import os
import argparse
import random
import numpy as np
import pickle
import json
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer
from biscope_utils import detect_single_sample, MODEL_ZOO
from dataset import CustomDataset
from metrics import AUROC, AUPR, TPR_at_FPR5, F1

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval'], 
                    help="Mode: train or eval")
parser.add_argument('--seed', type=int, default=42, 
                    help="Random seed for reproducibility")
parser.add_argument('--sample_clip', type=int, default=512, 
                    help="Max token length for samples")
parser.add_argument('--summary_model', type=str, default='none', 
                    help="Summary model key or 'none'")
parser.add_argument('--detect_model', type=str, required=True, 
                    help="Detection model key")
parser.add_argument('--train_data_path', type=str, default=None,
                    help='Path to training data (JSON format)')

# ====== 修改：将原来的 eval_data_path 替换为 base_dataset 和 generalization_datasets ======
parser.add_argument('--base_dataset', type=str, default=None,
                    help='Path to base evaluation data for finding optimal threshold')
parser.add_argument('--generalization_datasets', type=str, nargs='+', default=None,
                    help='List of paths to generalization datasets')

parser.add_argument('--data_format', type=str, default='MIRAGE',
                    help='Data format: MIRAGE or ImBD')
parser.add_argument('--save_path', type=str, default='./results/biscope',
                    help='Directory to save results and models')
parser.add_argument('--model_file', type=str, default='biscope_classifier.pkl',
                    help='Filename for saved model')
parser.add_argument('--save_file', type=str, default='eval_biscope_generalization.json',
                    help='Filename for evaluation results')

def load_models(args, device='cuda'):
    """Load summary model and detection model"""
    # Load summary model if specified
    if args.summary_model in MODEL_ZOO:
        summary_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ZOO[args.summary_model],
            torch_dtype=torch.bfloat16,
            device_map='auto'
        ).eval()
        summary_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ZOO[args.summary_model], padding_side='left'
        )
        summary_tokenizer.pad_token = summary_tokenizer.eos_token
    else:
        summary_model, summary_tokenizer = None, None
    
    # Load detection model
    if args.detect_model in MODEL_ZOO:
        detect_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ZOO[args.detect_model],
            torch_dtype=torch.bfloat16,
            device_map='auto'
        ).eval()
        detect_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ZOO[args.detect_model], padding_side='left'
        )
        detect_tokenizer.pad_token = detect_tokenizer.eos_token
    else:
        raise ValueError(f"Unknown detection model: {args.detect_model}")
    
    return detect_model, detect_tokenizer, summary_model, summary_tokenizer

def extract_features_from_data(args, data_path, fabric):
    """
    Extract features from data using CustomDataset
    Returns: (human_features, llm_features) as numpy arrays
    """
    if fabric.global_rank == 0:
        fabric.print(f"Loading data from {data_path}...")
        fabric.print(f"Data format: {args.data_format}")
        
        dataset = CustomDataset(data_path=data_path, data_format=args.data_format)
        fabric.print(f"Loaded {len(dataset)} samples")
        
        detect_model, detect_tokenizer, summary_model, summary_tokenizer = load_models(args)
        
        fabric.print("Extracting features from human texts...")
        human_features = []
        for i in tqdm(range(len(dataset)), desc="Processing human texts"):
            item = dataset[i]
            feats = detect_single_sample(
                args, detect_model, detect_tokenizer, 
                summary_model, summary_tokenizer, 
                item['human'], device='cuda'
            )
            human_features.append(feats)
        
        fabric.print("Extracting features from LLM texts...")
        llm_features = []
        for i in tqdm(range(len(dataset)), desc="Processing LLM texts"):
            item = dataset[i]
            feats = detect_single_sample(
                args, detect_model, detect_tokenizer,
                summary_model, summary_tokenizer,
                item['llm'], device='cuda'
            )
            llm_features.append(feats)
        
        return np.array(human_features), np.array(llm_features)
    else:
        return None, None

def find_optimal_threshold(human_scores, llm_scores):
    """
    在给定得分上寻找使 ACC 最优的阈值。
    对于 Random Forest，predict_proba[:, 1] 表示属于 LLM 的概率。
    概率越高越倾向于是 LLM。
    """
    y_true = [0] * len(human_scores) + [1] * len(llm_scores)
    y_scores = human_scores + llm_scores
    
    best_acc = 0.0
    best_thresh = 0.0
    
    unique_thresholds = np.unique(y_scores)
    
    for thresh in unique_thresholds:
        y_pred = [1 if s >= thresh else 0 for s in y_scores]
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            
    return best_thresh, best_acc

def train_mode(fabric: Fabric, args):
    """Training mode"""
    if fabric.global_rank == 0:
        fabric.print(f"="*60)
        fabric.print(f"Training mode")
        fabric.print(f"Data: {args.train_data_path}")
        fabric.print(f"Detection model: {args.detect_model}")
        fabric.print(f"Summary model: {args.summary_model}")
        fabric.print(f"="*60)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    human_features, llm_features = extract_features_from_data(args, args.train_data_path, fabric)
    
    if fabric.global_rank == 0 and human_features is not None:
        train_feats = np.vstack([human_features, llm_features])
        train_labels = np.concatenate([
            np.zeros(len(human_features)), 
            np.ones(len(llm_features))
        ])
        
        fabric.print(f"\nTraining with {len(train_feats)} samples...")
        
        fabric.print("\nTraining Random Forest classifier...")
        clf = RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1)
        clf.fit(train_feats, train_labels)
        
        os.makedirs(args.save_path, exist_ok=True)
        model_path = os.path.join(args.model_file)
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)
        fabric.print(f"\nModel saved to {model_path}")
        
        train_preds = clf.predict(train_feats)
        train_proba = clf.predict_proba(train_feats)[:, 1]
        
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)
        
        human_scores = train_proba[:len(human_features)].tolist()
        llm_scores = train_proba[len(human_features):].tolist()
        _, _, train_auroc = AUROC(neg_list=human_scores, pos_list=llm_scores)
        
        fabric.print(f"\nTraining Results:")
        fabric.print(f"  Accuracy: {train_acc:.4f}")
        fabric.print(f"  F1 Score: {train_f1:.4f}")
        fabric.print(f"  AUROC: {train_auroc:.4f}")
        
        train_results = {
            'detect_model': args.detect_model,
            'summary_model': args.summary_model,
            'train_dataset': args.train_data_path.split("/")[-1].split(".json")[0],
            'n_samples': len(train_feats),
            'n_human': len(human_features),
            'n_llm': len(llm_features),
            'train_accuracy': train_acc,
            'train_f1': train_f1,
            'train_auroc': train_auroc,
        }
        
        result_path = os.path.join(args.save_path, 'train_results.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(train_results, f, ensure_ascii=False, indent=4)
        fabric.print(f"Training results saved to {result_path}")

def eval_mode(fabric: Fabric, args):
    """Evaluation mode"""
    if fabric.global_rank == 0:
        fabric.print(f"="*60)
        fabric.print(f"Evaluation mode")
        fabric.print(f"Base Data: {args.base_dataset}")
        fabric.print(f"="*60)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if fabric.global_rank == 0:
        # Load trained model
        model_path = os.path.join(args.model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please train the model first using --mode train"
            )
        
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        fabric.print(f"Loaded model from {model_path}")
    
    # ==========================
    # 1. Base Dataset 提取与阈值计算
    # ==========================
    base_human_features, base_llm_features = extract_features_from_data(args, args.base_dataset, fabric)
    
    if fabric.global_rank == 0 and base_human_features is not None:
        fabric.print(f"\n[1/3] Finding optimal threshold on Base Dataset: {args.base_dataset}")
        # Get prediction probabilities (score for class 1 = LLM)
        base_human_scores = clf.predict_proba(base_human_features)[:, 1].tolist()
        base_llm_scores = clf.predict_proba(base_llm_features)[:, 1].tolist()
        
        best_threshold, best_base_acc = find_optimal_threshold(base_human_scores, base_llm_scores)
        fabric.print(f"--> Base Dataset Optimal Threshold: {best_threshold:.4f} | Base ACC: {best_base_acc:.4f}\n")
        
        result_dict = {
            'method': 'biscope',
            'detect_model': args.detect_model,
            'summary_model': args.summary_model,
            'base_dataset': args.base_dataset.split("/")[-1].split(".json")[0],
            'optimal_threshold': float(best_threshold),
            'base_dataset_acc': float(best_base_acc),
            'generalization_datasets_acc': {},
            'average_generalization_acc': 0.0
        }
        
        generalization_accs = []
        
        # ==========================
        # 2. Generalization Datasets 测试
        # ==========================
        fabric.print(f"[2/3] Evaluating on Generalization Datasets...")
        for gen_dataset_path in args.generalization_datasets:
            gen_human_features, gen_llm_features = extract_features_from_data(args, gen_dataset_path, fabric)
            
            gen_human_scores = clf.predict_proba(gen_human_features)[:, 1].tolist()
            gen_llm_scores = clf.predict_proba(gen_llm_features)[:, 1].tolist()
            
            y_true = [0] * len(gen_human_scores) + [1] * len(gen_llm_scores)
            y_scores = gen_human_scores + gen_llm_scores
            
            # 使用最优阈值进行分类预测
            y_pred = [1 if s >= best_threshold else 0 for s in y_scores]
            acc = accuracy_score(y_true, y_pred)
            
            dataset_name = os.path.basename(gen_dataset_path).split('.json')[0]
            result_dict['generalization_datasets_acc'][dataset_name] = float(acc)
            generalization_accs.append(acc)
            
            fabric.print(f"--> {dataset_name} ACC (using threshold {best_threshold:.4f}): {acc:.4f}")
        
        # ==========================
        # 3. 计算均值并保存
        # ==========================
        avg_gen_acc = float(np.mean(generalization_accs)) if generalization_accs else 0.0
        result_dict['average_generalization_acc'] = avg_gen_acc
        fabric.print(f"\n[3/3] Average Generalization ACC: {avg_gen_acc:.4f}")
        
        os.makedirs(args.save_path, exist_ok=True)
        result_path = os.path.join(args.save_path, args.save_file)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)
        fabric.print(f"\nResults saved to {result_path}")


def main(fabric: Fabric, args):
    if args.mode == 'train':
        if args.train_data_path is None:
            raise ValueError("--train_data_path is required for train mode")
        train_mode(fabric, args)
    elif args.mode == 'eval':
        # ====== 修改：确保传入了 base_dataset 和 generalization_datasets ======
        if args.base_dataset is None or args.generalization_datasets is None:
            raise ValueError("--base_dataset and --generalization_datasets are required for eval mode")
        eval_mode(fabric, args)

if __name__ == '__main__':
    args = parser.parse_args()
    
    # For BiSCoPE, we don't need distributed training since:
    # Models are loaded with device_map='auto' which handles multi-GPU

    fabric = Fabric(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,  # Single device, model handles multi-GPU internally
        precision="bf16-mixed"
    )
    
    fabric.launch(main, args)