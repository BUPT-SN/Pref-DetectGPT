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
parser.add_argument('--eval_data_path', type=str, default=None,
                    help='Path to evaluation data (JSON format)')
parser.add_argument('--data_format', type=str, default='MIRAGE',
                    help='Data format: MIRAGE or ImBD')
parser.add_argument('--save_path', type=str, default='./results/biscope',
                    help='Directory to save results and models')
parser.add_argument('--model_file', type=str, default='biscope_classifier.pkl',
                    help='Filename for saved model')
parser.add_argument('--save_file', type=str, default='eval_biscope.json',
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
    # Only load data and extract features on rank 0
    # Models are loaded with device_map='auto', so they don't need Fabric setup
    if fabric.global_rank == 0:
        fabric.print(f"Loading data from {data_path}...")
        fabric.print(f"Data format: {args.data_format}")
        
        # Use CustomDataset to load data
        dataset = CustomDataset(data_path=data_path, data_format=args.data_format)
        fabric.print(f"Loaded {len(dataset)} samples")
        
        # Load models (device_map='auto' handles multi-GPU)
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
        # Other ranks just return None
        return None, None

def train_mode(fabric: Fabric, args):
    """Training mode"""
    if fabric.global_rank == 0:
        fabric.print(f"="*60)
        fabric.print(f"Training mode")
        fabric.print(f"Data: {args.train_data_path}")
        fabric.print(f"Detection model: {args.detect_model}")
        fabric.print(f"Summary model: {args.summary_model}")
        fabric.print(f"="*60)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Extract features (only on rank 0)
    human_features, llm_features = extract_features_from_data(args, args.train_data_path, fabric)
    
    # Only rank 0 proceeds with training
    if fabric.global_rank == 0 and human_features is not None:
        # Combine features and labels
        train_feats = np.vstack([human_features, llm_features])
        train_labels = np.concatenate([
            np.zeros(len(human_features)), 
            np.ones(len(llm_features))
        ])
        
        fabric.print(f"\nTraining with {len(train_feats)} samples...")
        fabric.print(f"Human samples: {len(human_features)}")
        fabric.print(f"LLM samples: {len(llm_features)}")
        fabric.print(f"Feature shape: {train_feats.shape}")
        
        # Train Random Forest classifier
        fabric.print("\nTraining Random Forest classifier...")
        clf = RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1)
        clf.fit(train_feats, train_labels)
        
        # Save model
        os.makedirs(args.save_path, exist_ok=True)
        model_path = args.model_file
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)
        fabric.print(f"\nModel saved to {model_path}")
        
        # Compute training metrics
        train_preds = clf.predict(train_feats)
        train_proba = clf.predict_proba(train_feats)[:, 1]
        
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)
        
        # Compute AUROC on training set
        human_scores = train_proba[:len(human_features)].tolist()
        llm_scores = train_proba[len(human_features):].tolist()
        _, _, train_auroc = AUROC(neg_list=human_scores, pos_list=llm_scores)
        
        fabric.print(f"\nTraining Results:")
        fabric.print(f"  Accuracy: {train_acc:.4f}")
        fabric.print(f"  F1 Score: {train_f1:.4f}")
        fabric.print(f"  AUROC: {train_auroc:.4f}")
        
        # Save training results
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
            'human_scores': human_scores,
            'llm_scores': llm_scores,
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
        fabric.print(f"Data: {args.eval_data_path}")
        fabric.print(f"="*60)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Only rank 0 does evaluation
    if fabric.global_rank == 0:
        # Load trained model
        model_path = args.model_file
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please train the model first using --mode train"
            )
        
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        fabric.print(f"Loaded model from {model_path}")
    
    # Extract features from evaluation data (only on rank 0)
    human_features, llm_features = extract_features_from_data(args, args.eval_data_path, fabric)
    
    # Only rank 0 proceeds with evaluation
    if fabric.global_rank == 0 and human_features is not None:
        fabric.print(f"\nEvaluating on {len(human_features) + len(llm_features)} samples...")
        fabric.print(f"Human samples: {len(human_features)}")
        fabric.print(f"LLM samples: {len(llm_features)}")
        
        # Get prediction probabilities (score for class 1 = LLM)
        human_scores = clf.predict_proba(human_features)[:, 1].tolist()
        llm_scores = clf.predict_proba(llm_features)[:, 1].tolist()
        
        # Compute metrics
        fpr, tpr, eval_auroc = AUROC(neg_list=human_scores, pos_list=llm_scores)
        prec, recall, eval_aupr = AUPR(neg_list=human_scores, pos_list=llm_scores)
        tpr_at_5 = TPR_at_FPR5(neg_list=human_scores, pos_list=llm_scores)
        eval_f1 = F1(neg_list=human_scores, pos_list=llm_scores)
        
        human_score_mean = np.mean(human_scores)
        human_score_std = np.std(human_scores)
        llm_score_mean = np.mean(llm_scores)
        llm_score_std = np.std(llm_scores)
        
        fabric.print(f"\nEvaluation Results:")
        fabric.print(f"  AUROC: {eval_auroc:.4f}")
        fabric.print(f"  AUPR: {eval_aupr:.4f}")
        fabric.print(f"  TPR@FPR5%: {tpr_at_5:.4f}")
        fabric.print(f"  F1 Score: {eval_f1:.4f}")
        fabric.print(f"\nScore Statistics:")
        fabric.print(f"  Human: {human_score_mean:.4f} ± {human_score_std:.4f}")
        fabric.print(f"  LLM: {llm_score_mean:.4f} ± {llm_score_std:.4f}")
        
        # Save results
        result_dict = {
            'method': 'biscope',
            'detect_model': args.detect_model,
            'summary_model': args.summary_model,
            'eval_dataset': args.eval_data_path.split("/")[-1].split(".json")[0],
            'n_samples': len(human_features) + len(llm_features),
            'n_human': len(human_features),
            'n_llm': len(llm_features),
            'human_score_mean': float(human_score_mean),
            'human_score_std': float(human_score_std),
            'llm_score_mean': float(llm_score_mean),
            'llm_score_std': float(llm_score_std),
            'AUROC': float(eval_auroc),
            'AUPR': float(eval_aupr),
            'TPR_AT_FPR_5%': float(tpr_at_5),
            'F1': float(eval_f1),
            'human_scores': human_scores,
            'llm_scores': llm_scores,
        }
        
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
        if args.eval_data_path is None:
            raise ValueError("--eval_data_path is required for eval mode")
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