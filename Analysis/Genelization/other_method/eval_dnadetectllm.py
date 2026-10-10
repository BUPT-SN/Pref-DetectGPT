import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from dna_detectllm import DetectLLM
from dataset import CustomDataset
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5, F1, TPR_at_FPR10

parser = argparse.ArgumentParser()
# ====== 修改：新增 base_dataset 和 generalization_datasets ======
parser.add_argument('--base_dataset', type=str, required=True, help='The path to the base dataset to find the optimal threshold.')
parser.add_argument('--generalization_datasets', type=str, nargs='+', required=True, help='A list of paths to the generalization datasets.')

parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--save_path', type=str, default='./results/dna_detectllm')
parser.add_argument('--save_file', type=str, default='eval_dnadetectllm_generalization.json')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size for evaluation. Default: 1.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--observer_name_or_path', type=str, default='tiiuae/falcon-7b', help='Observer model name or local path (default: tiiuae/falcon-7b).')
parser.add_argument('--performer_name_or_path', type=str, default='tiiuae/falcon-7b-instruct', help='Performer model name or local path (default: tiiuae/falcon-7b-instruct).')

torch.set_grad_enabled(False)


def find_optimal_threshold(human_scores, llm_scores):
    """
    自动判断极性并寻找使 ACC 最优的阈值。
    """
    y_true = [0] * len(human_scores) + [1] * len(llm_scores)
    y_scores = human_scores + llm_scores
    
    best_acc = 0.0
    best_thresh = 0.0
    best_polarity = 1 # 1 表示 score >= thresh 是 LLM，-1 表示 score <= thresh 是 LLM
    
    unique_thresholds = np.unique(y_scores)
    
    for thresh in unique_thresholds:
        # 测试情况1：分数越高越像 LLM
        y_pred_pos = [1 if s >= thresh else 0 for s in y_scores]
        acc_pos = accuracy_score(y_true, y_pred_pos)
        
        # 测试情况2：分数越低越像 LLM
        y_pred_neg = [1 if s <= thresh else 0 for s in y_scores]
        acc_neg = accuracy_score(y_true, y_pred_neg)
        
        if acc_pos > best_acc:
            best_acc = acc_pos
            best_thresh = thresh
            best_polarity = 1
            
        if acc_neg > best_acc:
            best_acc = acc_neg
            best_thresh = thresh
            best_polarity = -1
            
    return best_thresh, best_acc, best_polarity

def evaluate_dataset(detector, dataset_path, args):
    """
    独立封装的针对单一数据集进行推理打分的过程
    """
    dataset = CustomDataset(data_path=dataset_path, data_format=args.eval_data_format)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    human_scores = []
    llm_scores = []

    dataset_name = os.path.basename(dataset_path)
    iterator = tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc=f"Evaluating DNA-DetectLLM on {dataset_name}",
    )

    for idx, item in iterator:
        h_scores = detector.compute_score(item['human'])
        l_scores = detector.compute_score(item['llm'])
        # 为与全局评测对齐（人类为负类、LLM 为正类，分数越大越像 LLM），对 DetectLLM 分数取反
        human_scores.extend([-s for s in h_scores])
        llm_scores.extend([-s for s in l_scores])

    return human_scores, llm_scores


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    detector = DetectLLM(
        observer_name_or_path=args.observer_name_or_path,
        performer_name_or_path=args.performer_name_or_path,
    )

    # ==========================================
    # 1. 评估 Base Dataset 并计算最佳阈值与极性
    # ==========================================
    print(f"\n[1/3] Extracting scores from Base Dataset: {args.base_dataset}")
    base_human_scores, base_llm_scores = evaluate_dataset(detector, args.base_dataset, args)

    best_threshold, best_base_acc, polarity = find_optimal_threshold(base_human_scores, base_llm_scores)
    direction_str = ">=" if polarity == 1 else "<="
    
    print(f"\n--> Base Dataset Optimal Threshold: {best_threshold:.4f} (Rule: LLM {direction_str} Thresh) | Base ACC: {best_base_acc:.4f}\n")

    result_dict = {
        'method': 'dna_detectllm',
        'observer_model_path': args.observer_name_or_path,
        'performer_model_path': args.performer_name_or_path,
        'base_dataset': os.path.basename(args.base_dataset).split('.json')[0],
        'eval_batch_size': args.eval_batch_size,
        'optimal_threshold': float(best_threshold),
        'threshold_polarity': int(polarity),
        'base_dataset_acc': float(best_base_acc),
        'generalization_datasets_acc': {},
        'average_generalization_acc': 0.0
    }

    # ==========================================
    # 2. 将阈值应用到 Generalization Datasets 
    # ==========================================
    print(f"[2/3] Evaluating on Generalization Datasets...")
    generalization_accs = []

    for gen_dataset_path in args.generalization_datasets:
        gen_human_scores, gen_llm_scores = evaluate_dataset(detector, gen_dataset_path, args)
        
        y_true = [0] * len(gen_human_scores) + [1] * len(gen_llm_scores)
        y_scores = gen_human_scores + gen_llm_scores
        
        if polarity == 1:
            y_pred = [1 if s >= best_threshold else 0 for s in y_scores]
        else:
            y_pred = [1 if s <= best_threshold else 0 for s in y_scores]
            
        acc = accuracy_score(y_true, y_pred)
        
        dataset_name = os.path.basename(gen_dataset_path).split('.json')[0]
        result_dict['generalization_datasets_acc'][dataset_name] = float(acc)
        generalization_accs.append(acc)
        
        print(f"--> {dataset_name} ACC (using threshold {best_threshold:.4f}): {acc:.4f}")

    # ==========================================
    # 3. 计算均值并保存结果
    # ==========================================
    avg_gen_acc = float(np.mean(generalization_accs)) if generalization_accs else 0.0
    result_dict['average_generalization_acc'] = avg_gen_acc
    
    print(f"\n[3/3] Average Generalization ACC: {avg_gen_acc:.4f}")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    save_file_path = os.path.join(args.save_path, args.save_file)
    with open(save_file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(result_dict, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)