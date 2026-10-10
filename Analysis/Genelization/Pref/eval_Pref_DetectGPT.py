import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from core import Pref
from dataset import CustomDataset
# 原有的 metrics 可以保留，如果不需要输出 AUROC 等指标也可以删除
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5, F1, TPR_at_FPR10

parser = argparse.ArgumentParser()
# ====== 修改：新增 base_dataset 和 generalization_datasets 参数 ======
parser.add_argument('--base_dataset', type=str, required=True, help='The path to the base dataset to find the optimal threshold.')
parser.add_argument('--generalization_datasets', type=str, nargs='+', required=True, help='A list of paths to the generalization datasets.')
parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--save_path', type=str, default='./results/Pref')
parser.add_argument('--save_file', type=str, default='eval_Pref_generalization.json')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size for evaluation. Default: 1.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--reference_name_or_path', type=str, default='tiiuae/falcon-7b', help='Observer model name or local path.')
parser.add_argument('--policy_name_or_path', type=str, default='tiiuae/falcon-7b-instruct', help='Performer model name or local path.')
parser.add_argument('--sample_name_or_path', type=str, default='tiiuae/falcon-7b', help='Sample model name or local path.')
parser.add_argument('--max_token_observed', type=int, default=512)

torch.set_grad_enabled(False)

def find_optimal_threshold(human_scores, llm_scores):
    """
    在给定得分上寻找使 ACC 最优的阈值。
    假设得分越高越倾向于 LLM（标签为 1），得分越低越倾向于 Human（标签为 0）。
    """
    y_true = [0] * len(human_scores) + [1] * len(llm_scores)
    y_scores = human_scores + llm_scores
    
    best_acc = 0.0
    best_thresh = 0.0
    
    # 遍历所有出现过的分数作为候选阈值（使用 unique 减少计算量）
    unique_thresholds = np.unique(y_scores)
    
    for thresh in unique_thresholds:
        y_pred = [1 if s >= thresh else 0 for s in y_scores]
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            
    return best_thresh, best_acc

def evaluate_dataset(dataset_path, detector, args):
    """
    独立封装的函数，用于获取某个数据集下的 human_scores 和 llm_scores
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
        desc=f"Evaluating on {dataset_name}",
    )

    for idx, item in iterator:
        h_scores = detector.compute_score(item['human'])
        l_scores = detector.compute_score(item['llm'])
        human_scores.extend([s for s in h_scores])
        llm_scores.extend([s for s in l_scores])
        
    return human_scores, llm_scores

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. 初始化模型
    detector = Pref(
        reference_name_or_path=args.reference_name_or_path,
        policy_name_or_path=args.policy_name_or_path,
        sample_name_or_path=args.sample_name_or_path,
        max_token_observed=args.max_token_observed
    )

    # 2. 在 Base Dataset 上寻找最优阈值
    print(f"\n[1/3] Extracting scores from Base Dataset: {args.base_dataset}")
    base_human_scores, base_llm_scores = evaluate_dataset(args.base_dataset, detector, args)
    
    best_threshold, best_base_acc = find_optimal_threshold(base_human_scores, base_llm_scores)
    print(f"--> Base Dataset Optimal Threshold: {best_threshold:.4f} | Base ACC: {best_base_acc:.4f}\n")

    # 3. 初始化用于保存结果的字典
    result_dict = {
        'method': 'Pref',
        'reference_model_path': args.reference_name_or_path,
        'policy_model_path': args.policy_name_or_path,
        'sample_model_path': args.sample_name_or_path,
        'base_dataset': args.base_dataset,
        'optimal_threshold': float(best_threshold),
        'base_dataset_acc': float(best_base_acc),
        'generalization_datasets_acc': {},
        'average_generalization_acc': 0.0
    }

    # 4. 在 Generalization Datasets 上测试泛化性能
    generalization_accs = []
    
    print(f"[2/3] Evaluating on Generalization Datasets...")
    for gen_dataset_path in args.generalization_datasets:
        gen_human_scores, gen_llm_scores = evaluate_dataset(gen_dataset_path, detector, args)
        
        y_true = [0] * len(gen_human_scores) + [1] * len(gen_llm_scores)
        y_scores = gen_human_scores + gen_llm_scores
        
        # 使用基于 base_dataset 找到的最优阈值
        y_pred = [1 if s >= best_threshold else 0 for s in y_scores]
        acc = accuracy_score(y_true, y_pred)
        
        dataset_name = os.path.basename(gen_dataset_path).split('.json')[0]
        result_dict['generalization_datasets_acc'][dataset_name] = float(acc)
        generalization_accs.append(acc)
        
        print(f"--> {dataset_name} ACC (using threshold {best_threshold:.4f}): {acc:.4f}")

    # 5. 计算并保存平均值
    avg_gen_acc = float(np.mean(generalization_accs)) if generalization_accs else 0.0
    result_dict['average_generalization_acc'] = avg_gen_acc
    print(f"\n[3/3] Average Generalization ACC: {avg_gen_acc:.4f}")

    # 6. 将结果保存到 JSON
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    save_file_path = os.path.join(args.save_path, args.save_file)
    with open(save_file_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
        
    print(f"Results successfully saved to {save_file_path}")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)