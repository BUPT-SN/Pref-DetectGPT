# Revise from https://github.com/baoguangsheng/fast-detect-gpt
import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
from tqdm import tqdm
from typing import List, Tuple, Dict, Union
from sklearn.metrics import accuracy_score

from dataset import CustomDataset
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5, F1
import torch.nn as nn
import copy
# from peft import AutoPeftModelForCausalLM


parser = argparse.ArgumentParser()
parser.add_argument("--scoring_model_name", type=str, default="Qwen2-0.5B")
parser.add_argument("--reference_model_name", type=str, default=None)
parser.add_argument("--cache_dir", type=str, default="")
parser.add_argument("--discrepancy_analytic", action="store_true", help="Whether to use the analytic version of sampling discrepancy.")

# ====== 修改：新增 base_dataset 和 generalization_datasets ======
parser.add_argument('--base_dataset', type=str, required=True, help='The path to the base dataset to find the optimal threshold.')
parser.add_argument('--generalization_datasets', type=str, nargs='+', required=True, help='A list of paths to the generalization datasets.')

parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--save_path', type=str, default='./results/IRM')
parser.add_argument('--save_file', type=str, default='eval_irm_generalization.json')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size for evaluation. Default: 1.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_token_observed', type=int, default=512)

torch.set_grad_enabled(False)

# 将 scoring / reference 模型分别放在两张卡上（在当前可见设备内）
DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

def get_likelihood(logits, labels, attention_mask=None):
    """
    计算序列的对数似然 (Log-Likelihood)。
    labels 为 input_ids 右移一位的结果。
    """
    # 确保 labels 维度正确 [batch, seq_len, 1]
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    
    # 获取 log_softmax
    lprobs = torch.log_softmax(logits, dim=-1)
    
    # 提取对应 label 的 log_prob
    log_likelihood = lprobs.gather(dim=-1, index=labels).squeeze(-1)

    if attention_mask is not None:
        # mask 同样需要右移对齐 labels
        mask = attention_mask[:, 1:].float()
        mask = mask[:, :log_likelihood.shape[1]]
        # iRM 通常直接取总和来代表该序列在模型下的绝对概率
        return (log_likelihood * mask).sum(dim=-1)
    else:
        return log_likelihood.sum(dim=-1)

class IRMDetector(nn.Module):
    def __init__(self,
                 scoring_model_name: str,
                 reference_model_name: str,
                 max_token_observed: int = 512):
        super().__init__()
        self.max_token_observed = max_token_observed
        
        # 1. 加载评分模型 (Base Model) - 放置在 DEVICE_1
        print(f"Loading Scoring Model: {scoring_model_name}...")
        self.scoring_model = AutoModelForCausalLM.from_pretrained(
            scoring_model_name, torch_dtype=torch.bfloat16
        ).to(DEVICE_1).eval()
        
        # self.scoring_model = AutoPeftModelForCausalLM.from_pretrained(
        #     scoring_model_name, torch_dtype=torch.bfloat16
        # ).to(DEVICE_1).eval()
        self.scoring_tokenizer = AutoTokenizer.from_pretrained(scoring_model_name)
        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token

        # 2. 加载参考模型 (Reference Model) - 尽量放置在 DEVICE_2
        print(f"Loading Reference Model: {reference_model_name}...")
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            reference_model_name, torch_dtype=torch.bfloat16
        ).to(DEVICE_2).eval()
        self.reference_tokenizer = AutoTokenizer.from_pretrained(reference_model_name)
        if self.reference_tokenizer.pad_token is None:
            self.reference_tokenizer.pad_token = self.reference_tokenizer.eos_token

    def compute_score(self, input_texts: Union[List[str], str]) -> List[float]:
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        with torch.no_grad():
            # --- 评分模型推理 ---
            tokenized_score = self.scoring_tokenizer(
                input_texts, return_tensors="pt", padding=True, 
                truncation=True, max_length=self.max_token_observed
            ).to(DEVICE_1)
            
            labels_score = tokenized_score.input_ids[:, 1:]
            logits_score = self.scoring_model(**tokenized_score).logits[:, :-1]
            ll_score = get_likelihood(logits_score, labels_score, tokenized_score.attention_mask)

            # --- 参考模型推理 ---
            # 注意：参考模型可能使用不同的 Tokenizer，必须重新编码
            tokenized_ref = self.reference_tokenizer(
                input_texts, return_tensors="pt", padding=True, 
                truncation=True, max_length=self.max_token_observed
            ).to(DEVICE_2)
            
            labels_ref = tokenized_ref.input_ids[:, 1:]
            logits_ref = self.reference_model(**tokenized_ref).logits[:, :-1]
            ll_ref = get_likelihood(logits_ref, labels_ref, tokenized_ref.attention_mask)

            # --- 计算 iRM 分数 ---
            # Score = LogLikelihood_Score - LogLikelihood_Ref
            # 分数越高，通常意味着文本由 AI 生成的可能性越大
            score = ll_score - ll_ref.to(DEVICE_1)
            
            return score.tolist()

    def forward(self, x):
        return self.compute_score(x)

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

def evaluate_dataset(model, dataset_path, args):
    """
    独立封装的针对单一数据集进行推理打分的过程
    """
    dataset = CustomDataset(data_path=dataset_path, data_format=args.eval_data_format)
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.eval_batch_size, 
        shuffle=False, 
        collate_fn=dataset.collate_fn
    )
    
    human_scores = []
    llm_scores = []
    
    dataset_name = os.path.basename(dataset_path)
    for idx, item in tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc=f"Evaluating IRM on {dataset_name}",
    ):
        human_scores.extend(model(item['human']))
        llm_scores.extend(model(item['llm']))
        
    return human_scores, llm_scores


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = IRMDetector(
        scoring_model_name=args.scoring_model_name,
        reference_model_name=args.reference_model_name,
        max_token_observed=args.max_token_observed,
    )

    # ==========================================
    # 1. 评估 Base Dataset 并计算最佳阈值与极性
    # ==========================================
    print(f"\n[1/3] Extracting scores from Base Dataset: {args.base_dataset}")
    base_human_scores, base_llm_scores = evaluate_dataset(model, args.base_dataset, args)

    best_threshold, best_base_acc, polarity = find_optimal_threshold(base_human_scores, base_llm_scores)
    direction_str = ">=" if polarity == 1 else "<="
    
    print(f"\n--> Base Dataset Optimal Threshold: {best_threshold:.4f} (Rule: LLM {direction_str} Thresh) | Base ACC: {best_base_acc:.4f}\n")

    result_dict = {
        'method': 'IRM',
        'scoring_model_name': args.scoring_model_name,
        'reference_model_name': args.reference_model_name,
        'discrepancy_analytic': args.discrepancy_analytic,
        'base_dataset': os.path.basename(args.base_dataset).split(".json")[0],
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
        gen_human_scores, gen_llm_scores = evaluate_dataset(model, gen_dataset_path, args)
        
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

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)