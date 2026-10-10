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
import torch.nn.functional as F
from lightning.fabric import Fabric

parser = argparse.ArgumentParser()
parser.add_argument("--scoring_model_name", type=str, default="Qwen2-0.5B")
parser.add_argument("--method", type=str, default="likelihood", choices=["likelihood", "rank", "logrank", "entropy"])

# ====== 修改：新增 base_dataset 和 generalization_datasets ======
parser.add_argument('--base_dataset', type=str, required=True, help='The path to the base dataset to find the optimal threshold.')
parser.add_argument('--generalization_datasets', type=str, nargs='+', required=True, help='A list of paths to the generalization datasets.')

parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--save_path', type=str, default='./results/FastDetectGPT')
parser.add_argument('--save_file', type=str, default='eval_baselines_generalization.json')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size for evaluation. Default: 1.')
parser.add_argument('--seed', type=int, default=42)

torch.set_grad_enabled(False)

def get_likelihood(logits, labels, attention_mask=None):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
        mask = mask[:, :log_likelihood.shape[1]]
        log_likelihood = (log_likelihood * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
    else:
        log_likelihood = log_likelihood.mean(dim=-1)
    return log_likelihood

def get_rank(logits, labels, attention_mask=None):
    correct_token_logits = logits.gather(dim=-1, index=labels.unsqueeze(-1))
    ranks = (logits > correct_token_logits).sum(dim=-1) + 1
    ranks = ranks.float()
    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
        mask = mask[:, :ranks.shape[1]]
        ranks = (ranks * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
    else:
        ranks = ranks.mean(dim=-1)
    return ranks

def get_logrank(logits, labels, attention_mask=None):
    correct_token_logits = logits.gather(dim=-1, index=labels.unsqueeze(-1))
    ranks = (logits > correct_token_logits).sum(dim=-1) + 1
    ranks = ranks.float()
    log_ranks = torch.log(ranks)
    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
        mask = mask[:, :log_ranks.shape[1]]
        log_ranks = (log_ranks * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
    else:
        log_ranks = log_ranks.mean(dim=-1)
    return -log_ranks

def get_entropy(logits, labels, attention_mask=None):
    entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    entropy = entropy.sum(-1)
    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
        mask = mask[:, :entropy.shape[1]]
        entropy = (entropy * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
    else:
        entropy = entropy.mean(dim=-1)
    return entropy

def from_pretrained(cls, model_name, kwargs):
    """Simple wrapper: always load directly from model_name (HF name or local path)."""
    return cls.from_pretrained(model_name, **kwargs)

methods = {
    'likelihood': get_likelihood,
    'rank': get_rank,
    'logrank': get_logrank,
    'entropy': get_entropy,
}


class Baselines(nn.Module):
    def __init__(self,
                 scoring_model_name: str=None,
                 scoring_model: AutoModelForCausalLM=None,
                 scoring_tokenizer: AutoTokenizer=None,
                 method: callable=get_rank,
                 ):
        super().__init__()
        if scoring_model_name is not None:
            self.scoring_model_name = scoring_model_name
            self.scoring_model = from_pretrained(AutoModelForCausalLM,
                                                 scoring_model_name,
                                                 kwargs=dict(torch_dtype=torch.bfloat16))
            self.scoring_tokenizer = from_pretrained(AutoTokenizer,
                                                     scoring_model_name,
                                                     kwargs={'padding_side': 'right',
                                                             'use_fast': True if 'facebook/opt-' not in scoring_model_name else False},
                                                     )
        else:
            assert scoring_model is not None and scoring_tokenizer is not None, "You should provide scoring_model_name or scoring_model and scoring_tokenizer."
            self.scoring_model = scoring_model
            self.scoring_tokenizer = scoring_tokenizer
            self.scoring_model_name = scoring_model.config._name_or_path
        
        self.method = method
        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token

    def forward(self, input_text):
        if isinstance(input_text, str):
            input_texts = [input_text]
        else:
            input_texts = input_text

        tokenized = self.scoring_tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_token_type_ids=False,
        )
        tokenized = {k: v.to(self.scoring_model.device) for k, v in tokenized.items()}
        labels = tokenized['input_ids'][:, 1:]
        attention_mask = tokenized['attention_mask'] if 'attention_mask' in tokenized else None
        with torch.no_grad():
            logits = self.scoring_model(**tokenized).logits[:, :-1]
            scores = self.method(logits, labels, attention_mask=attention_mask)
        return scores.tolist()


def find_optimal_threshold(human_scores, llm_scores):
    """
    由于 likelihood、entropy、rank 等得分判定 LLM 和 Human 的大小关系（极性）可能相反，
    该函数会自动遍历两种情况并返回在 Base Dataset 上准确率最高的一套阈值和极性方向。
    """
    y_true = [0] * len(human_scores) + [1] * len(llm_scores)
    y_scores = human_scores + llm_scores
    
    best_acc = 0.0
    best_thresh = 0.0
    best_polarity = 1 # 1 表示 score >= thresh 是 LLM，-1 表示 score <= thresh 是 LLM
    
    unique_thresholds = np.unique(y_scores)
    
    for thresh in unique_thresholds:
        # 测试情况1：分数越高越像 LLM (如 likelihood)
        y_pred_pos = [1 if s >= thresh else 0 for s in y_scores]
        acc_pos = accuracy_score(y_true, y_pred_pos)
        
        # 测试情况2：分数越低越像 LLM (如 entropy, rank)
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

def evaluate_dataset(fabric: Fabric, model, dataset_path, args):
    """
    独立封装的针对单一数据集进行推理打分的过程
    """
    dataset = CustomDataset(data_path=dataset_path, data_format=args.eval_data_format)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    data_loader = fabric.setup_dataloaders(data_loader)

    local_human_scores = []
    local_llm_scores = []

    rank = int(os.environ.get("RANK", 0))
    dataset_name = os.path.basename(dataset_path)
    
    iterator = tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc=f"Evaluating {args.method} on {dataset_name}",
        disable=False,
        position=rank,
        leave=True,
    )

    for idx, item in iterator:
        local_human_scores.extend(model(item['human']))
        local_llm_scores.extend(model(item['llm']))

    fabric.barrier()
    all_human_scores = fabric.all_gather(torch.tensor(local_human_scores, device=fabric.device)).flatten().cpu().tolist()
    all_llm_scores = fabric.all_gather(torch.tensor(local_llm_scores, device=fabric.device)).flatten().cpu().tolist()

    return all_human_scores, all_llm_scores
    
def main(fabric: Fabric, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = Baselines(scoring_model_name=args.scoring_model_name,
                      method=methods[args.method])
    model.eval()
    model.to(fabric.device)

    # ==========================================
    # 1. 评估 Base Dataset 并计算最佳阈值与极性
    # ==========================================
    if fabric.global_rank == 0:
        fabric.print(f"\n[1/3] Extracting scores from Base Dataset: {args.base_dataset}")
        
    base_human_scores, base_llm_scores = evaluate_dataset(fabric, model, args.base_dataset, args)

    best_threshold, best_base_acc, polarity = 0.0, 0.0, 1
    result_dict = {}

    if fabric.global_rank == 0:
        best_threshold, best_base_acc, polarity = find_optimal_threshold(base_human_scores, base_llm_scores)
        direction_str = ">=" if polarity == 1 else "<="
        
        fabric.print(f"\n--> Base Dataset Optimal Threshold: {best_threshold:.4f} (Rule: LLM {direction_str} Thresh) | Base ACC: {best_base_acc:.4f}\n")
        
        result_dict = {
            'method': args.method,
            'scoring_model_name': args.scoring_model_name,
            'base_dataset': args.base_dataset.split("/")[-1].split(".json")[0],
            'eval_batch_size': args.eval_batch_size,
            'optimal_threshold': float(best_threshold),
            'threshold_polarity': int(polarity),
            'base_dataset_acc': float(best_base_acc),
            'generalization_datasets_acc': {},
            'average_generalization_acc': 0.0
        }

    # 等待 Rank 0 处理完毕
    fabric.barrier()

    # ==========================================
    # 2. 将阈值应用到 Generalization Datasets 
    # ==========================================
    if fabric.global_rank == 0:
        fabric.print(f"[2/3] Evaluating on Generalization Datasets...")
        
    generalization_accs = []
    
    for gen_dataset_path in args.generalization_datasets:
        gen_human_scores, gen_llm_scores = evaluate_dataset(fabric, model, gen_dataset_path, args)
        
        if fabric.global_rank == 0:
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
            
            fabric.print(f"--> {dataset_name} ACC (using threshold {best_threshold:.4f}): {acc:.4f}")

    # ==========================================
    # 3. 计算均值并保存结果
    # ==========================================
    if fabric.global_rank == 0:
        avg_gen_acc = float(np.mean(generalization_accs)) if generalization_accs else 0.0
        result_dict['average_generalization_acc'] = avg_gen_acc
        
        fabric.print(f"\n[3/3] Average Generalization ACC: {avg_gen_acc:.4f}")

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
            
        save_file_path = os.path.join(args.save_path, args.save_file)
        with open(save_file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(result_dict, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    args = parser.parse_args()

    # 与 eval.py 保持一致：不使用 DDPStrategy，只用 Fabric 的多进程和通信能力
    torch.set_float32_matmul_precision("medium")

    if torch.cuda.is_available():
        total_gpus = torch.cuda.device_count()
        devices = total_gpus
    else:
        devices = 1

    fabric = Fabric(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=devices,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
    )

    fabric.launch(main, args)