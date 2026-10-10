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

from dataset import CustomDataset
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5, F1
import torch.nn as nn
import copy
# from peft import AutoPeftModelForCausalLM


parser = argparse.ArgumentParser()
parser.add_argument("--scoring_model_name", type=str, default="Qwen2-0.5B")
parser.add_argument("--reference_model_name", type=str, default=None)
parser.add_argument('--eval_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/rewrite.json', help='The path to the evaluation data. Default: ./data/MIRAGE_BENCH/DIG/rewrite.json.')
parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--save_path', type=str, default='./results/IRM')
parser.add_argument('--save_file', type=str, default='eval_irm.json')
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

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = IRMDetector(
        scoring_model_name=args.scoring_model_name,
        reference_model_name=args.reference_model_name,
        max_token_observed=args.max_token_observed,
    )

    dataset = CustomDataset(data_path=args.eval_data_path, data_format=args.eval_data_format)
    human_scores = []
    llm_scores = []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    for idx, item in tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc=f"Evaluating IRM on {args.eval_data_path.split('/')[-1]}",
    ):
        human_scores.extend(model(item['human']))
        llm_scores.extend(model(item['llm']))

    # 单进程上直接计算指标（保持 neg=human, pos=llm 的语义不变）
    fpr, tpr, eval_auroc = AUROC(neg_list=human_scores, pos_list=llm_scores)
    prec, recall, eval_aupr = AUPR(neg_list=human_scores, pos_list=llm_scores)
    tpr_at_5 = TPR_at_FPR5(neg_list=human_scores, pos_list=llm_scores)
    eval_f1 = F1(neg_list=human_scores, pos_list=llm_scores)
    human_score_mean = torch.mean(torch.tensor(human_scores)).item()
    human_score_std = torch.std(torch.tensor(human_scores)).item()
    llm_score_mean = torch.mean(torch.tensor(llm_scores)).item()
    llm_score_std = torch.std(torch.tensor(llm_scores)).item()

    print(f'Eval AUROC: {eval_auroc:.4f} | Eval AUPR: {eval_aupr:.4f} | TPR@FPR5%: {tpr_at_5:.4f} | F1: {eval_f1:.4f}')

    result_dict = {
        'method': 'IRM',
        'scoring_model_name': args.scoring_model_name,
        'reference_model_name': args.reference_model_name,
        'eval_dataset': args.eval_data_path.split("/")[-1].split(".json")[0],
        'eval_batch_size': args.eval_batch_size,
        'human_score_mean': human_score_mean,
        'human_score_std': human_score_std,
        'llm_score_mean': llm_score_mean,
        'llm_score_std': llm_score_std,
        'AUROC': eval_auroc,
        'AUPR': eval_aupr,
        'TPR_AT_FPR_5%': tpr_at_5,
        'F1': eval_f1,
        'human_scores': human_scores,
        'llm_scores': llm_scores,
    }

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, args.save_file), 'w', encoding='utf-8') as f:
        f.write(json.dumps(result_dict, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)