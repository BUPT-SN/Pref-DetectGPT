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
import torch.nn.functional as F
from lightning.fabric import Fabric

parser = argparse.ArgumentParser()
parser.add_argument("--scoring_model_name", type=str, default="Qwen2-0.5B")
parser.add_argument("--method", type=str, default="likelihood", choices=["likelihood", "rank", "logrank", "entropy"])
parser.add_argument('--eval_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/rewrite.json', help='The path to the evaluation data. Default: ./data/DIG/rewrite.json.')
parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--save_path', type=str, default='./results/FastDetectGPT')
parser.add_argument('--save_file', type=str, default='eval_baselines.json')
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
    """
    different from the human code, because argsort in the human code is memory-intensive.
    """
    correct_token_logits = logits.gather(dim=-1, index=labels.unsqueeze(-1))
    # Get ranks by counting how many logits are greater than the logit of the correct token.
    # Add 1 for 1-based ranking.
    # This avoids creating a huge sorted index tensor, which is memory-intensive.
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
    """
    different from the human code, because argsort in the human code is memory-intensive.
    """
    # Get the logits of the correct labels.
    # correct_token_logits will have shape (batch_size, seq_len, 1)
    correct_token_logits = logits.gather(dim=-1, index=labels.unsqueeze(-1))
    # Get ranks by counting how many logits are greater than the logit of the correct token.
    # Add 1 for 1-based ranking.
    # This avoids creating a huge sorted index tensor, which is memory-intensive.
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
    
def main(fabric: Fabric, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = Baselines(scoring_model_name=args.scoring_model_name,
                      method=methods[args.method])
    model.eval()

    dataset = CustomDataset(data_path=args.eval_data_path, data_format=args.eval_data_format)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    # 与 eval.py 一致：不包 DDP，仅将模型放到当前进程设备上，多卡仅做多进程推理 + all_gather
    model.to(fabric.device)
    data_loader = fabric.setup_dataloaders(data_loader)

    local_human_scores = []
    local_llm_scores = []

    # 每个 rank 单独显示一条进度条，位置用 rank 区分，避免互相覆盖
    rank = int(os.environ.get("RANK", 0))
    iterator = tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc=f"Evaluating {args.method}",
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

    if fabric.global_rank == 0:
        fpr, tpr, eval_auroc = AUROC(neg_list=all_human_scores, pos_list=all_llm_scores)
        prec, recall, eval_aupr = AUPR(neg_list=all_human_scores, pos_list=all_llm_scores)
        tpr_at_5 = TPR_at_FPR5(neg_list=all_human_scores, pos_list=all_llm_scores)
        human_score_mean = torch.mean(torch.tensor(all_human_scores)).item()
        eval_f1 = F1(neg_list=all_human_scores, pos_list=all_llm_scores)
        human_score_std = torch.std(torch.tensor(all_human_scores)).item()
        llm_score_mean = torch.mean(torch.tensor(all_llm_scores)).item()
        llm_score_std = torch.std(torch.tensor(all_llm_scores)).item()
        fabric.print(f'Eval AUROC: {eval_auroc:.4f} | Eval AUPR: {eval_aupr:.4f} | TPR@FPR5%: {tpr_at_5:.4f} | F1: {eval_f1:.4f}')


        result_dict = {
            'method': args.method,
            'scoring_model_name': args.scoring_model_name,
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
            'human_scores': all_human_scores,
            'llm_scores': all_llm_scores,
        }

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        with open(os.path.join(args.save_path, args.save_file), 'w', encoding='utf-8') as f:
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