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

parser = argparse.ArgumentParser()
parser.add_argument("--scoring_model_name", type=str, default="Qwen2-0.5B")
parser.add_argument("--reference_model_name", type=str, default=None)
parser.add_argument("--cache_dir", type=str, default="./model/")
parser.add_argument("--discrepancy_analytic", action="store_true", help="Whether to use the analytic version of sampling discrepancy.")

# ====== 修改：新增 base_dataset 和 generalization_datasets ======
parser.add_argument('--base_dataset', type=str, required=True, help='The path to the base dataset to find the optimal threshold.')
parser.add_argument('--generalization_datasets', type=str, nargs='+', required=True, help='A list of paths to the generalization datasets.')

parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--save_path', type=str, default='./results/FastDetectGPT')
parser.add_argument('--save_file', type=str, default='eval_fast_detect_gpt_generalization.json')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size for evaluation. Default: 1.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_token_observed', type=int, default=512)

torch.set_grad_enabled(False)

# 将 scoring / reference 模型分别放在两张卡上（在当前可见设备内）
DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

def get_samples(logits, labels):
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples

def get_likelihood(logits, labels, attention_mask=None):
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels).squeeze(-1)

    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
        mask = mask[:, :log_likelihood.shape[1]]
        log_likelihood = (log_likelihood * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
    else:
        log_likelihood = log_likelihood.mean(dim=-1)
    return log_likelihood

def get_sampling_discrepancy(logits_ref, logits_score, labels, attention_mask=None):
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels)
    log_likelihood_x = get_likelihood(logits_score, labels, attention_mask)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples, attention_mask)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    discrepancy = (log_likelihood_x - miu_tilde) / (sigma_tilde + 1e-8)
    return discrepancy

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels, attention_mask=None):
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)

    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
        mask = mask[:, :log_likelihood.shape[1]]
        log_likelihood = (log_likelihood * mask).sum(dim=-1)
        mean_ref = (mean_ref * mask).sum(dim=-1)
        var_ref = (var_ref * mask).sum(dim=-1)
    else:
        log_likelihood = log_likelihood.sum(dim=-1)
        mean_ref = mean_ref.sum(dim=-1)
        var_ref = var_ref.sum(dim=-1)

    discrepancy = (log_likelihood - mean_ref) / (var_ref.sqrt() + 1e-8)
    return discrepancy

    
def from_pretrained(cls, model_name, kwargs):
    """Always load directly from model_name (HF name or local path)."""
    return cls.from_pretrained(model_name, **kwargs)


class FastDetectGPT(nn.Module):
    def __init__(self,
                 scoring_model_name: str=None,
                 reference_model_name: str=None,
                 scoring_model: AutoModelForCausalLM=None,
                 reference_model: AutoModelForCausalLM=None,
                 scoring_tokenizer: AutoTokenizer=None,
                 reference_tokenizer: AutoTokenizer=None,
                 cache_dir: str=None,  # kept for CLI compatibility, unused
                 discrepancy_analytic: bool=False,
                 max_token_observed: int = 512,
                 ):
        super().__init__()
        self.cache_dir = cache_dir
        self.max_token_observed = max_token_observed
        if scoring_model_name is not None:
            self.scoring_model_name = scoring_model_name
            self.scoring_model = from_pretrained(AutoModelForCausalLM,
                                                 scoring_model_name,
                                                 kwargs=dict(torch_dtype=torch.bfloat16))
            # 放到第 0 张可见 GPU 上    
            self.scoring_model.to(DEVICE_1)
            self.scoring_tokenizer = from_pretrained(AutoTokenizer,
                                                     scoring_model_name,
                                                     kwargs={'padding_side': 'right',
                                         'use_fast': True if 'facebook/opt-' not in scoring_model_name else False},
                                                     )
        else: 
            if scoring_model is None or scoring_tokenizer is None:
                raise ValueError('You should provide scoring_model_name or scoring_model and scoring_tokenizer.')
            self.scoring_model = scoring_model
            self.scoring_tokenizer = scoring_tokenizer
            self.scoring_model_name = scoring_model.config._name_or_path
        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
            self.scoring_tokenizer.pad_token_id = self.scoring_tokenizer.eos_token_id

        if reference_model_name is not None:
            if reference_model_name != scoring_model_name:
                self.reference_model = from_pretrained(AutoModelForCausalLM,
                                                        reference_model_name,
                                                        kwargs=dict(torch_dtype=torch.bfloat16))
                # 放到第 1 张可见 GPU 上
                self.reference_model.to(DEVICE_2)
                self.reference_tokenizer = from_pretrained(AutoTokenizer,
                                                            reference_model_name,
                                                            kwargs={'padding_side': 'right',
                                        'use_fast': True if 'facebook/opt-' not in reference_model_name else False},
                                    )
                self.reference_model_name = reference_model_name
            else:
                self.reference_model = None
                self.reference_tokenizer = None
                self.reference_model_name = self.scoring_model_name
        else:
            if reference_model is None and reference_tokenizer is None:
                self.reference_model = None
                self.reference_tokenizer = None
                self.reference_model_name = self.scoring_model_name
            elif reference_model is not None and reference_tokenizer is not None:
                self.reference_model = reference_model
                self.reference_tokenizer = reference_tokenizer
                self.reference_model_name = reference_model.config._name_or_path
            else:
                raise ValueError('You should provide reference_model and reference_tokenizer at the same time.')
        self.discrepancy = get_sampling_discrepancy_analytic if discrepancy_analytic else get_sampling_discrepancy

        if self.reference_tokenizer is not None:
            if self.reference_tokenizer.pad_token is None:
                self.reference_tokenizer.pad_token = self.reference_tokenizer.eos_token
                self.reference_tokenizer.pad_token_id = self.reference_tokenizer.eos_token_id

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        if isinstance(input_text, str):
            input_texts = [input_text]
        else:
            input_texts = input_text
        with torch.no_grad():
            # 先在 DEVICE_1 上做 scoring 编码与前向
            tokenized_score = self.scoring_tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_token_observed,
                return_token_type_ids=False,
            ).to(DEVICE_1)
            labels = tokenized_score.input_ids[:, 1:]
            logits_score = self.scoring_model(**tokenized_score).logits[:, :-1]

            if self.scoring_model_name == self.reference_model_name:
                logits_ref = logits_score
            else:
                # reference 模型在 DEVICE_2 上前向，但后续所有计算统一放到 DEVICE_1
                tokenized_ref = self.reference_tokenizer(
                    input_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_token_observed,
                    return_token_type_ids=False,
                ).to(DEVICE_2)
                logits_ref = self.reference_model(**tokenized_ref).logits[:, :-1]

                # 调整序列长度对齐
                min_seq_len = min(logits_ref.shape[1], logits_score.shape[1])
                logits_ref = logits_ref[:, :min_seq_len, :]
                logits_score = logits_score[:, :min_seq_len, :]
                labels = labels[:, :min_seq_len]

            # 仿照 binoculars，将所有需要参与 discrepancy 计算的张量统一到同一设备上
            common_device = DEVICE_1
            logits_score = logits_score.to(common_device)
            logits_ref = logits_ref.to(common_device)
            labels = labels.to(common_device)
            attention_mask = tokenized_score.attention_mask.to(common_device)

            scores = self.discrepancy(logits_ref, logits_score, labels, attention_mask)

        return scores.tolist()
    
    def forward(self, input_text):
        return self.compute_score(input_text)


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
        y_pred_pos = [1 if s >= thresh else 0 for s in y_scores]
        acc_pos = accuracy_score(y_true, y_pred_pos)
        
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
        desc=f"Evaluating FastDetectGPT on {dataset_name}",
    ):
        human_scores.extend(model(item['human']))
        llm_scores.extend(model(item['llm']))
        
    return human_scores, llm_scores


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = FastDetectGPT(
        scoring_model_name=args.scoring_model_name,
        reference_model_name=args.reference_model_name,
        cache_dir=args.cache_dir,
        discrepancy_analytic=args.discrepancy_analytic,
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
        'method': 'FastDetectGPT',
        'scoring_model_name': args.scoring_model_name,
        'reference_model_name': args.reference_model_name,
        'discrepancy_analytic': args.discrepancy_analytic,
        'base_dataset': os.path.basename(args.base_dataset).split(".json")[0],
        'eval_batch_size': args.eval_batch_size,
        'optimal_threshold': float(best_threshold),
        'threshold_polarity': int(polarity),
        'base_dataset_acc': float(best_base_acc),
        'generalization_datasets_acc': {},
        'average_generalization_acc': 0.0,
        # 仅保留 Base Dataset 的分数，防止 JSON 文件过大
        "base_human_scores": base_human_scores,
        "base_llm_scores": base_llm_scores
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