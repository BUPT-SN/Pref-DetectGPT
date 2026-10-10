import random
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import argparse
import json
import time
import os
import sys
from scoring_methods import fastMDE
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings('ignore')

# 将上级目录 other_method 加入 sys.path，便于复用 dataset 和 metrics
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from dataset import CustomDataset
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5, F1

device = "cuda" if torch.cuda.is_available() else "cpu"
model_fullnames = { 
                     'meta-llama/Llama-3.2-1B':'meta-llama/Llama-3.2-1B',
                     'meta-llama/Llama-3.2-1B-Instruct':'meta-llama/Llama-3.2-1B-Instruct',
                     }
def load_model(model_name):
    model_fullname = model_fullnames[model_name]
    model_path = model_fullname

    print(f'Loading model {model_fullname}...')
    model_kwargs = {}
    
    model_kwargs.update(dict(torch_dtype=torch.bfloat16))

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs, device_map="auto")
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    # model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model

def load_tokenizer(model_name):
    model_fullname = model_fullnames[model_name]
    model_path = model_fullname

    optional_tok_kwargs = {}
    if "opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    optional_tok_kwargs['padding_side'] = 'right'

    base_tokenizer = AutoTokenizer.from_pretrained(model_path, **optional_tok_kwargs)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer

def load_data(input_file):
    # data_file = os.getcwd() + f"{input_file}.raw_data.json"
    data_file = f"{input_file}.raw_data.json"
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data

def get_samples(logits, labels, args):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = args.n_samples
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood

def get_lastde(log_likelihood, args):
    embed_size = args.embed_size
    epsilon = int(args.epsilon * log_likelihood.shape[1])
    
    # --- 修改开始：动态调整 tau_prime 以适应短文本 ---
    seq_len = log_likelihood.shape[1]
    
    # 理论推导：
    # calculate_DE 内部做 unfold 需要长度 >= embed_size
    # get_tau_scale_DE 产生的序列长度为 seq_len - tau + 1
    # 因此要求: seq_len - tau + 1 >= embed_size
    # 推导出: tau <= seq_len - embed_size + 1
    
    max_allowed_tau = seq_len - embed_size + 1
    
    # 如果文本极短（比如只有 3 个词），甚至连 tau=1 都算不了
    if max_allowed_tau < 1:
        # 这种情况下无法计算 MDE，返回一个默认值（比如 0 或 1），避免报错
        # 或者也可以抛出警告
        # print(f"Warning: Text too short (len={seq_len}) for LastDE calculation.")
        return torch.zeros_like(log_likelihood.mean(dim=1))

    # 取 min 确保不会超过 args.tau_prime，也不会超过物理极限
    current_tau_prime = min(args.tau_prime, max_allowed_tau)
    # --- 修改结束 ---

    templl = log_likelihood.mean(dim=1)
    
    # 使用动态调整后的 current_tau_prime
    aggmde = fastMDE.get_tau_multiscale_DE(
        ori_data=log_likelihood, 
        embed_size=embed_size, 
        epsilon=epsilon, 
        tau_prime=current_tau_prime  #这里传修正后的值
    )
    
    lastde = templl / aggmde 
    return lastde

def get_sampling_discrepancy(logits_ref, logits_score, labels, args):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels, args)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)


    # lastde
    lastde_x = get_lastde(log_likelihood_x, args)
    sampled_lastde = get_lastde(log_likelihood_x_tilde, args)

    miu_tilde = sampled_lastde.mean()
    sigma_tilde = sampled_lastde.std()
    discrepancy = (lastde_x - miu_tilde) / sigma_tilde

    return discrepancy.cpu().item()

def _compute_lastde_score(text, scoring_tokenizer, scoring_model,
                          reference_tokenizer, reference_model, args):
    """对单条文本计算 lastde_doubleplus 分数（与原始 experiment 中逻辑一致）。"""
    
    # --- 修改开始：添加 truncation 和 max_length ---
    tokenized = scoring_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,             # 开启截断
        max_length=args.max_length,  # 使用超参数控制长度
        return_token_type_ids=False,
    ).to(device)
    # --- 修改结束 ---

    labels = tokenized.input_ids[:, 1:]
    with torch.no_grad():
        logits_score = scoring_model(**tokenized).logits[:, :-1]
        if args.reference_model_name == args.scoring_model_name:
            logits_ref = logits_score
        else:
            # --- 修改开始：Reference Tokenizer 也要同步截断 ---
            tokenized_ref = reference_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,             # 开启截断
                max_length=args.max_length,  # 保持长度一致
                return_token_type_ids=False,
            ).to(device)
            # --- 修改结束 ---
            
            # 确保截断后的 input_ids 依然对齐
            assert torch.all(
                tokenized_ref.input_ids[:, 1:] == labels
            ), "Tokenizer is mismatch after truncation."
            logits_ref = reference_model(**tokenized_ref).logits[:, :-1]

        score = get_sampling_discrepancy(logits_ref, logits_score, labels, args)

    return score


def run_eval_like_baselines(args):
    """使用 LastDE-doubleplus 对 other_method/dataset.CustomDataset 进行评估，
    输出与 other_method/eval_baselines.py 一致的指标（AUROC/AUPR/TPR@FPR5/F1），
    并将结果保存为 JSON。
    """

    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name)
    scoring_model = load_model(args.scoring_model_name)
    scoring_model.eval()

    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name)
        reference_model = load_model(args.reference_model_name)
        reference_model.eval()
    else:
        reference_tokenizer = scoring_tokenizer
        reference_model = scoring_model

    # load data（与 other_method/eval_baselines.py 一致）
    dataset = CustomDataset(
        data_path=args.eval_data_path,
        data_format=args.eval_data_format,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    human_scores = []  # 负类（人类文本）
    llm_scores = []    # 正类（LLM 文本）

    pbar = tqdm.tqdm(data_loader, desc="Evaluating lastde_doubleplus")
    for batch in pbar:
        human_texts = batch["human"]
        llm_texts = batch["llm"]

        # 支持任意 eval_batch_size，逐条计算分数
        assert len(human_texts) == len(llm_texts)
        for human_text, llm_text in zip(human_texts, llm_texts):
            human_score = _compute_lastde_score(
                human_text,
                scoring_tokenizer,
                scoring_model,
                reference_tokenizer,
                reference_model,
                args,
            )
            llm_score = _compute_lastde_score(
                llm_text,
                scoring_tokenizer,
                scoring_model,
                reference_tokenizer,
                reference_model,
                args,
            )

            human_scores.append(human_score)
            llm_scores.append(llm_score)

    # 计算与 eval_baselines.py 对齐的指标
    fpr, tpr, eval_auroc = AUROC(neg_list=human_scores, pos_list=llm_scores)
    prec, recall, eval_aupr = AUPR(neg_list=human_scores, pos_list=llm_scores)
    tpr_at_5 = TPR_at_FPR5(neg_list=human_scores, pos_list=llm_scores)
    eval_f1 = F1(neg_list=human_scores, pos_list=llm_scores)

    human_score_mean = float(np.mean(human_scores))
    human_score_std = float(np.std(human_scores))
    llm_score_mean = float(np.mean(llm_scores))
    llm_score_std = float(np.std(llm_scores))

    print(
        f"Eval AUROC: {eval_auroc:.4f} | Eval AUPR: {eval_aupr:.4f} | "
        f"TPR@FPR5%: {tpr_at_5:.4f} | F1: {eval_f1:.4f}"
    )

    result_dict = {
        "method": "lastde_doubleplus",
        "scoring_model_name": args.scoring_model_name,
        "reference_model_name": args.reference_model_name,
        "eval_dataset": os.path.basename(args.eval_data_path),
        "eval_batch_size": args.eval_batch_size,
        "human_score_mean": human_score_mean,
        "human_score_std": human_score_std,
        "llm_score_mean": llm_score_mean,
        "llm_score_std": llm_score_std,
        "AUROC": float(eval_auroc),
        "AUPR": float(eval_aupr),
        "TPR_AT_FPR_5%": float(tpr_at_5),
        "F1": float(eval_f1),
        "human_scores": [float(s) for s in human_scores],
        "llm_scores": [float(s) for s in llm_scores],
    }

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    save_file = os.path.join(args.save_path, args.save_file)
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

    print(f"Results written into {save_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 与 other_method/eval_baselines.py 对齐的基础参数
    parser.add_argument(
        "--scoring_model_name",
        type=str,
        default="gptj_6b",
        help="用于打分的模型名称（应是 model_fullnames 中的键）",
    )
    parser.add_argument(
        "--reference_model_name",
        type=str,
        default="gptj_6b",
        help="参考模型名称（与 scoring_model_name 放在同一张表中）",
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default="./data/MIRAGE_BENCH/DIG/rewrite.json",
        help="评估数据路径，支持与 other_method/CustomDataset 相同的格式",
    )
    parser.add_argument(
        "--eval_data_format",
        type=str,
        default="MIRAGE",
        help="数据格式：MIRAGE 或 ImBD",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./results/Lastde",
        help="保存结果的目录",
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default="lastde_doubleplus_eval.json",
        help="保存结果的文件名",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="评估 batch size（可以大于 1）",
    )

    # LastDE 本身需要的超参数
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--embed_size", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=8)
    parser.add_argument("--tau_prime", type=int, default=15)

    parser.add_argument("--seed", type=int, default=0)
    
    # --- 新增参数：最大长度截断 ---
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="输入文本的最大 Token 长度截断，默认为 512",
    )
    # ---------------------------

    args = parser.parse_args()

    run_eval_like_baselines(args)
