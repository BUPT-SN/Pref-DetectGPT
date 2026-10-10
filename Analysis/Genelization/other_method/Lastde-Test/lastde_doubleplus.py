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
from sklearn.metrics import accuracy_score

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
    model_fullname = model_fullnames.get(model_name, model_name) # Fallback if not in dict
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
    model_fullname = model_fullnames.get(model_name, model_name)
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
    
    max_allowed_tau = seq_len - embed_size + 1
    
    if max_allowed_tau < 1:
        return torch.zeros_like(log_likelihood.mean(dim=1))

    current_tau_prime = min(args.tau_prime, max_allowed_tau)
    # --- 修改结束 ---

    templl = log_likelihood.mean(dim=1)
    
    aggmde = fastMDE.get_tau_multiscale_DE(
        ori_data=log_likelihood, 
        embed_size=embed_size, 
        epsilon=epsilon, 
        tau_prime=current_tau_prime
    )
    
    lastde = templl / aggmde 
    return lastde

def get_sampling_discrepancy(logits_ref, logits_score, labels, args):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
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
    discrepancy = (lastde_x - miu_tilde) / (sigma_tilde + 1e-8)

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


def evaluate_dataset(dataset_path, scoring_tokenizer, scoring_model, reference_tokenizer, reference_model, args):
    """
    提取出的专门用于单一数据集推理和打分的函数
    """
    dataset = CustomDataset(
        data_path=dataset_path,
        data_format=args.eval_data_format,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    human_scores = []
    llm_scores = []

    dataset_name = os.path.basename(dataset_path)
    pbar = tqdm.tqdm(data_loader, desc=f"Evaluating Lastde++ on {dataset_name}")
    for batch in pbar:
        human_texts = batch["human"]
        llm_texts = batch["llm"]

        # 逐条计算分数
        assert len(human_texts) == len(llm_texts)
        for human_text, llm_text in zip(human_texts, llm_texts):
            human_score = _compute_lastde_score(
                human_text, scoring_tokenizer, scoring_model,
                reference_tokenizer, reference_model, args
            )
            llm_score = _compute_lastde_score(
                llm_text, scoring_tokenizer, scoring_model,
                reference_tokenizer, reference_model, args
            )

            human_scores.append(human_score)
            llm_scores.append(llm_score)
            
    return human_scores, llm_scores


def run_generalization_eval(args):
    """
    使用 Base Dataset 找寻最优阈值，并在 Generalization Datasets 上进行泛化测试
    """
    # 1. Load models
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

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ==========================================
    # 2. 评估 Base Dataset 并计算最佳阈值与极性
    # ==========================================
    print(f"\n[1/3] Extracting scores from Base Dataset: {args.base_dataset}")
    base_human_scores, base_llm_scores = evaluate_dataset(
        args.base_dataset, scoring_tokenizer, scoring_model, reference_tokenizer, reference_model, args
    )

    best_threshold, best_base_acc, polarity = find_optimal_threshold(base_human_scores, base_llm_scores)
    direction_str = ">=" if polarity == 1 else "<="
    
    print(f"\n--> Base Dataset Optimal Threshold: {best_threshold:.4f} (Rule: LLM {direction_str} Thresh) | Base ACC: {best_base_acc:.4f}\n")

    result_dict = {
        "method": "lastde_doubleplus",
        "scoring_model_name": args.scoring_model_name,
        "reference_model_name": args.reference_model_name,
        "base_dataset": os.path.basename(args.base_dataset).split(".json")[0],
        "eval_batch_size": args.eval_batch_size,
        "optimal_threshold": float(best_threshold),
        "threshold_polarity": int(polarity),
        "base_dataset_acc": float(best_base_acc),
        "generalization_datasets_acc": {},
        "average_generalization_acc": 0.0
    }

    # ==========================================
    # 3. 将阈值应用到 Generalization Datasets 
    # ==========================================
    print(f"[2/3] Evaluating on Generalization Datasets...")
    generalization_accs = []

    for gen_dataset_path in args.generalization_datasets:
        gen_human_scores, gen_llm_scores = evaluate_dataset(
            gen_dataset_path, scoring_tokenizer, scoring_model, reference_tokenizer, reference_model, args
        )
        
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
    # 4. 计算均值并保存结果
    # ==========================================
    avg_gen_acc = float(np.mean(generalization_accs)) if generalization_accs else 0.0
    result_dict['average_generalization_acc'] = avg_gen_acc
    
    print(f"\n[3/3] Average Generalization ACC: {avg_gen_acc:.4f}")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    save_file = os.path.join(args.save_path, args.save_file)
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

    print(f"Results written into {save_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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
    
    # ====== 修改：新增 base_dataset 和 generalization_datasets ======
    parser.add_argument(
        '--base_dataset', 
        type=str, 
        required=True, 
        help='The path to the base dataset to find the optimal threshold.'
    )
    parser.add_argument(
        '--generalization_datasets', 
        type=str, 
        nargs='+', 
        required=True, 
        help='A list of paths to the generalization datasets.'
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
        default="lastde_doubleplus_generalization.json",
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

    run_generalization_eval(args)