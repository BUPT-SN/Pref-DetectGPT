import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
from tqdm import tqdm
from typing import Union
from sklearn.metrics import accuracy_score

from dataset import CustomDataset
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5, F1


parser = argparse.ArgumentParser()
parser.add_argument("--observer_model_path", type=str, default="./model/Qwen2-0.5B")
parser.add_argument("--performer_model_path", type=str, default="./model/Qwen2-0.5B")

# ====== 修改：新增 base_dataset 和 generalization_datasets ======
parser.add_argument('--base_dataset', type=str, required=True, help='The path to the base dataset to find the optimal threshold.')
parser.add_argument('--generalization_datasets', type=str, nargs='+', required=True, help='A list of paths to the generalization datasets.')

parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--save_path', type=str, default='./results/binoculars')
parser.add_argument('--save_file', type=str, default='eval_binoculars_generalization.json')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size for evaluation. Default: 1.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_token_observed', type=int, default=512)

torch.set_grad_enabled(False)

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)


def perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor):
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
           shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
    ppl = ppl.to("cpu").float().numpy()

    return ppl


def entropy(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits, q_logits

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    return agg_ce


def assert_tokenizer_consistency(model_id_1, model_id_2):
    identical_tokenizers = (
        AutoTokenizer.from_pretrained(model_id_1).vocab
        == AutoTokenizer.from_pretrained(model_id_2).vocab
    )
    if not identical_tokenizers:
        raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")


# 在当前可见设备内，把 observer 放在第 0 张卡，performer 放在第 1 张卡
DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1


class Binoculars:
    def __init__(self,
                 observer_name_or_path: str = "meta-llama/Meta-Llama-3.1-8B",
                 performer_name_or_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        dtype = torch.bfloat16 if use_bfloat16 else torch.float32

        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path,
            device_map={"": DEVICE_1},
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path,
            device_map={"": DEVICE_2},
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        ).to(DEVICE_1)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> tuple[torch.Tensor, torch.Tensor]:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1.startswith("cuda"):
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        binoculars_scores = - ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores


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
        collate_fn=dataset.collate_fn,
    )

    human_scores = []
    llm_scores = []

    dataset_name = os.path.basename(dataset_path)
    for idx, item in tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc=f"Evaluating binoculars on {dataset_name}",
    ):
        human_scores.extend(model.compute_score(item['human']))
        llm_scores.extend(model.compute_score(item['llm']))

    return human_scores, llm_scores


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = Binoculars(
        observer_name_or_path=args.observer_model_path,
        performer_name_or_path=args.performer_model_path,
        use_bfloat16=True,
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
        'method': 'binoculars',
        'observer_model_path': args.observer_model_path,
        'performer_model_path': args.performer_model_path,
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