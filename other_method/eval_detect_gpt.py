# Revise from https://github.com/baoguangsheng/fast-detect-gpt
import os.path

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import tqdm
import argparse
import json
from torch.utils.data import DataLoader
from dataset import PerturbedDataset
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy


parser = argparse.ArgumentParser()
parser.add_argument("--scoring_model_name", type=str, default="Qwen2-0.5B")
parser.add_argument('--eval_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/rewrite.json', help='The path to the evaluation data. Default: ./data/DIG/rewrite.json.')
parser.add_argument('--save_path', type=str, default='./results/FastDetectGPT')
parser.add_argument('--save_file', type=str, default='eval_fast_detect_gpt.json')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size for evaluation. Default: 1.')
parser.add_argument('--seed', type=int, default=42)

def from_pretrained(cls, model_name, kwargs):
    """Simple wrapper: always load directly from model_name (HF name or local path)."""
    return cls.from_pretrained(model_name, **kwargs)


class DetectGPT(nn.Module):
    def __init__(self, scoring_model_name):
        super().__init__()
        self.scoring_model = from_pretrained(AutoModelForCausalLM, scoring_model_name,
                                             dict(torch_dtype=torch.float16))
        self.scoring_tokenizer = from_pretrained(AutoTokenizer, scoring_model_name,
                                                 {'padding_side': 'right', 'use_fast': False})
        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token

    def get_lls(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
            
        tokenized = self.scoring_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512, return_token_type_ids=False).to(self.scoring_model.device)
        labels = tokenized.input_ids
        
        with torch.no_grad():
            logits = self.scoring_model(**tokenized).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        attention_mask = tokenized.attention_mask[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.shape)
        
        masked_loss = loss * attention_mask
        
        sequence_lengths = attention_mask.sum(dim=1).clamp(min=1)
        sequence_loss = masked_loss.sum(dim=1) / sequence_lengths
        
        return (-sequence_loss).tolist()

    def forward(self, texts, perturbed_texts):
        """
        Compute DetectGPT score for a batch of texts and their perturbed versions.
        Args:
            texts: list of str, the texts to score
            perturbed_texts: list of list of str, perturbed versions for each text
        Returns:
            list of dict, each with 'score' for the corresponding text
        """
        lls = self.get_lls(texts)
        results = []
        if len(texts) > 0 and len(perturbed_texts) > 0 and len(perturbed_texts[0]) > 0:
            for i in range(len(texts)):
                p_lls = self.get_lls(perturbed_texts[i])
                perturbed_ll_mean = np.mean(p_lls)
                perturbed_ll_std = np.std(p_lls) if len(p_lls) > 1 else 1
                if perturbed_ll_std == 0:
                    perturbed_ll_std = 1
                score = (lls[i] - perturbed_ll_mean) / perturbed_ll_std
                results.append(score)
        return results


def main(fabric: Fabric, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = DetectGPT(args.scoring_model_name)

    dataset = PerturbedDataset(data_path=args.eval_data_path)
    data_loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    model = fabric.setup(model)
    data_loader = fabric.setup_dataloaders(data_loader)

    local_human_scores = []
    local_llm_scores = []

    for item in tqdm.tqdm(data_loader, desc=f"Computing DetectGPT criterion", disable=not (fabric.global_rank == 0)):
        with torch.no_grad():
            local_human_scores.extend(model(item["human"], item["perturbed_human"]))
            local_llm_scores.extend(model(item["sampled"], item["perturbed_sampled"]))

    fabric.barrier()
    all_human_scores = fabric.all_gather(torch.tensor(local_human_scores, device=fabric.device)).flatten().cpu().tolist()
    all_llm_scores = fabric.all_gather(torch.tensor(local_llm_scores, device=fabric.device)).flatten().cpu().tolist()

    if fabric.global_rank == 0:
        fpr, tpr, roc_auc = AUROC(neg_list=all_human_scores, pos_list=all_llm_scores)
        p, r, pr_auc = AUPR(neg_list=all_human_scores, pos_list=all_llm_scores)
        tpr_at_5 = TPR_at_FPR5(neg_list=all_human_scores, pos_list=all_llm_scores)
        human_score_mean = torch.mean(torch.tensor(all_human_scores)).item()
        human_score_std = torch.std(torch.tensor(all_human_scores)).item()
        llm_score_mean = torch.mean(torch.tensor(all_llm_scores)).item()
        llm_score_std = torch.std(torch.tensor(all_llm_scores)).item()
        fabric.print(f'Eval AUROC: {roc_auc:.4f} | Eval AUPR: {pr_auc:.4f}')
        

        results = {
            'method': 'DetectGPT',
            'scoring_model_name': args.scoring_model_name,
            'dataset': args.eval_data_path,
            'eval_batch_size': args.eval_batch_size,
            'human_score_mean': human_score_mean,
            'human_score_std': human_score_std,
            'llm_score_mean': llm_score_mean,
            'llm_score_std': llm_score_std,
            'AUROC': roc_auc,
            'AUPR': pr_auc,
            'TPR_AT_FPR_5%': tpr_at_5,
            'human_scores': all_human_scores,
            'llm_scores': all_llm_scores,
        }
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        with open(os.path.join(args.save_path, args.save_file), 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, indent=4, ensure_ascii=False))
            fabric.print(f'Results written into {os.path.join(args.save_path, args.save_file)}')


if __name__ == '__main__':
    args = parser.parse_args()

    if torch.cuda.is_available():
        devices = torch.cuda.device_count()
    else:
        devices = 1

    if devices > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
        fabric = Fabric(accelerator="cuda", devices=devices, precision="bf16-mixed", strategy=strategy)
    else:
        fabric = Fabric(accelerator="cuda" if torch.cuda.is_available() else "cpu",
                        devices=1,
                        precision="bf16-mixed" if torch.cuda.is_available() else "32-true")

    fabric.launch(main, args)