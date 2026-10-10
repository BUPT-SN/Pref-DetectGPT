# Revise from https://github.com/baoguangsheng/fast-detect-gpt
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse
import json
from tqdm import tqdm
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from dataset import CustomDataset
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5, F1

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="openai-community/roberta-base-openai-detector")
parser.add_argument('--eval_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/rewrite.json', help='The path to the evaluation data.')
parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD.')
parser.add_argument('--save_path', type=str, default='./results/roberta')
parser.add_argument('--save_file', type=str, default='eval_roberta.json')
parser.add_argument('--eval_batch_size', type=int, default=8, help='The batch size for evaluation.')
parser.add_argument('--seed', type=int, default=42)

def from_pretrained(cls, model_name, kwargs):
    # use local model if it exists

    local_path = model_name

    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, device_map='auto')


class RobertaDetector(nn.Module):
    def __init__(self,
                 scoring_model_name: str=None,
                 scoring_model: AutoModelForSequenceClassification=None,
                 scoring_tokenizer: AutoTokenizer=None
                 ):
        super().__init__()
        if scoring_model_name is not None:
            self.scoring_model_name = scoring_model_name
            self.scoring_model = from_pretrained(AutoModelForSequenceClassification,
                                                 scoring_model_name,
                                                 kwargs=dict(torch_dtype=torch.float16))
            self.scoring_tokenizer = from_pretrained(AutoTokenizer,
                                                     scoring_model_name,
                                                     kwargs={'padding_side': 'right',
                                                             'use_fast': True if 'facebook/opt-' not in scoring_model_name else False})
        else:
            assert scoring_model is not None and scoring_tokenizer is not None, "You should provide scoring_model_name or scoring_model and scoring_tokenizer."
            self.scoring_model = scoring_model
            self.scoring_tokenizer = scoring_tokenizer
            self.scoring_model_name = scoring_model.config._name_or_path
        
        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
    '''
    查看标签映射:
    
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_name = "openai-community/roberta-base-openai-detector"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print(model.config.id2label)
    
    {0: 'Fake', 1: 'Real'}
    '''
    def forward(self, input_text: list[str]):
        tokenized = self.scoring_tokenizer(input_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.scoring_model.device)
        with torch.no_grad():
            logits = self.scoring_model(**tokenized).logits
            probs = torch.softmax(logits, dim=-1)
            scores = probs[:, 0]
        return scores.tolist()


def main(fabric: Fabric, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = RobertaDetector(scoring_model_name=args.model_name)
    model.eval()

    dataset = CustomDataset(data_path=args.eval_data_path, data_format=args.eval_data_format)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    model = fabric.setup(model)
    data_loader = fabric.setup_dataloaders(data_loader)

    local_human_scores = []
    local_llm_scores = []

    for idx, item in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Evaluating {args.model_name} on {args.eval_data_path.split('/')[-1]}", disable=not (fabric.global_rank == 0)):
        local_human_scores.extend(model(item['human']))
        local_llm_scores.extend(model(item['llm']))

    fabric.barrier()
    all_human_scores = fabric.all_gather(torch.tensor(local_human_scores, device=fabric.device)).flatten().cpu().tolist()
    all_llm_scores = fabric.all_gather(torch.tensor(local_llm_scores, device=fabric.device)).flatten().cpu().tolist()

    if fabric.global_rank == 0:
        fpr, tpr, eval_auroc = AUROC(neg_list=all_human_scores, pos_list=all_llm_scores)
        prec, recall, eval_aupr = AUPR(neg_list=all_human_scores, pos_list=all_llm_scores)
        tpr_at_5 = TPR_at_FPR5(neg_list=all_human_scores, pos_list=all_llm_scores)
        eval_f1 = F1(neg_list=all_human_scores, pos_list=all_llm_scores)

        human_score_mean = torch.mean(torch.tensor(all_human_scores)).item()
        human_score_std = torch.std(torch.tensor(all_human_scores)).item()
        llm_score_mean = torch.mean(torch.tensor(all_llm_scores)).item()
        llm_score_std = torch.std(torch.tensor(all_llm_scores)).item()

        fabric.print(f'Eval AUROC: {eval_auroc:.4f} | Eval AUPR: {eval_aupr:.4f} | TPR at FPR 5%: {tpr_at_5:.4f}')

        result_dict = {
            'method': 'roberta',
            'model_name': args.model_name,
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