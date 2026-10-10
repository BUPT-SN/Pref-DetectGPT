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
from sklearn.metrics import accuracy_score

from dataset import CustomDataset
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5, F1

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="openai-community/roberta-base-openai-detector")
# ====== 新增：base_dataset 和 generalization_datasets ======
parser.add_argument('--base_dataset', type=str, required=True, help='The path to the base dataset to find the optimal threshold.')
parser.add_argument('--generalization_datasets', type=str, nargs='+', required=True, help='A list of paths to the generalization datasets.')
parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD.')
parser.add_argument('--save_path', type=str, default='./results/roberta')
parser.add_argument('--save_file', type=str, default='eval_roberta_generalization.json')
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
            # 取 index=0 ('Fake') 的概率，分数越高越倾向于是 LLM 机器生成
            scores = probs[:, 0]
        return scores.tolist()


def find_optimal_threshold(human_scores, llm_scores):
    """
    寻找使 ACC 最优的阈值。
    """
    y_true = [0] * len(human_scores) + [1] * len(llm_scores)
    y_scores = human_scores + llm_scores
    
    best_acc = 0.0
    best_thresh = 0.0
    
    unique_thresholds = np.unique(y_scores)
    
    for thresh in unique_thresholds:
        # 分数越高越可能是 Fake(LLM)，所以大于阈值判为 1
        y_pred = [1 if s >= thresh else 0 for s in y_scores]
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            
    return best_thresh, best_acc


def evaluate_dataset(fabric: Fabric, model, dataset_path, args):
    """
    独立封装的数据集评估方法，处理 DDP 环境下的 dataloader 和 gather。
    """
    dataset = CustomDataset(data_path=dataset_path, data_format=args.eval_data_format)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    
    # 将 dataloader 包装给 fabric 使得支持多卡分布式读取
    data_loader = fabric.setup_dataloaders(data_loader)

    local_human_scores = []
    local_llm_scores = []

    dataset_name = os.path.basename(dataset_path)
    for idx, item in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Evaluating on {dataset_name}", disable=not (fabric.global_rank == 0)):
        local_human_scores.extend(model(item['human']))
        local_llm_scores.extend(model(item['llm']))

    fabric.barrier()
    
    # 汇总多卡上的结果
    all_human_scores = fabric.all_gather(torch.tensor(local_human_scores, device=fabric.device)).flatten().cpu().tolist()
    all_llm_scores = fabric.all_gather(torch.tensor(local_llm_scores, device=fabric.device)).flatten().cpu().tolist()
    
    return all_human_scores, all_llm_scores


def main(fabric: Fabric, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. 建立模型并注入 Fabric
    model = RobertaDetector(scoring_model_name=args.model_name)
    model.eval()
    model = fabric.setup(model)

    # 2. 评估 Base Dataset 以提取分数
    base_human_scores, base_llm_scores = evaluate_dataset(fabric, model, args.base_dataset, args)

    # 以下纯数据处理/保存逻辑，只让 rank 0 主进程执行
    best_threshold, best_base_acc = 0.0, 0.0
    result_dict = {}
    
    if fabric.global_rank == 0:
        best_threshold, best_base_acc = find_optimal_threshold(base_human_scores, base_llm_scores)
        fabric.print(f"\n--> Base Dataset Optimal Threshold: {best_threshold:.4f} | Base ACC: {best_base_acc:.4f}\n")
        
        result_dict = {
            'method': 'roberta',
            'model_name': args.model_name,
            'base_dataset': args.base_dataset.split("/")[-1].split(".json")[0],
            'optimal_threshold': float(best_threshold),
            'base_dataset_acc': float(best_base_acc),
            'generalization_datasets_acc': {},
            'average_generalization_acc': 0.0
        }

    # 同步状态（可选，确保所有进程在这个阶段对齐）
    fabric.barrier()
    
    # 将 best_threshold 广播给所有进程，或仅由 rank 0 计算（这里为了简单，都交给 rank 0 计算）
    # 但所有进程都需要参与 forward，否则 DDP 会卡死在 evaluate_dataset() 的 all_gather 处
    generalization_accs = []
    
    for gen_dataset_path in args.generalization_datasets:
        gen_human_scores, gen_llm_scores = evaluate_dataset(fabric, model, gen_dataset_path, args)
        
        if fabric.global_rank == 0:
            y_true = [0] * len(gen_human_scores) + [1] * len(gen_llm_scores)
            y_scores = gen_human_scores + gen_llm_scores
            
            y_pred = [1 if s >= best_threshold else 0 for s in y_scores]
            acc = accuracy_score(y_true, y_pred)
            
            dataset_name = os.path.basename(gen_dataset_path).split('.json')[0]
            result_dict['generalization_datasets_acc'][dataset_name] = float(acc)
            generalization_accs.append(acc)
            fabric.print(f"--> {dataset_name} ACC: {acc:.4f}")

    if fabric.global_rank == 0:
        avg_gen_acc = float(np.mean(generalization_accs)) if generalization_accs else 0.0
        result_dict['average_generalization_acc'] = avg_gen_acc
        fabric.print(f"\nAverage Generalization ACC: {avg_gen_acc:.4f}")

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        
        save_file_path = os.path.join(args.save_path, args.save_file)
        with open(save_file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(result_dict, ensure_ascii=False, indent=4))
        fabric.print(f"Results successfully saved to {save_file_path}")


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