import argparse
import os
import random
import time
import numpy as np
import torch
import datetime
import json

from torch.utils.data import DataLoader, Subset
from lightning.fabric import Fabric
from sklearn.metrics import accuracy_score

from core_imbd.model import DiscrepancyEstimator
from core_imbd.dataset import CustomDataset
from core_imbd.metrics import AUROC, AUPR, TPR_at_FPR5, TPR_at_FPR10, F1
from core_imbd.trainer import Trainer
from tqdm import tqdm

parser = argparse.ArgumentParser()
# Model
parser.add_argument('--scoring_model_name', type=str, default='gpt-neo-2.7B', help='The path or name of the scoring model. Default: gpt-neo-2.7B.')
parser.add_argument('--reference_model_name', type=str, default=None, help='The path or name of the reference model. Default: None. Which indicates that the reference model is the same as the scoring model when using DPO and None when using DDL.')
parser.add_argument('--train_method', type=str, default='DDL', help='The training method. Should be DDL or SPO. Default: DDL.')
parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, help='Pass a pretrained model name or path to load the pretrained model. Default: None.')
# Dataset
# ====== 修改：新增 base_dataset 和 generalization_datasets ======
parser.add_argument('--base_dataset', type=str, required=True, help='The path to the base dataset to find the optimal threshold.')
parser.add_argument('--generalization_datasets', type=str, nargs='+', required=True, help='A list of paths to the generalization datasets.')
parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size for evaluation. Default: 1.')
# WandB
parser.add_argument('--wandb', type=bool, default=False, help='Whether to use wandb for logging. Default: False.')
parser.add_argument('--wandb_dir', type=str, default='./log/', help='The directory to store the wandb logs. Default: ./log/.')
parser.add_argument('--wandb_entity', type=str, default=None, help='The entity of the wandb project. Default: None.')
# Save
parser.add_argument('--save_dir', type=str, default='./results/', help='The directory to save the evaluation results. Default: ./results/.')
parser.add_argument('--save_file', type=str, default=None, help='The file to save the evaluation results. Default: None.')
parser.add_argument('--device_num', type=int, default=None, help='Number of GPUs used by Fabric for evaluation. Default: all visible GPUs.')


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
        # 假设分数越高越倾向于是 LLM
        y_pred = [1 if s >= thresh else 0 for s in y_scores]
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            
    return best_thresh, best_acc

def evaluate_dataset(fabric: Fabric, model, trainer, dataset_path, args):
    """
    在指定数据集上运行推理，并汇总所有 GPU 卡上的评分结果
    """
    dataset = CustomDataset(dataset_path,
                            scoring_tokenizer=model.scoring_tokenizer,
                            reference_tokenizer=None,
                            data_format=args.eval_data_format)
    
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    loader = fabric.setup_dataloaders(loader)
    
    # 仅在主进程显示进度条
    show_pbar = (fabric.global_rank == 0)
    
    local_human_eval, local_llm_eval = trainer.eval(
        model, loader, show_progress_bar=show_pbar
    )
    fabric.barrier()
    
    # 将多张卡的结果汇总
    all_human_eval = fabric.all_gather(torch.tensor(local_human_eval, device=fabric.device)).flatten().cpu().tolist()
    all_llm_eval = fabric.all_gather(torch.tensor(local_llm_eval, device=fabric.device)).flatten().cpu().tolist()
    
    return all_human_eval, all_llm_eval

def main(fabric: Fabric, args):
    # Set seed for reproducibility
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.reference_model_name == "None":
        args.reference_model_name = None
        
    model = DiscrepancyEstimator(scoring_model_name=args.scoring_model_name,
                                 reference_model_name=args.reference_model_name,
                                 train_method=args.train_method,
                                 pretrained_ckpt=args.pretrained_model_name_or_path)
    
    if args.wandb:
        os.makedirs(args.wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = args.wandb_dir 
        
    if args.save_file is not None:
        save_name = f'{args.save_file}'
    else:
        # 使用 base_dataset 名字作为默认保存文件名的一部分
        base_name = args.base_dataset.split("/")[-1].split(".json")[0]
        save_name = f'{args.train_method}_generalization_score_imbd_{model.scoring_model.config._name_or_path.split("/")[-1]}_ref_{"None" if model.reference_model_name is None else model.reference_model_name.split("/")[-1]}_{base_name}_evalBS{args.eval_batch_size}'
    
    # Set up wandb
    if args.wandb is True and fabric.global_rank == 0:
        import wandb
        if args.wandb_entity is None:
            assert os.environ.get('WANDB_MODE') == 'offline', "Please set WANDB_MODE to offline or provide a wandb_entity"
        now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        wandb.init(
            project='Evaluate_Machine_Generate_Text_Detection',
            entity=args.wandb_entity,
            name=f"{save_name}_{now_time}",
            config={
                'scoring_model_name': model.scoring_model_name,
                'train_method': args.train_method,
                'reference_model_name': model.reference_model_name if model.reference_model_name is not None else 'None',
                'pretrained_model_name_or_path': args.pretrained_model_name_or_path if args.pretrained_model_name_or_path is not None else 'None',
                'base_dataset': args.base_dataset,
                'eval_batch_size': args.eval_batch_size,
                'wandb_dir': args.wandb_dir,
                'result_file': save_name,
            },
        )

    # 准备模型和 Trainer
    model.to(fabric.device)
    trainer = Trainer()

    # ==========================================
    # 1. 在 Base Dataset 上提取分数并计算最优阈值
    # ==========================================
    if fabric.global_rank == 0:
        fabric.print(f"\n[1/3] Extracting scores from Base Dataset: {args.base_dataset}")
        
    base_human_eval, base_llm_eval = evaluate_dataset(fabric, model, trainer, args.base_dataset, args)

    best_threshold, best_base_acc = 0.0, 0.0
    result_dict = {}
    
    if fabric.global_rank == 0:
        best_threshold, best_base_acc = find_optimal_threshold(base_human_eval, base_llm_eval)
        fabric.print(f"--> Base Dataset Optimal Threshold: {best_threshold:.4f} | Base ACC: {best_base_acc:.4f}\n")
        
        result_dict = {
            'ckpt_name': args.pretrained_model_name_or_path if args.pretrained_model_name_or_path is not None else args.scoring_model_name,
            'base_dataset': args.base_dataset.split("/")[-1].split(".json")[0],
            'eval_batch_size': args.eval_batch_size,
            'optimal_threshold': float(best_threshold),
            'base_dataset_acc': float(best_base_acc),
            'generalization_datasets_acc': {},
            'average_generalization_acc': 0.0
        }

    # 等待 rank 0 处理完毕
    fabric.barrier()

    # ==========================================
    # 2. 在 Generalization Datasets 上测试泛化效果
    # ==========================================
    if fabric.global_rank == 0:
        fabric.print(f"[2/3] Evaluating on Generalization Datasets...")
        
    generalization_accs = []
    
    for gen_dataset_path in args.generalization_datasets:
        gen_human_eval, gen_llm_eval = evaluate_dataset(fabric, model, trainer, gen_dataset_path, args)
        
        if fabric.global_rank == 0:
            y_true = [0] * len(gen_human_eval) + [1] * len(gen_llm_eval)
            y_scores = gen_human_eval + gen_llm_eval
            
            y_pred = [1 if s >= best_threshold else 0 for s in y_scores]
            acc = accuracy_score(y_true, y_pred)
            
            dataset_name = os.path.basename(gen_dataset_path).split('.json')[0]
            result_dict['generalization_datasets_acc'][dataset_name] = float(acc)
            generalization_accs.append(acc)
            
            fabric.print(f"--> {dataset_name} ACC (using threshold {best_threshold:.4f}): {acc:.4f}")

    # ==========================================
    # 3. 统计并保存结果
    # ==========================================
    if fabric.global_rank == 0:
        avg_gen_acc = float(np.mean(generalization_accs)) if generalization_accs else 0.0
        result_dict['average_generalization_acc'] = avg_gen_acc
        fabric.print(f"\n[3/3] Average Generalization ACC: {avg_gen_acc:.4f}")

        if args.wandb is True:
            import wandb
            wandb.log(result_dict, step=0)

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # 保存 JSON
        with open(os.path.join(args.save_dir, f'{save_name.strip(".json")}.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(result_dict, ensure_ascii=False, indent=4))

        # 追加写入 CSV 概要日志
        csv_path = os.path.join(args.save_dir, 'eval_generalization_summary.csv')
        need_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        header = 'time,ckpt_name,base_dataset,eval_batch_size,optimal_threshold,base_acc,avg_gen_acc\n'
        line = f'{now_str},{result_dict["ckpt_name"]},{result_dict["base_dataset"]},{result_dict["eval_batch_size"]},{best_threshold:.6f},{best_base_acc:.6f},{avg_gen_acc:.6f}\n'
        
        with open(csv_path, 'a', encoding='utf-8') as f_csv:
            if need_header:
                f_csv.write(header)
            f_csv.write(line)

if __name__ == '__main__':
    args = parser.parse_args()

    # 初始化 Fabric：支持多卡 DDP 推理
    torch.set_float32_matmul_precision("medium")

    if torch.cuda.is_available():
        total_gpus = torch.cuda.device_count()
        device_num = args.device_num or total_gpus
        device_num = max(1, min(device_num, total_gpus))
    else:
        device_num = 1

    fabric = Fabric(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        devices=device_num,
    )

    fabric.launch(main, args)