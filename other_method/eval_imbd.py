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
parser.add_argument('--eval_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/rewrite.json', help='The path to the evaluation data. Default: ./data/DIG/rewrite.json.')
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


def main(fabric: Fabric, args):
    # Set seed for reproducibility
    seed = 42
    # 全局随机种子：Python / NumPy / Torch（含 CUDA）
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
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
        os.environ["WANDB_DIR"] = args.wandb_dir  # 指定日志目录
    if args.save_file is not None:
        save_name = f'{args.save_file}'
    else:
        save_name = f'{args.train_method}_score_imbd_{model.scoring_model.config._name_or_path.split("/")[-1]}_ref_{"None" if model.reference_model_name is None else model.reference_model_name.split("/")[-1]}_{args.eval_data_path.split("/")[-1].split(".json")[0]}_evalBS{args.eval_batch_size}'
    
    # Set up wandb（评估阶段也直接使用 wandb 接口）
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
                'eval_dataset': args.eval_data_path,
                'eval_batch_size': args.eval_batch_size,
                'wandb_dir': args.wandb_dir,
                'result_file': save_name,
            },
        )
    
    eval_dataset = CustomDataset(args.eval_data_path,
                                 scoring_tokenizer=model.scoring_tokenizer,
                                 reference_tokenizer=None,
                                 data_format=args.eval_data_format)
    
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=eval_dataset.collate_fn)

    # 使用 Fabric 管理设备与数据加载，多卡下不再使用 DDP 包装模型，只做多进程推理 + all_gather 聚合
    model.to(fabric.device)
    eval_loader = fabric.setup_dataloaders(eval_loader)
    
    # Evaluate
    trainer = Trainer()

    local_human_eval, local_llm_eval = trainer.eval(
        model, eval_loader, show_progress_bar=True
    )
    fabric.barrier()
    
    all_human_eval = fabric.all_gather(torch.tensor(local_human_eval, device=fabric.device)).flatten().cpu().tolist()
    all_llm_eval = fabric.all_gather(torch.tensor(local_llm_eval, device=fabric.device)).flatten().cpu().tolist()
    if fabric.global_rank == 0:
        fpr, tpr, eval_auroc = AUROC(all_human_eval, all_llm_eval)
        prec, recall, eval_aupr = AUPR(all_human_eval, all_llm_eval)
        tpr_at_5 = TPR_at_FPR5(all_human_eval, all_llm_eval)
        tpr_at_10 = TPR_at_FPR10(all_human_eval, all_llm_eval)
        eval_f1 = F1(all_human_eval, all_llm_eval)
        human_score_mean = torch.mean(torch.tensor(all_human_eval)).item()
        human_score_std = torch.std(torch.tensor(all_human_eval)).item()
        llm_score_mean = torch.mean(torch.tensor(all_llm_eval)).item()
        llm_score_std = torch.std(torch.tensor(all_llm_eval)).item()
        fabric.print(f'Eval AUROC: {eval_auroc:.4f} | Eval AUPR: {eval_aupr:.4f} | Eval TPR@FPR=10%: {tpr_at_10:.4f} | Eval F1: {eval_f1:.4f}')
        
        result_dict = {
            'ckpt_name': args.pretrained_model_name_or_path if args.pretrained_model_name_or_path is not None else args.scoring_model_name,
            'eval_dataset': args.eval_data_path.split("/")[-1].split(".json")[0],
            'eval_batch_size': args.eval_batch_size,
            'human_score_mean': human_score_mean,
            'human_score_std': human_score_std,
            'llm_score_mean': llm_score_mean,
            'llm_score_std': llm_score_std,
            'AUROC': eval_auroc,
            'AUPR': eval_aupr,
            'TPR_AT_FPR_5%': tpr_at_5,
            'TPR_AT_FPR_10%': tpr_at_10,
            'F1': eval_f1,
        }
        if args.wandb is True:
            import wandb
            wandb.log(result_dict, step=0)

        result_dict['human_scores'] = all_human_eval
        result_dict['llm_scores'] = all_llm_eval

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        with open(os.path.join(args.save_dir, f'{save_name.strip(".json")}.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(result_dict, ensure_ascii=False, indent=4))

        # 额外将本次 eval 的关键指标追加写入一个 CSV 日志，便于对比不同 checkpoint
        csv_path = os.path.join(args.save_dir, 'eval_summary.csv')
        need_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        header = 'time,ckpt_name,eval_dataset,eval_batch_size,AUROC,TPR_AT_FPR_5%\n'
        line = f'{now_str},{result_dict["ckpt_name"]},{result_dict["eval_dataset"]},{result_dict["eval_batch_size"]},{result_dict["AUROC"]:.6f},{result_dict["TPR_AT_FPR_5%"]:.6f}\n'
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
        # 如果未显式指定 device_num，就默认用所有可见 GPU
        device_num = args.device_num or total_gpus
        device_num = max(1, min(device_num, total_gpus))
    else:
        device_num = 1

    # 评估阶段无需梯度同步，这里即便多卡也不使用 DDP 包模型，只利用 Fabric 的多进程与通信能力
    fabric = Fabric(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        devices=device_num,
    )

    # 使用 Fabric 启动（单/多进程均可）
    fabric.launch(main, args)