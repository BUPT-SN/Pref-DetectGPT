import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import re

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from core_imbd.model import DiscrepancyEstimator
from core_imbd.dataset import CustomDataset
from core_imbd.loss import calculate_DPO_loss, calculate_DDL_loss
from core_imbd.metrics import AUROC, AUPR
from core_imbd.trainer import Trainer
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from peft import LoraConfig, TaskType

parser = argparse.ArgumentParser()
# Model
parser.add_argument('--scoring_model_name', type=str, default='gpt-neo-2.7B', help='The path or name of the scoring model. Default: gpt-neo-2.7B.')
parser.add_argument('--reference_model_name', type=str, default=None, help='The path or name of the reference model. Default: None. Which indicates that the reference model is the same as the scoring model when using DPO and None when using DDL.')
parser.add_argument('--train_method', type=str, default='DDL', help='The training method. Should be DDL or SPO. Default: DDL.')
# LoRA
parser.add_argument('--lora_rank', type=int, default=8, help='The rank of LoRA. Default: 8.')
parser.add_argument('--lora_alpha', type=float, default=32, help='The alpha of LoRA. Default: 32.')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='The dropout of LoRA. Default: 0.1.')
# Data
parser.add_argument('--train_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/polish.json', help='The path to the training data. Default: ./data/DIG/polish.json.')
parser.add_argument('--train_data_format', type=str, default='MIRAGE', help='The format of the training data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--eval_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/rewrite.json', help='The path to the evaluation data. Default: ./data/DIG/rewrite.json.')
parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--train_batch_size', type=int, default=1, help='The batch size of training data. Default: 1.')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size of evaluation data. Default: 1.')
# Training
parser.add_argument('--learning_rate', type=float, default=1e-4, help='The learning rate. Default: 1e-4.')
parser.add_argument('--num_epochs', type=int, default=5, help='The number of epochs. Default: 5.')
parser.add_argument('--eval_freq', type=int, default=1, help='The frequency of evaluation. Default: 1.')
parser.add_argument('--save_freq', type=int, default=5, help='The frequency of saving the model. Default: 5.')
parser.add_argument('--save_directory', type=str, default='./ckpt/', help='The directory to save the model. Default: ./ckpt/.')
parser.add_argument('--eval', type=bool, default=False, help='Whether to evaluate the model. Default: False.')
# Loss
parser.add_argument('--DDL_target_human_crit', type=float, default=0., help='The target crit of human text when using DDL. Default: 0.')
parser.add_argument('--DDL_target_llm_crit', type=float, default=100., help='The target crit of llm text when using DDL. Default: 1.')
parser.add_argument('--DPO_beta', type=float, default=0.05, help='The beta of DPO. Default: 0.05.')
# Save
parser.add_argument('--ckpt_name', type=str, default=None, help='The name of the saved model. Default: None.')
parser.add_argument('--log_dir', type=str, default='./logs/', help='The directory to save training logs. Default: ./logs/.')


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
    
    # Set up model
    if args.reference_model_name == "None":
        args.reference_model_name = None
    model = DiscrepancyEstimator(scoring_model_name=args.scoring_model_name,
                                 reference_model_name=args.reference_model_name,
                                 train_method=args.train_method)
    lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    model.add_lora_config(lora_config)

    # 只对最后 n 层的 LoRA 参数进行微调：
    # 先根据参数名中出现的层号（如 transformer.h.N）找到最大的层索引，
    # 然后仅保留该层及其之前共 n 层上的 LoRA 参数为可训练，其余全部冻结。
    layer_pattern = re.compile(r"\.h\.(\d+)\.")
    max_layer_idx = -1
    for name, param in model.scoring_model.named_parameters():
        if "lora_" not in name:
            continue
        m = layer_pattern.search(name)
        if m:
            idx = int(m.group(1))
            if idx > max_layer_idx:
                max_layer_idx = idx


    if args.ckpt_name is not None:
        run_name = f'{args.ckpt_name}'
    elif args.train_method == 'DDL':
        run_name = f'DDL_score_{model.scoring_model.config._name_or_path.split("/")[-1]}_ref_{"None" if model.reference_model_name is None else model.reference_model_name.split("/")[-1]}_{args.train_data_path.split("/")[-1].split(".json")[0]}_lr{args.learning_rate}_bs{args.train_batch_size}_rewTgt{args.DDL_target_llm_crit}_oriTgt{args.DDL_target_human_crit}_r{args.lora_rank}'
    else:
        run_name = f'SPO_score_{model.scoring_model.config._name_or_path.split("/")[-1]}_ref_{"None" if model.reference_model_name is None else model.reference_model_name.split("/")[-1]}_{args.train_data_path.split("/")[-1].split(".json")[0]}_lr{args.learning_rate}_bs{args.train_batch_size}_beta{args.DPO_beta}_r{args.lora_rank}'
    
    if fabric.global_rank == 0:
        fabric.print(args)
        model.scoring_model.print_trainable_parameters()

    # Set up dataset
    train_dataset = CustomDataset(data_path=args.train_data_path,
                                  scoring_tokenizer=model.scoring_tokenizer,
                                  reference_tokenizer=model.reference_tokenizer,
                                  data_format=args.train_data_format)
    eval_dataset = CustomDataset(data_path=args.eval_data_path,
                                 scoring_tokenizer=model.scoring_tokenizer,
                                 reference_tokenizer=None,
                                 data_format=args.eval_data_format)

    # Set up loss function
    assert args.train_method in ['DDL', 'SPO'], "Invalid loss function"
    if args.train_method == 'DDL':
        loss_fn = calculate_DDL_loss
    else:
        loss_fn = calculate_DPO_loss


    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.use_deterministic_algorithms(True)
    
    # Set up trainer
    trainer = Trainer()

    trainer.train(fabric=fabric,
                  model=model,
                  train_dataset=train_dataset,
                  eval_dataset=eval_dataset,
                  train_batch_size=args.train_batch_size,
                  eval_batch_size=args.eval_batch_size,
                  loss_fn=loss_fn,
                  learning_rate=args.learning_rate,
                  num_epochs=args.num_epochs,
                  eval_freq=args.eval_freq,
                  save_freq=args.save_freq,
                  save_directory=args.save_directory,
                  DDL_target_human_crit=args.DDL_target_human_crit,
                  DDL_target_llm_crit=args.DDL_target_llm_crit,
                  DPO_beta=args.DPO_beta,
                  save_name=run_name,
                  eval=args.eval,
                  log_dir=args.log_dir)


if __name__ == '__main__':
    args = parser.parse_args()

    # 初始化 Fabric，支持多卡（与您示例类似）
    torch.set_float32_matmul_precision("medium")

    if torch.cuda.is_available():
        device_num = torch.cuda.device_count()
    else:
        device_num = 1

    if device_num > 1:
        ddp_strategy = DDPStrategy(find_unused_parameters=True)
        fabric = Fabric(
            accelerator="cuda",
            precision="bf16-mixed",
            devices=device_num,
            strategy=ddp_strategy,
        )
    else:
        fabric = Fabric(
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
            devices=device_num,
        )

    # 使用 Fabric 多进程启动
    fabric.launch(main, args)