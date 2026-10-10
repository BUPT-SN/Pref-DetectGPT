import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os

from torch.utils.data import DataLoader
from .model import DiscrepancyEstimator
from .dataset import CustomDataset
from .loss import calculate_DPO_loss, calculate_DDL_loss
from .metrics import AUROC, AUPR, TPR_at_FPR5
from lightning.fabric import Fabric
from tqdm import tqdm

class Trainer():
    def train(self,
              fabric: Fabric,
              model: DiscrepancyEstimator,
              train_dataset: CustomDataset,
              eval_dataset: CustomDataset,
              loss_fn = calculate_DDL_loss,
              learning_rate: float = 1e-4,
              num_epochs: int = 5,
              eval_freq: int = 1,
              save_freq: int = 5,
              save_directory: str='./ckpt/',
              save_name: str = None,
              DDL_target_human_crit: float = 0.,
              DDL_target_llm_crit: float = 100.,
              DPO_beta: float = 0.05,
              train_batch_size: int = 1,
              eval_batch_size: int = 1,
              eval: bool = False,
              log_dir: str = './logs/',):
        assert loss_fn in [calculate_DDL_loss, calculate_DPO_loss], "Invalid loss function"
        start_time = time.time()
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                                  collate_fn=train_dataset.collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False,
                                 collate_fn=eval_dataset.collate_fn)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=num_epochs * len(train_loader),
                                                            eta_min=0,
                                                            last_epoch=-1)
        # 使用 Fabric 做分布式准备
        model, optimizer = fabric.setup(model, optimizer)
        train_loader, eval_loader = fabric.setup_dataloaders(train_loader, eval_loader)

        # 日志文件：主进程将每个 epoch 的训练/验证指标追加写入本地日志
        log_file = None
        if fabric.global_rank == 0 and save_name is not None and log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            # 使用 .log 后缀，但内部为 CSV 结构，便于后处理
            log_file = os.path.join(log_dir, f"{save_name}.log")

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            train_losses = []
            epoch_human_discrepancy_train = []
            epoch_llm_discrepancy_train = []
            model.train()
            for step, batch in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Fine-tuning: {epoch+1} epoch",
                disable=not (fabric.global_rank == 0),
            ):
                outputs = model(batch['scoring']['human_input_ids'],
                                batch['scoring']['human_attention_mask'],
                                batch['scoring']['llm_input_ids'],
                                batch['scoring']['llm_attention_mask'],
                                batch['reference']['human_input_ids'],
                                batch['reference']['human_attention_mask'],
                                batch['reference']['llm_input_ids'],
                                batch['reference']['llm_attention_mask'])
                if loss_fn == calculate_DPO_loss:
                    loss, _, _, _, _ = loss_fn(
                        outputs['scoring_llm_logprob'],
                        outputs['scoring_human_logprob'],
                        outputs['reference_llm_logprob'],
                        outputs['reference_human_logprob'],
                        beta=DPO_beta)
                else:
                    loss = loss_fn(outputs['scoring_human_discrepancy'],
                                   outputs['scoring_llm_discrepancy'],
                                   target_human_crit=DDL_target_human_crit,
                                   target_llm_crit=DDL_target_llm_crit)
                optimizer.zero_grad()
                fabric.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                train_losses.append(loss.detach().cpu().item())
                epoch_human_discrepancy_train.extend(outputs['scoring_human_discrepancy'].detach().cpu().tolist())
                epoch_llm_discrepancy_train.extend(outputs['scoring_llm_discrepancy'].detach().cpu().tolist())

            fabric.barrier()
            all_train_losses = fabric.all_gather(torch.tensor(train_losses, device=fabric.device)).flatten()
            all_human_train = fabric.all_gather(torch.tensor(epoch_human_discrepancy_train, device=fabric.device)).flatten()
            all_llm_train = fabric.all_gather(torch.tensor(epoch_llm_discrepancy_train, device=fabric.device)).flatten()
            if fabric.global_rank == 0:
                fpr, tpr, train_epoch_auroc = AUROC(all_human_train.cpu().tolist(), all_llm_train.cpu().tolist())
                train_epoch_tpr_at_5 = TPR_at_FPR5(all_human_train.cpu().tolist(), all_llm_train.cpu().tolist())
                # AUPR 仍保留在日志中，便于 wandb/CSV 对比，但不再打印
                prec, recall, train_epoch_aupr = AUPR(all_human_train.cpu().tolist(), all_llm_train.cpu().tolist())
                epoch_loss = torch.mean(all_train_losses).item()
                human_discrepancy_mean = torch.mean(all_human_train).item()
                human_discrepancy_std = torch.std(all_human_train).item()
                llm_discrepancy_mean = torch.mean(all_llm_train).item()
                llm_discrepancy_std = torch.std(all_llm_train).item()

                # Prepare log dictionary
                log_dict = {
                    "train/auroc": train_epoch_auroc,
                    "train/tpr_at_fpr_5": train_epoch_tpr_at_5,
                    "train/aupr": train_epoch_aupr,
                    "train/human_discrepancy_mean": human_discrepancy_mean,
                    "train/human_discrepancy_std": human_discrepancy_std,
                    "train/llm_discrepancy_mean": llm_discrepancy_mean,
                    "train/llm_discrepancy_std": llm_discrepancy_std,
                }

                fabric.print(f'Epoch: {epoch + 1} | Time: {time.time() - epoch_start_time:.3f} sec')
                fabric.print(f'Train Loss: {epoch_loss:.8f}')
                fabric.print(f'Train AUROC: {train_epoch_auroc:.4f} | Train TPR@FPR=5%: {train_epoch_tpr_at_5:.4f}')
            
            if eval:
                if (epoch + 1) % eval_freq == 0 or (epoch + 1) == num_epochs:
                    epoch_human_discrepancy_eval, epoch_llm_discrepancy_eval = self.eval(
                        model, eval_loader, show_progress_bar=(fabric.global_rank == 0)
                    )
                    fabric.barrier()
                    all_human_eval = fabric.all_gather(torch.tensor(epoch_human_discrepancy_eval, device=fabric.device)).flatten()
                    all_llm_eval = fabric.all_gather(torch.tensor(epoch_llm_discrepancy_eval, device=fabric.device)).flatten()
                    if fabric.global_rank == 0:
                        fpr, tpr, eval_epoch_auroc = AUROC(all_human_eval.cpu().tolist(), all_llm_eval.cpu().tolist())
                        eval_epoch_tpr_at_5 = TPR_at_FPR5(all_human_eval.cpu().tolist(), all_llm_eval.cpu().tolist())
                        prec, recall, eval_epoch_aupr = AUPR(all_human_eval.cpu().tolist(), all_llm_eval.cpu().tolist())
                        human_discrepancy_mean = torch.mean(all_human_eval).item()
                        human_discrepancy_std = torch.std(all_human_eval).item()
                        llm_discrepancy_mean = torch.mean(all_llm_eval).item()
                        llm_discrepancy_std = torch.std(all_llm_eval).item()
                        log_dict.update({
                            "eval/auroc": eval_epoch_auroc,
                            "eval/tpr_at_fpr_5": eval_epoch_tpr_at_5,
                            "eval/aupr": eval_epoch_aupr,
                            "eval/human_discrepancy_mean": human_discrepancy_mean,
                            "eval/human_discrepancy_std": human_discrepancy_std,
                            "eval/llm_discrepancy_mean": llm_discrepancy_mean,
                            "eval/llm_discrepancy_std": llm_discrepancy_std,
                        })
                        fabric.print(f'Eval AUROC: {eval_epoch_auroc:.4f} | Eval TPR@FPR=5%: {eval_epoch_tpr_at_5:.4f}')
                fabric.barrier()

            # 将本 epoch 的关键指标追加写入本地日志文件（仅 AUROC 和 TPR@FPR=5%）
            if fabric.global_rank == 0 and log_file is not None:
                # 只保留关心的两个指标（分别记录 train 和 eval），外加 epoch 方便对齐
                eval_auroc_val = log_dict.get("eval/auroc", "") if 'log_dict' in locals() else ""
                eval_tpr5_val = log_dict.get("eval/tpr_at_fpr_5", "") if 'log_dict' in locals() else ""

                header = (
                    "epoch,"
                    "train_auroc,train_tpr_at_fpr_5,"
                    "eval_auroc,eval_tpr_at_fpr_5\n"
                )
                line = (
                    f"{epoch + 1},"
                    f"{train_epoch_auroc:.6f},{train_epoch_tpr_at_5:.6f},"
                    f"{eval_auroc_val if eval_auroc_val != '' else ''},"
                    f"{eval_tpr5_val if eval_tpr5_val != '' else ''}\n"
                )

                # 若文件不存在或为空，则先写入表头
                need_header = not os.path.exists(log_file) or os.path.getsize(log_file) == 0
                with open(log_file, "a", encoding="utf-8") as f:
                    if need_header:
                        f.write(header)
                    f.write(line)

            if (epoch + 1) == num_epochs or (epoch + 1) % save_freq == 0:
                fabric.barrier()
                fabric.print('saving model ...')
                if save_name is None:
                    raise ValueError('save_name should not be None')
                this_epoch_save_name = f'{save_name}_e{epoch+1}'
                if not os.path.exists(os.path.join(save_directory, this_epoch_save_name)):
                    os.makedirs(os.path.join(save_directory, this_epoch_save_name), exist_ok=True)
                if fabric.global_rank == 0:
                    # 某些 lightning.fabric 版本没有 unwrap 接口，这里直接使用当前模型对象的保存方法
                    model.save_pretrained(os.path.join(save_directory, this_epoch_save_name))

            if (epoch + 1) == num_epochs:
                if fabric.global_rank == 0:
                    fabric.print(f'Finished Training!')
                    fabric.print(f'Total Time: {time.time() - start_time:.3f} sec')

    def eval(self,
            model,
            eval_loader,
            show_progress_bar: bool = True):
        
        epoch_human_discrepancy_eval = []
        epoch_llm_discrepancy_eval = []
        model.eval()

        # --- 新增：获取当前显卡的 Rank ---
        # 如果系统环境变量里没有 RANK，默认设为 0 (单卡情况)
        rank = int(os.environ.get("RANK", 0))
        
        with torch.no_grad():
            # --- 修改：tqdm 参数 ---
            # 1. desc: 显示当前是哪个 Rank
            # 2. position: 第几张卡就在第几行，避免重叠
            # 3. leave=True: 跑完后保留进度条
            iterator = tqdm(
                enumerate(eval_loader),
                total=len(eval_loader),
                desc=f"Eval", 
                disable=not show_progress_bar,
                position=rank, 
                leave=True
            )

            for step, batch in iterator:
                outputs = model(
                    batch['scoring']['human_input_ids'],
                    batch['scoring']['human_attention_mask'],
                    batch['scoring']['llm_input_ids'],
                    batch['scoring']['llm_attention_mask'],
                    batch['reference']['human_input_ids'],
                    batch['reference']['human_attention_mask'],
                    batch['reference']['llm_input_ids'],
                    batch['reference']['llm_attention_mask']
                )
                epoch_human_discrepancy_eval.extend(outputs['scoring_human_discrepancy'].cpu().tolist())
                epoch_llm_discrepancy_eval.extend(outputs['scoring_llm_discrepancy'].cpu().tolist())

        return epoch_human_discrepancy_eval, epoch_llm_discrepancy_eval