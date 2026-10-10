import argparse
import os
import random
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from dataset import CustomDataset
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(
    train_data_path: str,
    train_data_format: str,
    eval_data_path: Optional[str],
    eval_data_format: str,
    batch_size: int,
):
    train_dataset = CustomDataset(data_path=train_data_path, data_format=train_data_format)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    eval_loader = None
    if eval_data_path is not None and eval_data_path != "":
        eval_dataset = CustomDataset(data_path=eval_data_path, data_format=eval_data_format)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=eval_dataset.collate_fn,
        )
    return train_loader, eval_loader


def train_epoch(model, tokenizer, loader, device, max_seq_len: int, optimizer, scheduler=None):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        human_texts = batch["human"]
        llm_texts = batch["llm"]

        texts = human_texts + llm_texts
        
        # 对于openai-community/roberta-base-openai-detector
        # print(model.config.id2label)
        # {0: 'Fake', 1: 'Real'}
        labels = torch.tensor(
            [1] * len(human_texts) + [0] * len(llm_texts), dtype=torch.long, device=device
        )

        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        ).to(device)

        outputs = model(**encodings, labels=labels)
        loss = outputs.loss
        # DataParallel 情况下各卡返回一个标量 loss，
        # gather 后会变成一维张量，这里统一取均值再反传。
        if loss.dim() > 0:
            loss = loss.mean()
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_examples += labels.size(0)

    avg_loss = total_loss / max(total_examples, 1)
    avg_acc = total_correct / max(total_examples, 1)
    return avg_loss, avg_acc


def main(args):
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    train_loader, eval_loader = build_dataloaders(
        train_data_path=args.train_data_path,
        train_data_format=args.train_data_format,
        eval_data_path=args.eval_data_path,
        eval_data_format=args.eval_data_format,
        batch_size=args.batch_size,
    )

    total_steps = args.num_epochs * len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}")
        train_loss, train_acc = train_epoch(
            model,
            tokenizer,
            train_loader,
            device,
            max_seq_len=args.max_sequence_length,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        print(f"  Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")


    # 最后再保存一份 "last" 模型
    last_dir = os.path.join(args.output_dir, "last")
    os.makedirs(last_dir, exist_ok=True)
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    model_to_save.save_pretrained(last_dir)
    tokenizer.save_pretrained(last_dir)
    print(f"Training finished. Last model saved to {last_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="roberta-base", help="HF 模型名或本地路径")
    parser.add_argument("--train_data_path", type=str, default="./data/ai_detection_500_polish.raw_data.json")
    parser.add_argument("--train_data_format", type=str, default="ImBD", help="MIRAGE 或 ImBD")
    parser.add_argument("--eval_data_path", type=str, default="", help="可选：验证集路径")
    parser.add_argument("--eval_data_format", type=str, default="ImBD")
    parser.add_argument("--output_dir", type=str, default="./model/roberta_detector")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
