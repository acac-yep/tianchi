#!/usr/bin/env python3
"""
HAT 模型第二阶段微调脚本 (Stage 2 Fine-tuning)

基于 cls_train.py 修改，主要特点：
1. 从 best ckpt (checkpoints/cls_hat512/hat_cls_best.pt) 加载权重
2. 重新初始化 optimizer 和 scheduler (不加载旧状态)
3. 更小的 learning_rate (3e-5) 和 warmup_ratio (0.05)
4. 增加 Early Stopping 机制 (基于 Macro-F1)

使用方法:
    python scripts/cls_finetune_stage2.py \
        --pretrained-ckpt checkpoints/cls_hat512/hat_cls_best.pt \
        --output-dir checkpoints/cls_hat512_stage2 \
        --lr 3e-5 --warmup-ratio 0.05 --num-epochs 3 --early-patience 2
"""

import os
import sys
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, accuracy_score

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description='HAT 模型分类第二阶段微调脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据
    parser.add_argument(
        '--train-path',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'processed' / 'train.csv'),
        help='训练数据路径（预处理后的 train.csv）'
    )
    parser.add_argument(
        '--val-path',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'processed' / 'val.csv'),
        help='验证数据路径（预处理后的 val.csv）'
    )
    parser.add_argument(
        '--class-weights',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'processed' / 'class_weights.npy'),
        help='类别权重文件路径（np.float32[num_labels]）'
    )

    # 预训练权重 (Stage 1 Best Checkpoint)
    parser.add_argument(
        '--pretrained-ckpt',
        type=str,
        required=True,
        help='第一阶段微调最好的 checkpoint 路径 (e.g. checkpoints/cls_hat512/hat_cls_best.pt)'
    )

    # 输出
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(PROJECT_ROOT / 'checkpoints' / 'cls_hat512_stage2'),
        help='第二阶段 checkpoint 输出目录'
    )

    # 训练超参 (默认值已调整为 Stage 2 建议值)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='训练 batch size'
    )
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=128,
        help='验证 batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-5,
        help='学习率 (建议 3e-5)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='权重衰减'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=3,
        help='训练 epoch 数 (建议 2-3)'
    )
    parser.add_argument(
        '--warmup-ratio',
        type=float,
        default=0.05,
        help='warmup 步数比例 (建议 0.05)'
    )
    parser.add_argument(
        '--log-every',
        type=int,
        default=50,
        help='训练日志打印间隔（step）'
    )
    parser.add_argument(
        '--early-patience',
        type=int,
        default=2,
        help='Early Stopping patience (连续 N 个 epoch Macro-F1 不提升则停止)'
    )

    # 设备 & 其他
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda / cpu）'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='DataLoader worker 数'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    parser.add_argument(
        '--grad-clip',
        type=float,
        default=1.0,
        help='梯度裁剪阈值（0 表示不裁剪）'
    )

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps)),
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, data_loader, device, loss_fn):
    """在验证集上计算 loss / accuracy / macro-F1"""
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    total_steps = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)      # [B, N, K]
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)            # [B]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs["logits"]  # [B, num_labels]

        loss = loss_fn(logits, labels)

        total_loss += loss.item()
        total_steps += 1

        preds = logits.argmax(dim=-1)                 # [B]
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    avg_loss = total_loss / max(1, total_steps)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, acc, macro_f1


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ========== 1. 读数据 ==========
    print(f"Loading data from {args.train_path} and {args.val_path}")
    train_df = pd.read_csv(args.train_path, sep='\t')
    val_df = pd.read_csv(args.val_path, sep='\t')

    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].astype(int).tolist()
    val_texts = val_df['text'].tolist()
    val_labels = val_df['label'].astype(int).tolist()

    # ========== 2. Dataset & DataLoader ==========
    from src.data_preprocess import (
        HATDataset,
        HATDataCollator,
        create_tokenizer,
        create_segmenter,
    )
    # from src.common_config import COMMON_CONFIG # unused

    tokenizer = create_tokenizer()
    segmenter = create_segmenter()

    train_dataset = HATDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        segmenter=segmenter,
        mode='train',
        cache_segments=False,
    )
    val_dataset = HATDataset(
        texts=val_texts,
        labels=val_labels,
        tokenizer=tokenizer,
        segmenter=segmenter,
        mode='eval',
        cache_segments=False,
    )

    collator = HATDataCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False,
    )

    # ========== 3. 创建模型 ==========
    from src.model import create_model, HATConfig

    config = HATConfig()
    model = create_model(config)
    model.to(device)

    # ========== 4. 加载预训练权重 (Stage 1 Best) ==========
    if Path(args.pretrained_ckpt).exists():
        print(f"\n加载 Stage 1 Best Checkpoint: {args.pretrained_ckpt}")
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        model_state_dict = ckpt.get('model_state_dict', ckpt)
        # 这里已经是分类模型，所以大概率 keys 完全匹配
        missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
        print(f"  加载完成，missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    else:
        raise FileNotFoundError(f"找不到预训练权重: {args.pretrained_ckpt}")

    # ========== 5. 类别权重 & 优化器 & 调度器 (重新构建) ==========
    if not Path(args.class_weights).exists():
        raise FileNotFoundError(f"类别权重文件不存在: {args.class_weights}")
    
    class_weights = np.load(args.class_weights).astype('float32')
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\n总训练步数: {total_steps}, warmup 步数: {warmup_steps}")
    print(f"学习率: {args.lr}, weight_decay: {args.weight_decay}")
    print(f"Early Patience: {args.early_patience} epochs")

    # ========== 6. 训练循环 ==========
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 Early Stopping 变量
    best_val_f1 = 0.0
    no_improve_epochs = 0
    
    # 初始评估 (Baseline)
    print("\n[Baseline Eval] ...")
    val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, loss_fn=loss_fn)
    print(f"Baseline Val Macro-F1: {val_f1:.4f}")
    best_val_f1 = val_f1 # 以加载的模型作为起点

    global_step = 0
    start_time = time.time()

    for epoch in range(args.num_epochs):
        print(f"\n========== Epoch {epoch+1}/{args.num_epochs} ==========")
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs["logits"]

            loss = loss_fn(logits, labels)

            if torch.isnan(loss):
                print(f"警告: step {global_step} 出现 NaN loss，跳过")
                continue

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_loss += loss.item()

            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                samples_per_sec = global_step * args.batch_size / elapsed

                print(
                    f"[Epoch {epoch+1}] "
                    f"Step {global_step}/{total_steps} | "
                    f"Train Loss: {avg_loss:.4f} | "
                    f"LR: {lr_now:.2e} | "
                    f"Speed: {samples_per_sec:.1f} samples/s"
                )
                sys.stdout.flush()
                running_loss = 0.0

        # ===== Epoch Eval =====
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, loss_fn=loss_fn)
        print(
            f"\n[Eval] Epoch {epoch+1} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        # Checkpoint & Early Stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_epochs = 0
            best_path = output_dir / "hat_cls_best.pt"
            torch.save(
                {
                    'epoch': epoch,
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_macro_f1': val_f1,
                },
                best_path,
            )
            print(f"  >> 新最佳模型，已保存到: {best_path}")
        else:
            no_improve_epochs += 1
            print(f"  >> Macro-F1 未提升 (连续 {no_improve_epochs} epoch)")
            if no_improve_epochs >= args.early_patience:
                print(f"\nEarly stopping triggered! (No improvement for {args.early_patience} epochs)")
                break

    elapsed = time.time() - start_time
    print(f"\nStage 2 微调完成！总步数: {global_step}, 最佳 Val Macro-F1: {best_val_f1:.4f}")
    print(f"总耗时: {elapsed / 60:.1f} 分钟")


if __name__ == "__main__":
    main()

