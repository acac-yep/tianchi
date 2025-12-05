#!/usr/bin/env python3
"""
HAT 模型多样性微调脚本

关键改进点：
1. 更激进的学习率 (5e-5 ~ 1e-4)
2. 支持冻结底层 encoder
3. 支持不同的 dropout 配置
4. 支持不同的滑窗策略
5. 禁止在 baseline 时保存（必须有实际训练改进）

使用方法:
    python scripts/cls_finetune_diverse.py \
        --pretrained-ckpt checkpoints/cls_hat512/hat_cls_best.pt \
        --output-dir checkpoints/diverse_ensemble/seed42_high_lr \
        --lr 5e-5 \
        --freeze-layers 4 \
        --dropout 0.15 \
        --seed 42
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
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, accuracy_score

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description='HAT 模型多样性微调脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据
    parser.add_argument('--train-path', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'processed' / 'train.csv'))
    parser.add_argument('--val-path', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'processed' / 'val.csv'))
    parser.add_argument('--class-weights', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'processed' / 'class_weights.npy'))

    # 预训练权重
    parser.add_argument('--pretrained-ckpt', type=str, required=True,
                        help='Stage 1 Best Checkpoint 路径')

    # 输出
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出目录')

    # ========== 多样性控制参数 ==========
    # 学习率（更激进）
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='学习率 (建议 5e-5 ~ 1e-4)')
    
    # 冻结层数
    parser.add_argument('--freeze-layers', type=int, default=0,
                        help='冻结前 N 层 encoder (0=不冻结, 4=冻结一半)')
    
    # Dropout 配置
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Hidden dropout (建议 0.1 ~ 0.25)')
    parser.add_argument('--attention-dropout', type=float, default=0.1,
                        help='Attention dropout')
    
    # 滑窗配置（segment 级别）
    parser.add_argument('--sliding-stride', type=int, default=4,
                        help='滑窗步长 (segment 数量，默认 4)')
    
    # 损失函数
    parser.add_argument('--loss-type', type=str,
                        choices=['ce', 'focal', 'focal_smooth'],
                        default='focal')
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--label-smoothing', type=float, default=0.1)

    # 训练超参（更短、更大胆）
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--num-epochs', type=int, default=5,
                        help='训练轮数 (建议 3-5)')
    parser.add_argument('--warmup-ratio', type=float, default=0.1)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--early-patience', type=int, default=3)
    parser.add_argument('--min-improvement', type=float, default=0.0005,
                        help='最小提升阈值 (必须超过 baseline 才保存)')

    # 设备 & 其他
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad-clip', type=float, default=1.0)

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


def freeze_encoder_layers(model, num_layers_to_freeze: int):
    """
    冻结 encoder 的前 N 层
    
    HAT 模型结构:
    - model.segment_encoder.embeddings
    - model.segment_encoder.encoder.layer[0..N]  (shared BERT layers)
    - model.document_encoder (document-level attention)
    - model.classifier
    """
    if num_layers_to_freeze <= 0:
        return
    
    print(f"\n冻结前 {num_layers_to_freeze} 层 encoder...")
    
    # 冻结 embeddings
    if hasattr(model, 'segment_encoder') and hasattr(model.segment_encoder, 'embeddings'):
        for param in model.segment_encoder.embeddings.parameters():
            param.requires_grad = False
        print("  冻结: segment_encoder.embeddings")
    
    # 冻结指定层数的 encoder layers
    if hasattr(model, 'segment_encoder') and hasattr(model.segment_encoder, 'encoder'):
        encoder_layers = model.segment_encoder.encoder.layer
        for i in range(min(num_layers_to_freeze, len(encoder_layers))):
            for param in encoder_layers[i].parameters():
                param.requires_grad = False
            print(f"  冻结: segment_encoder.encoder.layer[{i}]")
    
    # 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")


def modify_model_dropout(model, hidden_dropout: float, attention_dropout: float):
    """
    修改模型的 dropout 配置
    """
    print(f"\n设置 dropout: hidden={hidden_dropout}, attention={attention_dropout}")
    
    modified_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = hidden_dropout
            modified_count += 1
    
    print(f"  修改了 {modified_count} 个 Dropout 层")


@torch.no_grad()
def evaluate(model, data_loader, device, loss_fn):
    """在验证集上计算 loss / accuracy / macro-F1"""
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    total_steps = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs["logits"]

        loss = loss_fn(logits, labels)

        total_loss += loss.item()
        total_steps += 1

        preds = logits.argmax(dim=-1)
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

    # ========== 打印多样性配置 ==========
    print("\n" + "=" * 60)
    print("多样性配置")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Learning Rate: {args.lr}")
    print(f"Freeze Layers: {args.freeze_layers}")
    print(f"Dropout: {args.dropout}")
    print(f"Attention Dropout: {args.attention_dropout}")
    print(f"Sliding Stride: {args.sliding_stride}")
    print(f"Loss Type: {args.loss_type}")
    print("=" * 60)

    # ========== 1. 读数据 ==========
    print(f"\nLoading data from {args.train_path} and {args.val_path}")
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

    tokenizer = create_tokenizer()
    
    # 使用自定义滑窗配置
    from src.data_preprocess.config import SegmenterConfig
    segmenter_config = SegmenterConfig(
        sliding_window_stride=args.sliding_stride,
    )
    segmenter = create_segmenter(config=segmenter_config)
    print(f"\n滑窗配置: sliding_stride={args.sliding_stride} segments")

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

    # ========== 4. 加载预训练权重 ==========
    if Path(args.pretrained_ckpt).exists():
        print(f"\n加载 Stage 1 Best Checkpoint: {args.pretrained_ckpt}")
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        model_state_dict = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
        print(f"  加载完成，missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    else:
        raise FileNotFoundError(f"找不到预训练权重: {args.pretrained_ckpt}")

    # ========== 5. 应用多样性配置 ==========
    # 5.1 冻结层
    freeze_encoder_layers(model, args.freeze_layers)
    
    # 5.2 修改 dropout
    modify_model_dropout(model, args.dropout, args.attention_dropout)

    # ========== 6. 创建损失函数 ==========
    from src.losses import create_loss_fn

    if not Path(args.class_weights).exists():
        raise FileNotFoundError(f"类别权重文件不存在: {args.class_weights}")

    class_weights = np.load(args.class_weights).astype('float32')
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    loss_fn = create_loss_fn(
        loss_type=args.loss_type,
        class_weights=class_weights_tensor,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        device=str(device),
    )

    print(f"\n损失函数: {args.loss_type}")
    if 'focal' in args.loss_type:
        print(f"  Focal gamma: {args.focal_gamma}")

    # ========== 7. 优化器 & 调度器 ==========
    # 只优化可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        trainable_params,
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
    print(f"最小提升阈值: {args.min_improvement}")

    # ========== 8. 训练循环 ==========
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始评估（获取 baseline）
    print("\n[Baseline Eval] ...")
    baseline_loss, baseline_acc, baseline_f1 = evaluate(model, val_loader, device, loss_fn=loss_fn)
    print(f"Baseline Val Macro-F1: {baseline_f1:.4f}")
    
    best_val_f1 = baseline_f1
    no_improve_epochs = 0
    model_saved = False

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
        improvement = val_f1 - baseline_f1
        
        print(
            f"\n[Eval] Epoch {epoch+1} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_f1:.4f} | "
            f"vs Baseline: {improvement:+.4f}"
        )

        # Checkpoint - 必须超过 baseline + min_improvement
        if val_f1 > best_val_f1 + args.min_improvement:
            best_val_f1 = val_f1
            no_improve_epochs = 0
            
            # 只有真正比 baseline 好才保存
            if val_f1 > baseline_f1 + args.min_improvement:
                best_path = output_dir / "hat_cls_best.pt"
                torch.save(
                    {
                        'epoch': epoch,
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_macro_f1': val_f1,
                        'baseline_f1': baseline_f1,
                        'improvement': val_f1 - baseline_f1,
                        'loss_type': args.loss_type,
                        'focal_gamma': args.focal_gamma,
                        'seed': args.seed,
                        'lr': args.lr,
                        'freeze_layers': args.freeze_layers,
                        'dropout': args.dropout,
                        'attention_dropout': args.attention_dropout,
                        'sliding_stride': args.sliding_stride,
                    },
                    best_path,
                )
                model_saved = True
                print(f"  >> 新最佳模型 (超过 baseline {val_f1-baseline_f1:+.4f})，已保存到: {best_path}")
            else:
                print(f"  >> F1 提升但未超过 baseline 足够多，不保存")
        else:
            no_improve_epochs += 1
            print(f"  >> Macro-F1 未提升 (连续 {no_improve_epochs} epoch)")
            if no_improve_epochs >= args.early_patience:
                print(f"\nEarly stopping triggered! (No improvement for {args.early_patience} epochs)")
                break

    elapsed = time.time() - start_time
    print(f"\n训练完成！总步数: {global_step}")
    print(f"Baseline F1: {baseline_f1:.4f}")
    print(f"最佳 Val F1: {best_val_f1:.4f}")
    print(f"提升: {best_val_f1 - baseline_f1:+.4f}")
    print(f"总耗时: {elapsed / 60:.1f} 分钟")
    
    if not model_saved:
        print("\n⚠️  警告: 训练没有产生超过 baseline 的模型！")
        print("   可能需要调整超参数 (增大 lr / 减少 freeze_layers / 增加 epochs)")
        
        # 强制保存最后一个 checkpoint（带警告标记）
        last_path = output_dir / "hat_cls_last.pt"
        torch.save(
            {
                'epoch': epoch,
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_macro_f1': val_f1,
                'baseline_f1': baseline_f1,
                'improvement': val_f1 - baseline_f1,
                'warning': 'NOT_BETTER_THAN_BASELINE',
                'seed': args.seed,
            },
            last_path,
        )
        print(f"   最后一个 checkpoint 已保存到: {last_path}")


if __name__ == "__main__":
    main()

