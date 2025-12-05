#!/usr/bin/env python3
"""
HAT 模型 K-Fold 交叉验证训练脚本

功能:
    1. 使用 Stratified K-Fold 划分训练集（保证每折 label 分布接近整体）
    2. 对每个 fold：
       - 训练集 = 其他 K-1 折
       - 验证集 = 当前折
       - 完整训练一个模型
       - 保存为 hat_cls_fold{k}_best.pt
    3. 推理时使用所有 fold 模型进行 ensemble

数据流:
    1. 从 data/processed/train.csv 读取文本和标签
    2. HATDataset 进行分段处理，输出 [N, K] 的 input_ids（不含 CLS）
    3. HATDataCollator 打包 batch: [B, N, K], labels [B]
    4. HATInterleaved512ForClassification 模型内部添加 CLS，返回 {"logits": logits}
    5. 训练脚本统一负责计算 loss（使用带类别权重的 CrossEntropyLoss）

使用方法:
    python scripts/cls_train_kfold.py \
        --train-path data/processed/train.csv \
        --mlm-ckpt checkpoints/mlm_hat512/hat_mlm_final.pt \
        --output-dir checkpoints/cls_hat512_kfold \
        --n-folds 5
"""

import os
import sys
import argparse
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description='HAT 模型 K-Fold 交叉验证训练脚本',
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
        '--class-weights',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'processed' / 'class_weights.npy'),
        help='类别权重文件路径（np.float32[num_labels]）'
    )

    # K-Fold 参数
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='K-Fold 折数（建议 3-5）'
    )
    parser.add_argument(
        '--fold-seed',
        type=int,
        default=42,
        help='K-Fold 划分的随机种子'
    )

    # 预训练权重
    parser.add_argument(
        '--mlm-ckpt',
        type=str,
        default=str(PROJECT_ROOT / 'checkpoints' / 'mlm_hat512' / 'hat_mlm_final.pt'),
        help='MLM 预训练 checkpoint 路径（可选，为空则不加载）'
    )

    # 输出
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(PROJECT_ROOT / 'checkpoints' / 'cls_hat512_kfold'),
        help='分类 checkpoint 输出目录（每个 fold 模型会保存为 hat_cls_fold{k}_best.pt）'
    )

    # 训练超参
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='训练 batch size（H800 建议 64-128）'
    )
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=128,
        help='验证 batch size（H800 建议 128-256）'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='学习率'
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
        default=5,
        help='训练 epoch 数'
    )
    parser.add_argument(
        '--warmup-ratio',
        type=float,
        default=0.06,
        help='warmup 步数比例（相对于总训练步数）'
    )
    parser.add_argument(
        '--log-every',
        type=int,
        default=50,
        help='训练日志打印间隔（step）'
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
        help='DataLoader worker 数（H800 建议 4-8）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（用于训练过程）'
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

        # 模型返回字典格式 {"logits": logits}
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs["logits"]  # [B, num_labels]

        # 使用外部定义的损失函数（带类别权重）
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


def train_single_fold(
    fold_idx: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    full_dataset,
    args,
    device: torch.device,
    class_weights_tensor: torch.Tensor,
    tokenizer,
    segmenter,
    collator,
) -> Tuple[float, Path]:
    """
    训练单个 fold 的模型
    
    Returns:
        best_val_f1: 最佳验证集 F1
        best_model_path: 最佳模型路径
    """
    # 为每个 fold 设置不同的随机种子，增加模型多样性
    fold_seed = args.seed + fold_idx * 1000
    set_seed(fold_seed)
    
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}/{args.n_folds} (seed={fold_seed})")
    print(f"{'='*60}")
    
    # 创建 fold 的训练集和验证集
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    print(f"训练集样本数: {len(train_subset):,}")
    print(f"验证集样本数: {len(val_subset):,}")
    
    # DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False,
    )
    
    print(f"Train batches/epoch: {len(train_loader)}")
    print(f"Val batches/epoch: {len(val_loader)}")
    
    # 创建模型
    from src.model import create_model, HATConfig
    
    config = HATConfig()
    model = create_model(config)  # HATInterleaved512ForClassification
    model.to(device)
    
    # 加载 MLM 预训练权重（可选）
    if args.mlm_ckpt and Path(args.mlm_ckpt).exists():
        print(f"\n加载 MLM 预训练权重: {args.mlm_ckpt}")
        ckpt = torch.load(args.mlm_ckpt, map_location='cpu')
        mlm_state_dict = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = model.load_state_dict(mlm_state_dict, strict=False)
        print(f"  加载完成，missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    else:
        print("\n未提供或找不到 MLM 预训练权重，将从随机初始化开始训练。")
    
    # 损失函数
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # 优化器 & 调度器
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
    
    # 训练循环
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_f1 = 0.0
    global_step = 0
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        print(f"\n========== Fold {fold_idx + 1} | Epoch {epoch+1}/{args.num_epochs} ==========")
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
            logits = outputs["logits"]  # [B, num_labels]
            
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
                    f"[Fold {fold_idx + 1} | Epoch {epoch+1}] "
                    f"Step {global_step}/{total_steps} | "
                    f"Train Loss: {avg_loss:.4f} | "
                    f"LR: {lr_now:.2e} | "
                    f"Speed: {samples_per_sec:.1f} samples/s"
                )
                sys.stdout.flush()
                running_loss = 0.0
        
        # 每个 epoch 结束做一次验证
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, loss_fn=loss_fn)
        print(
            f"\n[Eval] Fold {fold_idx + 1} | Epoch {epoch+1} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )
        sys.stdout.flush()
        
        # 保存最好模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path = output_dir / f"hat_cls_fold{fold_idx}_best.pt"
            torch.save(
                {
                    'fold': fold_idx,
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
    
    elapsed = time.time() - start_time
    print(f"\nFold {fold_idx + 1} 训练完成！总步数: {global_step}, 最佳 Val Macro-F1: {best_val_f1:.4f}")
    print(f"耗时: {elapsed / 60:.1f} 分钟")
    
    # 确保模型已保存
    best_model_path = output_dir / f"hat_cls_fold{fold_idx}_best.pt"
    if not best_model_path.exists():
        # 如果模型未保存，保存当前模型
        print(f"警告: 模型文件不存在，保存当前模型...")
        torch.save(
            {
                'fold': fold_idx,
                'epoch': args.num_epochs - 1,
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'val_macro_f1': best_val_f1,
            },
            best_model_path,
        )
    
    # 清理显存：删除模型、优化器等，释放 GPU 内存
    del model, optimizer, scheduler, loss_fn
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"已清理 GPU 显存")
    
    return best_val_f1, best_model_path


def main():
    args = parse_args()
    
    # 设置全局随机种子（用于训练过程）
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # ========== 1. 读数据 ==========
    train_df = pd.read_csv(args.train_path, sep='\t')
    print(f"训练样本数: {len(train_df):,}")
    
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].astype(int).tolist()
    
    # ========== 2. K-Fold 划分 ==========
    print(f"\n使用 Stratified K-Fold (K={args.n_folds}, seed={args.fold_seed}) 划分数据...")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.fold_seed)
    
    # 获取所有 fold 的划分
    fold_splits = list(skf.split(train_texts, train_labels))
    
    print(f"\n各 Fold 样本分布:")
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        train_labels_fold = np.array(train_labels)[train_idx]
        val_labels_fold = np.array(train_labels)[val_idx]
        train_label_dist = np.bincount(train_labels_fold)
        val_label_dist = np.bincount(val_labels_fold)
        print(f"  Fold {fold_idx + 1}: 训练集 {len(train_idx):,} 样本, 验证集 {len(val_idx):,} 样本")
        print(f"    训练集标签分布: {train_label_dist}")
        print(f"    验证集标签分布: {val_label_dist}")
    
    # ========== 3. 创建完整数据集 ==========
    from src.data_preprocess import (
        HATDataset,
        HATDataCollator,
        create_tokenizer,
        create_segmenter,
    )
    
    tokenizer = create_tokenizer()
    segmenter = create_segmenter()
    collator = HATDataCollator()
    
    # 创建完整数据集（包含所有训练样本）
    full_dataset = HATDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        segmenter=segmenter,
        mode='train',
        cache_segments=False,
    )
    
    # ========== 4. 类别权重 ==========
    if not Path(args.class_weights).exists():
        raise FileNotFoundError(f"类别权重文件不存在: {args.class_weights}")
    
    class_weights = np.load(args.class_weights).astype('float32')  # [num_labels]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"\n类别权重示例: {class_weights[:5]} ...")
    
    # ========== 5. 训练每个 Fold ==========
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 在开始训练前清理显存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print(f"\n已清理初始显存，准备开始 K-Fold 训练...")
    
    fold_results = []
    total_start_time = time.time()
    
    for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
        best_val_f1, best_model_path = train_single_fold(
            fold_idx=fold_idx,
            train_indices=train_indices,
            val_indices=val_indices,
            full_dataset=full_dataset,
            args=args,
            device=device,
            class_weights_tensor=class_weights_tensor,
            tokenizer=tokenizer,
            segmenter=segmenter,
            collator=collator,
        )
        fold_results.append({
            'fold': fold_idx,
            'best_val_f1': best_val_f1,
            'model_path': best_model_path,
        })
        
        # 在每个 fold 之间清理显存，确保下一个 fold 有足够的显存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            print(f"\n已清理显存，准备训练下一个 fold...")
    
    # ========== 6. 总结 ==========
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"K-Fold 训练完成！")
    print(f"{'='*60}")
    print(f"总耗时: {total_elapsed / 60:.1f} 分钟")
    print(f"\n各 Fold 最佳验证集 Macro-F1:")
    
    all_f1s = []
    for result in fold_results:
        fold_idx = result['fold']
        best_f1 = result['best_val_f1']
        model_path = result['model_path']
        all_f1s.append(best_f1)
        print(f"  Fold {fold_idx + 1}: {best_f1:.4f} -> {model_path}")
    
    mean_f1 = np.mean(all_f1s)
    std_f1 = np.std(all_f1s)
    print(f"\n平均 Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"\n所有模型已保存到: {output_dir}")
    print(f"\n推理时使用以下命令进行 Ensemble:")
    print(f"  python scripts/infer.py \\")
    print(f"    --test-path data/processed/test.csv \\")
    print(f"    --model-paths {output_dir}/hat_cls_fold0_best.pt,{output_dir}/hat_cls_fold1_best.pt,... \\")
    print(f"    --output-path outputs/submission/submission_kfold.csv")


if __name__ == "__main__":
    main()

