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
    5. 训练脚本统一负责计算 loss（支持多种损失函数：CE / Label Smoothing / Focal Loss）

正则化策略:
    - Label Smoothing: 防止模型过度自信，提升泛化能力（默认 0.05）
    - Focal Loss: 关注难样本，适合类别不平衡（可选）
    - WeightedRandomSampler: 采样层面的类别平衡（可选）
    - Early Stopping: 防止过拟合，节省训练时间（可选）

模型增强:
    - AMP (混合精度): 加速训练并节省显存，通常还能轻微提升泛化（可选）
    - EMA (指数移动平均): 维护模型权重的移动平均，验证时使用 EMA 模型（可选）

使用方法:
    # 基础用法（Label Smoothing + 类别权重）
    python scripts/cls_train_kfold.py \
        --train-path data/processed/train.csv \
        --mlm-ckpt checkpoints/mlm_hat512/hat_mlm_final.pt \
        --output-dir checkpoints/cls_hat512_kfold \
        --n-folds 5 \
        --loss-type smooth \
        --label-smoothing 0.05

    # 使用 AMP + EMA（推荐）
    python scripts/cls_train_kfold.py \
        --use-amp \
        --use-ema \
        --ema-decay 0.9999

    # 使用 Focal Loss + Label Smoothing + AMP + EMA
    python scripts/cls_train_kfold.py \
        --loss-type focal_smooth \
        --label-smoothing 0.05 \
        --focal-gamma 2.0 \
        --use-amp \
        --use-ema
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
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.cuda.amp import GradScaler

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入损失函数模块
from src.losses import create_loss_fn


# =============================================================================
# 日志工具
# =============================================================================

def log_print(*args, **kwargs):
    """带时间戳的 print 函数"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}]", *args, **kwargs)


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
        default=3e-5,
        help='学习率（默认 3e-5）'
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

    # 损失函数相关
    parser.add_argument(
        '--loss-type',
        type=str,
        default='smooth',
        choices=['ce', 'smooth', 'focal', 'focal_smooth'],
        help='损失函数类型: ce=标准交叉熵, smooth=标签平滑, focal=Focal Loss, focal_smooth=Focal Loss+标签平滑'
    )
    parser.add_argument(
        '--label-smoothing',
        type=float,
        default=0.05,
        help='标签平滑系数（0-1，推荐 0.05 或 0.1）'
    )
    parser.add_argument(
        '--focal-gamma',
        type=float,
        default=2.0,
        help='Focal Loss 的聚焦参数 gamma（推荐 2.0）'
    )

    # 采样策略
    parser.add_argument(
        '--use-weighted-sampler',
        action='store_true',
        help='使用 WeightedRandomSampler 进行类别平衡采样（与 class_weights 配合使用）'
    )

    # Early Stopping
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=None,
        help='Early stopping patience（连续 N 个 epoch val F1 无提升则停止，None 表示不使用）'
    )

    # AMP (Automatic Mixed Precision)
    parser.add_argument(
        '--use-amp',
        action='store_true',
        help='使用混合精度训练（AMP），可加速训练并节省显存'
    )

    # EMA (Exponential Moving Average)
    parser.add_argument(
        '--use-ema',
        action='store_true',
        help='使用指数移动平均（EMA）权重，验证时使用 EMA 模型'
    )
    parser.add_argument(
        '--ema-decay',
        type=float,
        default=0.9999,
        help='EMA 衰减率（推荐 0.9999，范围 0-1）'
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


# =============================================================================
# EMA (Exponential Moving Average) 工具类
# =============================================================================

class ModelEMA:
    """
    模型指数移动平均（EMA）
    
    维护模型权重的指数移动平均，用于验证和推理。
    这是一种 cheap ensemble 方法，通常能提升模型泛化能力。
    
    Reference: https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model_ema.py
    """
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, device: torch.device = None):
        """
        Args:
            model: 要维护 EMA 的模型
            decay: EMA 衰减率（0-1，越大越接近当前权重）
            device: 设备（如果为 None，则使用 model 的设备）
        """
        self.model = model
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        
        # 创建 EMA 模型副本（深拷贝）
        self.ema_model = self._create_ema_model()
        self.ema_model.eval()
    
    def _create_ema_model(self):
        """创建 EMA 模型的副本（通过 state_dict 复制）"""
        import copy
        # 使用深拷贝创建模型副本（这是最安全的方式）
        # 注意：这要求模型是可序列化的，对于标准 PyTorch 模型通常没问题
        ema_model = copy.deepcopy(self.model)
        ema_model.to(self.device)
        
        # 确保 EMA 模型处于 eval 模式
        ema_model.eval()
        
        # 冻结 EMA 模型参数（只用于前向传播）
        for param in ema_model.parameters():
            param.requires_grad = False
        
        return ema_model
    
    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """
        更新 EMA 权重
        
        EMA 权重 = decay * EMA权重 + (1 - decay) * 当前权重
        """
        for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
            if model_param.requires_grad:
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)
    
    def state_dict(self):
        """返回 EMA 模型的状态字典"""
        return self.ema_model.state_dict()
    
    def load_state_dict(self, state_dict):
        """加载 EMA 模型的状态字典"""
        self.ema_model.load_state_dict(state_dict)


@torch.no_grad()
def evaluate(model, data_loader, device, loss_fn, use_amp=False):
    """
    在验证集上计算 loss / accuracy / macro-F1
    
    Args:
        model: 要评估的模型（可以是 EMA 模型）
        data_loader: 验证数据加载器
        device: 设备
        loss_fn: 损失函数
        use_amp: 是否使用混合精度
    """
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    total_steps = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)      # [B, N, K]
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)            # [B]

        # 使用 torch.amp.autocast 进行混合精度推理（如果启用）
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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
    train_labels: List[int],  # 添加 train_labels 用于 WeightedRandomSampler
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
    
    log_print(f"\n{'='*60}")
    log_print(f"Fold {fold_idx + 1}/{args.n_folds} (seed={fold_seed})")
    log_print(f"{'='*60}")
    
    # 创建 fold 的训练集和验证集
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    log_print(f"训练集样本数: {len(train_subset):,}")
    log_print(f"验证集样本数: {len(val_subset):,}")
    
    # 创建 DataLoader（可选使用 WeightedRandomSampler）
    train_sampler = None
    shuffle = True
    
    if args.use_weighted_sampler:
        # 获取训练集的标签
        train_labels_subset = np.array(train_labels)[train_indices]
        # 计算样本权重
        from src.data_preprocess.class_balance import ClassWeightCalculator
        calculator = ClassWeightCalculator()
        sample_weights = calculator.compute_sample_weights(train_labels_subset.tolist())
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(train_subset),
            replacement=True
        )
        shuffle = False  # 使用 sampler 时不能 shuffle
        log_print(f"使用 WeightedRandomSampler 进行类别平衡采样")
    
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
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
    
    log_print(f"Train batches/epoch: {len(train_loader)}")
    log_print(f"Val batches/epoch: {len(val_loader)}")
    
    # 创建模型
    from src.model import create_model, HATConfig
    
    config = HATConfig()
    model = create_model(config)  # HATInterleaved512ForClassification
    model.to(device)
    
    # 加载 MLM 预训练权重（可选）
    if args.mlm_ckpt and Path(args.mlm_ckpt).exists():
        log_print(f"\n加载 MLM 预训练权重: {args.mlm_ckpt}")
        ckpt = torch.load(args.mlm_ckpt, map_location='cpu', weights_only=False)
        mlm_state_dict = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = model.load_state_dict(mlm_state_dict, strict=False)
        log_print(f"  加载完成，missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    else:
        log_print("\n未提供或找不到 MLM 预训练权重，将从随机初始化开始训练。")
    
    # 损失函数（使用工厂函数创建，支持多种损失类型）
    loss_fn = create_loss_fn(
        loss_type=args.loss_type,
        class_weights=class_weights_tensor,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        device=str(device),
    )
    
    loss_info = f"损失函数: {args.loss_type}"
    if args.loss_type in ['smooth', 'focal_smooth']:
        loss_info += f" (label_smoothing={args.label_smoothing})"
    if args.loss_type in ['focal', 'focal_smooth']:
        loss_info += f" (gamma={args.focal_gamma})"
    if args.use_weighted_sampler:
        loss_info += " + WeightedRandomSampler"
    log_print(f"\n{loss_info}")
    
    # AMP (混合精度) 设置
    scaler = None
    if args.use_amp:
        scaler = GradScaler()
        log_print(f"启用 AMP (混合精度训练)")
    
    # EMA (指数移动平均) 设置
    ema = None
    if args.use_ema:
        ema = ModelEMA(model, decay=args.ema_decay, device=device)
        log_print(f"启用 EMA (衰减率={args.ema_decay})")
    
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
    
    log_print(f"\n总训练步数: {total_steps}, warmup 步数: {warmup_steps}")
    log_print(f"学习率: {args.lr}, weight_decay: {args.weight_decay}")
    
    # 训练循环
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_f1 = 0.0
    global_step = 0
    start_time = time.time()
    
    # Early stopping 相关
    patience = args.early_stopping_patience
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(args.num_epochs):
        log_print(f"\n========== Fold {fold_idx + 1} | Epoch {epoch+1}/{args.num_epochs} ==========")
        model.train()
        running_loss = 0.0
        
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            optimizer_was_run = True
            
            # 使用 AMP 进行混合精度训练
            if args.use_amp:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits = outputs["logits"]  # [B, num_labels]
                    loss = loss_fn(logits, labels)
                
                if torch.isnan(loss):
                    log_print(f"警告: step {global_step} 出现 NaN loss，跳过")
                    continue
                
                # 使用 scaler 进行反向传播
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer_was_run = scaler.step(optimizer)
                scaler.update()
            else:
                # 标准精度训练
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs["logits"]  # [B, num_labels]
                loss = loss_fn(logits, labels)
                
                if torch.isnan(loss):
                    log_print(f"警告: step {global_step} 出现 NaN loss，跳过")
                    continue
                
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            
            # 仅在优化器实际运行时再推进 scheduler，避免跳过首个 lr
            if optimizer_was_run or optimizer_was_run is None:  # 兼容旧版返回 None
                scheduler.step()
            
            # 更新 EMA 权重（如果启用）
            if ema is not None:
                ema.update(model)
            
            global_step += 1
            running_loss += loss.item()
            
            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                samples_per_sec = global_step * args.batch_size / elapsed
                
                log_print(
                    f"[Fold {fold_idx + 1} | Epoch {epoch+1}] "
                    f"Step {global_step}/{total_steps} | "
                    f"Train Loss: {avg_loss:.4f} | "
                    f"LR: {lr_now:.2e} | "
                    f"Speed: {samples_per_sec:.1f} samples/s"
                )
                sys.stdout.flush()
                running_loss = 0.0
        
        # 每个 epoch 结束做一次验证
        # 如果使用 EMA，验证时使用 EMA 模型
        eval_model = ema.ema_model if ema is not None else model
        val_loss, val_acc, val_f1 = evaluate(
            eval_model, val_loader, device, loss_fn=loss_fn, use_amp=args.use_amp
        )
        eval_model_name = "EMA" if ema is not None else "Standard"
        log_print(
            f"\n[Eval] Fold {fold_idx + 1} | Epoch {epoch+1} | Model: {eval_model_name} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )
        sys.stdout.flush()
        
        # 保存最好模型 & Early Stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0  # 重置 patience 计数器
            best_path = output_dir / f"hat_cls_fold{fold_idx}_best.pt"
            
            # 保存模型状态字典（如果使用 EMA，保存 EMA 权重）
            model_state_dict = ema.state_dict() if ema is not None else model.state_dict()
            
            save_dict = {
                'fold': fold_idx,
                'epoch': epoch,
                'step': global_step,
                'model_state_dict': model_state_dict,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_macro_f1': val_f1,
            }
            
            # 如果使用 EMA，也保存原始模型权重（用于继续训练）
            if ema is not None:
                save_dict['model_state_dict_original'] = model.state_dict()
                save_dict['ema_decay'] = args.ema_decay
            
            torch.save(save_dict, best_path)
            log_print(f"  >> 新最佳模型 ({eval_model_name})，已保存到: {best_path}")
        else:
            # F1 没有提升
            if patience is not None:
                patience_counter += 1
                if patience_counter >= patience:
                    log_print(
                        f"\nEarly Stopping: 连续 {patience} 个 epoch val F1 无提升 "
                        f"(最佳: {best_val_f1:.4f} @ epoch {best_epoch + 1})"
                    )
                    break
    
    elapsed = time.time() - start_time
    actual_epochs = epoch + 1
    log_print(f"\nFold {fold_idx + 1} 训练完成！")
    log_print(f"  实际训练 epoch: {actual_epochs}/{args.num_epochs}")
    log_print(f"  总步数: {global_step}")
    log_print(f"  最佳 Val Macro-F1: {best_val_f1:.4f} (epoch {best_epoch + 1})")
    log_print(f"  耗时: {elapsed / 60:.1f} 分钟")
    
    # 确保模型已保存
    best_model_path = output_dir / f"hat_cls_fold{fold_idx}_best.pt"
    if not best_model_path.exists():
        # 如果模型未保存，保存当前模型
        log_print(f"警告: 模型文件不存在，保存当前模型...")
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
    if ema is not None:
        del ema
    if scaler is not None:
        del scaler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        log_print(f"已清理 GPU 显存")
    
    return best_val_f1, best_model_path


def main():
    args = parse_args()
    
    # 设置全局随机种子（用于训练过程）
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    log_print(f"使用设备: {device}")
    if device.type == 'cuda':
        log_print(f"GPU: {torch.cuda.get_device_name(0)}")
        log_print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # ========== 1. 读数据 ==========
    train_df = pd.read_csv(args.train_path, sep='\t')
    log_print(f"训练样本数: {len(train_df):,}")
    
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].astype(int).tolist()
    
    # ========== 2. K-Fold 划分 ==========
    log_print(f"\n使用 Stratified K-Fold (K={args.n_folds}, seed={args.fold_seed}) 划分数据...")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.fold_seed)
    
    # 获取所有 fold 的划分
    fold_splits = list(skf.split(train_texts, train_labels))
    
    log_print(f"\n各 Fold 样本分布:")
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        train_labels_fold = np.array(train_labels)[train_idx]
        val_labels_fold = np.array(train_labels)[val_idx]
        train_label_dist = np.bincount(train_labels_fold)
        val_label_dist = np.bincount(val_labels_fold)
        log_print(f"  Fold {fold_idx + 1}: 训练集 {len(train_idx):,} 样本, 验证集 {len(val_idx):,} 样本")
        log_print(f"    训练集标签分布: {train_label_dist}")
        log_print(f"    验证集标签分布: {val_label_dist}")
    
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
    log_print(f"\n类别权重示例: {class_weights[:5]} ...")
    
    # ========== 5. 训练每个 Fold ==========
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 打印训练配置摘要
    log_print(f"\n{'='*60}")
    log_print(f"训练配置摘要")
    log_print(f"{'='*60}")
    log_print(f"损失函数: {args.loss_type}")
    if args.loss_type in ['smooth', 'focal_smooth']:
        log_print(f"  标签平滑: {args.label_smoothing}")
    if args.loss_type in ['focal', 'focal_smooth']:
        log_print(f"  Focal gamma: {args.focal_gamma}")
    log_print(f"学习率: {args.lr}")
    log_print(f"使用 WeightedRandomSampler: {args.use_weighted_sampler}")
    log_print(f"使用 AMP (混合精度): {args.use_amp}")
    log_print(f"使用 EMA (指数移动平均): {args.use_ema}")
    if args.use_ema:
        log_print(f"  EMA 衰减率: {args.ema_decay}")
    if args.early_stopping_patience:
        log_print(f"Early Stopping patience: {args.early_stopping_patience}")
    log_print(f"{'='*60}\n")
    
    # 在开始训练前清理显存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        log_print(f"已清理初始显存，准备开始 K-Fold 训练...")
    
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
            train_labels=train_labels,
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
            log_print(f"\n已清理显存，准备训练下一个 fold...")
    
    # ========== 6. 总结 ==========
    total_elapsed = time.time() - total_start_time
    log_print(f"\n{'='*60}")
    log_print(f"K-Fold 训练完成！")
    log_print(f"{'='*60}")
    log_print(f"总耗时: {total_elapsed / 60:.1f} 分钟")
    log_print(f"\n各 Fold 最佳验证集 Macro-F1:")
    
    all_f1s = []
    for result in fold_results:
        fold_idx = result['fold']
        best_f1 = result['best_val_f1']
        model_path = result['model_path']
        all_f1s.append(best_f1)
        log_print(f"  Fold {fold_idx + 1}: {best_f1:.4f} -> {model_path}")
    
    mean_f1 = np.mean(all_f1s)
    std_f1 = np.std(all_f1s)
    log_print(f"\n平均 Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")
    log_print(f"\n所有模型已保存到: {output_dir}")
    log_print(f"\n推理时使用以下命令进行 Ensemble:")
    log_print(f"  python scripts/infer.py \\")
    log_print(f"    --test-path data/processed/test.csv \\")
    log_print(f"    --model-paths {output_dir}/hat_cls_fold0_best.pt,{output_dir}/hat_cls_fold1_best.pt,... \\")
    log_print(f"    --output-path outputs/submission/submission_kfold.csv")


if __name__ == "__main__":
    main()

