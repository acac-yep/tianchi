#!/usr/bin/env python3
"""
HAT 模型 MLM 预训练脚本（改进版）

使用 Masked Language Modeling (MLM) 任务对 HAT 模型进行预训练。

主要改进:
    1. 添加验证集和验证循环，支持 val_loss 和 perplexity 监控
    2. 实现 EMA (Exponential Moving Average) 提升泛化能力
    3. 保存 best checkpoint（基于 val_loss）
    4. 支持 warmup_ratio 参数，更灵活的 warmup 设置
    5. 优化默认超参数（更长训练、更小 LR、更合理的 schedule）

数据流:
    1. 从 train.csv 读取文本（已经完成 token ID 重映射）
    2. HATDataset 进行分段处理，输出 [N, K] 的 input_ids（不含 CLS）
    3. MLMDataCollator 进行 mask 处理，生成 [B, N, K] 的 batch
    4. HATInterleaved512ForMLM 模型内部添加 CLS，处理为 [B, N, K+1]
    5. 模型内部将 mlm_labels pad 为 [B, N, K+1] 与 prediction_scores 对齐

形状约定（非常重要）:
    - 外部 input_ids: [B, N, K]  # K=512，不含 CLS
    - 外部 mlm_labels: [B, N, K]  # K=512，不含 CLS
    - 模型内部 hidden_states: [B, N, K+1, H]  # K+1=513，含 CLS
    - 模型内部 mlm_labels: [B, N, K+1]  # 模型内部 pad
    - prediction_scores: [B, N, K+1, vocab_size]

使用方法:
    python scripts/mlm_train.py
    python scripts/mlm_train.py --batch-size 8 --max-steps 10000 --warmup-ratio 0.06
"""

import os
import sys
import argparse
import random
import time
import copy
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# 日志工具
# =============================================================================

def log_print(*args, **kwargs):
    """带时间戳的 print 函数"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}]", *args, **kwargs)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='HAT 模型 MLM 预训练脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据相关
    parser.add_argument(
        '--train-path',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'processed' / 'train.csv'),
        help='训练数据路径（预处理后的 train.csv）'
    )
    
    # 输出相关
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(PROJECT_ROOT / 'checkpoints' / 'mlm_hat512'),
        help='checkpoint 输出目录'
    )
    
    # 训练超参数
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='batch size（根据 GPU 显存调整）'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-5,
        help='学习率（推荐 3e-5 到 5e-5，比原来的 1e-4 更温和）'
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
        default=1,
        help='训练 epoch 数'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        default=10000,
        help='最大训练步数（到达后提前停止，推荐 1e4 或更大）'
    )
    
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=None,
        help='warmup 步数（如果设置了 --warmup-ratio 则忽略）'
    )
    
    parser.add_argument(
        '--warmup-ratio',
        type=float,
        default=0.06,
        help='warmup 步数占总训练步数的比例（推荐 0.06，即 6%%）'
    )
    
    parser.add_argument(
        '--log-every',
        type=int,
        default=50,
        help='日志打印间隔（步数）'
    )
    
    parser.add_argument(
        '--eval-every',
        type=int,
        default=200,
        help='验证间隔（步数）'
    )
    
    parser.add_argument(
        '--save-every',
        type=int,
        default=500,
        help='checkpoint 保存间隔（步数）'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='验证集比例（从训练集中划分）'
    )
    
    # MLM 相关
    parser.add_argument(
        '--mlm-probability',
        type=float,
        default=0.15,
        help='MLM mask 概率'
    )
    
    # 设备相关
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备（cuda / cpu）'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=2,
        help='DataLoader 工作进程数'
    )
    
    # 其他
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
    
    # EMA 相关
    parser.add_argument(
        '--use-ema',
        action='store_true',
        default=True,
        help='使用 EMA (Exponential Moving Average)'
    )
    
    parser.add_argument(
        '--ema-decay',
        type=float,
        default=0.999,
        help='EMA 衰减率（推荐 0.999）'
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    创建带 warmup 的线性学习率调度器
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_ema_model(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    创建 EMA 模型副本
    
    Args:
        model: 原始模型
        device: 设备
        
    Returns:
        ema_model: EMA 模型（所有参数 requires_grad=False）
    """
    ema_model = copy.deepcopy(model)
    ema_model.to(device)
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


def update_ema(model: torch.nn.Module, ema_model: torch.nn.Module, decay: float = 0.999):
    """
    更新 EMA 模型权重
    
    Args:
        model: 原始模型
        ema_model: EMA 模型
        decay: EMA 衰减率
    """
    with torch.no_grad():
        model_state_dict = model.state_dict()
        ema_state_dict = ema_model.state_dict()
        
        for key in ema_state_dict:
            if key in model_state_dict:
                ema_state_dict[key].copy_(
                    decay * ema_state_dict[key] + (1.0 - decay) * model_state_dict[key]
                )


def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    在验证集上评估模型
    
    Args:
        model: 模型（可以是 EMA 模型）
        val_loader: 验证数据加载器
        device: 设备
        
    Returns:
        avg_loss: 平均 loss
        perplexity: 困惑度 (exp(avg_loss))
        accuracy: token-level 准确率（可选）
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mlm_labels = batch['mlm_labels'].to(device)
            
            # 前向传播
            loss, prediction_scores = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mlm_labels=mlm_labels,
            )
            
            # 统计被 mask 的 token 数量
            # mlm_labels 中非 -100 的位置就是被 mask 的位置
            masked_positions = (mlm_labels != -100)
            batch_masked_tokens = masked_positions.sum().item()
            
            if batch_masked_tokens > 0:
                # loss 是平均到所有被 mask 的 token 上的
                total_loss += loss.item() * batch_masked_tokens
                total_tokens += batch_masked_tokens
                
                # 计算准确率（可选）
                # 需要将 prediction_scores 与 mlm_labels 对齐
                # 注意：模型内部会将 mlm_labels pad 为 [B, N, K+1]
                # prediction_scores 是 [B, N, K+1, vocab_size]
                # 我们需要取 [B, N, 1:] 部分（跳过 CLS）与 mlm_labels 对齐
                # 但 mlm_labels 在模型内部已经被 pad，所以我们需要重新处理
                # 为了简化，这里先不计算准确率，只计算 loss 和 perplexity
    
    avg_loss = total_loss / max(1, total_tokens)
    perplexity = np.exp(avg_loss)
    
    # 准确率计算（简化版，暂时返回 0）
    accuracy = 0.0  # TODO: 实现完整的准确率计算
    
    model.train()
    return avg_loss, perplexity, accuracy


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    log_print(f"使用设备: {device}")
    
    if device.type == 'cuda':
        log_print(f"GPU: {torch.cuda.get_device_name(0)}")
        log_print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # ========== 1. 加载数据 ==========
    log_print("\n" + "=" * 60)
    log_print("加载训练数据")
    log_print("=" * 60)
    
    train_path = Path(args.train_path)
    if not train_path.exists():
        log_print(f"错误: 训练数据文件不存在: {train_path}")
        log_print("请先运行 scripts/run_preprocessing.py 进行数据预处理")
        sys.exit(1)
    
    df = pd.read_csv(train_path, sep='\t')
    texts = df['text'].tolist()
    log_print(f"总样本数: {len(texts):,}")
    
    # 划分训练集和验证集
    val_size = int(len(texts) * args.val_ratio)
    train_size = len(texts) - val_size
    
    # 随机打乱
    indices = list(range(len(texts)))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_texts = [texts[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    
    log_print(f"训练集: {len(train_texts):,} 样本")
    log_print(f"验证集: {len(val_texts):,} 样本 ({args.val_ratio*100:.1f}%)")
    
    # ========== 2. 创建 Dataset 和 DataLoader ==========
    log_print("\n" + "=" * 60)
    log_print("创建 Dataset 和 DataLoader")
    log_print("=" * 60)
    
    from src.data_preprocess import (
        HATDataset,
        MLMDataCollator,
        create_tokenizer,
        create_segmenter,
    )
    from src.common_config import COMMON_CONFIG
    
    # 创建 tokenizer 和 segmenter
    tokenizer = create_tokenizer()
    segmenter = create_segmenter()
    
    # 创建训练 Dataset
    # 注意：mode='pretrain' 时，HATDataset 内部会使用 'train' 模式进行分段
    # 输出形状: input_ids [N, K], attention_mask [N, K]，不含 CLS
    train_dataset = HATDataset(
        texts=train_texts,
        labels=None,  # MLM 不需要 label
        tokenizer=tokenizer,
        segmenter=segmenter,
        mode='pretrain',
        cache_segments=False,
    )
    
    # 创建验证 Dataset
    val_dataset = HATDataset(
        texts=val_texts,
        labels=None,
        tokenizer=tokenizer,
        segmenter=segmenter,
        mode='pretrain',
        cache_segments=False,
    )
    
    log_print(f"训练 Dataset 大小: {len(train_dataset)}")
    log_print(f"验证 Dataset 大小: {len(val_dataset)}")
    
    # 创建 MLM Collator
    # 输出形状: input_ids [B, N, K], attention_mask [B, N, K], mlm_labels [B, N, K]
    # 注意: 不含 CLS，CLS 由模型内部添加
    train_collator = MLMDataCollator(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        max_segments=COMMON_CONFIG.max_segments,
        segment_length=COMMON_CONFIG.segment_length,
        pad_token_id=COMMON_CONFIG.pad_token_id,
        pad_to_max=True,
    )
    
    val_collator = MLMDataCollator(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        max_segments=COMMON_CONFIG.max_segments,
        segment_length=COMMON_CONFIG.segment_length,
        pad_token_id=COMMON_CONFIG.pad_token_id,
        pad_to_max=True,
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_collator,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_collator,
        pin_memory=True,
        drop_last=False,
    )
    
    log_print(f"Batch size: {args.batch_size}")
    log_print(f"训练 Batches per epoch: {len(train_loader)}")
    log_print(f"验证 Batches: {len(val_loader)}")
    
    # 验证一个 batch 的形状
    sample_batch = next(iter(train_loader))
    log_print(f"\nBatch 形状验证（外部，不含 CLS）:")
    log_print(f"  input_ids: {sample_batch['input_ids'].shape}")
    log_print(f"  attention_mask: {sample_batch['attention_mask'].shape}")
    log_print(f"  mlm_labels: {sample_batch['mlm_labels'].shape}")
    log_print(f"  segment_mask: {sample_batch['segment_mask'].shape}")
    
    # ========== 3. 创建模型 ==========
    log_print("\n" + "=" * 60)
    log_print("创建 MLM 模型")
    log_print("=" * 60)
    
    from src.model import create_mlm_model, HATConfig
    
    config = HATConfig()
    model = create_mlm_model(config)
    model.to(device)
    
    log_print(f"模型参数量: {model.get_num_parameters():,}")
    log_print(f"模型配置:")
    log_print(f"  vocab_size: {config.vocab_size}")
    log_print(f"  hidden_size: {config.hidden_size}")
    log_print(f"  num_hat_layers: {config.num_hat_layers}")
    log_print(f"  segment_length: {config.segment_length} (不含 CLS)")
    log_print(f"  max_segments: {config.max_segments}")
    
    # ========== 4. 创建优化器和调度器 ==========
    log_print("\n" + "=" * 60)
    log_print("创建优化器和调度器")
    log_print("=" * 60)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # 计算总训练步数
    num_training_steps = min(args.max_steps, len(train_loader) * args.num_epochs)
    
    # 计算 warmup 步数
    if args.warmup_steps is not None:
        num_warmup_steps = args.warmup_steps
    else:
        num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    
    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    log_print(f"学习率: {args.lr}")
    log_print(f"权重衰减: {args.weight_decay}")
    log_print(f"Warmup 步数: {num_warmup_steps} ({num_warmup_steps/num_training_steps*100:.1f}%)")
    log_print(f"总训练步数: {num_training_steps}")
    
    # ========== 4.5 创建 EMA 模型 ==========
    ema_model = None
    if args.use_ema:
        log_print("\n" + "=" * 60)
        log_print("创建 EMA 模型")
        log_print("=" * 60)
        ema_model = create_ema_model(model, device)
        log_print(f"EMA 衰减率: {args.ema_decay}")
    
    # ========== 5. 训练循环 ==========
    log_print("\n" + "=" * 60)
    log_print("开始训练")
    log_print("=" * 60)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.train()
    global_step = 0
    total_loss = 0.0
    log_loss = 0.0
    start_time = time.time()
    
    # Best validation loss tracking
    best_val_loss = float('inf')
    best_val_ppl = float('inf')
    best_step = 0
    
    for epoch in range(args.num_epochs):
        log_print(f"\n--- Epoch {epoch + 1}/{args.num_epochs} ---")
        
        for batch_idx, batch in enumerate(train_loader):
            # 移动数据到设备
            # 形状: [B, N, K]，K=512，不含 CLS
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mlm_labels = batch['mlm_labels'].to(device)
            
            # 前向传播
            # 模型内部会:
            # 1. 添加 CLS，将 input_ids 从 [B, N, K] 变为 [B, N, K+1, H]
            # 2. 将 mlm_labels 从 [B, N, K] pad 为 [B, N, K+1]
            # 3. 计算 loss
            optimizer.zero_grad()
            loss, prediction_scores = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mlm_labels=mlm_labels,
            )
            
            # 检查 loss 是否为 NaN
            if torch.isnan(loss):
                log_print(f"警告: step {global_step} 出现 NaN loss，跳过")
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # 优化器更新
            optimizer.step()
            scheduler.step()
            
            # 更新 EMA
            if args.use_ema and ema_model is not None:
                update_ema(model, ema_model, decay=args.ema_decay)
            
            # 统计
            global_step += 1
            total_loss += loss.item()
            log_loss += loss.item()
            
            # 日志
            if global_step % args.log_every == 0:
                avg_loss = log_loss / args.log_every
                elapsed = time.time() - start_time
                samples_per_sec = global_step * args.batch_size / elapsed
                current_lr = scheduler.get_last_lr()[0]
                
                log_print(f"[Epoch {epoch+1}] "
                      f"Step {global_step}/{num_training_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Speed: {samples_per_sec:.1f} samples/s")
                
                log_loss = 0.0
            
            # 验证
            if global_step % args.eval_every == 0:
                log_print("\n" + "-" * 60)
                log_print(f"验证 (Step {global_step})")
                log_print("-" * 60)
                
                # 使用 EMA 模型进行验证（如果启用）
                eval_model = ema_model if (args.use_ema and ema_model is not None) else model
                
                val_loss, val_ppl, val_acc = evaluate(eval_model, val_loader, device)
                
                log_print(f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
                
                # 保存 best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_ppl = val_ppl
                    best_step = global_step
                    
                    best_path = output_dir / 'best_model.pt'
                    checkpoint_data = {
                        'step': global_step,
                        'epoch': epoch,
                        'model_state_dict': eval_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss,
                        'val_ppl': val_ppl,
                        'train_loss': total_loss / global_step,
                        'config': {
                            'vocab_size': config.vocab_size,
                            'hidden_size': config.hidden_size,
                            'num_hat_layers': config.num_hat_layers,
                            'segment_length': config.segment_length,
                            'max_segments': config.max_segments,
                        },
                    }
                    if args.use_ema:
                        checkpoint_data['is_ema'] = True
                        checkpoint_data['ema_decay'] = args.ema_decay
                    
                    torch.save(checkpoint_data, best_path)
                    log_print(f"  ✓ Best model 已保存 (Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f})")
                else:
                    log_print(f"  (Best: Step {best_step}, Val Loss: {best_val_loss:.4f}, Val PPL: {best_val_ppl:.2f})")
                
                log_print("-" * 60 + "\n")
            
            # 保存 checkpoint
            if global_step % args.save_every == 0:
                ckpt_path = output_dir / f'checkpoint_step{global_step}.pt'
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': total_loss / global_step,
                    'config': {
                        'vocab_size': config.vocab_size,
                        'hidden_size': config.hidden_size,
                        'num_hat_layers': config.num_hat_layers,
                    },
                }, ckpt_path)
                log_print(f"  Checkpoint 已保存: {ckpt_path}")
            
            # 达到最大步数
            if global_step >= args.max_steps:
                break
        
        if global_step >= args.max_steps:
            break
    
    # ========== 6. 最终验证和保存 ==========
    log_print("\n" + "=" * 60)
    log_print("最终验证")
    log_print("=" * 60)
    
    # 使用 EMA 模型进行最终验证（如果启用）
    final_eval_model = ema_model if (args.use_ema and ema_model is not None) else model
    final_val_loss, final_val_ppl, final_val_acc = evaluate(final_eval_model, val_loader, device)
    
    log_print(f"最终 Val Loss: {final_val_loss:.4f} | Val PPL: {final_val_ppl:.2f}")
    
    # 保存最终模型（使用 EMA 模型，如果启用）
    log_print("\n" + "=" * 60)
    log_print("保存最终模型")
    log_print("=" * 60)
    
    final_path = output_dir / 'hat_mlm_final.pt'
    final_checkpoint = {
        'step': global_step,
        'model_state_dict': final_eval_model.state_dict(),
        'train_loss': total_loss / global_step if global_step > 0 else 0,
        'val_loss': final_val_loss,
        'val_ppl': final_val_ppl,
        'best_val_loss': best_val_loss,
        'best_val_ppl': best_val_ppl,
        'best_step': best_step,
        'config': {
            'vocab_size': config.vocab_size,
            'hidden_size': config.hidden_size,
            'num_hat_layers': config.num_hat_layers,
            'segment_length': config.segment_length,
            'max_segments': config.max_segments,
        },
    }
    if args.use_ema:
        final_checkpoint['is_ema'] = True
        final_checkpoint['ema_decay'] = args.ema_decay
    
    torch.save(final_checkpoint, final_path)
    log_print(f"最终模型已保存: {final_path}")
    
    # 训练统计
    elapsed = time.time() - start_time
    log_print(f"\n" + "=" * 60)
    log_print("训练完成!")
    log_print("=" * 60)
    log_print(f"  总步数: {global_step}")
    log_print(f"  平均 Train Loss: {total_loss / global_step:.4f}")
    log_print(f"  最终 Val Loss: {final_val_loss:.4f}")
    log_print(f"  最终 Val PPL: {final_val_ppl:.2f}")
    log_print(f"  Best Val Loss: {best_val_loss:.4f} (Step {best_step})")
    log_print(f"  Best Val PPL: {best_val_ppl:.2f}")
    log_print(f"  总耗时: {elapsed / 60:.1f} 分钟")
    log_print(f"  平均速度: {global_step * args.batch_size / elapsed:.1f} samples/s")
    log_print(f"\n推荐使用 best_model.pt 进行下游任务微调")
    log_print("=" * 60)


if __name__ == '__main__':
    main()

