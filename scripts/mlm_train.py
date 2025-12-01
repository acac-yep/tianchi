#!/usr/bin/env python3
"""
HAT 模型 MLM 预训练脚本

使用 Masked Language Modeling (MLM) 任务对 HAT 模型进行预训练。

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
    python scripts/mlm_train.py --batch-size 8 --max-steps 5000
"""

import os
import sys
import argparse
import random
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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
        default=1,
        help='训练 epoch 数'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        default=1000,
        help='最大训练步数（到达后提前停止）'
    )
    
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=100,
        help='warmup 步数'
    )
    
    parser.add_argument(
        '--log-every',
        type=int,
        default=50,
        help='日志打印间隔（步数）'
    )
    
    parser.add_argument(
        '--save-every',
        type=int,
        default=500,
        help='checkpoint 保存间隔（步数）'
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


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # ========== 1. 加载数据 ==========
    print("\n" + "=" * 60)
    print("加载训练数据")
    print("=" * 60)
    
    train_path = Path(args.train_path)
    if not train_path.exists():
        print(f"错误: 训练数据文件不存在: {train_path}")
        print("请先运行 scripts/run_preprocessing.py 进行数据预处理")
        sys.exit(1)
    
    df = pd.read_csv(train_path, sep='\t')
    texts = df['text'].tolist()
    print(f"训练样本数: {len(texts):,}")
    
    # ========== 2. 创建 Dataset 和 DataLoader ==========
    print("\n" + "=" * 60)
    print("创建 Dataset 和 DataLoader")
    print("=" * 60)
    
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
    
    # 创建 Dataset
    # 注意：mode='pretrain' 时，HATDataset 内部会使用 'train' 模式进行分段
    # 输出形状: input_ids [N, K], attention_mask [N, K]，不含 CLS
    train_dataset = HATDataset(
        texts=texts,
        labels=None,  # MLM 不需要 label
        tokenizer=tokenizer,
        segmenter=segmenter,
        mode='pretrain',
        cache_segments=False,
    )
    
    print(f"Dataset 大小: {len(train_dataset)}")
    
    # 创建 MLM Collator
    # 输出形状: input_ids [B, N, K], attention_mask [B, N, K], mlm_labels [B, N, K]
    # 注意: 不含 CLS，CLS 由模型内部添加
    collator = MLMDataCollator(
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
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )
    
    print(f"Batch size: {args.batch_size}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # 验证一个 batch 的形状
    sample_batch = next(iter(train_loader))
    print(f"\nBatch 形状验证（外部，不含 CLS）:")
    print(f"  input_ids: {sample_batch['input_ids'].shape}")
    print(f"  attention_mask: {sample_batch['attention_mask'].shape}")
    print(f"  mlm_labels: {sample_batch['mlm_labels'].shape}")
    print(f"  segment_mask: {sample_batch['segment_mask'].shape}")
    
    # ========== 3. 创建模型 ==========
    print("\n" + "=" * 60)
    print("创建 MLM 模型")
    print("=" * 60)
    
    from src.model import create_mlm_model, HATConfig
    
    config = HATConfig()
    model = create_mlm_model(config)
    model.to(device)
    
    print(f"模型参数量: {model.get_num_parameters():,}")
    print(f"模型配置:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hat_layers: {config.num_hat_layers}")
    print(f"  segment_length: {config.segment_length} (不含 CLS)")
    print(f"  max_segments: {config.max_segments}")
    
    # ========== 4. 创建优化器和调度器 ==========
    print("\n" + "=" * 60)
    print("创建优化器和调度器")
    print("=" * 60)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # 计算总训练步数
    num_training_steps = min(args.max_steps, len(train_loader) * args.num_epochs)
    
    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    print(f"学习率: {args.lr}")
    print(f"权重衰减: {args.weight_decay}")
    print(f"Warmup 步数: {args.warmup_steps}")
    print(f"总训练步数: {num_training_steps}")
    
    # ========== 5. 训练循环 ==========
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.train()
    global_step = 0
    total_loss = 0.0
    log_loss = 0.0
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.num_epochs} ---")
        
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
                print(f"警告: step {global_step} 出现 NaN loss，跳过")
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # 优化器更新
            optimizer.step()
            scheduler.step()
            
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
                
                print(f"[Epoch {epoch+1}] "
                      f"Step {global_step}/{num_training_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Speed: {samples_per_sec:.1f} samples/s")
                
                log_loss = 0.0
            
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
                print(f"  Checkpoint 已保存: {ckpt_path}")
            
            # 达到最大步数
            if global_step >= args.max_steps:
                break
        
        if global_step >= args.max_steps:
            break
    
    # ========== 6. 保存最终模型 ==========
    print("\n" + "=" * 60)
    print("保存最终模型")
    print("=" * 60)
    
    final_path = output_dir / 'hat_mlm_final.pt'
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'loss': total_loss / global_step if global_step > 0 else 0,
        'config': {
            'vocab_size': config.vocab_size,
            'hidden_size': config.hidden_size,
            'num_hat_layers': config.num_hat_layers,
            'segment_length': config.segment_length,
            'max_segments': config.max_segments,
        },
    }, final_path)
    print(f"最终模型已保存: {final_path}")
    
    # 训练统计
    elapsed = time.time() - start_time
    print(f"\n训练完成!")
    print(f"  总步数: {global_step}")
    print(f"  平均 Loss: {total_loss / global_step:.4f}")
    print(f"  总耗时: {elapsed / 60:.1f} 分钟")
    print(f"  平均速度: {global_step * args.batch_size / elapsed:.1f} samples/s")


if __name__ == '__main__':
    main()

