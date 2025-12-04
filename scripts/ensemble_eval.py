#!/usr/bin/env python3
"""
Ensemble 模型评估脚本

功能:
1. 加载多个不同 seed 训练的模型 checkpoint
2. 对验证集做 logits 平均 (ensemble)
3. 计算 Macro-F1, Accuracy 等指标
4. 支持输出预测结果到 CSV

使用方法:
    # 自动扫描目录下的所有 checkpoint
    python scripts/ensemble_eval.py \
        --checkpoint-dir checkpoints/ensemble \
        --val-path data/processed/val.csv
    
    # 指定多个 checkpoint 路径
    python scripts/ensemble_eval.py \
        --checkpoint-paths \
            checkpoints/ensemble/cls_hat512_stage2_seed42/hat_cls_best.pt \
            checkpoints/ensemble/cls_hat512_stage2_seed13/hat_cls_best.pt \
            checkpoints/ensemble/cls_hat512_stage2_seed87/hat_cls_best.pt \
        --val-path data/processed/val.csv
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Ensemble 模型评估脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Checkpoint 来源 (二选一)
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='包含多个 seed checkpoint 的目录，自动扫描 **/hat_cls_best.pt'
    )
    parser.add_argument(
        '--checkpoint-paths',
        type=str,
        nargs='+',
        default=None,
        help='指定多个 checkpoint 文件路径'
    )
    
    # 数据路径
    parser.add_argument(
        '--val-path',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'processed' / 'val.csv'),
        help='验证数据路径'
    )
    parser.add_argument(
        '--test-path',
        type=str,
        default=None,
        help='测试数据路径 (可选，用于生成提交文件)'
    )
    
    # 推理参数
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='推理 batch size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备 (cuda / cpu)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='DataLoader worker 数'
    )
    
    # 输出
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(PROJECT_ROOT / 'outputs' / 'ensemble'),
        help='输出目录'
    )
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='是否保存预测结果到 CSV'
    )
    
    # Ensemble 策略
    parser.add_argument(
        '--ensemble-method',
        type=str,
        choices=['logits_avg', 'prob_avg', 'voting'],
        default='logits_avg',
        help='Ensemble 方法: logits_avg (logits平均), prob_avg (概率平均), voting (投票)'
    )
    
    return parser.parse_args()


def find_checkpoints(checkpoint_dir: str) -> List[str]:
    """
    在指定目录下扫描所有 hat_cls_best.pt 文件
    """
    pattern = os.path.join(checkpoint_dir, '**/hat_cls_best.pt')
    ckpts = glob.glob(pattern, recursive=True)
    ckpts = sorted(ckpts)  # 按路径排序
    return ckpts


def load_models(
    checkpoint_paths: List[str],
    device: torch.device,
) -> List[torch.nn.Module]:
    """
    加载多个模型
    """
    from src.model import create_model, HATConfig
    
    models = []
    config = HATConfig()
    
    for i, ckpt_path in enumerate(checkpoint_paths):
        print(f"[{i+1}/{len(checkpoint_paths)}] 加载模型: {ckpt_path}")
        
        # 创建新模型
        model = create_model(config)
        model.to(device)
        
        # 加载权重
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict, strict=True)
        
        # 设置为评估模式
        model.eval()
        
        # 打印 checkpoint 信息
        if 'val_macro_f1' in ckpt:
            print(f"    Val Macro-F1: {ckpt['val_macro_f1']:.4f}")
        
        models.append(model)
    
    print(f"\n共加载 {len(models)} 个模型")
    return models


def create_dataloader(
    data_path: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """
    创建数据加载器
    """
    from src.data_preprocess import (
        HATDataset,
        HATDataCollator,
        create_tokenizer,
        create_segmenter,
    )
    
    # 读取数据
    df = pd.read_csv(data_path, sep='\t')
    texts = df['text'].tolist()
    
    # 如果有 label 列则使用，否则用 -1 占位
    if 'label' in df.columns:
        labels = df['label'].astype(int).tolist()
    else:
        labels = [-1] * len(texts)
    
    tokenizer = create_tokenizer()
    segmenter = create_segmenter()
    
    dataset = HATDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        segmenter=segmenter,
        mode='eval',
        cache_segments=False,
    )
    
    collator = HATDataCollator()
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False,
    )
    
    return loader


@torch.no_grad()
def ensemble_predict(
    models: List[torch.nn.Module],
    data_loader: DataLoader,
    device: torch.device,
    method: str = 'logits_avg',
) -> Dict[str, np.ndarray]:
    """
    使用 ensemble 进行预测
    
    Args:
        models: 模型列表
        data_loader: 数据加载器
        device: 设备
        method: ensemble 方法
            - 'logits_avg': logits 平均
            - 'prob_avg': 概率平均 (softmax 后平均)
            - 'voting': 投票
    
    Returns:
        dict: {
            'predictions': 最终预测 [N],
            'logits_ensemble': ensemble 后的 logits [N, C],
            'labels': 真实标签 [N] (如果有),
        }
    """
    all_logits_list = [[] for _ in models]  # 每个模型的 logits
    all_labels = []
    
    print(f"\n开始 Ensemble 推理 (method={method}, {len(models)} models)...")
    
    for batch in tqdm(data_loader, desc="Inference"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']  # 可能是 -1
        
        # 每个模型分别推理
        batch_logits = []
        for i, model in enumerate(models):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs["logits"]  # [B, C]
            batch_logits.append(logits)
            all_logits_list[i].append(logits.cpu())
        
        all_labels.append(labels)
    
    # 拼接所有 batch
    all_logits = [torch.cat(logits_list, dim=0) for logits_list in all_logits_list]
    # all_logits: List of [N, C], length = num_models
    
    all_labels = torch.cat(all_labels, dim=0).numpy()  # [N]
    
    # 堆叠: [num_models, N, C]
    stacked_logits = torch.stack(all_logits, dim=0)
    
    if method == 'logits_avg':
        # Logits 平均
        ensemble_logits = stacked_logits.mean(dim=0)  # [N, C]
        predictions = ensemble_logits.argmax(dim=-1).numpy()
        
    elif method == 'prob_avg':
        # 概率平均 (softmax 后平均)
        probs = torch.softmax(stacked_logits, dim=-1)  # [num_models, N, C]
        ensemble_probs = probs.mean(dim=0)  # [N, C]
        ensemble_logits = torch.log(ensemble_probs + 1e-10)  # 转回 log 空间
        predictions = ensemble_probs.argmax(dim=-1).numpy()
        
    elif method == 'voting':
        # 投票
        votes = stacked_logits.argmax(dim=-1)  # [num_models, N]
        # 对每个样本统计投票
        predictions = []
        for i in range(votes.shape[1]):
            sample_votes = votes[:, i].numpy()
            # 使用 numpy 的 bincount 找最多票的类别
            vote_counts = np.bincount(sample_votes, minlength=stacked_logits.shape[-1])
            predictions.append(vote_counts.argmax())
        predictions = np.array(predictions)
        ensemble_logits = stacked_logits.mean(dim=0)  # 仍然保存平均 logits 用于分析
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return {
        'predictions': predictions,
        'logits_ensemble': ensemble_logits.numpy(),
        'labels': all_labels,
    }


def evaluate_ensemble(
    predictions: np.ndarray,
    labels: np.ndarray,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    评估 ensemble 结果
    """
    # 过滤掉 label=-1 的样本 (测试集没有标签)
    valid_mask = labels >= 0
    if valid_mask.sum() == 0:
        print("警告: 没有有效标签，无法评估")
        return {}
    
    valid_preds = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    
    # 计算指标
    macro_f1 = f1_score(valid_labels, valid_preds, average='macro')
    micro_f1 = f1_score(valid_labels, valid_preds, average='micro')
    weighted_f1 = f1_score(valid_labels, valid_preds, average='weighted')
    accuracy = accuracy_score(valid_labels, valid_preds)
    
    print("\n" + "=" * 60)
    print("Ensemble 评估结果")
    print("=" * 60)
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Micro-F1:    {micro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")
    print(f"Accuracy:    {accuracy:.4f}")
    print("=" * 60)
    
    # 详细分类报告
    print("\n分类报告:")
    print(classification_report(valid_labels, valid_preds, digits=4))
    
    # 保存结果
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存混淆矩阵
        cm = confusion_matrix(valid_labels, valid_preds)
        np.save(output_path / 'confusion_matrix.npy', cm)
        
        # 保存指标到文本文件
        with open(output_path / 'metrics.txt', 'w') as f:
            f.write(f"Macro-F1: {macro_f1:.4f}\n")
            f.write(f"Micro-F1: {micro_f1:.4f}\n")
            f.write(f"Weighted-F1: {weighted_f1:.4f}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write("\n分类报告:\n")
            f.write(classification_report(valid_labels, valid_preds, digits=4))
        
        print(f"\n结果已保存到: {output_path}")
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'accuracy': accuracy,
    }


def compare_single_vs_ensemble(
    models: List[torch.nn.Module],
    data_loader: DataLoader,
    device: torch.device,
    ensemble_predictions: np.ndarray,
    labels: np.ndarray,
) -> None:
    """
    对比单模型 vs Ensemble 的性能
    """
    valid_mask = labels >= 0
    if valid_mask.sum() == 0:
        return
    
    valid_labels = labels[valid_mask]
    
    print("\n" + "=" * 60)
    print("单模型 vs Ensemble 对比")
    print("=" * 60)
    print(f"{'Model':<20} {'Macro-F1':<12} {'Accuracy':<12}")
    print("-" * 60)
    
    # 评估每个单模型
    single_f1s = []
    for i, model in enumerate(models):
        all_preds = []
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
                all_preds.append(preds)
        
        all_preds = np.concatenate(all_preds)[valid_mask]
        f1 = f1_score(valid_labels, all_preds, average='macro')
        acc = accuracy_score(valid_labels, all_preds)
        single_f1s.append(f1)
        print(f"Model_{i+1:<14} {f1:<12.4f} {acc:<12.4f}")
    
    # Ensemble 结果
    ens_preds = ensemble_predictions[valid_mask]
    ens_f1 = f1_score(valid_labels, ens_preds, average='macro')
    ens_acc = accuracy_score(valid_labels, ens_preds)
    
    print("-" * 60)
    print(f"{'Ensemble':<20} {ens_f1:<12.4f} {ens_acc:<12.4f}")
    print("-" * 60)
    
    # 计算提升
    avg_single_f1 = np.mean(single_f1s)
    max_single_f1 = np.max(single_f1s)
    improvement_avg = (ens_f1 - avg_single_f1) * 100
    improvement_max = (ens_f1 - max_single_f1) * 100
    
    print(f"\nEnsemble vs 单模型平均: {improvement_avg:+.2f}% Macro-F1")
    print(f"Ensemble vs 单模型最佳: {improvement_max:+.2f}% Macro-F1")
    print("=" * 60)


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # ========== 1. 收集 checkpoint 路径 ==========
    if args.checkpoint_paths:
        ckpt_paths = args.checkpoint_paths
    elif args.checkpoint_dir:
        ckpt_paths = find_checkpoints(args.checkpoint_dir)
        if not ckpt_paths:
            raise FileNotFoundError(
                f"在 {args.checkpoint_dir} 下未找到任何 hat_cls_best.pt 文件"
            )
    else:
        raise ValueError("请指定 --checkpoint-dir 或 --checkpoint-paths")
    
    print(f"\n发现 {len(ckpt_paths)} 个 checkpoint:")
    for p in ckpt_paths:
        print(f"  - {p}")
    
    # ========== 2. 加载模型 ==========
    models = load_models(ckpt_paths, device)
    
    # ========== 3. 验证集评估 ==========
    if args.val_path and Path(args.val_path).exists():
        print(f"\n加载验证集: {args.val_path}")
        val_loader = create_dataloader(
            args.val_path,
            args.batch_size,
            args.num_workers,
        )
        
        # Ensemble 预测
        results = ensemble_predict(
            models, val_loader, device,
            method=args.ensemble_method,
        )
        
        # 评估
        metrics = evaluate_ensemble(
            results['predictions'],
            results['labels'],
            output_dir=args.output_dir,
        )
        
        # 对比单模型 vs Ensemble
        compare_single_vs_ensemble(
            models, val_loader, device,
            results['predictions'],
            results['labels'],
        )
        
        # 保存预测结果
        if args.save_predictions:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            np.save(output_path / 'val_predictions.npy', results['predictions'])
            np.save(output_path / 'val_logits_ensemble.npy', results['logits_ensemble'])
            print(f"\n预测结果已保存到: {output_path}")
    
    # ========== 4. 测试集预测 (可选) ==========
    if args.test_path and Path(args.test_path).exists():
        print(f"\n加载测试集: {args.test_path}")
        test_loader = create_dataloader(
            args.test_path,
            args.batch_size,
            args.num_workers,
        )
        
        test_results = ensemble_predict(
            models, test_loader, device,
            method=args.ensemble_method,
        )
        
        # 保存测试集预测
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / 'test_predictions.npy', test_results['predictions'])
        
        # 生成提交格式的 CSV
        test_df = pd.read_csv(args.test_path, sep='\t')
        submission_df = pd.DataFrame({
            'id': range(len(test_results['predictions'])),
            'label': test_results['predictions'],
        })
        submission_path = output_path / 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"\n提交文件已保存到: {submission_path}")
    
    print("\n✓ Ensemble 评估完成！")


if __name__ == "__main__":
    main()

