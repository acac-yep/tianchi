#!/usr/bin/env python3
"""
模型多样性分析脚本

功能:
1. 检查多个 checkpoint 的预测是否完全一致
2. 分析模型权重差异
3. 可视化错误样本分布

使用方法:
    python scripts/analyze_model_diversity.py \
        --checkpoint-dir checkpoints/ensemble_focal \
        --val-path data/processed/val.csv
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from collections import Counter

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description='模型多样性分析')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='包含多个 checkpoint 的目录')
    parser.add_argument('--val-path', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'processed' / 'val.csv'),
                        help='验证数据路径')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-samples', type=int, default=500,
                        help='用于详细分析的样本数')
    return parser.parse_args()


def find_checkpoints(checkpoint_dir: str) -> List[str]:
    """扫描 checkpoint"""
    pattern = os.path.join(checkpoint_dir, '**/hat_cls_best.pt')
    ckpts = glob.glob(pattern, recursive=True)
    return sorted(ckpts)


def load_model(ckpt_path: str, device: torch.device):
    """加载单个模型"""
    from src.model import create_model, HATConfig
    
    config = HATConfig()
    model = create_model(config)
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    model.eval()
    
    return model, ckpt


def analyze_weight_similarity(ckpt_paths: List[str]) -> Dict:
    """分析模型权重相似度"""
    print("\n" + "=" * 60)
    print("权重相似度分析")
    print("=" * 60)
    
    # 加载所有 checkpoint 的 state_dict
    state_dicts = []
    for path in ckpt_paths:
        ckpt = torch.load(path, map_location='cpu')
        sd = ckpt.get('model_state_dict', ckpt)
        state_dicts.append(sd)
    
    # 计算权重差异
    all_keys = list(state_dicts[0].keys())
    weight_diffs = []
    
    print(f"\n对比 {len(state_dicts)} 个模型的权重...")
    
    for key in all_keys[:10]:  # 只检查前10个参数
        weights = [sd[key].float().numpy().flatten() for sd in state_dicts]
        
        # 计算每对模型之间的最大差异
        max_diff = 0
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                diff = np.max(np.abs(weights[i] - weights[j]))
                max_diff = max(max_diff, diff)
        
        weight_diffs.append((key, max_diff))
        if max_diff > 1e-6:
            print(f"  {key[:50]}: 最大差异 = {max_diff:.2e}")
    
    # 判断是否完全相同
    total_max_diff = max(d[1] for d in weight_diffs)
    
    if total_max_diff < 1e-6:
        print("\n⚠️  所有模型权重完全相同！")
        print("   这意味着 Stage2 训练没有产生任何变化。")
    else:
        print(f"\n权重最大差异: {total_max_diff:.2e}")
    
    return {'max_diff': total_max_diff, 'per_layer': weight_diffs}


@torch.no_grad()
def get_predictions(model, data_loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """获取模型预测"""
    all_preds = []
    all_logits = []
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_logits.append(logits.cpu().numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_logits, axis=0)


def analyze_prediction_diversity(
    ckpt_paths: List[str],
    data_loader: DataLoader,
    labels: np.ndarray,
    device: torch.device,
) -> Dict:
    """分析预测多样性"""
    print("\n" + "=" * 60)
    print("预测多样性分析")
    print("=" * 60)
    
    # 收集每个模型的预测
    all_predictions = []
    all_model_logits = []
    
    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"\n加载模型 [{i+1}/{len(ckpt_paths)}]: {Path(ckpt_path).parent.name}")
        model, ckpt = load_model(ckpt_path, device)
        
        preds, logits = get_predictions(model, data_loader, device)
        all_predictions.append(preds)
        all_model_logits.append(logits)
        
        # 清理内存
        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    all_predictions = np.array(all_predictions)  # [num_models, num_samples]
    all_model_logits = np.array(all_model_logits)  # [num_models, num_samples, num_classes]
    
    num_models = len(ckpt_paths)
    num_samples = len(labels)
    
    # 1. 检查预测完全一致的比例
    pred_agree_all = np.all(all_predictions == all_predictions[0:1], axis=0)
    agree_ratio = pred_agree_all.mean()
    
    print(f"\n=== 预测一致性分析 ===")
    print(f"样本数: {num_samples}")
    print(f"所有模型预测一致的样本比例: {agree_ratio:.2%} ({pred_agree_all.sum()}/{num_samples})")
    
    if agree_ratio > 0.999:
        print("\n⚠️  几乎所有样本的预测完全一致！")
        print("   这证实了 5 个模型实际上是同一个模型的拷贝。")
    
    # 2. 分析不一致样本
    disagree_indices = np.where(~pred_agree_all)[0]
    print(f"\n不一致样本数: {len(disagree_indices)}")
    
    if len(disagree_indices) > 0:
        print("\n不一致样本的预测分布 (前20个):")
        for idx in disagree_indices[:20]:
            preds_at_idx = all_predictions[:, idx]
            true_label = labels[idx]
            print(f"  样本 {idx}: 真实={true_label}, 预测={list(preds_at_idx)}")
    
    # 3. 每个模型的错误集合对比
    print(f"\n=== 错误样本集合分析 ===")
    error_sets = []
    for i, preds in enumerate(all_predictions):
        errors = set(np.where(preds != labels)[0])
        error_sets.append(errors)
        f1 = f1_score(labels, preds, average='macro')
        print(f"Model_{i+1}: {len(errors)} 错误, Macro-F1={f1:.4f}")
    
    # 4. 错误集合交集/并集
    if len(error_sets) >= 2:
        intersection = error_sets[0]
        union = error_sets[0]
        for es in error_sets[1:]:
            intersection = intersection & es
            union = union | es
        
        print(f"\n错误样本交集大小: {len(intersection)} (所有模型都错的)")
        print(f"错误样本并集大小: {len(union)} (至少一个模型错的)")
        
        if len(union) > 0:
            jaccard = len(intersection) / len(union)
            print(f"Jaccard 相似度: {jaccard:.4f} (1.0 表示完全相同)")
            
            if jaccard > 0.99:
                print("\n⚠️  错误集合几乎完全相同！")
                print("   Ensemble 无法提供修正效果。")
    
    # 5. Logits 差异分析
    print(f"\n=== Logits 差异分析 ===")
    logits_std = np.std(all_model_logits, axis=0)  # [num_samples, num_classes]
    avg_logits_std = logits_std.mean()
    max_logits_std = logits_std.max()
    
    print(f"Logits 标准差平均值: {avg_logits_std:.6f}")
    print(f"Logits 标准差最大值: {max_logits_std:.6f}")
    
    if avg_logits_std < 1e-5:
        print("\n⚠️  Logits 几乎没有差异！")
        print("   模型输出完全相同。")
    
    return {
        'agree_ratio': agree_ratio,
        'disagree_count': len(disagree_indices),
        'predictions': all_predictions,
        'logits': all_model_logits,
        'error_sets': error_sets,
    }


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 找 checkpoint
    ckpt_paths = find_checkpoints(args.checkpoint_dir)
    if not ckpt_paths:
        raise FileNotFoundError(f"在 {args.checkpoint_dir} 下未找到 checkpoint")
    
    print(f"\n发现 {len(ckpt_paths)} 个 checkpoint:")
    for p in ckpt_paths:
        print(f"  - {p}")
    
    # 2. 分析权重相似度
    weight_analysis = analyze_weight_similarity(ckpt_paths)
    
    # 3. 创建数据加载器
    from src.data_preprocess import (
        HATDataset, HATDataCollator, 
        create_tokenizer, create_segmenter
    )
    
    print(f"\n加载验证集: {args.val_path}")
    df = pd.read_csv(args.val_path, sep='\t')
    texts = df['text'].tolist()
    labels = df['label'].astype(int).values
    
    tokenizer = create_tokenizer()
    segmenter = create_segmenter()
    
    dataset = HATDataset(
        texts=texts,
        labels=labels.tolist(),
        tokenizer=tokenizer,
        segmenter=segmenter,
        mode='eval',
        cache_segments=False,
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=HATDataCollator(),
        pin_memory=True,
    )
    
    # 4. 分析预测多样性
    pred_analysis = analyze_prediction_diversity(
        ckpt_paths, data_loader, labels, device
    )
    
    # 5. 总结
    print("\n" + "=" * 60)
    print("总结与建议")
    print("=" * 60)
    
    if weight_analysis['max_diff'] < 1e-6 and pred_analysis['agree_ratio'] > 0.999:
        print("""
诊断结果: 5 个模型是完全相同的！

原因分析:
1. Stage2 训练从同一个 Stage1 checkpoint 开始
2. 学习率 3e-5 太保守，配合 early stopping
3. 训练没有产生实质改进，保存的是 baseline checkpoint

解决方案:
1. 【推荐】运行新的多样性训练:
   sbatch scripts/slurm_scripts/train_diverse_ensemble.sh
   
2. 关键改动:
   - 更大学习率 (5e-5 ~ 1e-4)
   - 部分冻结底层 encoder
   - 不同的 dropout 配置
   - 不同的滑窗策略

3. 快速验证:
   运行完后再次执行本脚本检查多样性
""")
    else:
        improvement = (1 - pred_analysis['agree_ratio']) * 100
        print(f"\n模型存在 {improvement:.2f}% 的预测差异")
        print("可以尝试 ensemble，但提升可能有限。")
    
    print("\n✓ 分析完成！")


if __name__ == "__main__":
    main()

