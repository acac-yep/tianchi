#!/usr/bin/env python3
"""
HAT 模型推理脚本

功能:
1. 支持多个 checkpoint 的 ensemble（多 seed）
2. 支持滑动窗口处理超长文档
3. 窗口级聚合 + 模型级聚合
4. 生成符合天池竞赛格式的 submission.csv

数据流:
    1. 从 data/processed/test.csv 读取测试文本
    2. 对每个文档使用 segmenter.get_sliding_windows() 生成多个窗口
    3. 对每个窗口、每个模型跑一次前向，得到 logits
    4. 先在窗口维度聚合（同一 doc 的多个窗口求平均）
    5. 再在模型维度聚合（多模型 ensemble 求平均）
    6. 得到最终类别，写成 submission.csv

使用方法:
    # 单模型推理
    python scripts/infer.py \
        --test-path data/processed/test.csv \
        --model-paths checkpoints/cls_hat512/hat_cls_best.pt \
        --output-path submission.csv

    # 多模型 ensemble
    python scripts/infer.py \
        --test-path data/processed/test.csv \
        --model-paths checkpoints/cls_hat512/seed42_best.pt,checkpoints/cls_hat512/seed3407_best.pt \
        --output-path submission.csv \
        --batch-size 64

Author: HAT Project
Date: 2024
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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
        description='HAT 模型推理脚本（支持滑动窗口 + 多模型 ensemble）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据路径
    parser.add_argument(
        '--test-path',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'processed' / 'test.csv'),
        help='测试数据路径（预处理后的 test.csv，包含 text 列）'
    )
    
    # 模型路径（支持多个，用逗号分隔）
    parser.add_argument(
        '--model-paths',
        type=str,
        required=True,
        help='模型 checkpoint 路径，多个路径用逗号分隔'
    )
    
    # 输出路径
    parser.add_argument(
        '--output-path',
        type=str,
        default=str(PROJECT_ROOT / 'submission.csv'),
        help='预测结果输出路径'
    )
    
    # 参考的提交样例（用于确保格式一致）
    parser.add_argument(
        '--sample-submit-path',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'test_a_sample_submit.csv'),
        help='天池提交样例文件路径'
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
        help='设备（cuda / cpu）'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='DataLoader worker 数'
    )
    
    # 聚合策略
    parser.add_argument(
        '--window-agg',
        type=str,
        choices=['mean', 'max'],
        default='mean',
        help='窗口级聚合方式: mean (平均), max (取最大)'
    )
    parser.add_argument(
        '--model-agg',
        type=str,
        choices=['logits_avg', 'prob_avg', 'voting'],
        default='logits_avg',
        help='模型级聚合方式: logits_avg, prob_avg, voting'
    )
    
    # 可选: 保存详细 logits 用于调试/分析
    parser.add_argument(
        '--save-logits',
        action='store_true',
        help='是否保存聚合后的 logits（用于调试）'
    )
    
    # 可选: 验证集路径（用于 sanity check）
    parser.add_argument(
        '--val-path',
        type=str,
        default=None,
        help='验证集路径（可选，用于 sanity check）'
    )
    
    return parser.parse_args()


# =============================================================================
# 推理专用 Dataset
# =============================================================================

class InferenceDataset(Dataset):
    """
    推理专用 Dataset
    
    特点：
    1. 无标签
    2. 保留 doc_id 用于聚合
    3. 支持滑动窗口（在 collator 中展开）
    """
    
    def __init__(
        self,
        texts: List[str],
        doc_ids: List[int],
        tokenizer,
        segmenter,
    ):
        """
        Args:
            texts: 文本列表（空格分隔的 token ID 字符串）
            doc_ids: 文档 ID 列表
            tokenizer: HATTokenizer 实例
            segmenter: DocumentSegmenter 实例
        """
        self.texts = texts
        self.doc_ids = doc_ids
        self.tokenizer = tokenizer
        self.segmenter = segmenter
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个样本，返回该文档的所有滑动窗口
        
        Returns:
            dict: {
                "doc_id": int,
                "windows": List[SegmentedDocument],  # 滑动窗口列表
            }
        """
        text = self.texts[idx]
        doc_id = self.doc_ids[idx]
        
        # Tokenize
        token_ids = self.tokenizer.encode(text)
        
        # 获取滑动窗口
        windows = self.segmenter.get_sliding_windows(token_ids)
        
        return {
            "doc_id": doc_id,
            "windows": windows,
        }


class InferenceCollator:
    """
    推理专用 Collator
    
    将多个文档的多个窗口展平成一个 batch，同时记录 doc_id 以便聚合。
    """
    
    def __init__(
        self,
        max_segments: int = 8,
        segment_length: int = 512,
        pad_token_id: int = 0,
    ):
        self.max_segments = max_segments
        self.segment_length = segment_length
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        整理一个 batch 的数据
        
        Args:
            batch: List[{"doc_id": int, "windows": List[SegmentedDocument]}]
            
        Returns:
            dict: {
                "input_ids": [B_total, N, K],
                "attention_mask": [B_total, N, K],
                "doc_ids": [B_total],  # 每个窗口对应的 doc_id
            }
        """
        all_segment_ids = []
        all_attention_masks = []
        all_doc_ids = []
        
        for item in batch:
            doc_id = item["doc_id"]
            windows = item["windows"]
            
            for window in windows:
                # window 是 SegmentedDocument 对象
                # segment_ids: [N, K], segment_attention_masks: [N, K]
                num_segs = window.num_segments
                
                # Padding 到 max_segments
                padded_ids = np.full(
                    (self.max_segments, self.segment_length),
                    self.pad_token_id,
                    dtype=np.int64
                )
                padded_mask = np.zeros(
                    (self.max_segments, self.segment_length),
                    dtype=np.int64
                )
                
                # 填充实际数据
                padded_ids[:num_segs] = window.segment_ids
                padded_mask[:num_segs] = window.segment_attention_masks
                
                all_segment_ids.append(padded_ids)
                all_attention_masks.append(padded_mask)
                all_doc_ids.append(doc_id)
        
        # 转换为 tensor
        input_ids = torch.tensor(np.stack(all_segment_ids), dtype=torch.long)
        attention_mask = torch.tensor(np.stack(all_attention_masks), dtype=torch.long)
        doc_ids = torch.tensor(all_doc_ids, dtype=torch.long)
        
        return {
            "input_ids": input_ids,         # [B_total, N, K]
            "attention_mask": attention_mask,  # [B_total, N, K]
            "doc_ids": doc_ids,             # [B_total]
        }


# =============================================================================
# 模型加载
# =============================================================================

def load_models(model_paths: str, device: torch.device) -> List[torch.nn.Module]:
    """
    加载多个模型 checkpoint
    
    Args:
        model_paths: 逗号分隔的 checkpoint 路径
        device: 目标设备
        
    Returns:
        模型列表
    """
    from src.model import create_model, HATConfig
    
    paths = [p.strip() for p in model_paths.split(',') if p.strip()]
    models = []
    config = HATConfig()
    
    log_print(f"\n加载 {len(paths)} 个模型...")
    
    for i, path in enumerate(paths):
        log_print(f"  [{i+1}/{len(paths)}] {path}")
        
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {path}")
        
        # 创建模型
        model = create_model(config)
        
        # 加载权重
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict, strict=True)
        
        # 移动到设备并设为评估模式
        model.to(device)
        model.eval()
        
        # 打印 checkpoint 信息
        if 'val_macro_f1' in ckpt:
            log_print(f"      Val Macro-F1: {ckpt['val_macro_f1']:.4f}")
        
        models.append(model)
    
    log_print(f"  共加载 {len(models)} 个模型")
    return models


# =============================================================================
# 单模型推理（带窗口聚合）
# =============================================================================

@torch.no_grad()
def infer_single_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_labels: int,
    window_agg: str = 'mean',
) -> Dict[int, torch.Tensor]:
    """
    对单个模型进行推理，并在窗口维度聚合
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        num_labels: 类别数
        window_agg: 窗口聚合方式 ('mean' or 'max')
        
    Returns:
        doc_logits: {doc_id: Tensor[num_labels]} - 每个文档的聚合 logits
    """
    model.eval()
    
    # 按 doc_id 收集窗口 logits
    doc_window_logits = defaultdict(list)  # doc_id -> [Tensor[num_labels], ...]
    
    for batch in tqdm(test_loader, desc="Inference", leave=False):
        input_ids = batch["input_ids"].to(device)       # [B, N, K]
        attention_mask = batch["attention_mask"].to(device)  # [B, N, K]
        doc_ids = batch["doc_ids"]                       # [B]
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs["logits"]  # [B, num_labels]
        
        # 收集每个窗口的 logits
        for i in range(logits.size(0)):
            doc_id = doc_ids[i].item()
            doc_window_logits[doc_id].append(logits[i].cpu())
    
    # 窗口级聚合
    doc_logits = {}
    for doc_id, window_logits in doc_window_logits.items():
        stacked = torch.stack(window_logits, dim=0)  # [W, num_labels]
        
        if window_agg == 'mean':
            agg_logits = stacked.mean(dim=0)  # [num_labels]
        elif window_agg == 'max':
            agg_logits = stacked.max(dim=0).values  # [num_labels]
        else:
            raise ValueError(f"Unknown window aggregation: {window_agg}")
        
        doc_logits[doc_id] = agg_logits
    
    return doc_logits


# =============================================================================
# 多模型 Ensemble 推理
# =============================================================================

@torch.no_grad()
def run_inference(
    models: List[torch.nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    num_labels: int,
    window_agg: str = 'mean',
    model_agg: str = 'logits_avg',
) -> Tuple[Dict[int, int], Dict[int, np.ndarray]]:
    """
    多模型 ensemble 推理
    
    流程：
    1. 对每个模型，跑完整个 test_loader，得到每个 doc 的窗口聚合 logits
    2. 在模型维度做聚合
    3. 返回最终预测
    
    Args:
        models: 模型列表
        test_loader: 测试数据加载器
        device: 设备
        num_labels: 类别数
        window_agg: 窗口聚合方式
        model_agg: 模型聚合方式
        
    Returns:
        preds: {doc_id: label} - 每个文档的预测类别
        ensemble_logits: {doc_id: np.ndarray[num_labels]} - 聚合后的 logits
    """
    # logits_dict[doc_id] -> List[Tensor[num_labels]]，每个元素是一个模型的聚合 logits
    logits_dict = defaultdict(list)
    
    log_print(f"\n开始 Ensemble 推理 (window_agg={window_agg}, model_agg={model_agg})...")
    
    for i, model in enumerate(models):
        log_print(f"\n  模型 [{i+1}/{len(models)}] 推理中...")
        
        # 获取该模型对每个 doc 的 logits（已窗口聚合）
        doc_logits = infer_single_model(
            model, test_loader, device, num_labels, window_agg
        )
        
        # 添加到聚合容器
        for doc_id, logits in doc_logits.items():
            logits_dict[doc_id].append(logits)
    
    # 模型级聚合
    preds = {}
    ensemble_logits = {}
    
    log_print("\n  模型级聚合...")
    
    for doc_id, logit_list in logits_dict.items():
        stacked = torch.stack(logit_list, dim=0)  # [M, num_labels]
        
        if model_agg == 'logits_avg':
            # Logits 平均
            agg_logits = stacked.mean(dim=0)  # [num_labels]
            label = agg_logits.argmax(dim=-1).item()
            
        elif model_agg == 'prob_avg':
            # 概率平均（softmax 后平均）
            probs = torch.softmax(stacked, dim=-1)  # [M, num_labels]
            agg_probs = probs.mean(dim=0)  # [num_labels]
            agg_logits = torch.log(agg_probs + 1e-10)
            label = agg_probs.argmax(dim=-1).item()
            
        elif model_agg == 'voting':
            # 投票
            votes = stacked.argmax(dim=-1).numpy()  # [M]
            vote_counts = np.bincount(votes, minlength=num_labels)
            label = vote_counts.argmax()
            agg_logits = stacked.mean(dim=0)  # 保存平均 logits 用于分析
            
        else:
            raise ValueError(f"Unknown model aggregation: {model_agg}")
        
        preds[doc_id] = label
        ensemble_logits[doc_id] = agg_logits.numpy()
    
    return preds, ensemble_logits


# =============================================================================
# 保存结果
# =============================================================================

def save_submission(
    preds: Dict[int, int],
    output_path: str,
    sample_submit_path: Optional[str] = None,
) -> None:
    """
    保存预测结果为天池竞赛格式
    
    Args:
        preds: {doc_id: label} - 预测结果
        output_path: 输出路径
        sample_submit_path: 提交样例路径（用于确保格式一致）
    """
    # 按 doc_id 排序
    sorted_doc_ids = sorted(preds.keys())
    labels = [preds[doc_id] for doc_id in sorted_doc_ids]
    
    # 检查提交样例格式
    if sample_submit_path and Path(sample_submit_path).exists():
        sample_df = pd.read_csv(sample_submit_path)
        
        # 天池格式只有 label 列
        if 'label' in sample_df.columns and sample_df.shape[1] == 1:
            # 只有 label 列的格式
            submission_df = pd.DataFrame({'label': labels})
        elif 'id' in sample_df.columns and 'label' in sample_df.columns:
            # id + label 格式
            submission_df = pd.DataFrame({
                'id': sorted_doc_ids,
                'label': labels,
            })
        else:
            # 默认使用只有 label 的格式
            submission_df = pd.DataFrame({'label': labels})
    else:
        # 默认使用只有 label 的格式（符合天池要求）
        submission_df = pd.DataFrame({'label': labels})
    
    # 保存
    output_dir = Path(output_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    submission_df.to_csv(output_path, index=False)
    log_print(f"\n提交文件已保存到: {output_path}")
    log_print(f"  样本数: {len(labels)}")
    log_print(f"  类别分布: {np.bincount(labels, minlength=14)}")


# =============================================================================
# 可选：在验证集上做 sanity check
# =============================================================================

def evaluate_on_val(
    models: List[torch.nn.Module],
    val_path: str,
    device: torch.device,
    num_labels: int,
    batch_size: int,
    num_workers: int,
    window_agg: str,
    model_agg: str,
) -> None:
    """
    在验证集上评估（sanity check）
    """
    from sklearn.metrics import f1_score, accuracy_score
    from src.data_preprocess import create_tokenizer, create_segmenter
    
    log_print(f"\n在验证集上进行 Sanity Check: {val_path}")
    
    # 读取验证数据
    df = pd.read_csv(val_path, sep='\t')
    texts = df['text'].tolist()
    labels_true = df['label'].astype(int).tolist()
    doc_ids = list(range(len(texts)))
    
    # 创建数据集和加载器
    tokenizer = create_tokenizer()
    segmenter = create_segmenter()
    
    val_dataset = InferenceDataset(
        texts=texts,
        doc_ids=doc_ids,
        tokenizer=tokenizer,
        segmenter=segmenter,
    )
    
    collator = InferenceCollator()
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    
    # 推理
    preds, _ = run_inference(
        models, val_loader, device, num_labels, window_agg, model_agg
    )
    
    # 评估
    labels_pred = [preds[doc_id] for doc_id in doc_ids]
    
    macro_f1 = f1_score(labels_true, labels_pred, average='macro')
    accuracy = accuracy_score(labels_true, labels_pred)
    
    log_print(f"\n  Val Macro-F1: {macro_f1:.4f}")
    log_print(f"  Val Accuracy: {accuracy:.4f}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    args = parse_args()
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    log_print(f"使用设备: {device}")
    if device.type == 'cuda':
        log_print(f"  GPU: {torch.cuda.get_device_name(0)}")
        log_print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # ========== 1. 加载模型 ==========
    models = load_models(args.model_paths, device)
    
    # 获取 num_labels
    from src.common_config import COMMON_CONFIG
    num_labels = COMMON_CONFIG.num_labels
    log_print(f"\n类别数: {num_labels}")
    
    # ========== 2. 可选：验证集 sanity check ==========
    if args.val_path and Path(args.val_path).exists():
        evaluate_on_val(
            models=models,
            val_path=args.val_path,
            device=device,
            num_labels=num_labels,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            window_agg=args.window_agg,
            model_agg=args.model_agg,
        )
    
    # ========== 3. 构建测试集 DataLoader ==========
    log_print(f"\n加载测试数据: {args.test_path}")
    
    from src.data_preprocess import create_tokenizer, create_segmenter
    
    # 读取测试数据
    test_df = pd.read_csv(args.test_path, sep='\t')
    texts = test_df['text'].tolist()
    doc_ids = list(range(len(texts)))  # 使用行索引作为 doc_id
    
    log_print(f"  测试样本数: {len(texts)}")
    
    tokenizer = create_tokenizer()
    segmenter = create_segmenter()
    
    test_dataset = InferenceDataset(
        texts=texts,
        doc_ids=doc_ids,
        tokenizer=tokenizer,
        segmenter=segmenter,
    )
    
    collator = InferenceCollator()
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 重要：保持顺序
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    
    # ========== 4. 推理 ==========
    start_time = time.time()
    
    preds, ensemble_logits = run_inference(
        models=models,
        test_loader=test_loader,
        device=device,
        num_labels=num_labels,
        window_agg=args.window_agg,
        model_agg=args.model_agg,
    )
    
    elapsed = time.time() - start_time
    log_print(f"\n推理完成，耗时: {elapsed:.1f} 秒")
    
    # ========== 5. 保存结果 ==========
    save_submission(
        preds=preds,
        output_path=args.output_path,
        sample_submit_path=args.sample_submit_path,
    )
    
    # 可选：保存 logits
    if args.save_logits:
        logits_path = Path(args.output_path).with_suffix('.logits.npy')
        logits_array = np.array([ensemble_logits[doc_id] for doc_id in sorted(preds.keys())])
        np.save(logits_path, logits_array)
        log_print(f"Logits 已保存到: {logits_path}")
    
    log_print("\n✓ 推理完成！")


if __name__ == "__main__":
    main()

