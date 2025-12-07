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


def _parse_int_list(text: str) -> List[int]:
    """将逗号分隔的整数串解析为列表，过滤空项"""
    parts = [p.strip() for p in text.split(',')]
    ints = []
    for p in parts:
        if not p:
            continue
        try:
            ints.append(int(p))
        except ValueError:
            raise ValueError(f"无法解析整数: {p}")
    return ints


def _parse_float_list(text: str) -> List[float]:
    """将逗号分隔的浮点数串解析为列表"""
    parts = [p.strip() for p in text.split(',')]
    floats: List[float] = []
    for p in parts:
        if not p:
            continue
        try:
            floats.append(float(p))
        except ValueError:
            raise ValueError(f"无法解析浮点数: {p}")
    return floats


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
    
    parser.add_argument(
        '--window-tta-offsets',
        type=str,
        default='0',
        help='窗口 TTA 的 token offset 列表，逗号分隔。默认只使用 offset=0'
    )
    
    parser.add_argument(
        '--mc-dropout-runs',
        type=int,
        default=1,
        help='MC Dropout 前向次数 (>1 时推理阶段仅对 Dropout 启用训练模式)'
    )
    
    parser.add_argument(
        '--decision-threshold',
        type=float,
        default=None,
        help='二分类时的正类概率阈值，例如 0.6 表示 p(正类)>=0.6 才预测正类'
    )
    
    parser.add_argument(
        '--class-thresholds',
        type=str,
        default=None,
        help='按类别的概率阈值，逗号分隔，长度需等于 num_labels；若提供则优先于 decision-threshold'
    )
    parser.add_argument(
        '--tune-class-thresholds',
        action='store_true',
        help='在验证集上网格搜索统一阈值（14 类场景建议配合 --val-path 使用）'
    )
    parser.add_argument(
        '--threshold-grid',
        type=str,
        default='0.30,0.35,0.40,0.45,0.50,0.55,0.60',
        help='阈值网格，逗号分隔，用于 tune-class-thresholds'
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
        choices=['mean', 'max', 'mean_conf'],
        default='mean',
        help='窗口级聚合方式: mean / max / mean_conf(按窗口置信度加权平均)'
    )
    parser.add_argument(
        '--model-agg',
        type=str,
        choices=[
            'logits_avg',          # logits 均值
            'prob_avg',            # 概率均值
            'voting',              # 投票
            'logits_avg_weighted', # 按模型权重加权 logits 均值
            'prob_avg_weighted',   # 按模型权重加权概率均值
        ],
        default='logits_avg',
        help='模型级聚合方式: logits_avg / prob_avg / voting / *_weighted(按验证集指标加权)'
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
    4. 支持窗口 TTA（多种 token offset 视角）
    """
    
    def __init__(
        self,
        texts: List[str],
        doc_ids: List[int],
        tokenizer,
        segmenter,
        window_tta_offsets: Optional[List[int]] = None,
    ):
        """
        Args:
            texts: 文本列表（空格分隔的 token ID 字符串）
            doc_ids: 文档 ID 列表
            tokenizer: HATTokenizer 实例
            segmenter: DocumentSegmenter 实例
            window_tta_offsets: TTA 的 token offset 列表（含 0），按 offset 切子串做滑窗
        """
        self.texts = texts
        self.doc_ids = doc_ids
        self.tokenizer = tokenizer
        self.segmenter = segmenter
        self.window_tta_offsets = window_tta_offsets or [0]
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个样本，返回该文档的所有滑动窗口
        
        Returns:
            dict: {
                "doc_id": int,
                "windows": List[SegmentedDocument],  # 滑动窗口列表（含 TTA 视角）
            }
        """
        text = self.texts[idx]
        doc_id = self.doc_ids[idx]
        
        # Tokenize
        token_ids = self.tokenizer.encode(text)
        
        windows: List = []
        for offset in self.window_tta_offsets:
            if offset <= 0:
                token_view = token_ids
            elif offset >= len(token_ids):
                continue  # 该 offset 已超出长度，跳过
            else:
                token_view = token_ids[offset:]
            
            windows.extend(self.segmenter.get_sliding_windows(token_view))
        
        # 防御：确保至少有一个窗口
        if not windows:
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
# 推理辅助：MC Dropout & 阈值策略
# =============================================================================

def _enable_dropout_only(model: torch.nn.Module, train: bool = True) -> None:
    """仅对 Dropout 模块打开训练模式，避免 BatchNorm 等进入 train。"""
    dropout_layers = (
        torch.nn.Dropout,
        torch.nn.Dropout1d,
        torch.nn.Dropout2d,
        torch.nn.Dropout3d,
        torch.nn.AlphaDropout,
    )
    for module in model.modules():
        if isinstance(module, dropout_layers):
            module.train(mode=train)


def _apply_decision_thresholds(
    agg_logits: torch.Tensor,
    agg_probs: torch.Tensor,
    num_labels: int,
    decision_threshold: Optional[float] = None,
    class_thresholds: Optional[torch.Tensor] = None,
    fallback_vote: Optional[np.ndarray] = None,
) -> int:
    """
    根据阈值/投票/argmax 生成最终类别
    优先级：class_thresholds > binary decision_threshold > voting > argmax
    """
    if class_thresholds is not None:
        if class_thresholds.numel() != num_labels:
            raise ValueError("class_thresholds 长度必须等于 num_labels")
        mask = agg_probs >= class_thresholds
        if mask.any():
            masked_probs = agg_probs * mask.float()
            return int(masked_probs.argmax(dim=-1).item())
    
    if num_labels == 2 and decision_threshold is not None:
        return int(agg_probs[1] >= decision_threshold)
    
    if fallback_vote is not None:
        return int(fallback_vote.argmax())
    
    return int(agg_logits.argmax(dim=-1).item())


# =============================================================================
# 模型加载
# =============================================================================

def load_models(model_paths: str, device: torch.device) -> Tuple[List[torch.nn.Module], np.ndarray]:
    """
    加载多个模型 checkpoint，并返回模型及其权重
    
    Args:
        model_paths: 逗号分隔的 checkpoint 路径
        device: 目标设备
        
    Returns:
        (模型列表, 模型权重数组)，权重来源于 checkpoint 的 val_macro_f1
    """
    from src.model import create_model, HATConfig
    
    paths = [p.strip() for p in model_paths.split(',') if p.strip()]
    models: List[torch.nn.Module] = []
    model_scores: List[float] = []
    config = HATConfig()
    
    log_print(f"\n加载 {len(paths)} 个模型...")
    
    for i, path in enumerate(paths):
        log_print(f"  [{i+1}/{len(paths)}] {path}")
        
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {path}")
        
        # 创建模型
        model = create_model(config)
        
        # 加载权重
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict, strict=True)
        
        # 移动到设备并设为评估模式
        model.to(device)
        model.eval()
        
        # 打印 checkpoint 信息并记录作为权重
        score = ckpt.get('val_macro_f1', None)
        if score is not None:
            log_print(f"      Val Macro-F1: {score:.4f}")
            model_scores.append(float(score))
        else:
            log_print("      (未找到 val_macro_f1，将使用均匀权重)")
            model_scores.append(1.0)
        
        models.append(model)
    
    log_print(f"  共加载 {len(models)} 个模型")
    
    scores = np.asarray(model_scores, dtype=np.float32)
    if scores.sum() <= 0:
        weights = np.ones_like(scores) / len(scores)
    else:
        weights = scores / scores.sum()
    
    log_print(f"  模型权重: {weights.tolist()}")
    return models, weights


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
    mc_dropout_runs: int = 1,
) -> Dict[int, torch.Tensor]:
    """
    对单个模型进行推理，并在窗口维度聚合
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        num_labels: 类别数
        window_agg: 窗口聚合方式 ('mean' or 'max')
        mc_dropout_runs: MC Dropout 前向次数（>1 时启用 Dropout）
        
    Returns:
        doc_logits: {doc_id: Tensor[num_labels]} - 每个文档的聚合 logits
    """
    # 先整体设为 eval，再仅对 Dropout 打开训练模式，避免 BatchNorm 等进入 train
    model.eval()
    enable_mc_dropout = mc_dropout_runs and mc_dropout_runs > 1
    if enable_mc_dropout:
        _enable_dropout_only(model, train=True)
    
    # 按 doc_id 收集窗口 logits
    doc_window_logits = defaultdict(list)  # doc_id -> [Tensor[num_labels], ...]
    
    for batch in tqdm(test_loader, desc="Inference", leave=False):
        input_ids = batch["input_ids"].to(device)       # [B, N, K]
        attention_mask = batch["attention_mask"].to(device)  # [B, N, K]
        doc_ids = batch["doc_ids"]                       # [B]
        
        # 前向传播（支持 MC Dropout）
        if enable_mc_dropout:
            logits_runs = []
            for _ in range(mc_dropout_runs):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits_runs.append(outputs["logits"])
            logits = torch.stack(logits_runs, dim=0).mean(dim=0)  # [B, num_labels]
        else:
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
        elif window_agg == 'mean_conf':
            # 用窗口最大类别概率作为置信度做加权平均
            probs = torch.softmax(stacked, dim=-1)       # [W, num_labels]
            conf = probs.max(dim=-1).values              # [W]
            weights = conf / (conf.sum() + 1e-12)        # 归一化
            agg_logits = (stacked * weights.view(-1, 1)).sum(dim=0)
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
    model_weights: Optional[np.ndarray] = None,
    mc_dropout_runs: int = 1,
    decision_threshold: Optional[float] = None,
    class_thresholds: Optional[np.ndarray] = None,
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
    
    # 预处理模型权重
    weight_tensor: Optional[torch.Tensor] = None
    if model_weights is not None:
        weight_tensor = torch.as_tensor(model_weights, dtype=torch.float32)
    
    for i, model in enumerate(models):
        log_print(f"\n  模型 [{i+1}/{len(models)}] 推理中...")
        
        # 获取该模型对每个 doc 的 logits（已窗口聚合）
        doc_logits = infer_single_model(
            model=model,
            test_loader=test_loader,
            device=device,
            num_labels=num_labels,
            window_agg=window_agg,
            mc_dropout_runs=mc_dropout_runs,
        )
        
        # 添加到聚合容器
        for doc_id, logits in doc_logits.items():
            logits_dict[doc_id].append(logits)
    
    # 模型级聚合
    preds = {}
    ensemble_logits = {}
    
    log_print("\n  模型级聚合...")
    
    # 预处理阈值
    class_thresholds_tensor = None
    if class_thresholds is not None:
        class_thresholds_tensor = torch.as_tensor(class_thresholds, dtype=torch.float32)
    
    for doc_id, logit_list in logits_dict.items():
        stacked = torch.stack(logit_list, dim=0)  # [M, num_labels]
        
        # 对齐权重长度（正常情况下 M==模型数）
        w = None
        if weight_tensor is not None and weight_tensor.numel() >= stacked.size(0):
            w = weight_tensor[: stacked.size(0)]
            w = w / (w.sum() + 1e-12)  # 归一化
        
        vote_counts = None
        if model_agg == 'logits_avg':
            # Logits 平均
            agg_logits = stacked.mean(dim=0)  # [num_labels]
            agg_probs = torch.softmax(agg_logits, dim=-1)
            
        elif model_agg == 'logits_avg_weighted':
            if w is None:
                agg_logits = stacked.mean(dim=0)
            else:
                agg_logits = (stacked * w.view(-1, 1)).sum(dim=0)
            agg_probs = torch.softmax(agg_logits, dim=-1)
            
        elif model_agg == 'prob_avg':
            # 概率平均（softmax 后平均）
            probs = torch.softmax(stacked, dim=-1)  # [M, num_labels]
            agg_probs = probs.mean(dim=0)  # [num_labels]
            agg_logits = torch.log(agg_probs + 1e-10)
            
        elif model_agg == 'prob_avg_weighted':
            probs = torch.softmax(stacked, dim=-1)  # [M, num_labels]
            if w is None:
                agg_probs = probs.mean(dim=0)
            else:
                agg_probs = (probs * w.view(-1, 1)).sum(dim=0)
            agg_logits = torch.log(agg_probs + 1e-10)
            
        elif model_agg == 'voting':
            # 投票
            votes = stacked.argmax(dim=-1).numpy()  # [M]
            vote_counts = np.bincount(votes, minlength=num_labels)
            agg_logits = stacked.mean(dim=0)  # 保存平均 logits 用于分析
            agg_probs = torch.softmax(agg_logits, dim=-1)
            
        else:
            raise ValueError(f"Unknown model aggregation: {model_agg}")
        
        label = _apply_decision_thresholds(
            agg_logits=agg_logits,
            agg_probs=agg_probs,
            num_labels=num_labels,
            decision_threshold=decision_threshold,
            class_thresholds=class_thresholds_tensor,
            fallback_vote=vote_counts if model_agg == 'voting' else None,
        )
        
        preds[doc_id] = int(label)
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
    model_weights: Optional[np.ndarray] = None,
    mc_dropout_runs: int = 1,
    window_tta_offsets: Optional[List[int]] = None,
    decision_threshold: Optional[float] = None,
    class_thresholds: Optional[np.ndarray] = None,
    tune_class_thresholds: bool = False,
    threshold_grid: Optional[List[float]] = None,
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
        window_tta_offsets=window_tta_offsets,
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
    preds, ensemble_logits = run_inference(
        models=models,
        test_loader=val_loader,
        device=device,
        num_labels=num_labels,
        window_agg=window_agg,
        model_agg=model_agg,
        model_weights=model_weights,
        mc_dropout_runs=mc_dropout_runs,
        decision_threshold=decision_threshold,
        class_thresholds=class_thresholds,
    )
    
    # 评估
    labels_pred = [preds[doc_id] for doc_id in doc_ids]
    
    macro_f1 = f1_score(labels_true, labels_pred, average='macro')
    accuracy = accuracy_score(labels_true, labels_pred)
    
    log_print(f"\n  Val Macro-F1: {macro_f1:.4f}")
    log_print(f"  Val Accuracy: {accuracy:.4f}")
    
    # 可选：在验证集上网格搜索统一类阈值（多类时作为 soft filter）
    if tune_class_thresholds and threshold_grid:
        log_print("\n  启动阈值网格搜索 (统一阈值，按类过滤后再 argmax)...")
        # 将 logits 转 prob
        doc_ids_sorted = sorted(doc_ids)
        logits_tensor = torch.tensor([ensemble_logits[i] for i in doc_ids_sorted], dtype=torch.float32)
        probs_tensor = torch.softmax(logits_tensor, dim=-1)
        labels_true_tensor = torch.tensor(labels_true, dtype=torch.int64)
        
        best_f1 = -1.0
        best_t = None
        
        for t in threshold_grid:
            th = torch.full((num_labels,), float(t), dtype=torch.float32)
            preds_t = []
            for p, logit in zip(probs_tensor, logits_tensor):
                label = _apply_decision_thresholds(
                    agg_logits=logit,
                    agg_probs=p,
                    num_labels=num_labels,
                    decision_threshold=None,            # 14 类，不用二分类阈值
                    class_thresholds=th,
                    fallback_vote=None,
                )
                preds_t.append(label)
            macro_f1_t = f1_score(labels_true_tensor, preds_t, average='macro')
            if macro_f1_t > best_f1:
                best_f1 = macro_f1_t
                best_t = t
        
        if best_t is not None:
            log_print(f"  网格搜索最佳统一阈值: {best_t:.2f}, Val Macro-F1: {best_f1:.4f}")
            suggested = ",".join([f"{best_t:.2f}" for _ in range(num_labels)])
            log_print(f"  建议推理时添加参数: --class-thresholds {suggested}")
        else:
            log_print("  阈值网格搜索未找到更优解，保持默认 argmax。")


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
    models, model_weights = load_models(args.model_paths, device)
    
    # 解析窗口 TTA offsets
    window_tta_offsets = _parse_int_list(args.window_tta_offsets)
    if 0 not in window_tta_offsets:
        window_tta_offsets = [0] + window_tta_offsets
    window_tta_offsets = sorted(list({o for o in window_tta_offsets if o >= 0}))
    
    # 解析类别阈值
    class_thresholds = _parse_float_list(args.class_thresholds) if args.class_thresholds else None
    threshold_grid = _parse_float_list(args.threshold_grid) if args.threshold_grid else []
    
    # 获取 num_labels
    from src.common_config import COMMON_CONFIG
    num_labels = COMMON_CONFIG.num_labels
    log_print(f"\n类别数: {num_labels}")
    
    if class_thresholds and len(class_thresholds) != num_labels:
        raise ValueError(
            f"class-thresholds 长度({len(class_thresholds)})需等于 num_labels({num_labels})"
        )
    
    log_print(f"窗口 TTA offsets: {window_tta_offsets}")
    if args.mc_dropout_runs > 1:
        log_print(f"MC Dropout 前向次数: {args.mc_dropout_runs}")
    if args.decision_threshold is not None:
        log_print(f"二分类阈值: {args.decision_threshold}")
    if class_thresholds:
        log_print(f"类别阈值: {class_thresholds}")
    
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
            model_weights=model_weights,
            mc_dropout_runs=args.mc_dropout_runs,
            window_tta_offsets=window_tta_offsets,
            decision_threshold=args.decision_threshold,
            class_thresholds=class_thresholds,
            tune_class_thresholds=args.tune_class_thresholds,
            threshold_grid=threshold_grid,
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
        window_tta_offsets=window_tta_offsets,
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
        model_weights=model_weights,
        mc_dropout_runs=args.mc_dropout_runs,
        decision_threshold=args.decision_threshold,
        class_thresholds=class_thresholds,
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

