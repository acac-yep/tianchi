"""
类别平衡模块
处理类别不平衡问题：类别权重计算、加权采样器等
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from collections import Counter
import logging

from .config import ClassBalanceConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ClassWeightCalculator:
    """
    类别权重计算器
    
    支持多种权重计算方式：
    1. inverse_sqrt: 1 / sqrt(count)
    2. inverse_log: 1 / log(1 + count)
    3. effective_num: Effective Number of Samples 方法
    4. inverse_freq: 1 / count
    """
    
    def __init__(self, config: ClassBalanceConfig = None):
        self.config = config or DEFAULT_CONFIG.class_balance
    
    def compute_weights(
        self, 
        labels: List[int], 
        num_classes: Optional[int] = None,
        method: Optional[str] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        计算类别权重
        
        Args:
            labels: 标签列表
            num_classes: 类别数（如果未指定，从 labels 推断）
            method: 权重计算方法（覆盖配置）
            normalize: 是否归一化（使权重和为 num_classes）
            
        Returns:
            类别权重数组，形状 [num_classes]
        """
        method = method or self.config.weight_method
        
        if method == 'none':
            if num_classes is None:
                num_classes = max(labels) + 1
            return np.ones(num_classes)
        
        # 统计类别频率
        label_counts = Counter(labels)
        
        if num_classes is None:
            num_classes = max(labels) + 1
        
        # 初始化权重
        weights = np.zeros(num_classes)
        
        for c in range(num_classes):
            count = label_counts.get(c, 1)  # 避免除零
            
            if method == 'inverse_sqrt':
                weights[c] = 1.0 / np.sqrt(count)
            elif method == 'inverse_log':
                weights[c] = 1.0 / np.log1p(count)
            elif method == 'inverse_freq':
                weights[c] = 1.0 / count
            elif method == 'effective_num':
                # Effective Number of Samples: (1 - beta^n) / (1 - beta)
                beta = self.config.effective_num_beta
                effective_num = (1.0 - np.power(beta, count)) / (1.0 - beta)
                weights[c] = 1.0 / effective_num
            else:
                raise ValueError(f"Unknown weight method: {method}")
        
        # 归一化
        if normalize:
            weights = weights / weights.sum() * num_classes
        
        return weights
    
    def compute_sample_weights(
        self, 
        labels: List[int],
        class_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算每个样本的权重（用于 WeightedRandomSampler）
        
        Args:
            labels: 标签列表
            class_weights: 类别权重（如果未指定，自动计算）
            
        Returns:
            样本权重数组，形状 [num_samples]
        """
        if class_weights is None:
            class_weights = self.compute_weights(labels)
        
        sample_weights = np.array([class_weights[label] for label in labels])
        
        # 应用平滑因子
        smoothing = self.config.sampler_smoothing
        if smoothing < 1.0:
            # 平滑：在均匀权重和计算权重之间插值
            uniform_weight = 1.0 / len(sample_weights)
            sample_weights = smoothing * sample_weights + (1 - smoothing) * uniform_weight
        
        return sample_weights


class ClassDistributionAnalyzer:
    """类别分布分析器"""
    
    @staticmethod
    def analyze(
        labels: List[int], 
        num_classes: Optional[int] = None
    ) -> Dict:
        """
        分析类别分布
        
        Returns:
            包含分布统计的字典
        """
        label_counts = Counter(labels)
        
        if num_classes is None:
            num_classes = max(labels) + 1
        
        counts = [label_counts.get(c, 0) for c in range(num_classes)]
        total = sum(counts)
        
        # 基础统计
        stats = {
            'num_classes': num_classes,
            'total_samples': total,
            'class_counts': dict(label_counts),
            'class_ratios': {c: count / total * 100 for c, count in label_counts.items()},
        }
        
        # 不平衡度量
        max_count = max(counts)
        min_count = min(c for c in counts if c > 0)  # 排除空类
        
        stats['max_count'] = max_count
        stats['min_count'] = min_count
        stats['imbalance_ratio'] = max_count / min_count if min_count > 0 else float('inf')
        
        # 最大/最小类
        stats['majority_class'] = max(label_counts.items(), key=lambda x: x[1])[0]
        stats['minority_class'] = min(label_counts.items(), key=lambda x: x[1])[0]
        
        # Gini 系数（衡量不平衡程度）
        ratios = np.array([c / total for c in counts if c > 0])
        stats['gini_coefficient'] = ClassDistributionAnalyzer._compute_gini(ratios)
        
        return stats
    
    @staticmethod
    def _compute_gini(values: np.ndarray) -> float:
        """计算 Gini 系数"""
        n = len(values)
        if n == 0:
            return 0.0
        
        sorted_values = np.sort(values)
        index = np.arange(1, n + 1)
        
        return (2 * np.sum(index * sorted_values) / (n * np.sum(sorted_values))) - (n + 1) / n
    
    @staticmethod
    def print_distribution(
        labels: List[int],
        num_classes: Optional[int] = None,
        class_names: Optional[Dict[int, str]] = None
    ):
        """打印类别分布表格"""
        stats = ClassDistributionAnalyzer.analyze(labels, num_classes)
        
        print("\n" + "=" * 60)
        print("类别分布分析")
        print("=" * 60)
        print(f"总样本数: {stats['total_samples']:,}")
        print(f"类别数: {stats['num_classes']}")
        print(f"不平衡比: {stats['imbalance_ratio']:.2f}:1")
        print(f"Gini 系数: {stats['gini_coefficient']:.4f}")
        print()
        
        print(f"{'类别':<10} {'名称':<15} {'样本数':>10} {'占比':>10}")
        print("-" * 50)
        
        for c in range(stats['num_classes']):
            count = stats['class_counts'].get(c, 0)
            ratio = count / stats['total_samples'] * 100
            name = class_names.get(c, '') if class_names else ''
            print(f"{c:<10} {name:<15} {count:>10,} {ratio:>9.2f}%")
        
        print()


def compute_class_weights(
    labels: List[int],
    num_classes: Optional[int] = None,
    method: str = 'inverse_sqrt',
    **kwargs
) -> np.ndarray:
    """
    便捷函数：计算类别权重
    
    Args:
        labels: 标签列表
        num_classes: 类别数
        method: 权重计算方法
        **kwargs: 其他参数
        
    Returns:
        类别权重数组
    """
    calculator = ClassWeightCalculator()
    return calculator.compute_weights(labels, num_classes, method, **kwargs)


def create_weighted_sampler(
    labels: List[int],
    num_samples: Optional[int] = None,
    replacement: bool = True
):
    """
    创建 PyTorch WeightedRandomSampler
    
    Args:
        labels: 标签列表
        num_samples: 每个 epoch 采样的样本数（默认为数据集大小）
        replacement: 是否有放回采样
        
    Returns:
        WeightedRandomSampler 实例
    """
    try:
        from torch.utils.data import WeightedRandomSampler
    except ImportError:
        raise ImportError("需要安装 PyTorch: pip install torch")
    
    calculator = ClassWeightCalculator()
    sample_weights = calculator.compute_sample_weights(labels)
    
    if num_samples is None:
        num_samples = len(labels)
    
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=num_samples,
        replacement=replacement
    )


class FocalLossWeights:
    """
    Focal Loss 权重计算
    
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    @staticmethod
    def compute_alpha(
        labels: List[int],
        num_classes: Optional[int] = None,
        method: str = 'inverse_sqrt'
    ) -> np.ndarray:
        """
        计算 Focal Loss 的 alpha 参数
        
        Args:
            labels: 标签列表
            num_classes: 类别数
            method: 权重计算方法
            
        Returns:
            alpha 数组，形状 [num_classes]
        """
        calculator = ClassWeightCalculator()
        weights = calculator.compute_weights(labels, num_classes, method, normalize=True)
        
        # 归一化到 [0, 1] 范围
        weights = weights / weights.max()
        
        return weights
    
    @staticmethod
    def get_recommended_gamma(imbalance_ratio: float) -> float:
        """
        根据不平衡比推荐 gamma 值
        
        Args:
            imbalance_ratio: 类别不平衡比
            
        Returns:
            推荐的 gamma 值
        """
        if imbalance_ratio < 10:
            return 1.0
        elif imbalance_ratio < 50:
            return 2.0
        else:
            return 3.0


class LabelSmoothingConfig:
    """Label Smoothing 配置"""
    
    @staticmethod
    def get_smoothing_value(
        num_classes: int,
        base_smoothing: float = 0.1
    ) -> float:
        """
        根据类别数获取推荐的 smoothing 值
        
        Args:
            num_classes: 类别数
            base_smoothing: 基础 smoothing 值
            
        Returns:
            推荐的 smoothing 值
        """
        # 类别越多，smoothing 值越小
        return base_smoothing * (1 - np.log10(num_classes) / 10)



