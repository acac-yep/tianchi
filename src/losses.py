#!/usr/bin/env python3
"""
自定义损失函数模块

包含:
- FocalLoss: 针对类别不平衡和难样本的 Focal Loss
- LabelSmoothingCrossEntropy: 标签平滑交叉熵
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss - 专注于难样本的损失函数
    
    论文: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    公式: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    当样本被正确分类且置信度高时，(1 - p_t)^gamma 接近 0，
    从而降低了简单样本的损失权重，使模型更关注难样本。
    
    Args:
        gamma: 聚焦参数，gamma >= 0。gamma=0 退化为标准交叉熵。
               gamma 越大，对简单样本的惩罚越小。推荐值: 2.0
        alpha: 类别权重，形状为 [num_classes] 或标量。
               可以使用 class_weights 来平衡类别不均衡。
        reduction: 'none' | 'mean' | 'sum'
        label_smoothing: 标签平滑系数 (可选)
    
    Example:
        >>> criterion = FocalLoss(gamma=2.0, alpha=class_weights)
        >>> logits = model(input_ids, attention_mask)["logits"]  # [B, C]
        >>> loss = criterion(logits, labels)  # labels: [B]
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        # 注册 alpha 作为 buffer (不会被优化，但会随模型移动设备)
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算 Focal Loss
        
        Args:
            logits: 模型输出 logits，形状 [B, C]
            target: 真实标签，形状 [B]
            
        Returns:
            loss: 标量损失值
        """
        # 确保 alpha 在正确的设备上
        if self.alpha is not None and self.alpha.device != logits.device:
            self.alpha = self.alpha.to(logits.device)
        
        # 计算标准交叉熵 (不使用 reduction，保留每个样本的损失)
        # 使用 label_smoothing 如果设置了
        ce_loss = F.cross_entropy(
            logits, 
            target, 
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )  # [B]
        
        # 计算 p_t: 正确类别的预测概率
        # pt = exp(-ce_loss)，因为 ce_loss = -log(p_t)
        pt = torch.exp(-ce_loss)  # [B]
        
        # 计算 focal weight: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma  # [B]
        
        # 最终 focal loss
        focal_loss = focal_weight * ce_loss  # [B]
        
        # 应用 reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    带标签平滑的交叉熵损失
    
    标签平滑可以防止模型过于自信，提高泛化能力。
    
    Args:
        smoothing: 平滑系数，范围 [0, 1]。
                   smoothing=0 退化为标准交叉熵。
                   推荐值: 0.1
        weight: 类别权重
        reduction: 'none' | 'mean' | 'sum'
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
        if weight is not None:
            if isinstance(weight, (list, tuple)):
                weight = torch.tensor(weight, dtype=torch.float32)
            self.register_buffer('weight', weight)
        else:
            self.weight = None
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算带标签平滑的交叉熵
        
        Args:
            logits: 模型输出 logits，形状 [B, C]
            target: 真实标签，形状 [B]
            
        Returns:
            loss: 标量损失值
        """
        return F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction=self.reduction,
            label_smoothing=self.smoothing,
        )


class FocalLossWithLabelSmoothing(nn.Module):
    """
    结合 Focal Loss 和 Label Smoothing 的损失函数
    
    同时利用:
    - Focal Loss 的难样本聚焦能力
    - Label Smoothing 的正则化效果
    
    Args:
        gamma: Focal Loss 的聚焦参数
        alpha: 类别权重
        smoothing: 标签平滑系数
        reduction: 'none' | 'mean' | 'sum'
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.focal = FocalLoss(
            gamma=gamma,
            alpha=alpha,
            reduction=reduction,
            label_smoothing=smoothing,
        )
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.focal(logits, target)


def create_loss_fn(
    loss_type: str = "ce",
    class_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
    device: str = "cuda",
) -> nn.Module:
    """
    损失函数工厂函数
    
    Args:
        loss_type: 损失函数类型
            - "ce": CrossEntropyLoss (带类别权重)
            - "focal": FocalLoss
            - "focal_smooth": FocalLoss + LabelSmoothing
            - "smooth": LabelSmoothingCrossEntropy
        class_weights: 类别权重张量 [num_classes]
        gamma: Focal Loss 的聚焦参数
        label_smoothing: 标签平滑系数
        device: 设备
    
    Returns:
        nn.Module: 损失函数
    
    Example:
        >>> import numpy as np
        >>> class_weights = torch.tensor(np.load("class_weights.npy"))
        >>> criterion = create_loss_fn("focal", class_weights, gamma=2.0)
    """
    # 转换权重到正确的设备和类型
    if class_weights is not None:
        if not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        class_weights = class_weights.to(device)
    
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_type == "focal":
        return FocalLoss(
            gamma=gamma,
            alpha=class_weights,
            reduction="mean",
            label_smoothing=0.0,
        )
    
    elif loss_type == "focal_smooth":
        return FocalLossWithLabelSmoothing(
            gamma=gamma,
            alpha=class_weights,
            smoothing=label_smoothing,
            reduction="mean",
        )
    
    elif loss_type == "smooth":
        return LabelSmoothingCrossEntropy(
            smoothing=label_smoothing,
            weight=class_weights,
            reduction="mean",
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                        f"Supported: ce, focal, focal_smooth, smooth")


# 便捷函数: 直接从 npy 文件创建带权重的 Focal Loss
def create_focal_loss_from_weights(
    weights_path: str,
    gamma: float = 2.0,
    device: str = "cuda",
) -> FocalLoss:
    """
    从类别权重文件创建 Focal Loss
    
    Args:
        weights_path: class_weights.npy 文件路径
        gamma: 聚焦参数
        device: 设备
    
    Returns:
        FocalLoss 实例
    """
    import numpy as np
    
    class_weights = np.load(weights_path).astype('float32')
    class_weights = torch.tensor(class_weights, device=device)
    
    return FocalLoss(gamma=gamma, alpha=class_weights, reduction="mean")


if __name__ == "__main__":
    # 简单测试
    print("=== Focal Loss 测试 ===")
    
    # 模拟数据
    batch_size = 4
    num_classes = 14
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 模拟类别权重
    class_weights = torch.ones(num_classes)
    class_weights[0] = 2.0  # 假设类别 0 权重更高
    
    # 测试各种损失函数
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    focal_loss = FocalLoss(gamma=2.0, alpha=class_weights)
    
    print(f"CE Loss: {ce_loss(logits, labels).item():.4f}")
    print(f"Focal Loss (gamma=2.0): {focal_loss(logits, labels).item():.4f}")
    
    # 测试工厂函数
    for loss_type in ["ce", "focal", "focal_smooth", "smooth"]:
        criterion = create_loss_fn(
            loss_type=loss_type,
            class_weights=class_weights,
            gamma=2.0,
            label_smoothing=0.1,
            device="cpu",
        )
        loss_val = criterion(logits, labels)
        print(f"{loss_type}: {loss_val.item():.4f}")
    
    print("\n✓ 所有损失函数测试通过！")

