"""
HATInterleaved512ForClassification - 层次化 Transformer 长文本分类模型

实现 Hierarchical Attention Transformer (HAT) 的 Interleaved (I1) 布局，
用于 14 类单标签长文本分类任务。

模型结构：
- segment_len = 512
- max_seq_len = 4096 → 最多 8 个 segment
- 6 层 HATLayer，每层包含 SWE + CSE + 回注
- 等效于 BERT-base 规模 (hidden_size=768, 12 heads, ~100M 参数)

输入/输出约定：
- 输入 input_ids: [B, N, K] - 不含 CLS_SEG，由模型在 forward 时添加
- 输出: dict {"logits": logits} - logits 形状为 [B, num_labels]
- 损失计算由训练脚本负责（标准 HuggingFace transformers 模式）

Author: HAT Project
Date: 2024
"""

import math
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入公共配置
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.common_config import COMMON_CONFIG


# =============================================================================
# 1. 配置类
# =============================================================================

@dataclass
class HATConfig:
    """
    HAT 模型配置类
    
    关键参数从 COMMON_CONFIG 引用，确保与数据预处理一致。
    """
    # ========== 从 COMMON_CONFIG 引用的参数 ==========
    # 词表相关
    vocab_size: int = COMMON_CONFIG.vocab_size
    
    # 序列长度相关
    max_segments: int = COMMON_CONFIG.max_segments
    segment_length: int = COMMON_CONFIG.segment_length
    max_position_embeddings_segment: int = COMMON_CONFIG.max_position_embeddings_segment
    max_position_embeddings_segment_level: int = COMMON_CONFIG.max_position_embeddings_segment_level
    
    # 分类任务
    num_labels: int = COMMON_CONFIG.num_labels
    
    # 特殊 token ID
    pad_token_id: int = COMMON_CONFIG.pad_token_id
    cls_seg_token_id: int = COMMON_CONFIG.cls_token_id
    
    # ========== 模型特有参数 ==========
    # 模型尺寸
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072  # FFN 中间层维度 (4 * hidden_size)
    
    # 层数
    num_hat_layers: int = 6  # HATLayer 个数，每个 = SWE + CSE
    
    # 正则化
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # 初始化
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    
    def __post_init__(self):
        """验证配置有效性"""
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) 必须能被 num_attention_heads ({self.num_attention_heads}) 整除"


# =============================================================================
# 2. 嵌入模块
# =============================================================================

class HATEmbeddings(nn.Module):
    """
    HAT 嵌入层
    
    将 input_ids [B, N, K] 转换为初始隐藏表示 [B, N, K+1, H]。
    
    包含：
    - 词嵌入 (word embedding)
    - 段内位置嵌入 (position embedding within segment)
    - 段 ID 嵌入 (segment id embedding)
    - 为每个 segment 插入 CLS_SEG token
    """
    
    def __init__(self, config: HATConfig):
        super().__init__()
        self.config = config
        
        # 词嵌入: vocab_size -> hidden_size
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            padding_idx=config.pad_token_id
        )
        
        # 段内位置嵌入: max_position_embeddings_segment -> hidden_size
        # 位置 0 对应 CLS_SEG，1..K 对应真实 token
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings_segment,
            config.hidden_size
        )
        
        # 段 ID 嵌入: max_segments -> hidden_size
        # 区分不同 segment（第 0 段、第 1 段...）
        self.segment_embeddings = nn.Embedding(
            config.max_segments,
            config.hidden_size
        )
        
        # Layer Normalization
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # CLS_SEG token ID (常量)
        self.cls_seg_token_id = config.cls_seg_token_id
        
        # 注册位置 ID 缓冲区
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings_segment).unsqueeze(0)
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: [B, N, K] - B=batch_size, N=num_segments, K=segment_length
            
        Returns:
            hidden_states: [B, N, K+1, H] - 加入 CLS_SEG 后的嵌入表示
        """
        batch_size, num_segments, segment_len = input_ids.shape
        device = input_ids.device
        
        # Step 1: 为每个 segment 前面插入 CLS_SEG token
        # 创建 CLS_SEG token: [B, N, 1]
        cls_seg_tokens = torch.full(
            (batch_size, num_segments, 1),
            self.cls_seg_token_id,
            dtype=torch.long,
            device=device
        )
        
        # 拼接: [B, N, K] -> [B, N, K+1]
        input_ids_with_cls = torch.cat([cls_seg_tokens, input_ids], dim=2)
        seq_length = input_ids_with_cls.shape[2]  # K+1
        
        # Step 2: 词嵌入
        # [B, N, K+1] -> [B, N, K+1, H]
        word_embeds = self.word_embeddings(input_ids_with_cls)
        
        # Step 3: 段内位置嵌入
        # position_ids: [1, K+1]
        position_ids = self.position_ids[:, :seq_length]
        # [1, K+1] -> [1, K+1, H] -> broadcast to [B, N, K+1, H]
        position_embeds = self.position_embeddings(position_ids)
        
        # Step 4: 段 ID 嵌入
        # segment_ids: [N] -> [1, N, 1, H]
        segment_ids = torch.arange(num_segments, device=device)
        segment_embeds = self.segment_embeddings(segment_ids)  # [N, H]
        segment_embeds = segment_embeds.unsqueeze(0).unsqueeze(2)  # [1, N, 1, H]
        
        # Step 5: 合并所有嵌入
        # word_embeds: [B, N, K+1, H]
        # position_embeds: [1, K+1, H] -> 广播到 [B, N, K+1, H]
        # segment_embeds: [1, N, 1, H] -> 广播到 [B, N, K+1, H]
        embeddings = word_embeds + position_embeds + segment_embeds
        
        # Step 6: LayerNorm + Dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


# =============================================================================
# 3. 通用 Transformer Block
# =============================================================================

class TransformerEncoderBlock(nn.Module):
    """
    标准 Transformer Encoder Block (Pre-Norm 结构)
    
    结构：
        x -> LayerNorm -> MultiHeadAttention -> Dropout -> Residual
          -> LayerNorm -> FFN -> Dropout -> Residual
    
    供 SWE (Segment-Wise Encoder) 和 CSE (Cross-Segment Encoder) 复用。
    """
    
    def __init__(self, config: HATConfig):
        super().__init__()
        self.config = config
        
        # Pre-Norm: LayerNorm 在 attention 之前
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Multi-Head Self-Attention
        # 注意: PyTorch 的 MultiheadAttention 的 dropout 是 attention weights 上的 dropout
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True  # 输入形状为 [B, L, H]
        )
        
        # Attention 输出的 dropout
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Pre-Norm: LayerNorm 在 FFN 之前
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-Forward Network (FFN)
        # H -> 4H -> H
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: [B, L, H] - 输入隐藏状态
            attention_mask: [B, L] - attention mask
                - 1 表示有效 token（参与 attention）
                - 0 表示 padding token（被 mask 掉）
                - 会被转换为 key_padding_mask 传给 MultiheadAttention
                
        Returns:
            hidden_states: [B, L, H] - 输出隐藏状态
        """
        # ========== Self-Attention Block (Pre-Norm) ==========
        # 1. LayerNorm
        normed = self.attention_norm(hidden_states)
        
        # 2. Self-Attention
        # 处理 attention_mask: PyTorch MultiheadAttention 使用 key_padding_mask
        # key_padding_mask: [B, L], True 表示被 mask 的位置
        key_padding_mask = None
        if attention_mask is not None:
            # 原始 mask: 1=有效, 0=padding
            # key_padding_mask: True=padding (需要 mask), False=有效
            key_padding_mask = (attention_mask == 0)
        
        attn_output, _ = self.self_attention(
            query=normed,
            key=normed,
            value=normed,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # 3. Dropout + Residual
        hidden_states = hidden_states + self.attention_dropout(attn_output)
        
        # ========== FFN Block (Pre-Norm) ==========
        # 4. LayerNorm
        normed = self.ffn_norm(hidden_states)
        
        # 5. FFN + Residual
        hidden_states = hidden_states + self.ffn(normed)
        
        return hidden_states


# =============================================================================
# 4. HAT Layer (SWE + CSE + 回注)
# =============================================================================

class HATLayer(nn.Module):
    """
    HAT 层：一个 "层对"
    
    包含：
    - Segment-Wise Encoder (SWE): 段内 self-attention
    - Cross-Segment Encoder (CSE): 跨段 self-attention
    - 全局信息回注: 将 CSE 输出回注到 token 级表示
    
    这是 HAT Interleaved (I1) 布局的核心模块。
    """
    
    def __init__(self, config: HATConfig):
        super().__init__()
        self.config = config
        
        # Segment-Wise Encoder (SWE)
        # 对每个 segment 内部做 self-attention
        self.segment_encoder = TransformerEncoderBlock(config)
        
        # Cross-Segment Encoder (CSE)
        # 对 segment-level CLS tokens 做 self-attention
        self.document_encoder = TransformerEncoderBlock(config)
        
        # 段级位置嵌入 (用于 CSE 前)
        self.segment_position_embeddings = nn.Embedding(
            config.max_position_embeddings_segment_level,
            config.hidden_size
        )
        
        # 全局信息投影层 (用于回注)
        self.global_projection = nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        segment_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            hidden_states: [B, N, K+1, H]
                - B = batch_size
                - N = num_segments  
                - K+1 = segment_length + CLS_SEG
                - H = hidden_size
            token_mask: [B, N, K+1] - token 级 mask (可选)
                - 1 表示有效 token, 0 表示 padding
            segment_mask: [B, N] - segment 级 mask (可选)
                - 1 表示有效 segment, 0 表示 padding segment
                
        Returns:
            hidden_states: [B, N, K+1, H] - 更新后的 token 级表示
            cls_context: [B, N, H] - 更新后的 segment 级表示 (CLS tokens)
        """
        batch_size, num_segments, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # ========== Step 1: Segment-Wise Encoder (SWE) ==========
        # 对每个 segment 独立做 self-attention
        
        # Reshape: [B, N, K+1, H] -> [B*N, K+1, H]
        hidden_states_flat = hidden_states.view(batch_size * num_segments, seq_len, hidden_size)
        
        # 准备 token mask: [B, N, K+1] -> [B*N, K+1]
        token_mask_flat = None
        if token_mask is not None:
            token_mask_flat = token_mask.view(batch_size * num_segments, seq_len)
        
        # SWE forward
        hidden_states_flat = self.segment_encoder(hidden_states_flat, attention_mask=token_mask_flat)
        
        # Reshape back: [B*N, K+1, H] -> [B, N, K+1, H]
        hidden_states = hidden_states_flat.view(batch_size, num_segments, seq_len, hidden_size)
        
        # ========== Step 2: 提取 segment CLS tokens ==========
        # CLS_SEG 位于每个 segment 的第 0 位
        # cls_tokens: [B, N, H]
        cls_tokens = hidden_states[:, :, 0, :]
        
        # ========== Step 3: Cross-Segment Encoder (CSE) ==========
        # 对 segment-level CLS tokens 做 self-attention
        
        # 添加段级位置嵌入
        # position_ids: [N]
        position_ids = torch.arange(num_segments, device=device)
        # segment_position_embeds: [N, H] -> [1, N, H]
        segment_position_embeds = self.segment_position_embeddings(position_ids).unsqueeze(0)
        
        # cls_input: [B, N, H]
        cls_input = cls_tokens + segment_position_embeds
        
        # CSE forward
        # 使用 segment_mask 作为 attention mask
        cls_context = self.document_encoder(cls_input, attention_mask=segment_mask)
        
        # ========== Step 4: 全局信息回注到 token 级 ==========
        # 将 CSE 输出投影后加回到该 segment 内所有 token
        
        # global_info: [B, N, H]
        global_info = self.global_projection(cls_context)
        
        # 扩展维度以便广播: [B, N, H] -> [B, N, 1, H]
        global_info = global_info.unsqueeze(2)
        
        # 回注: [B, N, K+1, H] + [B, N, 1, H] -> [B, N, K+1, H]
        hidden_states = hidden_states + global_info
        
        return hidden_states, cls_context


# =============================================================================
# 5. HAT Encoder
# =============================================================================

class HATEncoder(nn.Module):
    """
    HAT Encoder
    
    包含多个 HATLayer 的堆叠，实现 Interleaved (I1) 布局。
    """
    
    def __init__(self, config: HATConfig):
        super().__init__()
        self.config = config
        
        # num_hat_layers 个 HATLayer
        self.layers = nn.ModuleList([
            HATLayer(config) for _ in range(config.num_hat_layers)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        segment_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            hidden_states: [B, N, K+1, H] - 输入嵌入
            token_mask: [B, N, K+1] - token 级 mask (可选)
            segment_mask: [B, N] - segment 级 mask (可选)
            
        Returns:
            hidden_states: [B, N, K+1, H] - 最终 token 级表示
            cls_final: [B, N, H] - 最终 segment 级表示
        """
        cls_final = None
        
        # 逐层前向传播
        for layer in self.layers:
            hidden_states, cls_final = layer(
                hidden_states,
                token_mask=token_mask,
                segment_mask=segment_mask
            )
        
        return hidden_states, cls_final


# =============================================================================
# 6. 分类模型
# =============================================================================

class HATInterleaved512ForClassification(nn.Module):
    """
    HAT Interleaved 512 长文本分类模型
    
    用于 14 类单标签分类任务。
    
    模型结构：
    - Embeddings: 词嵌入 + 位置嵌入 + 段 ID 嵌入
    - Encoder: 6 层 HATLayer (SWE + CSE + 回注)
    - Classifier: 文档级池化 + 线性分类头
    """
    
    def __init__(self, config: HATConfig):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.embeddings = HATEmbeddings(config)
        
        # 编码器
        self.encoder = HATEncoder(config)
        
        # 分类头
        self.classifier_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.classifier_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _create_masks(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        num_segments: int,
        seq_len: int,
        device: torch.device
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        根据输入的 attention_mask 创建 token_mask 和 segment_mask
        
        Args:
            attention_mask: [B, N, K] - 原始 attention mask (不含 CLS_SEG)
            batch_size: B
            num_segments: N
            seq_len: K+1 (含 CLS_SEG)
            device: 设备
            
        Returns:
            token_mask: [B, N, K+1] - 含 CLS_SEG 的 token mask
            segment_mask: [B, N] - segment 级 mask
        """
        if attention_mask is None:
            return None, None
        
        # 为 CLS_SEG 位置添加 mask (CLS_SEG 始终有效)
        # cls_mask: [B, N, 1]
        cls_mask = torch.ones(batch_size, num_segments, 1, dtype=attention_mask.dtype, device=device)
        
        # token_mask: [B, N, K+1]
        token_mask = torch.cat([cls_mask, attention_mask], dim=2)
        
        # segment_mask: 如果一个 segment 的所有 token 都是 padding，则该 segment 无效
        # [B, N, K] -> [B, N] (any token is valid)
        segment_mask = attention_mask.any(dim=2).long()
        
        return token_mask, segment_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None  # 保留参数以兼容，但不使用
    ) -> dict:
        """
        前向传播
        
        Args:
            input_ids: [B, N, K] - 输入 token IDs
                - B = batch_size
                - N = num_segments (≤ 8)
                - K = segment_length (512)
            attention_mask: [B, N, K] - attention mask (可选)
                - 1 表示有效 token, 0 表示 padding
            labels: [B] - 分类标签 (可选，保留以兼容，但不使用)
                - 损失计算由训练脚本负责
                
        Returns:
            dict: {"logits": logits} - logits 形状为 [B, num_labels]
        """
        batch_size, num_segments, segment_len = input_ids.shape
        device = input_ids.device
        
        # Step 1: 嵌入
        # input_ids: [B, N, K] -> hidden_states: [B, N, K+1, H]
        hidden_states = self.embeddings(input_ids)
        
        # Step 2: 创建 masks
        seq_len = hidden_states.shape[2]  # K+1
        token_mask, segment_mask = self._create_masks(
            attention_mask, batch_size, num_segments, seq_len, device
        )
        
        # Step 3: 编码器
        # hidden_states: [B, N, K+1, H], cls_final: [B, N, H]
        hidden_states, cls_final = self.encoder(
            hidden_states,
            token_mask=token_mask,
            segment_mask=segment_mask
        )
        
        # Step 4: 文档级池化
        # 对所有 segment 的 CLS 做平均池化
        if segment_mask is not None:
            # 使用 mask 加权平均，避免 padding segment 的影响
            # segment_mask: [B, N] -> [B, N, 1]
            mask_expanded = segment_mask.unsqueeze(-1).float()
            # 加权求和
            doc_repr = (cls_final * mask_expanded).sum(dim=1)  # [B, H]
            # 除以有效 segment 数
            doc_repr = doc_repr / mask_expanded.sum(dim=1).clamp(min=1)  # [B, H]
        else:
            # 简单平均
            doc_repr = cls_final.mean(dim=1)  # [B, H]
        
        # Step 5: 分类
        # LayerNorm + Dropout + Linear
        doc_repr = self.classifier_norm(doc_repr)
        doc_repr = self.classifier_dropout(doc_repr)
        logits = self.classifier(doc_repr)  # [B, num_labels]
        
        # 返回字典格式（HuggingFace transformers 标准格式）
        return {"logits": logits}
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """获取参数数量"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# 7. 辅助函数
# =============================================================================

def create_model(config: Optional[HATConfig] = None) -> HATInterleaved512ForClassification:
    """
    工厂函数：创建 HAT 模型
    
    Args:
        config: 模型配置（可选，使用默认配置）
        
    Returns:
        HATInterleaved512ForClassification 模型实例
    """
    if config is None:
        config = HATConfig()
    return HATInterleaved512ForClassification(config)


def print_model_info(model: HATInterleaved512ForClassification):
    """打印模型信息"""
    config = model.config
    
    print("=" * 60)
    print("HAT Interleaved 512 Model Info")
    print("=" * 60)
    print(f"Vocab Size: {config.vocab_size}")
    print(f"Hidden Size: {config.hidden_size}")
    print(f"Num Attention Heads: {config.num_attention_heads}")
    print(f"Intermediate Size: {config.intermediate_size}")
    print(f"Num HAT Layers: {config.num_hat_layers}")
    print(f"Max Segments: {config.max_segments}")
    print(f"Segment Length: {config.segment_length}")
    print(f"Num Labels: {config.num_labels}")
    print("-" * 60)
    print(f"Total Parameters: {model.get_num_parameters(trainable_only=False):,}")
    print(f"Trainable Parameters: {model.get_num_parameters(trainable_only=True):,}")
    print("=" * 60)


# =============================================================================
# 8. 自检代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HAT Model Self-Test")
    print("=" * 60)
    
    # 创建配置和模型
    config = HATConfig()
    model = create_model(config)
    
    # 打印模型信息
    print_model_info(model)
    
    # 构造测试输入
    batch_size = 2
    num_segments = 4  # 使用 4 个 segment 测试
    segment_len = 512
    
    print(f"\nTest Input Shape: [B={batch_size}, N={num_segments}, K={segment_len}]")
    
    # 随机生成 input_ids (范围: 5 ~ vocab_size-1，因为 0-4 是特殊 token)
    input_ids = torch.randint(
        5, config.vocab_size,
        (batch_size, num_segments, segment_len),
        dtype=torch.long
    )
    
    # 创建 attention_mask (模拟部分 padding)
    attention_mask = torch.ones(batch_size, num_segments, segment_len, dtype=torch.long)
    # 让最后一个 segment 的后半部分是 padding
    attention_mask[:, -1, segment_len//2:] = 0
    
    # 创建标签
    labels = torch.randint(0, config.num_labels, (batch_size,), dtype=torch.long)
    
    print(f"input_ids shape: {input_ids.shape}")
    print(f"attention_mask shape: {attention_mask.shape}")
    print(f"labels shape: {labels.shape}")
    
    # 测试 forward（模型返回字典格式）
    print("\n--- Test Forward (returns dict) ---")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, config.num_labels), "Logits shape mismatch!"
    print("✓ Forward pass: OK")
    
    # 测试损失计算（由外部负责）
    print("\n--- Test Loss Computation (external) ---")
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    print(f"Loss: {loss.item():.4f}")
    print("✓ Loss computation: OK")
    
    # 测试反向传播（使用外部计算的 loss）
    print("\n--- Test Backward ---")
    model.train()
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs["logits"]
    loss = loss_fn(logits, labels)
    loss.backward()
    print("✓ Backward pass: OK")
    
    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    if has_grad:
        print("✓ Gradients computed: OK")
    else:
        print("✗ Warning: No gradients found!")
    
    # 测试不同 segment 数量
    print("\n--- Test Different Segment Numbers ---")
    for n_seg in [1, 4, 8]:
        test_input = torch.randint(5, config.vocab_size, (1, n_seg, segment_len))
        with torch.no_grad():
            outputs = model(test_input)
            logits = outputs["logits"]
        print(f"  num_segments={n_seg}: logits shape = {logits.shape} ✓")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

