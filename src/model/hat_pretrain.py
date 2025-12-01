"""
HAT 预训练模型 - Masked Language Modeling (MLM)

用于在分类任务前对 HAT 模型进行无监督预训练。
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .hat_model import (
    HATConfig,
    HATEmbeddings,
    HATEncoder,
)


class MLMHead(nn.Module):
    """
    Masked Language Modeling 预测头
    
    将 hidden_states 映射到 vocab_size 的 logits。
    """
    
    def __init__(self, config: HATConfig):
        super().__init__()
        
        # 变换层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 输出投影到词表
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: [*, H] - 任意前导维度 + hidden_size
            
        Returns:
            prediction_scores: [*, vocab_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        prediction_scores = self.decoder(hidden_states)
        return prediction_scores


class HATInterleaved512ForMLM(nn.Module):
    """
    HAT Interleaved 512 预训练模型 (MLM)
    
    用于 Masked Language Modeling 任务的预训练。
    """
    
    def __init__(self, config: HATConfig):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.embeddings = HATEmbeddings(config)
        
        # 编码器
        self.encoder = HATEncoder(config)
        
        # MLM 头
        self.mlm_head = MLMHead(config)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 将 MLM head 的 decoder 权重与 word embeddings 共享
        self.mlm_head.decoder.weight = self.embeddings.word_embeddings.weight
    
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
        """创建 token_mask 和 segment_mask"""
        if attention_mask is None:
            return None, None
        
        # 为 CLS_SEG 位置添加 mask
        cls_mask = torch.ones(batch_size, num_segments, 1, dtype=attention_mask.dtype, device=device)
        token_mask = torch.cat([cls_mask, attention_mask], dim=2)
        
        # segment_mask
        segment_mask = attention_mask.any(dim=2).long()
        
        return token_mask, segment_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            input_ids: [B, N, K] - 输入 token IDs（包含被 mask 的位置）
            attention_mask: [B, N, K] - attention mask (可选)
            mlm_labels: [B, N, K+1] - MLM 标签 (可选)
                - 非 mask 位置为 -100（忽略）
                - mask 位置为原始 token ID
                
        Returns:
            - 如果 mlm_labels=None: prediction_scores [B, N, K+1, vocab_size]
            - 如果 mlm_labels 不为 None: (loss, prediction_scores)
        """
        batch_size, num_segments, segment_len = input_ids.shape
        device = input_ids.device
        
        # Step 1: 嵌入
        # [B, N, K] -> [B, N, K+1, H]
        hidden_states = self.embeddings(input_ids)
        
        # Step 2: 创建 masks
        seq_len = hidden_states.shape[2]
        token_mask, segment_mask = self._create_masks(
            attention_mask, batch_size, num_segments, seq_len, device
        )
        
        # Step 3: 编码器
        # [B, N, K+1, H]
        hidden_states, _ = self.encoder(
            hidden_states,
            token_mask=token_mask,
            segment_mask=segment_mask
        )
        
        # Step 4: MLM 预测
        # [B, N, K+1, H] -> [B, N, K+1, vocab_size]
        prediction_scores = self.mlm_head(hidden_states)
        
        # Step 5: 计算损失
        if mlm_labels is not None:
            loss_fn = nn.CrossEntropyLoss()  # 默认 ignore_index=-100
            # Flatten for loss computation
            # prediction_scores: [B, N, K+1, V] -> [B*N*(K+1), V]
            # mlm_labels: [B, N, K+1] -> [B*N*(K+1)]
            loss = loss_fn(
                prediction_scores.view(-1, self.config.vocab_size),
                mlm_labels.view(-1)
            )
            return loss, prediction_scores
        
        return prediction_scores
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """获取参数数量"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_mlm_model(config: Optional[HATConfig] = None) -> HATInterleaved512ForMLM:
    """
    工厂函数：创建 MLM 预训练模型
    
    Args:
        config: 模型配置（可选）
        
    Returns:
        HATInterleaved512ForMLM 模型实例
    """
    if config is None:
        config = HATConfig()
    return HATInterleaved512ForMLM(config)


# =============================================================================
# 自检代码
# =============================================================================

if __name__ == "__main__":
    from .hat_model import HATConfig, print_model_info
    
    print("=" * 60)
    print("HAT MLM Model Self-Test")
    print("=" * 60)
    
    # 创建配置和模型
    config = HATConfig()
    model = create_mlm_model(config)
    
    print(f"Total Parameters: {model.get_num_parameters():,}")
    
    # 构造测试输入
    batch_size = 2
    num_segments = 4
    segment_len = 512
    
    input_ids = torch.randint(5, config.vocab_size, (batch_size, num_segments, segment_len))
    attention_mask = torch.ones(batch_size, num_segments, segment_len, dtype=torch.long)
    
    # MLM labels: [B, N, K+1] (-100 表示不计算损失)
    mlm_labels = torch.full((batch_size, num_segments, segment_len + 1), -100, dtype=torch.long)
    # 随机选择一些位置作为 mask
    mask_positions = torch.rand(batch_size, num_segments, segment_len + 1) < 0.15
    mlm_labels[mask_positions] = torch.randint(5, config.vocab_size, mlm_labels.shape)[mask_positions]
    
    print(f"input_ids shape: {input_ids.shape}")
    print(f"mlm_labels shape: {mlm_labels.shape}")
    
    # 测试 forward
    model.eval()
    with torch.no_grad():
        loss, scores = model(input_ids, attention_mask=attention_mask, mlm_labels=mlm_labels)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Prediction scores shape: {scores.shape}")
    
    print("\n✓ MLM Model test passed!")

