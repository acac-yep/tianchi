# HAT-Interleaved 512 模型架构文档

## 概述

**HATInterleaved512ForClassification** 是一个层次化 Transformer 长文本分类模型，参考 Hierarchical Attention Transformer (HAT) 的 Interleaved (I1) 布局设计。

## 模型规格

| 参数 | 值 |
|------|-----|
| 词表大小 (vocab_size) | 7,555 |
| 隐藏层维度 (hidden_size) | 768 |
| 注意力头数 (num_attention_heads) | 12 |
| FFN 中间层维度 (intermediate_size) | 3,072 |
| HAT 层数 (num_hat_layers) | 6 |
| 每段最大长度 (segment_length) | 512 |
| 最大段数 (max_segments) | 8 |
| 最大序列长度 | 4,096 |
| 分类类别数 | 14 |
| **总参数量** | **~94.85M** |

## 模型结构

```
HATInterleaved512ForClassification
├── HATEmbeddings
│   ├── word_embeddings: [7555, 768]
│   ├── position_embeddings: [513, 768]  # 段内位置
│   ├── segment_embeddings: [8, 768]     # 段 ID
│   └── LayerNorm + Dropout
│
├── HATEncoder (6 × HATLayer)
│   └── HATLayer
│       ├── segment_encoder (SWE)        # 段内 self-attention
│       │   └── TransformerEncoderBlock
│       ├── document_encoder (CSE)       # 跨段 self-attention
│       │   └── TransformerEncoderBlock
│       ├── segment_position_embeddings  # 段级位置
│       └── global_projection            # 全局信息回注
│
└── Classifier
    ├── classifier_norm
    ├── classifier_dropout
    └── classifier: [768, 14]
```

## 核心组件

### 1. HATEmbeddings

将输入 `[B, N, K]` 转换为嵌入表示 `[B, N, K+1, H]`：

1. 为每个 segment 前插入 CLS_SEG token
2. 词嵌入 + 段内位置嵌入 + 段 ID 嵌入
3. LayerNorm + Dropout

### 2. TransformerEncoderBlock (Pre-Norm)

```
x -> LayerNorm -> MultiHeadAttention -> Dropout -> Residual
  -> LayerNorm -> FFN -> Dropout -> Residual
```

### 3. HATLayer (SWE + CSE + 回注)

每个 HATLayer 包含：

1. **SWE (Segment-Wise Encoder)**: 对每个 segment 内部做 self-attention
   - 输入: `[B, N, K+1, H]` → reshape → `[B*N, K+1, H]`
   - TransformerEncoderBlock
   - 输出: `[B, N, K+1, H]`

2. **CSE (Cross-Segment Encoder)**: 对 segment-level CLS tokens 做 self-attention
   - 提取 CLS tokens: `[B, N, H]`
   - 添加段级位置嵌入
   - TransformerEncoderBlock
   - 输出: `[B, N, H]`

3. **全局信息回注**: 将 CSE 输出回注到 token 级
   - `g = global_proj(cls_ctx)`: `[B, N, H]`
   - `h_next = h_seg + g.unsqueeze(2)`: 广播到所有 token

### 4. 文档级池化

对所有 segment 的 CLS token 做平均池化（考虑 segment_mask）：

```python
doc_repr = (cls_final * segment_mask).sum(dim=1) / segment_mask.sum(dim=1)
```

## 输入/输出

### 输入

```python
input_ids: torch.LongTensor      # [B, N, K] - token IDs (已偏移 +5)
attention_mask: torch.LongTensor  # [B, N, K] - 1=有效, 0=padding (可选)
labels: torch.LongTensor         # [B] - 分类标签 (可选)
```

### 输出

```python
# 如果 labels=None:
logits: torch.FloatTensor  # [B, num_labels]

# 如果 labels 不为 None:
(loss, logits)
```

## 使用示例

```python
from src.model import HATConfig, create_model

# 创建模型
config = HATConfig(
    vocab_size=7555,
    hidden_size=768,
    num_hat_layers=6,
    num_labels=14,
)
model = create_model(config)

# 前向传播
input_ids = torch.randint(5, 7555, (batch_size, 8, 512))
attention_mask = torch.ones(batch_size, 8, 512)
labels = torch.randint(0, 14, (batch_size,))

# 训练模式
loss, logits = model(input_ids, attention_mask, labels)

# 推理模式
logits = model(input_ids, attention_mask)
```

## 与数据预处理的配合

1. **Token ID 偏移**: 数据预处理时所有 token ID + 5，预留 0-4 给特殊 token
2. **分段**: 数据预处理完成 512 分段，模型不负责分段
3. **CLS_SEG 插入**: 模型在 forward 时自动插入 CLS_SEG (ID=2)

## 迁移到 Megatron-LM

模型设计便于迁移到 Megatron-LM 并行框架：

1. `TransformerEncoderBlock` 可替换为 Megatron 的 `ParallelTransformerLayer`
2. `HATEmbeddings` 可使用 Megatron 的 `ParallelEmbedding`
3. 分类头可使用 `ColumnParallelLinear`

## 预训练支持

提供 `HATInterleaved512ForMLM` 用于 MLM 预训练：

```python
from src.model import create_mlm_model

mlm_model = create_mlm_model()
loss, prediction_scores = mlm_model(input_ids, attention_mask, mlm_labels)
```

