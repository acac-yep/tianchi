# HAT-Interleaved 512 技术规格文档

## 概述

本文档描述 HAT-Interleaved 长文本分类系统的完整技术规格，包括数据预处理和模型架构。

## 1. 公共配置

### 1.1 配置来源

所有关键参数统一定义在 `src/common_config.py`，数据预处理和模型都从此引用：

```python
from src.common_config import COMMON_CONFIG

# 共享参数
vocab_size = COMMON_CONFIG.vocab_size           # 7555
segment_length = COMMON_CONFIG.segment_length   # 512
max_segments = COMMON_CONFIG.max_segments       # 8
num_labels = COMMON_CONFIG.num_labels           # 14
```

### 1.2 关键参数表

| 参数 | 值 | 说明 |
|------|-----|------|
| `vocab_size` | 7,555 | 原始 7550 + 5 特殊 token |
| `id_offset` | 5 | Token ID 偏移量 |
| `segment_length` | 512 | 每段 token 数（不含 CLS_SEG） |
| `max_segments` | 8 | 最大段数 |
| `max_seq_length` | 4,096 | 原始 token 级最大长度（不含 CLS） |
| `max_model_tokens` | 4,104 | 模型内部总 token 数（含每段 CLS_SEG） |
| `num_labels` | 14 | 分类类别数 |

**长度关系：**
- `max_seq_length = segment_length × max_segments = 512 × 8 = 4096`
- `max_model_tokens = max_segments × (segment_length + 1) = 8 × 513 = 4104`

### 1.3 特殊 Token ID

| Token | ID | 说明 |
|-------|-----|------|
| `[PAD]` | 0 | 填充 |
| `[UNK]` | 1 | 未知 |
| `[CLS]` | 2 | 段级 CLS (CLS_SEG)，文档表示通过所有段 CLS 的 masked 平均得到 |
| `[SEP]` | 3 | 分隔符 |
| `[MASK]` | 4 | MLM 掩码 |

> **注意**：当前方案不使用单独的 doc-level CLS token。文档级表示由所有 segment 的 CLS_SEG 向量经 masked 平均得到。

---

## 2. 数据预处理

### 2.1 处理流程

```
原始文本 (空格分隔的 token ID)
    ↓
1. Token ID 重映射 (+5)
    ↓
2. 数据清洗（去重、冲突处理）
    ↓
3. 文档分段（512 tokens/segment）
    ↓
4. 输出 [B, N, K] 格式（不含 CLS_SEG）
```

### 2.2 Token ID 重映射

原始 token ID 整体偏移 +5，为特殊 token 预留空间：

- 原始范围: `0 ~ 7549`
- 映射后范围: `5 ~ 7554`
- 预留 `0 ~ 4` 给特殊 token

```python
from src.data_preprocess import create_tokenizer

tokenizer = create_tokenizer()
encoded = tokenizer.encode("100 200 300")  # [105, 205, 305]
```

### 2.3 文档分段

**职责边界（重要）：**
- ✅ 分段器负责：切分 512 tokens/segment、尾段回拉、超长截断、padding
- ❌ 分段器不负责：添加任何特殊 token
- ✅ **CLS_SEG 由模型在 forward 时统一添加**

**分段策略：**

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `segment_length` | 512 | 每段长度 |
| `max_segments` | 8 | 最多 8 段 |
| `tail_pullback_threshold` | 0.5 | 尾段 < 256 时回拉 |
| `train_long_strategy` | `random_window` | 训练时随机选 8 段（数据增强） |
| `infer_long_strategy` | `sliding_window` | 推理时滑动窗口 |
| `sliding_window_stride` | 4 | 滑动步长 |

**超长文档处理：**
- **训练时**：推荐 `random_window`（随机选连续 8 段作为数据增强）
- **推理时**：使用滑动窗口，外部代码对多窗口 logits 聚合（求和/平均）

### 2.4 数据清洗

**处理顺序**：先清洗 → 再划分 train/val

| 问题类型 | 处理方式 |
|---------|---------|
| 完全重复样本 | 保留第一个 |
| 标签冲突（同文本不同标签） | 全部移除 |
| 空文本 | 移除 |
| 超短文本 (< 5 tokens) | 标记（可选移除） |

**空文档处理说明**：
> 空文本会在数据清洗阶段移除。Segmenter 中的空文档处理逻辑（`_create_empty_document()`）仅为防御性代码，一般不会触发。如果触发，返回一个全 PAD 的 segment，`segment_mask = [0]`，基本不会对训练产生影响。

### 2.5 类别平衡

**警告**：默认只使用一种平衡手段，避免双重补偿。

| 方式 | 配置 | 说明 |
|------|------|------|
| 损失权重 | `weight_method='inverse_sqrt'` | **推荐**，在 loss 函数中加权 |
| 采样器 | `use_weighted_sampler=False` | 默认关闭 |

**权重计算方法：**
- `inverse_sqrt`: $w_c = 1/\sqrt{n_c}$（推荐）
- `inverse_log`: $w_c = 1/\log(1+n_c)$
- `effective_num`: Effective Number of Samples

---

## 3. 模型架构

### 3.1 模型规格

| 参数 | 值 |
|------|-----|
| 词表大小 | 7,555 |
| 隐藏层维度 | 768 |
| 注意力头数 | 12 |
| FFN 中间层 | 3,072 |
| HAT 层数 | 6 |
| 段内位置嵌入 | 513 (含 CLS_SEG) |
| 段级位置嵌入 | 8 |
| **总参数量** | **~94.85M** |

### 3.2 模型结构

```
HATInterleaved512ForClassification
├── HATEmbeddings
│   ├── CLS_SEG 插入（模型负责）
│   ├── word_embeddings: [7555, 768]
│   ├── position_embeddings: [513, 768]
│   └── segment_embeddings: [8, 768]
│
├── HATEncoder (6 × HATLayer)
│   └── HATLayer
│       ├── SWE: 段内 self-attention
│       ├── CSE: 跨段 self-attention
│       └── 全局信息回注
│
└── Classifier
    ├── LayerNorm
    ├── Dropout
    └── Linear(768, 14)
```

### 3.3 输入/输出形状

**输入：**
```python
input_ids: [B, N, K]        # 不含 CLS_SEG
attention_mask: [B, N, K]   # 可选
labels: [B]                 # 可选
```

**模型内部形状演变：**

| 阶段 | 形状 | 说明 |
|------|------|------|
| 输入 | `[B, N, K]` | K=512，不含 CLS_SEG |
| Embedding 后 | `[B, N, K+1, H]` | 插入 CLS_SEG，K+1=513 |
| SWE | `[B*N, K+1, H]` | 段内 attention |
| CSE | `[B, N, H]` | 跨段 attention |
| 池化后 | `[B, H]` | 文档表示 |
| 输出 | `[B, 14]` | logits |

### 3.4 Mask 实现约定

**Mask 形状演变：**

| 阶段 | token_mask | segment_mask |
|------|------------|--------------|
| 输入 | `[B, N, K]` | - |
| Embedding 后 | `[B, N, K+1]` | `[B, N]` |
| SWE | `[B*N, K+1]` | - |
| CSE | - | `[B, N]` |

**Mask 实现方式：**

- **SWE（段内 self-attention）**：
  - 输入 `token_mask: [B, N, K+1]`，1=有效，0=padding。
  - 在内部 reshape 为 `[B * N, K+1]`，作为 `key_padding_mask` 使用。
  
- **CSE（跨段 self-attention）**：
  - 输入 `segment_mask: [B, N]`，1=有效，0=padding。
  - 在内部直接作为 `key_padding_mask: [B, N]` 使用。

> 本设计不使用复杂的 `[B, 1, 1, L]` 形状 `attn_mask`，而是通过 `key_padding_mask` 控制 padding 位置，与 PyTorch / Megatron 的标准实现保持一致。

**segment_mask 生成方式：**
> HATDataCollator 生成的 `segment_mask: [B, N]` 中，前 `num_segments` 位置为 1（有效段），后续位置为 0（padding 段），在跨段 self-attention 和文档级池化中共同使用。

### 3.5 使用示例

```python
from src.model import HATConfig, create_model

# 创建模型
model = create_model()

# 输入格式: [B, N, K]（不含 CLS_SEG，模型会自动添加）
input_ids = torch.randint(5, 7555, (batch_size, 8, 512))
attention_mask = torch.ones(batch_size, 8, 512)
labels = torch.randint(0, 14, (batch_size,))

# 训练
loss, logits = model(input_ids, attention_mask, labels)

# 推理
logits = model(input_ids, attention_mask)
```

---

## 4. MLM 预训练

### 4.1 MLMDataCollator 说明

**Mask 策略：**
- 15% token 被选中处理
- 其中 80% 替换为 `[MASK]` (ID=4)
- 其中 10% 替换为随机 token（范围 `[5, vocab_size-1]`，不含特殊 token）
- 其中 10% 保持不变

**不 mask 的位置：**
- PAD、UNK、SEP、MASK token (ID 0, 1, 3, 4)
- Padding 位置
- **注意**：不包含 CLS，因为数据预处理阶段不会出现 CLS token

### 4.2 MLM 模型接口

**接口设计原则：**
- 外部（预处理 & collator）只知道**原始 512 tokens**，不感知 CLS
- 模型内部负责插 CLS，并对齐 labels 的维度

```python
from src.model import create_mlm_model

mlm_model = create_mlm_model()

# 外部 / collator 提供
input_ids: [B, N, K]        # 包含 [MASK] 的输入，不含 CLS
attention_mask: [B, N, K]
mlm_labels: [B, N, K]       # 非 mask 位置为 -100

# 模型内部处理：
# - 对 mlm_labels 前面 pad 一列 -100，扩展为 [B, N, K+1]
# - 与 prediction_scores: [B, N, K+1, vocab_size] 对齐

# 输出
loss, prediction_scores = mlm_model(input_ids, attention_mask, mlm_labels)
# prediction_scores: [B, N, K+1, vocab_size]
```

---

## 5. 完整数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据预处理阶段                            │
├─────────────────────────────────────────────────────────────────┤
│  原始文本 "100 200 300 ..."                                      │
│      ↓ HATTokenizer.encode()                                    │
│  Token IDs [105, 205, 305, ...]  (ID + 5)                       │
│      ↓ DocumentSegmenter.segment_document()                     │
│  Segments [N, K]  (不含 CLS_SEG)                                 │
│      ↓ HATDataCollator()                                        │
│  Batch [B, N, K]                                                │
│  segment_mask [B, N] (前 num_segments 为 1，其余为 0)            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                          模型阶段                                │
├─────────────────────────────────────────────────────────────────┤
│  input_ids [B, N, K]                                            │
│      ↓ HATEmbeddings (插入 CLS_SEG)                             │
│  hidden_states [B, N, K+1, H]                                   │
│  token_mask [B, N, K+1] (CLS 位置始终为 1)                       │
│      ↓ HATEncoder (6 × HATLayer)                                │
│  cls_final [B, N, H]                                            │
│      ↓ 池化 (masked mean over segments)                         │
│  doc_repr [B, H]                                                │
│      ↓ Classifier                                               │
│  logits [B, 14]                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 配置文件关系

```
src/common_config.py          ← 公共参数（单一真相来源）
    ↑                ↑
    │                │
src/data_preprocess/config.py    src/model/hat_model.py
(DataConfig)                     (HATConfig)
```

**原则**：关键参数只在 `common_config.py` 定义一次，其他配置通过引用获取。

---

## 7. 注意事项

1. **CLS_SEG 添加位置**：只由模型负责，分段器不添加
2. **配置一致性**：修改 segment_length 等参数时，只需修改 common_config.py
3. **类别平衡**：避免同时使用 sampler 和 loss 权重，防止双重补偿
4. **超长文档**：训练用 random_window，推理用 sliding_window + logits 聚合
5. **数据清洗顺序**：先清洗再划分 train/val
6. **MLM labels 形状**：外部提供 `[B, N, K]`，模型内部 pad 为 `[B, N, K+1]`
7. **空文档**：在数据清洗阶段移除，segmenter 的空文档处理仅为防御性代码
