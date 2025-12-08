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

**预处理阶段（`scripts/run_preprocessing.py`）：**

```
原始文本 (空格分隔的 token ID)
    ↓
1. Token ID 重映射 (+5)
    ↓
2. 数据清洗（去重、冲突处理）
    ↓
3. 划分 train/val（stratified）
    ↓
4. 计算类别权重
    ↓
5. 保存处理后的文本（仍为原始长度，未分段）
```

**训练阶段（`HATDataset` 动态处理）：**

```
预处理后的文本（空格分隔的 token ID）
    ↓
1. Tokenize（解析 token ID 字符串）
    ↓
2. 文档分段（512 tokens/segment，最多 8 段）
    ↓
3. 输出 [N, K] 格式（不含 CLS_SEG）
    ↓
4. HATDataCollator 整理为 [B, N, K] batch
```

> **重要说明**：文档分段在训练时由 `HATDataset.__getitem__` 动态完成，而非预处理阶段。这样设计的好处是：
> - 预处理速度快，数据文件小
> - 支持不同的分段策略（训练时 random_window，推理时 sliding_window）
> - 灵活的数据增强（每次访问可能得到不同的分段结果）

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

**分段时机：**
- ⚠️ **不在预处理阶段**：预处理只保存重映射后的完整文本（原始长度）
- ✅ **在训练时动态完成**：由 `HATDataset.__getitem__` 调用 `DocumentSegmenter.segment_document()` 完成

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

**实现细节：**
```python
# HATDataset.__getitem__ 中的处理流程
def _process_text(self, idx: int) -> SegmentedDocument:
    text = self.texts[idx]  # 空格分隔的 token ID 字符串
    token_ids = self.tokenizer.encode(text)  # 解析为 token ID 列表
    segmented = self.segmenter.segment_document(
        token_ids, 
        mode='train' if self.mode != 'pretrain' else 'train'
    )
    return segmented  # 返回 [N, K] 格式的 SegmentedDocument
```

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
│                    预处理阶段 (run_preprocessing.py)              │
├─────────────────────────────────────────────────────────────────┤
│  原始文本 "100 200 300 ..."                                      │
│      ↓ Token ID 重映射 (+5)                                     │
│  重映射文本 "105 205 305 ..."                                    │
│      ↓ 数据清洗、划分 train/val                                  │
│  保存到 train.csv / val.csv (完整文本，未分段)                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  训练阶段 (HATDataset 动态处理)                   │
├─────────────────────────────────────────────────────────────────┤
│  文本 "105 205 305 ..."                                          │
│      ↓ HATTokenizer.encode()                                    │
│  Token IDs [105, 205, 305, ...]                                 │
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

1. **分段处理时机**：文档分段在训练时由 `HATDataset` 动态完成，不在预处理阶段
2. **CLS_SEG 添加位置**：只由模型负责，分段器不添加
3. **配置一致性**：修改 segment_length 等参数时，只需修改 common_config.py
4. **类别平衡**：避免同时使用 sampler 和 loss 权重，防止双重补偿
5. **超长文档**：训练用 random_window，推理用 sliding_window + logits 聚合
6. **数据清洗顺序**：先清洗再划分 train/val
7. **MLM labels 形状**：外部提供 `[B, N, K]`，模型内部 pad 为 `[B, N, K+1]`
8. **空文档**：在数据清洗阶段移除，segmenter 的空文档处理仅为防御性代码
9. **预处理输出**：预处理后的 CSV 文件保存的是完整文本（重映射后），长度可能超过 4096 tokens，分段在训练时处理

---

## 8. 训练与调优

### 8.1 MLM 预训练（Stage0，可选但推荐）
- 脚本：`scripts/mlm_train.py`，数据源为预处理后的 `train.csv`（默认 9:1 划分训练/验证）。
- 关键超参：`batch_size=4`、`lr=5e-5`、`warmup_ratio=0.06`（或 `warmup_steps`）、`max_steps=10000`、`weight_decay=0.01`、`grad_clip=1.0`、`mlm_probability=0.15`，默认启用 `EMA=0.999`，可选 `AMP`。
- 形状约定：外部 batch `[B,N,K]`（不含 CLS），模型内部补 `CLS_SEG` 得到 `[B,N,K+1]` 并自动对齐 `mlm_labels`。
- 产物：`hat_mlm_final.pt`、`best_model.pt`（基于验证 loss / perplexity），兼容后续分类加载。
- 报告指标：验证 PPL 从 5064 → 11.08（Step 12k），全程无反弹。

### 8.2 分类训练 Stage1（主力方案）
- 脚本：`scripts/cls_train_kfold.py`，默认 Stratified K-Fold=5，种子 42。
- 模型：`HATInterleaved512ForClassification`，可加载 MLM 权重；输入保持 `[B,N,K]`，模型内部补 `CLS_SEG`。
- 损失：`ce` / `smooth`（默认 label_smoothing=0.05）/ `focal` / `focal_smooth`，类别权重来自 `class_weights.npy`；可选 `WeightedRandomSampler`，避免与损失权重叠加。
- 优化：`AdamW(lr=3e-5, weight_decay=0.01, betas=(0.9,0.999), eps=1e-8)`，`warmup_ratio=0.06`，`grad_clip=1.0`；可选 `AMP`、`EMA(0.9999)`、`early_stopping`。
- 批次：`batch_size=64`（H800 建议 64~128），`eval_batch_size=128`，`num_epochs=5`。
- 产物：`hat_cls_fold{k}_best.pt`（每折以宏 F1 选优）。报告实测均值 `0.9602 ± 0.0019`，共约 946 分钟。

### 8.3 分类训练 Stage2（可选微调）
- 脚本：`scripts/cls_train_kfold_stage2.py`，在 Stage1 权重上小学习率再训练，若 `Stage2 F1 ≥ Stage1` 才写 `hat_cls_fold{k}_stage2_best.pt`，否则回退 Stage1。
- 超参：`lr=1e-5`，`warmup_ratio=0.05`，`num_epochs=1~2`，`batch_size≈64`，`grad_clip=1.0`，可选 `AMP`、`EMA(0.9999)`。
- 正则/增强：随机滑窗起点（token 偏移，默认 stride=128）、可选 `R-Drop (λ=0.5)`、`FGM (ε=0.5, ratio=1.0)`。
- 结果：报告中 Stage2 未稳定超越 Stage1，默认推理仍使用 Stage1 模型。

---

## 9. 推理与部署

- 单/多模型：`scripts/infer.py` 支持 `--window-agg [mean|max|mean_conf]`、`--model-agg [logits_avg|prob_avg|voting|logits_avg_weighted|prob_avg_weighted]`、`--window-tta-offsets`、`--mc-dropout-runs`、阈值调节（统一阈值或逐类阈值）和 logits 持久化。
- K-Fold 集成：`scripts/infer_kfold.py` 自动扫描目录，优先 `hat_cls_fold*_stage2_best.pt`，缺失时回退 `hat_cls_fold*_best.pt`，并按需以 `torchrun --nproc-per-node` 启动多 GPU。
- 聚合顺序：窗口级聚合 → 模型级聚合 → 写出 `outputs/submission/*.csv`（可同时保存 logits）。
- 形状：保持 `[B,N,K]` 输入，segmenter 在推理侧可滑窗分段；模型内部补 `CLS_SEG`、扩展 mask 对齐。

---

## 10. 代码与文档映射

- 公共超参：`src/common_config.py`（唯一真相来源，对齐 vocab/长度/类别数）。
- 预处理：`scripts/run_preprocessing.py` → `src/data_preprocess/preprocess.py`；分词/分段/数据集在 `tokenizer.py`、`segmenter.py`、`dataset.py`，类别平衡在 `class_balance.py`。
- 模型：`src/model/hat_model.py`（分类）、`src/model/hat_pretrain.py`（MLM）；损失函数 `src/losses.py`。
- 训练/推理脚本：`scripts/mlm_train.py`、`scripts/cls_train*.py`、`scripts/infer*.py`；集群模板见 `scripts/slurm_scripts/`。
- 报告参考：`report/report.tex`（实验过程与结果），本技术规格与报告保持一致。