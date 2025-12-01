# 数据预处理方案文档

## 概述

本文档描述了 HAT-I1 模型的完整数据预处理方案，针对 20 万条长文本分类任务进行了专门优化。

## 目录结构

```
src/data/
├── __init__.py          # 模块导出
├── config.py            # 配置管理
├── tokenizer.py         # Token 处理（ID 重映射）
├── segmenter.py         # 文档分段器
├── data_cleaner.py      # 数据清洗
├── class_balance.py     # 类别平衡
├── dataset.py           # PyTorch Dataset
└── preprocess.py        # 主预处理脚本
```

## 1. Token ID 重映射

### 问题
原始数据中 token ID 范围为 0-7549，与特殊 token (PAD=0, UNK=1, CLS=2, SEP=3, MASK=4) 冲突。

### 解决方案
所有原始 token ID 整体偏移 +5：
- 原始: `0-7549` → 新: `5-7554`
- 预留 0-4 给特殊 token
- 最终词表大小: 7555

### 使用示例

```python
from src.data import create_tokenizer

tokenizer = create_tokenizer()

# 编码文本
text = "100 200 300"
token_ids = tokenizer.encode(text)  # [105, 205, 305]

# 解码还原
decoded = tokenizer.decode(token_ids)  # "100 200 300"
```

## 2. 文档分段策略

### 配置
- 每段长度 K = 512 tokens
- 最大段数 N = 8（总长度 4096）
- 每段前添加 CLS_SEG token

### 尾段回拉
当最后一段长度 < 256 时（阈值=50%），从文档末尾回拉一整段 512 tokens。

### 超长文档处理

**训练时：**
- `random_window`: 随机选择连续 8 段（数据增强）
- `head_tail`: 前 7 段 + 最后 1 段
- `head_only`: 只取前 8 段

**推理时：**
- 滑动窗口，步长 4 段
- 多个窗口的 logits 求和/平均

### 使用示例

```python
from src.data import create_tokenizer, create_segmenter

tokenizer = create_tokenizer()
segmenter = create_segmenter()

# 分段处理
token_ids = tokenizer.encode(long_text)
segmented = segmenter.segment_document(token_ids, mode='train')

print(f"分段数: {segmented.num_segments}")
print(f"各段长度: {segmented.segment_lengths}")

## 3. 数据清洗

### 处理项目

| 问题类型 | 处理方式 | 配置选项 |
|---------|---------|---------|
| 完全重复样本 | 保留第一个 | `remove_duplicates=True` |
| 标签冲突 (同文本不同标签) | 全部移除 | `remove_label_conflicts=True` |
| 空文本 | 移除 | 默认 |
| 超短文本 (<5 tokens) | 标记/可选移除 | `min_text_length=5` |
| 极长文本 (>10000 tokens) | 标记 | `extreme_long_threshold=10000` |

### 使用示例

```python
from src.data import DataCleaner, DataCleaningConfig

config = DataCleaningConfig(
    remove_duplicates=True,
    remove_label_conflicts=True,
)

cleaner = DataCleaner(config)
cleaned_df, report = cleaner.clean(df)

print(report)  # 打印清洗报告

## 4. 类别平衡处理

### 类别权重计算

支持多种方法：
- `inverse_sqrt`: $w_c = 1/\sqrt{n_c}$（推荐）
- `inverse_log`: $w_c = 1/\log(1+n_c)$
- `effective_num`: Effective Number of Samples
- `inverse_freq`: $w_c = 1/n_c$

### 采样策略

使用 `WeightedRandomSampler`，可配置平滑因子降低过采样强度。

### 使用示例

```python
from src.data import compute_class_weights, create_weighted_sampler

# 计算类别权重（用于损失函数）
weights = compute_class_weights(labels, method='inverse_sqrt')
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))

# 创建加权采样器（用于 DataLoader）
sampler = create_weighted_sampler(labels)
dataloader = DataLoader(dataset, sampler=sampler)

## 5. PyTorch Dataset

### HATDataset

专为 HAT 模型设计的 Dataset：
- 支持懒加载和缓存
- 自动进行 tokenization 和分段
- 支持训练和推理模式

### HATDataCollator

批处理整理器：
- 将不同 segment 数量的样本 padding 到相同维度
- 生成 `segment_mask` 标记有效 segment

### MLMDataCollator

预训练专用整理器：
- 继承 HATDataCollator
- 额外处理 15% mask 策略 (80% [MASK], 10% 随机, 10% 保持)

### 使用示例

```python
from src.data import (
    HATDataset, 
    HATDataCollator, 
    create_dataloader
)

# 创建数据集
dataset = HATDataset(
    texts=df['text'].tolist(),
    labels=df['label'].tolist(),
    mode='train',
)

# 创建 DataLoader
collator = HATDataCollator()
dataloader = create_dataloader(
    dataset,
    batch_size=32,
    use_weighted_sampler=True,
    collator=collator,
)

# 训练循环
for batch in dataloader:
    input_ids = batch['input_ids']      # [B, N, K]
    attention_mask = batch['attention_mask']  # [B, N, K]
    segment_mask = batch['segment_mask']  # [B, N]
    labels = batch['labels']  # [B]
```

## 6. 完整预处理流程

### 运行预处理

```bash
# 方式 1: 使用模块
python -m src.data.preprocess

# 方式 2: 使用脚本
python scripts/run_preprocessing.py
```

### 输出文件

```
data/processed/
├── train.csv            # 清洗后的训练集
├── val.csv              # 验证集 (10%)
├── test.csv             # 测试集
├── class_weights.npy    # 类别权重
├── tokenizer/           # Tokenizer 配置
│   └── tokenizer_config.json
└── preprocessing_report.json  # 预处理报告
```

## 7. 配置参考

### 完整配置结构

```python
from src.data import DataConfig

config = DataConfig(
    # 数据路径
    data_dir=Path("/path/to/data"),
    output_dir=Path("/path/to/output"),
    
    # Tokenizer
    tokenizer=TokenizerConfig(
        id_offset=5,
        vocab_size=7555,
    ),
    
    # Segmenter
    segmenter=SegmenterConfig(
        segment_length=512,
        max_segments=8,
        train_long_strategy='random_window',
    ),
    
    # 数据清洗
    cleaning=DataCleaningConfig(
        remove_duplicates=True,
        remove_label_conflicts=True,
    ),
    
    # 类别平衡
    class_balance=ClassBalanceConfig(
        weight_method='inverse_sqrt',
        use_weighted_sampler=True,
    ),
    
    # 其他
    num_labels=14,
    val_split_ratio=0.1,
    seed=42,
)
```

## 8. 注意事项

1. **内存管理**: 对于大数据集，使用 `cache_segments=False` 避免内存溢出
2. **随机性控制**: 设置 `seed` 确保可复现性
3. **推理时**: 使用 `get_sliding_windows()` 处理超长文档
4. **预训练**: 使用 `MLMDataCollator` 替代 `HATDataCollator`

## 9. 性能优化建议

1. **预处理缓存**: 大规模数据集可预先处理并保存
2. **多进程加载**: 使用 `num_workers > 0`
3. **Pin Memory**: 启用 `pin_memory=True` 加速 GPU 传输
4. **混合精度**: 配合 bf16/fp16 训练
