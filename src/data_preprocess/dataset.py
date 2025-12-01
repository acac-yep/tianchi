"""
PyTorch Dataset 模块
实现 HAT 模型的数据集类和数据加载器
"""

from typing import List, Dict, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object

from .config import DataConfig, DEFAULT_CONFIG
from .tokenizer import HATTokenizer, create_tokenizer
from .segmenter import DocumentSegmenter, SegmentedDocument, create_segmenter
from .class_balance import ClassWeightCalculator, create_weighted_sampler

logger = logging.getLogger(__name__)


class HATDataset(Dataset):
    """
    HAT 模型的 PyTorch Dataset
    
    特点：
    1. 懒加载：只在访问时进行分段处理
    2. 缓存机制：可选择缓存分段结果
    3. 支持预训练和微调两种模式
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: HATTokenizer = None,
        segmenter: DocumentSegmenter = None,
        mode: str = 'train',  # 'train', 'eval', 'pretrain'
        cache_segments: bool = False,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            texts: 文本列表（空格分隔的 token ID 字符串）
            labels: 标签列表（微调时需要）
            tokenizer: HATTokenizer 实例
            segmenter: DocumentSegmenter 实例
            mode: 运行模式
            cache_segments: 是否缓存分段结果
            transform: 数据变换函数
        """
        if not TORCH_AVAILABLE:
            raise ImportError("需要安装 PyTorch: pip install torch")
        
        self.texts = texts
        self.labels = labels
        self.mode = mode
        self.cache_segments = cache_segments
        self.transform = transform
        
        # 初始化 tokenizer 和 segmenter
        self.tokenizer = tokenizer or create_tokenizer()
        self.segmenter = segmenter or create_segmenter()
        
        # 缓存
        self._cache: Dict[int, SegmentedDocument] = {}
        
        # 预分段（如果启用缓存）
        if cache_segments:
            logger.info("预处理并缓存所有分段结果...")
            self._preprocess_all()
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            包含以下键的字典：
            - input_ids: [num_segments, segment_length]
            - attention_mask: [num_segments, segment_length]
            - segment_mask: [num_segments] - 标记哪些 segment 是真实的
            - labels: 标签（如果有）
        """
        # 获取分段结果
        if self.cache_segments and idx in self._cache:
            segmented = self._cache[idx]
        else:
            segmented = self._process_text(idx)
            if self.cache_segments:
                self._cache[idx] = segmented
        
        # 转换为 tensor
        item = {
            'input_ids': torch.tensor(segmented.segment_ids, dtype=torch.long),
            'attention_mask': torch.tensor(segmented.segment_attention_masks, dtype=torch.long),
            'segment_mask': torch.ones(segmented.num_segments, dtype=torch.long),
            'num_segments': segmented.num_segments,
            'original_length': segmented.original_length,
        }
        
        # 添加标签
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # 应用变换
        if self.transform is not None:
            item = self.transform(item)
        
        return item
    
    def _process_text(self, idx: int) -> SegmentedDocument:
        """处理单个文本"""
        text = self.texts[idx]
        
        # Tokenize（编码 + 重映射）
        token_ids = self.tokenizer.encode(text)
        
        # 分段
        segmented = self.segmenter.segment_document(
            token_ids, 
            mode=self.mode if self.mode != 'pretrain' else 'train'
        )
        
        return segmented
    
    def _preprocess_all(self):
        """预处理所有样本"""
        for idx in range(len(self.texts)):
            if idx % 10000 == 0:
                logger.info(f"  处理进度: {idx}/{len(self.texts)}")
            self._cache[idx] = self._process_text(idx)
    
    def get_sliding_windows(self, idx: int) -> List[Dict[str, torch.Tensor]]:
        """
        获取超长文档的滑动窗口（推理时使用）
        
        Returns:
            窗口列表，每个窗口格式同 __getitem__
        """
        text = self.texts[idx]
        token_ids = self.tokenizer.encode(text)
        
        windows = self.segmenter.get_sliding_windows(token_ids)
        
        items = []
        for window in windows:
            item = {
                'input_ids': torch.tensor(window.segment_ids, dtype=torch.long),
                'attention_mask': torch.tensor(window.segment_attention_masks, dtype=torch.long),
                'segment_mask': torch.ones(window.num_segments, dtype=torch.long),
                'num_segments': window.num_segments,
            }
            if self.labels is not None:
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            items.append(item)
        
        return items


class HATDataCollator:
    """
    HAT 模型的数据整理器
    
    将不同长度的样本 padding 到相同的 segment 数量
    """
    
    def __init__(
        self,
        max_segments: int = 8,
        segment_length: int = 512,
        pad_token_id: int = 0,
        pad_to_max: bool = True,
    ):
        self.max_segments = max_segments
        self.segment_length = segment_length
        self.pad_token_id = pad_token_id
        self.pad_to_max = pad_to_max
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        整理一个 batch 的数据
        
        Args:
            batch: 样本列表
            
        Returns:
            batched tensors
        """
        batch_size = len(batch)
        
        # 确定目标 segment 数量
        if self.pad_to_max:
            target_num_segments = self.max_segments
        else:
            target_num_segments = max(item['num_segments'] for item in batch)
        
        # 初始化 batch tensors
        input_ids = torch.full(
            (batch_size, target_num_segments, self.segment_length),
            self.pad_token_id,
            dtype=torch.long
        )
        attention_mask = torch.zeros(
            (batch_size, target_num_segments, self.segment_length),
            dtype=torch.long
        )
        segment_mask = torch.zeros(
            (batch_size, target_num_segments),
            dtype=torch.long
        )
        
        # 填充
        for i, item in enumerate(batch):
            num_segs = item['num_segments']
            input_ids[i, :num_segs] = item['input_ids']
            attention_mask[i, :num_segs] = item['attention_mask']
            segment_mask[i, :num_segs] = 1
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'segment_mask': segment_mask,
        }
        
        # 处理标签
        if 'labels' in batch[0]:
            labels = torch.stack([item['labels'] for item in batch])
            result['labels'] = labels
        
        return result


class MLMDataCollator(HATDataCollator):
    """
    MLM 预训练的数据整理器
    
    额外处理 mask 标记
    """
    
    def __init__(
        self,
        tokenizer: HATTokenizer,
        mlm_probability: float = 0.15,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        
        # 特殊 token 集合（不能被 mask）
        self.special_token_ids = {
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.unk_token_id,
            tokenizer.mask_token_id,
        }
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        整理 batch 并添加 MLM masks
        """
        # 先调用父类的整理逻辑
        result = super().__call__(batch)
        
        # 创建 MLM 标签
        input_ids = result['input_ids'].clone()
        mlm_labels = result['input_ids'].clone()
        
        # 创建 mask 概率矩阵
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        
        # 特殊 token 位置不 mask
        special_tokens_mask = self._get_special_tokens_mask(input_ids)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # padding 位置不 mask
        probability_matrix.masked_fill_(result['attention_mask'] == 0, value=0.0)
        
        # 采样 mask 位置
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 非 mask 位置的标签设为 -100（忽略）
        mlm_labels[~masked_indices] = -100
        
        # 80% 替换为 [MASK]
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        
        # 10% 替换为随机 token
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            self.tokenizer.config.id_offset,  # 跳过特殊 token
            self.vocab_size,
            input_ids.shape,
            dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]
        
        # 10% 保持不变（已经设置了标签）
        
        result['input_ids'] = input_ids
        result['mlm_labels'] = mlm_labels
        
        return result
    
    def _get_special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """获取特殊 token 的掩码"""
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_id in self.special_token_ids:
            mask |= (input_ids == special_id)
        return mask


def create_dataloader(
    dataset: HATDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    use_weighted_sampler: bool = False,
    collator: HATDataCollator = None,
    **kwargs
) -> DataLoader:
    """
    创建 DataLoader
    
    Args:
        dataset: HATDataset 实例
        batch_size: batch 大小
        shuffle: 是否打乱（与 weighted_sampler 互斥）
        num_workers: 工作进程数
        use_weighted_sampler: 是否使用加权采样器
        collator: 数据整理器
        **kwargs: 其他 DataLoader 参数
        
    Returns:
        DataLoader 实例
    """
    if not TORCH_AVAILABLE:
        raise ImportError("需要安装 PyTorch: pip install torch")
    
    # 默认 collator
    if collator is None:
        collator = HATDataCollator()
    
    # 创建采样器
    sampler = None
    if use_weighted_sampler and dataset.labels is not None:
        sampler = create_weighted_sampler(dataset.labels)
        shuffle = False  # sampler 和 shuffle 互斥
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        **kwargs
    )


def load_dataset_from_csv(
    csv_path: Union[str, Path],
    text_col: str = 'text',
    label_col: Optional[str] = 'label',
    tokenizer: HATTokenizer = None,
    segmenter: DocumentSegmenter = None,
    mode: str = 'train',
    **kwargs
) -> HATDataset:
    """
    从 CSV 文件加载数据集
    
    Args:
        csv_path: CSV 文件路径
        text_col: 文本列名
        label_col: 标签列名（None 表示无标签）
        tokenizer: tokenizer 实例
        segmenter: segmenter 实例
        mode: 运行模式
        **kwargs: 传递给 HATDataset 的其他参数
        
    Returns:
        HATDataset 实例
    """
    df = pd.read_csv(csv_path, sep='\t')
    
    texts = df[text_col].tolist()
    labels = df[label_col].tolist() if label_col and label_col in df.columns else None
    
    return HATDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        segmenter=segmenter,
        mode=mode,
        **kwargs
    )


def save_processed_dataset(
    dataset: HATDataset,
    save_path: Union[str, Path]
):
    """
    保存预处理后的数据集
    
    Args:
        dataset: 已缓存的 HATDataset
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'cache': dataset._cache,
        'labels': dataset.labels,
        'mode': dataset.mode,
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"数据集已保存到: {save_path}")


def load_processed_dataset(
    load_path: Union[str, Path],
    tokenizer: HATTokenizer = None,
    segmenter: DocumentSegmenter = None,
) -> HATDataset:
    """
    加载预处理后的数据集
    
    Args:
        load_path: 加载路径
        tokenizer: tokenizer 实例
        segmenter: segmenter 实例
        
    Returns:
        HATDataset 实例
    """
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    
    # 创建空文本列表（只使用缓存）
    texts = [''] * len(data['cache'])
    
    dataset = HATDataset(
        texts=texts,
        labels=data['labels'],
        tokenizer=tokenizer,
        segmenter=segmenter,
        mode=data['mode'],
        cache_segments=True,
    )
    
    # 直接使用加载的缓存
    dataset._cache = data['cache']
    
    logger.info(f"数据集已从 {load_path} 加载")
    return dataset



