"""
文档分段器模块
实现 HAT 模型需要的长文档分段策略

注意：分段器只负责将长文档切分成多个 segment，不添加任何特殊 token。
CLS_SEG 由模型在 forward 时统一添加。
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import random
from dataclasses import dataclass

from .config import SegmenterConfig, SpecialTokens, DEFAULT_CONFIG


@dataclass
class SegmentedDocument:
    """
    分段后的文档数据结构
    
    注意：segment_ids 中不包含 CLS_SEG，CLS_SEG 由模型添加。
    因此每个 segment 的长度为 segment_length（如 512），
    模型添加 CLS_SEG 后变为 segment_length + 1（如 513）。
    """
    # 分段后的 token IDs，形状: [num_segments, segment_length]
    # 不含 CLS_SEG，由模型添加
    segment_ids: List[List[int]]
    
    # 每个 segment 的 attention mask，形状: [num_segments, segment_length]
    segment_attention_masks: List[List[int]]
    
    # 每个 segment 的实际 token 长度（不含 padding，不含 CLS_SEG）
    segment_lengths: List[int]
    
    # 原始文档的总 token 数
    original_length: int
    
    # 分段数量
    num_segments: int
    
    # 是否被截断
    is_truncated: bool
    
    # 截断策略（如果被截断）
    truncation_strategy: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'segment_ids': self.segment_ids,
            'segment_attention_masks': self.segment_attention_masks,
            'segment_lengths': self.segment_lengths,
            'original_length': self.original_length,
            'num_segments': self.num_segments,
            'is_truncated': self.is_truncated,
            'truncation_strategy': self.truncation_strategy
        }


class DocumentSegmenter:
    """
    文档分段器
    
    将长文档按照 HAT 模型的要求切分成多个 segment：
    1. 每个 segment 长度为 K=512 tokens
    2. 最多 8 个 segment（总长度 4096）
    3. 尾段回拉策略避免过短的尾段
    
    注意：
    - 分段器不添加任何特殊 token（CLS_SEG 由模型负责添加）
    - 输出的 segment_ids 形状为 [N, K]，模型接收后会变成 [N, K+1]
    """
    
    def __init__(
        self,
        config: SegmenterConfig = None,
        special_tokens: SpecialTokens = None,
    ):
        self.config = config or DEFAULT_CONFIG.segmenter
        self.special_tokens = special_tokens or DEFAULT_CONFIG.special_tokens
        
        # 预计算常用值
        self.segment_length = self.config.segment_length  # K = 512
        self.max_segments = self.config.max_segments      # 8
        self.max_seq_length = self.config.max_seq_length  # 4096
        
        # 尾段回拉阈值
        self.tail_threshold = int(self.segment_length * self.config.tail_pullback_threshold)
        
        # 特殊 token（仅用于 padding）
        self.pad_token_id = self.special_tokens.PAD
        
    def segment_document(
        self,
        token_ids: List[int],
        mode: str = 'train',
        random_seed: Optional[int] = None
    ) -> SegmentedDocument:
        """
        将文档分段
        
        Args:
            token_ids: 重映射后的 token ID 列表
            mode: 'train' 或 'infer'，决定超长文档的处理策略
            random_seed: 随机种子（用于训练时的随机窗口选择）
            
        Returns:
            SegmentedDocument 对象
            - segment_ids: [N, K] 不含 CLS_SEG
            - 模型会在 forward 时将其转换为 [N, K+1]
        """
        original_length = len(token_ids)
        
        if original_length == 0:
            # 空文档处理
            return self._create_empty_document()
        
        # Step 1: 按 segment_length 切分
        raw_segments = self._split_into_segments(token_ids)
        
        # Step 2: 尾段回拉处理
        segments = self._handle_tail_segment(raw_segments, token_ids)
        
        # Step 3: 处理超长文档
        is_truncated = len(segments) > self.max_segments
        truncation_strategy = None
        
        if is_truncated:
            if mode == 'train':
                segments, truncation_strategy = self._truncate_for_training(
                    segments, random_seed
                )
            else:
                # 推理时返回第一个窗口，外部逻辑处理滑动窗口
                segments = segments[:self.max_segments]
                truncation_strategy = 'head_only'
        
        # Step 4: Padding（不添加 CLS）
        segment_ids, segment_masks, segment_lengths = self._finalize_segments(segments)
        
        return SegmentedDocument(
            segment_ids=segment_ids,
            segment_attention_masks=segment_masks,
            segment_lengths=segment_lengths,
            original_length=original_length,
            num_segments=len(segment_ids),
            is_truncated=is_truncated,
            truncation_strategy=truncation_strategy
        )
    
    def _split_into_segments(self, token_ids: List[int]) -> List[List[int]]:
        """
        将 token 序列按 segment_length 切分
        """
        segments = []
        for i in range(0, len(token_ids), self.segment_length):
            segment = token_ids[i:i + self.segment_length]
            segments.append(segment)
        return segments
    
    def _handle_tail_segment(
        self, 
        segments: List[List[int]], 
        original_tokens: List[int]
    ) -> List[List[int]]:
        """
        尾段回拉处理
        
        如果最后一个 segment 太短（< threshold），则从原文末尾回拉一整段
        """
        if len(segments) <= 1:
            return segments
        
        last_segment = segments[-1]
        
        # 如果尾段长度小于阈值，执行回拉
        if len(last_segment) < self.tail_threshold:
            # 计算回拉起点：从文档末尾往前数 segment_length 个 token
            pullback_start = max(0, len(original_tokens) - self.segment_length)
            new_last_segment = original_tokens[pullback_start:]
            
            # 替换最后一个 segment
            segments[-1] = new_last_segment
        
        return segments
    
    def _truncate_for_training(
        self, 
        segments: List[List[int]], 
        random_seed: Optional[int] = None
    ) -> Tuple[List[List[int]], str]:
        """
        训练时的截断策略
        
        支持三种策略：
        1. random_window: 随机选择连续的 max_segments 个段（推荐，数据增强效果）
        2. head_tail: 前 (max_segments-1) 段 + 最后一段
        3. head_only: 只取前 max_segments 段
        """
        strategy = self.config.train_long_strategy
        
        if strategy == 'random_window':
            if random_seed is not None:
                random.seed(random_seed)
            
            # 随机选择一个起始位置
            max_start = len(segments) - self.max_segments
            start_idx = random.randint(0, max_start)
            selected = segments[start_idx:start_idx + self.max_segments]
            
        elif strategy == 'head_tail':
            # 前 (max_segments - 1) 段 + 最后一段
            head_count = self.max_segments - 1
            selected = segments[:head_count] + [segments[-1]]
            
        else:  # head_only
            selected = segments[:self.max_segments]
        
        return selected, strategy
    
    def _finalize_segments(
        self, 
        segments: List[List[int]]
    ) -> Tuple[List[List[int]], List[List[int]], List[int]]:
        """
        最终处理：Padding 到 segment_length
        
        注意：不添加 CLS_SEG，CLS_SEG 由模型负责添加
        
        Returns:
            (segment_ids, attention_masks, actual_lengths)
        """
        segment_ids = []
        attention_masks = []
        segment_lengths = []
        
        for segment in segments:
            actual_length = len(segment)
            segment_lengths.append(actual_length)
            
            # Padding 到 segment_length
            padding_length = self.segment_length - actual_length
            if padding_length > 0:
                padded_segment = segment + [self.pad_token_id] * padding_length
                attention_mask = [1] * actual_length + [0] * padding_length
            else:
                # 如果刚好等于 segment_length
                padded_segment = segment[:self.segment_length]
                attention_mask = [1] * self.segment_length
            
            segment_ids.append(padded_segment)
            attention_masks.append(attention_mask)
        
        return segment_ids, attention_masks, segment_lengths
    
    def _create_empty_document(self) -> SegmentedDocument:
        """创建空文档的分段结果"""
        # 空文档：全 padding
        segment = [self.pad_token_id] * self.segment_length
        mask = [0] * self.segment_length
        
        return SegmentedDocument(
            segment_ids=[segment],
            segment_attention_masks=[mask],
            segment_lengths=[0],
            original_length=0,
            num_segments=1,
            is_truncated=False,
            truncation_strategy=None
        )
    
    def get_sliding_windows(
        self, 
        token_ids: List[int]
    ) -> List[SegmentedDocument]:
        """
        推理时的滑动窗口处理
        
        将超长文档切成多个窗口，每个窗口独立分段。
        外部代码需要对多个窗口的 logits 进行聚合（如求和/平均）。
        
        Args:
            token_ids: 重映射后的 token ID 列表
            
        Returns:
            SegmentedDocument 列表，每个对应一个窗口
        """
        # 首先切分成所有 segments
        all_segments = self._split_into_segments(token_ids)
        all_segments = self._handle_tail_segment(all_segments, token_ids)
        
        if len(all_segments) <= self.max_segments:
            # 不需要滑动窗口
            return [self.segment_document(token_ids, mode='infer')]
        
        windows = []
        stride = self.config.sliding_window_stride
        
        for start in range(0, len(all_segments), stride):
            end = min(start + self.max_segments, len(all_segments))
            window_segments = all_segments[start:end]
            
            # 如果窗口太小（最后一个窗口可能不足），跳过
            if len(window_segments) < stride:
                continue
            
            # 为每个窗口创建 SegmentedDocument
            segment_ids, segment_masks, segment_lengths = self._finalize_segments(window_segments)
            
            windows.append(SegmentedDocument(
                segment_ids=segment_ids,
                segment_attention_masks=segment_masks,
                segment_lengths=segment_lengths,
                original_length=len(token_ids),
                num_segments=len(segment_ids),
                is_truncated=True,
                truncation_strategy='sliding_window'
            ))
        
        # 确保至少有一个窗口
        if not windows:
            return [self.segment_document(token_ids, mode='infer')]
        
        return windows
    
    def pad_batch(
        self, 
        documents: List[SegmentedDocument],
        pad_to_max_segments: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        将一批分段后的文档 padding 到相同的 segment 数量
        
        Args:
            documents: SegmentedDocument 列表
            pad_to_max_segments: 是否 pad 到 max_segments（否则 pad 到 batch 内最大值）
            
        Returns:
            包含 batched tensors 的字典:
            - input_ids: [B, N, K] - 不含 CLS_SEG
            - attention_mask: [B, N, K]
            - segment_mask: [B, N]
        """
        if pad_to_max_segments:
            target_num_segments = self.max_segments
        else:
            target_num_segments = max(doc.num_segments for doc in documents)
        
        batch_size = len(documents)
        
        # 初始化 batch tensors
        batch_segment_ids = np.full(
            (batch_size, target_num_segments, self.segment_length),
            self.pad_token_id,
            dtype=np.int64
        )
        batch_attention_masks = np.zeros(
            (batch_size, target_num_segments, self.segment_length),
            dtype=np.int64
        )
        batch_segment_mask = np.zeros(
            (batch_size, target_num_segments),
            dtype=np.int64
        )
        
        for i, doc in enumerate(documents):
            num_segs = doc.num_segments
            batch_segment_ids[i, :num_segs] = doc.segment_ids
            batch_attention_masks[i, :num_segs] = doc.segment_attention_masks
            batch_segment_mask[i, :num_segs] = 1
        
        return {
            'input_ids': batch_segment_ids,           # [B, N, K]
            'attention_mask': batch_attention_masks,  # [B, N, K]
            'segment_mask': batch_segment_mask,       # [B, N] - 哪些 segment 是真实的
        }


def create_segmenter(config: SegmenterConfig = None) -> DocumentSegmenter:
    """工厂函数：创建分段器实例"""
    return DocumentSegmenter(config=config)
