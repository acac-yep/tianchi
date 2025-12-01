"""
公共配置文件
统一定义数据预处理和模型共享的关键参数，避免重复定义导致不一致。

使用方式：
    from src.common_config import CommonConfig, COMMON_CONFIG
    
    # 数据预处理和模型都从这里获取共享参数
    vocab_size = COMMON_CONFIG.vocab_size
    segment_length = COMMON_CONFIG.segment_length
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CommonConfig:
    """
    项目公共配置（不可变）
    
    这些参数在数据预处理和模型中都会使用，
    统一定义以确保一致性。
    
    长度说明：
    - max_seq_length (4096): 原始 token 级最大长度，不含 CLS
    - max_model_tokens (4104): 模型内部总 token 数，含每段的 CLS_SEG
    """
    
    # ========== 词表相关 ==========
    # 原始 token ID 范围
    original_min_token_id: int = 0
    original_max_token_id: int = 7549
    
    # ID 偏移量（为特殊 token 预留空间）
    # 原始 token ID 会整体 +5，预留 0-4 给特殊 token
    id_offset: int = 5
    
    # 词表大小 = 原始词表 + 偏移量
    # 7549 - 0 + 1 = 7550 个原始 token，偏移后范围 5-7554
    # 加上特殊 token 0-4，总共 7555
    vocab_size: int = 7555
    
    # ========== 特殊 Token ID ==========
    pad_token_id: int = 0
    unk_token_id: int = 1
    # CLS token:
    # 当前设计中仅用作 segment-level CLS (CLS_SEG)。
    # 文档级表示通过所有 segment CLS 的 masked 平均得到，
    # 本方案中没有单独的 doc-level CLS token。
    cls_token_id: int = 2
    sep_token_id: int = 3
    mask_token_id: int = 4
    
    # ========== 序列长度相关 ==========
    # 每个 segment 的长度（不含 CLS_SEG）
    segment_length: int = 512
    
    # 最大 segment 数量
    max_segments: int = 8
    
    # 原始 token 级最大长度（不含 CLS）= segment_length * max_segments
    max_seq_length: int = 4096
    
    # 段内最大位置数（含 CLS_SEG）= segment_length + 1
    max_position_embeddings_segment: int = 513
    
    # 段级最大位置数 = max_segments
    max_position_embeddings_segment_level: int = 8
    
    # 模型内部真实总 token 数（含每段的 CLS_SEG）
    # = max_segments * (segment_length + 1) = 8 * 513 = 4104
    max_model_tokens: int = 4104
    
    # ========== 任务相关 ==========
    # 分类类别数
    num_labels: int = 14
    
    def __post_init__(self):
        """验证配置一致性"""
        # 长度一致性校验
        assert self.max_seq_length == self.segment_length * self.max_segments, \
            f"max_seq_length ({self.max_seq_length}) 应等于 segment_length * max_segments"
        assert self.max_position_embeddings_segment == self.segment_length + 1, \
            f"max_position_embeddings_segment 应等于 segment_length + 1"
        assert self.max_position_embeddings_segment_level == self.max_segments, \
            f"max_position_embeddings_segment_level 应等于 max_segments"
        assert self.max_model_tokens == self.max_segments * (self.segment_length + 1), \
            f"max_model_tokens ({self.max_model_tokens}) 应等于 max_segments * (segment_length + 1)"
        
        # 特殊 token ID 校验
        assert self.id_offset == self.mask_token_id + 1, \
            f"id_offset 应等于 mask_token_id + 1"
        assert self.vocab_size > self.id_offset, \
            f"vocab_size ({self.vocab_size}) 应大于 id_offset ({self.id_offset})"
        
        # 特殊 token ID 范围校验
        assert 0 <= self.pad_token_id < self.id_offset, \
            f"pad_token_id ({self.pad_token_id}) 应在 [0, {self.id_offset}) 范围内"
        assert 0 <= self.unk_token_id < self.id_offset, \
            f"unk_token_id ({self.unk_token_id}) 应在 [0, {self.id_offset}) 范围内"
        assert 0 <= self.cls_token_id < self.id_offset, \
            f"cls_token_id ({self.cls_token_id}) 应在 [0, {self.id_offset}) 范围内"
        assert 0 <= self.sep_token_id < self.id_offset, \
            f"sep_token_id ({self.sep_token_id}) 应在 [0, {self.id_offset}) 范围内"
        assert 0 <= self.mask_token_id < self.id_offset, \
            f"mask_token_id ({self.mask_token_id}) 应在 [0, {self.id_offset}) 范围内"
        
        # 任务相关校验
        assert self.num_labels > 0, \
            f"num_labels ({self.num_labels}) 应大于 0"


# 全局单例配置
COMMON_CONFIG = CommonConfig()


def get_common_config() -> CommonConfig:
    """获取公共配置实例"""
    return COMMON_CONFIG
