"""
数据预处理配置文件
HAT-I1 模型的数据处理参数配置
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class SpecialTokens:
    """特殊 Token 配置"""
    PAD: int = 0      # 填充 token
    UNK: int = 1      # 未知 token
    CLS_DOC: int = 2  # 文档级 CLS（可选）
    SEP: int = 3      # 分隔符
    MASK: int = 4     # MLM 掩码
    
    # 特殊 token 的数量，用于计算偏移量
    NUM_SPECIAL: int = 5
    
    def to_dict(self) -> Dict[str, int]:
        return {
            '[PAD]': self.PAD,
            '[UNK]': self.UNK,
            '[CLS]': self.CLS_DOC,
            '[SEP]': self.SEP,
            '[MASK]': self.MASK,
        }


@dataclass
class SegmenterConfig:
    """分段器配置"""
    # 每个 segment 的最大长度（不含 CLS_SEG）
    segment_length: int = 512
    
    # 最大 segment 数量（4096 / 512 = 8）
    max_segments: int = 8
    
    # 最大序列总长度
    max_seq_length: int = 4096
    
    # 尾段回拉的最小长度阈值（低于此值则回拉）
    tail_pullback_threshold: float = 0.5  # segment_length 的比例
    
    # 训练时对超长文档的处理策略: 'random_window', 'head_tail', 'head_only'
    train_long_strategy: str = 'random_window'
    
    # 推理时对超长文档的处理策略: 'sliding_window', 'head_only'
    infer_long_strategy: str = 'sliding_window'
    
    # 滑动窗口的步长（推理时使用）
    sliding_window_stride: int = 4  # segment 数量


@dataclass
class TokenizerConfig:
    """Tokenizer 配置"""
    # 原始 token ID 范围
    original_min_id: int = 0
    original_max_id: int = 7549
    
    # ID 偏移量（为特殊 token 预留空间）
    id_offset: int = 5
    
    # 词表大小（包含特殊 token）
    vocab_size: int = 7555  # 7549 + 1 + 5
    
    # 是否在每个 segment 前添加 CLS
    add_segment_cls: bool = True


@dataclass
class DataCleaningConfig:
    """数据清洗配置"""
    # 是否移除完全重复的样本
    remove_duplicates: bool = True
    
    # 是否移除标签冲突的样本（同一文本不同标签）
    remove_label_conflicts: bool = True
    
    # 最小文本长度（低于此长度的样本将被标记）
    min_text_length: int = 5
    
    # 是否移除超短文本
    remove_short_texts: bool = False
    
    # 极长文本阈值（超过此长度需要特殊处理）
    extreme_long_threshold: int = 10000


@dataclass
class ClassBalanceConfig:
    """类别平衡配置"""
    # 类别权重计算方式: 'inverse_sqrt', 'inverse_log', 'effective_num', 'none'
    weight_method: str = 'inverse_sqrt'
    
    # Effective Number of Samples 的 beta 参数（仅在 weight_method='effective_num' 时使用）
    effective_num_beta: float = 0.9999
    
    # 是否使用 WeightedRandomSampler
    use_weighted_sampler: bool = True
    
    # 采样器的权重平滑因子（降低过采样强度）
    sampler_smoothing: float = 0.5  # 0=均匀采样, 1=完全按权重采样


@dataclass
class DataConfig:
    """数据预处理总配置"""
    # 数据路径
    data_dir: Path = Path("/home/byhx/workspace/tianchi/data")
    train_file: str = "train_set.csv"
    test_file: str = "test_a.csv"
    
    # 输出路径
    output_dir: Path = Path("/home/byhx/workspace/tianchi/data/processed")
    
    # 子配置
    special_tokens: SpecialTokens = field(default_factory=SpecialTokens)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    cleaning: DataCleaningConfig = field(default_factory=DataCleaningConfig)
    class_balance: ClassBalanceConfig = field(default_factory=ClassBalanceConfig)
    
    # 类别数
    num_labels: int = 14
    
    # 验证集划分比例
    val_split_ratio: float = 0.1
    
    # 随机种子
    seed: int = 42
    
    @property
    def train_path(self) -> Path:
        return self.data_dir / self.train_file
    
    @property
    def test_path(self) -> Path:
        return self.data_dir / self.test_file
    
    def __post_init__(self):
        """确保输出目录存在"""
        self.output_dir.mkdir(parents=True, exist_ok=True)


def get_default_config() -> DataConfig:
    """获取默认配置"""
    return DataConfig()


# 导出默认配置实例
DEFAULT_CONFIG = get_default_config()



