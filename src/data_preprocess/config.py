"""
数据预处理配置文件
HAT-I1 模型的数据处理参数配置

注意：关键参数（vocab_size, segment_length 等）从 common_config 引用，
确保与模型配置一致。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path
import sys

# 添加项目根目录以导入 common_config
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.common_config import COMMON_CONFIG


@dataclass
class SpecialTokens:
    """
    特殊 Token 配置
    
    从 COMMON_CONFIG 引用，确保与模型一致。
    """
    PAD: int = COMMON_CONFIG.pad_token_id
    UNK: int = COMMON_CONFIG.unk_token_id
    CLS_DOC: int = COMMON_CONFIG.cls_token_id
    SEP: int = COMMON_CONFIG.sep_token_id
    MASK: int = COMMON_CONFIG.mask_token_id
    
    # 特殊 token 的数量
    NUM_SPECIAL: int = COMMON_CONFIG.id_offset
    
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
    """
    分段器配置
    
    注意：segment_length, max_segments, max_seq_length 从 COMMON_CONFIG 引用。
    分段器不添加 CLS_SEG，CLS_SEG 由模型负责添加。
    """
    # 从公共配置引用（确保与模型一致）
    segment_length: int = COMMON_CONFIG.segment_length
    max_segments: int = COMMON_CONFIG.max_segments
    max_seq_length: int = COMMON_CONFIG.max_seq_length
    
    # 尾段回拉的最小长度阈值（低于此值则回拉）
    tail_pullback_threshold: float = 0.5  # segment_length 的比例
    
    # 训练时对超长文档的处理策略: 'random_window', 'head_tail', 'head_only'
    # 推荐 random_window，可作为数据增强
    train_long_strategy: str = 'random_window'
    
    # 推理时对超长文档的处理策略: 'sliding_window', 'head_only'
    # 推荐 sliding_window，外部代码对多窗口 logits 聚合
    infer_long_strategy: str = 'sliding_window'
    
    # 滑动窗口的步长（推理时使用）
    sliding_window_stride: int = 4  # segment 数量


@dataclass
class TokenizerConfig:
    """
    Tokenizer 配置
    
    从 COMMON_CONFIG 引用关键参数。
    """
    # 从公共配置引用
    original_min_id: int = COMMON_CONFIG.original_min_token_id
    original_max_id: int = COMMON_CONFIG.original_max_token_id
    id_offset: int = COMMON_CONFIG.id_offset
    vocab_size: int = COMMON_CONFIG.vocab_size


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
    """
    类别平衡配置
    
    警告：默认只使用一种平衡手段。
    - use_weighted_sampler=True + 损失函数不加权：使用采样器平衡
    - use_weighted_sampler=False + 损失函数加权：使用损失权重平衡
    
    不建议同时使用两种方式，可能导致过度补偿。
    """
    # 类别权重计算方式: 'inverse_sqrt', 'inverse_log', 'effective_num', 'none'
    weight_method: str = 'inverse_sqrt'
    
    # Effective Number of Samples 的 beta 参数
    effective_num_beta: float = 0.9999
    
    # 是否使用 WeightedRandomSampler
    # 警告：如果同时使用 sampler 和 loss 权重，可能导致双重补偿
    use_weighted_sampler: bool = False  # 默认关闭，推荐使用 loss 权重
    
    # 采样器的权重平滑因子
    sampler_smoothing: float = 0.5


@dataclass
class DataConfig:
    """
    数据预处理总配置
    
    关键参数从 COMMON_CONFIG 引用，确保与模型一致。
    
    注意：data_dir 和 output_dir 的默认值为 None，应在使用时通过参数传入。
    这样可以避免硬编码路径导致的跨环境问题。
    """
    # 数据路径（默认 None，需通过参数传入）
    data_dir: Optional[Path] = None
    train_file: str = "train_set.csv"
    test_file: str = "test_a.csv"
    
    # 输出路径（默认 None，需通过参数传入）
    output_dir: Optional[Path] = None
    
    # 子配置
    special_tokens: SpecialTokens = field(default_factory=SpecialTokens)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    cleaning: DataCleaningConfig = field(default_factory=DataCleaningConfig)
    class_balance: ClassBalanceConfig = field(default_factory=ClassBalanceConfig)
    
    # 从公共配置引用
    num_labels: int = COMMON_CONFIG.num_labels
    
    # 验证集划分比例
    val_split_ratio: float = 0.1
    
    # 随机种子
    seed: int = 42
    
    @property
    def train_path(self) -> Path:
        if self.data_dir is None:
            raise ValueError("data_dir 未设置，请在使用前通过参数传入")
        return self.data_dir / self.train_file
    
    @property
    def test_path(self) -> Path:
        if self.data_dir is None:
            raise ValueError("data_dir 未设置，请在使用前通过参数传入")
        return self.data_dir / self.test_file
    
    def __post_init__(self):
        """确保输出目录存在（仅在 output_dir 已设置时）"""
        if self.output_dir is not None:
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError) as e:
                # 如果无法创建目录，记录警告但不抛出异常
                # 让调用者处理目录创建
                import warnings
                warnings.warn(
                    f"无法创建输出目录 {self.output_dir}: {e}. "
                    "请确保有写入权限，或在使用时手动创建目录。"
                )


def get_default_config() -> DataConfig:
    """获取默认配置"""
    return DataConfig()


# 导出默认配置实例
DEFAULT_CONFIG = get_default_config()
