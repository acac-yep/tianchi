"""
HAT 模型数据预处理模块

主要组件：
- config: 配置管理
- tokenizer: Token 处理（ID 重映射）
- segmenter: 文档分段器
- data_cleaner: 数据清洗
- class_balance: 类别平衡
- dataset: PyTorch Dataset
"""

from .config import (
    DataConfig,
    SpecialTokens,
    SegmenterConfig,
    TokenizerConfig,
    DataCleaningConfig,
    ClassBalanceConfig,
    get_default_config,
    DEFAULT_CONFIG,
)

from .tokenizer import (
    HATTokenizer,
    create_tokenizer,
)

from .segmenter import (
    DocumentSegmenter,
    SegmentedDocument,
    create_segmenter,
)

from .data_cleaner import (
    DataCleaner,
    CleaningReport,
    TextLengthAnalyzer,
    clean_dataset,
)

from .class_balance import (
    ClassWeightCalculator,
    ClassDistributionAnalyzer,
    FocalLossWeights,
    LabelSmoothingConfig,
    compute_class_weights,
    create_weighted_sampler,
)

from .dataset import (
    HATDataset,
    HATDataCollator,
    MLMDataCollator,
    create_dataloader,
    load_dataset_from_csv,
    save_processed_dataset,
    load_processed_dataset,
)

from .preprocess import (
    DataPreprocessor,
    run_preprocessing,
)


__all__ = [
    # Config
    'DataConfig',
    'SpecialTokens',
    'SegmenterConfig',
    'TokenizerConfig',
    'DataCleaningConfig',
    'ClassBalanceConfig',
    'get_default_config',
    'DEFAULT_CONFIG',
    
    # Tokenizer
    'HATTokenizer',
    'create_tokenizer',
    
    # Segmenter
    'DocumentSegmenter',
    'SegmentedDocument',
    'create_segmenter',
    
    # Data Cleaning
    'DataCleaner',
    'CleaningReport',
    'TextLengthAnalyzer',
    'clean_dataset',
    
    # Class Balance
    'ClassWeightCalculator',
    'ClassDistributionAnalyzer',
    'FocalLossWeights',
    'LabelSmoothingConfig',
    'compute_class_weights',
    'create_weighted_sampler',
    
    # Dataset
    'HATDataset',
    'HATDataCollator',
    'MLMDataCollator',
    'create_dataloader',
    'load_dataset_from_csv',
    'save_processed_dataset',
    'load_processed_dataset',
    
    # Preprocessing
    'DataPreprocessor',
    'run_preprocessing',
]



