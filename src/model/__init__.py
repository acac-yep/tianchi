"""
HAT 模型模块

包含：
- HATConfig: 模型配置
- HATInterleaved512ForClassification: 分类模型
- HATInterleaved512ForMLM: 预训练模型 (MLM)
"""

from .hat_model import (
    HATConfig,
    HATEmbeddings,
    TransformerEncoderBlock,
    HATLayer,
    HATEncoder,
    HATInterleaved512ForClassification,
    create_model,
    print_model_info,
)

try:
    from .hat_pretrain import (
        HATInterleaved512ForMLM,
        create_mlm_model,
    )
except ImportError:
    # hat_pretrain 依赖 hat_model，如果单独导入可能失败
    pass

__all__ = [
    # Config
    'HATConfig',
    
    # Model Components
    'HATEmbeddings',
    'TransformerEncoderBlock',
    'HATLayer',
    'HATEncoder',
    
    # Full Models
    'HATInterleaved512ForClassification',
    'HATInterleaved512ForMLM',
    
    # Factory Functions
    'create_model',
    'create_mlm_model',
    'print_model_info',
]

