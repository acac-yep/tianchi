"""
Token 处理模块
实现 Token ID 重映射和基本的 tokenization 功能
"""

from typing import List, Tuple, Optional, Dict
import numpy as np

from .config import TokenizerConfig, SpecialTokens, DEFAULT_CONFIG


class HATTokenizer:
    """
    HAT 模型的 Tokenizer
    
    主要功能：
    1. Token ID 重映射（解决特殊 token 冲突）
    2. 文本编码/解码
    3. 特殊 token 处理
    """
    
    def __init__(
        self,
        config: TokenizerConfig = None,
        special_tokens: SpecialTokens = None
    ):
        self.config = config or DEFAULT_CONFIG.tokenizer
        self.special_tokens = special_tokens or DEFAULT_CONFIG.special_tokens
        
        # 预计算一些常用值
        self.id_offset = self.config.id_offset
        self.vocab_size = self.config.vocab_size
        
        # 特殊 token IDs
        self.pad_token_id = self.special_tokens.PAD
        self.unk_token_id = self.special_tokens.UNK
        self.cls_token_id = self.special_tokens.CLS_DOC
        self.sep_token_id = self.special_tokens.SEP
        self.mask_token_id = self.special_tokens.MASK
        
    def remap_token_id(self, token_id: int) -> int:
        """
        将原始 token ID 重映射到新的 ID 空间
        
        原始: 0-7549 -> 新: 5-7554
        """
        return token_id + self.id_offset
    
    def inverse_remap_token_id(self, remapped_id: int) -> int:
        """
        将重映射后的 ID 还原为原始 ID
        
        新: 5-7554 -> 原始: 0-7549
        """
        if remapped_id < self.id_offset:
            # 这是一个特殊 token，返回 -1 表示无对应原始 ID
            return -1
        return remapped_id - self.id_offset
    
    def encode(self, text: str) -> List[int]:
        """
        将文本（空格分隔的 token ID 字符串）编码为重映射后的 token ID 列表
        
        Args:
            text: 空格分隔的 token ID 字符串，如 "1234 5678 910"
            
        Returns:
            重映射后的 token ID 列表
        """
        if not text or not text.strip():
            return []
        
        tokens = text.strip().split()
        remapped_ids = []
        
        for token in tokens:
            try:
                original_id = int(token)
                # 检查是否在有效范围内
                if self.config.original_min_id <= original_id <= self.config.original_max_id:
                    remapped_ids.append(self.remap_token_id(original_id))
                else:
                    # 超出范围的 token 映射为 UNK
                    remapped_ids.append(self.unk_token_id)
            except ValueError:
                # 无法解析的 token 映射为 UNK
                remapped_ids.append(self.unk_token_id)
        
        return remapped_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        将 token ID 列表解码为原始文本格式
        
        Args:
            token_ids: 重映射后的 token ID 列表
            skip_special_tokens: 是否跳过特殊 token
            
        Returns:
            空格分隔的原始 token ID 字符串
        """
        original_ids = []
        special_ids = {
            self.pad_token_id, 
            self.unk_token_id, 
            self.cls_token_id, 
            self.sep_token_id, 
            self.mask_token_id
        }
        
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            
            original_id = self.inverse_remap_token_id(token_id)
            if original_id >= 0:
                original_ids.append(str(original_id))
        
        return ' '.join(original_ids)
    
    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        """批量编码文本"""
        return [self.encode(text) for text in texts]
    
    def get_special_tokens_mask(
        self, 
        token_ids: List[int], 
        already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        获取特殊 token 的掩码
        
        Returns:
            掩码列表，特殊 token 位置为 1，其他位置为 0
        """
        if already_has_special_tokens:
            special_ids = {
                self.pad_token_id, 
                self.unk_token_id, 
                self.cls_token_id, 
                self.sep_token_id, 
                self.mask_token_id
            }
            return [1 if token_id in special_ids else 0 for token_id in token_ids]
        return [0] * len(token_ids)
    
    def create_attention_mask(
        self, 
        token_ids: List[int], 
        pad_token_id: Optional[int] = None
    ) -> List[int]:
        """
        创建 attention mask
        
        Returns:
            掩码列表，非 padding 位置为 1，padding 位置为 0
        """
        pad_id = pad_token_id if pad_token_id is not None else self.pad_token_id
        return [1 if token_id != pad_id else 0 for token_id in token_ids]
    
    def pad_sequence(
        self, 
        token_ids: List[int], 
        max_length: int, 
        padding_side: str = 'right',
        truncation: bool = True
    ) -> Tuple[List[int], List[int]]:
        """
        填充或截断序列
        
        Args:
            token_ids: token ID 列表
            max_length: 目标长度
            padding_side: 'left' 或 'right'
            truncation: 是否截断超长序列
            
        Returns:
            (padded_ids, attention_mask)
        """
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        padding_length = max_length - len(token_ids)
        
        if padding_length > 0:
            padding = [self.pad_token_id] * padding_length
            if padding_side == 'right':
                padded_ids = token_ids + padding
                attention_mask = [1] * len(token_ids) + [0] * padding_length
            else:
                padded_ids = padding + token_ids
                attention_mask = [0] * padding_length + [1] * len(token_ids)
        else:
            padded_ids = token_ids
            attention_mask = [1] * len(token_ids)
        
        return padded_ids, attention_mask
    
    @property
    def vocab_info(self) -> Dict:
        """返回词表信息"""
        return {
            'vocab_size': self.vocab_size,
            'id_offset': self.id_offset,
            'original_range': (self.config.original_min_id, self.config.original_max_id),
            'remapped_range': (
                self.config.original_min_id + self.id_offset,
                self.config.original_max_id + self.id_offset
            ),
            'special_tokens': self.special_tokens.to_dict()
        }
    
    def save_pretrained(self, save_dir: str):
        """保存 tokenizer 配置"""
        import json
        from pathlib import Path
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            'tokenizer_config': {
                'original_min_id': self.config.original_min_id,
                'original_max_id': self.config.original_max_id,
                'id_offset': self.config.id_offset,
                'vocab_size': self.config.vocab_size,
            },
            'special_tokens': self.special_tokens.to_dict()
        }
        
        with open(save_path / 'tokenizer_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, load_dir: str) -> 'HATTokenizer':
        """从保存的配置加载 tokenizer"""
        import json
        from pathlib import Path
        
        load_path = Path(load_dir)
        
        with open(load_path / 'tokenizer_config.json', 'r') as f:
            config_dict = json.load(f)
        
        tokenizer_config = TokenizerConfig(**config_dict['tokenizer_config'])
        special_tokens = SpecialTokens()
        
        return cls(config=tokenizer_config, special_tokens=special_tokens)


def create_tokenizer(config: TokenizerConfig = None) -> HATTokenizer:
    """工厂函数：创建 tokenizer 实例"""
    return HATTokenizer(config=config)



