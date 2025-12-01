"""
数据清洗模块
处理重复样本、标签冲突、异常样本等数据质量问题
"""

from typing import List, Tuple, Dict, Optional, Set
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import logging

from .config import DataCleaningConfig, DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    """数据清洗报告"""
    original_count: int
    final_count: int
    removed_count: int
    
    # 各类问题统计
    duplicate_count: int
    conflict_count: int
    short_text_count: int
    empty_text_count: int
    extreme_long_count: int
    
    # 冲突样本详情
    conflict_details: List[Dict]
    
    def __str__(self):
        return f"""
========== 数据清洗报告 ==========
原始样本数: {self.original_count:,}
清洗后样本数: {self.final_count:,}
移除样本数: {self.removed_count:,} ({self.removed_count/self.original_count*100:.2f}%)

问题样本统计:
  - 完全重复样本: {self.duplicate_count:,}
  - 标签冲突样本: {self.conflict_count:,}
  - 空文本样本: {self.empty_text_count:,}
  - 超短文本样本: {self.short_text_count:,}
  - 极长文本样本: {self.extreme_long_count:,}
=====================================
"""


class DataCleaner:
    """
    数据清洗器
    
    主要功能：
    1. 检测和移除完全重复的样本
    2. 检测和处理标签冲突（同一文本不同标签）
    3. 检测空文本和超短文本
    4. 检测极长文本
    5. 生成清洗报告
    """
    
    def __init__(self, config: DataCleaningConfig = None):
        self.config = config or DEFAULT_CONFIG.cleaning
        
    def clean(
        self, 
        df: pd.DataFrame, 
        text_col: str = 'text', 
        label_col: str = 'label',
        return_report: bool = True
    ) -> Tuple[pd.DataFrame, Optional[CleaningReport]]:
        """
        执行数据清洗
        
        Args:
            df: 输入 DataFrame
            text_col: 文本列名
            label_col: 标签列名
            return_report: 是否返回清洗报告
            
        Returns:
            (cleaned_df, report)
        """
        logger.info("开始数据清洗...")
        original_count = len(df)
        
        # 创建工作副本
        df = df.copy()
        
        # 添加文本哈希列用于快速比较
        df['_text_hash'] = df[text_col].apply(self._hash_text)
        
        # 添加文本长度列
        df['_text_length'] = df[text_col].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        
        # 初始化掩码（True = 保留）
        keep_mask = pd.Series([True] * len(df), index=df.index)
        
        # 1. 检测空文本
        empty_mask = df[text_col].isna() | (df[text_col].astype(str).str.strip() == '')
        empty_count = empty_mask.sum()
        logger.info(f"  空文本样本: {empty_count}")
        keep_mask &= ~empty_mask
        
        # 2. 检测超短文本
        short_mask = df['_text_length'] < self.config.min_text_length
        short_count = short_mask.sum()
        logger.info(f"  超短文本样本 (< {self.config.min_text_length} tokens): {short_count}")
        if self.config.remove_short_texts:
            keep_mask &= ~short_mask
        
        # 3. 检测极长文本
        extreme_long_mask = df['_text_length'] > self.config.extreme_long_threshold
        extreme_long_count = extreme_long_mask.sum()
        logger.info(f"  极长文本样本 (> {self.config.extreme_long_threshold} tokens): {extreme_long_count}")
        
        # 4. 检测完全重复样本
        duplicate_mask, duplicate_count = self._find_duplicates(df, text_col, label_col)
        logger.info(f"  完全重复样本: {duplicate_count}")
        if self.config.remove_duplicates:
            keep_mask &= ~duplicate_mask
        
        # 5. 检测标签冲突样本
        conflict_mask, conflict_count, conflict_details = self._find_label_conflicts(
            df, text_col, label_col
        )
        logger.info(f"  标签冲突样本组: {len(conflict_details)}, 涉及样本数: {conflict_count}")
        if self.config.remove_label_conflicts:
            keep_mask &= ~conflict_mask
        
        # 应用掩码
        cleaned_df = df[keep_mask].copy()
        
        # 清理临时列
        cleaned_df = cleaned_df.drop(columns=['_text_hash', '_text_length'])
        
        # 重置索引
        cleaned_df = cleaned_df.reset_index(drop=True)
        
        final_count = len(cleaned_df)
        removed_count = original_count - final_count
        
        logger.info(f"清洗完成: {original_count} -> {final_count} (移除 {removed_count})")
        
        if return_report:
            report = CleaningReport(
                original_count=original_count,
                final_count=final_count,
                removed_count=removed_count,
                duplicate_count=duplicate_count,
                conflict_count=conflict_count,
                short_text_count=short_count,
                empty_text_count=empty_count,
                extreme_long_count=extreme_long_count,
                conflict_details=conflict_details
            )
            return cleaned_df, report
        
        return cleaned_df, None
    
    def _hash_text(self, text) -> str:
        """计算文本哈希"""
        if pd.isna(text):
            return ''
        text_str = str(text).strip()
        return hashlib.md5(text_str.encode()).hexdigest()
    
    def _find_duplicates(
        self, 
        df: pd.DataFrame, 
        text_col: str, 
        label_col: str
    ) -> Tuple[pd.Series, int]:
        """
        查找完全重复的样本（text 和 label 都相同）
        
        Returns:
            (duplicate_mask, count) - 掩码标记需要移除的重复项（保留第一个）
        """
        # 标记重复项（保留第一个）
        duplicate_mask = df.duplicated(subset=[text_col, label_col], keep='first')
        count = duplicate_mask.sum()
        return duplicate_mask, count
    
    def _find_label_conflicts(
        self, 
        df: pd.DataFrame, 
        text_col: str, 
        label_col: str
    ) -> Tuple[pd.Series, int, List[Dict]]:
        """
        查找标签冲突样本（同一文本对应不同标签）
        
        Returns:
            (conflict_mask, count, details)
        """
        # 按文本分组，检查标签数量
        text_groups = df.groupby('_text_hash')[label_col].nunique()
        conflict_hashes = set(text_groups[text_groups > 1].index)
        
        # 创建冲突掩码
        conflict_mask = df['_text_hash'].isin(conflict_hashes)
        count = conflict_mask.sum()
        
        # 收集冲突详情
        details = []
        for text_hash in conflict_hashes:
            conflict_rows = df[df['_text_hash'] == text_hash]
            labels = conflict_rows[label_col].unique().tolist()
            sample_text = str(conflict_rows.iloc[0][text_col])[:100] + '...'
            
            details.append({
                'text_hash': text_hash,
                'labels': labels,
                'count': len(conflict_rows),
                'sample_text': sample_text
            })
        
        return conflict_mask, count, details
    
    def analyze_conflicts(
        self, 
        df: pd.DataFrame, 
        text_col: str = 'text', 
        label_col: str = 'label'
    ) -> pd.DataFrame:
        """
        详细分析标签冲突样本
        
        Returns:
            包含所有冲突样本的 DataFrame
        """
        df = df.copy()
        df['_text_hash'] = df[text_col].apply(self._hash_text)
        
        text_groups = df.groupby('_text_hash')[label_col].nunique()
        conflict_hashes = set(text_groups[text_groups > 1].index)
        
        conflict_df = df[df['_text_hash'].isin(conflict_hashes)].copy()
        conflict_df = conflict_df.drop(columns=['_text_hash'])
        
        return conflict_df
    
    def get_extreme_long_samples(
        self, 
        df: pd.DataFrame, 
        text_col: str = 'text',
        threshold: Optional[int] = None
    ) -> pd.DataFrame:
        """
        获取极长文本样本
        
        Returns:
            极长样本的 DataFrame
        """
        threshold = threshold or self.config.extreme_long_threshold
        
        df = df.copy()
        df['_text_length'] = df[text_col].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        
        extreme_df = df[df['_text_length'] > threshold].copy()
        extreme_df = extreme_df.sort_values('_text_length', ascending=False)
        
        return extreme_df


class TextLengthAnalyzer:
    """文本长度分析器"""
    
    @staticmethod
    def analyze(
        df: pd.DataFrame, 
        text_col: str = 'text'
    ) -> Dict:
        """
        分析文本长度分布
        
        Returns:
            包含各种统计指标的字典
        """
        lengths = df[text_col].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        
        stats = {
            'count': len(lengths),
            'mean': lengths.mean(),
            'std': lengths.std(),
            'min': lengths.min(),
            'p25': lengths.quantile(0.25),
            'median': lengths.median(),
            'p75': lengths.quantile(0.75),
            'p90': lengths.quantile(0.90),
            'p95': lengths.quantile(0.95),
            'p99': lengths.quantile(0.99),
            'max': lengths.max(),
        }
        
        # 计算各阈值的覆盖率
        thresholds = [256, 512, 1024, 2048, 4096, 8192]
        for t in thresholds:
            stats[f'coverage_{t}'] = (lengths <= t).mean() * 100
        
        return stats
    
    @staticmethod
    def get_length_distribution(
        df: pd.DataFrame, 
        text_col: str = 'text',
        bins: List[int] = None
    ) -> pd.DataFrame:
        """
        获取长度分布统计
        
        Returns:
            长度区间统计 DataFrame
        """
        if bins is None:
            bins = [0, 256, 512, 1024, 2048, 4096, 8192, float('inf')]
        
        lengths = df[text_col].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        
        labels = []
        for i in range(len(bins) - 1):
            if bins[i+1] == float('inf'):
                labels.append(f'>{bins[i]}')
            else:
                labels.append(f'{bins[i]}-{bins[i+1]}')
        
        df_result = pd.DataFrame({
            'length_range': pd.cut(lengths, bins=bins, labels=labels, right=True)
        })
        
        dist = df_result['length_range'].value_counts().sort_index()
        
        return pd.DataFrame({
            'range': dist.index,
            'count': dist.values,
            'percentage': dist.values / len(df) * 100
        })


def clean_dataset(
    df: pd.DataFrame,
    config: DataCleaningConfig = None,
    **kwargs
) -> Tuple[pd.DataFrame, CleaningReport]:
    """
    便捷函数：清洗数据集
    
    Args:
        df: 输入 DataFrame
        config: 清洗配置
        **kwargs: 传递给 DataCleaner.clean() 的其他参数
        
    Returns:
        (cleaned_df, report)
    """
    cleaner = DataCleaner(config)
    return cleaner.clean(df, **kwargs)



