#!/usr/bin/env python3
"""
数据预处理主模块

执行完整的数据预处理流程：
1. 加载原始数据
2. 数据清洗（去重、处理冲突、移除空文本）
3. Token ID 重映射 (+5)
4. 划分训练/验证集 (stratified)
5. 计算类别权重
6. 保存处理后的数据

输出文件结构：
    data/processed/
        train.csv           # text, label
        val.csv             # text, label
        test.csv            # text
        class_weights.npy
        tokenizer/tokenizer_config.json
        preprocessing_report.json
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocess.config import DataConfig, get_default_config
from src.data_preprocess.tokenizer import HATTokenizer, create_tokenizer
from src.data_preprocess.segmenter import DocumentSegmenter, create_segmenter
from src.data_preprocess.data_cleaner import DataCleaner, CleaningReport, TextLengthAnalyzer
from src.data_preprocess.class_balance import (
    ClassWeightCalculator, 
    ClassDistributionAnalyzer,
    compute_class_weights
)
from src.common_config import COMMON_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    数据预处理器
    
    整合所有预处理步骤的主类。
    
    输入数据格式：
        - train_set.csv: text (空格分隔的原始 token ID), label
        - test_a.csv: text (空格分隔的原始 token ID)
    
    输出数据格式：
        - text 列: 空格分隔的重映射后 token ID (原始 ID + 5)
        - label 列: 仅 train/val 有
    """
    
    def __init__(self, config: DataConfig = None):
        self.config = config or get_default_config()
        
        # 初始化组件
        self.tokenizer = create_tokenizer(self.config.tokenizer)
        self.segmenter = create_segmenter(self.config.segmenter)
        self.cleaner = DataCleaner(self.config.cleaning)
        self.weight_calculator = ClassWeightCalculator(self.config.class_balance)
        
        # 处理结果
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.class_weights: Optional[np.ndarray] = None
        self.cleaning_report: Optional[CleaningReport] = None
        
    def run(self, save_results: bool = True) -> Dict:
        """
        执行完整的预处理流程
        
        Args:
            save_results: 是否保存结果到文件
            
        Returns:
            包含处理结果统计的字典
        """
        logger.info("=" * 60)
        logger.info("开始数据预处理")
        logger.info("=" * 60)
        
        results = {}
        
        # Step 1: 加载数据
        logger.info("\nStep 1: 加载原始数据")
        train_raw, test_raw = self._load_data()
        results['raw_train_size'] = len(train_raw)
        results['raw_test_size'] = len(test_raw)
        
        # Step 2: 数据清洗
        logger.info("\nStep 2: 数据清洗")
        train_cleaned, self.cleaning_report = self.cleaner.clean(
            train_raw, 
            text_col='text', 
            label_col='label'
        )
        logger.info(str(self.cleaning_report))
        results['cleaned_train_size'] = len(train_cleaned)
        
        # Step 3: Token ID 重映射
        logger.info("\nStep 3: Token ID 重映射 (+5)")
        train_cleaned = self._remap_tokens(train_cleaned)
        test_remapped = self._remap_tokens(test_raw.copy())
        logger.info(f"  Token ID 范围: 原始 0~7549 -> 映射后 5~7554")
        
        # Step 4: 分析清洗后的数据
        logger.info("\nStep 4: 数据分析")
        self._analyze_data(train_cleaned, test_remapped)
        
        # Step 5: 划分训练/验证集
        logger.info("\nStep 5: 划分训练/验证集 (stratified)")
        self.train_df, self.val_df = self._split_train_val(train_cleaned)
        results['train_size'] = len(self.train_df)
        results['val_size'] = len(self.val_df)
        
        # Step 6: 处理测试集
        self.test_df = test_remapped
        results['test_size'] = len(self.test_df)
        
        # Step 7: 计算类别权重
        logger.info("\nStep 6: 计算类别权重 (inverse_sqrt)")
        self.class_weights = self._compute_class_weights()
        results['class_weights'] = self.class_weights.tolist()
        
        # Step 8: 验证处理结果
        logger.info("\nStep 7: 验证处理结果")
        self._validate_results()
        
        # Step 9: 保存结果
        if save_results:
            logger.info("\nStep 8: 保存处理结果")
            self._save_results(results)
        
        logger.info("\n" + "=" * 60)
        logger.info("数据预处理完成！")
        logger.info(f"  训练集: {results['train_size']:,} 条")
        logger.info(f"  验证集: {results['val_size']:,} 条")
        logger.info(f"  测试集: {results['test_size']:,} 条")
        logger.info("=" * 60)
        
        return results
    
    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载原始数据"""
        train_path = self.config.train_path
        test_path = self.config.test_path
        
        if not train_path.exists():
            raise FileNotFoundError(f"训练集文件不存在: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"测试集文件不存在: {test_path}")
        
        train_df = pd.read_csv(train_path, sep='\t')
        test_df = pd.read_csv(test_path, sep='\t')
        
        logger.info(f"  训练集: {len(train_df):,} 条")
        logger.info(f"  测试集: {len(test_df):,} 条")
        
        return train_df, test_df
    
    def _remap_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对 text 列进行 Token ID 重映射
        
        原始 ID + 5，为特殊 token (0~4) 预留空间
        """
        def remap_text(text: str) -> str:
            if pd.isna(text) or not text.strip():
                return ""
            
            tokens = text.strip().split()
            remapped = []
            for token in tokens:
                try:
                    original_id = int(token)
                    # 重映射: +5
                    remapped_id = original_id + self.tokenizer.id_offset
                    remapped.append(str(remapped_id))
                except ValueError:
                    # 无法解析的 token 跳过
                    continue
            
            return " ".join(remapped)
        
        df = df.copy()
        df['text'] = df['text'].apply(remap_text)
        
        return df
    
    def _analyze_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """分析数据特征"""
        # 类别分布
        logger.info("\n  类别分布:")
        ClassDistributionAnalyzer.print_distribution(
            train_df['label'].tolist(),
            num_classes=self.config.num_labels
        )
        
        # 长度分布
        logger.info("\n  长度分布 (tokens):")
        train_stats = TextLengthAnalyzer.analyze(train_df, 'text')
        test_stats = TextLengthAnalyzer.analyze(test_df, 'text')
        
        logger.info(f"  训练集: mean={train_stats['mean']:.1f}, median={train_stats['median']:.1f}, "
                   f"p90={train_stats['p90']:.1f}, max={train_stats['max']:.0f}")
        logger.info(f"  测试集: mean={test_stats['mean']:.1f}, median={test_stats['median']:.1f}, "
                   f"p90={test_stats['p90']:.1f}, max={test_stats['max']:.0f}")
        
        # 超长文本统计
        train_over_4096 = (train_df['text'].apply(lambda x: len(str(x).split())) > 4096).sum()
        logger.info(f"\n  超过 4096 tokens 的样本: {train_over_4096} ({train_over_4096/len(train_df)*100:.2f}%)")
    
    def _split_train_val(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """划分训练集和验证集 (stratified by label)"""
        train_df, val_df = train_test_split(
            df,
            test_size=self.config.val_split_ratio,
            random_state=self.config.seed,
            stratify=df['label']
        )
        
        logger.info(f"  训练集: {len(train_df):,} 条 ({100*(1-self.config.val_split_ratio):.0f}%)")
        logger.info(f"  验证集: {len(val_df):,} 条 ({100*self.config.val_split_ratio:.0f}%)")
        
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    
    def _compute_class_weights(self) -> np.ndarray:
        """计算类别权重"""
        labels = self.train_df['label'].tolist()
        
        weights = self.weight_calculator.compute_weights(
            labels,
            num_classes=self.config.num_labels
        )
        
        logger.info("  类别权重 (inverse_sqrt):")
        for i, w in enumerate(weights):
            logger.info(f"    Label {i:2d}: {w:.4f}")
        
        return weights
    
    def _validate_results(self):
        """验证处理结果"""
        # 验证 token ID 范围
        def check_token_range(text: str) -> bool:
            if pd.isna(text) or not text.strip():
                return True
            tokens = text.strip().split()
            for token in tokens:
                try:
                    tid = int(token)
                    if tid < 5 or tid >= self.tokenizer.vocab_size:
                        return False
                except ValueError:
                    return False
            return True
        
        # 检查训练集
        train_valid = self.train_df['text'].apply(check_token_range).all()
        val_valid = self.val_df['text'].apply(check_token_range).all()
        test_valid = self.test_df['text'].apply(check_token_range).all()
        
        if train_valid and val_valid and test_valid:
            logger.info("  ✓ 所有样本的 token ID 范围正确 [5, 7554]")
        else:
            logger.warning("  ✗ 存在 token ID 范围异常的样本")
        
        # 验证标签范围
        train_labels_valid = self.train_df['label'].between(0, self.config.num_labels - 1).all()
        val_labels_valid = self.val_df['label'].between(0, self.config.num_labels - 1).all()
        
        if train_labels_valid and val_labels_valid:
            logger.info(f"  ✓ 所有标签在有效范围内 [0, {self.config.num_labels - 1}]")
        else:
            logger.warning("  ✗ 存在标签范围异常的样本")
    
    def _save_results(self, results: Dict):
        """保存处理结果"""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存处理后的数据（CSV 格式，tab 分隔）
        self.train_df.to_csv(output_dir / 'train.csv', sep='\t', index=False)
        self.val_df.to_csv(output_dir / 'val.csv', sep='\t', index=False)
        self.test_df.to_csv(output_dir / 'test.csv', sep='\t', index=False)
        logger.info(f"  数据集已保存到: {output_dir}")
        
        # 保存类别权重
        np.save(output_dir / 'class_weights.npy', self.class_weights)
        logger.info(f"  类别权重已保存: class_weights.npy")
        
        # 保存 tokenizer 配置
        tokenizer_dir = output_dir / 'tokenizer'
        tokenizer_dir.mkdir(exist_ok=True)
        tokenizer_config = {
            'id_offset': self.tokenizer.id_offset,
            'vocab_size': self.tokenizer.vocab_size,
            'pad_token_id': self.tokenizer.pad_token_id,
            'unk_token_id': self.tokenizer.unk_token_id,
            'cls_token_id': self.tokenizer.cls_token_id,
            'sep_token_id': self.tokenizer.sep_token_id,
            'mask_token_id': self.tokenizer.mask_token_id,
        }
        with open(tokenizer_dir / 'tokenizer_config.json', 'w') as f:
            json.dump(tokenizer_config, f, indent=2)
        logger.info(f"  Tokenizer 配置已保存")
        
        # 保存处理报告
        # 转换 numpy 类型为 Python 原生类型，避免 JSON 序列化问题
        def to_native(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        report = {
            'timestamp': timestamp,
            'config': {
                'seed': to_native(self.config.seed),
                'val_split_ratio': to_native(self.config.val_split_ratio),
                'num_labels': to_native(self.config.num_labels),
                'segment_length': to_native(self.config.segmenter.segment_length),
                'max_segments': to_native(self.config.segmenter.max_segments),
                'max_seq_length': to_native(COMMON_CONFIG.max_seq_length),
                'vocab_size': to_native(self.config.tokenizer.vocab_size),
                'id_offset': to_native(self.config.tokenizer.id_offset),
            },
            'statistics': {
                'raw_train_size': to_native(results['raw_train_size']),
                'raw_test_size': to_native(results['raw_test_size']),
                'cleaned_train_size': to_native(results['cleaned_train_size']),
                'train_size': to_native(results['train_size']),
                'val_size': to_native(results['val_size']),
                'test_size': to_native(results['test_size']),
            },
            'cleaning_report': {
                'original_count': to_native(self.cleaning_report.original_count),
                'final_count': to_native(self.cleaning_report.final_count),
                'removed_count': to_native(self.cleaning_report.removed_count),
                'duplicate_count': to_native(self.cleaning_report.duplicate_count),
                'conflict_count': to_native(self.cleaning_report.conflict_count),
                'empty_text_count': to_native(self.cleaning_report.empty_text_count),
                'short_text_count': to_native(self.cleaning_report.short_text_count),
            },
            'class_weights': [to_native(w) for w in results['class_weights']],
        }
        
        with open(output_dir / 'preprocessing_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"  处理报告已保存: preprocessing_report.json")
    
    def get_datasets(self):
        """
        获取处理后的数据集（用于训练）
        
        Returns:
            (train_df, val_df, test_df, class_weights)
        """
        if self.train_df is None:
            raise ValueError("请先运行 run() 方法进行数据预处理")
        
        return self.train_df, self.val_df, self.test_df, self.class_weights


def run_preprocessing(config: DataConfig) -> Dict:
    """
    执行数据预处理流程（便捷函数）
    
    Args:
        config: DataConfig 配置对象
        
    Returns:
        处理结果统计字典
    """
    preprocessor = DataPreprocessor(config)
    return preprocessor.run(save_results=True)


def main():
    """主函数（命令行入口）"""
    import argparse
    
    parser = argparse.ArgumentParser(description='HAT 模型数据预处理')
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default=str(PROJECT_ROOT / 'data'),
        help='数据目录 (default: project_root/data)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=str(PROJECT_ROOT / 'data' / 'processed'),
        help='输出目录 (default: project_root/data/processed)'
    )
    parser.add_argument(
        '--val-ratio', 
        type=float, 
        default=0.1,
        help='验证集比例 (default: 0.1)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='随机种子 (default: 42)'
    )
    parser.add_argument(
        '--no-save', 
        action='store_true',
        help='不保存结果'
    )
    
    args = parser.parse_args()
    
    # 创建配置
    config = get_default_config()
    config.data_dir = Path(args.data_dir)
    config.output_dir = Path(args.output_dir)
    config.val_split_ratio = args.val_ratio
    config.seed = args.seed
    
    # 运行预处理
    preprocessor = DataPreprocessor(config)
    results = preprocessor.run(save_results=not args.no_save)
    
    return results


if __name__ == '__main__':
    main()
