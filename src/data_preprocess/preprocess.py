#!/usr/bin/env python3
"""
数据预处理主脚本

执行完整的数据预处理流程：
1. 加载原始数据
2. 数据清洗（去重、处理冲突）
3. Token ID 重映射
4. 划分训练/验证集
5. 计算类别权重
6. 保存处理后的数据

使用方法：
    python -m src.data.preprocess --config config.yaml
    或
    python src/data/preprocess.py
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import DataConfig, get_default_config
from src.data.tokenizer import HATTokenizer, create_tokenizer
from src.data.segmenter import DocumentSegmenter, create_segmenter
from src.data.data_cleaner import DataCleaner, CleaningReport, TextLengthAnalyzer
from src.data.class_balance import (
    ClassWeightCalculator, 
    ClassDistributionAnalyzer,
    compute_class_weights
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    数据预处理器
    
    整合所有预处理步骤的主类
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
        
        # Step 3: 分析清洗后的数据
        logger.info("\nStep 3: 数据分析")
        self._analyze_data(train_cleaned, test_raw)
        
        # Step 4: 划分训练/验证集
        logger.info("\nStep 4: 划分训练/验证集")
        self.train_df, self.val_df = self._split_train_val(train_cleaned)
        results['train_size'] = len(self.train_df)
        results['val_size'] = len(self.val_df)
        
        # Step 5: 处理测试集
        self.test_df = test_raw.copy()
        
        # Step 6: 计算类别权重
        logger.info("\nStep 5: 计算类别权重")
        self.class_weights = self._compute_class_weights()
        results['class_weights'] = self.class_weights.tolist()
        
        # Step 7: Token 重映射验证
        logger.info("\nStep 6: Token 重映射验证")
        self._validate_tokenization()
        
        # Step 8: 保存结果
        if save_results:
            logger.info("\nStep 7: 保存处理结果")
            self._save_results(results)
        
        logger.info("\n" + "=" * 60)
        logger.info("数据预处理完成！")
        logger.info("=" * 60)
        
        return results
    
    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载原始数据"""
        train_df = pd.read_csv(self.config.train_path, sep='\t')
        test_df = pd.read_csv(self.config.test_path, sep='\t')
        
        logger.info(f"  训练集: {len(train_df):,} 条")
        logger.info(f"  测试集: {len(test_df):,} 条")
        
        return train_df, test_df
    
    def _analyze_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """分析数据特征"""
        # 类别分布
        logger.info("\n  类别分布:")
        ClassDistributionAnalyzer.print_distribution(
            train_df['label'].tolist(),
            num_classes=self.config.num_labels
        )
        
        # 长度分布
        logger.info("\n  长度分布:")
        train_stats = TextLengthAnalyzer.analyze(train_df, 'text')
        test_stats = TextLengthAnalyzer.analyze(test_df, 'text')
        
        logger.info(f"  训练集: mean={train_stats['mean']:.1f}, median={train_stats['median']:.1f}, "
                   f"p90={train_stats['p90']:.1f}, max={train_stats['max']:.0f}")
        logger.info(f"  测试集: mean={test_stats['mean']:.1f}, median={test_stats['median']:.1f}, "
                   f"p90={test_stats['p90']:.1f}, max={test_stats['max']:.0f}")
    
    def _split_train_val(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """划分训练集和验证集"""
        train_df, val_df = train_test_split(
            df,
            test_size=self.config.val_split_ratio,
            random_state=self.config.seed,
            stratify=df['label']
        )
        
        logger.info(f"  训练集: {len(train_df):,} 条")
        logger.info(f"  验证集: {len(val_df):,} 条")
        
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    
    def _compute_class_weights(self) -> np.ndarray:
        """计算类别权重"""
        labels = self.train_df['label'].tolist()
        
        weights = self.weight_calculator.compute_weights(
            labels,
            num_classes=self.config.num_labels
        )
        
        logger.info("  类别权重:")
        for i, w in enumerate(weights):
            logger.info(f"    Label {i}: {w:.4f}")
        
        return weights
    
    def _validate_tokenization(self):
        """验证 token 处理"""
        # 随机采样几个样本进行验证
        sample_texts = self.train_df['text'].sample(3, random_state=42).tolist()
        
        logger.info("  Token 重映射验证:")
        for i, text in enumerate(sample_texts[:1]):  # 只显示第一个
            # 原始 tokens
            original_tokens = text.split()[:10]
            logger.info(f"    原始 tokens (前10): {original_tokens}")
            
            # 重映射后
            remapped = self.tokenizer.encode(text)[:10]
            logger.info(f"    重映射后 (前10): {remapped}")
            
            # 还原验证
            decoded = self.tokenizer.decode(remapped)
            logger.info(f"    还原后: {decoded}")
        
        # 验证词表信息
        vocab_info = self.tokenizer.vocab_info
        logger.info(f"\n  词表信息:")
        logger.info(f"    词表大小: {vocab_info['vocab_size']}")
        logger.info(f"    ID 偏移: {vocab_info['id_offset']}")
        logger.info(f"    原始范围: {vocab_info['original_range']}")
        logger.info(f"    映射后范围: {vocab_info['remapped_range']}")
    
    def _save_results(self, results: Dict):
        """保存处理结果"""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存处理后的数据
        self.train_df.to_csv(output_dir / 'train.csv', sep='\t', index=False)
        self.val_df.to_csv(output_dir / 'val.csv', sep='\t', index=False)
        self.test_df.to_csv(output_dir / 'test.csv', sep='\t', index=False)
        logger.info(f"  数据集已保存到: {output_dir}")
        
        # 保存类别权重
        np.save(output_dir / 'class_weights.npy', self.class_weights)
        logger.info(f"  类别权重已保存")
        
        # 保存 tokenizer 配置
        self.tokenizer.save_pretrained(output_dir / 'tokenizer')
        logger.info(f"  Tokenizer 配置已保存")
        
        # 保存处理报告
        report = {
            'timestamp': timestamp,
            'config': {
                'seed': self.config.seed,
                'val_split_ratio': self.config.val_split_ratio,
                'num_labels': self.config.num_labels,
                'segment_length': self.config.segmenter.segment_length,
                'max_segments': self.config.segmenter.max_segments,
                'vocab_size': self.config.tokenizer.vocab_size,
                'id_offset': self.config.tokenizer.id_offset,
            },
            'statistics': results,
            'cleaning_report': {
                'original_count': self.cleaning_report.original_count,
                'final_count': self.cleaning_report.final_count,
                'removed_count': self.cleaning_report.removed_count,
                'duplicate_count': self.cleaning_report.duplicate_count,
                'conflict_count': self.cleaning_report.conflict_count,
            }
        }
        
        with open(output_dir / 'preprocessing_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"  处理报告已保存")
    
    def get_datasets(self):
        """
        获取处理后的数据集（用于训练）
        
        Returns:
            (train_df, val_df, test_df, class_weights)
        """
        if self.train_df is None:
            raise ValueError("请先运行 run() 方法进行数据预处理")
        
        return self.train_df, self.val_df, self.test_df, self.class_weights


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='HAT 模型数据预处理')
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='/home/byhx/workspace/tianchi/data',
        help='数据目录'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='/home/byhx/workspace/tianchi/data/processed',
        help='输出目录'
    )
    parser.add_argument(
        '--val-ratio', 
        type=float, 
        default=0.1,
        help='验证集比例'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='随机种子'
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



