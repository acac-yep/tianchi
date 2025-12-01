#!/usr/bin/env python3
"""
数据预处理运行脚本

从官方数据生成 HAT 模型可直接使用的预处理结果：

输入:
    data/train_set.csv  # text (空格分隔的原始 token ID), label
    data/test_a.csv     # text (空格分隔的原始 token ID)

输出:
    data/processed/
        train.csv           # text (重映射后 ID), label
        val.csv             # text (重映射后 ID), label
        test.csv            # text (重映射后 ID)
        class_weights.npy   # 类别权重 (inverse_sqrt)
        tokenizer/tokenizer_config.json
        preprocessing_report.json

使用方法:
    python scripts/run_preprocessing.py
    python scripts/run_preprocessing.py --val-ratio 0.1 --seed 42
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='HAT 模型数据预处理脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(PROJECT_ROOT / 'data'),
        help='数据目录，包含 train_set.csv 和 test_a.csv'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'processed'),
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
        help='不保存结果（仅用于测试）'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 导入依赖（放在这里以便在解析参数后导入）
    from src.data_preprocess.config import (
        DataConfig,
        TokenizerConfig,
        SegmenterConfig,
        DataCleaningConfig,
        ClassBalanceConfig,
    )
    from src.data_preprocess.preprocess import run_preprocessing
    from src.common_config import COMMON_CONFIG
    
    # 构造路径
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    train_path = data_dir / 'train_set.csv'
    test_path = data_dir / 'test_a.csv'
    
    # 检查输入文件是否存在
    if not train_path.exists():
        print(f"错误: 训练集文件不存在: {train_path}")
        print(f"请确保数据文件位于正确位置")
        sys.exit(1)
    
    if not test_path.exists():
        print(f"错误: 测试集文件不存在: {test_path}")
        print(f"请确保数据文件位于正确位置")
        sys.exit(1)
    
    print("=" * 60)
    print("HAT 模型数据预处理")
    print("=" * 60)
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"验证集比例: {args.val_ratio}")
    print(f"随机种子: {args.seed}")
    print("=" * 60)
    
    # 构造配置
    config = DataConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        train_file='train_set.csv',
        test_file='test_a.csv',
        tokenizer=TokenizerConfig(
            id_offset=COMMON_CONFIG.id_offset,
            vocab_size=COMMON_CONFIG.vocab_size,
            original_min_id=COMMON_CONFIG.original_min_token_id,
            original_max_id=COMMON_CONFIG.original_max_token_id,
        ),
        segmenter=SegmenterConfig(
            segment_length=COMMON_CONFIG.segment_length,
            max_segments=COMMON_CONFIG.max_segments,
            max_seq_length=COMMON_CONFIG.max_seq_length,
            tail_pullback_threshold=0.5,
            train_long_strategy='random_window',
            infer_long_strategy='sliding_window',
            sliding_window_stride=4,
        ),
        cleaning=DataCleaningConfig(
            remove_duplicates=True,
            remove_label_conflicts=True,
            remove_short_texts=False,
            min_text_length=5,
            extreme_long_threshold=10000,
        ),
        class_balance=ClassBalanceConfig(
            weight_method='inverse_sqrt',
            use_weighted_sampler=False,
        ),
        num_labels=COMMON_CONFIG.num_labels,
        val_split_ratio=args.val_ratio,
        seed=args.seed,
    )
    
    # 运行预处理
    from src.data_preprocess.preprocess import DataPreprocessor
    
    preprocessor = DataPreprocessor(config)
    results = preprocessor.run(save_results=not args.no_save)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("预处理完成！")
    print("=" * 60)
    print(f"训练集: {results['train_size']:,} 条")
    print(f"验证集: {results['val_size']:,} 条")
    print(f"测试集: {results['test_size']:,} 条")
    
    if not args.no_save:
        print(f"\n输出文件:")
        print(f"  - {output_dir / 'train.csv'}")
        print(f"  - {output_dir / 'val.csv'}")
        print(f"  - {output_dir / 'test.csv'}")
        print(f"  - {output_dir / 'class_weights.npy'}")
        print(f"  - {output_dir / 'tokenizer' / 'tokenizer_config.json'}")
        print(f"  - {output_dir / 'preprocessing_report.json'}")
    
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
