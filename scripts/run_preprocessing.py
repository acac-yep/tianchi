#!/usr/bin/env python3
"""
数据预处理运行脚本

这个脚本演示了完整的数据预处理流程，包括：
1. 加载和清洗数据
2. Token ID 重映射
3. 分段处理示例
4. 类别权重计算
5. 创建 PyTorch DataLoader

使用方法:
    python scripts/run_preprocessing.py
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def demo_tokenizer():
    """演示 Tokenizer 使用"""
    print("\n" + "=" * 60)
    print("1. Tokenizer 演示")
    print("=" * 60)
    
    from src.data import create_tokenizer
    
    tokenizer = create_tokenizer()
    
    # 模拟一段文本
    sample_text = "0 1 2 3 4 100 200 7549"  # 包含与特殊 token 冲突的 ID
    
    print(f"\n原始文本: {sample_text}")
    
    # 编码
    encoded = tokenizer.encode(sample_text)
    print(f"编码后 (ID+5): {encoded}")
    
    # 解码
    decoded = tokenizer.decode(encoded)
    print(f"解码后: {decoded}")
    
    # 词表信息
    print(f"\n词表信息: {tokenizer.vocab_info}")


def demo_segmenter():
    """演示分段器使用"""
    print("\n" + "=" * 60)
    print("2. 分段器演示")
    print("=" * 60)
    
    from src.data import create_tokenizer, create_segmenter
    
    tokenizer = create_tokenizer()
    segmenter = create_segmenter()
    
    # 模拟不同长度的文本
    short_text = " ".join([str(i) for i in range(100)])      # 100 tokens
    medium_text = " ".join([str(i % 1000) for i in range(800)])   # 800 tokens
    long_text = " ".join([str(i % 1000) for i in range(5000)])    # 5000 tokens
    
    for name, text in [("短文本", short_text), ("中等文本", medium_text), ("长文本", long_text)]:
        token_ids = tokenizer.encode(text)
        segmented = segmenter.segment_document(token_ids, mode='train')
        
        print(f"\n{name}:")
        print(f"  原始长度: {segmented.original_length}")
        print(f"  分段数: {segmented.num_segments}")
        print(f"  各段长度: {segmented.segment_lengths}")
        print(f"  是否截断: {segmented.is_truncated}")
        if segmented.is_truncated:
            print(f"  截断策略: {segmented.truncation_strategy}")


def demo_data_cleaning():
    """演示数据清洗"""
    print("\n" + "=" * 60)
    print("3. 数据清洗演示")
    print("=" * 60)
    
    from src.data import DataCleaner, DataCleaningConfig
    
    # 创建模拟数据（包含各种问题）
    data = {
        'text': [
            "100 200 300",           # 正常
            "100 200 300",           # 重复
            "400 500 600",           # 正常
            "400 500 600",           # 相同文本
            "",                       # 空文本
            "1 2",                    # 超短文本
            " ".join([str(i) for i in range(20000)]),  # 极长文本
        ],
        'label': [
            0, 0,  # 完全重复
            1, 2,  # 标签冲突
            0,     # 空文本
            1,     # 超短
            3,     # 极长
        ]
    }
    df = pd.DataFrame(data)
    
    print(f"\n原始数据: {len(df)} 条")
    
    # 配置清洗选项
    config = DataCleaningConfig(
        remove_duplicates=True,
        remove_label_conflicts=True,
        remove_short_texts=False,
        min_text_length=5,
        extreme_long_threshold=10000,
    )
    
    cleaner = DataCleaner(config)
    cleaned_df, report = cleaner.clean(df)
    
    print(report)
    print(f"清洗后数据: {len(cleaned_df)} 条")


def demo_class_balance():
    """演示类别平衡处理"""
    print("\n" + "=" * 60)
    print("4. 类别平衡处理演示")
    print("=" * 60)
    
    from src.data import (
        ClassWeightCalculator, 
        ClassDistributionAnalyzer,
        compute_class_weights
    )
    
    # 模拟不平衡标签（类似真实数据分布）
    labels = (
        [0] * 38918 + [1] * 14996 + [2] * 30000 + [3] * 20000 +
        [4] * 8000 + [5] * 7000 + [6] * 6000 + [7] * 5000 +
        [8] * 4000 + [9] * 3000 + [10] * 2000 + [11] * 1500 +
        [12] * 1000 + [13] * 900
    )
    
    # 分析分布
    ClassDistributionAnalyzer.print_distribution(labels, num_classes=14)
    
    # 计算不同方法的权重
    print("\n不同权重计算方法对比:")
    for method in ['inverse_sqrt', 'inverse_log', 'effective_num']:
        weights = compute_class_weights(labels, num_classes=14, method=method)
        print(f"\n{method}:")
        for i in [0, 6, 13]:  # 展示头部、中部、尾部类别
            print(f"  Label {i}: {weights[i]:.4f}")


def demo_dataset_creation():
    """演示 Dataset 创建（需要真实数据）"""
    print("\n" + "=" * 60)
    print("5. Dataset 创建演示")
    print("=" * 60)
    
    data_path = PROJECT_ROOT / "data" / "train_set.csv"
    
    if not data_path.exists():
        print(f"\n数据文件不存在: {data_path}")
        print("跳过 Dataset 创建演示")
        return
    
    try:
        import torch
        from src.data import (
            HATDataset, 
            HATDataCollator, 
            create_dataloader,
            create_tokenizer,
            create_segmenter
        )
        
        # 加载少量数据用于演示
        df = pd.read_csv(data_path, sep='\t', nrows=100)
        
        tokenizer = create_tokenizer()
        segmenter = create_segmenter()
        
        # 创建 Dataset
        dataset = HATDataset(
            texts=df['text'].tolist(),
            labels=df['label'].tolist(),
            tokenizer=tokenizer,
            segmenter=segmenter,
            mode='train',
            cache_segments=False
        )
        
        print(f"\nDataset 大小: {len(dataset)}")
        
        # 获取一个样本
        sample = dataset[0]
        print(f"\n样本结构:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value}")
        
        # 创建 DataLoader
        collator = HATDataCollator()
        dataloader = create_dataloader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collator=collator
        )
        
        # 获取一个 batch
        batch = next(iter(dataloader))
        print(f"\nBatch 结构:")
        for key, value in batch.items():
            print(f"  {key}: shape={value.shape}")
        
    except ImportError:
        print("\n需要安装 PyTorch: pip install torch")
        print("跳过 Dataset 创建演示")


def run_full_preprocessing():
    """运行完整的预处理流程"""
    print("\n" + "=" * 60)
    print("6. 完整预处理流程")
    print("=" * 60)
    
    from src.data.preprocess import DataPreprocessor
    from src.data import get_default_config
    
    config = get_default_config()
    
    # 检查数据是否存在
    if not config.train_path.exists():
        print(f"\n数据文件不存在: {config.train_path}")
        print("请确保数据文件位于正确位置")
        return
    
    # 运行预处理
    preprocessor = DataPreprocessor(config)
    results = preprocessor.run(save_results=True)
    
    print("\n处理完成！")
    print(f"训练集: {results['train_size']} 条")
    print(f"验证集: {results['val_size']} 条")


def main():
    """主函数"""
    print("=" * 60)
    print("HAT 模型数据预处理演示")
    print("=" * 60)
    
    # 1. Tokenizer 演示
    demo_tokenizer()
    
    # 2. 分段器演示
    demo_segmenter()
    
    # 3. 数据清洗演示
    demo_data_cleaning()
    
    # 4. 类别平衡演示
    demo_class_balance()
    
    # 5. Dataset 创建演示
    demo_dataset_creation()
    
    # 6. 询问是否运行完整预处理
    print("\n" + "=" * 60)
    response = input("是否运行完整的数据预处理流程？(y/n): ").strip().lower()
    if response == 'y':
        run_full_preprocessing()
    else:
        print("已跳过完整预处理流程")
    
    print("\n演示完成！")


if __name__ == '__main__':
    main()



