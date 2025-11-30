#!/usr/bin/env python3
"""
数据集特征分析脚本（内存优化版）
分析训练集和测试集的各项统计指标，为模型选型提供依据
"""

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
import warnings
import gc
warnings.filterwarnings('ignore')

# =============================================================================
# 配置
# =============================================================================
DATA_DIR = Path("/home/byhx/workspace/tianchi/data")
TRAIN_PATH = DATA_DIR / "train_set.csv"
TEST_PATH = DATA_DIR / "test_a.csv"

# 特殊 token 预留 ID（可根据实际情况修改）
SPECIAL_TOKENS = {
    'PAD': 0,
    'UNK': 1,
    'CLS': 2,
    'SEP': 3,
    'MASK': 4,
}

# 截断阈值
TRUNCATION_THRESHOLDS = [256, 512, 1024, 2048, 4096, 8192]


def load_data():
    """加载数据集"""
    print("=" * 80)
    print("正在加载数据...")
    print("=" * 80)
    
    train_df = pd.read_csv(TRAIN_PATH, sep='\t')
    test_df = pd.read_csv(TEST_PATH, sep='\t')
    
    print(f"训练集加载完成: {len(train_df)} 条样本")
    print(f"测试集加载完成: {len(test_df)} 条样本")
    
    return train_df, test_df


def compute_length_stats(lengths):
    """计算长度统计指标"""
    lengths = np.array(lengths)
    stats = {
        'count': len(lengths),
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'P25': np.percentile(lengths, 25),
        'P50 (median)': np.percentile(lengths, 50),
        'P75': np.percentile(lengths, 75),
        'P90': np.percentile(lengths, 90),
        'P95': np.percentile(lengths, 95),
        'P99': np.percentile(lengths, 99),
        'max': np.max(lengths),
    }
    return stats


def print_length_stats(stats, name=""):
    """打印长度统计"""
    print(f"\n{name} 长度统计:")
    print("-" * 50)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:15s}: {value:,.2f}")
        else:
            print(f"  {key:15s}: {value:,}")


def analyze_label_distribution(train_df):
    """分析标签分布"""
    print("\n" + "=" * 80)
    print("一、数据规模 & 标签结构")
    print("=" * 80)
    
    # 1.1 样本量
    print(f"\n1.1 样本量")
    print(f"  训练集样本数: {len(train_df):,}")
    
    # 1.2 标签信息
    print(f"\n1.2 标签信息")
    label_counts = train_df['label'].value_counts().sort_index()
    num_labels = len(label_counts)
    print(f"  类别数 (num_labels): {num_labels}")
    
    print(f"\n  各类别样本数和占比:")
    print(f"  {'Label':<8} {'Count':>10} {'Ratio':>10}")
    print("  " + "-" * 30)
    
    total = len(train_df)
    for label, count in label_counts.items():
        ratio = count / total * 100
        print(f"  {label:<8} {count:>10,} {ratio:>9.2f}%")
    
    # 类别不平衡比
    max_count = label_counts.max()
    min_count = label_counts.min()
    imbalance_ratio = max_count / min_count
    print(f"\n  最多类样本数: {max_count:,} (label {label_counts.idxmax()})")
    print(f"  最少类样本数: {min_count:,} (label {label_counts.idxmin()})")
    print(f"  不平衡比 (max/min): {imbalance_ratio:.2f}:1")
    
    # 1.3 是否多标签
    print(f"\n1.3 是否多标签任务")
    # 检查是否有多个标签（用逗号或空格分隔）
    multi_label_count = train_df['label'].astype(str).str.contains(r'[,\s]', regex=True).sum()
    if multi_label_count > 0:
        print(f"  发现 {multi_label_count} 条可能包含多标签的样本")
    else:
        print(f"  这是单标签多类分类任务（每个样本只有一个标签）")
    
    return label_counts


def compute_lengths_chunked(df, text_col='text', chunk_size=10000):
    """分块计算文本长度，节省内存"""
    lengths = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size][text_col]
        chunk_lengths = chunk.apply(lambda x: len(str(x).split()))
        lengths.extend(chunk_lengths.tolist())
        gc.collect()
    return np.array(lengths)


def analyze_length_distribution(train_df, test_df):
    """分析文本长度分布"""
    print("\n" + "=" * 80)
    print("二、文本长度分布")
    print("=" * 80)
    
    # 计算长度（分块处理）
    print("\n正在计算文本长度...")
    train_lengths = compute_lengths_chunked(train_df)
    test_lengths = compute_lengths_chunked(test_df)
    
    # 2.1 整体长度分布
    print("\n2.1 整体长度分布")
    train_stats = compute_length_stats(train_lengths)
    test_stats = compute_length_stats(test_lengths)
    
    print_length_stats(train_stats, "训练集")
    print_length_stats(test_stats, "测试集")
    
    # 2.2 不同阈值的覆盖率
    print("\n2.2 不同阈值的样本覆盖率")
    print("-" * 60)
    print(f"  {'阈值':<10} {'训练集覆盖率':>15} {'测试集覆盖率':>15}")
    print("  " + "-" * 45)
    
    for threshold in TRUNCATION_THRESHOLDS:
        train_coverage = (train_lengths <= threshold).mean() * 100
        test_coverage = (test_lengths <= threshold).mean() * 100
        print(f"  ≤{threshold:<8} {train_coverage:>14.2f}% {test_coverage:>14.2f}%")
    
    # 2.3 Token 级别覆盖率（截断时保留的 token 比例）
    print("\n2.3 Token 级别覆盖率（截断时保留的 token 比例）")
    print("-" * 60)
    total_train_tokens = train_lengths.sum()
    total_test_tokens = test_lengths.sum()
    
    print(f"  训练集总 token 数: {total_train_tokens:,}")
    print(f"  测试集总 token 数: {total_test_tokens:,}")
    print()
    print(f"  {'截断阈值':<10} {'训练集Token覆盖率':>18} {'测试集Token覆盖率':>18}")
    print("  " + "-" * 50)
    
    for threshold in TRUNCATION_THRESHOLDS:
        train_retained = np.minimum(train_lengths, threshold).sum()
        test_retained = np.minimum(test_lengths, threshold).sum()
        train_token_coverage = train_retained / total_train_tokens * 100
        test_token_coverage = test_retained / total_test_tokens * 100
        print(f"  截断@{threshold:<6} {train_token_coverage:>17.2f}% {test_token_coverage:>17.2f}%")
    
    # 2.4 按类别的长度分布
    print("\n2.4 按类别的长度分布（训练集）")
    print("-" * 80)
    print(f"  {'Label':<8} {'Mean':>10} {'Median':>10} {'P90':>10} {'P95':>10} {'Max':>10}")
    print("  " + "-" * 65)
    
    # 临时添加长度列
    train_df['_length'] = train_lengths
    for label in sorted(train_df['label'].unique()):
        label_lengths = train_df[train_df['label'] == label]['_length'].values
        print(f"  {label:<8} {np.mean(label_lengths):>10.1f} {np.median(label_lengths):>10.1f} "
              f"{np.percentile(label_lengths, 90):>10.1f} {np.percentile(label_lengths, 95):>10.1f} "
              f"{np.max(label_lengths):>10}")
    
    return train_lengths, test_lengths


def analyze_token_statistics_streaming(train_df, test_df):
    """流式分析 token 统计，节省内存"""
    print("\n" + "=" * 80)
    print("三、Token 层面的统计")
    print("=" * 80)
    
    print("\n正在统计 token（流式处理）...")
    
    # 流式收集 token 统计
    train_token_counts = Counter()
    train_vocab = set()
    
    for i, text in enumerate(train_df['text']):
        tokens = [int(t) for t in str(text).split()]
        train_token_counts.update(tokens)
        train_vocab.update(tokens)
        if (i + 1) % 50000 == 0:
            print(f"  处理训练集: {i+1}/{len(train_df)}")
    
    test_token_counts = Counter()
    test_vocab = set()
    
    for i, text in enumerate(test_df['text']):
        tokens = [int(t) for t in str(text).split()]
        test_token_counts.update(tokens)
        test_vocab.update(tokens)
        if (i + 1) % 50000 == 0:
            print(f"  处理测试集: {i+1}/{len(test_df)}")
    
    total_vocab = train_vocab | test_vocab
    
    # 3.1 Vocab 信息
    print("\n3.1 Vocab 信息")
    print(f"  训练集不同 token 数 (vocab_size_train): {len(train_vocab):,}")
    print(f"  测试集不同 token 数 (vocab_size_test): {len(test_vocab):,}")
    print(f"  总不同 token 数 (vocab_size_total): {len(total_vocab):,}")
    print(f"  Token 最小 ID: {min(total_vocab)}")
    print(f"  Token 最大 ID: {max(total_vocab)}")
    
    # 3.2 频率分布
    print("\n3.2 频率分布")
    
    print(f"\n  训练集 Top 50 高频 token:")
    print(f"  {'Token':>8} {'Count':>12} {'Ratio':>10}")
    print("  " + "-" * 35)
    total_train = sum(train_token_counts.values())
    for token, count in train_token_counts.most_common(50):
        ratio = count / total_train * 100
        print(f"  {token:>8} {count:>12,} {ratio:>9.4f}%")
    
    # 低频 token 统计
    print(f"\n  低频 token 统计（训练集）:")
    hapax_count = sum(1 for count in train_token_counts.values() if count == 1)
    rare_count_5 = sum(1 for count in train_token_counts.values() if count <= 5)
    rare_count_10 = sum(1 for count in train_token_counts.values() if count <= 10)
    
    print(f"  出现次数 = 1 的 token 数: {hapax_count:,} ({hapax_count/len(train_vocab)*100:.2f}% of vocab)")
    print(f"  出现次数 ≤ 5 的 token 数: {rare_count_5:,} ({rare_count_5/len(train_vocab)*100:.2f}% of vocab)")
    print(f"  出现次数 ≤ 10 的 token 数: {rare_count_10:,} ({rare_count_10/len(train_vocab)*100:.2f}% of vocab)")
    
    # 3.3 特殊 token 预留空间
    print("\n3.3 特殊 token 预留空间")
    print(f"  建议预留的特殊 token ID:")
    for name, token_id in SPECIAL_TOKENS.items():
        in_data = "⚠️ 在数据中出现" if token_id in total_vocab else "✓ 未在数据中出现"
        print(f"    {name}: {token_id} - {in_data}")
    
    # 检查最小 token ID，建议偏移
    min_token = min(total_vocab)
    if min_token <= max(SPECIAL_TOKENS.values()):
        suggested_offset = max(SPECIAL_TOKENS.values()) + 1
        print(f"\n  ⚠️ 注意: 数据中最小 token ID ({min_token}) 与特殊 token 冲突")
        print(f"  建议: 将所有 token ID 偏移 +{suggested_offset}")
    
    return train_vocab, test_vocab, train_token_counts, test_token_counts


def analyze_data_quality(train_df, test_df, train_lengths, test_lengths):
    """分析数据质量和噪声情况"""
    print("\n" + "=" * 80)
    print("四、数据质量 & 噪声情况")
    print("=" * 80)
    
    # 4.1 缺失/异常样本
    print("\n4.1 缺失/异常样本")
    
    # 空文本
    train_empty = train_df['text'].isna() | (train_df['text'].astype(str).str.strip() == '')
    test_empty = test_df['text'].isna() | (test_df['text'].astype(str).str.strip() == '')
    print(f"  训练集空文本数: {train_empty.sum()}")
    print(f"  测试集空文本数: {test_empty.sum()}")
    
    # 超短文本
    print(f"\n  超短文本统计:")
    for threshold in [5, 10, 20]:
        train_short = (train_lengths < threshold).sum()
        test_short = (test_lengths < threshold).sum()
        print(f"    长度 < {threshold}: 训练集 {train_short:,} 条, 测试集 {test_short:,} 条")
    
    # 超短文本的标签分布（使用临时列）
    train_df['_length'] = train_lengths
    very_short = train_df[train_df['_length'] < 10]
    if len(very_short) > 0:
        print(f"\n  超短文本 (length < 10) 的标签分布:")
        short_label_dist = very_short['label'].value_counts().sort_index()
        for label, count in short_label_dist.items():
            print(f"    Label {label}: {count} 条")
    
    # 4.2 重复样本
    print("\n4.2 重复样本")
    
    # 完全重复
    train_duplicates = train_df.duplicated(subset=['text', 'label'], keep=False)
    num_duplicate_pairs = train_duplicates.sum()
    unique_duplicated = train_df[train_duplicates].drop_duplicates(subset=['text', 'label'])
    print(f"  训练集完全重复的样本数: {num_duplicate_pairs:,}")
    print(f"  去重后剩余的重复组数: {len(unique_duplicated):,}")
    
    # text 相同但 label 不同（采样检查）
    print("  正在检查标签冲突...")
    text_groups = train_df.groupby('text')['label'].nunique()
    conflict_texts = text_groups[text_groups > 1]
    print(f"  text 相同但 label 不同的冲突样本组数: {len(conflict_texts):,}")
    
    if len(conflict_texts) > 0 and len(conflict_texts) <= 10:
        print(f"  冲突样本示例:")
        for text in list(conflict_texts.index)[:3]:
            labels = train_df[train_df['text'] == text]['label'].unique()
            print(f"    text hash: {hash(text) % 10000}, labels: {labels}")
    
    # 4.3 极端长样本
    print("\n4.3 极端长样本")
    total_tokens_train = train_lengths.sum()
    
    for threshold in [10000, 20000, 30000]:
        long_mask = train_lengths > threshold
        long_count = long_mask.sum()
        long_tokens = train_lengths[long_mask].sum()
        print(f"  长度 > {threshold:,}:")
        print(f"    样本数: {long_count:,} ({long_count/len(train_lengths)*100:.4f}%)")
        if long_count > 0:
            print(f"    占总 token 数: {long_tokens/total_tokens_train*100:.2f}%")


def analyze_train_test_difference(train_lengths, test_lengths, train_vocab, test_vocab, train_token_counts, test_token_counts):
    """分析训练集和测试集的差异"""
    print("\n" + "=" * 80)
    print("五、训练集 vs 测试集的差异")
    print("=" * 80)
    
    # 5.1 长度分布差异
    print("\n5.1 长度分布对比")
    print("-" * 60)
    print(f"  {'指标':<15} {'训练集':>15} {'测试集':>15} {'差异':>15}")
    print("  " + "-" * 55)
    
    metrics = [
        ('均值', np.mean(train_lengths), np.mean(test_lengths)),
        ('中位数', np.median(train_lengths), np.median(test_lengths)),
        ('P90', np.percentile(train_lengths, 90), np.percentile(test_lengths, 90)),
        ('P95', np.percentile(train_lengths, 95), np.percentile(test_lengths, 95)),
        ('P99', np.percentile(train_lengths, 99), np.percentile(test_lengths, 99)),
        ('最大值', np.max(train_lengths), np.max(test_lengths)),
    ]
    
    for name, train_val, test_val in metrics:
        diff = test_val - train_val
        diff_pct = (test_val - train_val) / train_val * 100
        print(f"  {name:<15} {train_val:>15.1f} {test_val:>15.1f} {diff:>+10.1f} ({diff_pct:>+.1f}%)")
    
    # 5.2 Token 分布差异
    print("\n5.2 Token 分布差异")
    
    # OOV token
    oov_tokens = test_vocab - train_vocab
    print(f"  训练集 vocab size: {len(train_vocab):,}")
    print(f"  测试集 vocab size: {len(test_vocab):,}")
    print(f"  测试集中训练未见 token (OOV) 数: {len(oov_tokens):,}")
    print(f"  OOV 占测试集 vocab 比例: {len(oov_tokens)/len(test_vocab)*100:.2f}%")
    
    # OOV token 在测试集中的出现次数
    if len(oov_tokens) > 0:
        total_test_tokens = sum(test_token_counts.values())
        oov_occurrences = sum(test_token_counts[t] for t in oov_tokens)
        print(f"  OOV token 在测试集中的出现次数: {oov_occurrences:,}")
        print(f"  OOV token 占测试集总 token 比例: {oov_occurrences/total_test_tokens*100:.4f}%")
    
    # 高频 token 对比
    print(f"\n  高频 token 在 train/test 中的频率对比 (Top 20):")
    print(f"  {'Token':>8} {'Train Freq':>12} {'Test Freq':>12} {'Diff':>10}")
    print("  " + "-" * 45)
    
    total_train = sum(train_token_counts.values())
    total_test = sum(test_token_counts.values())
    
    for token, _ in train_token_counts.most_common(20):
        train_freq = train_token_counts[token] / total_train * 100
        test_freq = test_token_counts.get(token, 0) / total_test * 100
        diff = test_freq - train_freq
        print(f"  {token:>8} {train_freq:>11.4f}% {test_freq:>11.4f}% {diff:>+9.4f}%")


def run_truncation_experiments(train_df, sample_size=50000):
    """运行截断实验（简单 baseline），使用采样减少内存"""
    print("\n" + "=" * 80)
    print("六、截断实验（简单 Baseline）")
    print("=" * 80)
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        import time
    except ImportError:
        print("\n  ⚠️ 需要安装 scikit-learn 才能运行截断实验")
        print("  运行: pip install scikit-learn")
        return None
    
    print(f"\n正在运行截断实验 (TF-IDF + Logistic Regression)...")
    print(f"使用采样数据: {sample_size} 条样本")
    print("这可能需要几分钟时间...\n")
    
    # 采样数据
    if len(train_df) > sample_size:
        sample_df = train_df.sample(n=sample_size, random_state=42)
    else:
        sample_df = train_df
    
    gc.collect()
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        sample_df['text'].values, 
        sample_df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=sample_df['label'].values
    )
    
    del sample_df
    gc.collect()
    
    results = []
    truncation_lengths = [128, 256, 512, 1024, 2048]
    
    for max_len in truncation_lengths:
        start_time = time.time()
        
        # 截断文本
        def truncate_text(text, max_tokens):
            tokens = str(text).split()[:max_tokens]
            return ' '.join(tokens)
        
        X_train_trunc = [truncate_text(x, max_len) for x in X_train]
        X_val_trunc = [truncate_text(x, max_len) for x in X_val]
        
        # TF-IDF 特征
        vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        
        X_train_tfidf = vectorizer.fit_transform(X_train_trunc)
        X_val_tfidf = vectorizer.transform(X_val_trunc)
        
        del X_train_trunc, X_val_trunc
        gc.collect()
        
        # 训练模型
        clf = LogisticRegression(
            max_iter=500,
            C=1.0,
            solver='lbfgs',
            multi_class='multinomial',
            n_jobs=-1
        )
        clf.fit(X_train_tfidf, y_train)
        
        # 预测
        y_pred = clf.predict(X_val_tfidf)
        
        del X_train_tfidf, X_val_tfidf, clf, vectorizer
        gc.collect()
        
        # 评估
        acc = accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average='macro')
        f1_weighted = f1_score(y_val, y_pred, average='weighted')
        
        elapsed = time.time() - start_time
        
        results.append({
            'max_len': max_len,
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'time': elapsed
        })
        
        print(f"  截断@{max_len:>5}: Acc={acc:.4f}, F1_macro={f1_macro:.4f}, F1_weighted={f1_weighted:.4f} ({elapsed:.1f}s)")
    
    # 打印汇总
    print("\n截断实验结果汇总:")
    print("-" * 70)
    print(f"  {'Max Len':>8} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12} {'Time':>8}")
    print("  " + "-" * 55)
    for r in results:
        print(f"  {r['max_len']:>8} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} {r['f1_weighted']:>12.4f} {r['time']:>7.1f}s")
    
    # 分析
    print("\n截断实验分析:")
    best_result = max(results, key=lambda x: x['f1_macro'])
    print(f"  最佳截断长度: {best_result['max_len']} (F1_macro = {best_result['f1_macro']:.4f})")
    
    # 计算增益
    print(f"\n  从短到长的 F1 增益:")
    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]
        gain = curr['f1_macro'] - prev['f1_macro']
        print(f"    {prev['max_len']} → {curr['max_len']}: {gain:+.4f}")
    
    return results


def generate_summary(train_df, test_df, label_counts, train_vocab, test_vocab, train_lengths, test_lengths):
    """生成总结报告"""
    print("\n" + "=" * 80)
    print("总结报告")
    print("=" * 80)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              数据集特征分析总结                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. 整体规模 & 标签                                                          │
├─────────────────────────────────────────────────────────────────────────────┤""")
    print(f"│   训练集样本数: {len(train_df):>10,}                                            │")
    print(f"│   测试集样本数: {len(test_df):>10,}                                            │")
    print(f"│   类别数:       {len(label_counts):>10}                                            │")
    print(f"│   任务类型:     单标签多类分类                                            │")
    
    max_count = label_counts.max()
    min_count = label_counts.min()
    print(f"│   类别不平衡比: {max_count/min_count:>10.2f}:1                                          │")
    
    print("""├─────────────────────────────────────────────────────────────────────────────┤
│ 2. 长度分布                                                                 │
├─────────────────────────────────────────────────────────────────────────────┤""")
    
    train_mean = np.mean(train_lengths)
    train_median = np.median(train_lengths)
    train_p90 = np.percentile(train_lengths, 90)
    train_max = np.max(train_lengths)
    
    print(f"│   训练集均值:   {train_mean:>10.1f}                                            │")
    print(f"│   训练集中位数: {train_median:>10.1f}                                            │")
    print(f"│   训练集P90:    {train_p90:>10.1f}                                            │")
    print(f"│   训练集最大值: {train_max:>10}                                            │")
    
    coverage_512 = (train_lengths <= 512).mean() * 100
    coverage_1024 = (train_lengths <= 1024).mean() * 100
    coverage_2048 = (train_lengths <= 2048).mean() * 100
    
    print(f"│   ≤512覆盖率:   {coverage_512:>10.2f}%                                           │")
    print(f"│   ≤1024覆盖率:  {coverage_1024:>10.2f}%                                           │")
    print(f"│   ≤2048覆盖率:  {coverage_2048:>10.2f}%                                           │")
    
    print("""├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Token 统计                                                               │
├─────────────────────────────────────────────────────────────────────────────┤""")
    
    print(f"│   训练集词表大小: {len(train_vocab):>8,}                                            │")
    print(f"│   总词表大小:     {len(train_vocab | test_vocab):>8,}                                            │")
    print(f"│   OOV token数:    {len(test_vocab - train_vocab):>8,}                                            │")
    
    print("""├─────────────────────────────────────────────────────────────────────────────┤
│ 4. 建议                                                                     │
├─────────────────────────────────────────────────────────────────────────────┤""")
    
    if coverage_512 > 95:
        print("│   • 超过95%样本≤512，可优先使用512上下文模型                          │")
    elif coverage_1024 > 95:
        print("│   • 超过95%样本≤1024，建议使用1024上下文模型                          │")
    else:
        print("│   • 存在大量长文本，建议考虑长上下文/层次化模型                        │")
    
    if max_count/min_count > 10:
        print("│   • 类别不平衡严重，建议使用过采样或类别加权                          │")
    
    print("│   • 词表较小，可使用embedding预训练提升低频token表示                   │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")


def main():
    """主函数"""
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                        数据集特征分析工具                                  ║")
    print("║                    Dataset Feature Analysis Tool                          ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    
    # 加载数据
    train_df, test_df = load_data()
    
    # 一、分析标签分布
    label_counts = analyze_label_distribution(train_df)
    
    # 二、分析长度分布
    train_lengths, test_lengths = analyze_length_distribution(train_df, test_df)
    
    # 三、分析 token 统计
    train_vocab, test_vocab, train_token_counts, test_token_counts = analyze_token_statistics_streaming(train_df, test_df)
    
    # 四、分析数据质量
    analyze_data_quality(train_df, test_df, train_lengths, test_lengths)
    
    # 五、分析训练测试差异
    analyze_train_test_difference(train_lengths, test_lengths, train_vocab, test_vocab, train_token_counts, test_token_counts)
    
    # 六、运行截断实验
    run_truncation_experiments(train_df, sample_size=30000)
    
    # 生成总结
    generate_summary(train_df, test_df, label_counts, train_vocab, test_vocab, train_lengths, test_lengths)
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()
