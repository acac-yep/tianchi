#!/usr/bin/env python3
"""
K-Fold 模型推理辅助脚本

功能:
    自动扫描 K-fold 目录下的所有模型，并调用 infer.py 进行 ensemble 推理

使用方法:
    python scripts/infer_kfold.py \
        --kfold-dir checkpoints/cls_hat512_kfold \
        --test-path data/processed/test.csv \
        --output-path outputs/submission/submission_kfold.csv
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description='K-Fold 模型推理辅助脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--kfold-dir',
        type=str,
        required=True,
        help='K-fold 模型目录（包含 hat_cls_fold{k}_best.pt 文件）'
    )
    
    parser.add_argument(
        '--test-path',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'processed' / 'test.csv'),
        help='测试数据路径'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default=str(PROJECT_ROOT / 'outputs' / 'submission' / 'submission_kfold.csv'),
        help='输出路径'
    )
    
    parser.add_argument(
        '--val-path',
        type=str,
        default=None,
        help='验证集路径（可选，用于 sanity check）'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='推理 batch size'
    )
    
    parser.add_argument(
        '--window-agg',
        type=str,
        choices=['mean', 'max'],
        default='mean',
        help='窗口级聚合方式'
    )
    
    parser.add_argument(
        '--model-agg',
        type=str,
        choices=['logits_avg', 'prob_avg', 'voting'],
        default='logits_avg',
        help='模型级聚合方式'
    )
    
    parser.add_argument(
        '--save-logits',
        action='store_true',
        help='是否保存 logits'
    )
    
    return parser.parse_args()


def find_kfold_models(kfold_dir: Path) -> list:
    """扫描 K-fold 目录，找到所有 fold 模型"""
    kfold_dir = Path(kfold_dir)
    
    if not kfold_dir.exists():
        raise FileNotFoundError(f"K-fold 目录不存在: {kfold_dir}")
    
    # 查找所有 hat_cls_fold{k}_best.pt 文件
    model_files = sorted(kfold_dir.glob("hat_cls_fold*_best.pt"))
    
    if not model_files:
        raise FileNotFoundError(
            f"在 {kfold_dir} 中未找到任何 hat_cls_fold*_best.pt 文件"
        )
    
    print(f"找到 {len(model_files)} 个 K-fold 模型:")
    for i, model_path in enumerate(model_files):
        print(f"  [{i+1}] {model_path}")
    
    return [str(p) for p in model_files]


def main():
    args = parse_args()
    
    # 查找所有 K-fold 模型
    model_paths = find_kfold_models(args.kfold_dir)
    model_paths_str = ",".join(model_paths)
    
    print(f"\n使用 {len(model_paths)} 个模型进行 Ensemble 推理")
    print(f"窗口聚合: {args.window_agg}, 模型聚合: {args.model_agg}")
    print()
    
    # 构建 infer.py 命令
    infer_script = PROJECT_ROOT / "scripts" / "infer.py"
    
    cmd = [
        sys.executable,
        str(infer_script),
        "--test-path", args.test_path,
        "--model-paths", model_paths_str,
        "--output-path", args.output_path,
        "--batch-size", str(args.batch_size),
        "--window-agg", args.window_agg,
        "--model-agg", args.model_agg,
    ]
    
    if args.val_path:
        cmd.extend(["--val-path", args.val_path])
    
    if args.save_logits:
        cmd.append("--save-logits")
    
    # 执行推理
    print("执行推理命令:")
    print(" ".join(cmd))
    print()
    
    result = subprocess.run(cmd, check=True)
    
    print()
    print("=" * 60)
    print("K-Fold Ensemble 推理完成！")
    print(f"输出文件: {args.output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

