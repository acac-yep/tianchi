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
import re
import subprocess
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# 日志工具
# =============================================================================

def log_print(*args, **kwargs):
    """带时间戳的 print 函数"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}]", *args, **kwargs)


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
        choices=['mean', 'max', 'mean_conf'],
        default='mean',
        help='窗口级聚合方式: mean / max / mean_conf(置信度加权)'
    )
    
    parser.add_argument(
        '--model-agg',
        type=str,
        choices=[
            'logits_avg',
            'prob_avg',
            'voting',
            'logits_avg_weighted',
            'prob_avg_weighted',
        ],
        default='prob_avg_weighted',
        help='模型级聚合方式（含按验证集指标加权）'
    )
    
    parser.add_argument(
        '--save-logits',
        action='store_true',
        help='是否保存 logits'
    )
    
    return parser.parse_args()


def _extract_fold_idx(path: Path) -> int:
    """从文件名中提取 fold 编号"""
    m = re.search(r"hat_cls_fold(\d+)", path.name)
    if not m:
        return -1
    return int(m.group(1))


def find_kfold_models(kfold_dir: Path) -> list:
    """扫描 K-fold 目录，优先返回 Stage2 模型，否则回退 Stage1"""
    kfold_dir = Path(kfold_dir)
    
    if not kfold_dir.exists():
        raise FileNotFoundError(f"K-fold 目录不存在: {kfold_dir}")
    
    stage2_files = { _extract_fold_idx(p): p for p in kfold_dir.glob("hat_cls_fold*_stage2_best.pt") }
    stage1_files = { _extract_fold_idx(p): p for p in kfold_dir.glob("hat_cls_fold*_best.pt") }
    
    all_folds = sorted(set(stage1_files.keys()) | set(stage2_files.keys()))
    if not all_folds:
        raise FileNotFoundError(
            f"在 {kfold_dir} 中未找到 hat_cls_fold*_stage2_best.pt 或 hat_cls_fold*_best.pt"
        )
    
    model_files = []
    for fold in all_folds:
        if fold in stage2_files:
            model_files.append(stage2_files[fold])
            log_print(f"Fold {fold}: 使用 Stage2 {stage2_files[fold]}")
        elif fold in stage1_files:
            model_files.append(stage1_files[fold])
            log_print(f"Fold {fold}: Stage2 缺失，回退 Stage1 {stage1_files[fold]}")
        else:
            raise FileNotFoundError(f"Fold {fold} 缺少模型文件")
    
    log_print(f"共选用 {len(model_files)} 个模型用于 ensemble")
    return [str(p) for p in model_files]


def main():
    args = parse_args()
    
    # 查找所有 K-fold 模型
    model_paths = find_kfold_models(args.kfold_dir)
    model_paths_str = ",".join(model_paths)
    
    log_print(f"\n使用 {len(model_paths)} 个模型进行 Ensemble 推理")
    log_print(f"窗口聚合: {args.window_agg}, 模型聚合: {args.model_agg}")
    log_print()
    
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
    log_print("执行推理命令:")
    log_print(" ".join(cmd))
    log_print()
    
    result = subprocess.run(cmd, check=True)
    
    log_print()
    log_print("=" * 60)
    log_print("K-Fold Ensemble 推理完成！")
    log_print(f"输出文件: {args.output_path}")
    log_print("=" * 60)


if __name__ == "__main__":
    main()

