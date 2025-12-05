#!/bin/bash
#SBATCH -J diverse_relax
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 72:00:00
#SBATCH --array=0-4
#SBATCH -o slurm-diverse-relax-%A_%a.out

# ============================================================================
# 多样性 Ensemble 训练脚本 (放宽阈值版本)
# 
# 关键改动：
#   - min_improvement 从 0.0005 降低到 0.0001
#   - 允许保存接近 baseline 的模型（用于 ensemble 多样性）
#   
# 使用方法:
#   sbatch scripts/slurm_scripts/train_diverse_ensemble_relaxed.sh
# ============================================================================

# 配置矩阵
declare -a CONFIGS=(
    "42 5e-5 0 0.10 0.10"
    "13 7e-5 2 0.15 0.10"
    "87 5e-5 4 0.10 0.15"
    "2025 1e-4 0 0.20 0.10"
    "7 7e-5 3 0.15 0.15"
)

CONFIG_STR=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
read -r SEED LR FREEZE_LAYERS DROPOUT ATTN_DROPOUT <<< "$CONFIG_STR"

echo "=============================================="
echo "多样性训练 (放宽阈值) - Job $SLURM_ARRAY_TASK_ID"
echo "Seed: $SEED, LR: $LR, Freeze: $FREEZE_LAYERS"
echo "=============================================="

# 环境设置
unset LD_LIBRARY_PATH
module load cuda/12.4
module load cudnn/9.11.0.98_cuda12
source $(conda info --base)/etc/profile.d/conda.sh
conda activate megatron_lzxenv

export CUDNN_PATH=/data/apps/cudnn/cudnn-linux-x86_64-9.11.0.98_cuda12-archive
export CFLAGS="-I${CUDNN_PATH}/include $CFLAGS"
export CPATH="${CUDNN_PATH}/include:$CPATH"
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib:/data/apps/cuda/12.4/lib64:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib

cd /data/home/scyb226/lzx/study/lab/tianchi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

OUTPUT_DIR="checkpoints/diverse_ensemble_relaxed/seed${SEED}_lr${LR}_freeze${FREEZE_LAYERS}_drop${DROPOUT}"

# 训练（降低 min_improvement 阈值）
CUDA_VISIBLE_DEVICES=0 \
python -u scripts/cls_finetune_diverse.py \
  --train-path data/processed/train.csv \
  --val-path data/processed/val.csv \
  --class-weights data/processed/class_weights.npy \
  --pretrained-ckpt checkpoints/cls_hat512/hat_cls_best.pt \
  --output-dir "$OUTPUT_DIR" \
  --loss-type focal \
  --focal-gamma 2.0 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --lr $LR \
  --weight-decay 0.01 \
  --num-epochs 5 \
  --warmup-ratio 0.1 \
  --log-every 50 \
  --early-patience 3 \
  --min-improvement 0.0001 \
  --grad-clip 1.0 \
  --device cuda \
  --num-workers 4 \
  --seed $SEED \
  --freeze-layers $FREEZE_LAYERS \
  --dropout $DROPOUT \
  --attention-dropout $ATTN_DROPOUT

echo ""
echo "=== 训练完成 ==="
echo "输出目录: $OUTPUT_DIR"

