#!/bin/bash
#SBATCH -J diverse_ens
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 72:00:00
#SBATCH --array=0-4
#SBATCH -o slurm-diverse-%A_%a.out

# ============================================================================
# 多样性 Ensemble 训练脚本
# 
# 核心改进:
#   1. 更激进的学习率 (5e-5 ~ 1e-4)
#   2. 部分冻结底层 encoder
#   3. 不同的 dropout 配置
#   4. 不同的随机种子
#   
# 设计原则:
#   - 每个模型有足够的差异性
#   - 确保训练能产生超过 baseline 的结果
#   - 禁止保存等于 baseline 的 checkpoint
#   
# 使用方法:
#   sbatch scripts/slurm_scripts/train_diverse_ensemble.sh
# ============================================================================

# 配置矩阵: 每个 job 有不同的超参组合
# [seed, lr, freeze_layers, dropout, attention_dropout]
declare -a CONFIGS=(
    "42 5e-5 0 0.10 0.10"    # Config 0: 标准配置，更大 lr
    "13 7e-5 2 0.15 0.10"    # Config 1: 冻结2层，更高 dropout
    "87 5e-5 4 0.10 0.15"    # Config 2: 冻结4层，更高 attention dropout
    "2025 1e-4 0 0.20 0.10"  # Config 3: 最大 lr，最高 dropout
    "7 7e-5 3 0.15 0.15"     # Config 4: 均衡配置
)

# 获取当前 job 的配置
CONFIG_STR=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
read -r SEED LR FREEZE_LAYERS DROPOUT ATTN_DROPOUT <<< "$CONFIG_STR"

echo "=============================================="
echo "多样性 Ensemble 训练 - Job $SLURM_ARRAY_TASK_ID"
echo "=============================================="
echo "Seed: $SEED"
echo "Learning Rate: $LR"
echo "Freeze Layers: $FREEZE_LAYERS"
echo "Dropout: $DROPOUT"
echo "Attention Dropout: $ATTN_DROPOUT"
echo "=============================================="

# 清理环境变量
unset LD_LIBRARY_PATH

# 加载cuda和cudnn
module load cuda/12.4
module load cudnn/9.11.0.98_cuda12

# 初始化 conda 并激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate megatron_lzxenv

# 设置cudnn编译路径
export CUDNN_PATH=/data/apps/cudnn/cudnn-linux-x86_64-9.11.0.98_cuda12-archive

export CFLAGS="-I${CUDNN_PATH}/include $CFLAGS"
export CPATH="${CUDNN_PATH}/include:$CPATH"

# 设置基础LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib:/data/apps/cuda/12.4/lib64:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib

echo ""
echo "=== 环境变量检查 ==="
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# 切换到项目根目录
cd /data/home/scyb226/lzx/study/lab/tianchi

echo "=== 开始多样性训练 ==="
echo "工作目录: $(pwd)"
echo ""

# 设置 PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# 构建输出目录名
OUTPUT_DIR="checkpoints/diverse_ensemble/seed${SEED}_lr${LR}_freeze${FREEZE_LAYERS}_drop${DROPOUT}"

# 多样性训练命令
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
  --min-improvement 0.0005 \
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

# 检查是否成功生成了 best checkpoint
if [ -f "$OUTPUT_DIR/hat_cls_best.pt" ]; then
    echo "✓ Best checkpoint 已生成"
else
    echo "⚠️ 警告: 未生成 best checkpoint (训练未超过 baseline)"
    if [ -f "$OUTPUT_DIR/hat_cls_last.pt" ]; then
        echo "  Last checkpoint 已保存"
    fi
fi

