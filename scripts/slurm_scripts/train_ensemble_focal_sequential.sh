#!/bin/bash
#SBATCH -J ens_focal_seq
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 72:00:00

# ============================================================================
# 多 Seed Ensemble 训练脚本 - Focal Loss 版本 (顺序执行)
# 
# 说明:
#   在单个 GPU 上顺序执行 5 个不同 seed 的训练任务
#   适合 GPU 资源紧张的情况（只需要 1 个 GPU）
#   
# 使用方法:
#   sbatch scripts/slurm_scripts/train_ensemble_focal_sequential.sh
# ============================================================================

# 定义不同的随机种子
SEEDS=(42 13 87 2025 7)

echo "=============================================="
echo "Sequential Focal Loss Ensemble Training"
echo "Seeds: ${SEEDS[@]}"
echo "需要 GPU 数量: 1 (顺序执行)"
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

echo "=== 环境变量检查 ==="
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# 切换到项目根目录
cd /data/home/scyb226/lzx/study/lab/tianchi

# 设置 PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# 循环训练每个 seed
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=============================================="
    echo "=== 开始训练 Seed=$SEED (Focal Loss) ==="
    echo "=============================================="
    echo ""
    
    # 检查是否已存在该 seed 的 checkpoint (支持断点续训)
    CKPT_PATH="checkpoints/ensemble_focal/cls_hat512_stage2_seed${SEED}/hat_cls_best.pt"
    if [ -f "$CKPT_PATH" ]; then
        echo "发现已存在的 checkpoint: $CKPT_PATH"
        echo "跳过 seed=$SEED 的训练..."
        continue
    fi
    
    CUDA_VISIBLE_DEVICES=0 \
    python -u scripts/cls_finetune_stage2_focal.py \
      --train-path data/processed/train.csv \
      --val-path data/processed/val.csv \
      --class-weights data/processed/class_weights.npy \
      --pretrained-ckpt checkpoints/cls_hat512/hat_cls_best.pt \
      --output-dir checkpoints/ensemble_focal/cls_hat512_stage2_seed${SEED} \
      --loss-type focal \
      --focal-gamma 2.0 \
      --batch-size 32 \
      --eval-batch-size 64 \
      --lr 3e-5 \
      --weight-decay 0.01 \
      --num-epochs 3 \
      --warmup-ratio 0.05 \
      --log-every 50 \
      --early-patience 2 \
      --grad-clip 1.0 \
      --device cuda \
      --num-workers 4 \
      --seed $SEED
    
    echo ""
    echo "=== Seed $SEED (Focal Loss) 训练完成 ==="
done

echo ""
echo "=============================================="
echo "=== 所有 Seed 训练完成！ ==="
echo "=============================================="
echo ""
echo "接下来可以运行 ensemble 评估:"
echo "  python scripts/ensemble_eval.py --checkpoint-dir checkpoints/ensemble_focal"

