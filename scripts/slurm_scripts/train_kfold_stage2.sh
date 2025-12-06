#!/bin/bash
#SBATCH -J stage2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 24:00:00
#SBATCH -o slurm-kfold-stage2-%j.out

# ============================================================================
# HAT 模型 K-Fold Stage2 微调脚本
#
# 作用：
#   - 基于 Stage1 最优 ckpt (hat_cls_fold{k}_best.pt) 进行小学习率二次微调
#   - 可选启用 R-Drop、FGM、随机滑窗起点
#   - 训练步数少（1~2 epoch），用早停保护
#
# 用法：
#   sbatch scripts/slurm_scripts/train_kfold_stage2.sh
# ============================================================================

# 清理环境变量
unset LD_LIBRARY_PATH

# 加载 cuda 和 cudnn
module load cuda/12.4
module load cudnn/9.11.0.98_cuda12

# 初始化 conda 并激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate megatron_lzxenv

# 设置 cudnn 路径
export CUDNN_PATH=/data/apps/cudnn/cudnn-linux-x86_64-9.11.0.98_cuda12-archive
export CFLAGS="-I${CUDNN_PATH}/include $CFLAGS"
export CPATH="${CUDNN_PATH}/include:$CPATH"
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib:/data/apps/cuda/12.4/lib64:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib

echo "=== 环境变量检查 ==="
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# 切换到项目根目录
cd /data/home/scyb226/lzx/study/lab/tianchi

echo "=== 开始 K-Fold Stage2 微调 ==="
echo "工作目录: $(pwd)"
echo ""

# PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ============================================================================
# 配置区
# ============================================================================

# K-Fold 参数（保持与 Stage1 一致）
N_FOLDS=5
FOLD_SEED=42

# 数据路径
TRAIN_PATH="data/processed/train.csv"
CLASS_WEIGHTS="data/processed/class_weights.npy"

# Stage1 & 输出
STAGE1_DIR="checkpoints/cls_hat512_kfold"
OUTPUT_DIR="checkpoints/cls_hat512_kfold_stage2"

# 训练超参（小学习率 + 少 epoch）
BATCH_SIZE=32
EVAL_BATCH_SIZE=64
LR=1e-5
WEIGHT_DECAY=0.01
NUM_EPOCHS=3
WARMUP_RATIO=0.05
LOG_EVERY=50
GRAD_CLIP=1.0

# 损失函数
LOSS_TYPE="smooth"          # ce / smooth / focal / focal_smooth
LABEL_SMOOTHING=0.05
FOCAL_GAMMA=2.0

# 正则与增强
USE_RDROP=true
RDROP_ALPHA=0.5
USE_FGM=false               # 如需开启设为 true
FGM_EPS=0.5
FGM_LOSS_RATIO=1.0
RANDOM_OFFSET=true          # 随机滑窗起点
RANDOM_OFFSET_STRIDE=128

# Early Stopping
EARLY_STOPPING_PATIENCE=1

# AMP / EMA
USE_AMP=true
USE_EMA=true
EMA_DECAY=0.99

# 其他
DEVICE="cuda"
NUM_WORKERS=4
SEED=42

# ============================================================================

echo "=== 配置信息 ==="
echo "Stage1 目录: ${STAGE1_DIR}"
echo "Stage2 输出: ${OUTPUT_DIR}"
echo "学习率: ${LR}"
echo "Epochs: ${NUM_EPOCHS}"
echo "损失函数: ${LOSS_TYPE}"
echo "R-Drop: ${USE_RDROP} (alpha=${RDROP_ALPHA})"
echo "FGM: ${USE_FGM} (eps=${FGM_EPS}, ratio=${FGM_LOSS_RATIO})"
echo "随机滑窗起点: ${RANDOM_OFFSET} (stride=${RANDOM_OFFSET_STRIDE})"
echo ""

# 构建命令
CMD_ARGS=(
  --train-path "${TRAIN_PATH}"
  --class-weights "${CLASS_WEIGHTS}"
  --n-folds ${N_FOLDS}
  --fold-seed ${FOLD_SEED}
  --stage1-dir "${STAGE1_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --batch-size ${BATCH_SIZE}
  --eval-batch-size ${EVAL_BATCH_SIZE}
  --lr ${LR}
  --weight-decay ${WEIGHT_DECAY}
  --num-epochs ${NUM_EPOCHS}
  --warmup-ratio ${WARMUP_RATIO}
  --log-every ${LOG_EVERY}
  --grad-clip ${GRAD_CLIP}
  --device ${DEVICE}
  --num-workers ${NUM_WORKERS}
  --seed ${SEED}
  --loss-type "${LOSS_TYPE}"
  --label-smoothing ${LABEL_SMOOTHING}
  --focal-gamma ${FOCAL_GAMMA}
  --early-stopping-patience ${EARLY_STOPPING_PATIENCE}
)

# 条件参数
if [ "${USE_AMP}" = "true" ]; then
  CMD_ARGS+=(--use-amp)
fi
if [ "${USE_EMA}" = "true" ]; then
  CMD_ARGS+=(--use-ema --ema-decay ${EMA_DECAY})
fi
if [ "${USE_RDROP}" = "true" ]; then
  CMD_ARGS+=(--use-rdrop --rdrop-alpha ${RDROP_ALPHA})
fi
if [ "${USE_FGM}" = "true" ]; then
  CMD_ARGS+=(--use-fgm --fgm-epsilon ${FGM_EPS} --fgm-loss-ratio ${FGM_LOSS_RATIO})
fi
if [ "${RANDOM_OFFSET}" = "true" ]; then
  CMD_ARGS+=(--random-offset --random-offset-stride ${RANDOM_OFFSET_STRIDE})
fi

# 执行训练（单卡示例，可按需改 CUDA_VISIBLE_DEVICES）
CUDA_VISIBLE_DEVICES=0 \
python -u scripts/cls_train_kfold_stage2.py "${CMD_ARGS[@]}"

echo ""
echo "=== Stage2 微调完成 ==="

# 列出输出
if [ -d "${OUTPUT_DIR}" ]; then
    echo "生成的模型文件（如存在 Stage2 提升）:"
    ls -lh "${OUTPUT_DIR}"/hat_cls_fold*_stage2_best.pt 2>/dev/null || echo "  Stage2 未生成新的 best，可能使用 Stage1 回退"
    echo ""
    echo "推理示例:"
    echo "  python scripts/infer_kfold.py \\"
    echo "    --kfold-dir ${OUTPUT_DIR} \\"
    echo "    --test-path data/processed/test.csv \\"
    echo "    --output-path outputs/submission/submission_kfold_stage2.csv"
fi

echo ""
echo "=== 全部完成 ==="


