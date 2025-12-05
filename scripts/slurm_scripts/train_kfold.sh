#!/bin/bash
#SBATCH -J cls_kfold
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 48:00:00
#SBATCH -o slurm-kfold-%j.out

# ============================================================================
# HAT 模型 K-Fold 交叉验证训练脚本
# 
# 功能:
#   1. 使用 Stratified K-Fold 划分训练集
#   2. 对每个 fold 训练一个模型
#   3. 保存为 hat_cls_fold{k}_best.pt
#   
# 新增功能:
#   - Label Smoothing: 防止模型过度自信，提升泛化能力
#   - Focal Loss: 关注难样本，适合类别不平衡
#   - WeightedRandomSampler: 采样层面的类别平衡
#   - Early Stopping: 防止过拟合，节省训练时间
#   - AMP (混合精度): 加速训练并节省显存，通常还能轻微提升泛化
#   - EMA (指数移动平均): 维护模型权重的移动平均，验证时使用 EMA 模型
#   
# 使用方法:
#   sbatch scripts/slurm_scripts/train_kfold.sh
#   
# 配置说明:
#   - LOSS_TYPE: 损失函数类型 (ce/smooth/focal/focal_smooth)
#   - LABEL_SMOOTHING: 标签平滑系数 (推荐 0.05 或 0.1)
#   - USE_WEIGHTED_SAMPLER: 是否使用加权采样器 (true/false)
#   - EARLY_STOPPING_PATIENCE: Early stopping patience (留空表示不使用)
#   - USE_AMP: 是否使用混合精度训练 (true/false，推荐 true)
#   - USE_EMA: 是否使用指数移动平均 (true/false，推荐 true)
#   - EMA_DECAY: EMA 衰减率 (推荐 0.9999)
# ============================================================================

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

echo "=== 开始 K-Fold 交叉验证训练 ==="
echo "工作目录: $(pwd)"
echo ""

# 设置 PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ============================================================================
# 配置区
# ============================================================================

# K-Fold 参数
N_FOLDS=5
FOLD_SEED=42

# 数据路径
TRAIN_PATH="data/processed/train.csv"
CLASS_WEIGHTS="data/processed/class_weights.npy"

# 预训练权重（可选）
MLM_CKPT="checkpoints/mlm_hat512/hat_mlm_final.pt"

# 输出目录
OUTPUT_DIR="checkpoints/cls_hat512_kfold"

# 训练超参（减小 batch size 以避免显存不足）
BATCH_SIZE=32
EVAL_BATCH_SIZE=64
LR=3e-5
WEIGHT_DECAY=0.01
NUM_EPOCHS=5
WARMUP_RATIO=0.06
LOG_EVERY=50
GRAD_CLIP=1.0

# 损失函数相关
LOSS_TYPE="smooth"           # ce / smooth / focal / focal_smooth
LABEL_SMOOTHING=0.05         # 标签平滑系数（推荐 0.05 或 0.1）
FOCAL_GAMMA=2.0              # Focal Loss 的 gamma 参数

# 采样策略
USE_WEIGHTED_SAMPLER=false   # 是否使用 WeightedRandomSampler（true/false）

# Early Stopping
EARLY_STOPPING_PATIENCE=""   # Early stopping patience（留空表示不使用，例如: 2）

# AMP (混合精度训练)
USE_AMP=true                 # 是否使用混合精度训练（推荐 true）

# EMA (指数移动平均)
USE_EMA=true                 # 是否使用指数移动平均（推荐 true）
EMA_DECAY=0.9999            # EMA 衰减率（推荐 0.9999）

# 其他
DEVICE="cuda"
NUM_WORKERS=4
SEED=42

# ============================================================================

echo "=== 配置信息 ==="
echo "K-Fold 折数: ${N_FOLDS}"
echo "训练数据: ${TRAIN_PATH}"
echo "输出目录: ${OUTPUT_DIR}"
echo "Batch Size: ${BATCH_SIZE}"
echo "学习率: ${LR}"
echo "Epochs: ${NUM_EPOCHS}"
echo "损失函数: ${LOSS_TYPE}"
if [ "${LOSS_TYPE}" = "smooth" ] || [ "${LOSS_TYPE}" = "focal_smooth" ]; then
    echo "  标签平滑: ${LABEL_SMOOTHING}"
fi
if [ "${LOSS_TYPE}" = "focal" ] || [ "${LOSS_TYPE}" = "focal_smooth" ]; then
    echo "  Focal gamma: ${FOCAL_GAMMA}"
fi
if [ "${USE_WEIGHTED_SAMPLER}" = "true" ]; then
    echo "使用 WeightedRandomSampler: 是"
fi
if [ -n "${EARLY_STOPPING_PATIENCE}" ]; then
    echo "Early Stopping patience: ${EARLY_STOPPING_PATIENCE}"
fi
if [ "${USE_AMP}" = "true" ]; then
    echo "使用 AMP (混合精度): 是"
fi
if [ "${USE_EMA}" = "true" ]; then
    echo "使用 EMA (指数移动平均): 是"
    echo "  EMA 衰减率: ${EMA_DECAY}"
fi
echo ""

# 运行 K-Fold 训练
# 构建命令行参数
CMD_ARGS=(
  --train-path "${TRAIN_PATH}"
  --class-weights "${CLASS_WEIGHTS}"
  --n-folds ${N_FOLDS}
  --fold-seed ${FOLD_SEED}
  --mlm-ckpt "${MLM_CKPT}"
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
)

# 条件添加参数
if [ "${USE_WEIGHTED_SAMPLER}" = "true" ]; then
  CMD_ARGS+=(--use-weighted-sampler)
fi

if [ -n "${EARLY_STOPPING_PATIENCE}" ]; then
  CMD_ARGS+=(--early-stopping-patience ${EARLY_STOPPING_PATIENCE})
fi

if [ "${USE_AMP}" = "true" ]; then
  CMD_ARGS+=(--use-amp)
fi

if [ "${USE_EMA}" = "true" ]; then
  CMD_ARGS+=(--use-ema --ema-decay ${EMA_DECAY})
fi

# 执行训练
CUDA_VISIBLE_DEVICES=0 \
python -u scripts/cls_train_kfold.py "${CMD_ARGS[@]}"

echo ""
echo "=== K-Fold 训练完成 ==="

# 检查输出
if [ -d "${OUTPUT_DIR}" ]; then
    echo ""
    echo "生成的模型文件:"
    ls -lh "${OUTPUT_DIR}"/hat_cls_fold*_best.pt 2>/dev/null || echo "  未找到模型文件"
    
    echo ""
    echo "推理时使用以下命令:"
    echo "  python scripts/infer.py \\"
    echo "    --test-path data/processed/test.csv \\"
    echo "    --model-paths ${OUTPUT_DIR}/hat_cls_fold0_best.pt,${OUTPUT_DIR}/hat_cls_fold1_best.pt,${OUTPUT_DIR}/hat_cls_fold2_best.pt,${OUTPUT_DIR}/hat_cls_fold3_best.pt,${OUTPUT_DIR}/hat_cls_fold4_best.pt \\"
    echo "    --output-path outputs/submission/submission_kfold.csv"
fi

echo ""
echo "=== 全部完成 ==="

