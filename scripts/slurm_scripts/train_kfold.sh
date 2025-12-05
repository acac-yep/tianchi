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
# 使用方法:
#   sbatch scripts/slurm_scripts/train_kfold.sh
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
LR=1e-4
WEIGHT_DECAY=0.01
NUM_EPOCHS=5
WARMUP_RATIO=0.06
LOG_EVERY=50
GRAD_CLIP=1.0

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
echo ""

# 运行 K-Fold 训练
CUDA_VISIBLE_DEVICES=0 \
python -u scripts/cls_train_kfold.py \
  --train-path "${TRAIN_PATH}" \
  --class-weights "${CLASS_WEIGHTS}" \
  --n-folds ${N_FOLDS} \
  --fold-seed ${FOLD_SEED} \
  --mlm-ckpt "${MLM_CKPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size ${BATCH_SIZE} \
  --eval-batch-size ${EVAL_BATCH_SIZE} \
  --lr ${LR} \
  --weight-decay ${WEIGHT_DECAY} \
  --num-epochs ${NUM_EPOCHS} \
  --warmup-ratio ${WARMUP_RATIO} \
  --log-every ${LOG_EVERY} \
  --grad-clip ${GRAD_CLIP} \
  --device ${DEVICE} \
  --num-workers ${NUM_WORKERS} \
  --seed ${SEED}

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

