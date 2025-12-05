#!/bin/bash
#SBATCH -J kfold_infer
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 6:00:00
#SBATCH -o slurm-kfold-infer-%j.out

# ============================================================================
# K-Fold 模型 Ensemble 推理脚本
# 
# 功能:
#   1. 自动扫描 K-fold 目录下的所有模型
#   2. 使用所有 fold 模型进行 ensemble 推理
#   3. 生成测试集预测结果
#   
# 使用方法:
#   sbatch scripts/slurm_scripts/infer_kfold.sh
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

echo "=== 开始 K-Fold Ensemble 推理 ==="
echo "工作目录: $(pwd)"
echo ""

# 设置 PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ============================================================================
# 配置区
# ============================================================================

# K-Fold 模型目录
KFOLD_DIR="checkpoints/cls_hat512_kfold"

# 数据路径
TEST_PATH="data/processed/test.csv"
VAL_PATH="data/processed/val.csv"

# 输出配置
OUTPUT_DIR="outputs/submission"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PATH="${OUTPUT_DIR}/submission_kfold_${TIMESTAMP}.csv"

# 推理参数
BATCH_SIZE=64
NUM_WORKERS=4
WINDOW_AGG="mean"
MODEL_AGG="logits_avg"

# ============================================================================

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "=== 配置信息 ==="
echo "K-Fold 模型目录: ${KFOLD_DIR}"
echo "测试数据: ${TEST_PATH}"
echo "验证数据: ${VAL_PATH}"
echo "输出路径: ${OUTPUT_PATH}"
echo "Batch Size: ${BATCH_SIZE}"
echo "窗口聚合: ${WINDOW_AGG}"
echo "模型聚合: ${MODEL_AGG}"
echo ""

# 运行 K-Fold 推理
echo "=== 开始推理 ==="
CUDA_VISIBLE_DEVICES=0 \
python -u scripts/infer_kfold.py \
  --kfold-dir "${KFOLD_DIR}" \
  --test-path "${TEST_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  --val-path "${VAL_PATH}" \
  --batch-size ${BATCH_SIZE} \
  --window-agg ${WINDOW_AGG} \
  --model-agg ${MODEL_AGG} \
  --save-logits

echo ""
echo "=== 推理完成 ==="

# 显示结果
if [ -f "$OUTPUT_PATH" ]; then
    echo ""
    echo "提交文件: ${OUTPUT_PATH}"
    echo "文件大小: $(du -h "$OUTPUT_PATH" | cut -f1)"
    echo "行数: $(wc -l < "$OUTPUT_PATH")"
    echo ""
    echo "提交文件预览（前 10 行）:"
    head -10 "$OUTPUT_PATH"
    
    # 创建一个指向最新 K-fold 提交的软链接
    ln -sf "$(basename "$OUTPUT_PATH")" "${OUTPUT_DIR}/submission_kfold_latest.csv"
    echo ""
    echo "已创建软链接: ${OUTPUT_DIR}/submission_kfold_latest.csv"
fi

# 如果有 logits 文件，也显示信息
LOGITS_PATH="${OUTPUT_PATH%.csv}.logits.npy"
if [ -f "$LOGITS_PATH" ]; then
    echo ""
    echo "Logits 文件: ${LOGITS_PATH}"
    echo "文件大小: $(du -h "$LOGITS_PATH" | cut -f1)"
fi

echo ""
echo "=== 全部完成 ==="

