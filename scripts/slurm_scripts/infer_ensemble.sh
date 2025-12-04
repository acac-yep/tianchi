#!/bin/bash
#SBATCH -J hat_ensemble_infer
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 6:00:00
#SBATCH -o slurm-ensemble-infer-%j.out

# ============================================================================
# HAT 多模型 Ensemble 推理脚本
# 
# 功能:
#   1. 自动扫描指定目录下的所有 checkpoint 进行 ensemble
#   2. 支持滑动窗口处理超长文档
#   3. 先在验证集上做 sanity check，再生成测试集预测
#   
# 使用方法:
#   sbatch scripts/slurm_scripts/infer_ensemble.sh
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

echo "=== 开始 Ensemble 推理 ==="
echo "工作目录: $(pwd)"
echo ""

# 设置 PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ============================================================================
# 配置区
# ============================================================================

# Checkpoint 目录（会自动扫描该目录下所有 hat_cls_best.pt）
CHECKPOINT_DIR="checkpoints/ensemble_focal"

# 或者手动指定模型列表（取消注释使用）
# MANUAL_MODEL_PATHS="checkpoints/cls_hat512/seed42_best.pt,checkpoints/cls_hat512/seed3407_best.pt"

# 数据路径
TEST_PATH="data/processed/test.csv"
VAL_PATH="data/processed/val.csv"

# 输出配置
OUTPUT_DIR="outputs/submission"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PATH="${OUTPUT_DIR}/submission_${TIMESTAMP}.csv"

# 推理参数
BATCH_SIZE=64
NUM_WORKERS=4
WINDOW_AGG="mean"
MODEL_AGG="logits_avg"

# ============================================================================

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 自动扫描或使用手动指定的模型
if [ -n "$MANUAL_MODEL_PATHS" ]; then
    MODEL_PATHS="$MANUAL_MODEL_PATHS"
    echo "使用手动指定的模型路径"
else
    echo "扫描目录: ${CHECKPOINT_DIR}"
    MODEL_PATHS=$(find "$CHECKPOINT_DIR" -name "hat_cls_best.pt" 2>/dev/null | sort | tr '\n' ',' | sed 's/,$//')
    
    if [ -z "$MODEL_PATHS" ]; then
        echo "错误: 在 ${CHECKPOINT_DIR} 目录下未找到 hat_cls_best.pt 文件"
        exit 1
    fi
fi

echo ""
echo "=== 将使用以下模型进行 Ensemble ==="
IFS=',' read -ra MODEL_ARRAY <<< "$MODEL_PATHS"
for i in "${!MODEL_ARRAY[@]}"; do
    echo "  [$((i+1))] ${MODEL_ARRAY[$i]}"
done
echo "共 ${#MODEL_ARRAY[@]} 个模型"
echo ""

echo "=== 配置信息 ==="
echo "测试数据: ${TEST_PATH}"
echo "验证数据: ${VAL_PATH}"
echo "输出路径: ${OUTPUT_PATH}"
echo "Batch Size: ${BATCH_SIZE}"
echo "窗口聚合: ${WINDOW_AGG}"
echo "模型聚合: ${MODEL_AGG}"
echo ""

# 运行推理（包含验证集 sanity check）
echo "=== 开始推理 ==="
CUDA_VISIBLE_DEVICES=0 \
python -u scripts/infer.py \
  --test-path "${TEST_PATH}" \
  --model-paths "${MODEL_PATHS}" \
  --output-path "${OUTPUT_PATH}" \
  --batch-size ${BATCH_SIZE} \
  --num-workers ${NUM_WORKERS} \
  --window-agg ${WINDOW_AGG} \
  --model-agg ${MODEL_AGG} \
  --val-path "${VAL_PATH}" \
  --save-logits \
  --device cuda

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
    
    # 创建一个指向最新提交的软链接
    ln -sf "$(basename "$OUTPUT_PATH")" "${OUTPUT_DIR}/submission_latest.csv"
    echo ""
    echo "已创建软链接: ${OUTPUT_DIR}/submission_latest.csv"
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

