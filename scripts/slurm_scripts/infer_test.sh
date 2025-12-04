#!/bin/bash
#SBATCH -J hat_infer
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 4:00:00
#SBATCH -o slurm-%j.out

# ============================================================================
# HAT 模型推理脚本
# 
# 功能:
#   1. 支持多模型 ensemble（多 seed）
#   2. 支持滑动窗口处理超长文档
#   3. 生成天池竞赛格式的 submission.csv
#   
# 使用方法:
#   # 单模型推理
#   sbatch scripts/slurm_scripts/infer_test.sh
#   
#   # 或修改下面的 MODEL_PATHS 使用多模型 ensemble
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

echo "=== 开始 HAT 模型推理 ==="
echo "工作目录: $(pwd)"
echo ""

# 设置 PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ============================================================================
# 配置区（根据需要修改）
# ============================================================================

# 模型路径（多个路径用逗号分隔）
# 单模型示例:
MODEL_PATHS="checkpoints/cls_hat512/hat_cls_best.pt"

# 多模型 ensemble 示例（取消注释使用）:
# MODEL_PATHS="checkpoints/cls_hat512/seed42_best.pt,checkpoints/cls_hat512/seed3407_best.pt,checkpoints/cls_hat512/seed13_best.pt"

# 如果使用 ensemble_focal 目录下的多个 seed
# MODEL_PATHS=$(find checkpoints/ensemble_focal -name "hat_cls_best.pt" | tr '\n' ',' | sed 's/,$//')

# 测试数据路径
TEST_PATH="data/processed/test.csv"

# 输出路径
OUTPUT_PATH="submission.csv"

# 推理参数
BATCH_SIZE=64
NUM_WORKERS=4

# 聚合策略
WINDOW_AGG="mean"    # 窗口聚合: mean, max
MODEL_AGG="logits_avg"  # 模型聚合: logits_avg, prob_avg, voting

# 是否保存 logits（用于调试）
SAVE_LOGITS=""  # 取消注释启用: SAVE_LOGITS="--save-logits"

# 是否在验证集上做 sanity check
# VAL_PATH="data/processed/val.csv"  # 取消注释启用

# ============================================================================

echo "=== 配置信息 ==="
echo "模型路径: ${MODEL_PATHS}"
echo "测试数据: ${TEST_PATH}"
echo "输出路径: ${OUTPUT_PATH}"
echo "Batch Size: ${BATCH_SIZE}"
echo "窗口聚合: ${WINDOW_AGG}"
echo "模型聚合: ${MODEL_AGG}"
echo ""

# 检查模型文件是否存在
IFS=',' read -ra MODEL_ARRAY <<< "$MODEL_PATHS"
for model_path in "${MODEL_ARRAY[@]}"; do
    if [ ! -f "$model_path" ]; then
        echo "错误: 模型文件不存在: $model_path"
        exit 1
    fi
done
echo "所有模型文件检查通过"
echo ""

# 检查测试数据是否存在
if [ ! -f "$TEST_PATH" ]; then
    echo "错误: 测试数据文件不存在: $TEST_PATH"
    exit 1
fi
echo "测试数据文件检查通过"
echo ""

# 构建命令
CMD="python -u scripts/infer.py \
  --test-path ${TEST_PATH} \
  --model-paths ${MODEL_PATHS} \
  --output-path ${OUTPUT_PATH} \
  --batch-size ${BATCH_SIZE} \
  --num-workers ${NUM_WORKERS} \
  --window-agg ${WINDOW_AGG} \
  --model-agg ${MODEL_AGG} \
  --device cuda"

# 添加可选参数
if [ -n "$SAVE_LOGITS" ]; then
    CMD="$CMD --save-logits"
fi

if [ -n "$VAL_PATH" ] && [ -f "$VAL_PATH" ]; then
    CMD="$CMD --val-path ${VAL_PATH}"
    echo "将在验证集上做 sanity check: ${VAL_PATH}"
fi

echo ""
echo "=== 执行命令 ==="
echo "$CMD"
echo ""

# 运行推理
CUDA_VISIBLE_DEVICES=0 $CMD

echo ""
echo "=== 推理完成 ==="
echo "提交文件: ${OUTPUT_PATH}"
echo ""

# 显示提交文件前几行
if [ -f "$OUTPUT_PATH" ]; then
    echo "提交文件预览（前 10 行）:"
    head -10 "$OUTPUT_PATH"
    echo ""
    echo "提交文件行数: $(wc -l < $OUTPUT_PATH)"
fi

