#!/bin/bash
#SBATCH -J test_dynamic_pp
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 12:00:00
#SBATCH -o slurm-infer-%j.out

# ============================================================================
# HAT 模型 K-Fold 推理脚本（SLURM）
# 
# 功能:
#   1. 自动扫描 K-Fold 目录下的所有 hat_cls_fold{k}_best.pt
#   2. 使用 scripts/infer_kfold.py 进行多模型加权集成推理
#   3. 默认采用置信度加权窗口融合 + 验证集指标加权概率平均
# 
# 使用方法:
#   sbatch scripts/slurm_scripts/infer_kfold.sh
# 
# 如需自定义参数，可在下方配置区修改。
# ============================================================================

# 清理环境变量
unset LD_LIBRARY_PATH

# 加载 cuda / cudnn
module load cuda/12.4
module load cudnn/9.11.0.98_cuda12

# 初始化 conda 并激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate megatron_lzxenv

# 设置 cudnn 路径
export CUDNN_PATH=/data/apps/cudnn/cudnn-linux-x86_64-9.11.0.98_cuda12-archive
export CFLAGS="-I${CUDNN_PATH}/include $CFLAGS"
export CPATH="${CUDNN_PATH}/include:$CPATH"

# 设置基础 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib:/data/apps/cuda/12.4/lib64:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib

echo "=== 环境变量检查 ==="
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# 切换到项目根目录
cd /data/home/scyb226/lzx/study/lab/tianchi
echo "=== 开始 K-Fold 推理 ==="
echo "工作目录: $(pwd)"
echo ""

# 运行时优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ============================================================================
# 配置区
# ============================================================================
# K-Fold 模型目录
KFOLD_DIR="checkpoints/cls_hat512_kfold"

# 数据路径
TEST_PATH="data/processed/test.csv"
VAL_PATH="/data/home/scyb226/lzx/study/lab/tianchi/data/processed/val.csv"  # 验证集路径，用于阈值搜索

# 输出
OUTPUT_PATH="outputs/submission/submission_kfold.csv"

# 推理超参
BATCH_SIZE=32                  # 稳定留显存，兼顾 TTA/MC-Dropout
WINDOW_AGG="mean_conf"         # mean / max / mean_conf
MODEL_AGG="prob_avg_weighted"  # logits_avg / prob_avg / voting / *_weighted
SAVE_LOGITS=true
WINDOW_TTA_OFFSETS="0,128,256,384" # 更丰富的起点视角，至少包含 0
MC_DROPOUT_RUNS=4              # 多次采样提升稳健性，留意耗时
DECISION_THRESHOLD=""          # 如需二分类阈值调优，示例: 0.55
CLASS_THRESHOLDS=""            # 若多分类阈值已调优，填入逗号分隔列表
TUNE_CLASS_THRESHOLDS=true     # 14 类场景可在验证集上网格搜索统一阈值
THRESHOLD_GRID="0.30,0.35,0.40,0.45,0.50,0.55,0.60"  # 搜索网格

# 设备与并行
DEVICE="cuda"
NUM_WORKERS=4
# ============================================================================

echo "=== 配置信息 ==="
echo "K-Fold 模型目录: ${KFOLD_DIR}"
echo "测试数据: ${TEST_PATH}"
if [ -n "${VAL_PATH}" ]; then
  echo "验证数据: ${VAL_PATH}"
fi
echo "输出文件: ${OUTPUT_PATH}"
echo "Batch Size: ${BATCH_SIZE}"
echo "窗口聚合: ${WINDOW_AGG}"
echo "模型聚合: ${MODEL_AGG}"
echo "保存 logits: ${SAVE_LOGITS}"
echo "窗口 TTA offsets: ${WINDOW_TTA_OFFSETS}"
echo "MC Dropout runs: ${MC_DROPOUT_RUNS}"
if [ -n "${DECISION_THRESHOLD}" ]; then
  echo "二分类阈值: ${DECISION_THRESHOLD}"
fi
if [ -n "${CLASS_THRESHOLDS}" ]; then
  echo "类别阈值: ${CLASS_THRESHOLDS}"
fi
echo ""

# 构建推理命令
CMD=(
  python -u scripts/infer_kfold.py
  --kfold-dir "${KFOLD_DIR}"
  --test-path "${TEST_PATH}"
  --output-path "${OUTPUT_PATH}"
  --batch-size ${BATCH_SIZE}
  --window-agg "${WINDOW_AGG}"
  --model-agg "${MODEL_AGG}"
  --device "${DEVICE}"
  --num-workers ${NUM_WORKERS}
  --window-tta-offsets "${WINDOW_TTA_OFFSETS}"
  --mc-dropout-runs ${MC_DROPOUT_RUNS}
)

if [ -n "${VAL_PATH}" ]; then
  CMD+=(--val-path "${VAL_PATH}")
fi

if [ "${SAVE_LOGITS}" = "true" ]; then
  CMD+=(--save-logits)
fi

if [ -n "${DECISION_THRESHOLD}" ]; then
  CMD+=(--decision-threshold "${DECISION_THRESHOLD}")
fi

if [ -n "${CLASS_THRESHOLDS}" ]; then
  CMD+=(--class-thresholds "${CLASS_THRESHOLDS}")
fi

if [ "${TUNE_CLASS_THRESHOLDS}" = "true" ] && [ -n "${VAL_PATH}" ]; then
  CMD+=(--tune-class-thresholds --threshold-grid "${THRESHOLD_GRID}")
fi

echo "执行推理命令:"
echo "  ${CMD[@]}"
echo ""

# 执行推理
CUDA_VISIBLE_DEVICES=0 "${CMD[@]}"

echo ""
echo "=== K-Fold 推理完成 ==="
if [ -f "${OUTPUT_PATH}" ]; then
  echo "生成的提交文件: ${OUTPUT_PATH}"
else
  echo "未找到输出文件，请检查日志。"
fi
echo "======================="

