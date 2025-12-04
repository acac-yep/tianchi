#!/bin/bash
#SBATCH -J ensemble_eval
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 2:00:00

# ============================================================================
# Ensemble 模型评估脚本
# 
# 说明:
#   在验证集上评估 ensemble 模型的 Macro-F1
#   自动扫描 checkpoints/ensemble 目录下的所有 seed checkpoint
#   
# 使用方法:
#   sbatch scripts/slurm_scripts/eval_ensemble.sh
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

echo "=== 开始 Ensemble 评估 ==="
echo "工作目录: $(pwd)"
echo ""

# 设置 PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# 列出将要评估的 checkpoint
echo "扫描 checkpoints/ensemble 目录..."
find checkpoints/ensemble -name "hat_cls_best.pt" 2>/dev/null
echo ""

# 运行 ensemble 评估
CUDA_VISIBLE_DEVICES=0 \
python -u scripts/ensemble_eval.py \
  --checkpoint-dir checkpoints/ensemble \
  --val-path data/processed/val.csv \
  --batch-size 64 \
  --device cuda \
  --num-workers 4 \
  --ensemble-method logits_avg \
  --output-dir outputs/ensemble \
  --save-predictions

echo ""
echo "=== Ensemble 评估完成 ==="
echo "结果保存在: outputs/ensemble/"

