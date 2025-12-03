#!/bin/bash
#SBATCH -J cls_train_hat512
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 24:00:00

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
# 需要添加cudnn、cuda、torch的库路径
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib:/data/apps/cuda/12.4/lib64:$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib

# 设置PYTHONPATH（如果需要）
# export PYTHONPATH="/data/home/scyb226/lzx/Megatron-LM:$PYTHONPATH"

echo "=== 环境变量检查 ==="
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# 切换到项目根目录
cd /data/home/scyb226/lzx/study/lab/tianchi

echo "=== 开始分类训练 ==="
echo "工作目录: $(pwd)"
echo ""

# 单卡训练命令
CUDA_VISIBLE_DEVICES=0 \
python scripts/cls_train.py \
  --train-path data/processed/train.csv \
  --val-path data/processed/val.csv \
  --class-weights data/processed/class_weights.npy \
  --mlm-ckpt checkpoints/mlm_hat512/hat_mlm_final.pt \
  --output-dir checkpoints/cls_hat512 \
  --batch-size 64 \
  --eval-batch-size 128 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --num-epochs 5 \
  --warmup-ratio 0.06 \
  --log-every 50 \
  --grad-clip 1.0 \
  --device cuda \
  --num-workers 4 \
  --seed 42

echo ""
echo "=== 训练完成 ==="

