#!/bin/bash
# ============================================================================
# 快速修复：使用 last checkpoint 做 ensemble
# 
# 说明：
#   由于多样性训练没有超过 baseline，但 last checkpoint 可能有不同的预测
#   这个脚本会使用 hat_cls_last.pt 做 ensemble
# ============================================================================

cd /data/home/scyb226/lzx/study/lab/tianchi

# 使用 last checkpoint
CHECKPOINT_DIR="checkpoints/diverse_ensemble"
VAL_PATH="data/processed/val.csv"
TEST_PATH="data/processed/test.csv"

echo "=== 使用 Last Checkpoint 做 Ensemble ==="

# 查找所有 last checkpoint
MODEL_PATHS=$(find "$CHECKPOINT_DIR" -name "hat_cls_last.pt" 2>/dev/null | sort | tr '\n' ',' | sed 's/,$//')

if [ -z "$MODEL_PATHS" ]; then
    echo "错误: 未找到任何 hat_cls_last.pt 文件"
    exit 1
fi

echo "找到的模型:"
echo "$MODEL_PATHS" | tr ',' '\n' | nl

# 在验证集上测试
echo ""
echo "=== 在验证集上测试 ==="
python scripts/ensemble_eval.py \
    --checkpoint-paths $(echo "$MODEL_PATHS" | tr ',' ' ') \
    --val-path "$VAL_PATH" \
    --ensemble-method temp_scaled \
    --temperature 1.5 \
    --batch-size 64

echo ""
echo "=== 如果结果可接受，可以继续生成测试集预测 ==="

