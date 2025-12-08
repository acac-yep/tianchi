# Tianchi NLP新闻分类 

基于 HAT-Interleaved 模型的长文本分类方案，面向天池 2025 赛题：NLP新闻分类。项目包含数据分析、数据预处理、MLM 预训练、分类训练、K-Fold/Stage2 微调与推理全流程脚本。

## 环境依赖
- Python 3.9+
- PyTorch ≥ 2.0（根据 CUDA 版本安装对应轮子）
- 其余依赖见 `requirements.txt`

示例安装（需先自行安装匹配的 PyTorch）:
```bash
pip install -r requirements.txt
```

## 目录结构
```
.
├── data_analyze/                # 简单数据分析脚本与结果
├── docs/                        # 技术规格文档
│   └── technical_specification.md
├── outputs/
│   └── submission/              # 推理产生的提交文件
├── report/                      # 论文/报告（课程报告）
├── scripts/                     # 可直接运行的训练/推理脚本
│   ├── run_preprocessing.py     # 数据预处理
│   ├── mlm_train.py             # MLM 预训练
│   ├── cls_train.py             # 单模型分类微调
│   ├── cls_train_kfold.py       # K-Fold 训练
│   ├── cls_train_kfold_stage2.py# Stage2 二次微调 (R-Drop/FGM 等)
│   ├── infer.py                 # 单/多模型推理
│   ├── infer_kfold.py           # K-Fold 模型自动 ensemble 推理
│   └── slurm_scripts/           # 相关Slurm 启动脚本
├── src/
│   ├── common_config.py         # 共享超参（vocab/长度/类别数）
│   ├── data_preprocess/         # 分词、分段、清洗、配置
│   ├── model/                   # HAT 模型定义
│   └── losses.py                # 训练损失定义
├── requirements.txt
└── README.md
```

## 数据准备
将官方数据放入 `data/`（与项目同级），包含：
- `train_set.csv`：列 `text`（空格分隔原始 token id）、`label`
- `test_a.csv`：列 `text`

## 使用流程
### 0) 数据分析
见`data_analyze/`下简单数据分析脚本与结果
### 1) 数据预处理
生成重映射后的 CSV、类别权重与 tokenizer：
```bash
python scripts/run_preprocessing.py \
  --data-dir data \
  --output-dir data/processed \
  --val-ratio 0.1 --seed 42
```
输出：
- `data/processed/train.csv`, `val.csv`, `test.csv`
- `data/processed/class_weights.npy`
- `data/processed/tokenizer/tokenizer_config.json`

### 2) MLM 预训练（可选但推荐）
```bash
python scripts/mlm_train.py \
  --train-path data/processed/train.csv \
  --output-dir checkpoints/mlm_hat512 \
  --batch-size 4 --max-steps 10000 --warmup-ratio 0.06
```
产物：`checkpoints/mlm_hat512/hat_mlm_best.pt` 与 `hat_mlm_final.pt`

### 3) 分类微调（基础，效果不好）
```bash
python scripts/cls_train.py \
  --train-path data/processed/train.csv \
  --val-path data/processed/val.csv \
  --class-weights data/processed/class_weights.npy \
  --mlm-ckpt checkpoints/mlm_hat512/hat_mlm_final.pt \
  --output-dir checkpoints/cls_hat512 \
  --batch-size 64 --lr 1e-4 --num-epochs 5
```

### 4) K-Fold 训练 (实际使用)
```bash
python scripts/cls_train_kfold.py \
  --train-path data/processed/train.csv \
  --class-weights data/processed/class_weights.npy \
  --mlm-ckpt checkpoints/mlm_hat512/hat_mlm_final.pt \
  --output-dir checkpoints/cls_hat512_kfold \
  --n-folds 5 --loss-type smooth --label-smoothing 0.05 \
  --use-amp --use-ema
```
每折保存 `hat_cls_fold{k}_best.pt`。

### 5) Stage2 二次微调（可选）
在 K-Fold 基础上做小学习率+R-Drop/FGM：
```bash
python scripts/cls_train_kfold_stage2.py \
  --train-path data/processed/train.csv \
  --class-weights data/processed/class_weights.npy \
  --mlm-ckpt checkpoints/mlm_hat512/hat_mlm_final.pt \
  --output-dir checkpoints/cls_hat512_kfold \
  --n-folds 5 --use-amp --use-ema --use-rdrop --lambda-rdrop 0.5 \
  --use-fgm --fgm-epsilon 0.5 --num-epochs 2 --lr 1e-5
```
产物：`hat_cls_fold{k}_stage2_best.pt`（若指标更好则替换）。

### 6) 推理与提交
- 单/多模型：
```bash
python scripts/infer.py \
  --test-path data/processed/test.csv \
  --model-paths checkpoints/cls_hat512/hat_cls_best.pt \
  --output-path outputs/submission/submission.csv \
  --batch-size 64
```
多模型 ensemble 用逗号分隔 `--model-paths`。

- 自动扫描 K-Fold 目录并 ensemble：
```bash
python scripts/infer_kfold.py \
  --kfold-dir checkpoints/cls_hat512_kfold \
  --test-path data/processed/test.csv \
  --output-path outputs/submission/submission_kfold.csv \
  --window-agg mean --model-agg prob_avg_weighted
```

## Slurm 使用示例
在集群环境下，推荐使用 `scripts/slurm_scripts/` 下的模板提交作业。常见用法：
```bash
# 进入项目根目录
# 预处理/训练/推理按需选择脚本并修改超参
sbatch scripts/slurm_scripts/cls_train_kfold.sbatch
```

脚本内关键变量：
- `GPUS`, `CPUS_PER_TASK`, `MEM_PER_CPU`：资源请求
- `OUTPUT_DIR`：日志与模型保存目录
- `CONDA_ENV` 或 `PYTHON`：环境配置
- `CMD`：实际运行命令，可替换为 `run_preprocessing.py`、`mlm_train.py`、`cls_train.py`、`cls_train_kfold_stage2.py`、`infer_kfold.py` 等

提交后可用 `squeue -u <username>` 查看队列，`sacct -j <job_id>` 查看完成状态，`scancel <job_id>` 取消任务。日志通常写入 `slurm-%j.out` 或脚本内指定的 `OUTPUT_LOG`。

## 训练配置（H800）
本项目使用H800进行训练，如果使用其他环境，请自行修改配置
- 当前训练脚本默认单卡运行，如需多卡请手动调整 `CUDA_VISIBLE_DEVICES`，本项目未内置 DDP/ZeRO，谨慎修改。
- 当前推理脚本默认多卡运行，否则回退

