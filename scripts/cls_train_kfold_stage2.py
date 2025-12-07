#!/usr/bin/env python3
"""
HAT 模型 K-Fold Stage2 微调脚本

设计目标：
    - 在 Stage1 K-fold 基础上，加载 hat_cls_fold{k}_best.pt 做小学习率二次微调
    - 支持 R-Drop（双前向 + KL）与 FGM（embedding 级对抗扰动）
    - 轻量数据增强：随机滑窗起点（token 级随机偏移）
    - 每折只训少量 epoch（默认 1~2），warm-start + 小学习率
    - 若 Stage2 最优 F1 >= Stage1 最优，则保存为 hat_cls_fold{k}_stage2_best.pt，否则继续用原 ckpt
"""

import os
import sys
import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.losses import create_loss_fn
from src.data_preprocess import (
    HATDataset,
    HATDataCollator,
    create_tokenizer,
    create_segmenter,
)


# =============================================================================
# 工具函数
# =============================================================================

def log_print(*args, **kwargs):
    """带时间戳的 print 函数"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}]", *args, **kwargs)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# 数据增强：随机滑窗起点 Dataset
# =============================================================================


class Stage2HATDataset(HATDataset):
    """
    继承 HATDataset，新增 token 级随机偏移（随机滑窗起点）。

    - 每次 __getitem__ 时，若开启 random_offset，则在 [0, stride) 范围内随机偏移起点
    - 仅在训练模式下使用，缓存关闭以保证每个 epoch 都会重新分段
    """

    def __init__(
        self,
        *args,
        random_offset: bool = False,
        random_offset_stride: int = 128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.random_offset = random_offset
        self.random_offset_stride = max(1, int(random_offset_stride))

    def _process_text(self, idx: int):
        text = self.texts[idx]
        token_ids = self.tokenizer.encode(text)

        if (
            self.random_offset
            and len(token_ids) > self.segmenter.segment_length
        ):
            # 在 [0, stride) 内随机偏移起点
            max_offset = min(self.random_offset_stride, self.segmenter.segment_length)
            offset = np.random.randint(0, max_offset)
            token_ids = token_ids[offset:]

        segmented = self.segmenter.segment_document(
            token_ids,
            mode=self.mode if self.mode != "pretrain" else "train",
        )
        return segmented


# =============================================================================
# EMA & FGM
# =============================================================================


class ModelEMA:
    """指数移动平均权重，评估时使用"""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, device=None):
        self.model = model
        self.decay = decay
        self.device = device or next(model.parameters()).device
        self.ema_model = self._create_ema_model()
        self.ema_model.eval()

    def _create_ema_model(self):
        import copy

        ema_model = copy.deepcopy(self.model)
        ema_model.to(self.device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False
        return ema_model

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            if model_p.requires_grad:
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()


class FGM:
    """Fast Gradient Method，对 embedding 层添加单步对抗扰动"""

    def __init__(self, model, emb_name: str = "embeddings.word_embeddings", epsilon: float = 0.5):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.backup: Dict[str, torch.Tensor] = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                grad = param.grad
                norm = torch.norm(grad)
                if norm == 0 or torch.isnan(norm):
                    continue
                r_at = self.epsilon * grad / (norm + 1e-8)
                self.backup[name] = param.data.clone()
                param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# 损失计算：R-Drop
# =============================================================================


def compute_loss(
    model,
    batch,
    loss_fn,
    lambda_rdrop: float = 0.5,
    use_rdrop: bool = False,
):
    """计算主损失；返回 (loss, ce_loss, kl_loss)"""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    if not use_rdrop:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        ce_loss = loss_fn(logits, labels)
        return ce_loss, ce_loss, torch.tensor(0.0, device=logits.device)

    outputs1 = model(input_ids=input_ids, attention_mask=attention_mask)
    outputs2 = model(input_ids=input_ids, attention_mask=attention_mask)
    logits1 = outputs1["logits"]
    logits2 = outputs2["logits"]

    ce1 = loss_fn(logits1, labels)
    ce2 = loss_fn(logits2, labels)
    ce_loss = 0.5 * (ce1 + ce2)

    log_p1 = F.log_softmax(logits1, dim=-1)
    log_p2 = F.log_softmax(logits2, dim=-1)
    p1 = log_p1.exp()
    p2 = log_p2.exp()
    kl_12 = F.kl_div(log_p1, p2, reduction="batchmean")
    kl_21 = F.kl_div(log_p2, p1, reduction="batchmean")
    kl_loss = 0.5 * (kl_12 + kl_21)

    loss = ce_loss + lambda_rdrop * kl_loss
    return loss, ce_loss, kl_loss


# =============================================================================
# 评估
# =============================================================================


@torch.no_grad()
def evaluate(model, data_loader, device, loss_fn, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    all_labels, all_preds = [], []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            loss = loss_fn(logits, labels)

        total_loss += loss.item()
        total_steps += 1
        preds = logits.argmax(dim=-1)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    avg_loss = total_loss / max(1, total_steps)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1


# =============================================================================
# 训练单折
# =============================================================================


def train_single_fold(
    fold_idx: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    train_dataset,
    val_dataset,
    args,
    device: torch.device,
    class_weights_tensor: torch.Tensor,
    tokenizer,
    segmenter,
    collator,
    stage1_dir: Path,
) -> Tuple[float, float, Path]:
    """返回 (stage2_best_f1, stage1_best_f1, best_model_path)"""
    fold_seed = args.seed + fold_idx * 1000
    set_seed(fold_seed)

    log_print(f"\n{'='*70}")
    log_print(f"Stage2 | Fold {fold_idx + 1}/{args.n_folds} (seed={fold_seed})")
    log_print(f"{'='*70}")

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    log_print(f"训练集: {len(train_subset):,} | 验证集: {len(val_subset):,}")

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False,
    )

    # 创建模型 & 加载 Stage1 权重
    from src.model import create_model, HATConfig

    config = HATConfig()
    model = create_model(config).to(device)

    ckpt_path = stage1_dir / f"hat_cls_fold{fold_idx}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 Stage1 权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_to_load = ckpt.get("model_state_dict_original") or ckpt.get("model_state_dict") or ckpt
    missing, unexpected = model.load_state_dict(state_to_load, strict=False)
    log_print(f"加载 Stage1 权重: {ckpt_path}")
    log_print(f"  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    stage1_best_f1 = float(ckpt.get("val_macro_f1", 0.0))

    # 损失函数
    loss_fn = create_loss_fn(
        loss_type=args.loss_type,
        class_weights=class_weights_tensor,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        device=str(device),
    )
    loss_info = f"{args.loss_type}"
    if args.loss_type in ["smooth", "focal_smooth"]:
        loss_info += f" (smoothing={args.label_smoothing})"
    if args.loss_type in ["focal", "focal_smooth"]:
        loss_info += f" (gamma={args.focal_gamma})"
    log_print(f"损失: {loss_info}")
    if args.use_rdrop:
        log_print(f"R-Drop 开启，lambda={args.rdrop_alpha}")
    if args.use_fgm:
        log_print(f"FGM 开启，epsilon={args.fgm_epsilon}, ratio={args.fgm_loss_ratio}")

    # AMP / EMA / FGM
    scaler = GradScaler() if args.use_amp else None
    ema = ModelEMA(model, decay=args.ema_decay, device=device) if args.use_ema else None
    fgm = FGM(model, emb_name="embeddings.word_embeddings", epsilon=args.fgm_epsilon) if args.use_fgm else None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    log_print(f"总步数: {total_steps}, warmup: {warmup_steps}, lr: {args.lr}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_stage2_f1 = -1.0
    best_model_path = output_dir / f"hat_cls_fold{fold_idx}_stage2_best.pt"
    patience = args.early_stopping_patience
    patience_counter = 0
    start_time = time.time()
    global_step = 0

    for epoch in range(args.num_epochs):
        log_print(f"\n---- Fold {fold_idx + 1} | Epoch {epoch+1}/{args.num_epochs} ----")
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            optimizer_was_run = True

            if args.use_amp:
                with torch.amp.autocast(device_type=device.type):
                    loss, ce_loss, kl_loss = compute_loss(
                        model, batch, loss_fn,
                        lambda_rdrop=args.rdrop_alpha,
                        use_rdrop=args.use_rdrop,
                    )
                if torch.isnan(loss):
                    log_print(f"警告: step {global_step} NaN loss，跳过")
                    continue
                scaler.scale(loss).backward()

                if fgm is not None:
                    fgm.attack()
                    with torch.amp.autocast(device_type=device.type):
                        adv_loss, _, _ = compute_loss(
                            model, batch, loss_fn,
                            lambda_rdrop=args.rdrop_alpha,
                            use_rdrop=args.use_rdrop,
                        )
                        adv_loss = adv_loss * args.fgm_loss_ratio
                    scaler.scale(adv_loss).backward()
                    fgm.restore()

                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # 若出现 inf，GradScaler 会缩放 scale，表示本步已跳过优化器
                optimizer_was_run = scaler.get_scale() >= scale_before
            else:
                loss, ce_loss, kl_loss = compute_loss(
                    model, batch, loss_fn,
                    lambda_rdrop=args.rdrop_alpha,
                    use_rdrop=args.use_rdrop,
                )
                if torch.isnan(loss):
                    log_print(f"警告: step {global_step} NaN loss，跳过")
                    continue
                loss.backward()

                if fgm is not None:
                    fgm.attack()
                    adv_loss, _, _ = compute_loss(
                        model, batch, loss_fn,
                        lambda_rdrop=args.rdrop_alpha,
                        use_rdrop=args.use_rdrop,
                    )
                    adv_loss = adv_loss * args.fgm_loss_ratio
                    adv_loss.backward()
                    fgm.restore()

                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer_was_run = True

            if optimizer_was_run:
                scheduler.step()
            if ema is not None:
                ema.update(model)

            global_step += 1
            running_loss += loss.item()

            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                samples_per_sec = global_step * args.batch_size / max(1e-6, elapsed)
                log_print(
                    f"[Fold {fold_idx + 1} | Ep {epoch+1}] "
                    f"Step {global_step}/{total_steps} | "
                    f"Loss {avg_loss:.4f} (CE {ce_loss.item():.4f}, KL {kl_loss.item():.4f}) | "
                    f"LR {lr_now:.2e} | Speed {samples_per_sec:.1f} samples/s"
                )
                running_loss = 0.0

        # 验证
        eval_model = ema.ema_model if ema is not None else model
        val_loss, val_acc, val_f1 = evaluate(
            eval_model, val_loader, device, loss_fn=loss_fn, use_amp=args.use_amp
        )
        log_print(
            f"[Eval] Fold {fold_idx + 1} | Epoch {epoch+1} | "
            f"Val Loss {val_loss:.4f} | Acc {val_acc:.4f} | Macro-F1 {val_f1:.4f} "
            f"(Stage1 best {stage1_best_f1:.4f})"
        )

        if val_f1 > best_stage2_f1:
            best_stage2_f1 = val_f1
            patience_counter = 0
            if val_f1 >= stage1_best_f1:
                model_state = ema.state_dict() if ema is not None else eval_model.state_dict()
                torch.save(
                    {
                        "fold": fold_idx,
                        "epoch": epoch,
                        "step": global_step,
                        "model_state_dict": model_state,
                        "val_macro_f1": val_f1,
                        "stage1_best_f1": stage1_best_f1,
                        "args": vars(args),
                    },
                    best_model_path,
                )
                log_print(f"  >> Stage2 提升或持平，已保存: {best_model_path}")
            else:
                log_print("  >> Stage2 暂未超过 Stage1，不保存。")
        else:
            if patience is not None:
                patience_counter += 1
                if patience_counter >= patience:
                    log_print(f"Early stopping: {patience} 个 epoch 未提升")
                    break

    elapsed = time.time() - start_time
    log_print(
        f"\nFold {fold_idx + 1} 完成 | Stage2 best F1 {best_stage2_f1:.4f} | "
        f"Stage1 best {stage1_best_f1:.4f} | 用时 {elapsed/60:.1f} 分钟"
    )

    # 选择最终模型路径：若 Stage2 没有超过 Stage1，则返回 Stage1 路径
    final_path = best_model_path if best_stage2_f1 >= stage1_best_f1 and best_model_path.exists() else ckpt_path
    if final_path == ckpt_path:
        log_print("Stage2 未超过 Stage1，继续使用原模型。")
    else:
        log_print(f"Stage2 模型将用于推理: {final_path}")

    # 释放显存
    del model, optimizer, scheduler, loss_fn
    if ema is not None:
        del ema
    if scaler is not None:
        del scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_stage2_f1, stage1_best_f1, final_path


# =============================================================================
# 主流程
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="HAT 模型 K-Fold Stage2 微调脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 数据
    parser.add_argument(
        "--train-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "train.csv"),
        help="训练数据路径（预处理后的 train.csv）",
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "class_weights.npy"),
        help="类别权重文件路径（np.float32[num_labels]）",
    )
    # K-Fold
    parser.add_argument("--n-folds", type=int, default=5, help="K-Fold 折数")
    parser.add_argument("--fold-seed", type=int, default=42, help="K-Fold 随机种子（保持与 Stage1 一致）")
    # Stage1 & 输出
    parser.add_argument(
        "--stage1-dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "cls_hat512_kfold"),
        help="Stage1 K-fold ckpt 目录（包含 hat_cls_fold{k}_best.pt）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "cls_hat512_kfold_stage2"),
        help="Stage2 输出目录（保存 hat_cls_fold{k}_stage2_best.pt）",
    )
    # 训练超参
    parser.add_argument("--batch-size", type=int, default=64, help="训练 batch size")
    parser.add_argument("--eval-batch-size", type=int, default=128, help="验证 batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Stage2 学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--num-epochs", type=int, default=2, help="Stage2 训练 epoch 数")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="warmup 步数比例")
    parser.add_argument("--log-every", type=int, default=50, help="日志打印间隔（step）")
    # 设备 & 随机性
    parser.add_argument("--device", type=str, default="cuda", help="设备（cuda/cpu）")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数")
    parser.add_argument("--seed", type=int, default=42, help="训练随机种子")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值，0 表示不裁剪")
    # 损失
    parser.add_argument(
        "--loss-type",
        type=str,
        default="smooth",
        choices=["ce", "smooth", "focal", "focal_smooth"],
        help="损失函数类型",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="标签平滑系数")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal gamma")
    # Regularization
    parser.add_argument("--early-stopping-patience", type=int, default=1, help="早停 patience，None 关闭")
    parser.add_argument("--use-amp", action="store_true", help="使用 AMP 混合精度")
    parser.add_argument("--use-ema", action="store_true", help="使用 EMA 权重验证/保存")
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="EMA 衰减率")
    # R-Drop
    parser.add_argument("--use-rdrop", action="store_true", help="启用 R-Drop 一致性正则")
    parser.add_argument("--rdrop-alpha", type=float, default=0.5, help="R-Drop KL 系数")
    # FGM
    parser.add_argument("--use-fgm", action="store_true", help="启用 FGM 对抗训练")
    parser.add_argument("--fgm-epsilon", type=float, default=0.5, help="FGM 扰动尺度 epsilon")
    parser.add_argument("--fgm-loss-ratio", type=float, default=1.0, help="对抗损失权重系数")
    # 数据增强
    parser.add_argument(
        "--random-offset",
        action="store_true",
        help="训练时启用随机滑窗起点（token 级随机偏移）",
    )
    parser.add_argument(
        "--random-offset-stride",
        type=int,
        default=128,
        help="随机偏移的最大范围（0~stride-1）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log_print(f"使用设备: {device}")
    if device.type == "cuda":
        log_print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. 读数据
    train_df = pd.read_csv(args.train_path, sep="\t")
    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].astype(int).tolist()
    log_print(f"训练样本: {len(train_texts):,}")

    # 2. K-Fold 划分
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.fold_seed)
    fold_splits = list(skf.split(train_texts, train_labels))

    log_print("\n各 Fold 分布:")
    for fold_idx, (tr, va) in enumerate(fold_splits):
        log_print(f"  Fold {fold_idx + 1}: train {len(tr):,} | val {len(va):,}")

    # 3. 准备分词/分段/数据集
    tokenizer = create_tokenizer()
    segmenter = create_segmenter()
    collator = HATDataCollator()

    # 训练集：开启随机偏移；验证集：关闭随机偏移、可缓存以固定切分
    train_dataset = Stage2HATDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        segmenter=segmenter,
        mode="train",
        cache_segments=False,
        random_offset=args.random_offset,
        random_offset_stride=args.random_offset_stride,
    )
    val_dataset = HATDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        segmenter=segmenter,
        mode="train",
        cache_segments=True,
    )

    # 4. 类别权重
    class_weights = np.load(args.class_weights).astype("float32")
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    log_print(f"类别权重示例: {class_weights[:5]}")

    # 5. 训练各 fold
    stage1_dir = Path(args.stage1_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    total_start = time.time()
    for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
        best_stage2_f1, stage1_best_f1, final_path = train_single_fold(
            fold_idx=fold_idx,
            train_indices=train_indices,
            val_indices=val_indices,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=args,
            device=device,
            class_weights_tensor=class_weights_tensor,
            tokenizer=tokenizer,
            segmenter=segmenter,
            collator=collator,
            stage1_dir=stage1_dir,
        )
        fold_results.append(
            {
                "fold": fold_idx,
                "stage1_best_f1": stage1_best_f1,
                "stage2_best_f1": best_stage2_f1,
                "final_path": final_path,
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
            import gc

            gc.collect()
            log_print("已清理显存，进入下一折")

    total_elapsed = time.time() - total_start
    log_print("\n================== Stage2 总结 ==================")
    log_print(f"总耗时: {total_elapsed/60:.1f} 分钟")
    stage1_f1s = [f["stage1_best_f1"] for f in fold_results]
    stage2_f1s = [f["stage2_best_f1"] for f in fold_results]
    log_print(f"Stage1 平均 F1: {np.mean(stage1_f1s):.4f}")
    log_print(f"Stage2 平均 F1: {np.mean(stage2_f1s):.4f}")
    log_print("\n最终用于推理的模型路径：")
    for r in fold_results:
        log_print(f"  Fold {r['fold'] + 1}: {r['final_path']}")
    log_print("\n推理示例：")
    log_print(
        f"  python scripts/infer_kfold.py --kfold-dir {output_dir} "
        f"--test-path data/processed/test.csv "
        f"--output-path outputs/submission/submission_kfold_stage2.csv"
    )


if __name__ == "__main__":
    main()

