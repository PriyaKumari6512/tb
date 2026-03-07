"""
Training loop for SegFormer B4 segmentation model.
Features: AMP, DataParallel, differential LR, cosine warmup, early stopping.
"""

import argparse
import logging
import os
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from segformer_b4.model import build_segformer, build_criterion
from segformer_b4.dataset import build_dataloaders
from segformer_b4.utils import (
    load_config, set_seed, get_device, save_checkpoint,
    load_checkpoint, dice_score, log_metrics, setup_logging,
)

logger = logging.getLogger(__name__)


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
        else:
            import math
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * factor)


def validate(model, val_loader, criterion, device, amp_enabled):
    model.eval()
    total_loss = 0
    total_dice = 0
    count = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            with autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(images)
                logits_up = nn.functional.interpolate(
                    logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
                loss = criterion(logits_up, masks)

            pred = logits_up.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_dice += dice_score(pred, masks) * images.size(0)
            count += images.size(0)

    return total_loss / count, total_dice / count


def train(cfg: dict, resume_path: str = None):
    setup_logging(cfg.get("log_dir", "logs"))
    set_seed(cfg.get("seed", 42))
    device = get_device()
    logger.info(f"Device: {device}")

    # Data
    loaders = build_dataloaders(cfg)
    logger.info(f"Train: {len(loaders['train'].dataset)}, Val: {len(loaders['val'].dataset)}")

    # Model
    model_cfg = cfg["model"]
    model = build_segformer(
        variant=model_cfg.get("variant", "b4"),
        num_classes=model_cfg.get("num_classes", 2),
        pretrained_weights=model_cfg.get("pretrained_weights", None),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
    )
    model = model.to(device)

    # Multi-GPU
    train_cfg = cfg["training"]
    if train_cfg.get("multi_gpu", False) and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Loss
    loss_cfg = train_cfg.get("loss", {})
    criterion = build_criterion(
        loss_type=loss_cfg.get("type", "dice_bce"),
        pos_weight=loss_cfg.get("pos_weight", 5.0),
    ).to(device)

    # Optimizer with differential LR
    lr = train_cfg["lr"]
    lr_mult = train_cfg.get("encoder_lr_mult", 0.1)
    base_model = model.module if hasattr(model, "module") else model
    param_groups = base_model.get_param_groups(lr, lr_mult)
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # Scheduler
    epochs = train_cfg["epochs"]
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
        total_epochs=epochs,
    )

    # AMP
    amp_enabled = train_cfg.get("amp", True) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    # Resume
    start_epoch = 0
    best_dice = 0.0
    if resume_path and os.path.exists(resume_path):
        state = load_checkpoint(resume_path, base_model, optimizer, device)
        start_epoch = state.get("epoch", 0) + 1
        best_dice = state.get("best_dice", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best_dice={best_dice:.4f}")

    # TensorBoard
    ckpt_dir = cfg.get("ckpt_dir", "checkpoints")
    log_dir = cfg.get("log_dir", "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(log_dir, "tb_segformer"))

    # Early stopping
    patience = train_cfg.get("patience", 20)
    no_improve = 0
    grad_clip = train_cfg.get("grad_clip", 1.0)

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        scheduler.step(epoch)
        epoch_loss = 0
        epoch_dice = 0
        n_batches = 0
        t0 = time.time()

        for images, masks in loaders["train"]:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            with autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(images)
                logits_up = nn.functional.interpolate(
                    logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
                loss = criterion(logits_up, masks)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            pred = logits_up.detach().argmax(dim=1)
            epoch_loss += loss.item()
            epoch_dice += dice_score(pred, masks)
            n_batches += 1

        train_loss = epoch_loss / n_batches
        train_dice = epoch_dice / n_batches
        val_loss, val_dice = validate(model, loaders["val"], criterion, device, amp_enabled)
        elapsed = time.time() - t0

        cur_lr = optimizer.param_groups[-1]["lr"]
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} | "
            f"lr={cur_lr:.2e} | {elapsed:.1f}s"
        )

        # TensorBoard
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("dice", {"train": train_dice, "val": val_dice}, epoch)
        writer.add_scalar("lr", cur_lr, epoch)

        # JSONL metrics
        log_metrics({
            "epoch": epoch + 1, "train_loss": train_loss, "train_dice": train_dice,
            "val_loss": val_loss, "val_dice": val_dice, "lr": cur_lr,
        }, os.path.join(log_dir, "segformer_metrics.jsonl"))

        # Checkpointing
        save_state = {
            "epoch": epoch,
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": max(best_dice, val_dice),
            "config": cfg,
        }

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(save_state, os.path.join(ckpt_dir, "segformer_best.pth"))
            no_improve = 0
            logger.info(f"  ★ New best val_dice: {best_dice:.4f}")
        else:
            no_improve += 1

        if (epoch + 1) % train_cfg.get("save_every", 10) == 0:
            save_checkpoint(save_state, os.path.join(ckpt_dir, "segformer_latest.pth"))

        # Early stopping
        if no_improve >= patience:
            logger.info(f"Early stopping — no improvement for {patience} epochs")
            break

    writer.close()
    logger.info(f"Training complete. Best val_dice: {best_dice:.4f}")
    return best_dice


def main():
    parser = argparse.ArgumentParser(description="SegFormer B4 Training")
    parser.add_argument("--config", type=str, default="configs/segformer_config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smoke_test:
        cfg["training"]["epochs"] = 2

    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
