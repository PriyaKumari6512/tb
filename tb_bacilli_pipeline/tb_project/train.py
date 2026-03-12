"""
TB SegFormer Training with:
  - Dice + BCE combined loss (handles class imbalance)
  - WeightedRandomSampler (oversamples bacilli-rich images)
  - Mixed precision (AMP)
  - Multi-GPU (DataParallel)
  - Cosine schedule with warmup
  - Best + periodic checkpointing
  - Early stopping
"""

import argparse
import logging
import os
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from tb_project.dataset import build_dataloaders
from tb_project.model import build_criterion, build_model
from tb_project.utils import (
    compute_all_metrics,
    dice_score,
    get_device,
    load_checkpoint,
    load_config,
    log_metrics,
    print_env_info,
    save_checkpoint,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            import math
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Compute validation loss, Dice, IoU."""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    count = 0

    for batch in val_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(images)
            # Extract class-1 logit for binary loss
            binary_logits = logits[:, 1:2] - logits[:, 0:1]
            binary_targets = masks.unsqueeze(1).float()
            loss = criterion(binary_logits, binary_targets)

        # Dice on predictions
        probs = torch.softmax(logits, dim=1)
        preds = (probs[:, 1] > 0.5).long()

        for i in range(preds.shape[0]):
            total_dice += float(dice_score(preds[i], masks[i]))
            count += 1

        total_loss += loss.item() * images.shape[0]

    avg_loss = total_loss / max(count, 1)
    avg_dice = total_dice / max(count, 1)
    model.train()
    return avg_loss, avg_dice


def train(cfg: dict, smoke_test: bool = False):
    """Main TB training function."""
    train_cfg = cfg["training"]
    ckpt_cfg = cfg["checkpoint"]
    log_cfg = cfg["logging"]

    set_seed(train_cfg.get("seed", 42))
    run_dir = setup_logging(log_cfg["dir"], log_cfg["run_id"])
    device = get_device()
    print_env_info()

    # Save config
    import yaml
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Data
    train_loader, val_loader, _ = build_dataloaders(cfg)

    # Model
    model = build_model(cfg)
    model = model.to(device)

    if train_cfg.get("multi_gpu", False) and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # Loss
    criterion = build_criterion(cfg).to(device)

    # Optimizer (different LR for backbone vs. decode head)
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "decode_head" in name or "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": train_cfg["lr"] * 0.1},  # Lower LR for pretrained backbone
        {"params": head_params, "lr": train_cfg["lr"]},
    ], weight_decay=train_cfg.get("weight_decay", 0.01),
       betas=tuple(train_cfg.get("betas", [0.9, 0.999])))

    total_epochs = 2 if smoke_test else train_cfg["epochs"]
    scheduler = CosineWarmupScheduler(
        optimizer, train_cfg.get("warmup_epochs", 3),
        total_epochs, train_cfg.get("min_lr", 1e-7),
    )

    # AMP
    use_amp = train_cfg.get("amp", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Resume
    start_epoch = 0
    best_dice = 0.0
    patience_counter = 0
    if ckpt_cfg.get("resume"):
        ckpt = load_checkpoint(ckpt_cfg["resume"], model, optimizer, scheduler, scaler, device)
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt["metrics"].get("val_dice", 0)

    # Tensorboard
    tb_writer = None
    if log_cfg.get("tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(os.path.join(run_dir, "tensorboard"))
        except ImportError:
            pass

    metrics_file = os.path.join(run_dir, "metrics.jsonl")
    logger.info(f"Starting TB training for {total_epochs} epochs")

    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_loss = 0
        epoch_dice = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                # Extract class-1 logit for binary loss
                binary_logits = logits[:, 1:2] - logits[:, 0:1]
                binary_targets = masks.unsqueeze(1).float()
                loss = criterion(binary_logits, binary_targets)

            scaler.scale(loss).backward()

            if train_cfg.get("clip_grad", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["clip_grad"])

            scaler.step(optimizer)
            scaler.update()

            # Training Dice
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                preds = (probs[:, 1] > 0.5).long()
                batch_dice = float(dice_score(preds.view(-1), masks.view(-1)))

            epoch_loss += loss.item()
            epoch_dice += batch_dice
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{batch_dice:.3f}"})

            if smoke_test and batch_idx >= 5:
                break

        epoch_loss /= max(batch_idx + 1, 1)
        epoch_dice /= max(batch_idx + 1, 1)
        scheduler.step()

        # Validation
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[0]["lr"]
        metrics = {
            "epoch": epoch,
            "train_loss": round(epoch_loss, 6),
            "train_dice": round(epoch_dice, 4),
            "val_loss": round(val_loss, 6),
            "val_dice": round(val_dice, 4),
            "lr": lr_current,
            "time_sec": round(elapsed, 1),
        }
        log_metrics(metrics, metrics_file)
        logger.info(
            f"Epoch {epoch:03d} | loss: {epoch_loss:.4f} | dice: {epoch_dice:.3f} | "
            f"val_loss: {val_loss:.4f} | val_dice: {val_dice:.3f} | time: {elapsed:.0f}s"
        )

        if tb_writer:
            tb_writer.add_scalar("train/loss", epoch_loss, epoch)
            tb_writer.add_scalar("train/dice", epoch_dice, epoch)
            tb_writer.add_scalar("val/loss", val_loss, epoch)
            tb_writer.add_scalar("val/dice", val_dice, epoch)

        # Checkpointing
        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics,
                os.path.join(ckpt_cfg["dir"], "best_model.pth"), scaler,
            )
        else:
            patience_counter += 1

        if (epoch + 1) % train_cfg.get("save_every", 5) == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics,
                os.path.join(ckpt_cfg["dir"], f"epoch_{epoch:03d}.pth"), scaler,
            )

        if patience_counter >= train_cfg.get("patience", 20):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    logger.info(f"Training complete. Best Dice: {best_dice:.4f}")
    if tb_writer:
        tb_writer.close()
    return best_dice


def main():
    parser = argparse.ArgumentParser(description="Train SegFormer for TB segmentation")
    parser.add_argument("--config", type=str, default="configs/tb_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.resume:
        cfg["checkpoint"]["resume"] = args.resume

    train(cfg, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
