"""
Swin2SR Finetuning on the DDS3 TB microscopy dataset.

Features:
  - Pretrained HuggingFace Swin2SR x4 model
  - Staged fine-tuning (L1 → L1 + Perceptual)
  - Differential learning rates (encoder vs head)
  - Optional encoder freezing
  - Mixed precision (AMP)
  - Multi-GPU (DataParallel)
  - Cosine schedule with warmup
  - Best + periodic checkpointing
  - Early stopping
"""

import argparse
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from swin2sr.dataset import build_dataloaders
from swin2sr.model import build_model
from swin2sr.utils import (
    compute_psnr,
    compute_ssim,
    get_device,
    load_checkpoint,
    load_config,
    log_metrics,
    print_env_info,
    save_checkpoint,
    set_seed,
    setup_logging,
    tensor_to_img,
)

logger = logging.getLogger(__name__)

# Reconstruction head layer name patterns (not pretrained body).
_HEAD_KEYWORDS = ("upsample", "conv_last", "final_convolution", "reconstruction")


def _is_head_param(name: str) -> bool:
    """Return True if parameter name belongs to the reconstruction head."""
    return any(kw in name.lower() for kw in _HEAD_KEYWORDS)


# =============================================================================
# Perceptual Loss (VGG-based)
# =============================================================================

class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG-19 features."""

    def __init__(self, layer_weights=None):
        super().__init__()
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),   # relu1_2
            nn.Sequential(*list(vgg.children())[4:9]),  # relu2_2
            nn.Sequential(*list(vgg.children())[9:18]), # relu3_4
            nn.Sequential(*list(vgg.children())[18:27]),# relu4_4
        ])
        self.weights = layer_weights or [1.0, 1.0, 1.0, 1.0]
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std
        loss = 0.0
        x_sr, x_hr = sr, hr
        for i, layer in enumerate(self.slices):
            x_sr = layer(x_sr)
            x_hr = layer(x_hr)
            loss += self.weights[i] * nn.functional.l1_loss(x_sr, x_hr)
        return loss


# =============================================================================
# Learning rate scheduler with warmup
# =============================================================================

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
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def validate(model, val_loader, device):
    """Compute validation PSNR and SSIM."""
    model.eval()
    psnr_list, ssim_list = [], []

    for batch in val_loader:
        lr = batch["lr"].to(device)
        hr = batch["hr"]

        with autocast(device_type="cuda", enabled=device.type == "cuda"):
            sr = model(lr)

        for i in range(sr.shape[0]):
            sr_img = tensor_to_img(sr[i])
            hr_img = tensor_to_img(hr[i])
            psnr_list.append(compute_psnr(sr_img, hr_img))
            ssim_list.append(compute_ssim(sr_img, hr_img))

    avg_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0
    avg_ssim = sum(ssim_list) / len(ssim_list) if ssim_list else 0
    model.train()
    return avg_psnr, avg_ssim


# =============================================================================
# Training
# =============================================================================

def train(cfg: dict, smoke_test: bool = False):
    """Main finetuning function for Swin2SR on DDS3."""
    train_cfg = cfg["training"]
    ckpt_cfg = cfg["checkpoint"]
    log_cfg = cfg["logging"]

    # Setup
    set_seed(train_cfg.get("seed", 42))
    run_dir = setup_logging(log_cfg["dir"], log_cfg["run_id"])
    device = get_device()
    print_env_info()

    # Save config snapshot
    import yaml
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Data
    train_loader, val_loader, _ = build_dataloaders(cfg)

    # Model
    model = build_model(cfg)
    model = model.to(device)

    # Multi-GPU
    if train_cfg.get("multi_gpu", False) and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # Optimizer — differential LR for finetuning pretrained model
    if cfg["model"].get("pretrained", True):
        body_params = []
        head_params = []
        for name, param in model.named_parameters():
            if _is_head_param(name):
                head_params.append(param)
            else:
                body_params.append(param)
        encoder_lr_mult = train_cfg.get("encoder_lr_mult", 0.1)
        optimizer = torch.optim.AdamW([
            {"params": body_params, "lr": train_cfg["lr"] * encoder_lr_mult},
            {"params": head_params, "lr": train_cfg["lr"]},
        ], weight_decay=train_cfg.get("weight_decay", 0),
           betas=tuple(train_cfg.get("betas", [0.9, 0.99])))
        logger.info(f"Finetuning: body LR={train_cfg['lr'] * encoder_lr_mult:.2e}, "
                     f"head LR={train_cfg['lr']:.2e}")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("weight_decay", 0),
            betas=tuple(train_cfg.get("betas", [0.9, 0.99])),
        )

    # Scheduler
    total_epochs = 2 if smoke_test else train_cfg["epochs"]
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
        total_epochs=total_epochs,
        min_lr=train_cfg.get("min_lr", 1e-7),
    )

    # Loss
    pixel_loss_fn = nn.L1Loss() if train_cfg["loss"]["pixel"] == "l1" else nn.MSELoss()
    perceptual_loss_fn = None
    if train_cfg["loss"].get("perceptual", False):
        perceptual_loss_fn = VGGPerceptualLoss().to(device)
        perceptual_loss_fn.eval()
    perceptual_weight = train_cfg["loss"].get("perceptual_weight", 0.1)
    stage1_epochs = train_cfg["loss"].get("staged_finetuning", {}).get("stage1_epochs", 30)

    # AMP
    use_amp = train_cfg.get("amp", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Resume
    start_epoch = 0
    best_psnr = 0.0
    patience_counter = 0
    if ckpt_cfg.get("resume"):
        ckpt = load_checkpoint(ckpt_cfg["resume"], model, optimizer, scheduler, scaler, device)
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt["metrics"].get("val_psnr", 0)
        logger.info(f"Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")

    # Tensorboard
    tb_writer = None
    if log_cfg.get("tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(os.path.join(run_dir, "tensorboard"))
        except ImportError:
            logger.warning("tensorboard not installed, skipping")

    metrics_file = os.path.join(run_dir, "metrics.jsonl")
    logger.info(f"Starting Swin2SR finetuning for {total_epochs} epochs")
    logger.info(f"Stage 1 (L1 only): epochs 0-{stage1_epochs - 1}")
    if perceptual_loss_fn:
        logger.info(f"Stage 2 (L1 + perceptual): epochs {stage1_epochs}+")

    # Optional encoder freeze during initial epochs
    freeze_encoder_epochs = train_cfg.get("freeze_encoder_epochs", 0)
    if freeze_encoder_epochs > 0:
        logger.info(f"Freezing encoder for first {freeze_encoder_epochs} epochs")

    for epoch in range(start_epoch, total_epochs):
        model.train()

        # Freeze/unfreeze encoder
        if freeze_encoder_epochs > 0:
            base_model = model.module if hasattr(model, "module") else model
            freeze = epoch < freeze_encoder_epochs
            for name, param in base_model.named_parameters():
                if not _is_head_param(name):
                    param.requires_grad = not freeze
            if epoch == freeze_encoder_epochs:
                logger.info(f"Epoch {epoch}: unfreezing encoder parameters")

        epoch_loss = 0.0
        use_perceptual = perceptual_loss_fn is not None and epoch >= stage1_epochs
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_amp):
                sr = model(lr)
                loss = pixel_loss_fn(sr, hr)

                if use_perceptual:
                    loss = loss + perceptual_weight * perceptual_loss_fn(sr, hr)

            scaler.scale(loss).backward()

            if train_cfg.get("clip_grad", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["clip_grad"])

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if smoke_test and batch_idx >= 5:
                break

        epoch_loss /= max(batch_idx + 1, 1)
        scheduler.step()

        # Validation
        val_psnr, val_ssim = validate(model, val_loader, device)

        # Logging
        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[0]["lr"]
        metrics = {
            "epoch": epoch,
            "train_loss": round(epoch_loss, 6),
            "val_psnr": round(val_psnr, 4),
            "val_ssim": round(val_ssim, 6),
            "lr": lr_current,
            "time_sec": round(elapsed, 1),
            "stage": "perceptual" if use_perceptual else "l1",
        }
        log_metrics(metrics, metrics_file)
        logger.info(
            f"Epoch {epoch:03d} | loss: {epoch_loss:.4f} | "
            f"PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f} | "
            f"lr: {lr_current:.2e} | time: {elapsed:.0f}s"
        )

        if tb_writer:
            tb_writer.add_scalar("train/loss", epoch_loss, epoch)
            tb_writer.add_scalar("val/psnr", val_psnr, epoch)
            tb_writer.add_scalar("val/ssim", val_ssim, epoch)
            tb_writer.add_scalar("train/lr", lr_current, epoch)

        # Checkpointing
        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics,
                os.path.join(ckpt_cfg["dir"], "best_model.pth"), scaler,
            )
        else:
            patience_counter += 1

        if (epoch + 1) % train_cfg.get("save_every", 10) == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics,
                os.path.join(ckpt_cfg["dir"], f"epoch_{epoch:03d}.pth"), scaler,
            )

        # Early stopping
        if patience_counter >= train_cfg.get("patience", 30):
            logger.info(f"Early stopping at epoch {epoch} (patience={train_cfg['patience']})")
            break

    logger.info(f"Training complete. Best PSNR: {best_psnr:.2f} dB")
    if tb_writer:
        tb_writer.close()
    return best_psnr


def main():
    parser = argparse.ArgumentParser(description="Finetune Swin2SR super-resolution model")
    parser.add_argument("--config", type=str, default="swin2sr/configs/swin2sr_config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--smoke_test", action="store_true", help="Quick 2-epoch smoke test")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.resume:
        cfg["checkpoint"]["resume"] = args.resume

    train(cfg, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
