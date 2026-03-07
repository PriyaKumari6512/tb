"""
SR Evaluation — compute PSNR/SSIM on test set + generate visual comparison grids.
"""

import argparse
import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from sr_project.dataset import build_dataloaders
from sr_project.model import build_model
from sr_project.utils import (
    compute_psnr,
    compute_ssim,
    get_device,
    load_checkpoint,
    load_config,
    save_image,
    tensor_to_img,
)

logger = logging.getLogger(__name__)


def evaluate(cfg: dict, checkpoint_path: str, split: str = "test"):
    """Evaluate SR model on val or test set."""
    logging.basicConfig(level=logging.INFO)
    device = get_device()

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    loader = test_loader if split == "test" else val_loader

    # Model
    model = build_model(cfg)
    load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()

    output_dir = os.path.join(cfg["inference"]["output_dir"], f"eval_{split}")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    grid_samples = []  # For visual grid

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc=f"Evaluating {split}")):
            lr = batch["lr"].to(device)
            hr = batch["hr"]

            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                sr = model(lr)

            for i in range(sr.shape[0]):
                sr_img = tensor_to_img(sr[i])
                hr_img = tensor_to_img(hr[i])
                lr_img = tensor_to_img(lr[i])

                psnr_val = compute_psnr(sr_img, hr_img)
                ssim_val = compute_ssim(sr_img, hr_img)

                # Bicubic baseline
                lr_bicubic = cv2.resize(lr_img, (hr_img.shape[1], hr_img.shape[0]),
                                        interpolation=cv2.INTER_CUBIC)
                psnr_bic = compute_psnr(lr_bicubic, hr_img)
                ssim_bic = compute_ssim(lr_bicubic, hr_img)

                fname = batch.get("path", [f"img_{idx:04d}"])[i] if "path" in batch else f"img_{idx:04d}"
                fname = os.path.basename(str(fname))

                results.append({
                    "filename": fname,
                    "psnr_sr": round(psnr_val, 4),
                    "ssim_sr": round(ssim_val, 6),
                    "psnr_bicubic": round(psnr_bic, 4),
                    "ssim_bicubic": round(ssim_bic, 6),
                    "psnr_gain": round(psnr_val - psnr_bic, 4),
                })

                # Collect first 16 for visual grid
                if len(grid_samples) < 16:
                    grid_samples.append({
                        "lr": lr_bicubic,
                        "sr": sr_img,
                        "hr": hr_img,
                        "psnr": psnr_val,
                    })

    # Save metrics CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)

    # Summary
    summary = {
        "split": split,
        "num_images": len(results),
        "psnr_sr_mean": float(df["psnr_sr"].mean()),
        "psnr_sr_std": float(df["psnr_sr"].std()),
        "ssim_sr_mean": float(df["ssim_sr"].mean()),
        "ssim_sr_std": float(df["ssim_sr"].std()),
        "psnr_bicubic_mean": float(df["psnr_bicubic"].mean()),
        "psnr_gain_mean": float(df["psnr_gain"].mean()),
    }

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write(f"SR Evaluation Summary ({split})\n")
        f.write("=" * 50 + "\n")
        for k, v in summary.items():
            f.write(f"  {k}: {v}\n")

    logger.info(f"PSNR (SR): {summary['psnr_sr_mean']:.2f} ± {summary['psnr_sr_std']:.2f} dB")
    logger.info(f"SSIM (SR): {summary['ssim_sr_mean']:.4f} ± {summary['ssim_sr_std']:.4f}")
    logger.info(f"PSNR gain over bicubic: +{summary['psnr_gain_mean']:.2f} dB")

    # Generate visual grid (4×4, each row: LR_bicubic | SR | HR | difference)
    if grid_samples:
        _save_visual_grid(grid_samples, os.path.join(output_dir, "eval_grid.png"))
        logger.info(f"Visual grid saved to {output_dir}/eval_grid.png")

    logger.info(f"Results saved to {output_dir}")
    return summary


def _save_visual_grid(samples, path, max_rows=4):
    """Save comparison grid: LR (bicubic) | SR | HR | |SR-HR| for each row."""
    n = min(len(samples), max_rows)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["LR (Bicubic ↑)", "SR (Ours)", "HR (Ground Truth)", "|SR − HR| (×5)"]

    for i in range(n):
        s = samples[i]
        diff = np.abs(s["sr"].astype(float) - s["hr"].astype(float))
        diff = np.clip(diff * 5, 0, 255).astype(np.uint8)

        for j, img in enumerate([s["lr"], s["sr"], s["hr"], diff]):
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=12)
        axes[i, 0].set_ylabel(f"PSNR: {s['psnr']:.1f}", fontsize=11, rotation=0, labelpad=60)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate SR model")
    parser.add_argument("--config", type=str, default="configs/sr_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint, args.split)


if __name__ == "__main__":
    main()
