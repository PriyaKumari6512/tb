"""
Swin2SR Evaluation — compute PSNR/SSIM on test set + generate visual comparison grids.
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

from swin2sr.dataset import build_dataloaders
from swin2sr.model import build_model
from swin2sr.utils import (
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
    """Evaluate Swin2SR model on val or test set."""
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

    output_dir = cfg["inference"].get("output_dir", "./outputs/swin2sr_eval")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    logger.info(f"Evaluating on {split} set ({len(loader)} images)")

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
            lr = batch["lr"].to(device)
            hr = batch["hr"]

            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                sr = model(lr)

            sr_img = tensor_to_img(sr[0])
            hr_img = tensor_to_img(hr[0])
            lr_img = tensor_to_img(lr[0].cpu())

            psnr_val = compute_psnr(sr_img, hr_img)
            ssim_val = compute_ssim(sr_img, hr_img)

            results.append({
                "index": idx,
                "psnr": round(psnr_val, 4),
                "ssim": round(ssim_val, 6),
            })

            # Save visual comparison for first few images
            if idx < 10:
                # Upscale LR for side-by-side
                lr_up = cv2.resize(lr_img, (hr_img.shape[1], hr_img.shape[0]),
                                   interpolation=cv2.INTER_CUBIC)

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                axes[0].imshow(lr_up)
                axes[0].set_title("LR (bicubic upscale)")
                axes[0].axis("off")
                axes[1].imshow(sr_img)
                axes[1].set_title(f"Swin2SR (PSNR: {psnr_val:.2f})")
                axes[1].axis("off")
                axes[2].imshow(hr_img)
                axes[2].set_title("HR (ground truth)")
                axes[2].axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"comparison_{idx:04d}.png"),
                            dpi=100, bbox_inches="tight")
                plt.close()

    # Summary
    df = pd.DataFrame(results)
    avg_psnr = df["psnr"].mean()
    avg_ssim = df["ssim"].mean()

    logger.info(f"Average PSNR: {avg_psnr:.4f} dB")
    logger.info(f"Average SSIM: {avg_ssim:.6f}")

    df.to_csv(os.path.join(output_dir, "eval_results.csv"), index=False)

    summary = {
        "split": split,
        "num_images": len(results),
        "avg_psnr": round(avg_psnr, 4),
        "avg_ssim": round(avg_ssim, 6),
    }
    logger.info(f"Results saved to {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Swin2SR model")
    parser.add_argument("--config", type=str, default="swin2sr/configs/swin2sr_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint, args.split)


if __name__ == "__main__":
    main()
