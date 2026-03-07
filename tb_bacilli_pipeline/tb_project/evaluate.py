"""
TB Evaluation — compute Dice/IoU/Precision/Recall/F1 + generate visual grids.
Also shows per-subfolder breakdown.
"""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from tb_project.dataset import build_dataloaders
from tb_project.model import build_model
from tb_project.postprocess import count_bacilli, detect_bacilli
from tb_project.utils import (
    compute_all_metrics,
    create_overlay_with_count,
    get_device,
    load_checkpoint,
    load_config,
    save_image,
    read_image,
)

logger = logging.getLogger(__name__)


def evaluate(cfg: dict, checkpoint_path: str, split: str = "test"):
    logging.basicConfig(level=logging.INFO)
    device = get_device()

    _, val_loader, test_loader = build_dataloaders(cfg)
    loader = test_loader if split == "test" else val_loader

    model = build_model(cfg)
    load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()

    pp_cfg = cfg.get("postprocess", {})
    min_area = pp_cfg.get("min_area", 10)
    connectivity = pp_cfg.get("connectivity", 2)
    threshold = pp_cfg.get("conf_threshold", 0.5)

    output_dir = os.path.join(cfg["inference"]["output_dir"], f"eval_{split}")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    grid_samples = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split}"):
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()
            filenames = batch["filename"]
            subfolders = batch["subfolder"]

            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images)

            probs = torch.softmax(logits, dim=1)
            preds = (probs[:, 1] > threshold).cpu().numpy().astype(np.uint8)

            for i in range(preds.shape[0]):
                pred_mask = preds[i]
                gt_mask = masks[i]

                # Metrics
                metrics = compute_all_metrics(pred_mask, gt_mask)

                # Bacilli counts
                pred_count = count_bacilli(pred_mask, min_area, connectivity)
                gt_count = count_bacilli(gt_mask, min_area, connectivity)

                results.append({
                    "filename": filenames[i],
                    "subfolder": subfolders[i],
                    "dice": round(metrics["dice"], 4),
                    "iou": round(metrics["iou"], 4),
                    "precision": round(metrics["precision"], 4),
                    "recall": round(metrics["recall"], 4),
                    "f1": round(metrics["f1"], 4),
                    "pred_count": pred_count,
                    "gt_count": gt_count,
                    "count_diff": pred_count - gt_count,
                })

                # Collect for grid
                if len(grid_samples) < 8:
                    # Denormalize image for visualization
                    img_tensor = images[i].cpu()
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_denorm = (img_tensor * std + mean).clamp(0, 1)
                    img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                    detections = detect_bacilli(pred_mask, min_area, connectivity)
                    overlay = create_overlay_with_count(img_np, pred_mask, detections)

                    grid_samples.append({
                        "image": img_np,
                        "gt_mask": gt_mask,
                        "pred_mask": pred_mask,
                        "overlay": overlay,
                        "dice": metrics["dice"],
                        "pred_count": pred_count,
                    })

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    # Summary
    summary = {
        "split": split,
        "num_images": len(results),
        "dice_mean": float(df["dice"].mean()),
        "dice_std": float(df["dice"].std()),
        "iou_mean": float(df["iou"].mean()),
        "precision_mean": float(df["precision"].mean()),
        "recall_mean": float(df["recall"].mean()),
        "f1_mean": float(df["f1"].mean()),
        "avg_pred_count": float(df["pred_count"].mean()),
        "avg_gt_count": float(df["gt_count"].mean()),
        "avg_count_diff": float(df["count_diff"].mean()),
    }

    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("=" * 50 + "\n")
        f.write(f"TB Segmentation Evaluation ({split})\n")
        f.write("=" * 50 + "\n")
        for k, v in summary.items():
            f.write(f"  {k}: {v}\n")

    logger.info(f"Dice: {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}")
    logger.info(f"IoU:  {summary['iou_mean']:.4f}")
    logger.info(f"F1:   {summary['f1_mean']:.4f}")
    logger.info(f"Avg bacilli count: pred={summary['avg_pred_count']:.1f}, "
                f"gt={summary['avg_gt_count']:.1f}")

    # Visual grid
    if grid_samples:
        _save_visual_grid(grid_samples, os.path.join(output_dir, "eval_grid.png"))

    logger.info(f"Results saved to {output_dir}")
    return summary


def _save_visual_grid(samples, path, max_rows=4):
    """Grid: Image | GT Mask | Pred Mask | Overlay."""
    n = min(len(samples), max_rows)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = ["Input", "GT Mask", "Pred Mask", "Overlay (Pred)"]
    for i in range(n):
        s = samples[i]
        axes[i, 0].imshow(s["image"])
        axes[i, 1].imshow(s["gt_mask"], cmap="gray")
        axes[i, 2].imshow(s["pred_mask"], cmap="gray")
        axes[i, 3].imshow(s["overlay"])
        for j in range(4):
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=12)
        axes[i, 0].set_ylabel(
            f"Dice: {s['dice']:.2f}\nCount: {s['pred_count']}",
            fontsize=10, rotation=0, labelpad=60,
        )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate TB segmentation")
    parser.add_argument("--config", type=str, default="configs/tb_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint, args.split)


if __name__ == "__main__":
    main()
