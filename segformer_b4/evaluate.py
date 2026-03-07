"""
Evaluation module for SegFormer B4.
Computes Dice, IoU, Precision, Recall, F1 on test set.
"""

import argparse
import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from segformer_b4.model import build_segformer
from segformer_b4.dataset import build_dataloaders
from segformer_b4.utils import (
    load_config, get_device, load_checkpoint, set_seed,
    dice_score, iou_score, precision_recall_f1, setup_logging,
)

logger = logging.getLogger(__name__)


def evaluate(cfg: dict, checkpoint_path: str):
    """Full evaluation on test set."""
    setup_logging()
    set_seed(cfg.get("seed", 42))
    device = get_device()

    # Model
    model_cfg = cfg["model"]
    model = build_segformer(
        variant=model_cfg.get("variant", "b4"),
        num_classes=model_cfg.get("num_classes", 2),
    )
    load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()

    # Data
    loaders = build_dataloaders(cfg)
    test_loader = loaders["test"]

    all_dice, all_iou, all_prec, all_rec, all_f1 = [], [], [], [], []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            logits = torch.nn.functional.interpolate(
                logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
            preds = logits.argmax(dim=1)

            for i in range(preds.size(0)):
                all_dice.append(dice_score(preds[i], masks[i]))
                all_iou.append(iou_score(preds[i], masks[i]))
                prf = precision_recall_f1(preds[i], masks[i])
                all_prec.append(prf["precision"])
                all_rec.append(prf["recall"])
                all_f1.append(prf["f1"])

    results = {
        "dice": np.mean(all_dice),
        "iou": np.mean(all_iou),
        "precision": np.mean(all_prec),
        "recall": np.mean(all_rec),
        "f1": np.mean(all_f1),
        "n_images": len(all_dice),
    }

    logger.info("=" * 50)
    logger.info("  SegFormer B4 — Test Evaluation Results")
    logger.info("=" * 50)
    for k, v in results.items():
        if isinstance(v, float):
            logger.info(f"  {k:>12s}: {v:.4f}")
        else:
            logger.info(f"  {k:>12s}: {v}")
    logger.info("=" * 50)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
