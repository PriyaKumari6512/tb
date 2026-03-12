"""
Full Swin2SR → SegFormer pipeline for TB bacilli detection.

Pipeline stages:
  1. Super-Resolution: Swin2SR x4 enhances microscopy images
  2. Segmentation: SegFormer detects TB bacilli in enhanced images
  3. Post-processing: Connected-component analysis for bacilli counting
  4. Comparison: Original vs SR-enhanced detection results

Supports:
  - Tiled SR inference for large microscopy images
  - Side-by-side comparison with ground truth
  - Metrics: bacilli count, Dice, IoU, Precision, Recall, F1
  - CSV report generation
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

# Add parent directory to path so we can import swin2sr and tb_bacilli_pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swin2sr.model import build_model as build_sr_model
from swin2sr.utils import (
    get_device,
    img_to_tensor,
    load_checkpoint as load_sr_checkpoint,
    load_config,
    read_image,
    save_image,
    tensor_to_img,
    tiled_inference,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Segmentation helpers (self-contained to avoid hard dependency on tb_project)
# =============================================================================

def _build_segmentation_model(seg_cfg: dict):
    """Build SegFormer segmentation model from config."""
    try:
        from tb_bacilli_pipeline.tb_project.model import build_model
        return build_model(seg_cfg)
    except ImportError:
        # Fallback: build directly from HuggingFace
        from transformers import SegformerForSemanticSegmentation
        import torch.nn as nn
        import torch.nn.functional as F

        class SimpleSegFormer(nn.Module):
            def __init__(self, backbone="nvidia/mit-b4", num_classes=2):
                super().__init__()
                self.num_classes = num_classes
                try:
                    self.model = SegformerForSemanticSegmentation.from_pretrained(
                        backbone, num_labels=num_classes, ignore_mismatched_sizes=True,
                    )
                except (OSError, RuntimeError):
                    from transformers import SegformerConfig
                    config = SegformerConfig(
                        num_labels=num_classes,
                        hidden_sizes=[64, 128, 320, 512],
                        depths=[3, 8, 27, 3],
                        num_attention_heads=[1, 2, 5, 8],
                    )
                    self.model = SegformerForSemanticSegmentation(config)

            def forward(self, x):
                outputs = self.model(pixel_values=x)
                logits = outputs.logits
                return F.interpolate(logits, size=x.shape[2:], mode="bilinear",
                                     align_corners=False)

        mcfg = seg_cfg.get("model", {})
        return SimpleSegFormer(
            backbone=mcfg.get("backbone", "nvidia/mit-b4"),
            num_classes=mcfg.get("num_classes", 2),
        )


def _load_seg_checkpoint(path, model, device):
    """Load segmentation checkpoint."""
    try:
        from tb_bacilli_pipeline.tb_project.utils import load_checkpoint
        return load_checkpoint(path, model, device=device)
    except ImportError:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        return ckpt


def _segment_image(model, image, device, image_size=512, threshold=0.5):
    """Run segmentation on a single image (numpy RGB HWC uint8).

    Returns binary mask (H, W) uint8.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    orig_h, orig_w = image.shape[:2]
    augmented = transform(image=image)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_mask = (probs[0, 1] > threshold).cpu().numpy().astype(np.uint8)

    # Resize back to original
    if pred_mask.shape != (orig_h, orig_w):
        pred_mask = cv2.resize(pred_mask, (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)
    return pred_mask


def _detect_bacilli(mask, min_area=10, connectivity=2):
    """Connected-component analysis for bacilli detection."""
    from skimage.measure import label, regionprops
    labeled = label(mask, connectivity=connectivity)
    detections = []
    for region in regionprops(labeled):
        if region.area >= min_area:
            y0, x0, y1, x1 = region.bbox
            detections.append({
                "bbox": (x0, y0, x1, y1),
                "centroid": (int(region.centroid[1]), int(region.centroid[0])),
                "area": region.area,
            })
    return detections


def _compute_metrics(pred_mask, gt_mask):
    """Compute Dice and IoU between predicted and ground truth masks."""
    pred = pred_mask.astype(bool).flatten()
    gt = gt_mask.astype(bool).flatten()

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    dice = 2.0 * intersection / (pred.sum() + gt.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    precision = intersection / (pred.sum() + 1e-8)
    recall = intersection / (gt.sum() + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _create_overlay(image, mask, detections, alpha=0.4):
    """Create visualization overlay with mask and bounding boxes."""
    overlay = image.copy()
    # Red mask overlay
    red_mask = np.zeros_like(image)
    red_mask[mask > 0] = [255, 0, 0]
    overlay = cv2.addWeighted(overlay, 1.0, red_mask, alpha, 0)

    # Draw bounding boxes and count
    for det in detections:
        x0, y0, x1, y1 = det["bbox"]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # Add count text
    count = len(detections)
    cv2.putText(overlay, f"Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return overlay


def _save_side_by_side(img_orig, img_sr, save_path, count_orig, count_sr):
    """Save side-by-side comparison image."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(img_orig)
    axes[0].set_title(f"Original (count: {count_orig})")
    axes[0].axis("off")
    axes[1].imshow(img_sr)
    axes[1].set_title(f"Swin2SR Enhanced (count: {count_sr})")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline(pipeline_cfg: dict):
    """Run the full Swin2SR → Segmentation pipeline.

    Steps:
        1. Load Swin2SR SR model and SegFormer segmentation model
        2. For each test image:
           a. Run segmentation on original image → count bacilli
           b. Run Swin2SR → segmentation on SR output → count bacilli
           c. Compare results, generate overlays
        3. Generate comparison report (CSV + summary)
    """
    logging.basicConfig(level=logging.INFO)
    device = get_device()

    # Load SR config and segmentation config
    sr_cfg = load_config(pipeline_cfg["sr"]["config"])
    seg_cfg = load_config(pipeline_cfg["segmentation"]["config"])

    test_dir = pipeline_cfg["data"]["test_dir"]
    mask_dir = pipeline_cfg["data"].get("mask_dir")
    sr_enabled = pipeline_cfg["sr"].get("enabled", True)

    # Setup output directories
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = os.path.join(pipeline_cfg["output"]["dir"], run_id)
    dirs = {
        "original": os.path.join(output_base, "original"),
        "sr_enhanced": os.path.join(output_base, "sr_enhanced"),
        "side_by_side": os.path.join(output_base, "side_by_side"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Save config snapshot
    with open(os.path.join(output_base, "config_snapshot.yaml"), "w") as f:
        yaml.dump(pipeline_cfg, f)

    # Load segmentation model
    logger.info("Loading segmentation model...")
    seg_model = _build_segmentation_model(seg_cfg)
    _load_seg_checkpoint(pipeline_cfg["segmentation"]["checkpoint"], seg_model, device)
    seg_model = seg_model.to(device)
    seg_model.eval()

    # Load SR model (if enabled)
    sr_model = None
    if sr_enabled:
        logger.info("Loading Swin2SR model...")
        sr_model = build_sr_model(sr_cfg)
        load_sr_checkpoint(pipeline_cfg["sr"]["checkpoint"], sr_model, device=device)
        sr_model = sr_model.to(device)
        sr_model.eval()

    # Collect images
    extensions = (".bmp", ".png", ".jpg", ".jpeg", ".tif")
    image_files = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith(extensions)
    ])

    logger.info(f"Processing {len(image_files)} images")

    # Pipeline settings
    seg_image_size = seg_cfg["data"].get("image_size", 512)
    pp_cfg = seg_cfg.get("postprocess", {})
    threshold = pp_cfg.get("conf_threshold", 0.5)
    min_area = pp_cfg.get("min_area", 10)
    connectivity = pp_cfg.get("connectivity", 2)
    tile_size = sr_cfg["inference"].get("tile_size", 256)
    tile_overlap = sr_cfg["inference"].get("tile_overlap", 32)
    scale = sr_cfg["data"]["scale"]

    results = []
    t0 = time.time()

    for fname in image_files:
        img_path = os.path.join(test_dir, fname)
        stem = os.path.splitext(fname)[0]

        # Read original image
        image = read_image(img_path)

        # --- Stage A: Segmentation on original image ---
        mask_orig = _segment_image(seg_model, image, device, seg_image_size, threshold)
        detections_orig = _detect_bacilli(mask_orig, min_area, connectivity)
        count_orig = len(detections_orig)
        overlay_orig = _create_overlay(image, mask_orig, detections_orig)
        save_image(overlay_orig, os.path.join(dirs["original"], f"{stem}_overlay.png"))

        # --- Stage B: Swin2SR → Segmentation ---
        count_sr = 0
        overlay_sr = overlay_orig  # fallback
        dice_sr = 0.0
        if sr_enabled and sr_model is not None:
            # Super-resolve
            sr_img = tiled_inference(sr_model, image, scale, tile_size, tile_overlap, device)

            # Run segmentation on SR output
            mask_sr = _segment_image(seg_model, sr_img, device, seg_image_size, threshold)
            detections_sr = _detect_bacilli(mask_sr, min_area, connectivity)
            count_sr = len(detections_sr)
            overlay_sr = _create_overlay(sr_img, mask_sr, detections_sr)
            save_image(overlay_sr, os.path.join(dirs["sr_enhanced"], f"{stem}_overlay.png"))

            # Save SR image
            save_image(sr_img, os.path.join(dirs["sr_enhanced"], f"{stem}_sr.png"))

        # --- Ground truth comparison ---
        dice_orig = 0.0
        gt_count = 0
        if mask_dir:
            gt_path = os.path.join(mask_dir, fname)
            if os.path.exists(gt_path):
                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                gt_mask = (gt_mask > 0).astype(np.uint8)
                gt_detections = _detect_bacilli(gt_mask, min_area, connectivity)
                gt_count = len(gt_detections)

                # Metrics for original
                if mask_orig.shape != gt_mask.shape:
                    mask_orig_resized = cv2.resize(
                        mask_orig, (gt_mask.shape[1], gt_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
                else:
                    mask_orig_resized = mask_orig
                metrics_orig = _compute_metrics(mask_orig_resized, gt_mask)
                dice_orig = metrics_orig["dice"]

                # Metrics for SR-enhanced
                if sr_enabled:
                    if mask_sr.shape != gt_mask.shape:
                        mask_sr_resized = cv2.resize(
                            mask_sr, (gt_mask.shape[1], gt_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
                    else:
                        mask_sr_resized = mask_sr
                    metrics_sr = _compute_metrics(mask_sr_resized, gt_mask)
                    dice_sr = metrics_sr["dice"]

        # Side-by-side visualization
        if sr_enabled and pipeline_cfg["output"].get("save_side_by_side", True):
            _save_side_by_side(
                overlay_orig, overlay_sr,
                os.path.join(dirs["side_by_side"], f"{stem}_comparison.png"),
                count_orig, count_sr,
            )

        results.append({
            "filename": fname,
            "count_original": count_orig,
            "count_sr": count_sr if sr_enabled else "N/A",
            "count_gt": gt_count,
            "delta": count_sr - count_orig if sr_enabled else 0,
            "dice_original": round(dice_orig, 4),
            "dice_sr": round(dice_sr, 4) if sr_enabled else "N/A",
        })

        logger.info(f"{fname}: orig={count_orig}, sr={count_sr}, gt={gt_count}")

    elapsed = time.time() - t0

    # Save report
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_base, "comparison_report.csv"), index=False)

    # Summary
    summary_lines = [
        "=" * 60,
        "Swin2SR → Segmentation Pipeline Report",
        "=" * 60,
        f"Run ID: {run_id}",
        f"Images processed: {len(results)}",
        f"Time: {elapsed:.1f}s ({elapsed / max(len(results), 1):.1f}s/img)",
        "",
        "Original Detection:",
        f"  Avg count:  {df['count_original'].mean():.1f}",
    ]
    if mask_dir:
        summary_lines.append(f"  Avg Dice:   {df['dice_original'].mean():.4f}")
    if sr_enabled:
        sr_counts = [r["count_sr"] for r in results if r["count_sr"] != "N/A"]
        summary_lines += [
            "",
            "SR-Enhanced Detection:",
        ]
        if sr_counts:
            summary_lines.append(f"  Avg count:  {np.mean(sr_counts):.1f}")
        if mask_dir:
            sr_dices = [r["dice_sr"] for r in results if r["dice_sr"] != "N/A"]
            if sr_dices:
                summary_lines.append(f"  Avg Dice:   {np.mean(sr_dices):.4f}")

    summary_text = "\n".join(summary_lines)
    logger.info("\n" + summary_text)

    with open(os.path.join(output_base, "summary.txt"), "w") as f:
        f.write(summary_text)

    logger.info(f"Results saved to {output_base}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Swin2SR → Segmentation Pipeline for TB Bacilli Detection"
    )
    parser.add_argument(
        "--config", type=str,
        default="swin2sr_pipeline/configs/pipeline_config.yaml",
        help="Path to pipeline config YAML",
    )
    args = parser.parse_args()

    pipeline_cfg = load_config(args.config)
    run_pipeline(pipeline_cfg)


if __name__ == "__main__":
    main()
