"""
Combined SR → TB inference pipeline.
Runs TB detection on original images AND SR-enhanced images, produces comparison report.
"""

import argparse
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml

from sr_project.model import build_model as build_sr_model
from sr_project.utils import (
    get_device,
    img_to_tensor,
    load_checkpoint,
    load_config,
    read_image,
    save_image,
    tensor_to_img,
    tiled_inference,
)
from tb_project.inference import infer_single as tb_infer_single
from tb_project.model import build_model as build_tb_model
from tb_project.postprocess import detect_bacilli, detection_summary
from tb_project.utils import compute_all_metrics, create_overlay_with_count

logger = logging.getLogger(__name__)


def run_combined_pipeline(pipeline_cfg: dict):
    """Run TB inference on original vs SR-enhanced images."""
    logging.basicConfig(level=logging.INFO)
    device = get_device()

    # Load configs
    sr_cfg = load_config(pipeline_cfg["sr"]["config"])
    tb_cfg = load_config(pipeline_cfg["tb"]["config"])

    test_dir = pipeline_cfg["data"]["test_dir"]
    mask_dir = pipeline_cfg["data"].get("mask_dir")
    sr_enabled = pipeline_cfg["sr"].get("enabled", True)

    # Setup output
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

    # Load TB model
    logger.info("Loading TB model...")
    tb_model = build_tb_model(tb_cfg)
    load_checkpoint(pipeline_cfg["tb"]["checkpoint"], tb_model, device=device)
    tb_model = tb_model.to(device)
    tb_model.eval()

    # Load SR model (if enabled)
    sr_model = None
    if sr_enabled:
        logger.info("Loading SR model...")
        sr_model = build_sr_model(sr_cfg)
        load_checkpoint(pipeline_cfg["sr"]["checkpoint"], sr_model, device=device)
        sr_model = sr_model.to(device)
        sr_model.eval()

    # Collect images
    image_files = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif'))
    ])

    logger.info(f"Processing {len(image_files)} images")

    # Config values
    tb_image_size = tb_cfg["data"].get("image_size", 512)
    pp_cfg = tb_cfg.get("postprocess", {})
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

        # --- Option A: TB on original ---
        result_orig = tb_infer_single(
            tb_model, img_path, device,
            tb_image_size, threshold, min_area, connectivity,
        )
        count_orig = result_orig["summary"]["count"]
        save_image(result_orig["overlay"],
                    os.path.join(dirs["original"], f"{stem}_overlay.png"))

        # --- Option B: SR → TB ---
        count_sr = 0
        sr_dice = 0.0
        if sr_enabled and sr_model is not None:
            # Super-resolve
            image = read_image(img_path)
            sr_img = tiled_inference(sr_model, image, scale, tile_size, tile_overlap, device)

            # Save SR image temporarily and run TB inference
            sr_tmp_path = os.path.join(dirs["sr_enhanced"], f"{stem}_sr.png")
            save_image(sr_img, sr_tmp_path)

            result_sr = tb_infer_single(
                tb_model, sr_tmp_path, device,
                tb_image_size, threshold, min_area, connectivity,
            )
            count_sr = result_sr["summary"]["count"]
            save_image(result_sr["overlay"],
                        os.path.join(dirs["sr_enhanced"], f"{stem}_overlay.png"))

        # Load ground truth mask if available
        dice_orig = 0.0
        dice_sr = 0.0
        gt_count = 0
        if mask_dir:
            mask_path = os.path.join(mask_dir, fname)
            if os.path.exists(mask_path):
                import cv2
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                gt_mask = (gt_mask > 0).astype(np.uint8)
                gt_count = len(detect_bacilli(gt_mask, min_area, connectivity))

                # Resize pred masks to match gt
                if result_orig["mask"].shape != gt_mask.shape:
                    pred_orig_resized = cv2.resize(
                        result_orig["mask"], (gt_mask.shape[1], gt_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
                else:
                    pred_orig_resized = result_orig["mask"]
                metrics_orig = compute_all_metrics(pred_orig_resized, gt_mask)
                dice_orig = metrics_orig["dice"]

                if sr_enabled:
                    if result_sr["mask"].shape != gt_mask.shape:
                        pred_sr_resized = cv2.resize(
                            result_sr["mask"], (gt_mask.shape[1], gt_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
                    else:
                        pred_sr_resized = result_sr["mask"]
                    metrics_sr = compute_all_metrics(pred_sr_resized, gt_mask)
                    dice_sr = metrics_sr["dice"]

        # Side-by-side visualization
        if sr_enabled and pipeline_cfg["output"].get("save_side_by_side", True):
            _save_side_by_side(
                result_orig["overlay"],
                result_sr["overlay"] if sr_enabled else result_orig["overlay"],
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
        "Combined Pipeline Comparison Report",
        "=" * 60,
        f"Run ID: {run_id}",
        f"Images processed: {len(results)}",
        f"Time: {elapsed:.1f}s ({elapsed / max(len(results), 1):.1f}s/img)",
        "",
        "--- Original Images ---",
        f"  Mean bacilli count: {df['count_original'].mean():.2f}",
        f"  Total bacilli: {df['count_original'].sum()}",
    ]

    if sr_enabled:
        sr_counts = pd.to_numeric(df["count_sr"], errors="coerce")
        summary_lines.extend([
            "",
            "--- SR-Enhanced Images ---",
            f"  Mean bacilli count: {sr_counts.mean():.2f}",
            f"  Total bacilli: {sr_counts.sum():.0f}",
            "",
            "--- Comparison ---",
            f"  Mean count delta (SR - orig): {pd.to_numeric(df['delta']).mean():.2f}",
        ])

    if mask_dir:
        summary_lines.extend([
            f"  Mean Dice (original): {df['dice_original'].mean():.4f}",
            f"  Mean Dice (SR):       {pd.to_numeric(df['dice_sr'], errors='coerce').mean():.4f}",
        ])

    summary_lines.append("=" * 60)
    summary_text = "\n".join(summary_lines)

    with open(os.path.join(output_base, "summary.txt"), "w") as f:
        f.write(summary_text)

    print(summary_text)
    logger.info(f"Results saved to {output_base}")


def _save_side_by_side(img_a, img_b, path, count_a, count_b):
    """Stitch two overlay images side by side with labels."""
    import cv2
    h = max(img_a.shape[0], img_b.shape[0])
    w_a, w_b = img_a.shape[1], img_b.shape[1]

    canvas = np.zeros((h + 30, w_a + w_b + 10, 3), dtype=np.uint8)
    canvas[:img_a.shape[0], :w_a] = img_a
    canvas[:img_b.shape[0], w_a + 10:] = img_b

    # Labels
    cv2.putText(canvas, f"Original (n={count_a})", (10, h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(canvas, f"SR Enhanced (n={count_b})", (w_a + 20, h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    save_image(canvas, path)


def main():
    parser = argparse.ArgumentParser(description="Combined SR + TB pipeline")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml")
    args = parser.parse_args()

    pipeline_cfg = load_config(args.config)
    run_combined_pipeline(pipeline_cfg)


if __name__ == "__main__":
    main()
