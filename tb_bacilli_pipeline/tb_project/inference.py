"""
TB Inference — predict masks + count bacilli + draw bbox overlays.
Supports single image, folder, and optional SR-enhanced input.
"""

import argparse
import logging
import os
import time

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast
from tqdm import tqdm

from tb_project.model import build_model
from tb_project.postprocess import detect_bacilli, detection_summary
from tb_project.utils import (
    create_overlay_with_count,
    get_device,
    load_checkpoint,
    load_config,
    read_image,
    save_image,
)

logger = logging.getLogger(__name__)


def get_inference_transform(image_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def infer_single(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    image_size: int = 512,
    threshold: float = 0.5,
    min_area: int = 10,
    connectivity: int = 2,
) -> dict:
    """Run inference on a single image. Returns detection results."""
    image = read_image(image_path)
    orig_h, orig_w = image.shape[:2]

    # Transform
    transform = get_inference_transform(image_size)
    augmented = transform(image=image)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        with autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_mask = (probs[0, 1] > threshold).cpu().numpy().astype(np.uint8)

    # Resize mask back to original resolution
    if pred_mask.shape != (orig_h, orig_w):
        pred_mask_full = cv2.resize(pred_mask, (orig_w, orig_h),
                                      interpolation=cv2.INTER_NEAREST)
    else:
        pred_mask_full = pred_mask

    # Post-process
    detections = detect_bacilli(pred_mask_full, min_area, connectivity)
    summary = detection_summary(detections)

    # Create overlay on original image
    overlay = create_overlay_with_count(image, pred_mask_full, detections)

    return {
        "image": image,
        "mask": pred_mask_full,
        "overlay": overlay,
        "detections": detections,
        "summary": summary,
    }


def infer_folder(cfg: dict, checkpoint_path: str, folder_path: str,
                  output_dir: str = None):
    """Run TB inference on all images in a folder."""
    logging.basicConfig(level=logging.INFO)
    device = get_device()

    model = build_model(cfg)
    load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)

    if cfg["training"].get("multi_gpu", False) and torch.cuda.device_count() > 1:
        import torch.nn as nn
        model = nn.DataParallel(model)

    model.eval()

    pp_cfg = cfg.get("postprocess", {})
    inf_cfg = cfg.get("inference", {})
    image_size = cfg["data"].get("image_size", 512)
    threshold = pp_cfg.get("conf_threshold", 0.5)
    min_area = pp_cfg.get("min_area", 10)
    connectivity = pp_cfg.get("connectivity", 2)

    output_dir = output_dir or inf_cfg.get("output_dir", "outputs/tb")
    mask_dir = os.path.join(output_dir, "masks")
    overlay_dir = os.path.join(output_dir, "overlays")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # Collect images
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif'))
    ])

    if not image_files:
        logger.warning(f"No images found in {folder_path}")
        return

    logger.info(f"Processing {len(image_files)} images from {folder_path}")

    all_results = []
    t0 = time.time()

    infer_model = model.module if hasattr(model, "module") else model

    for fname in tqdm(image_files, desc="TB Inference"):
        src = os.path.join(folder_path, fname)
        result = infer_single(infer_model, src, device, image_size,
                               threshold, min_area, connectivity)

        stem = os.path.splitext(fname)[0]

        # Save outputs
        if inf_cfg.get("save_mask", True):
            save_image(result["mask"] * 255, os.path.join(mask_dir, f"{stem}_mask.png"))
        if inf_cfg.get("save_overlay", True):
            save_image(result["overlay"], os.path.join(overlay_dir, f"{stem}_overlay.png"))

        all_results.append({
            "filename": fname,
            "bacilli_count": result["summary"]["count"],
            "avg_area": result["summary"]["avg_area"],
            "total_area": result["summary"]["total_area"],
        })

    # Save CSV summary
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, "counts.csv")
    df.to_csv(csv_path, index=False)

    elapsed = time.time() - t0
    total_count = df["bacilli_count"].sum()
    avg_count = df["bacilli_count"].mean()

    logger.info("=" * 50)
    logger.info(f"Total bacilli detected: {total_count}")
    logger.info(f"Average per image: {avg_count:.2f}")
    logger.info(f"Processed {len(image_files)} images in {elapsed:.1f}s")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="TB segmentation inference")
    parser.add_argument("--config", type=str, default="configs/tb_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Single image path")
    group.add_argument("--folder", type=str, help="Folder of images")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.threshold:
        cfg["postprocess"]["conf_threshold"] = args.threshold

    if args.image:
        logging.basicConfig(level=logging.INFO)
        device = get_device()
        model = build_model(cfg)
        load_checkpoint(args.checkpoint, model, device=device)
        model = model.to(device)
        model.eval()

        result = infer_single(
            model, args.image, device,
            cfg["data"].get("image_size", 512),
            cfg["postprocess"].get("conf_threshold", 0.5),
            cfg["postprocess"].get("min_area", 10),
            cfg["postprocess"].get("connectivity", 2),
        )
        output_dir = args.output_dir or cfg["inference"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.image))[0]
        save_image(result["overlay"], os.path.join(output_dir, f"{stem}_overlay.png"))
        save_image(result["mask"] * 255, os.path.join(output_dir, f"{stem}_mask.png"))
        logger.info(f"Bacilli count: {result['summary']['count']}")
    else:
        infer_folder(cfg, args.checkpoint, args.folder, args.output_dir)


if __name__ == "__main__":
    main()
