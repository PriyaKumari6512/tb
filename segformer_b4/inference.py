"""
Inference module for SegFormer B4.
Supports single image and folder-level inference with bacilli detection.
"""

import argparse
import logging
import os
import time

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast
from tqdm import tqdm

from segformer_b4.model import build_segformer
from segformer_b4.utils import (
    get_device, load_checkpoint, load_config,
    read_image, save_image, detect_bacilli, detection_summary,
    create_overlay_with_count,
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
) -> dict:
    """Run segmentation on a single image."""
    image = read_image(image_path)
    orig_h, orig_w = image.shape[:2]

    transform = get_inference_transform(image_size)
    augmented = transform(image=image)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        with autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(input_tensor)
            logits = torch.nn.functional.interpolate(
                logits, size=(image_size, image_size),
                mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1)
        pred_mask = (probs[0, 1] > threshold).cpu().numpy().astype(np.uint8)

    # Resize mask to original resolution
    if pred_mask.shape != (orig_h, orig_w):
        pred_mask_full = cv2.resize(pred_mask, (orig_w, orig_h),
                                    interpolation=cv2.INTER_NEAREST)
    else:
        pred_mask_full = pred_mask

    detections = detect_bacilli(pred_mask_full, min_area)
    summary = detection_summary(detections)
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
    """Run inference on all images in a folder."""
    logging.basicConfig(level=logging.INFO)
    device = get_device()

    model_cfg = cfg["model"]
    model = build_segformer(
        variant=model_cfg.get("variant", "b4"),
        num_classes=model_cfg.get("num_classes", 2),
    )
    load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()

    image_size = cfg["data"].get("image_size", 512)
    threshold = cfg.get("postprocess", {}).get("conf_threshold", 0.5)
    min_area = cfg.get("postprocess", {}).get("min_area", 10)

    output_dir = output_dir or cfg.get("inference", {}).get("output_dir", "outputs/segformer")
    mask_dir = os.path.join(output_dir, "masks")
    overlay_dir = os.path.join(output_dir, "overlays")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif'))
    ])

    if not image_files:
        logger.warning(f"No images found in {folder_path}")
        return []

    logger.info(f"Processing {len(image_files)} images from {folder_path}")

    all_results = []
    t0 = time.time()

    for fname in tqdm(image_files, desc="SegFormer Inference"):
        src = os.path.join(folder_path, fname)
        result = infer_single(model, src, device, image_size, threshold, min_area)

        stem = os.path.splitext(fname)[0]
        save_image(result["mask"] * 255, os.path.join(mask_dir, f"{stem}_mask.png"))
        save_image(result["overlay"], os.path.join(overlay_dir, f"{stem}_overlay.png"))

        all_results.append({
            "filename": fname,
            "bacilli_count": result["summary"]["count"],
            "avg_area": result["summary"]["avg_area"],
            "total_area": result["summary"]["total_area"],
            "detections": result["detections"],
        })

    elapsed = time.time() - t0
    logger.info(f"Done — {len(image_files)} images in {elapsed:.1f}s")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="SegFormer B4 Inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str)
    group.add_argument("--folder", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.image:
        logging.basicConfig(level=logging.INFO)
        device = get_device()
        model_cfg = cfg["model"]
        model = build_segformer(
            variant=model_cfg.get("variant", "b4"),
            num_classes=model_cfg.get("num_classes", 2),
        )
        load_checkpoint(args.checkpoint, model, device=device)
        model = model.to(device)
        result = infer_single(model, args.image, device)
        print(f"Bacilli count: {result['summary']['count']}")
    else:
        infer_folder(cfg, args.checkpoint, args.folder, args.output_dir)


if __name__ == "__main__":
    main()
