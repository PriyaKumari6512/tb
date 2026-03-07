"""
SR Inference — single image, folder batch, or tiled mode for large images.
"""

import argparse
import logging
import os
import time

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from sr_project.model import build_model
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

logger = logging.getLogger(__name__)


def infer_single(
    model: torch.nn.Module,
    image_path: str,
    output_path: str,
    device: torch.device,
    tiled: bool = False,
    tile_size: int = 256,
    tile_overlap: int = 32,
    scale: int = 4,
) -> str:
    """Run SR on a single image."""
    img = read_image(image_path)

    if tiled:
        sr_img = tiled_inference(model, img, scale, tile_size, tile_overlap, device)
    else:
        model.eval()
        with torch.no_grad():
            lr_tensor = img_to_tensor(img).unsqueeze(0).to(device)
            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                sr_tensor = model(lr_tensor)
            sr_img = tensor_to_img(sr_tensor.squeeze(0))

    save_image(sr_img, output_path)
    return output_path


def infer_folder(cfg: dict, checkpoint_path: str, folder_path: str,
                 tiled: bool = False, output_dir: str = None):
    """Run SR inference on all images in a folder."""
    logging.basicConfig(level=logging.INFO)
    device = get_device()

    # Model
    model = build_model(cfg)
    load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)

    if cfg["training"].get("multi_gpu", False) and torch.cuda.device_count() > 1:
        import torch.nn as nn
        model = nn.DataParallel(model)

    model.eval()

    # Collect images
    ext = cfg["data"].get("extension", ".bmp")
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(ext.lower()) or f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))
    ])

    if not image_files:
        logger.warning(f"No images found in {folder_path}")
        return

    output_dir = output_dir or cfg["inference"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    tile_size = cfg["inference"].get("tile_size", 256)
    tile_overlap = cfg["inference"].get("tile_overlap", 32)
    scale = cfg["data"]["scale"]
    out_fmt = cfg["inference"].get("output_format", "png")

    logger.info(f"Processing {len(image_files)} images from {folder_path}")
    logger.info(f"  Tiled: {tiled} | tile_size: {tile_size} | overlap: {tile_overlap}")

    t0 = time.time()
    for fname in tqdm(image_files, desc="SR Inference"):
        src = os.path.join(folder_path, fname)
        stem = os.path.splitext(fname)[0]
        dst = os.path.join(output_dir, f"{stem}_sr.{out_fmt}")

        infer_model = model.module if hasattr(model, "module") else model
        infer_single(
            infer_model, src, dst, device,
            tiled=tiled, tile_size=tile_size,
            tile_overlap=tile_overlap, scale=scale,
        )

    elapsed = time.time() - t0
    logger.info(f"Done — {len(image_files)} images in {elapsed:.1f}s "
                f"({elapsed / len(image_files):.1f}s/img)")
    logger.info(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="SwinIR SR inference")
    parser.add_argument("--config", type=str, default="configs/sr_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Single image path")
    group.add_argument("--folder", type=str, help="Folder of images")
    parser.add_argument("--tiled", action="store_true", help="Use tiled inference")
    parser.add_argument("--tile_size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.tile_size:
        cfg["inference"]["tile_size"] = args.tile_size
    if args.overlap:
        cfg["inference"]["tile_overlap"] = args.overlap

    if args.image:
        logging.basicConfig(level=logging.INFO)
        device = get_device()
        model = build_model(cfg)
        load_checkpoint(args.checkpoint, model, device=device)
        model = model.to(device)
        model.eval()

        output_dir = args.output_dir or cfg["inference"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.image))[0]
        out_fmt = cfg["inference"].get("output_format", "png")
        out_path = os.path.join(output_dir, f"{stem}_sr.{out_fmt}")

        infer_single(model, args.image, out_path, device,
                      tiled=args.tiled,
                      tile_size=cfg["inference"]["tile_size"],
                      tile_overlap=cfg["inference"]["tile_overlap"],
                      scale=cfg["data"]["scale"])
        logger.info(f"Saved: {out_path}")
    else:
        infer_folder(cfg, args.checkpoint, args.folder,
                      tiled=args.tiled, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
