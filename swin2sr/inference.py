"""
Swin2SR Inference — single image, folder batch, or tiled mode for large images.
"""

import argparse
import logging
import os
import time

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from swin2sr.model import build_model
from swin2sr.utils import (
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
    """Run Swin2SR on a single image."""
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
    """Run Swin2SR inference on all images in a folder."""
    logging.basicConfig(level=logging.INFO)
    device = get_device()

    model = build_model(cfg)
    load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()

    if output_dir is None:
        output_dir = cfg["inference"].get("output_dir", "./outputs/swin2sr")
    os.makedirs(output_dir, exist_ok=True)

    extensions = (".bmp", ".png", ".jpg", ".jpeg", ".tif")
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(extensions)
    ])

    tile_size = cfg["inference"].get("tile_size", 256)
    tile_overlap = cfg["inference"].get("tile_overlap", 32)
    scale = cfg["data"]["scale"]
    output_fmt = cfg["inference"].get("output_format", "png")

    logger.info(f"Processing {len(image_files)} images from {folder_path}")
    t0 = time.time()

    for fname in tqdm(image_files, desc="Swin2SR Inference"):
        img_path = os.path.join(folder_path, fname)
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, f"{stem}_sr.{output_fmt}")

        infer_single(
            model, img_path, out_path, device,
            tiled=tiled, tile_size=tile_size, tile_overlap=tile_overlap, scale=scale,
        )

    elapsed = time.time() - t0
    logger.info(f"Done — {len(image_files)} images in {elapsed:.1f}s "
                f"({elapsed / max(len(image_files), 1):.2f}s/img)")


def main():
    parser = argparse.ArgumentParser(description="Swin2SR inference")
    parser.add_argument("--config", type=str, default="swin2sr/configs/swin2sr_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Image or folder path")
    parser.add_argument("--output", type=str, default=None, help="Output path/directory")
    parser.add_argument("--tiled", action="store_true", help="Use tiled inference for large images")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if os.path.isfile(args.input):
        device = get_device()
        model = build_model(cfg)
        load_checkpoint(args.checkpoint, model, device=device)
        model = model.to(device)
        output_path = args.output or args.input.replace(".", "_sr.")
        infer_single(model, args.input, output_path, device, tiled=args.tiled)
        logger.info(f"Saved: {output_path}")
    elif os.path.isdir(args.input):
        infer_folder(cfg, args.checkpoint, args.input,
                     tiled=args.tiled, output_dir=args.output)
    else:
        raise FileNotFoundError(f"Input not found: {args.input}")


if __name__ == "__main__":
    main()
