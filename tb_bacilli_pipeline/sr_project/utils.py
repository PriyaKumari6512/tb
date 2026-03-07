"""
Shared utility functions for the SR pipeline.
- Config loading with env-var interpolation
- Image I/O (BMP/PNG ↔ tensor)
- PSNR / SSIM metrics (Y-channel)
- Tiled inference helpers
- Checkpoint save / load
- Reproducibility seed
"""

import os
import re
import json
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

logger = logging.getLogger(__name__)

# =============================================================================
# Config helpers
# =============================================================================

def _interpolate_env(value: str) -> str:
    """Replace ${VAR:default} patterns with environment variable or default."""
    pattern = r"\$\{(\w+):([^}]*)\}"
    def replacer(match):
        var_name, default = match.group(1), match.group(2)
        return os.environ.get(var_name, default)
    return re.sub(pattern, replacer, value) if isinstance(value, str) else value


def _walk_and_interpolate(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _walk_and_interpolate(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_walk_and_interpolate(v) for v in obj]
    elif isinstance(obj, str):
        return _interpolate_env(obj)
    return obj


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config with env-var interpolation."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return _walk_and_interpolate(cfg)


def generate_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Image I/O
# =============================================================================

def read_image(path: str) -> np.ndarray:
    """Read image as RGB uint8 HWC numpy array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(img: np.ndarray, path: str):
    """Save RGB uint8 HWC numpy array."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 [0,255] → CHW float32 [0,1]."""
    return torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)


def tensor_to_img(tensor: torch.Tensor) -> np.ndarray:
    """CHW float32 [0,1] → HWC uint8 [0,255]."""
    arr = tensor.detach().cpu().clamp(0, 1).numpy()
    arr = (arr.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return arr


# =============================================================================
# Metrics (computed on Y channel for proper SR evaluation)
# =============================================================================

def rgb_to_y(img: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 to Y channel (ITU-R BT.601)."""
    return np.dot(img[..., :3].astype(np.float64),
                  [65.481, 128.553, 24.966]) / 255.0 + 16.0


def compute_psnr(sr: np.ndarray, hr: np.ndarray, crop_border: int = 4) -> float:
    """PSNR on Y channel with border cropping."""
    y_sr = rgb_to_y(sr)
    y_hr = rgb_to_y(hr)
    if crop_border > 0:
        y_sr = y_sr[crop_border:-crop_border, crop_border:-crop_border]
        y_hr = y_hr[crop_border:-crop_border, crop_border:-crop_border]
    return float(peak_signal_noise_ratio(y_hr, y_sr, data_range=235.0))


def compute_ssim(sr: np.ndarray, hr: np.ndarray, crop_border: int = 4) -> float:
    """SSIM on Y channel with border cropping."""
    y_sr = rgb_to_y(sr)
    y_hr = rgb_to_y(hr)
    if crop_border > 0:
        y_sr = y_sr[crop_border:-crop_border, crop_border:-crop_border]
        y_hr = y_hr[crop_border:-crop_border, crop_border:-crop_border]
    return float(structural_similarity(y_hr, y_sr, data_range=235.0))


# =============================================================================
# Tiled inference helpers
# =============================================================================

def compute_tiles(h: int, w: int, tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """Return list of (y_start, x_start, y_end, x_end) tile coordinates."""
    stride = tile_size - overlap
    tiles = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            tiles.append((y_start, x_start, y_end, x_end))
    # Deduplicate
    return list(dict.fromkeys(tiles))


def create_weight_map(tile_size: int, overlap: int) -> np.ndarray:
    """Create a 2D cosine blending weight map for a tile."""
    w = np.ones((tile_size, tile_size), dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap, dtype=np.float32)
        # Left / right
        w[:, :overlap] *= ramp[np.newaxis, :]
        w[:, -overlap:] *= ramp[np.newaxis, ::-1]
        # Top / bottom
        w[:overlap, :] *= ramp[:, np.newaxis]
        w[-overlap:, :] *= ramp[::-1, np.newaxis]
    return w


def tiled_inference(
    model: torch.nn.Module,
    lr_img: np.ndarray,
    scale: int,
    tile_size: int,
    overlap: int,
    device: torch.device,
) -> np.ndarray:
    """Run SR inference with tiled processing and blending."""
    h, w, c = lr_img.shape
    out_h, out_w = h * scale, w * scale

    # Accumulation buffers
    output = np.zeros((out_h, out_w, c), dtype=np.float64)
    weight_sum = np.zeros((out_h, out_w), dtype=np.float64)

    tiles = compute_tiles(h, w, tile_size, overlap)
    wmap = create_weight_map(tile_size, overlap)

    model.eval()
    with torch.no_grad():
        for y0, x0, y1, x1 in tiles:
            tile = lr_img[y0:y1, x0:x1]
            th, tw = tile.shape[:2]

            tile_t = img_to_tensor(tile).unsqueeze(0).to(device)
            sr_tile_t = model(tile_t)
            sr_tile = tensor_to_img(sr_tile_t.squeeze(0))

            # Crop weight map if tile is smaller than tile_size
            wm = wmap[:th * scale, :tw * scale] if (th < tile_size or tw < tile_size) else np.repeat(np.repeat(wmap, scale, axis=0), scale, axis=1)
            # Actually we need to expand the weight map by scale
            wm_full = np.repeat(np.repeat(wmap[:th, :tw], scale, axis=0), scale, axis=1)

            oy0, ox0 = y0 * scale, x0 * scale
            oy1, ox1 = y0 * scale + sr_tile.shape[0], x0 * scale + sr_tile.shape[1]

            output[oy0:oy1, ox0:ox1] += sr_tile.astype(np.float64) * wm_full[..., np.newaxis]
            weight_sum[oy0:oy1, ox0:ox1] += wm_full

    # Normalize
    weight_sum = np.maximum(weight_sum, 1e-8)
    output = output / weight_sum[..., np.newaxis]
    return output.clip(0, 255).astype(np.uint8)


# =============================================================================
# Checkpoint helpers
# =============================================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    scaler: Optional[Any] = None,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)
    logger.info(f"Checkpoint saved: {path} (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    # Handle DataParallel prefix
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    logger.info(f"Loaded checkpoint: {path} (epoch {ckpt['epoch']})")
    return ckpt


# =============================================================================
# Logging helpers
# =============================================================================

def setup_logging(log_dir: str, run_id: str) -> str:
    """Setup file + console logging. Returns run log directory."""
    if run_id == "auto":
        run_id = generate_run_id()
    run_dir = os.path.join(log_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return run_dir


def log_metrics(metrics: Dict[str, Any], path: str):
    """Append metrics dict as JSON line."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(metrics) + "\n")


# =============================================================================
# Device helpers
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_env_info():
    """Print environment diagnostics."""
    logger.info("=" * 60)
    logger.info("Environment Info")
    logger.info(f"  PyTorch:    {torch.__version__}")
    logger.info(f"  CUDA avail: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA ver:   {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name} ({props.total_mem / 1e9:.1f} GB)")
    logger.info(f"  Device:     {get_device()}")
    logger.info("=" * 60)
