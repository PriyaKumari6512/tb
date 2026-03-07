"""
Utility functions for SegFormer B4 segmentation module.
"""

import os
import re
import json
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import yaml
from skimage import measure

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================

def _interpolate_env(value):
    pattern = r"\$\{(\w+):([^}]*)\}"
    def replacer(match):
        return os.environ.get(match.group(1), match.group(2))
    return re.sub(pattern, replacer, value) if isinstance(value, str) else value


def _walk_and_interpolate(obj):
    if isinstance(obj, dict):
        return {k: _walk_and_interpolate(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_walk_and_interpolate(v) for v in obj]
    elif isinstance(obj, str):
        return _interpolate_env(obj)
    return obj


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return _walk_and_interpolate(cfg)


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
# Device
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Image I/O
# =============================================================================

def read_image(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(img: np.ndarray, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if img.ndim == 3 and img.shape[2] == 3:
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(path), img)


# =============================================================================
# Checkpoint
# =============================================================================

def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None,
                    device: torch.device = None) -> dict:
    device = device or get_device()
    state = torch.load(path, map_location=device, weights_only=False)
    model_state = state.get("model_state_dict", state.get("state_dict", state))

    # Handle DataParallel prefix
    cleaned = {}
    for k, v in model_state.items():
        cleaned[k.replace("module.", "")] = v
    model.load_state_dict(cleaned, strict=False)

    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    logger.info(f"Loaded checkpoint: {path}")
    return state


# =============================================================================
# Metrics
# =============================================================================

def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    pred_f = pred.float().flatten()
    target_f = target.float().flatten()
    intersection = (pred_f * target_f).sum()
    return float((2.0 * intersection + smooth) / (pred_f.sum() + target_f.sum() + smooth))


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    pred_f = pred.float().flatten()
    target_f = target.float().flatten()
    intersection = (pred_f * target_f).sum()
    union = pred_f.sum() + target_f.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def precision_recall_f1(pred: torch.Tensor, target: torch.Tensor) -> dict:
    pred_f = pred.float().flatten()
    target_f = target.float().flatten()
    tp = (pred_f * target_f).sum()
    fp = (pred_f * (1 - target_f)).sum()
    fn = ((1 - pred_f) * target_f).sum()
    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}


# =============================================================================
# Post-processing
# =============================================================================

def detect_bacilli(mask: np.ndarray, min_area: int = 10,
                   connectivity: int = 2) -> List[Dict]:
    binary = (mask > 0).astype(np.uint8)
    labeled = measure.label(binary, connectivity=connectivity)
    regions = measure.regionprops(labeled)

    detections = []
    det_id = 1
    for region in regions:
        if region.area < min_area:
            continue
        detections.append({
            "id": det_id,
            "bbox": list(region.bbox),
            "area": int(region.area),
            "centroid": [round(c, 1) for c in region.centroid],
        })
        det_id += 1
    return detections


def detection_summary(detections: List[Dict]) -> Dict:
    if not detections:
        return {"count": 0, "avg_area": 0, "min_area": 0, "max_area": 0, "total_area": 0}
    areas = [d["area"] for d in detections]
    return {
        "count": len(detections),
        "avg_area": round(sum(areas) / len(areas), 1),
        "min_area": min(areas),
        "max_area": max(areas),
        "total_area": sum(areas),
    }


# =============================================================================
# Visualization
# =============================================================================

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask > 0] = [255, 0, 0]  # Red for bacilli
    return out


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    colored = colorize_mask(mask)
    overlay = image.copy()
    mask_region = mask > 0
    overlay[mask_region] = (
        (1 - alpha) * image[mask_region] + alpha * colored[mask_region]
    ).astype(np.uint8)
    return overlay


def create_overlay_with_count(image, mask, detections):
    overlay = overlay_mask(image, mask)
    for det in detections:
        y1, x1, y2, x2 = det["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(overlay, f"Count: {len(detections)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return overlay


# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_dir: str = "logs", log_file: str = "segformer.log"):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, log_file)),
        ],
    )


def log_metrics(metrics: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(metrics) + "\n")
