"""
Shared utility functions for the TB segmentation pipeline.
- Dice / IoU / Precision / Recall / F1 metric functions
- Overlay drawing (mask + bboxes on image)
- Mask colorization
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Re-export shared utils from sr_project
from sr_project.utils import (
    get_device,
    generate_run_id,
    load_config,
    load_checkpoint,
    log_metrics,
    print_env_info,
    read_image,
    save_checkpoint,
    save_image,
    set_seed,
    setup_logging,
)


# =============================================================================
# Segmentation metrics (tensor-based for training, numpy for evaluation)
# =============================================================================

def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Compute Dice score. Inputs are binary (0/1) tensors."""
    pred_flat = pred.float().view(-1)
    target_flat = target.float().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Compute IoU (Jaccard)."""
    pred_flat = pred.float().view(-1)
    target_flat = target.float().view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def precision_recall_f1(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute precision, recall, F1 from binary masks."""
    pred_flat = pred.astype(bool).ravel()
    target_flat = target.astype(bool).ravel()

    tp = (pred_flat & target_flat).sum()
    fp = (pred_flat & ~target_flat).sum()
    fn = (~pred_flat & target_flat).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_all_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute all segmentation metrics."""
    pred_t = torch.from_numpy(pred.astype(np.float32))
    target_t = torch.from_numpy(target.astype(np.float32))

    metrics = {
        "dice": float(dice_score(pred_t, target_t)),
        "iou": float(iou_score(pred_t, target_t)),
    }
    metrics.update(precision_recall_f1(pred, target))
    return metrics


# =============================================================================
# Visualization helpers
# =============================================================================

def colorize_mask(mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """Convert binary mask to RGB colored version."""
    h, w = mask.shape[:2]
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    colored[mask > 0] = color
    return colored


def overlay_mask(image: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.4, color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """Overlay colored mask on image."""
    overlay = image.copy()
    colored = colorize_mask(mask, color)
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(
        image[mask_bool].reshape(-1, 3), 1 - alpha,
        colored[mask_bool].reshape(-1, 3), alpha, 0
    ).reshape(-1, 3)
    return overlay


def draw_bboxes(image: np.ndarray, bboxes: List[Dict],
                color: Tuple[int, int, int] = (0, 255, 0),
                thickness: int = 2) -> np.ndarray:
    """Draw bounding boxes with IDs on image."""
    result = image.copy()
    for det in bboxes:
        y0, x0, y1, x1 = det["bbox"]
        cv2.rectangle(result, (x0, y0), (x1, y1), color, thickness)
        cv2.putText(result, str(det["id"]), (x0, y0 - 4),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return result


def create_overlay_with_count(
    image: np.ndarray,
    mask: np.ndarray,
    detections: List[Dict],
    alpha: float = 0.4,
    mask_color: Tuple[int, int, int] = (255, 0, 0),
    bbox_color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Create full overlay: mask + bboxes + count text."""
    result = overlay_mask(image, mask, alpha, mask_color)
    result = draw_bboxes(result, detections, bbox_color)

    # Add count text
    count = len(detections)
    text = f"Bacilli: {count}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(result, (5, 5), (tw + 15, th + 15), (0, 0, 0), -1)
    cv2.putText(result, text, (10, th + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return result
