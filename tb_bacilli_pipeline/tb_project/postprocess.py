"""
Post-processing: connected-component analysis for bacilli detection + counting.
- Label connected components from binary mask using OpenCV (no skimage required)
- Filter by minimum area
- Extract bounding boxes, centroids, area
"""

import logging
from typing import Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_bacilli(
    mask: np.ndarray,
    min_area: int = 10,
    connectivity: int = 8,
) -> List[Dict]:
    """
    Detect individual bacilli from binary segmentation mask.

    Args:
        mask: Binary mask (H, W), uint8 or bool. 1 = bacilli.
        min_area: Minimum connected-component area to keep (filters noise).
        connectivity: 4 or 8 (OpenCV connectedComponentsWithStats connectivity).

    Returns:
        List of detection dicts: {id, bbox, area, centroid}
        bbox format: (y_min, x_min, y_max, x_max)
    """
    binary = (mask > 0).astype(np.uint8)
    # OpenCV connectedComponentsWithStats: connectivity must be 4 or 8
    cv2_conn = 8 if connectivity >= 8 else 4
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=cv2_conn
    )

    detections = []
    det_id = 1
    # Label 0 is background
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[label][0]), float(centroids[label][1])
        detections.append({
            "id": det_id,
            "bbox": [y, x, y + h, x + w],  # (y_min, x_min, y_max, x_max)
            "area": area,
            "centroid": [round(cy, 1), round(cx, 1)],
        })
        det_id += 1

    return detections


def count_bacilli(mask: np.ndarray, min_area: int = 10, connectivity: int = 8) -> int:
    """Quick bacilli count from mask."""
    return len(detect_bacilli(mask, min_area, connectivity))


def detection_summary(detections: List[Dict]) -> Dict:
    """Generate summary statistics from detections."""
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
