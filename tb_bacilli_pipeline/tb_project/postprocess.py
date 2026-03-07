"""
Post-processing: connected-component analysis for bacilli detection + counting.
- Label connected components from binary mask
- Filter by minimum area
- Extract bounding boxes, centroids, area
"""

import logging
from typing import Dict, List

import numpy as np
from skimage import measure

logger = logging.getLogger(__name__)


def detect_bacilli(
    mask: np.ndarray,
    min_area: int = 10,
    connectivity: int = 2,
) -> List[Dict]:
    """
    Detect individual bacilli from binary segmentation mask.

    Args:
        mask: Binary mask (H, W), uint8 or bool. 1 = bacilli.
        min_area: Minimum connected-component area to keep (filters noise).
        connectivity: 1 (4-connected) or 2 (8-connected).

    Returns:
        List of detection dicts: {id, bbox, area, centroid}
        bbox format: (y_min, x_min, y_max, x_max)
    """
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
            "bbox": list(region.bbox),  # (y_min, x_min, y_max, x_max)
            "area": int(region.area),
            "centroid": [round(c, 1) for c in region.centroid],
        })
        det_id += 1

    return detections


def count_bacilli(mask: np.ndarray, min_area: int = 10, connectivity: int = 2) -> int:
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
