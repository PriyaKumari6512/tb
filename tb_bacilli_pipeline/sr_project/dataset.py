"""
SR Dataset for DDS3.
- Walks TRAINING SET subfolders to collect HR images
- Generates LR on-the-fly via bicubic downsampling
- Paired augmentations (crop, flip, rotate) for training
- Full-image loading for validation/test
"""

import os
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from sr_project.utils import read_image, img_to_tensor

logger = logging.getLogger(__name__)


def collect_image_paths(
    data_root: str,
    split: str = "TRAINING SET",
    subfolders: Optional[List[str]] = None,
    image_dir: str = "IMAGE",
) -> List[str]:
    """Collect all image paths from DDS3-style directory structure.

    Args:
        data_root: Root dataset directory.
        split: Split directory (e.g. 'TRAINING SET').
        subfolders: List of subfolder names; if None, all subfolders are used.
        image_dir: Sub-directory containing images (default 'IMAGE').

    Returns:
        Sorted list of absolute paths to image files.
    """
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    if subfolders is None:
        subfolders = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])

    paths: List[str] = []
    for sf in subfolders:
        folder = os.path.join(split_dir, sf, image_dir)
        if not os.path.isdir(folder):
            logger.warning(f"Image dir not found: {folder}")
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif')):
                paths.append(os.path.join(folder, fname))

    logger.info(f"Collected {len(paths)} images from split '{split}'")
    return paths


class SRDataset(Dataset):
    """Super-Resolution dataset — generates LR/HR pairs on-the-fly.

    Args:
        image_paths: List of HR image paths.
        scale: Super-resolution scale factor.
        patch_size: HR patch size for training; None means full image.
        augment: If True, apply random flip/rotate augmentations.
    """

    def __init__(
        self,
        image_paths: List[str],
        scale: int = 4,
        patch_size: Optional[int] = 64,
        augment: bool = True,
    ):
        self.image_paths = image_paths
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def _augment_pair(self, hr: np.ndarray, lr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply consistent random flip/rotation to an HR/LR pair."""
        if np.random.random() > 0.5:
            hr = hr[:, ::-1, :].copy()
            lr = lr[:, ::-1, :].copy()
        if np.random.random() > 0.5:
            hr = hr[::-1, :, :].copy()
            lr = lr[::-1, :, :].copy()
        k = np.random.randint(0, 4)
        if k > 0:
            hr = np.rot90(hr, k).copy()
            lr = np.rot90(lr, k).copy()
        return hr, lr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr = read_image(self.image_paths[idx])
        h, w, _ = hr.shape

        if self.patch_size is not None:
            # Training: random crop
            ps = self.patch_size
            lps = ps // self.scale

            # Ensure image is large enough
            if h < ps or w < ps:
                hr = cv2.resize(hr, (max(w, ps), max(h, ps)), interpolation=cv2.INTER_CUBIC)
                h, w, _ = hr.shape

            top = np.random.randint(0, max(1, h - ps + 1))
            left = np.random.randint(0, max(1, w - ps + 1))
            hr_patch = hr[top:top + ps, left:left + ps]
            lr_patch = cv2.resize(hr_patch, (lps, lps), interpolation=cv2.INTER_CUBIC)

            if self.augment:
                hr_patch, lr_patch = self._augment_pair(hr_patch, lr_patch)

            return img_to_tensor(lr_patch), img_to_tensor(hr_patch)
        else:
            # Validation / test: full image, make dims divisible by scale
            h_new = h - h % self.scale
            w_new = w - w % self.scale
            hr_full = hr[:h_new, :w_new]
            lr_full = cv2.resize(
                hr_full, (w_new // self.scale, h_new // self.scale),
                interpolation=cv2.INTER_CUBIC
            )
            return img_to_tensor(lr_full), img_to_tensor(hr_full)


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders from config.

    Returns:
        (train_loader, val_loader) tuple.
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    data_root = data_cfg["data_root"]
    scale = data_cfg.get("scale", 4)
    patch_size = data_cfg.get("patch_size", 64)
    batch_size = train_cfg.get("batch_size", 16)
    num_workers = train_cfg.get("num_workers", 4)

    # --- Training ---
    train_split = data_cfg.get("train_split", "TRAINING SET")
    train_subs = data_cfg.get("train_subfolders", None)
    train_paths = collect_image_paths(data_root, train_split, train_subs)
    train_ds = SRDataset(train_paths, scale=scale, patch_size=patch_size, augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    # --- Validation ---
    val_split = data_cfg.get("val_split", "VALIDATION SET")
    val_subs = data_cfg.get("val_subfolders", None)
    val_paths = collect_image_paths(data_root, val_split, val_subs)
    val_ds = SRDataset(val_paths, scale=scale, patch_size=None, augment=False)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    logger.info(f"SR DataLoaders — train: {len(train_ds)}, val: {len(val_ds)}")
    return train_loader, val_loader
