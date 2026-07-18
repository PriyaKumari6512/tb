"""
TB Dataset for DDS3 — IMAGE/MASK paired loading.
- Walks multiple subfolders (100% / 90% / 50% NEGATIVE)
- Heavy Albumentations augmentations with mask co-transforms
- Returns (image_tensor, mask_tensor) pairs for binary segmentation
"""

import logging
import os
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


def collect_image_mask_pairs(
    data_root: str,
    split: str = "TRAINING SET",
    subfolders: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Collect (image_path, mask_path) tuples from DDS3-style directory.

    Args:
        data_root: Root directory (e.g. './DDS3').
        split: Split directory name (e.g. 'TRAINING SET').
        subfolders: Optional list of subfolder names. If None, all are used.

    Returns:
        List of (image_path, mask_path) tuples.
    """
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    if subfolders is None:
        subfolders = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])

    pairs: List[Tuple[str, str]] = []
    for sf in subfolders:
        img_dir = os.path.join(split_dir, sf, "IMAGE")
        mask_dir = os.path.join(split_dir, sf, "MASK")
        if not os.path.isdir(img_dir):
            logger.warning(f"IMAGE dir not found: {img_dir}")
            continue

        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif')):
                continue
            img_path = os.path.join(img_dir, fname)
            mask_path = os.path.join(mask_dir, fname)
            if not os.path.isfile(mask_path):
                logger.warning(f"Mask missing for {img_path}")
                continue
            pairs.append((img_path, mask_path))

    logger.info(f"Collected {len(pairs)} image-mask pairs from {split}")
    return pairs


def _get_train_transforms(image_size: int) -> A.Compose:
    """Heavy augmentation pipeline for training.

    Includes spatial, colour, noise/blur, morphological and dropout augmentations
    to replicate real-world microscopy image variations and reduce false positives
    on negative images.
    """
    return A.Compose([
        # --- Spatial ---
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.4, 1.0),
            ratio=(0.75, 1.333),
            interpolation=cv2.INTER_LINEAR,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=30, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, p=0.4),
        A.Perspective(scale=(0.02, 0.06), p=0.3),
        A.ElasticTransform(alpha=60, sigma=6, p=0.3),

        # --- Colour / intensity ---
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.08, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
        ], p=0.6),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),

        # --- Noise / blur ---
        A.OneOf([
            A.GaussNoise(std_range=(0.04, 0.12), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.85, 1.15), per_channel=True, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.Defocus(radius=(1, 3), p=1.0),
        ], p=0.35),
        A.ImageCompression(quality_range=(60, 95), p=0.2),

        # --- Dropout / occlusion ---
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0, p=0.2,
        ),

        # --- Normalize & convert ---
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def _get_val_transforms(image_size: int) -> A.Compose:
    """Validation/test: just resize + normalize."""
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class TBDataset(Dataset):
    """TB bacilli segmentation dataset.

    Args:
        pairs: List of (image_path, mask_path) tuples.
        image_size: Spatial size to resize images/masks to.
        augment: If True, apply heavy training augmentations.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        image_size: int = 512,
        augment: bool = False,
    ):
        self.pairs = pairs
        self.image_size = image_size
        self.transform = (
            _get_train_transforms(image_size) if augment else _get_val_transforms(image_size)
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")
        mask = (mask > 127).astype(np.uint8)

        augmented = self.transform(image=image, mask=mask)
        image_t: torch.Tensor = augmented["image"]          # (3, H, W) float32
        mask_t: torch.Tensor = augmented["mask"].unsqueeze(0).float()  # (1, H, W) float32

        return image_t, mask_t


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders from config.

    Args:
        cfg: Config dict with 'data' and 'training' sections.

    Returns:
        (train_loader, val_loader) tuple.
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    data_root = data_cfg["data_root"]
    image_size = data_cfg.get("image_size", 512)
    batch_size = train_cfg.get("batch_size", 8)
    num_workers = train_cfg.get("num_workers", 4)

    # --- Training ---
    train_split = data_cfg.get("train_split", "TRAINING SET")
    train_subs = data_cfg.get("train_subfolders", None)
    train_pairs = collect_image_mask_pairs(data_root, train_split, train_subs)
    train_ds = TBDataset(train_pairs, image_size=image_size, augment=True)

    # Weighted sampling: give each pair a weight based on its subfolder
    subfolder_weights: dict = data_cfg.get("subfolder_weights", {})
    if subfolder_weights and train_subs:
        # Map back from the paired list to subfolder name
        sample_weights = []
        for img_path, _ in train_pairs:
            sf = _infer_subfolder(img_path, data_root, train_split)
            sample_weights.append(subfolder_weights.get(sf, 1.0))
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

    # --- Validation ---
    val_split = data_cfg.get("val_split", "VALIDATION SET")
    val_subs = data_cfg.get("val_subfolders", None)
    val_pairs = collect_image_mask_pairs(data_root, val_split, val_subs)
    val_ds = TBDataset(val_pairs, image_size=image_size, augment=False)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    logger.info(f"TB DataLoaders — train: {len(train_ds)}, val: {len(val_ds)}")
    return train_loader, val_loader


def _infer_subfolder(img_path: str, data_root: str, split: str) -> str:
    """Extract subfolder name from image path."""
    rel = os.path.relpath(img_path, os.path.join(data_root, split))
    parts = rel.split(os.sep)
    return parts[0] if parts else ""
