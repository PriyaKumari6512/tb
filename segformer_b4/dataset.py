"""
Dataset & data loading for TB bacilli segmentation with SegFormer B4.
Supports DDS3-style IMAGE/MASK subfolder layout.
Includes heavy data augmentation to improve real-world generalization
and reduce false positives on negative images.
"""

import os
import logging
from typing import List, Tuple, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)


def collect_image_mask_pairs(
    data_root: str,
    split: str = "TRAINING SET",
    subfolders: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Collect (image_path, mask_path, subfolder_name) tuples.
    """
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    if subfolders is None:
        subfolders = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])

    pairs = []
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
            pairs.append((img_path, mask_path, sf))

    logger.info(f"Collected {len(pairs)} image-mask pairs from {split}")
    return pairs


def get_train_transforms(image_size: int = 512) -> A.Compose:
    """Heavy augmentation pipeline to improve generalization on real-world data.

    Includes spatial, colour, noise/blur, morphological and dropout augmentations
    to replicate real-world microscopy image variations and reduce false positives.
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
        A.ChannelShuffle(p=0.1),

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


def get_val_transforms(image_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class TBDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str, str]], transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path, _ = self.pairs[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()

        return image, mask


def build_dataloaders(cfg: dict) -> dict:
    """Build train/val/test dataloaders from config."""
    data_cfg = cfg["data"]
    data_root = data_cfg["data_root"]
    image_size = data_cfg.get("image_size", 512)
    batch_size = data_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 4)

    loaders = {}

    # Training
    train_subs = data_cfg.get("train_subfolders", None)
    train_pairs = collect_image_mask_pairs(data_root, "TRAINING SET", train_subs)
    train_ds = TBDataset(train_pairs, get_train_transforms(image_size))

    # Weighted sampling
    subfolder_weights = data_cfg.get("subfolder_weights", {})
    if subfolder_weights:
        weights = [subfolder_weights.get(p[2], 1.0) for p in train_pairs]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        loaders["train"] = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
    else:
        loaders["train"] = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

    # Validation
    val_subs = data_cfg.get("val_subfolders", None)
    val_pairs = collect_image_mask_pairs(data_root, "VALIDATION SET", val_subs)
    val_ds = TBDataset(val_pairs, get_val_transforms(image_size))
    loaders["val"] = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Test
    test_subs = data_cfg.get("test_subfolders", None)
    test_pairs = collect_image_mask_pairs(data_root, "TEST SET", test_subs)
    test_ds = TBDataset(test_pairs, get_val_transforms(image_size))
    loaders["test"] = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return loaders
