"""
Dataset & data loading for TB bacilli segmentation with SegFormer B4.
Supports DDS3-style IMAGE/MASK subfolder layout.
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
    return A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
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
