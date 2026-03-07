"""
TB Dataset for DDS3 — IMAGE/MASK paired loading.
- Walks multiple subfolders (100% / 90% / 50% NEGATIVE)
- Albumentations augmentations with mask co-transforms
- Returns image tensor + binary mask tensor + metadata
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sr_project.utils import load_config, read_image

logger = logging.getLogger(__name__)


def collect_image_mask_pairs(
    root: str,
    subfolders: List[str],
    image_dir: str = "IMAGE",
    mask_dir: str = "MASK",
    extension: str = ".bmp",
) -> List[Dict[str, str]]:
    """Collect paired image/mask paths from multiple subfolders."""
    pairs = []
    for subfolder in subfolders:
        img_folder = os.path.join(root, subfolder, image_dir)
        msk_folder = os.path.join(root, subfolder, mask_dir)

        if not os.path.isdir(img_folder):
            logger.warning(f"Image folder not found, skipping: {img_folder}")
            continue
        if not os.path.isdir(msk_folder):
            logger.warning(f"Mask folder not found, skipping: {msk_folder}")
            continue

        for fname in sorted(os.listdir(img_folder)):
            if not fname.lower().endswith(extension.lower()):
                continue
            img_path = os.path.join(img_folder, fname)
            msk_path = os.path.join(msk_folder, fname)

            if not os.path.exists(msk_path):
                logger.warning(f"Mask not found for {fname} in {subfolder}, skipping")
                continue

            pairs.append({
                "image": img_path,
                "mask": msk_path,
                "subfolder": subfolder,
                "filename": fname,
            })

    logger.info(f"Collected {len(pairs)} image/mask pairs from {len(subfolders)} subfolder(s)")
    return pairs


def get_train_transforms(image_size: int = 512) -> A.Compose:
    """Training augmentations — co-applied to image and mask."""
    return A.Compose([
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.5, 1.0),
            ratio=(0.75, 1.333),
            interpolation=cv2.INTER_LINEAR,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        ], p=0.5),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 512) -> A.Compose:
    """Validation/test transforms — just resize + normalize."""
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class TBDataset(Dataset):
    """TB bacilli segmentation dataset with IMAGE/MASK pairs."""

    def __init__(
        self,
        pairs: List[Dict[str, str]],
        transforms: Optional[A.Compose] = None,
        image_size: int = 512,
    ):
        self.pairs = pairs
        self.transforms = transforms
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        pair = self.pairs[idx]

        # Read image (RGB)
        image = read_image(pair["image"])

        # Read mask (grayscale → binary)
        mask = cv2.imread(pair["mask"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {pair['mask']}")
        # Binarize: any non-zero pixel is bacilli
        mask = (mask > 0).astype(np.uint8)

        # Apply transforms (co-applied to image and mask)
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]  # Already a tensor (C, H, W)
            mask = augmented["mask"]    # (H, W) tensor
        else:
            # Fallback: resize and convert
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size),
                              interpolation=cv2.INTER_NEAREST)
            image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
            mask = torch.from_numpy(mask)

        mask = mask.long()

        return {
            "image": image,
            "mask": mask,
            "filename": pair["filename"],
            "subfolder": pair["subfolder"],
        }


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders from config."""
    data_cfg = cfg["data"]
    root = data_cfg["root"]
    image_size = data_cfg.get("image_size", 512)

    # Collect pairs
    train_pairs = collect_image_mask_pairs(
        root, data_cfg["train_subfolders"],
        data_cfg["image_dir"], data_cfg["mask_dir"],
        data_cfg["extension"],
    )
    val_pairs = collect_image_mask_pairs(
        root, [data_cfg["val_subfolder"]],
        data_cfg["image_dir"], data_cfg["mask_dir"],
        data_cfg["extension"],
    )
    test_pairs = collect_image_mask_pairs(
        root, [data_cfg["test_subfolder"]],
        data_cfg["image_dir"], data_cfg["mask_dir"],
        data_cfg["extension"],
    )

    # Transforms
    train_transforms = get_train_transforms(image_size)
    val_transforms = get_val_transforms(image_size)

    # Datasets
    train_ds = TBDataset(train_pairs, train_transforms, image_size)
    val_ds = TBDataset(val_pairs, val_transforms, image_size)
    test_ds = TBDataset(test_pairs, val_transforms, image_size)

    # Weighted sampler: oversample images with bacilli
    # Approximate: 100% NEGATIVE = weight 0.5, 90% NEGATIVE = weight 1.0, 50% NEGATIVE = weight 1.5
    subfolder_weights = {
        "TRAINING SET/100% NEGATIVE": 0.5,
        "TRAINING SET/90% NEGATIVE": 1.0,
        "TRAINING SET/50% NEGATIVE": 1.5,
    }
    sample_weights = []
    for pair in train_pairs:
        w = subfolder_weights.get(pair["subfolder"], 1.0)
        sample_weights.append(w)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_pairs),
        replacement=True,
    )

    loader_kwargs = {
        "num_workers": data_cfg.get("num_workers", 4),
        "pin_memory": data_cfg.get("pin_memory", True),
    }
    train_loader = DataLoader(
        train_ds, batch_size=data_cfg["batch_size"],
        sampler=sampler, drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=data_cfg["batch_size"],
        shuffle=False, **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, **loader_kwargs,
    )

    logger.info(f"TB DataLoaders — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify TB dataset loading")
    parser.add_argument("--config", type=str, default="configs/tb_config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    batch = next(iter(train_loader))
    print(f"✓ Train batch — image: {batch['image'].shape}, mask: {batch['mask'].shape}")
    print(f"  Unique mask values: {batch['mask'].unique().tolist()}")
    print(f"  Filenames: {batch['filename'][:3]}")
    print(f"✓ Dataset verification passed!")
