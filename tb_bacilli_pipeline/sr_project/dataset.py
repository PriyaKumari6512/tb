"""
SR Dataset for DDS3.
- Walks TRAINING SET subfolders to collect HR images
- Generates LR on-the-fly via bicubic downsampling
- Paired augmentations (crop, flip, rotate) for training
- Full-image loading for validation/test
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from sr_project.utils import load_config, read_image, img_to_tensor

logger = logging.getLogger(__name__)


def collect_image_paths(
    root: str,
    subfolders: List[str],
    image_dir: str = "IMAGE",
    extension: str = ".bmp",
) -> List[str]:
    """Collect all image paths from multiple subfolders.
    
    Returns list of absolute paths sorted for reproducibility.
    """
    paths = []
    for subfolder in subfolders:
        folder = os.path.join(root, subfolder, image_dir)
        if not os.path.isdir(folder):
            logger.warning(f"Subfolder not found, skipping: {folder}")
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(extension.lower()):
                paths.append(os.path.join(folder, fname))
    logger.info(f"Collected {len(paths)} images from {len(subfolders)} subfolder(s)")
    return paths


class SRDataset(Dataset):
    """Super-Resolution dataset — generates LR/HR pairs.

    Training: random crop + augmentation, LR generated on-the-fly.
    Val/Test: full-image, LR generated on-the-fly.
    """

    def __init__(
        self,
        image_paths: List[str],
        scale: int = 4,
        patch_size: int = 64,
        is_train: bool = True,
    ):
        self.image_paths = image_paths
        self.scale = scale
        self.patch_size = patch_size  # HR patch size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.image_paths)

    def _augment(self, hr: np.ndarray, lr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random flip and rotation to both HR and LR identically."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            hr = hr[:, ::-1, :].copy()
            lr = lr[:, ::-1, :].copy()
        # Random vertical flip
        if np.random.random() > 0.5:
            hr = hr[::-1, :, :].copy()
            lr = lr[::-1, :, :].copy()
        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            hr = np.rot90(hr, k).copy()
            lr = np.rot90(lr, k).copy()
        return hr, lr

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hr = read_image(self.image_paths[idx])
        h, w, _ = hr.shape

        if self.is_train:
            # Random crop (HR)
            ps = self.patch_size
            lps = ps // self.scale

            # Ensure image is large enough
            if h < ps or w < ps:
                hr = cv2.resize(hr, (max(w, ps), max(h, ps)), interpolation=cv2.INTER_CUBIC)
                h, w, _ = hr.shape

            top = np.random.randint(0, h - ps + 1)
            left = np.random.randint(0, w - ps + 1)
            hr_patch = hr[top:top + ps, left:left + ps]

            # Generate LR by downsampling
            lr_patch = cv2.resize(hr_patch, (lps, lps), interpolation=cv2.INTER_CUBIC)

            # Augmentation
            hr_patch, lr_patch = self._augment(hr_patch, lr_patch)

            return {
                "lr": img_to_tensor(lr_patch),
                "hr": img_to_tensor(hr_patch),
            }
        else:
            # Validation / test: full image
            # Ensure dimensions are divisible by scale
            h_new = h - h % self.scale
            w_new = w - w % self.scale
            hr = hr[:h_new, :w_new]
            lr = cv2.resize(hr, (w_new // self.scale, h_new // self.scale),
                            interpolation=cv2.INTER_CUBIC)
            return {
                "lr": img_to_tensor(lr),
                "hr": img_to_tensor(hr),
                "path": self.image_paths[idx],
            }


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders from config."""
    data_cfg = cfg["data"]
    root = data_cfg["root"]
    scale = data_cfg["scale"]
    patch_size = data_cfg["patch_size"]

    # Collect paths
    train_paths = collect_image_paths(
        root, data_cfg["train_subfolders"],
        data_cfg["image_dir"], data_cfg["extension"],
    )
    val_paths = collect_image_paths(
        root, [data_cfg["val_subfolder"]],
        data_cfg["image_dir"], data_cfg["extension"],
    )
    test_paths = collect_image_paths(
        root, [data_cfg["test_subfolder"]],
        data_cfg["image_dir"], data_cfg["extension"],
    )

    # Datasets
    train_ds = SRDataset(train_paths, scale, patch_size, is_train=True)
    val_ds = SRDataset(val_paths, scale, patch_size, is_train=False)
    test_ds = SRDataset(test_paths, scale, patch_size, is_train=False)

    # DataLoaders
    loader_kwargs = {
        "num_workers": data_cfg.get("num_workers", 4),
        "pin_memory": data_cfg.get("pin_memory", True),
    }
    train_loader = DataLoader(
        train_ds, batch_size=data_cfg["batch_size"], shuffle=True,
        drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, **loader_kwargs,
    )

    logger.info(f"DataLoaders — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Quick verification of dataset loading."""
    import argparse
    parser = argparse.ArgumentParser(description="Verify SR dataset loading")
    parser.add_argument("--config", type=str, default="configs/sr_config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # Verify shapes
    batch = next(iter(train_loader))
    scale = cfg["data"]["scale"]
    ps = cfg["data"]["patch_size"]
    assert batch["lr"].shape == (cfg["data"]["batch_size"], 3, ps // scale, ps // scale), \
        f"LR shape mismatch: {batch['lr'].shape}"
    assert batch["hr"].shape == (cfg["data"]["batch_size"], 3, ps, ps), \
        f"HR shape mismatch: {batch['hr'].shape}"
    print(f"✓ Train batch — LR: {batch['lr'].shape}, HR: {batch['hr'].shape}")

    val_batch = next(iter(val_loader))
    print(f"✓ Val batch — LR: {val_batch['lr'].shape}, HR: {val_batch['hr'].shape}")
    print(f"✓ Dataset verification passed!")
