"""Tests for TB segmentation dataset and loading."""
import os
import sys
import tempfile
import shutil
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def synthetic_data_root():
    """Create temporary DDS3-like structure with IMAGE + MASK pairs."""
    root = tempfile.mkdtemp(prefix="test_tb_")
    subfolders = {
        "TRAINING SET/50% NEGATIVE": 6,
        "VALIDATION SET/50% NEGATIVE": 2,
    }
    for rel_path, n in subfolders.items():
        img_dir = os.path.join(root, rel_path, "IMAGE")
        mask_dir = os.path.join(root, rel_path, "MASK")
        os.makedirs(img_dir)
        os.makedirs(mask_dir)
        for i in range(1, n + 1):
            img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(img_dir, f"MSC{i}.bmp"), img)
            # Binary mask: 0 background, 255 foreground blob
            mask = np.zeros((64, 64, 3), dtype=np.uint8)
            mask[25:40, 25:40] = 255
            cv2.imwrite(os.path.join(mask_dir, f"MSC{i}.bmp"), mask)
    yield root
    shutil.rmtree(root)


class TestCollectImageMaskPairs:
    """Test pair collection."""

    def test_pair_count(self, synthetic_data_root):
        from tb_project.dataset import collect_image_mask_pairs

        pairs = collect_image_mask_pairs(
            synthetic_data_root, "TRAINING SET", ["50% NEGATIVE"]
        )
        assert len(pairs) == 6

    def test_masks_exist(self, synthetic_data_root):
        from tb_project.dataset import collect_image_mask_pairs

        pairs = collect_image_mask_pairs(
            synthetic_data_root, "VALIDATION SET", ["50% NEGATIVE"]
        )
        for img_p, mask_p in pairs:
            assert os.path.isfile(img_p)
            assert os.path.isfile(mask_p)


class TestTBDataset:
    """Test the TBDataset class."""

    def test_output_shapes(self, synthetic_data_root):
        from tb_project.dataset import TBDataset, collect_image_mask_pairs

        pairs = collect_image_mask_pairs(
            synthetic_data_root, "TRAINING SET", ["50% NEGATIVE"]
        )
        ds = TBDataset(pairs, image_size=32, augment=True)
        img, mask = ds[0]
        assert img.shape == (3, 32, 32), f"Image shape: {img.shape}"
        assert mask.shape == (1, 32, 32), f"Mask shape: {mask.shape}"

    def test_mask_is_binary(self, synthetic_data_root):
        from tb_project.dataset import TBDataset, collect_image_mask_pairs

        pairs = collect_image_mask_pairs(
            synthetic_data_root, "TRAINING SET", ["50% NEGATIVE"]
        )
        ds = TBDataset(pairs, image_size=32, augment=False)
        for i in range(len(ds)):
            _, mask = ds[i]
            unique = mask.unique()
            assert all(v in [0.0, 1.0] for v in unique.tolist()), (
                f"Non-binary mask values: {unique}"
            )

    def test_val_no_augment(self, synthetic_data_root):
        from tb_project.dataset import TBDataset, collect_image_mask_pairs

        pairs = collect_image_mask_pairs(
            synthetic_data_root, "VALIDATION SET", ["50% NEGATIVE"]
        )
        ds = TBDataset(pairs, image_size=32, augment=False)
        img, mask = ds[0]
        assert img.shape[1] == 32 and img.shape[2] == 32


class TestTBDataloaders:
    """Test dataloader construction."""

    def test_builds_ok(self, synthetic_data_root):
        from tb_project.dataset import build_dataloaders

        config = {
            "data": {
                "data_root": synthetic_data_root,
                "train_split": "TRAINING SET",
                "train_subfolders": ["50% NEGATIVE"],
                "val_split": "VALIDATION SET",
                "val_subfolders": ["50% NEGATIVE"],
                "image_size": 32,
                "subfolder_weights": {"50% NEGATIVE": 1.0},
            },
            "training": {"batch_size": 2, "num_workers": 0},
        }
        train_dl, val_dl = build_dataloaders(config)
        imgs, masks = next(iter(train_dl))
        assert imgs.ndim == 4
        assert masks.ndim == 4
