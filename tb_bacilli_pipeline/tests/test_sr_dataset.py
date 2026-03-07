"""Tests for SR dataset and data loading."""
import os
import sys
import tempfile
import shutil
import numpy as np
import cv2
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def synthetic_data_root():
    """Create a temporary DDS3-like directory with synthetic BMP images."""
    root = tempfile.mkdtemp(prefix="test_sr_")
    subfolders = {
        "TRAINING SET/100% NEGATIVE/IMAGE": 4,
        "TRAINING SET/50% NEGATIVE/IMAGE": 4,
        "VALIDATION SET/50% NEGATIVE/IMAGE": 2,
    }
    for rel_path, n_images in subfolders.items():
        img_dir = os.path.join(root, rel_path)
        os.makedirs(img_dir, exist_ok=True)
        # Also create MASK dir for TB tests
        mask_dir = img_dir.replace("/IMAGE", "/MASK")
        os.makedirs(mask_dir, exist_ok=True)
        for i in range(1, n_images + 1):
            # Create a small synthetic colour image (64×64)
            img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(img_dir, f"MSC{i}.bmp"), img)
            # Create a binary mask
            mask = np.zeros((64, 64, 3), dtype=np.uint8)
            mask[20:40, 20:40] = 255  # synthetic blob
            cv2.imwrite(os.path.join(mask_dir, f"MSC{i}.bmp"), mask)
    yield root
    shutil.rmtree(root)


class TestCollectImagePaths:
    """Test image collection from DDS3 folder structure."""

    def test_collect_training(self, synthetic_data_root):
        from sr_project.dataset import collect_image_paths

        paths = collect_image_paths(
            data_root=synthetic_data_root,
            split="TRAINING SET",
            subfolders=["100% NEGATIVE", "50% NEGATIVE"],
        )
        assert len(paths) == 8  # 4 + 4

    def test_collect_validation(self, synthetic_data_root):
        from sr_project.dataset import collect_image_paths

        paths = collect_image_paths(
            data_root=synthetic_data_root,
            split="VALIDATION SET",
            subfolders=["50% NEGATIVE"],
        )
        assert len(paths) == 2

    def test_all_paths_exist(self, synthetic_data_root):
        from sr_project.dataset import collect_image_paths

        paths = collect_image_paths(
            data_root=synthetic_data_root,
            split="TRAINING SET",
            subfolders=["100% NEGATIVE"],
        )
        for p in paths:
            assert os.path.isfile(p), f"File not found: {p}"


class TestSRDataset:
    """Test the SRDataset class."""

    def test_dataset_length(self, synthetic_data_root):
        from sr_project.dataset import SRDataset, collect_image_paths

        paths = collect_image_paths(
            synthetic_data_root, "TRAINING SET", ["100% NEGATIVE"]
        )
        ds = SRDataset(paths, scale=4, patch_size=32, augment=True)
        assert len(ds) == 4

    def test_output_shapes_training(self, synthetic_data_root):
        from sr_project.dataset import SRDataset, collect_image_paths

        paths = collect_image_paths(
            synthetic_data_root, "TRAINING SET", ["100% NEGATIVE"]
        )
        ds = SRDataset(paths, scale=4, patch_size=32, augment=True)
        lr, hr = ds[0]
        # LR patch = patch_size // scale = 8
        assert lr.shape == (3, 8, 8), f"LR shape: {lr.shape}"
        assert hr.shape == (3, 32, 32), f"HR shape: {hr.shape}"

    def test_output_shapes_validation(self, synthetic_data_root):
        from sr_project.dataset import SRDataset, collect_image_paths

        paths = collect_image_paths(
            synthetic_data_root, "VALIDATION SET", ["50% NEGATIVE"]
        )
        ds = SRDataset(paths, scale=4, patch_size=None, augment=False)
        lr, hr = ds[0]
        # Full image: 64×64 HR → 16×16 LR
        assert lr.shape[0] == 3
        assert hr.shape[0] == 3
        assert hr.shape[1] == lr.shape[1] * 4
        assert hr.shape[2] == lr.shape[2] * 4

    def test_pixel_range(self, synthetic_data_root):
        from sr_project.dataset import SRDataset, collect_image_paths

        paths = collect_image_paths(
            synthetic_data_root, "TRAINING SET", ["50% NEGATIVE"]
        )
        ds = SRDataset(paths, scale=4, patch_size=32, augment=False)
        lr, hr = ds[0]
        assert lr.min() >= 0.0 and lr.max() <= 1.0
        assert hr.min() >= 0.0 and hr.max() <= 1.0


class TestBuildDataloaders:
    """Test dataloader construction."""

    def test_dataloaders_run(self, synthetic_data_root):
        from sr_project.dataset import build_dataloaders

        config = {
            "data": {
                "data_root": synthetic_data_root,
                "train_split": "TRAINING SET",
                "train_subfolders": ["100% NEGATIVE", "50% NEGATIVE"],
                "val_split": "VALIDATION SET",
                "val_subfolders": ["50% NEGATIVE"],
                "scale": 4,
                "patch_size": 32,
            },
            "training": {"batch_size": 2, "num_workers": 0},
        }
        train_dl, val_dl = build_dataloaders(config)
        lr_batch, hr_batch = next(iter(train_dl))
        assert lr_batch.shape[0] <= 2
        assert lr_batch.ndim == 4
