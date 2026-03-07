"""Tests for model forward passes (CPU, small inputs)."""
import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSwinIRForward:
    """Test SwinIR model forward pass with small inputs."""

    def test_forward_shape(self):
        from sr_project.model import SwinIR

        model = SwinIR(
            img_size=64,
            in_chans=3,
            embed_dim=60,
            depths=[2, 2],
            num_heads=[3, 3],
            window_size=8,
            upscale=4,
        )
        model.eval()
        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 3, 64, 64), f"Output shape: {y.shape}"

    def test_forward_scale2(self):
        from sr_project.model import SwinIR

        model = SwinIR(
            img_size=64,
            in_chans=3,
            embed_dim=60,
            depths=[2],
            num_heads=[3],
            window_size=8,
            upscale=2,
        )
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 3, 64, 64)

    def test_output_range(self):
        from sr_project.model import SwinIR

        model = SwinIR(
            img_size=64,
            in_chans=3,
            embed_dim=60,
            depths=[2],
            num_heads=[3],
            window_size=8,
            upscale=4,
        )
        model.eval()
        x = torch.rand(1, 3, 16, 16)  # [0,1] input
        with torch.no_grad():
            y = model(x)
        # Output should be finite
        assert torch.isfinite(y).all(), "Non-finite values in output"

    def test_batch_consistency(self):
        from sr_project.model import SwinIR

        model = SwinIR(
            img_size=64,
            in_chans=3,
            embed_dim=60,
            depths=[2],
            num_heads=[3],
            window_size=8,
            upscale=4,
        )
        model.eval()
        x = torch.rand(1, 3, 16, 16)
        x_batch = x.repeat(2, 1, 1, 1)
        with torch.no_grad():
            y1 = model(x)
            y_batch = model(x_batch)
        torch.testing.assert_close(y1, y_batch[0:1], atol=1e-5, rtol=1e-5)


class TestSegFormerForward:
    """Test TB SegFormer model forward pass."""

    def test_forward_shape(self):
        from tb_project.model import TBSegFormer

        model = TBSegFormer(
            backbone="nvidia/mit-b0",  # smallest for testing
            num_classes=2,
            image_size=64,
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 2, 64, 64), f"Logits shape: {logits.shape}"

    def test_predict(self):
        from tb_project.model import TBSegFormer

        model = TBSegFormer(
            backbone="nvidia/mit-b0",
            num_classes=2,
            image_size=64,
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            mask = model.predict(x)
        assert mask.shape == (1, 1, 64, 64)
        assert mask.dtype == torch.float32
        assert mask.min() >= 0.0 and mask.max() <= 1.0


class TestLosses:
    """Test loss functions."""

    def test_dice_loss(self):
        from tb_project.model import DiceLoss

        loss_fn = DiceLoss()
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = loss_fn(pred, target)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0

    def test_dice_bce_loss(self):
        from tb_project.model import DiceBCELoss

        loss_fn = DiceBCELoss(pos_weight=5.0)
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_focal_loss(self):
        from tb_project.model import FocalLoss

        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = loss_fn(pred, target)
        assert loss.item() >= 0


class TestPostprocessing:
    """Test bacilli detection post-processing."""

    def test_detect_bacilli(self):
        from tb_project.postprocess import detect_bacilli
        import numpy as np

        # Create a mask with two distinct blobs
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5:15, 5:15] = 1    # blob 1: area 100
        mask[40:50, 40:50] = 1  # blob 2: area 100
        detections = detect_bacilli(mask, min_area=10)
        assert len(detections) == 2
        for det in detections:
            assert "bbox" in det
            assert "area" in det
            assert det["area"] >= 10

    def test_count_bacilli(self):
        from tb_project.postprocess import count_bacilli
        import numpy as np

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        count = count_bacilli(mask, min_area=5)
        assert count == 1

    def test_empty_mask(self):
        from tb_project.postprocess import detect_bacilli
        import numpy as np

        mask = np.zeros((64, 64), dtype=np.uint8)
        detections = detect_bacilli(mask, min_area=10)
        assert len(detections) == 0


class TestMetrics:
    """Test metric computations."""

    def test_dice_score(self):
        from tb_project.utils import dice_score

        pred = torch.ones(1, 1, 32, 32)
        target = torch.ones(1, 1, 32, 32)
        d = dice_score(pred, target)
        assert abs(d.item() - 1.0) < 1e-5

    def test_iou_score(self):
        from tb_project.utils import iou_score

        pred = torch.ones(1, 1, 32, 32)
        target = torch.ones(1, 1, 32, 32)
        iou = iou_score(pred, target)
        assert abs(iou.item() - 1.0) < 1e-5

    def test_psnr(self):
        from sr_project.utils import compute_psnr
        import numpy as np

        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        psnr = compute_psnr(img, img)
        assert psnr == float("inf") or psnr > 50  # identical images
