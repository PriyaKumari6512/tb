"""Tests for Swin2SR model forward passes (CPU, small inputs)."""
import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestSwin2SRForward:
    """Test Swin2SR model forward pass with small inputs."""

    def test_forward_shape_x4(self):
        from swin2sr.model import Swin2SRModel

        model = Swin2SRModel(pretrained=False, upscale=4)
        model.eval()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 3, 256, 256), f"Output shape: {y.shape}"

    def test_output_range(self):
        from swin2sr.model import Swin2SRModel

        model = Swin2SRModel(pretrained=False, upscale=4)
        model.eval()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert torch.isfinite(y).all(), "Non-finite values in output"
        assert y.min() >= 0.0, f"Output min {y.min()} < 0"
        assert y.max() <= 1.0, f"Output max {y.max()} > 1"

    def test_batch_consistency(self):
        from swin2sr.model import Swin2SRModel

        model = Swin2SRModel(pretrained=False, upscale=4)
        model.eval()
        x = torch.rand(1, 3, 64, 64)
        x_batch = x.repeat(2, 1, 1, 1)
        with torch.no_grad():
            y1 = model(x)
            y_batch = model(x_batch)
        torch.testing.assert_close(y1, y_batch[0:1], atol=1e-5, rtol=1e-5)

    def test_parameter_grouping(self):
        from swin2sr.model import Swin2SRModel

        model = Swin2SRModel(pretrained=False, upscale=4)
        encoder_params = model.get_encoder_params()
        head_params = model.get_head_params()
        total = sum(p.numel() for p in model.parameters())
        grouped = sum(p.numel() for p in encoder_params) + sum(p.numel() for p in head_params)
        assert grouped == total, f"Parameter grouping incomplete: {grouped} != {total}"

    def test_freeze_unfreeze_encoder(self):
        from swin2sr.model import Swin2SRModel

        model = Swin2SRModel(pretrained=False, upscale=4)
        model.freeze_encoder()
        for param in model.get_encoder_params():
            assert not param.requires_grad
        for param in model.get_head_params():
            assert param.requires_grad

        model.unfreeze_encoder()
        for param in model.get_encoder_params():
            assert param.requires_grad

    def test_build_model_from_config(self):
        from swin2sr.model import build_model

        cfg = {
            "model": {
                "pretrained": False,
                "upscale": 4,
                "finetuned_weights": None,
            }
        }
        model = build_model(cfg)
        assert model is not None
        x = torch.rand(1, 3, 64, 64)
        model.eval()
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 3, 256, 256)
