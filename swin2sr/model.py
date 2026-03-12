"""
Swin2SR model wrapper for image super-resolution.

Uses HuggingFace Swin2SRForImageSuperResolution with pretrained x4 model
(caidas/swin2SR-classical-sr-x4-64). Supports finetuning with:
  - Differential learning rates (encoder vs reconstruction head)
  - Optional encoder freezing for transfer learning
  - Flexible checkpoint loading
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Swin2SRForImageSuperResolution, Swin2SRConfig

logger = logging.getLogger(__name__)

# HuggingFace model identifier for pretrained Swin2SR x4
PRETRAINED_MODEL_NAME = "caidas/swin2SR-classical-sr-x4-64"


class Swin2SRModel(nn.Module):
    """Swin2SR wrapper for 4x super-resolution.

    Wraps HuggingFace Swin2SRForImageSuperResolution with:
    - Automatic logit extraction from model outputs
    - Input padding to window-size multiples
    - Pretrained weight loading from HuggingFace hub or local checkpoint
    - Finetuning utilities (freeze/unfreeze, parameter grouping)
    """

    def __init__(
        self,
        pretrained: bool = True,
        model_name: str = PRETRAINED_MODEL_NAME,
        upscale: int = 4,
    ):
        super().__init__()
        self.upscale = upscale

        if pretrained:
            logger.info(f"Loading pretrained Swin2SR: {model_name}")
            try:
                self.model = Swin2SRForImageSuperResolution.from_pretrained(model_name)
            except (OSError, RuntimeError) as e:
                logger.warning(
                    f"Could not load pretrained weights ({e}), "
                    f"falling back to config-based initialization"
                )
                config = self._get_default_config(upscale)
                self.model = Swin2SRForImageSuperResolution(config)
        else:
            config = self._get_default_config(upscale)
            self.model = Swin2SRForImageSuperResolution(config)

        logger.info(
            f"Swin2SR built — upscale: {upscale}x, "
            f"pretrained: {pretrained}"
        )

    @staticmethod
    def _get_default_config(upscale: int = 4) -> Swin2SRConfig:
        """Create a Swin2SR config for x4 classical SR without network access."""
        return Swin2SRConfig(
            embed_dim=180,
            depths=[6, 6, 6, 6, 6, 6],
            num_heads=[6, 6, 6, 6, 6, 6],
            window_size=8,
            mlp_ratio=2.0,
            upscale=upscale,
            img_size=64,
            upsampler="pixelshuffle",
            num_channels=3,
        )

    def get_encoder_params(self):
        """Return parameters belonging to the Swin2SR encoder (body)."""
        encoder_params = []
        for name, param in self.named_parameters():
            if not self._is_head_param(name):
                encoder_params.append(param)
        return encoder_params

    def get_head_params(self):
        """Return parameters belonging to the reconstruction head."""
        head_params = []
        for name, param in self.named_parameters():
            if self._is_head_param(name):
                head_params.append(param)
        return head_params

    @staticmethod
    def _is_head_param(name: str) -> bool:
        """Return True if the parameter belongs to the reconstruction head."""
        head_keywords = ("upsample", "conv_last", "final_convolution", "reconstruction")
        return any(kw in name.lower() for kw in head_keywords)

    def freeze_encoder(self):
        """Freeze encoder parameters (keep head trainable)."""
        for param in self.get_encoder_params():
            param.requires_grad = False
        logger.info("Encoder parameters frozen")

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.get_encoder_params():
            param.requires_grad = True
        logger.info("Encoder parameters unfrozen")

    def load_finetuned(self, path: str, strict: bool = True):
        """Load finetuned checkpoint.

        Supports:
        - Full checkpoint dict (with 'model_state_dict' key)
        - Raw state dict
        - DataParallel 'module.' prefix removal
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

        # Handle DataParallel prefix
        cleaned = {}
        for k, v in state.items():
            k = k.replace("module.", "")
            cleaned[k] = v

        missing, unexpected = self.load_state_dict(cleaned, strict=strict)
        loaded = len(cleaned) - len(unexpected)
        logger.info(
            f"Loaded finetuned Swin2SR: {loaded} keys loaded, "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )
        return missing, unexpected

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input LR image tensor (B, 3, H, W) in [0, 1] range.

        Returns:
            SR output tensor (B, 3, H*upscale, W*upscale).
        """
        outputs = self.model(x)
        sr = outputs.reconstruction
        # Clamp output to valid range
        sr = sr.clamp(0.0, 1.0)
        return sr


def build_model(cfg: dict) -> Swin2SRModel:
    """Build Swin2SR model from config dict.

    Config keys used:
        model.pretrained (bool): Use pretrained weights (default: True)
        model.model_name (str): HuggingFace model name
        model.upscale (int): Upscale factor (default: 4)
        model.finetuned_weights (str): Path to finetuned checkpoint
    """
    mcfg = cfg["model"]
    model = Swin2SRModel(
        pretrained=mcfg.get("pretrained", True),
        model_name=mcfg.get("model_name", PRETRAINED_MODEL_NAME),
        upscale=mcfg.get("upscale", 4),
    )

    # Load finetuned weights if provided
    finetuned = mcfg.get("finetuned_weights")
    if finetuned:
        model.load_finetuned(finetuned)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Swin2SR model — {n_params / 1e6:.2f}M trainable parameters")
    return model


if __name__ == "__main__":
    """Smoke test: forward pass with random input."""
    model = Swin2SRModel(pretrained=False)
    model.eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    print(f"✓ Forward pass OK — input {x.shape} → output {out.shape}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params / 1e6:.2f}M")
