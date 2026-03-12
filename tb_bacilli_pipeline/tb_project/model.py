"""
SegFormer model for TB bacilli binary segmentation.
Uses HuggingFace transformers SegformerForSemanticSegmentation with MIT backbone.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig

logger = logging.getLogger(__name__)


class TBSegFormer(nn.Module):
    """SegFormer wrapper for binary TB bacilli segmentation.

    Supports:
    - HuggingFace pretrained backbones (MIT-B0 through B5)
    - Custom num_classes (default: 2 for background + bacilli)
    - Automatic logit upsampling to input resolution
    """

    def __init__(
        self,
        backbone: str = "nvidia/mit-b4",
        num_classes: int = 2,
        pretrained: bool = True,
        image_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        if pretrained:
            # Load pretrained and replace classification head
            logger.info(f"Loading pretrained SegFormer backbone: {backbone}")
            try:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    backbone,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                )
            except (OSError, RuntimeError) as e:
                logger.warning(f"Could not load pretrained weights ({e}), "
                               f"falling back to random initialization")
                config = self._get_local_config(backbone, num_classes)
                self.model = SegformerForSemanticSegmentation(config)
        else:
            # Build from local config without network access
            config = self._get_local_config(backbone, num_classes)
            self.model = SegformerForSemanticSegmentation(config)

        logger.info(f"SegFormer built — backbone: {backbone}, num_classes: {num_classes}")

    @staticmethod
    def _get_local_config(backbone: str, num_classes: int) -> SegformerConfig:
        """Create SegformerConfig locally without requiring network access."""
        # Known SegFormer variant configurations
        variant_configs = {
            "nvidia/mit-b0": dict(
                hidden_sizes=[32, 64, 160, 256], depths=[2, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8], decoder_hidden_size=256,
            ),
            "nvidia/mit-b1": dict(
                hidden_sizes=[64, 128, 320, 512], depths=[2, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8], decoder_hidden_size=256,
            ),
            "nvidia/mit-b2": dict(
                hidden_sizes=[64, 128, 320, 512], depths=[3, 4, 6, 3],
                num_attention_heads=[1, 2, 5, 8], decoder_hidden_size=768,
            ),
            "nvidia/mit-b3": dict(
                hidden_sizes=[64, 128, 320, 512], depths=[3, 4, 18, 3],
                num_attention_heads=[1, 2, 5, 8], decoder_hidden_size=768,
            ),
            "nvidia/mit-b4": dict(
                hidden_sizes=[64, 128, 320, 512], depths=[3, 8, 27, 3],
                num_attention_heads=[1, 2, 5, 8], decoder_hidden_size=768,
            ),
            "nvidia/mit-b5": dict(
                hidden_sizes=[64, 128, 320, 512], depths=[3, 6, 40, 3],
                num_attention_heads=[1, 2, 5, 8], decoder_hidden_size=768,
            ),
        }
        if backbone in variant_configs:
            vcfg = variant_configs[backbone]
            return SegformerConfig(
                num_labels=num_classes,
                hidden_sizes=vcfg["hidden_sizes"],
                depths=vcfg["depths"],
                num_attention_heads=vcfg["num_attention_heads"],
                decoder_hidden_size=vcfg["decoder_hidden_size"],
            )
        # Fallback: try loading from pretrained (requires network)
        try:
            config = SegformerConfig.from_pretrained(backbone)
            config.num_labels = num_classes
            return config
        except Exception:
            raise ValueError(
                f"Unknown backbone '{backbone}'. Use one of: {list(variant_configs.keys())}"
            )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) normalized image tensor.
        Returns:
            logits: (B, num_classes, H, W) upsampled to input resolution.
        """
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits  # (B, num_classes, H/4, W/4)

        # Upsample to input resolution
        logits = F.interpolate(
            logits,
            size=pixel_values.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits

    def predict(self, pixel_values: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return probability map for the bacilli class.

        Returns:
            probs: (B, 1, H, W) float32 tensor with values in [0, 1].
        """
        logits = self.forward(pixel_values)
        if self.num_classes == 2:
            probs = torch.softmax(logits, dim=1)[:, 1:2]
        else:
            probs = torch.softmax(logits, dim=1).max(dim=1, keepdim=True).values
        return probs


# =============================================================================
# Loss functions
# =============================================================================

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Accepts single-channel logits (B, 1, H, W) and targets (B, 1, H, W).
    Applies sigmoid to logits internally.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.float().view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Combined Dice + weighted BCE loss for binary segmentation.

    Accepts single-channel logits (B, 1, H, W) and targets (B, 1, H, W).
    """

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 0.5,
                 pos_weight: float = 5.0, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        bce = self.bce_loss(logits, targets.float())
        return self.dice_weight * dice + self.bce_weight * bce


class FocalLoss(nn.Module):
    """Binary Focal loss for handling class imbalance.

    Accepts single-channel logits (B, 1, H, W) and targets (B, 1, H, W).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


def build_criterion(cfg: dict) -> nn.Module:
    """Build loss function from config."""
    loss_cfg = cfg["training"]["loss"]
    loss_type = loss_cfg["type"]

    if loss_type == "dice_bce":
        criterion = DiceBCELoss(
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            bce_weight=loss_cfg.get("bce_weight", 0.5),
            pos_weight=loss_cfg.get("pos_weight", 5.0),
            smooth=loss_cfg.get("smooth", 1.0),
        )
    elif loss_type == "dice":
        criterion = DiceLoss(smooth=loss_cfg.get("smooth", 1.0))
    elif loss_type == "focal":
        criterion = FocalLoss()
    elif loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([loss_cfg.get("pos_weight", 5.0)])
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    logger.info(f"Loss function: {loss_type}")
    return criterion


def build_model(cfg: dict) -> TBSegFormer:
    """Build SegFormer model from config."""
    mcfg = cfg["model"]
    model = TBSegFormer(
        backbone=mcfg["backbone"],
        num_classes=mcfg.get("num_classes", 2),
        pretrained=mcfg.get("pretrained", True),
        image_size=cfg.get("data", {}).get("image_size"),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"SegFormer model — {n_params / 1e6:.2f}M trainable parameters")
    return model


if __name__ == "__main__":
    """Smoke test: forward pass."""
    model = TBSegFormer(backbone="nvidia/mit-b0", num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    logits = model(x)
    assert logits.shape == (2, 2, 512, 512), f"Shape mismatch: {logits.shape}"
    mask = model.predict(x)
    assert mask.shape == (2, 1, 512, 512), f"Mask shape: {mask.shape}"
    assert mask.dtype == torch.float32, f"Mask dtype: {mask.dtype}"
    print(f"✓ Forward pass OK — input {x.shape} → logits {logits.shape} → mask {mask.shape}")
