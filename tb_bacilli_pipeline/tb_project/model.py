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
    ):
        super().__init__()
        self.num_classes = num_classes

        if pretrained:
            # Load pretrained and replace classification head
            logger.info(f"Loading pretrained SegFormer backbone: {backbone}")
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                backbone,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig.from_pretrained(backbone)
            config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)

        logger.info(f"SegFormer built — backbone: {backbone}, num_classes: {num_classes}")

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
        """Return binary mask prediction."""
        logits = self.forward(pixel_values)
        if self.num_classes == 2:
            probs = torch.softmax(logits, dim=1)
            mask = (probs[:, 1] > threshold).long()
        else:
            mask = logits.argmax(dim=1)
        return mask


# =============================================================================
# Loss functions
# =============================================================================

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1]  # Bacilli class probability
        targets_f = targets.float()

        intersection = (probs * targets_f).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets_f.sum(dim=(1, 2))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Combined Dice + weighted BCE loss."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 0.5,
                 pos_weight: float = 5.0, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth)
        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_weight]),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        return self.dice_weight * dice + self.bce_weight * ce


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
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
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, loss_cfg.get("pos_weight", 5.0)])
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
    assert mask.shape == (2, 512, 512), f"Mask shape: {mask.shape}"
    print(f"✓ Forward pass OK — input {x.shape} → logits {logits.shape} → mask {mask.shape}")
