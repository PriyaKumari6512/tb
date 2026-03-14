"""
SegFormer model for TB bacilli binary segmentation.
Uses HuggingFace transformers SegformerForSemanticSegmentation with MIT backbone.
"""

import logging
from typing import Optional

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
        """Return probability map for positive (bacilli) class.

        Returns:
            probs: (B, 1, H, W) float32 in [0, 1]
        """
        logits = self.forward(pixel_values)
        if self.num_classes == 2:
            probs = torch.softmax(logits, dim=1)[:, 1:2]  # (B, 1, H, W)
        else:
            probs = torch.sigmoid(logits[:, 1:2])  # (B, 1, H, W)
        return probs.float()


# =============================================================================
# Loss functions
# =============================================================================

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation with sigmoid inputs.

    Accepts logits of shape (B, 1, H, W) and float binary targets (B, 1, H, W).
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_f = probs.view(probs.shape[0], -1)
        targets_f = targets.float().view(targets.shape[0], -1)

        intersection = (probs_f * targets_f).sum(dim=1)
        union = probs_f.sum(dim=1) + targets_f.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Combined Dice + weighted BCE loss for binary sigmoid inputs."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 0.5,
                 pos_weight: float = 5.0, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth)
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=self.pos_weight
        )
        return self.dice_weight * dice + self.bce_weight * bce


class FocalLoss(nn.Module):
    """Focal loss for class imbalance — binary sigmoid variant.

    Accepts logits (B, 1, H, W) and float binary targets (B, 1, H, W).
    alpha < 0.5 down-weights the positive class to suppress false positives.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_f = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets_f, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets_f + (1 - self.alpha) * (1 - targets_f)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


class FocalDiceLoss(nn.Module):
    """Focal Loss + Soft Dice combination for maximum false-positive reduction."""

    def __init__(self, focal_weight: float = 1.0, dice_weight: float = 1.0,
                 alpha: float = 0.25, gamma: float = 2.0, smooth: float = 1.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self._focal = FocalLoss(alpha, gamma)
        self._dice = DiceLoss(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.focal_weight * self._focal(logits, targets)
                + self.dice_weight * self._dice(logits, targets))


def build_criterion(cfg: dict) -> nn.Module:
    """Build loss function from config."""
    loss_cfg = cfg["training"]["loss"]
    loss_type = loss_cfg["type"]

    if loss_type == "focal_dice":
        criterion = FocalDiceLoss(
            focal_weight=loss_cfg.get("focal_weight", 1.0),
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            alpha=loss_cfg.get("alpha", 0.25),
            gamma=loss_cfg.get("gamma", 2.0),
            smooth=loss_cfg.get("smooth", 1.0),
        )
    elif loss_type == "dice_bce":
        criterion = DiceBCELoss(
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            bce_weight=loss_cfg.get("bce_weight", 0.5),
            pos_weight=loss_cfg.get("pos_weight", 5.0),
            smooth=loss_cfg.get("smooth", 1.0),
        )
    elif loss_type == "dice":
        criterion = DiceLoss(smooth=loss_cfg.get("smooth", 1.0))
    elif loss_type == "focal":
        criterion = FocalLoss(
            alpha=loss_cfg.get("alpha", 0.25),
            gamma=loss_cfg.get("gamma", 2.0),
        )
    elif loss_type == "bce":
        pos_w = loss_cfg.get("pos_weight", 5.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]))
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
        image_size=mcfg.get("image_size", None),
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
    probs = model.predict(x)
    assert probs.shape == (2, 1, 512, 512), f"Mask shape: {probs.shape}"
    print(f"✓ Forward pass OK — input {x.shape} → logits {logits.shape} → probs {probs.shape}")
