"""
SegFormer model for TB bacilli binary segmentation.
Uses HuggingFace transformers SegformerForSemanticSegmentation with MIT backbone.
"""

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard-coded backbone configs for offline use (no internet required when
# pretrained=False).  Values are taken from the official HuggingFace model
# cards for the nvidia/mit-b* family.
# ---------------------------------------------------------------------------
_BACKBONE_OFFLINE_CONFIGS = {
    "nvidia/mit-b0": dict(
        hidden_sizes=[32, 64, 160, 256], depths=[2, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
    ),
    "nvidia/mit-b1": dict(
        hidden_sizes=[64, 128, 320, 512], depths=[2, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
    ),
    "nvidia/mit-b2": dict(
        hidden_sizes=[64, 128, 320, 512], depths=[3, 4, 6, 3],
        num_attention_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
    ),
    "nvidia/mit-b3": dict(
        hidden_sizes=[64, 128, 320, 512], depths=[3, 4, 18, 3],
        num_attention_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
    ),
    "nvidia/mit-b4": dict(
        hidden_sizes=[64, 128, 320, 512], depths=[3, 8, 27, 3],
        num_attention_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
    ),
    "nvidia/mit-b5": dict(
        hidden_sizes=[64, 128, 320, 512], depths=[3, 6, 40, 3],
        num_attention_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
    ),
}


class TBSegFormer(nn.Module):
    """SegFormer wrapper for binary TB bacilli segmentation.

    Supports:
    - HuggingFace pretrained backbones (MIT-B0 through B5)
    - Custom num_classes (default: 2 for background + bacilli)
    - Automatic logit upsampling to input resolution
    - Offline initialization via ``pretrained=False`` (no internet required)
    - Loading local pretrained weights via ``local_pretrained_path``
    """

    def __init__(
        self,
        backbone: str = "nvidia/mit-b4",
        num_classes: int = 2,
        pretrained: bool = True,
        image_size: Optional[int] = None,
        local_pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        if pretrained and local_pretrained_path is None:
            # Load pretrained backbone from HuggingFace (requires internet)
            logger.info(f"Loading pretrained SegFormer backbone: {backbone}")
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                backbone,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            # Offline construction — use hard-coded configs for known backbones
            if backbone in _BACKBONE_OFFLINE_CONFIGS:
                bcfg = _BACKBONE_OFFLINE_CONFIGS[backbone]
                config = SegformerConfig(
                    num_encoder_blocks=4,
                    hidden_sizes=bcfg["hidden_sizes"],
                    depths=bcfg["depths"],
                    num_attention_heads=bcfg["num_attention_heads"],
                    mlp_ratios=bcfg["mlp_ratios"],
                    num_labels=num_classes,
                )
            else:
                # Unknown backbone — attempt to load config from HuggingFace or
                # local directory; fall back gracefully.
                logger.warning(
                    f"Unknown backbone '{backbone}' for offline init — "
                    "attempting SegformerConfig.from_pretrained()"
                )
                config = SegformerConfig.from_pretrained(backbone)
                config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)

        # Optionally load weights from a local checkpoint
        if local_pretrained_path is not None:
            self._load_local_weights(local_pretrained_path)

        logger.info(f"SegFormer built — backbone: {backbone}, num_classes: {num_classes}")

    def _load_local_weights(self, path: str) -> None:
        """Load model weights from a local checkpoint file.

        Accepts checkpoints saved by :func:`save_checkpoint` (which wrap the
        state dict under ``"model_state_dict"``) as well as plain state dicts.

        .. warning::
            Only load checkpoints from trusted sources.  ``torch.load`` with
            ``weights_only=False`` uses Python ``pickle`` internally and can
            execute arbitrary code if the file is malicious.

        Args:
            path: Path to the ``.pth`` checkpoint file.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Local pretrained checkpoint not found: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Unwrap common wrapper keys
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(ckpt, dict) and key in ckpt:
                ckpt = ckpt[key]
                break

        # Strip DataParallel prefix
        if any(k.startswith("module.") for k in ckpt):
            ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

        missing, unexpected = self.load_state_dict(ckpt, strict=False)
        loaded = len(ckpt) - len(unexpected)
        logger.info(
            f"Loaded local pretrained weights from '{path}' — "
            f"{loaded}/{len(ckpt)} keys loaded, "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
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
    """Build SegFormer model from config.

    Config keys (under ``cfg["model"]``):
      - ``backbone``: HuggingFace model ID (e.g. ``"nvidia/mit-b4"``)
      - ``num_classes``: number of output classes (default 2)
      - ``pretrained``: load pretrained HuggingFace backbone (default True)
      - ``image_size``: expected input size (optional)
      - ``local_pretrained_path``: path to a local ``.pth`` checkpoint to load
        instead of downloading from HuggingFace (optional)
    """
    mcfg = cfg["model"]
    model = TBSegFormer(
        backbone=mcfg["backbone"],
        num_classes=mcfg.get("num_classes", 2),
        pretrained=mcfg.get("pretrained", True),
        image_size=mcfg.get("image_size", None),
        local_pretrained_path=mcfg.get("local_pretrained_path", None),
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
