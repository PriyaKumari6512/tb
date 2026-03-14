"""
SegFormer B4 — Full from-scratch implementation.

Architecture overview:
  ┌──────────────────────────────────────────────────┐
  │  Input Image (B, 3, H, W)                        │
  │       ↓                                          │
  │  ┌─ Stage 1: Overlap Patch Embed (stride=4)      │
  │  │   → Mix-TF Blocks × 3  → C1 (H/4, W/4, 64)  │
  │  ├─ Stage 2: Overlap Patch Embed (stride=2)      │
  │  │   → Mix-TF Blocks × 8  → C2 (H/8, W/8, 128) │
  │  ├─ Stage 3: Overlap Patch Embed (stride=2)      │
  │  │   → Mix-TF Blocks × 27 → C3 (H/16,W/16,320) │
  │  ├─ Stage 4: Overlap Patch Embed (stride=2)      │
  │  │   → Mix-TF Blocks × 3  → C4 (H/32,W/32,512) │
  │  └──────────────────────────────────────────────  │
  │       ↓                                          │
  │  All-MLP Decoder Head                             │
  │   → Linear unify each Ci to 768                   │
  │   → Upsample all to H/4 × W/4                    │
  │   → Concat → Linear fuse → Linear classify       │
  │       ↓                                          │
  │  Logits (B, num_classes, H/4, W/4)               │
  └──────────────────────────────────────────────────┘

Mix Transformer Block:
  LayerNorm → Efficient Self-Attention (with spatial reduction)
  → LayerNorm → Mix-FFN (Linear → DWConv3×3 → GELU → Linear)

B4 config:
  embed_dims  = [64, 128, 320, 512]
  depths      = [3, 8, 27, 3]
  num_heads   = [1, 2, 5, 8]
  sr_ratios   = [8, 4, 2, 1]  (spatial reduction ratios for efficient attention)
  mlp_ratios  = [4, 4, 4, 4]
  decoder_dim = 768
"""

import math
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Helper: Drop Path (Stochastic Depth)
# =============================================================================

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# =============================================================================
# Overlapping Patch Embedding
# =============================================================================

class OverlapPatchEmbed(nn.Module):
    """
    Overlapping patch embedding using a convolution with kernel > stride.
    Stage 1: patch_size=7, stride=4  →  H/4, W/4
    Stage 2-4: patch_size=3, stride=2  →  H/2, W/2 relative to input
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 64,
                 patch_size: int = 7, stride: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)               # (B, C, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', C)
        x = self.norm(x)
        return x, H, W


# =============================================================================
# Efficient Self-Attention (with Spatial Reduction)
# =============================================================================

class EfficientSelfAttention(nn.Module):
    """
    Multi-head self-attention with optional spatial reduction.
    When sr_ratio > 1, the K and V are spatially downsampled before attention,
    reducing complexity from O(N²) to O(N · N/R²).
    """

    def __init__(self, dim: int, num_heads: int = 1, sr_ratio: int = 1,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Spatial reduction for K, V
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape

        # Query
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Spatial reduction on K, V
        if self.sr_ratio > 1:
            x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_reduced = self.sr(x_2d).reshape(B, C, -1).permute(0, 2, 1)
            x_reduced = self.sr_norm(x_reduced)
            kv = self.kv(x_reduced).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv.unbind(0)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# =============================================================================
# Mix-FFN (Feed-Forward with Depth-Wise Convolution)
# =============================================================================

class MixFFN(nn.Module):
    """
    Mix Feed-Forward Network.
    Unlike standard FFN, it includes a 3×3 depth-wise convolution
    between the two linear layers, which encodes positional information
    implicitly (no need for positional embedding).
    """

    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        # 3×3 depth-wise conv for implicit positional encoding
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features,
            kernel_size=3, stride=1, padding=1, groups=hidden_features,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# =============================================================================
# Mix Transformer Block
# =============================================================================

class MixTransformerBlock(nn.Module):
    """
    One transformer block in a MiT stage.
    LayerNorm → Efficient Self-Attention → DropPath
    → LayerNorm → Mix-FFN → DropPath
    """

    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(
            dim, num_heads=num_heads, sr_ratio=sr_ratio,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


# =============================================================================
# Mix Transformer Encoder (MiT)
# =============================================================================

class MixTransformerEncoder(nn.Module):
    """
    Hierarchical Mix Transformer encoder with 4 stages.
    Each stage outputs feature maps at a different scale:
      Stage 1: H/4 × W/4
      Stage 2: H/8 × W/8
      Stage 3: H/16 × W/16
      Stage 4: H/32 × W/32
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: List[int] = [64, 128, 320, 512],
        depths: List[int] = [3, 8, 27, 3],
        num_heads: List[int] = [1, 2, 5, 8],
        sr_ratios: List[int] = [8, 4, 2, 1],
        mlp_ratios: List[int] = [4, 4, 4, 4],
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.num_stages = 4
        self.embed_dims = embed_dims

        # Stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        cur = 0
        for i in range(self.num_stages):
            # Patch embedding
            if i == 0:
                patch_embed = OverlapPatchEmbed(
                    in_channels=in_channels, embed_dim=embed_dims[i],
                    patch_size=7, stride=4,
                )
            else:
                patch_embed = OverlapPatchEmbed(
                    in_channels=embed_dims[i - 1], embed_dim=embed_dims[i],
                    patch_size=3, stride=2,
                )

            # Transformer blocks for this stage
            blocks = nn.ModuleList([
                MixTransformerBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    sr_ratio=sr_ratios[i],
                    mlp_ratio=mlp_ratios[i],
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j],
                )
                for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", blocks)
            setattr(self, f"norm{i + 1}", norm)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns multi-scale feature maps [C1, C2, C3, C4]
        where Ci has shape (B, embed_dims[i], H/(4*2^i), W/(4*2^i)).
        Actually: C1=(B,64,H/4,W/4), C2=(B,128,H/8,W/8), etc.
        """
        outputs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            blocks = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x)
            for blk in blocks:
                x = blk(x, H, W)
            x = norm(x)

            # Reshape to 2D feature map for next stage
            B, _, C = x.shape
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            outputs.append(x)

        return outputs


# =============================================================================
# All-MLP Decoder Head
# =============================================================================

class SegFormerDecoderHead(nn.Module):
    """
    Lightweight All-MLP decoder head.
    1. Linear project each stage output to a common channel dim
    2. Upsample all to stage-1 resolution (H/4 × W/4)
    3. Concatenate
    4. Fuse with a linear layer
    5. Classify with a linear layer
    """

    def __init__(
        self,
        encoder_dims: List[int] = [64, 128, 320, 512],
        decoder_dim: int = 768,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_stages = len(encoder_dims)

        # Linear projection for each encoder stage
        self.linear_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dims[i], decoder_dim),
            )
            for i in range(self.num_stages)
        ])

        # Fuse concatenated features
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * self.num_stages, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of [C1, C2, C3, C4] from encoder.
        Returns:
            logits: (B, num_classes, H/4, W/4)
        """
        # Target resolution = stage 1 resolution
        target_H, target_W = features[0].shape[2:]

        projected = []
        for i, feat in enumerate(features):
            B, C, H, W = feat.shape
            # Linear projection: reshape to (B, H*W, C) → Linear → back to (B, decoder_dim, H, W)
            x = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
            x = self.linear_layers[i](x)          # (B, H*W, decoder_dim)
            x = x.transpose(1, 2).reshape(B, -1, H, W)  # (B, decoder_dim, H, W)

            # Upsample to target resolution
            if H != target_H or W != target_W:
                x = F.interpolate(x, size=(target_H, target_W),
                                  mode="bilinear", align_corners=False)
            projected.append(x)

        # Concatenate and fuse
        x = torch.cat(projected, dim=1)   # (B, decoder_dim * 4, H/4, W/4)
        x = self.linear_fuse(x)           # (B, decoder_dim, H/4, W/4)
        x = self.dropout(x)
        logits = self.classifier(x)       # (B, num_classes, H/4, W/4)

        return logits


# =============================================================================
# SegFormer B4 Full Model
# =============================================================================

# Predefined configs for all SegFormer variants
SEGFORMER_CONFIGS = {
    "b0": dict(embed_dims=[32, 64, 160, 256],   depths=[2, 2, 2, 2],   num_heads=[1, 2, 5, 8], decoder_dim=256),
    "b1": dict(embed_dims=[64, 128, 320, 512],  depths=[2, 2, 2, 2],   num_heads=[1, 2, 5, 8], decoder_dim=256),
    "b2": dict(embed_dims=[64, 128, 320, 512],  depths=[3, 4, 6, 3],   num_heads=[1, 2, 5, 8], decoder_dim=768),
    "b3": dict(embed_dims=[64, 128, 320, 512],  depths=[3, 4, 18, 3],  num_heads=[1, 2, 5, 8], decoder_dim=768),
    "b4": dict(embed_dims=[64, 128, 320, 512],  depths=[3, 8, 27, 3],  num_heads=[1, 2, 5, 8], decoder_dim=768),
    "b5": dict(embed_dims=[64, 128, 320, 512],  depths=[3, 6, 40, 3],  num_heads=[1, 2, 5, 8], decoder_dim=768),
}


class SegFormerB4(nn.Module):
    """
    SegFormer B4: Hierarchical Mix Transformer encoder + All-MLP decoder.

    From-scratch implementation — no HuggingFace dependency.

    Args:
        in_channels: Input image channels (3 for RGB).
        num_classes: Number of segmentation classes.
        variant: SegFormer variant ("b0" through "b5"). Default "b4".
        drop_rate: Dropout rate in transformer blocks.
        drop_path_rate: Stochastic depth rate.
        pretrained_weights: Optional path to pretrained encoder weights.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        variant: str = "b4",
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        pretrained_weights: Optional[str] = None,
    ):
        super().__init__()
        assert variant in SEGFORMER_CONFIGS, f"Unknown variant: {variant}. Choose from {list(SEGFORMER_CONFIGS.keys())}"
        cfg = SEGFORMER_CONFIGS[variant]

        self.num_classes = num_classes
        self.variant = variant

        # Encoder
        self.encoder = MixTransformerEncoder(
            in_channels=in_channels,
            embed_dims=cfg["embed_dims"],
            depths=cfg["depths"],
            num_heads=cfg["num_heads"],
            sr_ratios=[8, 4, 2, 1],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Decoder
        self.decoder = SegFormerDecoderHead(
            encoder_dims=cfg["embed_dims"],
            decoder_dim=cfg["decoder_dim"],
            num_classes=num_classes,
        )

        # Load pretrained encoder weights if provided
        if pretrained_weights:
            self._load_pretrained(pretrained_weights)

    def _load_pretrained(self, path: str):
        """Load pretrained encoder weights (e.g., ImageNet-1K trained MiT)."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]

        # Filter to encoder keys only and strip prefix if needed
        encoder_state = {}
        for k, v in state.items():
            if k.startswith("encoder."):
                encoder_state[k[len("encoder."):]] = v
            elif not k.startswith("decoder.") and not k.startswith("classifier"):
                encoder_state[k] = v

        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        print(f"Loaded pretrained encoder: {len(state) - len(unexpected)} keys loaded, "
              f"{len(missing)} missing, {len(unexpected)} unexpected")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input images.
        Returns:
            logits: (B, num_classes, H/4, W/4) — NOT upsampled to full resolution.
                    Call upsample_logits() if you need full-resolution output.
        """
        features = self.encoder(x)  # [C1, C2, C3, C4]
        logits = self.decoder(features)  # (B, num_classes, H/4, W/4)
        return logits

    def forward_with_upsample(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with output upsampled to input resolution."""
        logits = self.forward(x)
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary mask at input resolution."""
        logits = self.forward_with_upsample(x)
        if self.num_classes == 2:
            probs = torch.softmax(logits, dim=1)
            return (probs[:, 1] > threshold).long()
        return logits.argmax(dim=1)

    def get_param_groups(self, lr: float, lr_mult_encoder: float = 0.1):
        """
        Differential LR: encoder parameters at lr * lr_mult_encoder,
        decoder parameters at full lr.
        """
        encoder_params = list(self.encoder.parameters())
        decoder_params = list(self.decoder.parameters())
        return [
            {"params": encoder_params, "lr": lr * lr_mult_encoder},
            {"params": decoder_params, "lr": lr},
        ]


# =============================================================================
# Loss Functions
# =============================================================================

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1]
        targets_f = targets.float()
        intersection = (probs * targets_f).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets_f.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Combined Dice + weighted Cross-Entropy loss."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 0.5,
                 pos_weight: float = 5.0, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth)
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        return self.dice_weight * dice + self.bce_weight * ce


class FocalLoss(nn.Module):
    """Focal loss for class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


class FocalDiceLoss(nn.Module):
    """Focal Loss + Soft Dice Loss combination for maximum false-positive reduction.

    Focal loss focuses on hard negatives and suppresses easy-negative contribution,
    while Dice loss optimises the overlap directly — together they tackle both
    class imbalance and boundary precision.
    """

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


# =============================================================================
# Builder utilities
# =============================================================================

def build_segformer(
    variant: str = "b4",
    num_classes: int = 2,
    pretrained_weights: Optional[str] = None,
    drop_path_rate: float = 0.1,
) -> SegFormerB4:
    """Build a SegFormer model from variant string."""
    model = SegFormerB4(
        num_classes=num_classes,
        variant=variant,
        drop_path_rate=drop_path_rate,
        pretrained_weights=pretrained_weights,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SegFormer-{variant.upper()} built — {n_params / 1e6:.2f}M trainable parameters")
    return model


def build_criterion(loss_type: str = "dice_bce", pos_weight: float = 5.0) -> nn.Module:
    if loss_type == "dice_bce":
        return DiceBCELoss(pos_weight=pos_weight)
    elif loss_type == "dice":
        return DiceLoss()
    elif loss_type == "focal":
        return FocalLoss()
    elif loss_type == "focal_dice":
        return FocalDiceLoss()
    elif loss_type == "ce":
        return nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]))
    raise ValueError(f"Unknown loss: {loss_type}")


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SegFormer B4 — From-scratch smoke test")
    print("=" * 60)

    for variant in ["b0", "b2", "b4"]:
        model = build_segformer(variant=variant, num_classes=2)
        x = torch.randn(2, 3, 512, 512)
        logits = model(x)
        logits_up = model.forward_with_upsample(x)
        mask = model.predict(x)

        cfg = SEGFORMER_CONFIGS[variant]
        expected_h = 512 // 4
        assert logits.shape == (2, 2, expected_h, expected_h), f"B{variant} logits: {logits.shape}"
        assert logits_up.shape == (2, 2, 512, 512), f"B{variant} upsampled: {logits_up.shape}"
        assert mask.shape == (2, 512, 512), f"B{variant} mask: {mask.shape}"

        enc_params = sum(p.numel() for p in model.encoder.parameters())
        dec_params = sum(p.numel() for p in model.decoder.parameters())
        print(f"  SegFormer-{variant.upper()}: encoder={enc_params/1e6:.2f}M, "
              f"decoder={dec_params/1e6:.2f}M, "
              f"logits={logits.shape} ✓")

    print("\n✓ All smoke tests passed!")
