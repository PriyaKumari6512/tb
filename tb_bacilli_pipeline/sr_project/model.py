"""
SwinIR: Image Restoration Using Swin Transformer.

Architecture:
  Input → Shallow Feature Extraction (Conv)
       → Deep Feature Extraction (N × RSTB)
       → Reconstruction (Conv + PixelShuffle)
       → Output

Each RSTB (Residual Swin Transformer Block):
  Input → M × STL (Swin Transformer Layer) → Conv → + Input

STL uses window-based multi-head self-attention with shifted windows.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# =============================================================================
# Core attention components
# =============================================================================

class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


# =============================================================================
# MLP
# =============================================================================

class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# =============================================================================
# Swin Transformer Layer
# =============================================================================

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition into non-overlapping windows. B, H, W, C → (B*nW), Wh*Ww, C."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition. (B*nW), Wh*Ww, C → B, H, W, C."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerLayer(nn.Module):
    """A single Swin Transformer layer with (shifted) window attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor, x_size: tuple) -> torch.Tensor:
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, f"Input size mismatch: {L} != {H}*{W}"

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Pad to multiples of window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._compute_mask(Hp, Wp, x.device)
        else:
            shifted_x = x
            attn_mask = None

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Reverse
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

    def _compute_mask(self, Hp: int, Wp: int, device: torch.device) -> torch.Tensor:
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h_s in h_slices:
            for w_s in w_slices:
                img_mask[:, h_s, w_s, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.squeeze(-1)      # nW, Wh*Ww
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        return attn_mask


# =============================================================================
# Residual Swin Transformer Block (RSTB)
# =============================================================================

class RSTB(nn.Module):
    """Residual Swin Transformer Block: N × STL + Conv + residual."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        resi_connection: str = "1conv",
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            SwinTransformerLayer(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)
        ])

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

    def forward(self, x: torch.Tensor, x_size: tuple) -> torch.Tensor:
        res = x
        for layer in self.layers:
            x = layer(x, x_size)
        # Reshape to 2D for conv
        B, L, C = x.shape
        H, W = x_size
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv(x)
        x = x.view(B, C, H * W).transpose(1, 2) + res
        return x


# =============================================================================
# Upsampling
# =============================================================================

class PixelShuffleUpsampler(nn.Module):
    """PixelShuffle-based upsampling for 4× scale."""

    def __init__(self, dim: int, upscale: int = 4):
        super().__init__()
        layers = []
        if upscale == 4:
            layers.extend([
                nn.Conv2d(dim, dim * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim, dim * 4, 3, 1, 1),
                nn.PixelShuffle(2),
            ])
        elif upscale == 2:
            layers.extend([
                nn.Conv2d(dim, dim * 4, 3, 1, 1),
                nn.PixelShuffle(2),
            ])
        elif upscale == 3:
            layers.extend([
                nn.Conv2d(dim, dim * 9, 3, 1, 1),
                nn.PixelShuffle(3),
            ])
        self.up = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


# =============================================================================
# SwinIR Model
# =============================================================================

class SwinIR(nn.Module):
    """
    SwinIR: Image Restoration Using Swin Transformer.

    Args:
        img_channels: Number of input/output image channels (3 for RGB).
        embed_dim: Embedding dimension.
        depths: Number of STL layers in each RSTB.
        num_heads: Number of attention heads in each RSTB.
        window_size: Window size for attention.
        mlp_ratio: MLP hidden dim ratio.
        upscale: Upscaling factor.
        resi_connection: Residual connection type ("1conv" or "3conv").
    """

    def __init__(
        self,
        img_channels: int = 3,
        embed_dim: int = 180,
        depths: List[int] = [6, 6, 6, 6, 6, 6, 6, 6],
        num_heads: List[int] = [6, 6, 6, 6, 6, 6, 6, 6],
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        upscale: int = 4,
        resi_connection: str = "1conv",
    ):
        super().__init__()
        self.window_size = window_size
        self.upscale = upscale

        # ---------- Shallow feature extraction ----------
        self.conv_first = nn.Conv2d(img_channels, embed_dim, 3, 1, 1)

        # ---------- Deep feature extraction ----------
        self.rstb_blocks = nn.ModuleList([
            RSTB(
                dim=embed_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                resi_connection=resi_connection,
            )
            for i in range(len(depths))
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # ---------- Reconstruction ----------
        self.upsample = PixelShuffleUpsampler(embed_dim, upscale)
        self.conv_last = nn.Conv2d(embed_dim, img_channels, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Pad input to window_size multiples
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, Hp, Wp = x.shape

        # Shallow features
        shallow = self.conv_first(x)

        # Deep features
        x_deep = shallow.flatten(2).transpose(1, 2)  # B, H*W, C
        x_size = (Hp, Wp)
        for rstb in self.rstb_blocks:
            x_deep = rstb(x_deep, x_size)
        x_deep = self.norm(x_deep)
        x_deep = x_deep.transpose(1, 2).view(B, -1, Hp, Wp)
        x_deep = self.conv_after_body(x_deep) + shallow

        # Reconstruction
        out = self.conv_last(self.upsample(x_deep))

        # Remove padding (scaled)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, : H * self.upscale, : W * self.upscale]

        return out


def build_model(cfg: dict) -> SwinIR:
    """Build SwinIR model from config."""
    mcfg = cfg["model"]
    model = SwinIR(
        img_channels=mcfg.get("img_channels", 3),
        embed_dim=mcfg["embed_dim"],
        depths=mcfg["depths"],
        num_heads=mcfg["num_heads"],
        window_size=mcfg.get("window_size", 8),
        mlp_ratio=mcfg.get("mlp_ratio", 2.0),
        upscale=mcfg.get("upscale", 4),
        resi_connection=mcfg.get("resi_connection", "1conv"),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SwinIR model built — {n_params / 1e6:.2f}M trainable parameters")
    return model


if __name__ == "__main__":
    """Smoke test: forward pass with random input."""
    model = SwinIR(
        embed_dim=60, depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6], window_size=8, upscale=4,
    )
    x = torch.randn(1, 3, 16, 16)
    out = model(x)
    assert out.shape == (1, 3, 64, 64), f"Shape mismatch: {out.shape}"
    print(f"✓ Forward pass OK — input {x.shape} → output {out.shape}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params / 1e6:.2f}M")
