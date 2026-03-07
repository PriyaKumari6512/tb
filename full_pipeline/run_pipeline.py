#!/usr/bin/env python3
"""
=============================================================================
  TB Bacilli Full Pipeline — Single-file Runner
=============================================================================

  Run the ENTIRE pipeline with one command:

      python run_pipeline.py
      python run_pipeline.py --config pipeline_config.yaml
      python run_pipeline.py --sr_checkpoint /path/to/sr.pth --seg_checkpoint /path/to/seg.pth
      python run_pipeline.py --skip_sr   # skip SR, run segmentation on original images

  Pipeline steps:
    1. Discover images in testingDataset/{Positive, Negative}
    2. Super-resolve each image with SwinIR (4× upscale, tiled for large images)
    3. Save SR images to testingDataset_SR/{Positive, Negative}
    4. Segment SR images with SegFormer B4 (from-scratch, not HuggingFace)
    5. Detect + count TB bacilli via connected-component analysis
    6. Save everything as JSON (per-image results + global summary)
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from skimage import measure
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 1: SwinIR Super-Resolution Model (self-contained)                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class SwinTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=2.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            mask = self._compute_mask(Hp, Wp, x.device)
        else:
            shifted_x = x
            mask = None
        x_win = window_partition(shifted_x, self.window_size)
        attn_win = self.attn(x_win, mask=mask)
        shifted_x = window_reverse(attn_win, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

    def _compute_mask(self, Hp, Wp, device):
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mw = window_partition(img_mask, self.window_size).squeeze(-1)
        return (mw.unsqueeze(1) - mw.unsqueeze(2)).masked_fill_(
            (mw.unsqueeze(1) - mw.unsqueeze(2)) != 0, -100.0)


class RSTB(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=8, mlp_ratio=2.0):
        super().__init__()
        self.layers = nn.ModuleList([
            SwinTransformerLayer(dim, num_heads, window_size,
                                shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio)
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):
        res = x
        for layer in self.layers:
            x = layer(x, x_size)
        B, L, C = x.shape
        H, W = x_size
        x = self.conv(x.transpose(1, 2).view(B, C, H, W))
        return x.view(B, C, H * W).transpose(1, 2) + res


class SwinIR(nn.Module):
    def __init__(self, img_channels=3, embed_dim=180, depths=(6,)*8,
                 num_heads=(6,)*8, window_size=8, mlp_ratio=2.0, upscale=4):
        super().__init__()
        self.window_size = window_size
        self.upscale = upscale
        self.conv_first = nn.Conv2d(img_channels, embed_dim, 3, 1, 1)
        self.rstb_blocks = nn.ModuleList([
            RSTB(embed_dim, depths[i], num_heads[i], window_size, mlp_ratio)
            for i in range(len(depths))
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        # PixelShuffle 4×
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1), nn.PixelShuffle(2),
        )
        self.conv_last = nn.Conv2d(embed_dim, img_channels, 3, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, Hp, Wp = x.shape
        shallow = self.conv_first(x)
        deep = shallow.flatten(2).transpose(1, 2)
        for rstb in self.rstb_blocks:
            deep = rstb(deep, (Hp, Wp))
        deep = self.norm(deep).transpose(1, 2).view(B, -1, Hp, Wp)
        deep = self.conv_after_body(deep) + shallow
        out = self.conv_last(self.upsample(deep))
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H * self.upscale, :W * self.upscale]
        return out


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 2: SegFormer B4 Segmentation Model (self-contained)              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    return x * x.new_empty(shape).bernoulli_(keep).div_(keep)


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p
    def forward(self, x):
        return _drop_path(x, self.p, self.training)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride, patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=1, sr_ratio=1, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x2d = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_r = self.sr(x2d).reshape(B, C, -1).permute(0, 2, 1)
            x_r = self.sr_norm(x_r)
            kv = self.kv(x_r).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MixFFN(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        x = self.dwconv(x.transpose(1, 2).view(B, C, H, W)).flatten(2).transpose(1, 2)
        x = self.act(x)
        return self.fc2(x)


class MixTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio))

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


SEGFORMER_B4_CONFIG = dict(
    embed_dims=[64, 128, 320, 512],
    depths=[3, 8, 27, 3],
    num_heads=[1, 2, 5, 8],
    sr_ratios=[8, 4, 2, 1],
    mlp_ratios=[4, 4, 4, 4],
    decoder_dim=768,
)


class SegFormerB4(nn.Module):
    """SegFormer B4 — full from-scratch implementation."""

    def __init__(self, num_classes=2, drop_path_rate=0.1):
        super().__init__()
        cfg = SEGFORMER_B4_CONFIG
        self.num_classes = num_classes
        total_depth = sum(cfg["depths"])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        cur = 0
        for i in range(4):
            in_ch = 3 if i == 0 else cfg["embed_dims"][i - 1]
            ps = 7 if i == 0 else 3
            st = 4 if i == 0 else 2
            setattr(self, f"patch_embed{i+1}",
                    OverlapPatchEmbed(in_ch, cfg["embed_dims"][i], ps, st))
            setattr(self, f"block{i+1}", nn.ModuleList([
                MixTransformerBlock(cfg["embed_dims"][i], cfg["num_heads"][i],
                                    cfg["sr_ratios"][i], cfg["mlp_ratios"][i], dpr[cur + j])
                for j in range(cfg["depths"][i])
            ]))
            setattr(self, f"norm{i+1}", nn.LayerNorm(cfg["embed_dims"][i]))
            cur += cfg["depths"][i]

        # Decoder
        d = cfg["decoder_dim"]
        self.linear_layers = nn.ModuleList([nn.Linear(cfg["embed_dims"][i], d) for i in range(4)])
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(d * 4, d, 1, bias=False), nn.BatchNorm2d(d), nn.ReLU(True))
        self.classifier = nn.Conv2d(d, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None: m.bias.data.zero_()

    def forward_encoder(self, x):
        features = []
        for i in range(4):
            x, H, W = getattr(self, f"patch_embed{i+1}")(x)
            for blk in getattr(self, f"block{i+1}"):
                x = blk(x, H, W)
            x = getattr(self, f"norm{i+1}")(x)
            B, _, C = x.shape
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            features.append(x)
        return features

    def forward_decoder(self, features):
        target_H, target_W = features[0].shape[2:]
        projected = []
        for i, feat in enumerate(features):
            B, C, H, W = feat.shape
            x = feat.flatten(2).transpose(1, 2)
            x = self.linear_layers[i](x)
            x = x.transpose(1, 2).reshape(B, -1, H, W)
            if H != target_H or W != target_W:
                x = F.interpolate(x, (target_H, target_W), mode="bilinear", align_corners=False)
            projected.append(x)
        x = torch.cat(projected, dim=1)
        x = self.linear_fuse(x)
        x = self.dropout(x)
        return self.classifier(x)

    def forward(self, x):
        return self.forward_decoder(self.forward_encoder(x))

    def predict(self, x, threshold=0.5):
        logits = self.forward(x)
        logits = F.interpolate(logits, x.shape[2:], mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1)
        return (probs[:, 1] > threshold).long()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 3: Image Utilities & Post-processing                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def read_image(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(img: np.ndarray, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if img.ndim == 3 and img.shape[2] == 3:
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(path), img)


def img_to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    return (t.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)


def detect_bacilli(mask: np.ndarray, min_area: int = 10,
                   connectivity: int = 2) -> List[Dict]:
    binary = (mask > 0).astype(np.uint8)
    labeled = measure.label(binary, connectivity=connectivity)
    regions = measure.regionprops(labeled)
    detections = []
    for i, r in enumerate(regions, 1):
        if r.area < min_area:
            continue
        detections.append({
            "id": i,
            "bbox": [int(c) for c in r.bbox],
            "area": int(r.area),
            "centroid": [round(float(c), 1) for c in r.centroid],
        })
    return detections


def colorize_mask(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask > 0] = [255, 0, 0]
    return out


def create_overlay(image, mask, detections):
    overlay = image.copy()
    colored = colorize_mask(mask)
    region = mask > 0
    overlay[region] = (0.6 * image[region] + 0.4 * colored[region]).astype(np.uint8)
    for det in detections:
        y1, x1, y2, x2 = det["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(overlay, f"Count: {len(detections)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return overlay


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 4: Tiled SR Inference (for 2448×2048 images)                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def compute_tiles(h, w, tile_size, overlap):
    stride = tile_size - overlap
    tiles = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            tiles.append((y_start, x_start, y_end, x_end))
    return list(dict.fromkeys(tiles))


def tiled_sr_inference(model, lr_img, scale, tile_size, overlap, device):
    h, w, c = lr_img.shape
    out_h, out_w = h * scale, w * scale
    output = np.zeros((out_h, out_w, c), dtype=np.float64)
    weight_sum = np.zeros((out_h, out_w), dtype=np.float64)
    tiles = compute_tiles(h, w, tile_size, overlap)

    # Build blending weight
    wmap = np.ones((tile_size, tile_size), dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap, dtype=np.float32)
        wmap[:, :overlap] *= ramp[None, :]
        wmap[:, -overlap:] *= ramp[None, ::-1]
        wmap[:overlap, :] *= ramp[:, None]
        wmap[-overlap:, :] *= ramp[::-1, None]

    model.eval()
    with torch.no_grad():
        for y0, x0, y1, x1 in tiles:
            tile = lr_img[y0:y1, x0:x1]
            th, tw = tile.shape[:2]
            tile_t = img_to_tensor(tile).unsqueeze(0).to(device)

            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                sr_tile_t = model(tile_t)
            sr_tile = tensor_to_img(sr_tile_t.squeeze(0))

            # Weight for this tile
            wm = np.repeat(np.repeat(wmap[:th, :tw], scale, axis=0), scale, axis=1)

            sy0, sx0 = y0 * scale, x0 * scale
            sy1, sx1 = sy0 + th * scale, sx0 + tw * scale

            output[sy0:sy1, sx0:sx1] += sr_tile.astype(np.float64) * wm[:, :, None]
            weight_sum[sy0:sy1, sx0:sx1] += wm

    weight_sum = np.maximum(weight_sum, 1e-8)
    output = (output / weight_sum[:, :, None]).clip(0, 255).astype(np.uint8)
    return output


def simple_sr_inference(model, lr_img, device):
    model.eval()
    with torch.no_grad():
        t = img_to_tensor(lr_img).unsqueeze(0).to(device)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            sr_t = model(t)
        return tensor_to_img(sr_t.squeeze(0))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 5: Segmentation Inference                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def segment_image(model, image, device, image_size=512, threshold=0.5, min_area=10):
    """Run SegFormer on one image, return mask + detections."""
    orig_h, orig_w = image.shape[:2]

    # Preprocess: resize + normalize (ImageNet stats)
    resized = cv2.resize(image, (image_size, image_size))
    x = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        logits = F.interpolate(logits, (image_size, image_size), mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1)
        pred_mask = (probs[0, 1] > threshold).cpu().numpy().astype(np.uint8)

    # Back to original resolution
    if (pred_mask.shape[0] != orig_h) or (pred_mask.shape[1] != orig_w):
        pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    detections = detect_bacilli(pred_mask, min_area)
    return pred_mask, detections


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 6: Checkpoint Loading Helpers                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_model_checkpoint(model, ckpt_path, device):
    """Load checkpoint into model, handling DataParallel prefix."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_state = state.get("model_state_dict", state.get("state_dict", state))
    cleaned = {k.replace("module.", ""): v for k, v in model_state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning(f"  Missing keys: {len(missing)}")
    if unexpected:
        logger.warning(f"  Unexpected keys: {len(unexpected)}")
    logger.info(f"Loaded checkpoint: {ckpt_path}")
    return model


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 7: Pipeline Orchestrator                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def discover_images(dataset_dir: str, categories: List[str],
                    extensions: List[str]) -> Dict[str, List[str]]:
    """Discover images per category folder."""
    found = {}
    for cat in categories:
        cat_dir = os.path.join(dataset_dir, cat)
        if not os.path.isdir(cat_dir):
            logger.warning(f"Category directory not found: {cat_dir}")
            found[cat] = []
            continue
        files = sorted([
            f for f in os.listdir(cat_dir)
            if os.path.splitext(f)[1].lower() in extensions
        ])
        found[cat] = files
        logger.info(f"  {cat}: {len(files)} images")
    return found


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_pipeline(config_path: str = None,
                 sr_checkpoint: str = None,
                 seg_checkpoint: str = None,
                 skip_sr: bool = False):
    """
    Execute the full SR → Segmentation pipeline.
    """
    t_start = time.time()

    # ── Load config ──
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # ── Resolve paths ──
    input_cfg = cfg.get("input", {})
    dataset_dir = input_cfg.get("dataset_dir",
                                "/Users/priyakumari/Desktop/t24157_backup/testingDataset")
    categories = input_cfg.get("categories", ["Positive", "Negative"])
    extensions = input_cfg.get("image_extensions", [".jpg", ".jpeg", ".png", ".bmp", ".tif"])

    sr_cfg = cfg.get("sr", {})
    sr_ckpt = sr_checkpoint or sr_cfg.get("checkpoint", "")
    sr_model_cfg = sr_cfg.get("model", {})
    use_tiled = sr_cfg.get("tiled", True)
    tile_size = sr_cfg.get("tile_size", 256)
    tile_overlap = sr_cfg.get("tile_overlap", 32)
    sr_scale = sr_model_cfg.get("upscale", 4)
    sr_output_suffix = sr_cfg.get("output_suffix", "_SR")
    sr_output_format = sr_cfg.get("output_format", "png")

    seg_cfg = cfg.get("segmentation", {})
    seg_ckpt = seg_checkpoint or seg_cfg.get("checkpoint", "")
    seg_model_cfg = seg_cfg.get("model", {})
    seg_image_size = seg_cfg.get("image_size", 512)
    seg_threshold = seg_cfg.get("threshold", 0.5)
    seg_min_area = seg_cfg.get("min_area", 10)

    out_cfg = cfg.get("output", {})
    output_base = out_cfg.get("base_dir",
                              "/Users/priyakumari/Desktop/t24157_backup/full_pipeline/pipeline_outputs")
    save_sr = out_cfg.get("save_sr_images", True)
    save_masks = out_cfg.get("save_masks", True)
    save_overlays = out_cfg.get("save_overlays", True)
    results_filename = out_cfg.get("results_filename", "pipeline_results.json")
    summary_filename = out_cfg.get("summary_filename", "pipeline_summary.json")

    # SR output dir (same structure as input)
    sr_output_root = dataset_dir + sr_output_suffix

    device = get_device()
    logger.info("=" * 70)
    logger.info("  TB Bacilli Full Pipeline")
    logger.info("=" * 70)
    logger.info(f"  Device:       {device}")
    logger.info(f"  Dataset:      {dataset_dir}")
    logger.info(f"  Categories:   {categories}")
    logger.info(f"  Skip SR:      {skip_sr}")
    logger.info(f"  SR Checkpoint: {sr_ckpt}")
    logger.info(f"  Seg Checkpoint:{seg_ckpt}")
    logger.info(f"  Output:       {output_base}")
    logger.info("=" * 70)

    # ── Step 1: Discover images ──
    logger.info("\n[Step 1/4] Discovering images...")
    image_map = discover_images(dataset_dir, categories, extensions)
    total_images = sum(len(v) for v in image_map.values())
    if total_images == 0:
        logger.error("No images found. Aborting.")
        return
    logger.info(f"  Total: {total_images} images")

    # ── Step 2: Super-Resolution ──
    sr_model = None
    if not skip_sr:
        logger.info("\n[Step 2/4] Loading SwinIR model...")
        sr_model = SwinIR(
            img_channels=sr_model_cfg.get("img_channels", 3),
            embed_dim=sr_model_cfg.get("embed_dim", 180),
            depths=sr_model_cfg.get("depths", [6]*8),
            num_heads=sr_model_cfg.get("num_heads", [6]*8),
            window_size=sr_model_cfg.get("window_size", 8),
            mlp_ratio=sr_model_cfg.get("mlp_ratio", 2.0),
            upscale=sr_scale,
        )

        if sr_ckpt and os.path.exists(sr_ckpt):
            load_model_checkpoint(sr_model, sr_ckpt, device)
        else:
            logger.warning(f"  SR checkpoint not found at '{sr_ckpt}' — using random weights")

        sr_model = sr_model.to(device)
        sr_model.eval()
        n_params = sum(p.numel() for p in sr_model.parameters())
        logger.info(f"  SwinIR loaded — {n_params/1e6:.2f}M params")

        logger.info("\n[Step 2/4] Running super-resolution...")
        sr_count = 0
        for cat, files in image_map.items():
            cat_sr_dir = os.path.join(sr_output_root, cat)
            os.makedirs(cat_sr_dir, exist_ok=True)

            for fname in tqdm(files, desc=f"SR [{cat}]", leave=True):
                src = os.path.join(dataset_dir, cat, fname)
                stem = os.path.splitext(fname)[0]
                dst = os.path.join(cat_sr_dir, f"{stem}_sr.{sr_output_format}")

                # Skip if already exists
                if os.path.exists(dst):
                    sr_count += 1
                    continue

                lr_img = read_image(src)
                if use_tiled:
                    sr_img = tiled_sr_inference(sr_model, lr_img, sr_scale,
                                                tile_size, tile_overlap, device)
                else:
                    sr_img = simple_sr_inference(sr_model, lr_img, device)

                if save_sr:
                    save_image(sr_img, dst)
                sr_count += 1

        logger.info(f"  SR complete: {sr_count} images → {sr_output_root}")

        # Free SR model memory
        del sr_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.info("\n[Step 2/4] Skipping SR — will segment original images")

    # ── Step 3: Segmentation ──
    logger.info("\n[Step 3/4] Loading SegFormer B4 model...")
    seg_model = SegFormerB4(
        num_classes=seg_model_cfg.get("num_classes", 2),
        drop_path_rate=seg_model_cfg.get("drop_path_rate", 0.1),
    )

    if seg_ckpt and os.path.exists(seg_ckpt):
        load_model_checkpoint(seg_model, seg_ckpt, device)
    else:
        logger.warning(f"  Seg checkpoint not found at '{seg_ckpt}' — using random weights")

    seg_model = seg_model.to(device)
    seg_model.eval()
    n_params = sum(p.numel() for p in seg_model.parameters())
    logger.info(f"  SegFormer B4 loaded — {n_params/1e6:.2f}M params")

    logger.info("\n[Step 3/4] Running segmentation...")
    all_results = []  # Per-image results for JSON
    os.makedirs(output_base, exist_ok=True)

    for cat, files in image_map.items():
        # Determine input dir for segmentation
        if skip_sr:
            seg_input_dir = os.path.join(dataset_dir, cat)
        else:
            seg_input_dir = os.path.join(sr_output_root, cat)

        # Output dirs
        mask_dir = os.path.join(output_base, cat, "masks")
        overlay_dir = os.path.join(output_base, cat, "overlays")
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)

        for fname in tqdm(files, desc=f"Seg [{cat}]", leave=True):
            # Resolve the actual filename (SR images have _sr suffix)
            stem = os.path.splitext(fname)[0]
            if skip_sr:
                src = os.path.join(seg_input_dir, fname)
            else:
                sr_fname = f"{stem}_sr.{sr_output_format}"
                src = os.path.join(seg_input_dir, sr_fname)
                if not os.path.exists(src):
                    # Fallback to original if SR image not found
                    src = os.path.join(dataset_dir, cat, fname)
                    logger.warning(f"  SR image not found, using original: {fname}")

            try:
                image = read_image(src)
            except FileNotFoundError:
                logger.error(f"  Cannot read: {src}")
                continue

            mask, detections = segment_image(
                seg_model, image, device, seg_image_size, seg_threshold, seg_min_area)

            # Summary stats
            areas = [d["area"] for d in detections]
            summary = {
                "count": len(detections),
                "avg_area": round(sum(areas) / len(areas), 1) if areas else 0,
                "total_area": sum(areas),
                "min_area": min(areas) if areas else 0,
                "max_area": max(areas) if areas else 0,
            }

            # Save mask
            if save_masks:
                save_image(mask * 255, os.path.join(mask_dir, f"{stem}_mask.png"))

            # Save overlay
            if save_overlays:
                overlay = create_overlay(image, mask, detections)
                save_image(overlay, os.path.join(overlay_dir, f"{stem}_overlay.png"))

            # Collect result
            all_results.append({
                "filename": fname,
                "category": cat,
                "source_image": src,
                "sr_applied": not skip_sr,
                "image_size": {"height": image.shape[0], "width": image.shape[1]},
                "bacilli_count": summary["count"],
                "summary": summary,
                "detections": detections,
            })

    # Free seg model
    del seg_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Step 4: Save JSON results ──
    logger.info("\n[Step 4/4] Saving results to JSON...")

    # Per-image results
    results_path = os.path.join(output_base, results_filename)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"  Per-image results: {results_path}")

    # Global summary
    total_elapsed = time.time() - t_start
    pos_results = [r for r in all_results if r["category"] == "Positive"]
    neg_results = [r for r in all_results if r["category"] == "Negative"]

    summary_data = {
        "pipeline_run": {
            "timestamp": datetime.now().isoformat(),
            "device": str(device),
            "sr_applied": not skip_sr,
            "sr_checkpoint": sr_ckpt if not skip_sr else None,
            "seg_checkpoint": seg_ckpt,
            "total_images": total_images,
            "elapsed_seconds": round(total_elapsed, 1),
            "avg_seconds_per_image": round(total_elapsed / max(total_images, 1), 2),
        },
        "overall": {
            "total_bacilli_detected": sum(r["bacilli_count"] for r in all_results),
            "images_with_bacilli": sum(1 for r in all_results if r["bacilli_count"] > 0),
            "images_without_bacilli": sum(1 for r in all_results if r["bacilli_count"] == 0),
            "avg_bacilli_per_image": round(
                sum(r["bacilli_count"] for r in all_results) / max(total_images, 1), 2),
        },
        "by_category": {},
    }

    for cat_name, cat_results in [("Positive", pos_results), ("Negative", neg_results)]:
        if not cat_results:
            continue
        counts = [r["bacilli_count"] for r in cat_results]
        summary_data["by_category"][cat_name] = {
            "num_images": len(cat_results),
            "total_bacilli": sum(counts),
            "avg_bacilli_per_image": round(sum(counts) / len(counts), 2),
            "images_with_bacilli": sum(1 for c in counts if c > 0),
            "images_without_bacilli": sum(1 for c in counts if c == 0),
            "max_bacilli_in_single_image": max(counts) if counts else 0,
        }

    summary_path = os.path.join(output_base, summary_filename)
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"  Pipeline summary: {summary_path}")

    # Final report
    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total images processed: {total_images}")
    logger.info(f"  Total bacilli detected: {summary_data['overall']['total_bacilli_detected']}")
    logger.info(f"  Images with bacilli:    {summary_data['overall']['images_with_bacilli']}")
    logger.info(f"  Time elapsed:           {total_elapsed:.1f}s")
    for cat_name, cat_data in summary_data["by_category"].items():
        logger.info(f"  [{cat_name}] {cat_data['num_images']} images, "
                     f"{cat_data['total_bacilli']} bacilli, "
                     f"avg {cat_data['avg_bacilli_per_image']}/img")
    logger.info(f"\n  Results:  {results_path}")
    logger.info(f"  Summary:  {summary_path}")
    logger.info(f"  SR imgs:  {sr_output_root}")
    logger.info(f"  Outputs:  {output_base}")
    logger.info("=" * 70)

    return summary_data


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN ENTRY POINT                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="TB Bacilli Full Pipeline: SR → Segmentation → Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py
  python run_pipeline.py --config pipeline_config.yaml
  python run_pipeline.py --skip_sr
  python run_pipeline.py --sr_checkpoint path/to/sr.pth --seg_checkpoint path/to/seg.pth
  python run_pipeline.py --dataset_dir /path/to/testingDataset
        """,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to pipeline_config.yaml (optional)")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Override input dataset directory")
    parser.add_argument("--sr_checkpoint", type=str, default=None,
                        help="Override SR model checkpoint path")
    parser.add_argument("--seg_checkpoint", type=str, default=None,
                        help="Override segmentation model checkpoint path")
    parser.add_argument("--skip_sr", action="store_true",
                        help="Skip SR, run segmentation on original images")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # Auto-detect config location if not specified
    config_path = args.config
    if config_path is None:
        # Try to find it relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_config = os.path.join(script_dir, "pipeline_config.yaml")
        if os.path.exists(default_config):
            config_path = default_config

    # Apply CLI overrides to config
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    if args.dataset_dir:
        cfg.setdefault("input", {})["dataset_dir"] = args.dataset_dir
    if args.output_dir:
        cfg.setdefault("output", {})["base_dir"] = args.output_dir

    # Re-save patched config for downstream use
    if config_path and os.path.exists(config_path):
        # Write back patched version to temp
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(cfg, tmp)
        tmp.close()
        config_path = tmp.name

    run_pipeline(
        config_path=config_path,
        sr_checkpoint=args.sr_checkpoint,
        seg_checkpoint=args.seg_checkpoint,
        skip_sr=args.skip_sr,
    )


if __name__ == "__main__":
    main()
