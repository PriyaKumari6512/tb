# TB Bacilli Detection Pipeline — Full Details

End-to-end pipeline for detecting and counting **Mycobacterium tuberculosis (TB) bacilli** in microscopy images using:

1. **SwinIR** — 4× super-resolution to enhance low-resolution images
2. **SegFormer (MIT-B4)** — semantic segmentation to detect and count bacilli
3. **Combined inference** — side-by-side comparison of original vs. SR-enhanced detection

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Dataset](#2-dataset)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Super-Resolution Module (SwinIR)](#4-super-resolution-module-swinir)
   - [Architecture](#41-architecture)
   - [Dataset & Augmentations](#42-dataset--augmentations)
   - [Training](#43-training)
   - [Evaluation](#44-evaluation)
   - [Inference](#45-inference)
   - [Configuration Reference](#46-configuration-reference)
5. [Segmentation Module (SegFormer)](#5-segmentation-module-segformer)
   - [Architecture](#51-architecture)
   - [Dataset & Augmentations](#52-dataset--augmentations)
   - [Training](#53-training)
   - [Post-processing](#54-post-processing)
   - [Evaluation](#55-evaluation)
   - [Inference](#56-inference)
   - [Configuration Reference](#57-configuration-reference)
6. [Combined SR → Segmentation Pipeline](#6-combined-sr--segmentation-pipeline)
   - [Flow](#61-flow)
   - [Outputs](#62-outputs)
   - [Configuration Reference](#63-configuration-reference)
7. [Experiment Management](#7-experiment-management)
8. [Setup & Installation](#8-setup--installation)
9. [Running the Pipeline](#9-running-the-pipeline)
   - [Training](#91-training)
   - [Evaluation](#92-evaluation)
   - [Inference](#93-inference)
10. [Monitoring & Logging](#10-monitoring--logging)
11. [Testing](#11-testing)
12. [Key Design Decisions](#12-key-design-decisions)

---

## 1. Repository Structure

```
tb_bacilli_pipeline/              ← Main production codebase
├── configs/
│   ├── sr_config.yaml            # SwinIR super-resolution config
│   ├── tb_config.yaml            # SegFormer segmentation config
│   └── pipeline_config.yaml      # Combined pipeline config
│
├── sr_project/                   # Super-resolution module
│   ├── model.py                  # SwinIR architecture
│   ├── dataset.py                # LR/HR pair dataset (on-the-fly downsampling)
│   ├── train.py                  # Training loop (AMP, multi-GPU, staged finetuning)
│   ├── evaluate.py               # PSNR/SSIM evaluation + visual grids
│   ├── inference.py              # Single image / folder / tiled inference
│   └── utils.py                  # Shared utilities (checkpointing, metrics, tiling)
│
├── tb_project/                   # Segmentation module
│   ├── model.py                  # SegFormer wrapper + loss functions
│   ├── dataset.py                # Image/mask dataset with augmentations
│   ├── train.py                  # Training with differential learning rates
│   ├── evaluate.py               # Dice/IoU/Precision/Recall/F1 evaluation
│   ├── inference.py              # Inference with bacilli counting
│   ├── postprocess.py            # Connected-component analysis
│   └── utils.py                  # Segmentation metrics & visualization helpers
│
├── pipelines/                    # Combined pipelines
│   ├── combined_inference.py     # SR → Segmentation comparison pipeline
│   ├── experiment_runner.py      # Config-driven experiment launcher
│   └── report.py                 # Report generation utilities
│
├── scripts/
│   ├── run_sr_train.sh           # SR training wrapper script
│   ├── run_tb_train.sh           # TB training wrapper script
│   ├── run_combined_inference.sh # Combined inference wrapper
│   ├── prepare_data.sh           # Dataset verification
│   └── check_status.sh           # Environment diagnostics
│
├── tests/
│   ├── test_sr_dataset.py
│   ├── test_tb_dataset.py
│   └── test_model_forward.py
│
├── requirements.txt
├── pytest.ini
└── README.md

full_pipeline/                    # Older standalone pipeline version
├── run_pipeline.py
└── pipeline_config.yaml

segformer_b4/                     # Standalone SegFormer reference implementation
├── model.py                      # From-scratch MiT backbone
├── train.py
├── dataset.py
├── evaluate.py
├── inference.py
├── utils.py
└── configs/segformer_config.yaml
```

---

## 2. Dataset

The pipeline uses the **DDS3** microscopy dataset organised as follows:

```
DDS3/
├── TRAINING SET/
│   ├── 100% NEGATIVE/
│   │   ├── IMAGE/   (MSC1.bmp … MSC1500.bmp)
│   │   └── MASK/
│   ├── 90% NEGATIVE/
│   │   ├── IMAGE/   (MSC1.bmp … MSC1500.bmp)
│   │   └── MASK/
│   └── 50% NEGATIVE/
│       ├── IMAGE/   (MSC1.bmp … MSC3000.bmp)
│       └── MASK/
├── VALIDATION SET/
│   └── 50% NEGATIVE/
│       ├── IMAGE/   (~1000 images)
│       └── MASK/
└── TEST SET/
    └── 50% NEGATIVE/
        ├── IMAGE/   (~1000 images)
        └── MASK/
```

| Property | Value |
|---|---|
| Format | BMP (also supports PNG, JPG, TIFF) |
| Image resolution | ~2448 × 2048 px (RGB, 3-channel) |
| Mask encoding | Binary: 0 = background, >0 = bacilli region |
| Subfolders | Named by approximate positive (bacilli-containing) fraction |

---

## 3. Pipeline Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                      TRAINING PHASE                                    │
│                                                                        │
│  DDS3 HR Images ──→ on-the-fly bicubic downsample ──→ LR/HR pairs     │
│         │                                                              │
│         ▼                                                              │
│  SwinIR Training ──────────────────────→ checkpoints/sr/best_model.pth│
│                                                                        │
│  DDS3 Images + Masks ─────────────────→ SegFormer Training            │
│         │                              ──→ checkpoints/tb/best_model.pth
│         └──────────────────────────────────────────────────────────── │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                      INFERENCE PHASE                                   │
│                                                                        │
│  Test Image                                                            │
│      │                                                                 │
│      ├──→ [SwinIR 4×] ──→ SR Image ──→ [SegFormer] ──→ Mask ──→ Count │
│      │                                                                 │
│      └──────────────────→ [SegFormer] ──→ Mask ──→ Count              │
│                                                                        │
│  Comparison: original count vs. SR-enhanced count (+ Dice vs GT)      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Super-Resolution Module (SwinIR)

### 4.1 Architecture

**SwinIR** (Swin Transformer for Image Restoration) transforms a low-resolution image into a 4× upscaled output.

```
LR Input (B, 3, H, W)
       │
       ▼
Shallow Feature Extraction  ← single 3×3 Conv
       │
       ▼
Deep Feature Extraction
  ┌─────────────────────────────┐
  │  RSTB × 8  (each block):   │
  │    STL → STL → … → STL     │  M Swin Transformer Layers
  │    + 1×1 or 3×3 conv       │
  │    + residual connection    │
  └─────────────────────────────┘
       │
       ▼
Conv + PixelShuffle 4×  ← subpixel convolution upsampler
       │
       ▼
SR Output (B, 3, 4H, 4W)
```

**Key components:**

| Component | Description |
|---|---|
| `WindowAttention` | Multi-head self-attention computed within non-overlapping local windows; uses relative position bias |
| `STL` (Swin Transformer Layer) | Window-based attention with alternating regular / shifted-window partitioning |
| `RSTB` (Residual Swin Transformer Block) | Stack of M STL layers + a conv layer + residual skip |
| `SwinIR` | Full model: shallow conv → 8×RSTB → conv → PixelShuffle |

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Embedding dim | 180 |
| RSTB blocks | 8 |
| Heads per block | 6 |
| Window size | 8 |
| MLP ratio | 2.0 |
| Upscale factor | 4 |
| Parameters | ~12 M |

### 4.2 Dataset & Augmentations

`sr_project/dataset.py` generates LR/HR pairs **on-the-fly** at training time — no pre-computed LR images are needed.

| Split | Augmentations |
|---|---|
| Train | Random crop (64×64 HR / 16×16 LR), random horizontal flip, random vertical flip, random 90° rotation |
| Val / Test | Full image, no augmentation |

```
HR image  ──→  random 64×64 crop  ──→  HR patch (64×64)
                                   └──→  bicubic ↓4  ──→  LR patch (16×16)
```

### 4.3 Training

File: `sr_project/train.py`

**Loss — staged finetuning:**

| Epochs | Loss |
|---|---|
| 0 – 30 | L1 pixel loss only |
| 31 – 200 | L1 pixel loss + VGG-19 Perceptual loss (weight 0.1) |

Staged finetuning ensures model stability: pixel-accurate reconstruction first, then perceptual quality.

**Perceptual loss layers:** `relu1_2`, `relu2_2`, `relu3_4`, `relu4_4` of VGG-19.

**Optimizer & Scheduler:**

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| LR | 2e-4 |
| Min LR | 1e-7 |
| Betas | (0.9, 0.99) |
| Weight decay | 0.0 |
| Scheduler | Cosine decay with linear warmup |
| Warmup epochs | 5 |
| Gradient clipping | 1.0 |

**Training infrastructure:**
- **AMP (Automatic Mixed Precision):** Reduces memory and speeds training
- **Multi-GPU:** `torch.nn.DataParallel` across all available GPUs
- **Early stopping:** patience = 30 epochs (monitors validation PSNR)
- **Checkpoints:** Best model + every 10 epochs

### 4.4 Evaluation

File: `sr_project/evaluate.py`

Metrics computed on Y-channel (luminance) after converting to YCbCr:

| Metric | Description |
|---|---|
| PSNR | Peak Signal-to-Noise Ratio (dB) |
| SSIM | Structural Similarity Index |

Also computes **bicubic baseline** (simple 4× bicubic upsampling) for comparison. Outputs:
- Per-image CSV with PSNR and SSIM for both SwinIR and bicubic
- Visual comparison grids: `[LR | Bicubic | SwinIR SR | HR]`

### 4.5 Inference

File: `sr_project/inference.py`

Three modes:

| Mode | Description |
|---|---|
| Single image | `infer_single(image_path)` → saves SR image |
| Folder batch | `infer_folder(folder_path)` → processes all images |
| Tiled | Splits large images into 256×256 tiles with 32 px overlap, processes each tile, blends back |

Tiled inference is critical for the large ~2448×2048 px microscopy images that would exceed GPU memory if processed in full.

### 4.6 Configuration Reference

File: `configs/sr_config.yaml`

```yaml
data:
  root: "${DATA_ROOT:./DDS3}"
  scale: 4
  patch_size: 64          # HR crop (LR = patch_size / scale = 16)
  batch_size: 16

model:
  embed_dim: 180
  depths: [6, 6, 6, 6, 6, 6, 6, 6]   # 8 RSTB blocks, 6 STL each
  num_heads: [6, 6, 6, 6, 6, 6, 6, 6]
  window_size: 8
  upscale: 4

training:
  epochs: 200
  lr: 2.0e-4
  staged_finetuning:
    stage1_epochs: 30    # L1 only
    stage2_start: 31     # L1 + perceptual

checkpoint:
  dir: "${CKPT_DIR:./checkpoints}/sr"

inference:
  tile_size: 256
  tile_overlap: 32
```

---

## 5. Segmentation Module (SegFormer)

### 5.1 Architecture

**SegFormer** is a hierarchical vision Transformer for semantic segmentation.

```
Input Image (B, 3, 512, 512)
       │
       ▼
MIT-B4 Encoder (4 stages, hierarchical)
  Stage 1 → Feature map 1/4  (H/4,  W/4)
  Stage 2 → Feature map 1/8  (H/8,  W/8)
  Stage 3 → Feature map 1/16 (H/16, W/16)
  Stage 4 → Feature map 1/32 (H/32, W/32)
       │
       ▼
All-MLP Decoder Head
  (concatenates multi-scale features → unified representation)
       │
       ▼
Bilinear upsample → original resolution
       │
       ▼
Logits (B, 2, 512, 512)   ← 2 classes: background (0), bacilli (1)
       │
       ▼
Softmax / argmax → Binary Mask (B, 512, 512)
```

**Implementation:** `TBSegFormer` wraps HuggingFace `SegformerForSemanticSegmentation` with the `nvidia/mit-b4` pretrained backbone, replacing the classification head for `num_classes=2`.

### 5.2 Dataset & Augmentations

File: `tb_project/dataset.py`

**Training augmentations** (via Albumentations):

| Augmentation | Parameters |
|---|---|
| RandomResizedCrop | scale (0.5, 1.0) |
| HorizontalFlip | p=0.5 |
| VerticalFlip | p=0.5 |
| RandomRotate90 | p=0.5 |
| ColorJitter | brightness, contrast, saturation |
| GaussNoise | |
| GaussianBlur | |
| Normalize | ImageNet mean/std |

**Validation / Test:** Resize to 512×512 + Normalize only.

**Class imbalance handling:** `WeightedRandomSampler` oversamples images from bacilli-rich subfolders (`50% NEGATIVE` weighted higher than `100% NEGATIVE`).

### 5.3 Training

File: `tb_project/train.py`

**Loss function — DiceBCE:**

```
Loss = dice_weight × DiceLoss + bce_weight × BCEWithLogitsLoss
```

| Parameter | Value |
|---|---|
| Dice weight | 1.0 |
| BCE weight | 0.5 |
| `pos_weight` (BCE) | 5.0 — heavily penalises missing bacilli |
| Dice smooth | 1.0 |

DiceLoss is computed on sigmoid-activated logits in binary (B, 1, H, W) form; `to_binary_logits()` converts 2-class output before loss computation.

**Differential learning rates:**

| Part | LR multiplier |
|---|---|
| MIT-B4 backbone (encoder) | 0.1× (i.e. 6e-6) |
| MLP decode head | 1.0× (i.e. 6e-5) |

This prevents overwriting the well-pretrained backbone features while allowing the decoder to learn quickly.

**Optimizer & Scheduler:**

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| LR | 6e-5 |
| Min LR | 1e-7 |
| Betas | (0.9, 0.999) |
| Weight decay | 0.01 |
| Scheduler | Cosine with linear warmup (3 epochs) |
| Gradient clipping | 1.0 |

**Training infrastructure:**
- **AMP:** Mixed precision enabled
- **Multi-GPU:** DataParallel
- **Early stopping:** patience = 20 epochs (monitors validation Dice)
- **Checkpoints:** Best model + every 5 epochs
- **Logging:** JSONL metrics + TensorBoard

### 5.4 Post-processing

File: `tb_project/postprocess.py`

After the model outputs a soft probability map, post-processing turns it into discrete bacilli detections:

```
Soft logits  ──→  sigmoid  ──→  threshold (0.5)  ──→  binary mask
                                                           │
                                         scikit-image label (connected components)
                                                           │
                                    filter: min_area = 10 px (remove noise)
                                                           │
                                    extract: bbox, centroid, area per component
                                                           │
                                              Detection list + summary
```

`detection_summary()` provides: total count, average/min/max area.

### 5.5 Evaluation

File: `tb_project/evaluate.py`

Full segmentation evaluation metrics:

| Metric | Description |
|---|---|
| Dice | 2·TP / (2·TP + FP + FN) |
| IoU | TP / (TP + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | Harmonic mean of Precision and Recall |
| Bacilli count error | |predicted count − GT count| |

Results are broken down per subfolder (`100% NEGATIVE`, `90% NEGATIVE`, `50% NEGATIVE`) to show performance at different positive rates. Outputs per-image CSV and visual inspection grids.

### 5.6 Inference

File: `tb_project/inference.py`

`infer_single(image)` returns:
- `image` — original image
- `mask` — predicted binary mask
- `overlay` — image with green bounding boxes drawn around each detected bacillus and a count label
- `detections` — list of `{id, bbox, area, centroid}` per bacillus
- `summary` — count, avg/min/max area

`infer_folder(folder)` processes all images and outputs:
- Predicted masks
- Overlay images
- `counts.csv` with per-image bacilli counts

### 5.7 Configuration Reference

File: `configs/tb_config.yaml`

```yaml
data:
  root: "${DATA_ROOT:./DDS3}"
  image_size: 512
  batch_size: 8

model:
  backbone: "nvidia/mit-b4"
  num_classes: 2
  pretrained: true

training:
  epochs: 100
  lr: 6.0e-5
  loss:
    type: "dice_bce"
    pos_weight: 5.0

postprocess:
  min_area: 10
  connectivity: 2
  conf_threshold: 0.5

checkpoint:
  dir: "${CKPT_DIR:./checkpoints}/tb"
```

---

## 6. Combined SR → Segmentation Pipeline

### 6.1 Flow

File: `pipelines/combined_inference.py`

For each test image, two parallel paths are run and compared:

```
Test Image
    │
    ├──── PATH A: Original ─────────────────────────────────────────────┐
    │     Image (native resolution)                                     │
    │     → TBSegFormer → Binary Mask → Post-process → Count_A, Overlay_A│
    │                                                                   │
    ├──── PATH B: SR-Enhanced ──────────────────────────────────────────┤
    │     Image → SwinIR 4× → SR Image (4× resolution)                 │
    │     → TBSegFormer → Binary Mask → Post-process → Count_B, Overlay_B│
    │                                                                   │
    └──── COMPARISON ───────────────────────────────────────────────────┘
          Count_A vs Count_B vs Count_GT (if available)
          Dice_A vs Dice_B vs ground-truth mask
          Side-by-side visualisation
```

### 6.2 Outputs

All results are saved under `experiments/{type}_{timestamp}/`:

```
experiments/
└── combined_20240314_120000/
    ├── comparison_report.csv     ← main results table
    ├── summary.txt               ← aggregate statistics
    ├── original/
    │   └── *.png                 ← overlay images (original path)
    ├── sr_enhanced/
    │   ├── *_sr.png              ← 4× SR images
    │   └── *_overlay.png        ← SR path overlays
    └── side_by_side/
        └── *.png                 ← [Original | SR | GT Mask] comparisons
```

**`comparison_report.csv` columns:**

| Column | Description |
|---|---|
| `filename` | Image filename |
| `count_original` | Bacilli count on original image |
| `count_sr` | Bacilli count on SR-enhanced image |
| `count_gt` | Ground-truth bacilli count |
| `delta` | count_sr − count_original |
| `dice_original` | Dice score on original image |
| `dice_sr` | Dice score on SR-enhanced image |

### 6.3 Configuration Reference

File: `configs/pipeline_config.yaml`

```yaml
sr:
  config: "configs/sr_config.yaml"
  checkpoint: "${CKPT_DIR:./checkpoints}/sr/best_model.pth"
  enabled: true

tb:
  config: "configs/tb_config.yaml"
  checkpoint: "${CKPT_DIR:./checkpoints}/tb/best_model.pth"

data:
  test_dir: "${DATA_ROOT:./DDS3}/TEST SET/50% NEGATIVE/IMAGE"
  mask_dir: "${DATA_ROOT:./DDS3}/TEST SET/50% NEGATIVE/MASK"

output:
  dir: "experiments/"
  save_overlays: true
  save_side_by_side: true

comparison:
  modes: ["original", "sr_enhanced"]
  metrics: ["bacilli_count", "dice", "iou"]
```

---

## 7. Experiment Management

File: `pipelines/experiment_runner.py`

Each run gets a unique timestamped ID and an isolated output directory:

```
experiments/
├── sr_20240314_091523/
│   ├── config_snapshot.yaml
│   ├── logs/
│   └── checkpoints/
└── tb_20240314_153010/
    ├── config_snapshot.yaml
    ├── logs/
    └── checkpoints/
```

```bash
python -m pipelines.experiment_runner \
  --config configs/sr_config.yaml \
  --type sr \
  --name "sr_embed180_ep200"
```

---

## 8. Setup & Installation

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure paths
cp .env.example .env
# Edit .env and set DATA_ROOT, CKPT_DIR, LOG_DIR, OUTPUT_DIR as needed

# 4. Verify dataset and environment
bash scripts/prepare_data.sh
bash scripts/check_status.sh
```

**Environment variables (with defaults):**

| Variable | Default | Description |
|---|---|---|
| `DATA_ROOT` | `./DDS3` | Path to DDS3 dataset root |
| `CKPT_DIR` | `./checkpoints` | Checkpoint save directory |
| `LOG_DIR` | `./logs` | Log file directory |
| `OUTPUT_DIR` | `./outputs` | Inference output directory |

---

## 9. Running the Pipeline

### 9.1 Training

#### Super-Resolution (SwinIR)

```bash
# Standard training (via script)
bash scripts/run_sr_train.sh

# Direct (with config override)
python -m sr_project.train --config configs/sr_config.yaml

# Smoke test (1 epoch, small subset — verifies setup)
python -m sr_project.train --config configs/sr_config.yaml --smoke_test

# Resume from checkpoint
python -m sr_project.train \
  --config configs/sr_config.yaml \
  --resume checkpoints/sr/sr_latest.pth

# Background execution (GPU server)
nohup bash scripts/run_sr_train.sh > logs/sr_train.log 2>&1 &
```

#### Segmentation (SegFormer)

```bash
bash scripts/run_tb_train.sh

# Or directly
python -m tb_project.train --config configs/tb_config.yaml

# Smoke test
python -m tb_project.train --config configs/tb_config.yaml --smoke_test
```

### 9.2 Evaluation

```bash
# SR — PSNR/SSIM vs bicubic baseline
python -m sr_project.evaluate \
  --config configs/sr_config.yaml \
  --checkpoint checkpoints/sr/best_model.pth

# Segmentation — Dice/IoU/Precision/Recall/F1
python -m tb_project.evaluate \
  --config configs/tb_config.yaml \
  --checkpoint checkpoints/tb/best_model.pth
```

### 9.3 Inference

#### Super-Resolution

```bash
# Single image
python -m sr_project.inference \
  --config configs/sr_config.yaml \
  --checkpoint checkpoints/sr/best_model.pth \
  --image path/to/image.bmp \
  --output output_sr.png

# Folder of images
python -m sr_project.inference \
  --config configs/sr_config.yaml \
  --checkpoint checkpoints/sr/best_model.pth \
  --folder path/to/images/ \
  --output_dir outputs/sr/

# Tiled inference for large images
python -m sr_project.inference \
  --config configs/sr_config.yaml \
  --checkpoint checkpoints/sr/best_model.pth \
  --image large_image.bmp \
  --output output.png \
  --tiled
```

#### Segmentation + Bacilli Counting

```bash
python -m tb_project.inference \
  --config configs/tb_config.yaml \
  --checkpoint checkpoints/tb/best_model.pth \
  --folder path/to/images/ \
  --output_dir outputs/tb/
```

Outputs: predicted masks, overlay images with bounding boxes, `counts.csv`.

#### Combined SR → Segmentation

```bash
bash scripts/run_combined_inference.sh

# Or directly
python -m pipelines.combined_inference \
  --config configs/pipeline_config.yaml
```

---

## 10. Monitoring & Logging

| Output | Path | Format |
|---|---|---|
| TensorBoard (SR) | `logs/sr/` | TensorBoard events |
| TensorBoard (TB) | `logs/tb/` | TensorBoard events |
| SR metrics | `logs/sr_metrics.jsonl` | JSON Lines |
| TB metrics | `logs/tb_metrics.jsonl` | JSON Lines |
| Best SR checkpoint | `checkpoints/sr/best_model.pth` | PyTorch state dict |
| Best TB checkpoint | `checkpoints/tb/best_model.pth` | PyTorch state dict |

```bash
# Launch TensorBoard
tensorboard --logdir logs/
```

---

## 11. Testing

```bash
# Run all tests
pytest

# Individual test files
pytest tests/test_sr_dataset.py
pytest tests/test_tb_dataset.py
pytest tests/test_model_forward.py

# Filter by name
pytest tests/test_model_forward.py -k "SwinIR"

# Verbose output
pytest -v
```

---

## 12. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Staged SR finetuning** | L1-only for 30 epochs ensures numerically stable start; adding VGG perceptual loss later improves texture quality without divergence |
| **Differential LR for SegFormer** | Backbone (MIT-B4) is pretrained on ImageNet; using 0.1× LR prevents overwriting useful representations while the MLP decoder learns at full speed |
| **`pos_weight=5.0` in BCE** | Bacilli pixels are a small fraction of each image; heavily penalising false negatives forces the model to detect all bacilli |
| **`WeightedRandomSampler`** | Balances subfolder sampling so the `50% NEGATIVE` (most positive) images are seen proportionally more during training |
| **Tiled inference with overlap** | Large microscopy images (~2448×2048) exceed GPU memory; 256×256 tiles with 32 px overlap + blending avoid tile boundary artefacts |
| **Connected-component post-processing** | Converts soft predictions to discrete, countable detections; `min_area=10` removes single-pixel noise while preserving real bacilli |
| **AMP + DataParallel** | Mixed precision halves memory and speeds computation; DataParallel enables multi-GPU scaling with no code changes |
| **On-the-fly LR generation (SR)** | Avoids storing a separate LR dataset; any augmentation or scale change is immediately reflected during training |
| **Experiment runner with timestamped IDs** | Reproducible, isolated runs; config snapshot ensures experiments can be exactly reproduced |
| **Environment-variable interpolation in YAML** | `${VAR:default}` syntax allows the same configs to run locally and on GPU clusters without editing files |
