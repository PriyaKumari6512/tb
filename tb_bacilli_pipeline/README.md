# TB Bacilli Imaging Pipeline

End-to-end pipeline for TB bacilli detection in microscopy images:
1. **Super-Resolution (SwinIR)** — enhance low-resolution microscopy images (4× upscaling)
2. **Segmentation (SegFormer)** — detect and count TB bacilli using semantic segmentation

## Project Structure

```
tb_bacilli_pipeline/
├── configs/                     # YAML configuration files
│   ├── sr_config.yaml           # Super-resolution training config
│   ├── tb_config.yaml           # Segmentation training config
│   └── pipeline_config.yaml     # Combined pipeline config
├── sr_project/                  # Super-resolution module
│   ├── model.py                 # SwinIR architecture
│   ├── dataset.py               # LR/HR pair dataset with on-the-fly downsampling
│   ├── train.py                 # Training loop (AMP, multi-GPU, staged finetuning)
│   ├── evaluate.py              # PSNR/SSIM evaluation + visual grids
│   ├── inference.py             # Single/folder/tiled inference
│   └── utils.py                 # Shared utilities
├── tb_project/                  # Segmentation module
│   ├── model.py                 # SegFormer (MIT-B4) wrapper + losses
│   ├── dataset.py               # Image/mask dataset with augmentations
│   ├── train.py                 # Training with differential LR
│   ├── evaluate.py              # Dice/IoU/P/R/F1 evaluation
│   ├── inference.py             # Inference with bacilli counting
│   ├── postprocess.py           # Connected-component analysis
│   └── utils.py                 # Segmentation metrics & visualization
├── pipelines/                   # Combined pipelines
│   ├── combined_inference.py    # SR→Segmentation comparison pipeline
│   ├── experiment_runner.py     # Config-driven experiment launcher
│   └── report.py                # Report generation
├── scripts/                     # Shell scripts
│   ├── run_sr_train.sh          # SR training wrapper
│   ├── run_tb_train.sh          # TB training wrapper
│   ├── run_combined_inference.sh
│   ├── prepare_data.sh          # Data verification
│   └── check_status.sh          # Environment check
├── tests/                       # Unit tests
├── requirements.txt
├── pytest.ini
└── .env.example
```

## Setup

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env to set DATA_ROOT, CKPT_DIR, etc.
```

### 3. Verify Data

```bash
bash scripts/prepare_data.sh
bash scripts/check_status.sh
```

The DDS3 dataset should be at `DATA_ROOT` (default `./DDS3/`) with this structure:
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
│       ├── IMAGE/   (1000 images)
│       └── MASK/
└── TEST SET/
    └── 50% NEGATIVE/
        ├── IMAGE/   (1000 images)
        └── MASK/
```

## Training

### Super-Resolution

```bash
# Standard training
bash scripts/run_sr_train.sh

# With overrides
python -m sr_project.train --config configs/sr_config.yaml

# Smoke test (1 epoch, small subset)
python -m sr_project.train --config configs/sr_config.yaml --smoke_test

# Resume from checkpoint
python -m sr_project.train --config configs/sr_config.yaml --resume checkpoints/sr_latest.pth

# Background (GPU server)
nohup bash scripts/run_sr_train.sh > logs/sr_train.log 2>&1 &
```

**Key parameters** (in `configs/sr_config.yaml`):
- `embed_dim: 180`, 8 RSTB blocks, window size 8
- Staged finetuning: L1 loss only → L1 + Perceptual (VGG-19) after epoch 30
- Cosine LR with linear warmup, AdamW, AMP enabled
- Batch size 16, patch size 64, 200 epochs

### Segmentation

```bash
bash scripts/run_tb_train.sh

# Or directly
python -m tb_project.train --config configs/tb_config.yaml

# Smoke test
python -m tb_project.train --config configs/tb_config.yaml --smoke_test
```

**Key parameters** (in `configs/tb_config.yaml`):
- SegFormer MIT-B4 backbone (ImageNet pretrained)
- Dice + weighted BCE loss (pos_weight 5.0 for class imbalance)
- Differential LR: backbone 0.1×, decode head 1×
- WeightedRandomSampler to balance subfolder sampling
- Batch size 8, image size 512, 100 epochs

## Evaluation

```bash
# SR evaluation with bicubic baseline comparison
python -m sr_project.evaluate \
  --config configs/sr_config.yaml \
  --checkpoint checkpoints/sr_best.pth

# TB segmentation evaluation
python -m tb_project.evaluate \
  --config configs/tb_config.yaml \
  --checkpoint checkpoints/tb_best.pth
```

## Inference

### Super-Resolution

```bash
# Single image
python -m sr_project.inference \
  --config configs/sr_config.yaml \
  --checkpoint checkpoints/sr_best.pth \
  --image path/to/image.bmp \
  --output output_sr.png

# Folder of images
python -m sr_project.inference \
  --config configs/sr_config.yaml \
  --checkpoint checkpoints/sr_best.pth \
  --folder path/to/images/ \
  --output_dir outputs/sr/

# Tiled inference (for large images)
python -m sr_project.inference \
  --config configs/sr_config.yaml \
  --checkpoint checkpoints/sr_best.pth \
  --image large_image.bmp \
  --output output.png \
  --tiled
```

### Segmentation + Bacilli Counting

```bash
python -m tb_project.inference \
  --config configs/tb_config.yaml \
  --checkpoint checkpoints/tb_best.pth \
  --folder path/to/images/ \
  --output_dir outputs/tb/
```

Outputs: predicted masks, overlay images with bounding boxes, `counts.csv` with per-image bacilli counts.

### Combined SR → Segmentation Pipeline

```bash
bash scripts/run_combined_inference.sh

# Or directly
python -m pipelines.combined_inference \
  --config configs/pipeline_config.yaml
```

Compares segmentation results on original vs. SR-enhanced images, producing side-by-side comparisons and a summary report.

## Testing

```bash
pytest                          # All tests
pytest tests/test_sr_dataset.py # SR dataset tests only
pytest tests/test_model_forward.py -k "SwinIR"  # SwinIR forward pass only
pytest -v                       # Verbose output
```

## Experiment Management

```bash
python -m pipelines.experiment_runner \
  --config configs/sr_config.yaml \
  --name "sr_embed180_ep200"
```

Each experiment run gets a unique ID, config snapshot, and organized output directory under `experiments/`.

## Architecture Details

### SwinIR (Super-Resolution)
- Shifted Window Transformer with residual connections
- 8 RSTB (Residual Swin Transformer Block) layers
- PixelShuffle 4× upsampling
- ~12M parameters

### SegFormer (Segmentation)
- MIT-B4 hierarchical encoder (ImageNet pretrained)
- Lightweight MLP decoder head
- 2-class output (background + bacilli)
- Post-processing: connected-component analysis for bacilli detection

## Monitoring

Training logs are written to:
- **TensorBoard**: `tensorboard --logdir logs/`
- **JSONL metrics**: `logs/sr_metrics.jsonl`, `logs/tb_metrics.jsonl`
- **Checkpoints**: `checkpoints/sr_best.pth`, `checkpoints/tb_best.pth`
