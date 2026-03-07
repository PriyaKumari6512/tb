#!/usr/bin/env bash
# =============================================================================
# Environment & Status Check
# =============================================================================
cd "$(dirname "$0")/.."

echo "============================================"
echo "  TB Bacilli Pipeline — Environment Check"
echo "============================================"
echo ""

echo "--- System ---"
echo "  Date:     $(date)"
echo "  Hostname: $(hostname)"
echo "  OS:       $(uname -s) $(uname -r)"
echo ""

echo "--- Python ---"
python3 --version 2>/dev/null || echo "  Python3 not found!"
echo ""

echo "--- PyTorch & CUDA ---"
python3 -c "
import torch
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA avail:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {p.name} ({p.total_mem / 1e9:.1f} GB)')
else:
    print('  No GPU detected')
" 2>/dev/null || echo "  PyTorch not installed"
echo ""

echo "--- Key Packages ---"
python3 -c "
import importlib
for pkg in ['torchvision', 'transformers', 'timm', 'albumentations', 'cv2', 'skimage']:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, '__version__', 'ok')
        print(f'  {pkg}: {v}')
    except ImportError:
        print(f'  {pkg}: NOT INSTALLED')
" 2>/dev/null
echo ""

echo "--- Data Directory ---"
DATA_ROOT="${DATA_ROOT:-./DDS3}"
if [ -d "$DATA_ROOT" ]; then
    echo "  $DATA_ROOT exists"
    for split in "TRAINING SET" "VALIDATION SET" "TEST SET"; do
        if [ -d "$DATA_ROOT/$split" ]; then
            count=$(find "$DATA_ROOT/$split" -name "*.bmp" 2>/dev/null | wc -l)
            echo "  $split: $count .bmp files"
        fi
    done
else
    echo "  $DATA_ROOT NOT FOUND — set DATA_ROOT env var"
fi
echo ""

echo "--- Checkpoints ---"
CKPT_DIR="${CKPT_DIR:-./checkpoints}"
if [ -d "$CKPT_DIR" ]; then
    find "$CKPT_DIR" -name "*.pth" -exec echo "  {}" \;
else
    echo "  No checkpoints found"
fi
echo ""
echo "============================================"
