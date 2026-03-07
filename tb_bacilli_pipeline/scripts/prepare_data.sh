#!/usr/bin/env bash
# =============================================================================
# Data Preparation & Verification
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-./DDS3}"

echo "=== Data Verification ==="

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: $DATA_ROOT not found. Please set DATA_ROOT."
    exit 1
fi

echo ""
echo "--- Dataset Summary ---"
total=0
for split_dir in "$DATA_ROOT"/*; do
    [ -d "$split_dir" ] || continue
    split_name=$(basename "$split_dir")
    echo ""
    echo "  [$split_name]"
    for subfolder in "$split_dir"/*; do
        [ -d "$subfolder" ] || continue
        sub_name=$(basename "$subfolder")
        img_dir="$subfolder/IMAGE"
        mask_dir="$subfolder/MASK"
        if [ -d "$img_dir" ]; then
            n_img=$(find "$img_dir" -name "*.bmp" | wc -l | tr -d ' ')
        else
            n_img=0
        fi
        if [ -d "$mask_dir" ]; then
            n_mask=$(find "$mask_dir" -name "*.bmp" | wc -l | tr -d ' ')
        else
            n_mask=0
        fi
        echo "    $sub_name: $n_img images, $n_mask masks"
        total=$((total + n_img))
    done
done
echo ""
echo "  Total images: $total"

# Verify image/mask pairing
echo ""
echo "--- Pair Verification ---"
missing=0
for img_path in $(find "$DATA_ROOT" -path "*/IMAGE/*.bmp" 2>/dev/null | head -100); do
    mask_path=$(echo "$img_path" | sed 's|/IMAGE/|/MASK/|')
    if [ ! -f "$mask_path" ]; then
        echo "  MISSING MASK: $mask_path"
        missing=$((missing + 1))
    fi
done
if [ $missing -eq 0 ]; then
    echo "  All sampled image-mask pairs verified OK"
else
    echo "  WARNING: $missing missing masks found (checked first 100)"
fi

# Create output directories
echo ""
echo "--- Creating directories ---"
for dir in outputs logs checkpoints experiments; do
    mkdir -p "$dir"
    echo "  Created $dir/"
done

echo ""
echo "=== Done ==="
