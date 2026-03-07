"""
Full Pipeline — Single-file runner.

Usage:
    python run_pipeline.py

What it does (fully automatic):
  1. Discovers testingDataset/Positive and testingDataset/Negative images
  2. Runs SwinIR 4× super-resolution on every image
  3. Saves SR images to testingDataset_SR/Positive and testingDataset_SR/Negative
  4. Runs SegFormer B4 segmentation on the SR images
  5. Detects & counts TB bacilli via connected-component analysis
  6. Saves all results (masks, overlays, counts, summary) in JSON format

All outputs go into full_pipeline/pipeline_outputs/
"""
