"""
Report generation utilities for combined pipeline results.
"""

import os
from typing import Dict, List, Optional

import pandas as pd


def generate_comparison_report(
    results: List[Dict],
    output_dir: str,
    run_id: str,
) -> str:
    """Generate CSV and text summary from comparison results."""
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "comparison_report.csv")
    df.to_csv(csv_path, index=False)

    summary = []
    summary.append(f"Run: {run_id}")
    summary.append(f"Total images: {len(df)}")

    if "count_original" in df.columns:
        summary.append(f"Original — Mean count: {df['count_original'].mean():.2f}")
    if "count_sr" in df.columns:
        sr = pd.to_numeric(df["count_sr"], errors="coerce")
        summary.append(f"SR — Mean count: {sr.mean():.2f}")
    if "dice_original" in df.columns:
        summary.append(f"Original — Mean Dice: {df['dice_original'].mean():.4f}")
    if "dice_sr" in df.columns:
        sr_dice = pd.to_numeric(df["dice_sr"], errors="coerce")
        summary.append(f"SR — Mean Dice: {sr_dice.mean():.4f}")

    summary_text = "\n".join(summary)
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary_text)

    return summary_text
