"""
Experiment runner — config-driven experiment management with run IDs.
"""

import argparse
import logging
import os
import shutil
from datetime import datetime

import yaml

from sr_project.utils import load_config

logger = logging.getLogger(__name__)


def run_experiment(config_path: str, experiment_type: str = "sr"):
    """Launch a tracked experiment run."""
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(config_path)

    # Generate run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", f"{experiment_type}_{run_id}")
    os.makedirs(exp_dir, exist_ok=True)

    # Snapshot config
    shutil.copy2(config_path, os.path.join(exp_dir, "config.yaml"))
    logger.info(f"Experiment {run_id} — type: {experiment_type}, dir: {exp_dir}")

    # Override log/checkpoint dirs to experiment dir
    cfg["logging"]["dir"] = os.path.join(exp_dir, "logs")
    cfg["logging"]["run_id"] = run_id
    cfg["checkpoint"]["dir"] = os.path.join(exp_dir, "checkpoints")

    if experiment_type == "sr":
        from sr_project.train import train
        train(cfg)
    elif experiment_type == "tb":
        from tb_project.train import train
        train(cfg)
    elif experiment_type == "pipeline":
        from pipelines.combined_inference import run_combined_pipeline
        cfg["output"]["dir"] = exp_dir
        run_combined_pipeline(cfg)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    logger.info(f"Experiment {run_id} complete. Results in {exp_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run tracked experiment")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--type", type=str, choices=["sr", "tb", "pipeline"], required=True)
    args = parser.parse_args()
    run_experiment(args.config, args.type)


if __name__ == "__main__":
    main()
