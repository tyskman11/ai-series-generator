#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "ai_series_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse
import json
from pathlib import Path

from support_scripts.pipeline_common import (
    headline,
    ok,
    info,
    load_config,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize training parameters for models."
    )
    parser.add_argument(
        "--model-type", default="all",
        choices=["all", "foundation", "adapter", "fine_tune", "voice", "video", "image"],
        help="Model type to optimize."
    )
    parser.add_argument(
        "--dataset-path",
        help="Training dataset path."
    )
    parser.add_argument(
        "--batch-size", type=int,
        help="Batch size."
    )
    parser.add_argument(
        "--learning-rate", type=float,
        help="Learning rate."
    )
    parser.add_argument(
        "--epochs", type=int,
        help="Number of epochs."
    )
    parser.add_argument(
        "--mixed-precision", action="store_true",
        help="Use mixed precision."
    )
    return parser.parse_args()


def optimize_image_training(cfg: dict) -> dict:
    return {
        "batch_size": cfg.get("batch_size", 4),
        "learning_rate": cfg.get("learning_rate", 1e-4),
        "epochs": cfg.get("epochs", 100),
        "mixed_precision": cfg.get("mixed_precision", True),
        "optimizer": "adamw",
        "scheduler": "cosine",
    }


def optimize_voice_training(cfg: dict) -> dict:
    return {
        "batch_size": cfg.get("batch_size", 8),
        "learning_rate": cfg.get("learning_rate", 3e-4),
        "epochs": cfg.get("epochs", 50),
        "mixed_precision": cfg.get("mixed_precision", True),
        "use_amp": True,
    }


def optimize_video_training(cfg: dict) -> dict:
    return {
        "batch_size": cfg.get("batch_size", 2),
        "learning_rate": cfg.get("learning_rate", 1e-5),
        "epochs": cfg.get("epochs", 30),
        "frame_skip": 2,
    }


def main() -> None:
    args = parse_args()
    headline("Training Optimizer")
    cfg = load_config()
    
    results = {}
    
    if args.model_type in ["all", "image"]:
        results["image"] = optimize_image_training(cfg)
    
    if args.model_type in ["all", "voice"]:
        results["voice"] = optimize_voice_training(cfg)
    
    if args.model_type in ["all", "video"]:
        results["video"] = optimize_video_training(cfg)
    
    output_path = resolve_project_path("generation/training_optimization.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    ok(f"Optimized {len(results)} model types")


if __name__ == "__main__":
    main()
