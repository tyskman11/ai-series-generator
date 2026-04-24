#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_common import (
    headline,
    info,
    error,
    load_config,
    episode_quality_assessment,
    clamp_quality_score,
    generated_episode_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality gate check before release.")
    parser.add_argument("--episode-id", help="Target episode ID.")
    parser.add_argument("--min-quality", type=float, default=0.68, help="Minimum quality threshold.")
    parser.add_argument("--max-weak-scenes", type=int, default=2, help="Maximum weak scenes allowed.")
    parser.add_argument("--strict", action="store_true", help="Fail on any warning.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    headline("Quality Gate Check")
    cfg = load_config()
    
    episode_id = args.episode_id or "latest"
    artifacts = generated_episode_artifacts(cfg, episode_id)
    
    if not artifacts:
        error(f"No artifacts found for {episode_id}")
        return
    
    quality = artifacts.get("quality_score", 0.0)
    quality_percent = int(quality * 100)
    weak_scenes = artifacts.get("scenes_below_release_threshold", 0)
    
    print(f"Episode: {episode_id}")
    print(f"Quality: {quality_percent}%")
    print(f"Weak scenes: {weak_scenes}/{args.max_weak_scenes}")
    
    passed = True
    
    if quality < args.min_quality:
        error(f"Quality {quality_percent}% below minimum {int(args.min_quality * 100)}%")
        passed = False
    
    if weak_scenes > args.max_weak_scenes:
        error(f"Too many weak scenes: {weak_scenes} > {args.max_weak_scenes}")
        passed = False
    
    if passed:
        info("QUALITY GATE PASSED")
    else:
        error("QUALITY GATE FAILED")
    
    return


if __name__ == "__main__":
    main()