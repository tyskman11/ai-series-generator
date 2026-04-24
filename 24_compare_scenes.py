#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_common import (
    headline,
    info,
    load_config,
    resolve_project_path,
    read_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two scenes visually.")
    parser.add_argument("--scene-a", required=True, help="First scene ID.")
    parser.add_argument("--scene-b", required=True, help="Second scene ID.")
    parser.add_argument("--episode-id", help="Episode containing scenes.")
    return parser.parse_args()


def compare_scenes(scene_a_path: Path, scene_b_path: Path) -> dict[str, Any]:
    result = {
        "scene_a": str(scene_a_path),
        "scene_b": str(scene_b_path),
        "both_exist": scene_a_path.exists() and scene_b_path.exists(),
        "size_a": scene_a_path.stat().st_size if scene_a_path.exists() else 0,
        "size_b": scene_b_path.stat().st_size if scene_b_path.exists() else 0,
        "similar": False,
    }
    
    if result["both_exist"] and result["size_a"] > 0:
        size_ratio = min(result["size_a"], result["size_b"]) / max(result["size_a"], result["size_b"])
        result["similar"] = size_ratio > 0.9
        result["size_ratio"] = round(size_ratio, 3)
    
    return result


def main() -> None:
    args = parse_args()
    headline("Compare Scenes")
    cfg = load_config()
    
    package_root = resolve_project_path("generation/final_episode_packages")
    episode_dir = package_root / (args.episode_id or "latest")
    
    scene_a_path = episode_dir / "scenes" / f"{args.scene_a}.png"
    scene_b_path = episode_dir / "scenes" / f"{args.scene_b}.png"
    
    comparison = compare_scenes(scene_a_path, scene_b_path)
    
    print(f"Scene A: {comparison['scene_a']}")
    print(f"Scene B: {comparison['scene_b']}")
    print(f"Both exist: {comparison['both_exist']}")
    print(f"Size ratio: {comparison.get('size_ratio', 0)}")
    print(f"Similar: {comparison['similar']}")


if __name__ == "__main__":
    main()