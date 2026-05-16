#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse
import json
from pathlib import Path

from support_scripts.pipeline_common import (
    headline,
    info,
    ok,
    load_config,
    resolve_project_path,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple short episodes into one season."
    )
    parser.add_argument(
        "--episodes", required=True, nargs="+",
        help="Episode IDs to merge."
    )
    parser.add_argument(
        "--output", required=True,
        help="Output episode ID."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    headline("Merge Episodes")
    cfg = load_config()
    
    all_scenes = []
    scene_index = 1
    
    for ep_id in args.episodes:
        package_root = resolve_project_path(f"generation/final_episode_packages/{ep_id}")
        manifest_path = package_root / f"{ep_id}_production_package.json"
        
        if not manifest_path.exists():
            info(f"Skipping {ep_id} (not found)")
            continue
        
        manifest = read_json(manifest_path, {})
        
        for scene in manifest.get("scenes", []):
            scene["scene_id"] = f"scene_{scene_index:04d}"
            scene["original_episode"] = ep_id
            all_scenes.append(scene)
            scene_index += 1
    
    merged = {
        "episode_id": args.output,
        "source_episodes": args.episodes,
        "scene_count": len(all_scenes),
        "scenes": all_scenes,
    }
    
    output_path = resolve_project_path("generation/final_episode_packages") / args.output / f"{args.output}_production_package.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, merged)
    
    ok(f"Merged {len(args.episodes)} episodes into {args.output} ({len(all_scenes)} scenes)")


if __name__ == "__main__":
    main()
