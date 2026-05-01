#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from support_scripts.pipeline_common import (
    headline,
    load_config,
    resolve_project_path,
    read_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show character appearance timeline.")
    parser.add_argument("--character", required=True, help="Character name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    headline(f"Character Timeline: {args.character}")
    cfg = load_config()
    
    timeline = defaultdict(list)
    
    linked_dir = resolve_project_path("data/processed/linked_segments")
    for ep_file in linked_dir.glob("*.json"):
        episode_data = read_json(ep_file, {})
        episode_id = ep_file.stem
        for segment in episode_data.get("segments", []):
            if segment.get("speaker_name") == args.character:
                timeline[episode_id].append({
                    "scene": segment.get("scene_id"),
                    "text": segment.get("text", "")[:80],
                })
    
    if not timeline:
        print(f"No appearances found for {args.character}")
        return
    
    for episode_id, appearances in sorted(timeline.items()):
        print(f"\n--- {episode_id} ---")
        for app in appearances:
            print(f"  {app.get('scene')}: {app.get('text')}")


if __name__ == "__main__":
    main()
