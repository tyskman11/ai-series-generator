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
from collections import defaultdict

from support_scripts.pipeline_common import (
    headline,
    ok,
    load_config,
    resolve_project_path,
    read_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track character outfits across episodes."
    )
    parser.add_argument(
        "--character",
        help="Character name to track."
    )
    parser.add_argument(
        "--season", type=int,
        help="Season to analyze."
    )
    return parser.parse_args()


def collect_character_outfits(episodes: list[dict], character: str) -> dict:
    outfits = defaultdict(list)
    
    for ep in episodes:
        for scene in ep["data"].get("scenes", []):
            for line in scene.get("lines", []):
                if line.get("character") == character:
                    outfit = line.get("outfit", "unknown")
                    if outfit != "unknown":
                        outfits[outfit].append({
                            "episode": ep["id"],
                            "scene": scene.get("scene_id", ""),
                        })
    
    return dict(outfits)


def main() -> None:
    args = parse_args()
    headline("Character Outfit Tracker")
    cfg = load_config()
    
    packages_dir = resolve_project_path("generation/final_episode_packages")
    episodes = []
    
    for ep_dir in packages_dir.iterdir():
        if not ep_dir.is_dir():
            continue
        
        voice_plan = list(ep_dir.glob("*_voice_plan.json"))
        if not voice_plan:
            continue
        
        data = read_json(voice_plan[0], {})
        if args.season and data.get("season") != args.season:
            continue
        episodes.append({"id": ep_dir.name, "data": data})
    
    if args.character:
        outfits = collect_character_outfits(episodes, args.character)
        output_path = resolve_project_path(f"generation/outfits_{args.character}.json")
        output_path.write_text(json.dumps(outfits, indent=2), encoding="utf-8")
        ok(f"Tracked {len(outfits)} outfits for {args.character}")
    else:
        all_chars = set()
        for ep in episodes:
            for scene in ep["data"].get("scenes", []):
                for line in scene.get("lines", []):
                    all_chars.add(line.get("character", ""))
        
        for char in sorted(all_chars)[:5]:
            outfits = collect_character_outfits(episodes, char)
            print(f"{char}: {len(outfits)} outfits")


if __name__ == "__main__":
    main()
