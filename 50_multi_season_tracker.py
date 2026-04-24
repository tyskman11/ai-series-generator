#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

from pipeline_common import (
    headline,
    ok,
    info,
    load_config,
    resolve_project_path,
    read_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track continuity across multiple seasons."
    )
    parser.add_argument(
        "--season", type=int,
        help="Season number to analyze."
    )
    parser.add_argument(
        "--track", default="characters",
        choices=["characters", "locations", "relationships"],
        help="What to track."
    )
    return parser.parse_args()


def collect_season_data(season: int = None) -> list[dict]:
    packages_dir = resolve_project_path("generation/final_episode_packages")
    episodes = []
    
    for ep_dir in packages_dir.iterdir():
        if not ep_dir.is_dir():
            continue
        
        voice_plan = list(ep_dir.glob("*_voice_plan.json"))
        if not voice_plan:
            continue
        
        data = read_json(voice_plan[0], {})
        ep_season = data.get("season", 0)
        
        if season is None or ep_season == season:
            episodes.append({
                "id": ep_dir.name,
                "season": ep_season,
                "data": data,
            })
    
    return sorted(episodes, key=lambda x: x["id"])


def track_characters_cross_seasons(episodes: list[dict]) -> dict:
    appearances = defaultdict(list)
    
    for ep in episodes:
        chars = set()
        for scene in ep["data"].get("scenes", []):
            for line in scene.get("lines", []):
                char = line.get("character", "")
                if char:
                    chars.add(char)
        
        for char in chars:
            appearances[char].append(ep["id"])
    
    return dict(appearances)


def track_relationships(episodes: list[dict]) -> dict:
    relationships = defaultdict(lambda: defaultdict(int))
    
    for ep in episodes:
        chars = set()
        for scene in ep["data"].get("scenes", []):
            for line in scene.get("lines", []):
                char = line.get("character", "")
                if char:
                    chars.add(char)
        
        chars = sorted(chars)
        for i, c1 in enumerate(chars):
            for c2 in chars[i+1:]:
                key = tuple(sorted([c1, c2]))
                relationships[key]["count"] += 1
                relationships[key]["episodes"].append(ep["id"])
    
    result = {}
    for key, val in relationships.items():
        result[f"{key[0]}-{key[1]}"] = val
    
    return result


def main() -> None:
    args = parse_args()
    headline("Multi-Season Continuity Tracker")
    cfg = load_config()
    
    episodes = collect_season_data(args.season)
    
    if not episodes:
        info("No episodes found")
        return
    
    if args.track == "characters":
        results = track_characters_cross_seasons(episodes)
    elif args.track == "relationships":
        results = track_relationships(episodes)
    else:
        results = {}
    
    output_path = resolve_project_path(f"generation/{args.track}_continuity.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    ok(f"Tracked {len(results)} {args.track} across {len(episodes)} episodes")


if __name__ == "__main__":
    main()