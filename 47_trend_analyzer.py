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
        description="Analyze series trends across episodes."
    )
    parser.add_argument(
        "--season", type=int,
        help="Analyze specific season."
    )
    parser.add_argument(
        "--metrics", nargs="+",
        default=["characters", "locations", "themes"],
        help="Metrics to analyze."
    )
    return parser.parse_args()


def collect_episodes(season: int = None) -> list[dict]:
    packages_dir = resolve_project_path("generation/final_episode_packages")
    episodes = []
    
    for ep_dir in packages_dir.iterdir():
        if not ep_dir.is_dir():
            continue
        
        voice_plan = list(ep_dir.glob("*_voice_plan.json"))
        if not voice_plan:
            continue
        
        try:
            data = read_json(voice_plan[0], {})
            ep_season = data.get("season", 0)
            
            if season is None or ep_season == season:
                episodes.append({
                    "id": ep_dir.name,
                    "season": ep_season,
                    "data": data,
                })
        except Exception:
            continue
    
    return sorted(episodes, key=lambda x: x["id"])


def analyze_character_trends(episodes: list[dict]) -> dict:
    appearances = defaultdict(int)
    pairs = defaultdict(int)
    
    for ep in episodes:
        characters = set()
        
        for scene in ep["data"].get("scenes", []):
            for line in scene.get("lines", []):
                char = line.get("character", "")
                if char:
                    characters.add(char)
                    appearances[char] += 1
        
        chars = sorted(characters)
        for i, c1 in enumerate(chars):
            for c2 in chars[i+1:]:
                key = tuple(sorted([c1, c2]))
                pairs[key] += 1
    
    return {
        "total_characters": len(appearances),
        "appearances": dict(appearances),
        "pair_cooccurrences": {f"{k[0]}-{k[1]}": v for k, v in pairs.items()},
    }


def analyze_location_trends(episodes: list[dict]) -> dict:
    locations = defaultdict(int)
    
    for ep in episodes:
        for scene in ep["data"].get("scenes", []):
            loc = scene.get("location", "")
            if loc:
                locations[loc] += 1
    
    return {"locations": dict(locations)}


def analyze_theme_trends(episodes: list[dict]) -> dict:
    themes = defaultdict(int)
    
    theme_keywords = {
        "romance": ["love", "kiss", "heart", "relationship"],
        "conflict": ["fight", "enemy", "battle", "war"],
        "mystery": ["secret", "mystery", "strange", "clue"],
        "family": ["family", "parent", "child", "brother"],
        "money": ["money", "rich", "pay", "deal"],
    }
    
    for ep in episodes:
        text_all = ""
        for scene in ep["data"].get("scenes", []):
            for line in scene.get("lines", []):
                text_all += " " + line.get("text", "").lower()
        
        for theme, keywords in theme_keywords.items():
            if any(kw in text_all for kw in keywords):
                themes[theme] += 1
    
    return {"themes": dict(themes)}


def main() -> None:
    args = parse_args()
    headline("Trend Analyzer")
    cfg = load_config()
    
    episodes = collect_episodes(args.season)
    
    if not episodes:
        info("No episodes found")
        return
    
    results = {
        "season": args.season,
        "episode_count": len(episodes),
    }
    
    if "characters" in args.metrics:
        results["characters"] = analyze_character_trends(episodes)
    
    if "locations" in args.metrics:
        results["locations"] = analyze_location_trends(episodes)
    
    if "themes" in args.metrics:
        results["themes"] = analyze_theme_trends(episodes)
    
    output_path = resolve_project_path("generation/series_trends.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    ok(f"Analyzed {len(episodes)} episodes")


if __name__ == "__main__":
    main()