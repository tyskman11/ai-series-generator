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
    read_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate recap episode from season episodes."
    )
    parser.add_argument(
        "--season", type=int, required=True,
        help="Season number."
    )
    parser.add_argument(
        "--max-duration", type=float, default=3600.0,
        help="Max recap duration in seconds."
    )
    parser.add_argument(
        "--format", default="season",
        choices=["season", "annual", "highlights"],
        help="Recap format."
    )
    parser.add_argument(
        "--top-scenes", type=int, default=20,
        help="Number of key scenes to include."
    )
    return parser.parse_args()


def collect_season_episodes(season: int) -> list[dict]:
    packages_dir = resolve_project_path("generation/final_episode_packages")
    
    episodes = []
    for ep_dir in packages_dir.iterdir():
        if not ep_dir.is_dir():
            continue
        
        manifest_file = list(ep_dir.glob("*_production_package.json"))
        if not manifest_file:
            continue
        
        try:
            manifest = read_json(manifest_file[0], {})
            ep_season = manifest.get("season", int(ep_dir.name.split("_")[0].replace("S", "")))
            
            if ep_season == season:
                episodes.append({
                    "episode_id": ep_dir.name,
                    "season": ep_season,
                    "scene_count": manifest.get("scene_count", 0),
                })
        except Exception:
            continue
    
    return sorted(episodes, key=lambda x: x["episode_id"])


def extract_key_moments(episodes: list[dict], top_n: int) -> list[dict]:
    moments = []
    
    for ep in episodes[:min(5, len(episodes))]:
        package_root = resolve_project_path(f"generation/final_episode_packages/{ep['episode_id']}")
        voice_plan_path = package_root / f"{ep['episode_id']}_voice_plan.json"
        
        if not voice_plan_path.exists():
            continue
        
        voice_plan = read_json(voice_plan_path, {})
        
        for scene in voice_plan.get("scenes", [])[:top_n // 5]:
            lines = scene.get("lines", [])
            if not lines:
                continue
            
            text_preview = " ".join(line.get("text", "")[:100] for line in lines[:2])
            
            moments.append({
                "episode_id": ep["episode_id"],
                "scene_id": scene.get("scene_id", ""),
                "start_time": scene.get("start_time_seconds", 0),
                "duration": scene.get("duration_seconds", 0),
                "preview": text_preview[:150],
                "type": scene.get("scene_type", "dialogue"),
            })
    
    return moments[:top_n]


def generate_recap_metadata(
    episodes: list[dict],
    moments: list[dict],
    season: int,
    format: str
) -> dict:
    total_duration = sum(m["duration"] for m in moments)
    
    return {
        "recap_id": f"recap_S{season:02d}_{format}",
        "season": season,
        "format": format,
        "source_episodes": [ep["episode_id"] for ep in episodes],
        "episode_count": len(episodes),
        "moment_count": len(moments),
        "total_duration": total_duration,
        "moments": moments,
    }


def main() -> None:
    args = parse_args()
    headline("Recap Generator")
    cfg = load_config()
    
    episodes = collect_season_episodes(args.season)
    
    if not episodes:
        info(f"No episodes found for season {args.season}")
        return
    
    moments = extract_key_moments(episodes, args.top_scenes)
    
    metadata = generate_recap_metadata(episodes, moments, args.season, args.format)
    
    output_path = resolve_project_path("generation/final_episode_packages") / f"recap_S{args.season:02d}_{args.format}.json"
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    
    ok(f"Generated recap for S{args.season:02d} with {len(moments)} key moments")


if __name__ == "__main__":
    main()
