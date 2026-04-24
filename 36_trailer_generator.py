#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
        description="Generate trailer from episode footage."
    )
    parser.add_argument("--episode-id", help="Source episode ID.")
    parser.add_argument(
        "--season", help="Season number for multi-episode trailer."
    )
    parser.add_argument(
        "--duration", type=float, default=90.0,
        help="Trailer duration in seconds."
    )
    parser.add_argument(
        "--style", default="action",
        choices=["action", "drama", "comedy", "teaser", "finale"],
        help="Trailer style."
    )
    return parser.parse_args()


STYLE_MARKERS = {
    "action": ["fight", "explosion", "chase", "car", "gun"],
    "drama": ["confession", "love", "tears", "alone", "silence"],
    "comedy": ["joke", "laugh", "funny", "mistake", "awkward"],
    "teaser": ["mystery", "question", "reveal", "hint"],
    "finale": ["final", "ending", "conclusion", "last"],
}


def select_trailer_clips(voice_plan: dict, duration: float, style: str) -> list[dict]:
    clips = []
    total_duration = 0.0
    
    for scene in voice_plan.get("scenes", []):
        scene_duration = scene.get("duration_seconds", 0)
        scene_text = " ".join(
            line.get("text", "") for line in scene.get("lines", [])
        ).lower()
        
        style_match = any(
            marker in scene_text for marker in STYLE_MARKERS.get(style, [])
        )
        
        priority = 2 if style_match else 1
        
        clips.append({
            "scene_id": scene.get("scene_id", ""),
            "start_time": scene.get("start_time_seconds", 0),
            "duration": scene_duration,
            "priority": priority,
            "type": scene.get("scene_type", "dialogue"),
        })
        
        total_duration += scene_duration
        
        if total_duration >= duration:
            break
    
    clips.sort(key=lambda x: (-x["priority"], x["start_time"]))
    
    selected = []
    total = 0.0
    for clip in clips:
        if total + clip["duration"] <= duration * 1.2:
            selected.append(clip)
            total += clip["duration"]
    
    return selected[:10]


def generate_trailer_metadata(
    clips: list[dict],
    episode_id: str,
    duration: float,
    style: str
) -> dict:
    return {
        "trailer_id": f"trailer_{episode_id}_{style}",
        "source_episode": episode_id,
        "style": style,
        "target_duration": duration,
        "actual_duration": sum(c["duration"] for c in clips),
        "clip_count": len(clips),
        "clips": [
            {
                "scene_id": c["scene_id"],
                "start_time": c["start_time"],
                "duration": c["duration"],
            }
            for c in clips
        ],
    }


def main() -> None:
    args = parse_args()
    headline("Trailer Generator")
    cfg = load_config()
    
    package_root = resolve_project_path(f"generation/final_episode_packages/{args.episode_id or 'latest'}")
    voice_plan_path = package_root / f"{args.episode_id or 'latest'}_voice_plan.json"
    
    if not voice_plan_path.exists():
        info("No voice plan found")
        return
    
    voice_plan = read_json(voice_plan_path, {})
    clips = select_trailer_clips(voice_plan, args.duration, args.style)
    
    metadata = generate_trailer_metadata(clips, args.episode_id or "latest", args.duration, args.style)
    
    output_path = package_root / f"trailer_{args.style}.json"
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    
    ok(f"Generated {args.style} trailer with {len(clips)} clips")


if __name__ == "__main__":
    main()