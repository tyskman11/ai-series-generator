#!/usr/bin/env python3
from __future__ import annotations

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
        description="Generate social media clips (TikTok, Reels, Shorts)."
    )
    parser.add_argument("--episode-id", help="Episode ID.")
    parser.add_argument(
        "--platform", default="tiktok",
        choices=["tiktok", "reels", "shorts"],
        help="Target platform."
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Max clip duration in seconds."
    )
    parser.add_argument(
        "--count", type=int, default=5,
        help="Number of clips to generate."
    )
    parser.add_argument(
        "--aspect", default="9:16",
        choices=["9:16", "1:1"],
        help="Aspect ratio."
    )
    return parser.parse_args()


PLATFORM_SETTINGS = {
    "tiktok": {"max_duration": 180, "aspect": "9:16", "hashtag": "#fyp #trending"},
    "reels": {"max_duration": 90, "aspect": "9:16", "hashtag": "#reels #viral"},
    "shorts": {"max_duration": 60, "aspect": "9:16", "hashtag": "#shorts #youtube"},
}


def extract_clip_segments(voice_plan: dict, max_duration: float) -> list[dict]:
    segments = []
    
    for scene in voice_plan.get("scenes", []):
        duration = scene.get("duration_seconds", 0)
        
        if duration > max_duration or duration < 5:
            continue
        
        lines = scene.get("lines", [])
        if not lines:
            continue
        
        text = " ".join(line.get("text", "")[:50] for line in lines[:3])
        
        segments.append({
            "scene_id": scene.get("scene_id", ""),
            "start_time": scene.get("start_time_seconds", 0),
            "duration": duration,
            "preview_text": text[:100],
        })
        
        if len(segments) >= 20:
            break
    
    return segments[:20]


def generate_clips(segments: list[dict], platform: str, count: int, aspect: str) -> list[dict]:
    settings = PLATFORM_SETTINGS.get(platform, PLATFORM_SETTINGS["tiktok"])
    clips = []
    
    for idx in range(min(count, len(segments))):
        seg = segments[idx]
        
        clip = {
            "clip_id": f"clip_{platform}_{idx+1:02d}",
            "source_scene": seg["scene_id"],
            "start_time": seg["start_time"],
            "duration": min(seg["duration"], settings["max_duration"]),
            "aspect": aspect,
            "platform": platform,
            "hashtags": settings["hashtag"],
        }
        
        clips.append(clip)
    
    return clips


def main() -> None:
    args = parse_args()
    headline("Social Media Clips")
    cfg = load_config()
    
    package_root = resolve_project_path(f"generation/final_episode_packages/{args.episode_id or 'latest'}")
    voice_plan_path = package_root / f"{args.episode_id or 'latest'}_voice_plan.json"
    
    if not voice_plan_path.exists():
        info("No voice plan found")
        return
    
    voice_plan = read_json(voice_plan_path, {})
    segments = extract_clip_segments(voice_plan, args.duration)
    
    clips = generate_clips(segments, args.platform, args.count, args.aspect)
    
    output = {
        "episode_id": args.episode_id or "latest",
        "platform": args.platform,
        "aspect": args.aspect,
        "clip_count": len(clips),
        "clips": clips,
    }
    
    output_path = package_root / f"{args.platform}_clips.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    
    ok(f"Generated {len(clips)} {args.platform} clips")


if __name__ == "__main__":
    main()
