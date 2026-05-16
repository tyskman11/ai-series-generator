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
    ok,
    info,
    load_config,
    resolve_project_path,
    read_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-edit episode based on mood/target."
    )
    parser.add_argument("--episode-id", help="Episode ID.")
    parser.add_argument(
        "--target", default="broadcast",
        choices=["broadcast", "short", "trailer", "social"],
        help="Target format."
    )
    parser.add_argument(
        "--mood", default="auto",
        choices=["auto", "fast", "smooth", "dynamic"],
        help="Edit pacing style."
    )
    parser.add_argument(
        "--max-duration", type=float,
        help="Max duration in seconds."
    )
    return parser.parse_args()


TRANSITIONS = {
    "fast": ["cut", "dissolve"],
    "smooth": ["dissolve", "fade"],
    "dynamic": ["wipe", "zoom", "cut"],
}


def auto_edit(
    voice_plan: dict,
    target: str,
    mood: str,
    max_duration: float = None
) -> dict:
    scenes = voice_plan.get("scenes", [])
    
    max_dur = max_duration or 3600.0
    edit_list = []
    current_time = 0.0
    
    transition_pool = TRANSITIONS.get(mood, ["dissolve"])
    
    for idx, scene in enumerate(scenes):
        scene_dur = scene.get("duration_seconds", 0)
        
        if current_time + scene_dur > max_dur:
            break
        
        transition = transition_pool[idx % len(transition_pool)]
        
        edit_list.append({
            "index": idx,
            "scene_id": scene.get("scene_id", ""),
            "start": current_time,
            "duration": scene_dur,
            "transition": transition,
            "action": "keep",
        })
        
        current_time += scene_dur
    
    output = {
        "episode_id": voice_plan.get("episode_id", "unknown"),
        "target": target,
        "mood": mood,
        "scene_count": len(edit_list),
        "total_duration": current_time,
        "edits": edit_list,
    }
    
    return output


def main() -> None:
    args = parse_args()
    headline("Auto Editor")
    cfg = load_config()
    
    package_root = resolve_project_path(f"generation/final_episode_packages/{args.episode_id or 'latest'}")
    voice_plan_path = package_root / f"{args.episode_id or 'latest'}_voice_plan.json"
    
    if not voice_plan_path.exists():
        info("No voice plan found")
        return
    
    voice_plan = read_json(voice_plan_path, {})
    edit_plan = auto_edit(voice_plan, args.target, args.mood, args.max_duration)
    
    output_path = package_root / f"auto_edit_{args.target}.json"
    output_path.write_text(json.dumps(edit_plan, indent=2), encoding="utf-8")
    
    ok(f"Auto-edited {edit_plan['scene_count']} scenes for {args.target}")


if __name__ == "__main__":
    main()
