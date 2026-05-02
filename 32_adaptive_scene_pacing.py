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
        description="Adaptive scene pacing based on dialog."
    )
    parser.add_argument("--episode-id", help="Episode ID.")
    parser.add_argument(
        "--mode", default="auto",
        choices=["auto", "fast", "calm", "dynamic"],
        help="Pacing mode."
    )
    return parser.parse_args()


PACING_RULES = {
    "fast": {"dialog_multiplier": 0.8, "min_duration": 10, "max_duration": 120},
    "calm": {"dialog_multiplier": 1.2, "min_duration": 30, "max_duration": 180},
    "dynamic": {"dialog_multiplier": 1.0, "min_duration": 15, "max_duration": 150},
    "auto": {"dialog_multiplier": 1.0, "min_duration": 20, "max_duration": 120},
}


def calculate_scene_duration(
    scene: dict,
    rules: dict
) -> float:
    lines = scene.get("lines", [])
    char_count = sum(len(line.get("text", "")) for line in lines)
    
    estimated = char_count / 150 * 60 * rules["dialog_multiplier"]
    
    estimated = max(estimated, rules["min_duration"])
    estimated = min(estimated, rules["max_duration"])
    
    return estimated


def adjust_pacing(voice_plan: dict, mode: str) -> dict:
    rules = PACING_RULES.get(mode, PACING_RULES["auto"])
    
    adjusted_scenes = []
    total_original = 0.0
    total_adjusted = 0.0
    
    for scene in voice_plan.get("scenes", []):
        orig_dur = scene.get("duration_seconds", 0)
        new_dur = calculate_scene_duration(scene, rules)
        
        adjusted_scenes.append({
            "scene_id": scene.get("scene_id", ""),
            "original_duration": orig_dur,
            "adjusted_duration": new_dur,
            "change": new_dur - orig_dur,
        })
        
        total_original += orig_dur
        total_adjusted += new_dur
    
    return {
        "episode_id": voice_plan.get("episode_id", "unknown"),
        "mode": mode,
        "rules": rules,
        "scene_count": len(adjusted_scenes),
        "total_original": total_original,
        "total_adjusted": total_adjusted,
        "scenes": adjusted_scenes,
    }


def main() -> None:
    args = parse_args()
    headline("Adaptive Scene Pacing")
    cfg = load_config()
    
    package_root = resolve_project_path(f"generation/final_episode_packages/{args.episode_id or 'latest'}")
    voice_plan_path = package_root / f"{args.episode_id or 'latest'}_voice_plan.json"
    
    if not voice_plan_path.exists():
        info("No voice plan found")
        return
    
    voice_plan = read_json(voice_plan_path, {})
    adjusted = adjust_pacing(voice_plan, args.mode)
    
    output_path = package_root / f"scene_pacing_{args.mode}.json"
    output_path.write_text(json.dumps(adjusted, indent=2), encoding="utf-8")
    
    ok(f"Adjusted {adjusted['scene_count']} scenes to {args.mode} pacing")


if __name__ == "__main__":
    main()
