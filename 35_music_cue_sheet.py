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
    load_config,
    resolve_project_path,
    read_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate music cue sheet for scenes."
    )
    parser.add_argument("--episode-id", help="Episode ID.")
    parser.add_argument(
        "--genre", default="cinematic",
        choices=["cinematic", "drama", "comedy", "action", "romance"],
        help="Music genre/style."
    )
    return parser.parse_args()


def generate_cue_sheet(voice_plan: dict, genre: str) -> list[dict]:
    cues = []
    genre_templates = {
        "cinematic": {"intro": "epic_01", "scene": "cinematic_dramatic", "outro": "epic_outro"},
        "drama": {"intro": "drama_intro", "scene": "drama_underscore", "outro": "drama_outro"},
        "comedy": {"intro": "comedy_upbeat", "scene": "comedy_light", "outro": "comedy_end"},
        "action": {"intro": "action_intro", "scene": "action_tension", "outro": "action_finale"},
        "romance": {"intro": "romance_piano", "scene": "romance_soft", "outro": "romance_end"},
    }
    
    template = genre_templates.get(genre, genre_templates["cinematic"])
    
    for idx, scene in enumerate(voice_plan.get("scenes", [])):
        scene_id = scene.get("scene_id", f"scene_{idx}")
        duration = scene.get("duration_seconds", 0)
        lines = scene.get("lines", [])
        
        cue = {
            "cue_id": f"CUE_{idx+1:03d}",
            "scene_id": scene_id,
            "start_time": 0,
            "duration": duration,
            "track": template["scene"],
            "type": "scene",
        }
        
        if idx == 0:
            cue["track"] = template["intro"]
            cue["type"] = "intro"
        elif idx == len(voice_plan.get("scenes", [])) - 1:
            cue["track"] = template["outro"]
            cue["type"] = "outro"
        
        if lines:
            cue["track"] = template["scene"]
        
        cues.append(cue)
    
    return cues


def main() -> None:
    args = parse_args()
    headline("Music Cue Sheet")
    cfg = load_config()
    
    package_root = resolve_project_path(f"generation/final_episode_packages/{args.episode_id or 'latest'}")
    voice_plan_path = package_root / f"{args.episode_id or 'latest'}_voice_plan.json"
    
    if not voice_plan_path.exists():
        print("No voice plan found")
        return
    
    voice_plan = read_json(voice_plan_path, {})
    cues = generate_cue_sheet(voice_plan, args.genre)
    
    output_path = package_root / "music_cue_sheet.json"
    output_path.write_text(json.dumps(cues, indent=2), encoding="utf-8")
    
    ok(f"Wrote {len(cues)} cue points to {output_path}")


if __name__ == "__main__":
    main()
