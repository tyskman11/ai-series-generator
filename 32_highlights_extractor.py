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
        description="Extract highlights from episodes."
    )
    parser.add_argument("--episode-id", help="Episode ID.")
    parser.add_argument(
        "--count", type=int, default=10,
        help="Number of highlights to extract."
    )
    parser.add_argument(
        "-- criteria", default="engagement",
        choices=["engagement", "emotion", "action", "drama"],
        help="Extraction criteria."
    )
    return parser.parse_args()


EMOTION_KEYWORDS = {
    "joy": ["laugh", "happy", "love", "joy", "wonderful"],
    "sadness": ["cry", "tears", "sad", "miss", "lost"],
    "anger": ["angry", "hate", "furious", "rage"],
    "surprise": ["wow", "unexpected", "shock", "cannot believe"],
    "tension": ["danger", "threat", "fear", "worried"],
}


def score_scene(
    scene: dict,
    criteria: str
) -> float:
    score = 0.0
    
    lines = scene.get("lines", [])
    text = " ".join(line.get("text", "") for line in lines).lower()
    
    if criteria == "engagement":
        score = scene.get("duration_seconds", 0) * len(lines)
        for kw in EMOTION_KEYWORDS.values():
            if any(w in text for w in kw):
                score += 10
    
    elif criteria == "emotion":
        for emotion, keywords in EMOTION_KEYWORDS.items():
            if any(w in text for w in keywords):
                score += 5
    
    elif criteria == "action":
        action_words = ["run", "fight", "chase", "kick", "shoot", "explosion"]
        if any(w in text for w in action_words):
            score += 20
    
    elif criteria == "drama":
        drama_words = ["confession", "secret", "reveal", "truth", "lie"]
        if any(w in text for w in drama_words):
            score += 15
    
    return score


def extract_highlights(
    voice_plan: dict,
    count: int,
    criteria: str
) -> list[dict]:
    highlights = []
    
    for scene in voice_plan.get("scenes", []):
        score = score_scene(scene, criteria)
        
        lines = scene.get("lines", [])
        preview = " ".join(line.get("text", "")[:80] for line in lines[:2])
        
        highlights.append({
            "scene_id": scene.get("scene_id", ""),
            "start_time": scene.get("start_time_seconds", 0),
            "duration": scene.get("duration_seconds", 0),
            "score": score,
            "preview": preview[:150],
            "type": scene.get("scene_type", "dialogue"),
        })
    
    highlights.sort(key=lambda x: x["score"], reverse=True)
    
    return highlights[:count]


def main() -> None:
    args = parse_args()
    headline("Highlights Extractor")
    cfg = load_config()
    
    package_root = resolve_project_path(f"generation/final_episode_packages/{args.episode_id or 'latest'}")
    voice_plan_path = package_root / f"{args.episode_id or 'latest'}_voice_plan.json"
    
    if not voice_plan_path.exists():
        info("No voice plan found")
        return
    
    voice_plan = read_json(voice_plan_path, {})
    highlights = extract_highlights(voice_plan, args.count, args.criteria)
    
    metadata = {
        "episode_id": args.episode_id or "latest",
        "criteria": args.criteria,
        "count": len(highlights),
        "highlights": highlights,
    }
    
    output_path = package_root / f"highlights_{args.criteria}.json"
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    
    ok(f"Extracted {len(highlights)} highlights ({args.criteria})")


if __name__ == "__main__":
    main()
