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
        description="Clone voice emotion from source."
    )
    parser.add_argument(
        "--source-episode",
        help="Source episode ID."
    )
    parser.add_argument(
        "--character",
        help="Character name."
    )
    return parser.parse_args()


def extract_emotion_profile(voice_plan: dict, character: str) -> dict:
    emotions = []
    
    for scene in voice_plan.get("scenes", []):
        for line in scene.get("lines", []):
            if line.get("character") == character:
                text = line.get("text", "").lower()
                
                if any(w in ["!", "!!", "??"] for w in text):
                    emotions.append("intense")
                elif any(w in text for w in ["?", "??", "..."]):
                    emotions.append("questioning")
                elif any(w in text for w in ["...", "...."]):
                    emotions.append("thoughtful")
                elif any(w in text for w in ["!", "wow"]):
                    emotions.append("excited")
                else:
                    emotions.append("neutral")
    
    return {
        "character": character,
        "emotion_counts": {e: emotions.count(e) for e in set(emotions)},
        "dominant": max(set(emotions), key=emotions.count) if emotions else "neutral",
    }


def main() -> None:
    args = parse_args()
    headline("Voice Emotion Cloning")
    cfg = load_config()
    
    if not args.source_episode or not args.character:
        info("--source-episode and --character required")
        return
    
    package_root = resolve_project_path(f"generation/final_episode_packages/{args.source_episode}")
    voice_plan_path = package_root / f"{args.source_episode}_voice_plan.json"
    
    if not voice_plan_path.exists():
        info("No voice plan found")
        return
    
    voice_plan = read_json(voice_plan_path, {})
    profile = extract_emotion_profile(voice_plan, args.character)
    
    output_path = resolve_project_path(f"generation/emotion_profile_{args.character}.json")
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    
    ok(f"Cloned emotion profile for {args.character}")


if __name__ == "__main__":
    main()
