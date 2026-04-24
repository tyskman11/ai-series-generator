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
        description="Detect emotions in dialog."
    )
    parser.add_argument("--episode-id", help="Episode ID.")
    return parser.parse_args()


EMOTIONS = {
    "joy": ["happy", "laugh", "joy", "wonderful", "great"],
    "sadness": ["sad", "cry", "tears", "miss", "lost"],
    "anger": ["angry", "hate", "furious", "rage"],
    "fear": ["afraid", "fear", "scared", "worried"],
    "surprise": ["wow", "shock", "cannot believe", "unexpected"],
    "love": ["love", "heart", "kiss", "care"],
}


def detect_emotion(text: str) -> str:
    text_lower = text.lower()
    
    for emotion, keywords in EMOTIONS.items():
        if any(kw in text_lower for kw in keywords):
            return emotion
    
    return "neutral"


def analyze_episode(voice_plan: dict) -> dict:
    emotion_counts = {e: 0 for e in EMOTIONS}
    emotion_counts["neutral"] = 0
    
    line_count = 0
    for scene in voice_plan.get("scenes", []):
        for line in scene.get("lines", []):
            text = line.get("text", "")
            emotion = detect_emotion(text)
            emotion_counts[emotion] += 1
            line_count += 1
    
    return {
        "episode_id": voice_plan.get("episode_id", "unknown"),
        "line_count": line_count,
        "emotions": emotion_counts,
    }


def main() -> None:
    args = parse_args()
    headline("Emotion-Aware Generation")
    cfg = load_config()
    
    package_root = resolve_project_path(f"generation/final_episode_packages/{args.episode_id or 'latest'}")
    voice_plan_path = package_root / f"{args.episode_id or 'latest'}_voice_plan.json"
    
    if not voice_plan_path.exists():
        info("No voice plan found")
        return
    
    voice_plan = read_json(voice_plan_path, {})
    results = analyze_episode(voice_plan)
    
    output_path = package_root / "emotion_analysis.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    ok(f"Analyzed {results['line_count']} lines")


if __name__ == "__main__":
    main()