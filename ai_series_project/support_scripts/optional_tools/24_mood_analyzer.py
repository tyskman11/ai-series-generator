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
        description="Analyze mood/emotions across episodes."
    )
    parser.add_argument("--episode-id", help="Episode ID.")
    parser.add_argument(
        "--season", type=int,
        help="Analyze full season."
    )
    parser.add_argument(
        "--granularity", default="scene",
        choices=["scene", "act", "episode"],
        help="Analysis granularity."
    )
    return parser.parse_args()


MOOD_KEYWORDS = {
    "tense": ["danger", "fear", "worry", "threat", "escape"],
    "joyful": ["laugh", "happy", "love", "celebrate", "wonderful"],
    "sad": ["cry", " tears", "miss", "lost", "grief"],
    "romantic": ["love", "kiss", "heart", "feelings"],
    "action": ["run", "fight", "chase", "explosion"],
    "mysterious": ["mystery", "secret", "strange", "unknown"],
    "comedic": ["joke", "funny", "laugh", "humor"],
    "dramatic": ["confession", "truth", "reveal", "confrontation"],
}


def analyze_scene_mood(scene: dict) -> dict:
    lines = scene.get("lines", [])
    text = " ".join(line.get("text", "") for line in lines).lower()
    
    mood_scores = {}
    for mood, keywords in MOOD_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            mood_scores[mood] = score
    
    dominant = max(mood_scores.items(), key=lambda x: x[1])[0] if mood_scores else "neutral"
    
    return {
        "scene_id": scene.get("scene_id", ""),
        "duration": scene.get("duration_seconds", 0),
        "moods": mood_scores,
        "dominant": dominant,
    }


def analyze_episode_mood(voice_plan: dict, granularity: str) -> dict:
    results = {
        "episode_id": voice_plan.get("episode_id", "unknown"),
        "granularity": granularity,
        "scenes": [],
        "summary": {},
    }
    
    for scene in voice_plan.get("scenes", []):
        analysis = analyze_scene_mood(scene)
        results["scenes"].append(analysis)
    
    all_moods = {}
    for scene_result in results["scenes"]:
        for mood, score in scene_result.get("moods", {}).items():
            all_moods[mood] = all_moods.get(mood, 0) + score
    
    results["summary"] = all_moods
    
    if all_moods:
        results["overall_dominant"] = max(all_moods.items(), key=lambda x: x[1])[0]
    
    return results


def main() -> None:
    args = parse_args()
    headline("Mood Analyzer")
    cfg = load_config()
    
    package_root = resolve_project_path(f"generation/final_episode_packages/{args.episode_id or 'latest'}")
    voice_plan_path = package_root / f"{args.episode_id or 'latest'}_voice_plan.json"
    
    if not voice_plan_path.exists():
        info("No voice plan found")
        return
    
    voice_plan = read_json(voice_plan_path, {})
    analysis = analyze_episode_mood(voice_plan, args.granularity)
    
    output_path = package_root / "mood_analysis.json"
    output_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    
    dominant = analysis.get("overall_dominant", "N/A")
    ok(f"Analyzed {len(analysis['scenes'])} scenes. Dominant: {dominant}")


if __name__ == "__main__":
    main()
