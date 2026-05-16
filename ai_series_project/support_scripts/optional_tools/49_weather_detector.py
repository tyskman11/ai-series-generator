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
        description="Detect weather conditions in scenes."
    )
    parser.add_argument("--episode-id", help="Episode ID.")
    return parser.parse_args()


WEATHER_KEYWORDS = {
    "sunny": ["sun", "bright", "clear sky", "sunshine"],
    "rainy": ["rain", "raining", "drizzle", "storm"],
    "cloudy": ["cloud", "overcast", "gray sky"],
    "night": ["night", "dark", "moon", "stars"],
    "foggy": ["fog", "mist", "haze"],
}


def detect_weather(scene: dict) -> str:
    text = " ".join(
        line.get("text", "") for line in scene.get("lines", [])
    ).lower()
    
    for weather, keywords in WEATHER_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return weather
    
    return "indoor"


def process_episode(episode_id: str, voice_plan: dict) -> dict:
    results = []
    
    for scene in voice_plan.get("scenes", []):
        weather = detect_weather(scene)
        results.append({
            "scene_id": scene.get("scene_id", ""),
            "weather": weather,
            "location": scene.get("location", ""),
        })
    
    return {
        "episode_id": episode_id,
        "weather_scenes": results,
    }


def main() -> None:
    args = parse_args()
    headline("Weather Detector")
    cfg = load_config()
    
    package_root = resolve_project_path(f"generation/final_episode_packages/{args.episode_id or 'latest'}")
    voice_plan_path = package_root / f"{args.episode_id or 'latest'}_voice_plan.json"
    
    if not voice_plan_path.exists():
        info("No voice plan found")
        return
    
    voice_plan = read_json(voice_plan_path, {})
    results = process_episode(args.episode_id or "latest", voice_plan)
    
    output_path = package_root / "weather_detection.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    weather_counts = {}
    for scene in results["weather_scenes"]:
        w = scene["weather"]
        weather_counts[w] = weather_counts.get(w, 0) + 1
    
    ok(f"Detected: {weather_counts}")


if __name__ == "__main__":
    main()
