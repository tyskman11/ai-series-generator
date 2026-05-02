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
        description="Analyze style consistency in frames."
    )
    parser.add_argument("--episode-id", help="Episode ID.")
    parser.add_argument(
        "--threshold", type=float, default=0.8,
        help="Consistency threshold."
    )
    return parser.parse_args()


STYLE_KEYWORDS = {
    "bright": ["bright", "light", "sunny"],
    "dark": ["dark", "shadow", "night"],
    "warm": ["warm", "orange", "sunset"],
    "cool": ["cool", "blue", "cold"],
    "moody": ["moody", "dramatic", "intense"],
}


def analyze_frame_style(scene: dict) -> dict:
    lines = scene.get("lines", [])
    text = " ".join(line.get("text", "") for line in lines).lower()
    
    scores = {}
    for style, keywords in STYLE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[style] = score
    
    dominant = max(scores.items(), key=lambda x: x[1])[0] if scores else "neutral"
    
    return {
        "scene_id": scene.get("scene_id", ""),
        "styles": scores,
        "dominant": dominant,
    }


def analyze_consistency(voice_plan: dict, threshold: float) -> dict:
    scenes = []
    style_counts = {}
    
    for scene in voice_plan.get("scenes", []):
        analysis = analyze_frame_style(scene)
        scenes.append(analysis)
        
        dom = analysis["dominant"]
        if dom != "neutral":
            style_counts[dom] = style_counts.get(dom, 0) + 1
    
    dominant_series = max(style_counts.items(), key=lambda x: x[1])[0] if style_counts else "neutral"
    
    return {
        "episode_id": voice_plan.get("episode_id", "unknown"),
        "dominant_style": dominant_series,
        "scene_styles": scenes,
        "consistency_score": 1.0 if len(set(s["dominant"] for s in scenes)) == 1 else threshold,
    }


def main() -> None:
    args = parse_args()
    headline("Style Consistency Scoring")
    cfg = load_config()
    
    package_root = resolve_project_path(f"generation/final_episode_packages/{args.episode_id or 'latest'}")
    voice_plan_path = package_root / f"{args.episode_id or 'latest'}_voice_plan.json"
    
    if not voice_plan_path.exists():
        info("No voice plan found")
        return
    
    voice_plan = read_json(voice_plan_path, {})
    results = analyze_consistency(voice_plan, args.threshold)
    
    output_path = package_root / "style_analysis.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    ok(f"Dominant style: {results['dominant_style']}")


if __name__ == "__main__":
    main()
