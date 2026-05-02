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
from collections import defaultdict

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
        description="Find similar scenes across episodes."
    )
    parser.add_argument(
        "--scene-id",
        help="Scene ID to findsimilar."
    )
    parser.add_argument(
        "--episode-id",
        help="Episode containing scene."
    )
    parser.add_argument(
        "--max-results", type=int, default=10,
        help="Max similar scenes to return."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7,
        help="Similarity threshold."
    )
    return parser.parse_args()


def extract_scene_features(scene: dict) -> dict:
    lines = scene.get("lines", [])
    text = " ".join(line.get("text", "") for line in lines).lower()
    
    location = scene.get("location", "").lower()
    time_of_day = scene.get("time_of_day", "").lower()
    
    keywords = set()
    for word in text.split():
        if len(word) > 4:
            keywords.add(word)
    
    return {
        "location": location,
        "time_of_day": time_of_day,
        "keywords": keywords,
        "text_preview": text[:200],
    }


def calculate_similarity(features1: dict, features2: dict) -> float:
    score = 0.0
    
    if features1.get("location") and features1["location"] == features2.get("location"):
        score += 0.3
    
    if features1.get("time_of_day") and features1["time_of_day"] == features2.get("time_of_day"):
        score += 0.2
    
    kw1 = features1.get("keywords", set())
    kw2 = features2.get("keywords", set())
    if kw1 and kw2:
        overlap = len(kw1 & kw2)
        union = len(kw1 | kw2)
        if union > 0:
            score += (overlap / union) * 0.5
    
    return score


def find_similar_scenes(
    target_scene: dict,
    episodes: list[dict],
    max_results: int,
    threshold: float
) -> list[dict]:
    target_features = extract_scene_features(target_scene)
    results = []
    
    for ep in episodes:
        for scene in ep["data"].get("scenes", []):
            if scene.get("scene_id") == target_scene.get("scene_id"):
                continue
            
            comp_features = extract_scene_features(scene)
            similarity = calculate_similarity(target_features, comp_features)
            
            if similarity >= threshold:
                results.append({
                    "episode_id": ep["id"],
                    "scene_id": scene.get("scene_id", ""),
                    "similarity": similarity,
                    "location": scene.get("location", ""),
                })
    
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:max_results]


def main() -> None:
    args = parse_args()
    headline("Similar Scene Finder")
    cfg = load_config()
    
    packages_dir = resolve_project_path("generation/final_episode_packages")
    episodes = []
    
    for ep_dir in packages_dir.iterdir():
        if not ep_dir.is_dir():
            continue
        
        voice_plan = list(ep_dir.glob("*_voice_plan.json"))
        if not voice_plan:
            continue
        
        data = read_json(voice_plan[0], {})
        episodes.append({"id": ep_dir.name, "data": data})
    
    target_scene = {}
    if args.episode_id and args.scene_id:
        pkg = next((e for e in episodes if e["id"] == args.episode_id), None)
        if pkg:
            target_scene = next(
                (s for s in pkg["data"].get("scenes", []) if s.get("scene_id") == args.scene_id),
                {}
            )
    
    if not target_scene:
        info("No target scene found")
        return
    
    similar = find_similar_scenes(target_scene, episodes, args.max_results, args.threshold)
    
    output = {
        "target_scene": args.scene_id,
        "episode": args.episode_id,
        "results": similar,
    }
    
    output_path = resolve_project_path("generation/similar_scenes.json")
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    
    ok(f"Found {len(similar)} similar scenes")


if __name__ == "__main__":
    main()
