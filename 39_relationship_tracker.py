#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

from support_scripts.pipeline_common import (
    headline,
    ok,
    load_config,
    resolve_project_path,
    read_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track character relationships."
    )
    parser.add_argument(
        "--season", type=int,
        help="Season to analyze."
    )
    parser.add_argument(
        "--min-appearances", type=int, default=3,
        help="Minimum co-appearances."
    )
    return parser.parse_args()


RELATIONSHIP_KEYWORDS = {
    "ally": ["friend", " ally", "support", "help", "together"],
    "enemy": ["enemy", "fight", "against", "oppose", "hostile"],
    "family": ["family", "parent", "child", "brother", "sister", "son", "daughter"],
    "friend": ["friend", "friend", "close", "trust", "love"],
    "romantic": ["love", "kiss", "heart", "romantic", "relationship"],
}


def detect_relationship(text: str) -> str:
    text_lower = text.lower()
    
    for rel, keywords in RELATIONSHIP_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return rel
    
    return "neutral"


def track_relationships(episodes: list[dict], min_app: int) -> dict:
    relations = defaultdict(lambda: defaultdict(int))
    
    for ep in episodes:
        chars_in_scene = set()
        current_scene = ""
        
        for scene in ep["data"].get("scenes", []):
            scene_id = scene.get("scene_id", "")
            
            for line in scene.get("lines", []):
                char = line.get("character", "")
                if char:
                    chars_in_scene.add(char)
                    
                    text = line.get("text", "")
                    rel = detect_relationship(text)
                    
                    if rel != "neutral":
                        for c in chars_in_scene:
                            if c != char:
                                relations[f"{char}-{c}"][rel] += 1
            
            chars_in_scene.clear()
    
    filtered = {}
    for pair, rels in relations.items():
        total = sum(rels.values())
        if total >= min_app:
            dominant = max(rels.items(), key=lambda x: x[1])[0]
            filtered[pair] = {"type": dominant, "count": total}
    
    return filtered


def main() -> None:
    args = parse_args()
    headline("Relationship Tracker")
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
        if args.season and data.get("season") != args.season:
            continue
        episodes.append({"id": ep_dir.name, "data": data})
    
    relations = track_relationships(episodes, args.min_appearances)
    
    output_path = resolve_project_path("generation/relationships.json")
    output_path.write_text(json.dumps(relations, indent=2), encoding="utf-8")
    
    ok(f"Tracked {len(relations)} relationships")


if __name__ == "__main__":
    main()
