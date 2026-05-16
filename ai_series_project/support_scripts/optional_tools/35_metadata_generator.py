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
        description="Generate platform metadata for episodes."
    )
    parser.add_argument("--episode-id", help="Episode ID.")
    parser.add_argument(
        "--platforms", nargs="+",
        default=["youtube", "netflix", "spotify"],
        help="Target platforms."
    )
    return parser.parse_args()


PLATFORM_SCHEMAS = {
    "youtube": {
        "title": str,
        "description": str,
        "tags": list,
        "category": str,
        "privacy": str,
    },
    "netflix": {
        "title": str,
        "synopsis": str,
        "genre": str,
        "maturity_rating": str,
        "features": list,
    },
    "spotify": {
        "title": str,
        "description": str,
        "tags": list,
        "explicit": bool,
    },
}


def generate_metadata(
    episode_id: str,
    voice_plan: dict,
    platforms: list[str]
) -> dict:
    metadata = {
        "episode_id": episode_id,
        "generated_at": "2026-01-01T00:00:00Z",
    }
    
    first_scene = voice_plan.get("scenes", [{}])[0]
    first_lines = first_scene.get("lines", [])
    preview = " ".join(line.get("text", "")[:50] for line in first_lines[:2])
    
    last_scene = voice_plan.get("scenes", [-1]) if voice_plan.get("scenes") else {}
    last_lines = last_scene.get("lines", []) if last_scene else []
    ending = last_lines[-1].get("text", "") if last_lines else ""
    
    duration = sum(s.get("duration_seconds", 0) for s in voice_plan.get("scenes", []))
    
    for platform in platforms:
        schema = PLATFORM_SCHEMAS.get(platform, {})
        
        platform_meta = {
            "title": f"{episode_id} - Full Episode",
            "description": f"Full episode. {preview[:100]}...",
            "duration_seconds": duration,
        }
        
        if platform == "youtube":
            platform_meta.update({
                "tags": ["series", "drama", "episode"],
                "category": "Entertainment",
                "privacy": "public",
            })
        elif platform == "netflix":
            platform_meta.update({
                "synopsis": preview[:200],
                "genre": "Drama",
                "maturity_rating": "TV-14",
                "features": ["hd", "5.1"],
            })
        elif platform == "spotify":
            platform_meta.update({
                "tags": ["series", "podcast", "drama"],
                "explicit": False,
            })
        
        metadata[platform] = platform_meta
    
    return metadata


def main() -> None:
    args = parse_args()
    headline("Metadata Generator")
    cfg = load_config()
    
    package_root = resolve_project_path(f"generation/final_episode_packages/{args.episode_id or 'latest'}")
    voice_plan_path = package_root / f"{args.episode_id or 'latest'}_voice_plan.json"
    
    if not voice_plan_path.exists():
        info("No voice plan found")
        return
    
    voice_plan = read_json(voice_plan_path, {})
    
    metadata = generate_metadata(args.episode_id or "latest", voice_plan, args.platforms)
    
    output_path = package_root / "platform_metadata.json"
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    
    ok(f"Generated metadata for {len(args.platforms)} platforms")


if __name__ == "__main__":
    main()
