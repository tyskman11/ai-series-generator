#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "ai_series_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse
import json
from collections import Counter
from pathlib import Path

from support_scripts.pipeline_common import (
    headline,
    load_config,
    resolve_project_path,
    read_json,
    latest_matching_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze recurring patterns in the series.")
    parser.add_argument("--output", default="patterns.json", help="Output file.")
    return parser.parse_args()


def analyze_patterns(story_prompts_dir: Path) -> dict[str, Any]:
    patterns = {
        "location_frequency": Counter(),
        "character_coappearance": Counter(),
        "dialog_starts": Counter(),
        "scene_lengths": [],
    }
    
    for prompt_file in story_prompts_dir.glob("*.md"):
        try:
            content = prompt_file.read_text(encoding="utf-8")
            lines = content.split("\n")
            for line in lines:
                if line.startswith("INT.") or line.startswith("EXT."):
                    patterns["dialog_starts"][line.split(".")[0]] += 1
        except Exception:
            continue
    
    return {
        "locations": dict(patterns["location_frequency"].most_common(10)),
        "dialog_starts": dict(patterns["dialog_starts"]),
    }


def main() -> None:
    args = parse_args()
    headline("Analyze Series Patterns")
    cfg = load_config()
    
    prompts_dir = resolve_project_path("generation/story_prompts")
    patterns = analyze_patterns(prompts_dir)
    
    output_path = resolve_project_path("series_bible") / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(patterns, indent=2), encoding="utf-8")
    
    print(f"Patterns written to {output_path}")


if __name__ == "__main__":
    main()
