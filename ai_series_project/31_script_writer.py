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
        description="Generate script/dialog suggestions."
    )
    parser.add_argument("--episode-id", help="Episode ID.")
    parser.add_argument(
        "--scene-id",
        help="Scene ID to generate dialog for."
    )
    parser.add_argument(
        "--style", default="natural",
        choices=["natural", "dramatic", "comedic", "action"],
        help="Dialog style."
    )
    parser.add_argument(
        "--lines", type=int, default=5,
        help="Number of lines to generate."
    )
    return parser.parse_args()


STYLE_TEMPLATES = {
    "natural": [
        "I need to tell you something.",
        "Have you heard about what happened?",
        "It's not what you think.",
        "We should talk about this later.",
    ],
    "dramatic": [
        "This changes everything!",
        "I never expected it to come to this.",
        "The truth will finally be revealed.",
        "You have no idea what I've done.",
    ],
    "comedic": [
        "Wait, did I just say that out loud?",
        "This is definitely not my finest moment.",
        "Plot twist: I'm actually terrible at this.",
        "Spoiler alert: That didn't work.",
    ],
    "action": [
        "We have to move now!",
        "Get down!",
        "Follow me!",
        "It's a trap!",
    ],
}


def generate_dialog(
    style: str,
    count: int,
    characters: list[str]
) -> list[dict]:
    import random
    
    templates = STYLE_TEMPLATES.get(style, STYLE_TEMPLATES["natural"])
    dialog = []
    
    for i in range(count):
        char = characters[i % len(characters)] if characters else "CHAR"
        line = random.choice(templates)
        
        dialog.append({
            "line_id": f"line_{i+1}",
            "character": char,
            "text": line,
            "style": style,
        })
    
    return dialog


def main() -> None:
    args = parse_args()
    headline("Script Writer")
    cfg = load_config()
    
    package_root = resolve_project_path(f"generation/final_episode_packages/{args.episode_id or 'latest'}")
    voice_plan_path = package_root / f"{args.episode_id or 'latest'}_voice_plan.json"
    
    characters = []
    if voice_plan_path.exists():
        voice_plan = read_json(voice_plan_path, {})
        
        for scene in voice_plan.get("scenes", []):
            for line in scene.get("lines", []):
                char = line.get("character", "")
                if char and char not in characters:
                    characters.append(char)
    
    if not characters:
        characters = ["PROTAG", "SUPPORTING", "ACTOR"]
    
    dialog = generate_dialog(args.style, args.lines, characters)
    
    output = {
        "episode_id": args.episode_id or "latest",
        "scene_id": args.scene_id or "new",
        "style": args.style,
        "dialog": dialog,
    }
    
    output_path = package_root / f"generated_dialog_{args.style}.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    
    ok(f"Generated {len(dialog)} lines in {args.style} style")


if __name__ == "__main__":
    main()
