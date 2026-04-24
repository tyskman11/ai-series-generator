#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_common import (
    headline,
    ok,
    info,
    load_config,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select scene transition type."
    )
    parser.add_argument(
        "--from-scene",
        help="Source scene ID."
    )
    parser.add_argument(
        "--to-scene",
        help="Target scene ID."
    )
    parser.add_argument(
        "--style", default="auto",
        choices=["auto", "fade", "dissolve", "wipe", "cut", "zoom"],
        help="Transition style."
    )
    return parser.parse_args()


TRANSITIONS = {
    "fade": {"duration": 1.0, "type": "opacity", "ease": "in_out"},
    "dissolve": {"duration": 0.5, "type": "crossfade", "ease": "linear"},
    "wipe": {"duration": 0.3, "type": "directional", "ease": "ease_in"},
    "cut": {"duration": 0.0, "type": "hard", "ease": "none"},
    "zoom": {"duration": 0.8, "type": "scale", "ease": "ease_out"},
}


def select_transition(style: str) -> dict:
    if style == "auto":
        return TRANSITIONS["dissolve"]
    return TRANSITIONS.get(style, TRANSITIONS["dissolve"])


def main() -> None:
    args = parse_args()
    headline("Scene Transition Selector")
    cfg = load_config()
    
    transition = select_transition(args.style)
    
    output = {
        "from_scene": args.from_scene or "previous",
        "to_scene": args.to_scene or "next",
        "style": args.style,
        "params": transition,
    }
    
    output_path = resolve_project_path("generation/scene_transition.json")
    output_path.write_text(str(output), encoding="utf-8")
    
    ok(f"Selected {args.style} transition")


if __name__ == "__main__":
    main()