#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        description="Auto-detect chapter markers from episode content."
    )
    parser.add_argument("--episode-id", help="Episode ID.")
    parser.add_argument(
        "--min-duration", type=float, default=60.0,
        help="Minimum scene duration in seconds to be a chapter."
    )
    parser.add_argument(
        "--format", default="ffmetadata",
        choices=["ffmetadata", "xml", "json"],
        help="Output format."
    )
    return parser.parse_args()


CHAPTER_KEYWORDS = [
    "previously", "last time", "continue", "to be continued",
    "prologue", "opening", "cold open", "title",
    "teaser", "cold open",
    "act one", "act two", "act three",
    "scene", "transition",
    "final", "the end", "credits",
]


def detect_chapters(voice_plan: dict, min_duration: float = 60.0) -> list[dict]:
    chapters = []
    current_chapter = None
    chapter_index = 1
    
    for idx, scene in enumerate(voice_plan.get("scenes", [])):
        scene_id = scene.get("scene_id", f"scene_{idx}")
        duration = scene.get("duration_seconds", 0)
        
        if duration < min_duration:
            continue
        
        scene_type = scene.get("scene_type", "dialogue")
        
        lines = scene.get("lines", [])
        if lines:
            first_line = lines[0].get("text", "")[:100]
            last_line = lines[-1].get("text", "")[:100] if lines else ""
        else:
            first_line = ""
            last_line = ""
        
        is_opening = idx == 0
        is_ending = idx == len(voice_plan.get("scenes", [])) - 1
        has_transition = any(kw in first_line.lower() for kw in CHAPTER_KEYWORDS)
        
        chapter = {
            "chapter_id": f"chapter_{chapter_index}",
            "scene_id": scene_id,
            "start_time": scene.get("start_time_seconds", 0),
            "duration": duration,
            "type": scene_type,
            "title": f"{scene_type.title()} {chapter_index}",
        }
        
        if is_opening:
            chapter["title"] = "Opening"
            chapter["type"] = "opening"
        elif is_ending:
            chapter["title"] = "Final Scene"
            chapter["type"] = "finale"
        
        chapters.append(chapter)
        chapter_index += 1
    
    return chapters


def format_ffmetadata(chapters: list[dict]) -> str:
    lines = ["; FFMetadata"]
    for ch in chapters:
        lines.append(f"[CHAPTER]")
        lines.append(f"title={ch['title']}")
        lines.append(f"start={int(ch['start_time'] * 1000)}")
        lines.append(f"end={int((ch['start_time'] + ch['duration']) * 1000)}")
        lines.append("")
    return "\n".join(lines)


def format_xml(chapters: list[dict]) -> str:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<chapters>"]
    for ch in chapters:
        lines.append(f"  <chapter id=\"{ch['chapter_id']}\" time=\"{ch['start_time']}\">")
        lines.append(f"    <title>{ch['title']}</title>")
        lines.append(f"    <duration>{ch['duration']}</duration>")
        lines.append(f"  </chapter>")
    lines.append("</chapters>")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    headline("Chapter Markers")
    cfg = load_config()
    
    package_root = resolve_project_path(f"generation/final_episode_packages/{args.episode_id or 'latest'}")
    voice_plan_path = package_root / f"{args.episode_id or 'latest'}_voice_plan.json"
    
    if not voice_plan_path.exists():
        info("No voice plan found")
        return
    
    voice_plan = read_json(voice_plan_path, {})
    chapters = detect_chapters(voice_plan, args.min_duration)
    
    if args.format == "ffmetadata":
        output = format_ffmetadata(chapters)
    elif args.format == "xml":
        output = format_xml(chapters)
    else:
        import json
        output = json.dumps(chapters, indent=2)
    
    output_path = package_root / f"{args.episode_id or 'latest'}_chapters.{'txt' if args.format == 'ffmetadata' else args.format}"
    output_path.write_text(output, encoding="utf-8")
    
    ok(f"Detected {len(chapters)} chapters")


if __name__ == "__main__":
    main()
