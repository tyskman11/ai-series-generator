#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse
from pathlib import Path

from support_scripts.pipeline_common import (
    headline,
    ok,
    load_config,
    resolve_project_path,
    read_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate subtitles from episode dialog."
    )
    parser.add_argument("--episode-id", help="Episode ID.")
    parser.add_argument(
        "--format", default="srt",
        choices=["srt", "vtt", "ass"],
        help="Subtitle format."
    )
    return parser.parse_args()


def format_srt(lines: list[dict]) -> str:
    output = []
    for idx, line in enumerate(lines, 1):
        start = format_time_srt(line.get("start_time", 0))
        end = format_time_srt(line.get("end_time", 0))
        text = line.get("text", "")
        output.append(f"{idx}\n{start} --> {end}\n{text}\n")
    return "\n".join(output)


def format_vtt(lines: list[dict]) -> str:
    output = ["WEBVTT", ""]
    for line in lines:
        start = format_time_vtt(line.get("start_time", 0))
        end = format_time_vtt(line.get("end_time", 0))
        text = line.get("text", "")
        output.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(output)


def format_time_srt(ms: float) -> str:
    hours = int(ms // 3600)
    minutes = int((ms % 3600) // 60)
    seconds = int(ms % 60)
    millis = int((ms % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def format_time_vtt(ms: float) -> str:
    hours = int(ms // 3600)
    minutes = int((ms % 3600) // 60)
    seconds = int(ms % 60)
    millis = int((ms % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def main() -> None:
    args = parse_args()
    headline("Generate Subtitles")
    cfg = load_config()
    
    package_root = resolve_project_path(f"generation/final_episode_packages/{args.episode_id or 'latest'}")
    
    voice_plan_path = package_root / f"{args.episode_id or 'latest'}_voice_plan.json"
    if not voice_plan_path.exists():
        print("No voice plan found")
        return
    
    voice_plan = read_json(voice_plan_path, {})
    all_lines = []
    for scene in voice_plan.get("scenes", []):
        all_lines.extend(scene.get("lines", []))
    
    if args.format == "srt":
        output = format_srt(all_lines)
    else:
        output = format_vtt(all_lines)
    
    output_path = package_root / f"{args.episode_id or 'latest'}_subtitles.{args.format}"
    write_text(output_path, output)
    
    ok(f"Wrote {len(all_lines)} lines to {output_path}")


if __name__ == "__main__":
    main()
