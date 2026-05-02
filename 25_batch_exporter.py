#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "ai_series_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse
import json
import subprocess
from pathlib import Path

from support_scripts.pipeline_common import (
    headline,
    ok,
    info,
    load_config,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch export episode to multiple formats."
    )
    parser.add_argument("--episode-id", help="Episode ID.")
    parser.add_argument(
        "--formats", nargs="+",
        default=["mp4", "webm", "mp3", "srt"],
        help="Export formats."
    )
    parser.add_argument(
        "--resolutions", nargs="+",
        default=["1080p", "720p"],
        help="Video resolutions."
    )
    return parser.parse_args()


EXPORT_CONFIGS = {
    "mp4": {"video": True, "audio": True, "codec": "h264"},
    "webm": {"video": True, "audio": True, "codec": "vp9"},
    "mp3": {"video": False, "audio": True, "codec": "mp3"},
    "m4a": {"video": False, "audio": True, "codec": "aac"},
    "srt": {"video": False, "audio": False, "subtitles": True},
    "vtt": {"video": False, "audio": False, "subtitles": True},
}


def generate_export_tasks(
    episode_id: str,
    formats: list[str],
    resolutions: list[str]
) -> list[dict]:
    tasks = []
    
    for fmt in formats:
        config = EXPORT_CONFIGS.get(fmt, {})
        if not config:
            continue
        
        if config.get("video"):
            for res in resolutions:
                tasks.append({
                    "format": fmt,
                    "resolution": res,
                    "output": f"{episode_id}_{res}.{fmt}",
                })
        else:
            tasks.append({
                "format": fmt,
                "output": f"{episode_id}.{fmt}",
            })
    
    return tasks


def main() -> None:
    args = parse_args()
    headline("Batch Exporter")
    cfg = load_config()
    
    package_root = resolve_project_path(f"generation/final_episode_packages/{args.episode_id or 'latest'}")
    
    tasks = generate_export_tasks(
        args.episode_id or "latest",
        args.formats,
        args.resolutions
    )
    
    manifest = {
        "episode_id": args.episode_id or "latest",
        "tasks": tasks,
        "status": "pending",
    }
    
    output_path = package_root / "batch_export_manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    
    ok(f"Created {len(tasks)} export tasks")


if __name__ == "__main__":
    main()
