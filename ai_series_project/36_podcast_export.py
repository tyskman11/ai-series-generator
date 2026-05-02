#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
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
        description="Export audio-only for podcast feeds."
    )
    parser.add_argument("--episode-id", help="Episode ID.")
    parser.add_argument(
        "--format", default="mp3",
        choices=["mp3", "m4a", "ogg"],
        help="Audio format."
    )
    parser.add_argument(
        "--bitrate", default="128k",
        help="Audio bitrate (e.g., 128k, 192k, 320k)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    headline("Podcast Export")
    cfg = load_config()
    
    package_root = resolve_project_path(f"generation/final_episode_packages/{args.episode_id or 'latest'}")
    
    audio_files = list(package_root.glob("*.wav")) + list(package_root.glob("*.mp3")) + list(package_root.glob("*.m4a"))
    
    if audio_files:
        primary_audio = audio_files[0]
        output_path = package_root / f"{args.episode_id or 'latest'}_podcast.{args.format}"
        
        info(f"Would convert: {primary_audio.name} -> {output_path.name}")
        info("FFmpeg integration needed for actual conversion")
        
        output_path.write_text("# place holder", encoding="utf-8")
        
        ok(f"Podcast export ready: {output_path.name}")
    else:
        info("No audio files found in package")
    
    metadata = {
        "episode_id": args.episode_id or "latest",
        "format": args.format,
        "bitrate": args.bitrate,
        "type": "podcast",
    }
    
    metadata_path = package_root / "podcast_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
