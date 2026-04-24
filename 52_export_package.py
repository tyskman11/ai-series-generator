#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from pipeline_common import (
    headline,
    info,
    ok,
    load_config,
    resolve_project_path,
    episode_production_package_root,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export finished episode package for external tools (DaVinci, Premiere)."
    )
    parser.add_argument("--episode-id", help="Target a specific episode ID.")
    parser.add_argument("--format", default="da Vinci Resolve", help="Export format.")
    return parser.parse_args()


def export_for_davinci(episode_package_root: Path, export_root: Path) -> dict[str, Any]:
    manifest_path = episode_package_root / f"{episode_package_root.name}_production_package.json"
    if not manifest_path.exists():
        return {"error": "No production package found"}
    
    manifest = read_json(manifest_path, {})
    scenes = manifest.get("scenes", [])
    
    project_data = {
        "project_name": manifest.get("episode_id", "export"),
        "resolution": {"width": 1280, "height": 720, "fps": 30},
        "tracks": [],
    }
    
    for idx, scene in enumerate(scenes):
        scene_id = scene.get("scene_id", f"scene_{idx}")
        video_path = scene.get("video_path")
        audio_path = scene.get("audio_path")
        
        track = {
            "index": idx,
            "scene_id": scene_id,
            "media": [],
        }
        
        if video_path and Path(video_path).exists():
            track["media"].append({"type": "video", "path": str(video_path)})
        if audio_path and Path(audio_path).exists():
            track["media"].append({"type": "audio", "path": str(audio_path)})
        
        project_data["tracks"].append(track)
    
    return project_data


def export_for_premiere(episode_package_root: Path, export_root: Path) -> dict[str, Any]:
    xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<project>']
    
    manifest_path = episode_package_root / f"{episode_package_root.name}_production_package.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path, {})
        for scene in manifest.get("scenes", []):
            scene_id = scene.get("scene_id", "unknown")
            xml_parts.append(f'  <clip id="{scene_id}">')
            xml_parts.append(f'    <name>{scene_id}</name>')
            xml_parts.append('  </clip>')
    
    xml_parts.append('</project>')
    return {"xml": "\n".join(xml_parts)}


def main() -> None:
    args = parse_args()
    headline("Export Package for External Tools")
    cfg = load_config()
    
    episode_id = args.episode_id or "latest"
    package_root = episode_production_package_root(cfg, episode_id)
    
    if not package_root.exists():
        info(f"No package found for {episode_id}")
        return
    
    export_formats = {
        "da Vinci Resolve": export_for_davinci,
        "Premiere Pro": export_for_premiere,
    }
    
    export_func = export_formats.get(args.format, export_for_davinci)
    result = export_func(package_root, resolve_project_path("exports"))
    
    ok(f"Exported package for {args.format}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()