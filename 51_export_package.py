#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from pipeline_common import (
    generated_episode_artifacts,
    headline,
    info,
    latest_generated_episode_artifacts,
    load_config,
    ok,
    read_json,
    resolve_project_path,
    resolve_stored_project_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a finished episode package for external tools such as DaVinci Resolve or Premiere."
    )
    parser.add_argument("--episode-id", help="Target a specific episode ID. Uses the newest generated episode by default.")
    parser.add_argument(
        "--format",
        default="davinci",
        choices=("davinci", "premiere", "json"),
        help="Target export format. Default: davinci.",
    )
    parser.add_argument(
        "--copy-media",
        action="store_true",
        help="Copy referenced media into the export folder for easier handoff.",
    )
    return parser.parse_args()


def export_root(cfg: dict[str, Any], episode_id: str, format_name: str) -> Path:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    base = resolve_project_path(str(paths.get("export_packages", "exports/packages")))
    return base / episode_id / format_name


def path_ready(path_value: object) -> bool:
    candidate = resolve_stored_project_path(path_value)
    return bool(str(candidate).strip()) and candidate.exists() and candidate.is_file()


def stored_path_text(path_value: object) -> str:
    raw_text = str(path_value or "").strip()
    if not raw_text:
        return ""
    return str(resolve_stored_project_path(path_value))


def resolve_episode_artifacts(cfg: dict[str, Any], episode_id: str | None) -> dict[str, Any]:
    requested = str(episode_id or "").strip()
    if not requested or requested.lower() == "latest":
        return latest_generated_episode_artifacts(cfg)
    return generated_episode_artifacts(cfg, requested)


def load_production_package(artifacts: dict[str, Any]) -> dict[str, Any]:
    package_path = resolve_stored_project_path(artifacts.get("production_package", ""))
    if not package_path.exists():
        return {}
    payload = read_json(package_path, {})
    return payload if isinstance(payload, dict) else {}


def preferred_scene_video(scene_payload: dict[str, Any]) -> str:
    current_outputs = (
        scene_payload.get("current_generated_outputs", {})
        if isinstance(scene_payload.get("current_generated_outputs"), dict)
        else {}
    )
    mastering = scene_payload.get("mastering", {}) if isinstance(scene_payload.get("mastering"), dict) else {}
    target_outputs = mastering.get("target_outputs", {}) if isinstance(mastering.get("target_outputs"), dict) else {}
    candidates = [
        current_outputs.get("scene_master_clip", ""),
        current_outputs.get("video_source_path", ""),
        target_outputs.get("scene_master_clip", ""),
    ]
    for value in candidates:
        if path_ready(value):
            return str(resolve_stored_project_path(value))
    return ""


def preferred_scene_audio(scene_payload: dict[str, Any]) -> str:
    current_outputs = (
        scene_payload.get("current_generated_outputs", {})
        if isinstance(scene_payload.get("current_generated_outputs"), dict)
        else {}
    )
    voice_clone = scene_payload.get("voice_clone", {}) if isinstance(scene_payload.get("voice_clone"), dict) else {}
    target_outputs = voice_clone.get("target_outputs", {}) if isinstance(voice_clone.get("target_outputs"), dict) else {}
    candidates = [
        current_outputs.get("scene_dialogue_audio", ""),
        target_outputs.get("scene_dialogue_audio", ""),
    ]
    for value in candidates:
        if path_ready(value):
            return str(resolve_stored_project_path(value))
    return ""


def preferred_scene_frame(scene_payload: dict[str, Any]) -> str:
    current_outputs = (
        scene_payload.get("current_generated_outputs", {})
        if isinstance(scene_payload.get("current_generated_outputs"), dict)
        else {}
    )
    image_generation = scene_payload.get("image_generation", {}) if isinstance(scene_payload.get("image_generation"), dict) else {}
    target_outputs = image_generation.get("target_outputs", {}) if isinstance(image_generation.get("target_outputs"), dict) else {}
    candidates = [
        current_outputs.get("generated_primary_frame", ""),
        current_outputs.get("generated_layered_storyboard_frame", ""),
        target_outputs.get("primary_frame", ""),
        target_outputs.get("layered_storyboard_frame", ""),
    ]
    for value in candidates:
        if path_ready(value):
            return str(resolve_stored_project_path(value))
    return ""


def collect_scene_export_rows(production_package: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenes = production_package.get("scenes", []) if isinstance(production_package.get("scenes"), list) else []
    for index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue
        quality = scene.get("quality_assessment", {}) if isinstance(scene.get("quality_assessment"), dict) else {}
        storyboard = scene.get("storyboard", {}) if isinstance(scene.get("storyboard"), dict) else {}
        rows.append(
            {
                "index": index,
                "scene_id": str(scene.get("scene_id", "") or f"scene_{index:03d}"),
                "title": str(scene.get("title", "") or ""),
                "summary": str(scene.get("summary", "") or ""),
                "duration_seconds": float(scene.get("duration_seconds", 0.0) or 0.0),
                "characters": list(scene.get("characters", []) or []) if isinstance(scene.get("characters"), list) else [],
                "video_path": preferred_scene_video(scene),
                "audio_path": preferred_scene_audio(scene),
                "frame_path": preferred_scene_frame(scene),
                "scene_package_path": stored_path_text(storyboard.get("scene_package_path", "")),
                "quality_label": str(quality.get("quality_label", "") or ""),
                "quality_percent": int(quality.get("quality_percent", 0) or 0),
                "weaknesses": list(quality.get("weaknesses", []) or []) if isinstance(quality.get("weaknesses"), list) else [],
            }
        )
    return rows


def build_common_export_payload(
    artifacts: dict[str, Any],
    production_package: dict[str, Any],
    format_name: str,
    target_root: Path,
) -> dict[str, Any]:
    scene_rows = collect_scene_export_rows(production_package)
    return {
        "format": format_name,
        "export_root": str(target_root),
        "episode_id": str(artifacts.get("episode_id", "") or ""),
        "display_title": str(artifacts.get("display_title", "") or ""),
        "episode_title": str(artifacts.get("episode_title", "") or ""),
        "render_mode": str(artifacts.get("render_mode", "") or ""),
        "production_readiness": str(artifacts.get("production_readiness", "") or ""),
        "quality_label": str(artifacts.get("quality_label", "") or ""),
        "quality_percent": int(artifacts.get("quality_percent", 0) or 0),
        "final_render": str(artifacts.get("final_render", "") or ""),
        "full_generated_episode": str(artifacts.get("full_generated_episode", "") or ""),
        "dialogue_audio": str(artifacts.get("dialogue_audio", "") or ""),
        "subtitle_preview": str(artifacts.get("subtitle_preview", "") or ""),
        "voice_plan": str(artifacts.get("voice_plan", "") or ""),
        "render_manifest": str(artifacts.get("render_manifest", "") or ""),
        "production_package": str(artifacts.get("production_package", "") or ""),
        "production_prompt_preview": str(artifacts.get("production_prompt_preview", "") or ""),
        "delivery_bundle_root": str(artifacts.get("delivery_bundle_root", "") or ""),
        "delivery_manifest": str(artifacts.get("delivery_manifest", "") or ""),
        "delivery_episode": str(artifacts.get("delivery_episode", "") or ""),
        "scene_count": len(scene_rows),
        "scenes": scene_rows,
    }


def copy_referenced_media(export_payload: dict[str, Any], media_root: Path) -> dict[str, str]:
    media_root.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    copied_by_source: dict[str, str] = {}
    seen_sources: set[str] = set()
    top_level_fields = (
        "final_render",
        "full_generated_episode",
        "dialogue_audio",
        "subtitle_preview",
        "voice_plan",
        "render_manifest",
        "production_package",
        "production_prompt_preview",
    )
    for field in top_level_fields:
        source = str(export_payload.get(field, "") or "").strip()
        if not path_ready(source):
            continue
        source_path = resolve_stored_project_path(source)
        if str(source_path) in seen_sources:
            continue
        seen_sources.add(str(source_path))
        target_path = media_root / source_path.name
        shutil.copy2(source_path, target_path)
        copied[field] = str(target_path)
        copied_by_source[str(source_path)] = str(target_path)

    for scene in export_payload.get("scenes", []):
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene.get("scene_id", "") or "scene")
        scene_dir = media_root / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        for field in ("video_path", "audio_path", "frame_path"):
            source = str(scene.get(field, "") or "").strip()
            if not path_ready(source):
                continue
            source_path = resolve_stored_project_path(source)
            if str(source_path) in seen_sources:
                scene[f"{field}_copied"] = copied_by_source.get(str(source_path), "")
                continue
            seen_sources.add(str(source_path))
            target_path = scene_dir / source_path.name
            shutil.copy2(source_path, target_path)
            scene[f"{field}_copied"] = str(target_path)
            copied_by_source[str(source_path)] = str(target_path)
    return copied


def export_for_davinci(export_payload: dict[str, Any], target_root: Path) -> dict[str, Any]:
    timeline = {
        "project_name": export_payload.get("episode_id") or "export",
        "display_title": export_payload.get("display_title") or "",
        "resolution": {"width": 1280, "height": 720, "fps": 30},
        "production_readiness": export_payload.get("production_readiness", ""),
        "quality_percent": export_payload.get("quality_percent", 0),
        "tracks": [
            {
                "track_name": "episode_timeline",
                "clips": [
                    {
                        "index": scene.get("index", 0),
                        "scene_id": scene.get("scene_id", ""),
                        "title": scene.get("title", ""),
                        "video": scene.get("video_path_copied") or scene.get("video_path", ""),
                        "audio": scene.get("audio_path_copied") or scene.get("audio_path", ""),
                        "frame": scene.get("frame_path_copied") or scene.get("frame_path", ""),
                        "duration_seconds": scene.get("duration_seconds", 0.0),
                        "quality_percent": scene.get("quality_percent", 0),
                    }
                    for scene in export_payload.get("scenes", [])
                    if isinstance(scene, dict)
                ],
            }
        ],
    }
    output_path = target_root / "davinci_timeline.json"
    write_json(output_path, timeline)
    return {"timeline_path": str(output_path), "clip_count": len(timeline["tracks"][0]["clips"])}


def export_for_premiere(export_payload: dict[str, Any], target_root: Path) -> dict[str, Any]:
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<project>",
        f"  <name>{escape(str(export_payload.get('display_title') or export_payload.get('episode_id') or 'export'))}</name>",
        "  <sequence>",
    ]
    for scene in export_payload.get("scenes", []):
        if not isinstance(scene, dict):
            continue
        clip_name = escape(str(scene.get("title") or scene.get("scene_id") or "scene"))
        clip_id = escape(str(scene.get("scene_id") or "scene"))
        clip_path = escape(str(scene.get("video_path_copied") or scene.get("video_path") or scene.get("frame_path_copied") or scene.get("frame_path") or ""))
        xml_lines.extend(
            [
                f'    <clip id="{clip_id}">',
                f"      <name>{clip_name}</name>",
                f"      <path>{clip_path}</path>",
                f"      <duration_seconds>{float(scene.get('duration_seconds', 0.0) or 0.0):.3f}</duration_seconds>",
                "    </clip>",
            ]
        )
    xml_lines.extend(["  </sequence>", "</project>"])
    output_path = target_root / "premiere_project.xml"
    output_path.write_text("\n".join(xml_lines), encoding="utf-8")
    return {"project_path": str(output_path), "clip_count": len(export_payload.get("scenes", []))}


def export_for_json(export_payload: dict[str, Any], target_root: Path) -> dict[str, Any]:
    output_path = target_root / "export_manifest.json"
    write_json(output_path, export_payload)
    return {"manifest_path": str(output_path)}


def main() -> None:
    args = parse_args()
    headline("Export Package for External Tools")
    cfg = load_config()

    artifacts = resolve_episode_artifacts(cfg, args.episode_id)
    if not artifacts:
        info("No generated episode artifacts were found.")
        raise SystemExit(1)

    production_package = load_production_package(artifacts)
    if not production_package:
        info(f"No production package found for {artifacts.get('episode_id') or args.episode_id or 'latest'}.")
        raise SystemExit(1)

    target_root = export_root(cfg, str(artifacts.get("episode_id", "") or "latest"), args.format)
    target_root.mkdir(parents=True, exist_ok=True)
    export_payload = build_common_export_payload(artifacts, production_package, args.format, target_root)

    copied_top_level: dict[str, str] = {}
    if args.copy_media:
        copied_top_level = copy_referenced_media(export_payload, target_root / "media")
    export_payload["copied_top_level_media"] = copied_top_level

    export_manifest_path = target_root / "export_manifest.json"
    write_json(export_manifest_path, export_payload)

    export_handlers = {
        "davinci": export_for_davinci,
        "premiere": export_for_premiere,
        "json": export_for_json,
    }
    handler_result = export_handlers[args.format](export_payload, target_root)

    summary = {
        "episode_id": export_payload.get("episode_id", ""),
        "format": args.format,
        "export_root": str(target_root),
        "export_manifest": str(export_manifest_path),
        "copy_media": bool(args.copy_media),
        **handler_result,
    }
    write_json(target_root / "export_summary.json", summary)

    ok(f"Exported package for {args.format}: {target_root}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
