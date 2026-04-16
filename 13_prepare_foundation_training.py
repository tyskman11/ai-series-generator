#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from pipeline_common import (
    SCRIPT_DIR,
    coalesce_text,
    detect_tool,
    error,
    has_primary_person_name,
    headline,
    info,
    is_background_person_name,
    load_config,
    ok,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
    runtime_python,
    write_json,
    write_text,
)

PROCESS_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Foundation-Training fuer Bild/Video/Stimmen vorbereiten"
    )
    parser.add_argument("--download-models", action="store_true", help="Lade konfigurierte Basis-Modelle herunter.")
    parser.add_argument("--skip-downloads", action="store_true", help="Ueberspringe automatische Basis-Downloads fuer diesen Lauf.")
    parser.add_argument("--force", action="store_true", help="Erzeuge vorhandene Exporte und Manifeste neu.")
    parser.add_argument("--episode", help="Optional nur eine bestimmte Quellfolge beruecksichtigen.")
    parser.add_argument("--limit-characters", type=int, default=0, help="Optional Kandidatenzahl begrenzen.")
    return parser.parse_args()


def slugify(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in coalesce_text(value))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "figur"


def build_download_targets(cfg: dict) -> list[dict]:
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    targets: list[dict] = []
    for kind, key in (
        ("image", "image_base_model"),
        ("video", "video_base_model"),
        ("voice", "voice_base_model"),
    ):
        model_id = coalesce_text(foundation_cfg.get(key, ""))
        if not model_id:
            continue
        targets.append(
            {
                "kind": kind,
                "model_id": model_id,
                "target_dir": str(resolve_project_path(cfg["paths"]["foundation_downloads"]) / kind / slugify(model_id)),
            }
        )
    return targets


def model_target_ready(target: dict) -> bool:
    target_dir = Path(str(target.get("target_dir", "")))
    if not target_dir.exists():
        return False
    return any(target_dir.iterdir())


def character_training_candidates(char_map: dict, series_model: dict, cfg: dict) -> list[dict]:
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    min_scenes = int(foundation_cfg.get("min_character_scene_count", 3))
    min_lines = int(foundation_cfg.get("min_character_line_count", 3))
    stats_by_name = {
        coalesce_text(row.get("name", "")): row
        for row in series_model.get("characters", [])
        if has_primary_person_name(coalesce_text(row.get("name", "")))
    }
    candidates: dict[str, dict] = {}
    for cluster_id, payload in (char_map.get("clusters", {}) or {}).items():
        if payload.get("ignored"):
            continue
        name = coalesce_text(payload.get("name", ""))
        if not has_primary_person_name(name) or is_background_person_name(name):
            continue
        stats = stats_by_name.get(name, {})
        scene_count = max(int(payload.get("scene_count", 0) or 0), int(stats.get("scene_count", 0) or 0))
        line_count = int(stats.get("line_count", 0) or 0)
        if scene_count < min_scenes and line_count < min_lines:
            continue
        existing = candidates.setdefault(
            name,
            {
                "name": name,
                "slug": slugify(name),
                "priority": bool(payload.get("priority", False) or stats.get("priority", False)),
                "scene_count": scene_count,
                "line_count": line_count,
                "face_clusters": [],
            },
        )
        existing["priority"] = existing["priority"] or bool(payload.get("priority", False) or stats.get("priority", False))
        existing["scene_count"] = max(existing["scene_count"], scene_count)
        existing["line_count"] = max(existing["line_count"], line_count)
        if cluster_id not in existing["face_clusters"]:
            existing["face_clusters"].append(cluster_id)
    return sorted(
        candidates.values(),
        key=lambda row: (
            0 if row.get("priority") else 1,
            -(int(row.get("scene_count", 0)) + int(row.get("line_count", 0))),
            row.get("name", ""),
        ),
    )


def read_dataset_rows(cfg: dict, episode_filter: str = "") -> list[dict]:
    dataset_root = resolve_project_path(cfg["paths"]["datasets_video_training"])
    rows: list[dict] = []
    for dataset_path in sorted(dataset_root.glob("*_dataset.json")):
        episode_id = dataset_path.stem.replace("_dataset", "")
        if episode_filter and episode_filter not in {episode_id, dataset_path.name, dataset_path.stem}:
            continue
        for row in read_json(dataset_path, []):
            prepared = dict(row)
            prepared["episode_id"] = coalesce_text(row.get("episode_id", "")) or episode_id
            rows.append(prepared)
    return rows


def cluster_name_map(char_map: dict) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for cluster_id, payload in (char_map.get("clusters", {}) or {}).items():
        name = coalesce_text(payload.get("name", ""))
        if has_primary_person_name(name) and not is_background_person_name(name) and not payload.get("ignored"):
            mapping[str(cluster_id)] = name
    return mapping


def scene_character_names(row: dict, known_clusters: dict[str, str]) -> set[str]:
    names: set[str] = set()
    for name in row.get("characters_visible", []) or []:
        candidate = coalesce_text(name)
        if has_primary_person_name(candidate):
            names.add(candidate)
        mapped = known_clusters.get(candidate, "")
        if mapped:
            names.add(mapped)
    for cluster_id in row.get("face_clusters", []) or []:
        mapped = known_clusters.get(coalesce_text(cluster_id), "")
        if mapped:
            names.add(mapped)
    for segment in row.get("transcript_segments", []) or []:
        speaker_name = coalesce_text(segment.get("speaker_name", ""))
        if has_primary_person_name(speaker_name):
            names.add(speaker_name)
        for name in segment.get("visible_character_names", []) or []:
            candidate = coalesce_text(name)
            if has_primary_person_name(candidate):
                names.add(candidate)
            mapped = known_clusters.get(candidate, "")
            if mapped:
                names.add(mapped)
        for cluster_id in segment.get("visible_face_clusters", []) or []:
            mapped = known_clusters.get(coalesce_text(cluster_id), "")
            if mapped:
                names.add(mapped)
    return names


def collect_character_rows(rows: list[dict], character_name: str, known_clusters: dict[str, str]) -> list[dict]:
    return [row for row in rows if character_name in scene_character_names(row, known_clusters)]


def voice_reference_candidates(cfg: dict, character_name: str) -> list[Path]:
    voice_samples_dir = resolve_project_path(cfg["paths"]["voice_samples"])
    voice_models_dir = resolve_project_path(cfg["paths"]["voice_models"])
    slug = slugify(character_name)
    matches: list[Path] = []
    for candidate in sorted(voice_samples_dir.glob(f"{slug}*.wav")):
        matches.append(candidate)
    model_path = voice_models_dir / f"{slug}_voice_model.json"
    if model_path.exists():
        payload = read_json(model_path, {})
        ref_audio = coalesce_text(payload.get("reference_audio", ""))
        if ref_audio:
            candidate = Path(ref_audio)
            if candidate.exists() and candidate not in matches:
                matches.append(candidate)
    return matches


def ffmpeg_resize_filter(width: int, height: int) -> str:
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
    )


def export_frame(ffmpeg_path: Path, video_file: Path, timestamp_seconds: float, output_path: Path, width: int, height: int, force: bool) -> bool:
    if output_path.exists() and not force:
        return True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg_path),
        "-y" if force else "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, timestamp_seconds):.3f}",
        "-i",
        str(video_file),
        "-frames:v",
        "1",
        "-vf",
        ffmpeg_resize_filter(width, height),
        str(output_path),
    ]
    result = subprocess.run(command, check=False)
    return result.returncode == 0 and output_path.exists()


def export_clip(
    ffmpeg_path: Path,
    video_file: Path,
    start_seconds: float,
    end_seconds: float,
    output_path: Path,
    width: int,
    height: int,
    force: bool,
) -> bool:
    if output_path.exists() and not force:
        return True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.6, float(end_seconds) - float(start_seconds))
    command = [
        str(ffmpeg_path),
        "-y" if force else "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start_seconds):.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(video_file),
        "-an",
        "-vf",
        ffmpeg_resize_filter(width, height),
        "-r",
        "12",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        str(output_path),
    ]
    result = subprocess.run(command, check=False)
    return result.returncode == 0 and output_path.exists()


def ensure_runtime_package(module_name: str, package_name: str) -> None:
    if importlib.util.find_spec(module_name) is not None:
        return
    subprocess.run(
        [str(runtime_python()), "-m", "pip", "install", "--upgrade", package_name],
        check=True,
    )


def download_models(cfg: dict, targets: list[dict]) -> list[dict]:
    ensure_runtime_package("huggingface_hub", "huggingface_hub")
    from huggingface_hub import snapshot_download

    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    token_env = coalesce_text(foundation_cfg.get("huggingface_token_env", "HF_TOKEN")) or "HF_TOKEN"
    token = coalesce_text(os.environ.get(token_env, ""))
    results: list[dict] = []
    for target in targets:
        target_dir = Path(target["target_dir"])
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if model_target_ready(target):
            results.append({**target, "snapshot_path": str(target_dir), "downloaded": False})
            continue
        info(f"Lade Basis-Modell ({target['kind']}): {target['model_id']}")
        snapshot_path = snapshot_download(
            repo_id=target["model_id"],
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            token=token or None,
            resume_download=True,
        )
        results.append({**target, "snapshot_path": str(snapshot_path), "downloaded": True})
    return results


def prepare_character_dataset(
    ffmpeg_path: Path,
    character: dict,
    rows: list[dict],
    cfg: dict,
    force: bool,
) -> dict:
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    width = int(foundation_cfg.get("frame_width", 1280))
    height = int(foundation_cfg.get("frame_height", 720))
    max_frames = int(foundation_cfg.get("max_frame_samples_per_character", 48))
    max_clips = int(foundation_cfg.get("max_video_clips_per_character", 18))
    max_voice = int(foundation_cfg.get("max_voice_segments_per_character", 48))

    frames_root = resolve_project_path(cfg["paths"]["foundation_frames"]) / character["slug"]
    video_root = resolve_project_path(cfg["paths"]["foundation_video"]) / character["slug"]
    voice_root = resolve_project_path(cfg["paths"]["foundation_voice"]) / character["slug"]
    voice_root.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -len(row.get("transcript", "")),
            row.get("episode_id", ""),
            row.get("scene_id", ""),
        ),
    )

    frame_samples: list[dict] = []
    video_samples: list[dict] = []
    for row_index, row in enumerate(sorted_rows):
        video_file = Path(str(row.get("video_file", "")))
        if not video_file.exists():
            continue
        segment_rows = row.get("transcript_segments", []) or []
        midpoint = max(0.0, float(row.get("duration_seconds", 0.0) or 0.0) / 2.0)
        if segment_rows:
            first_segment = segment_rows[0]
            midpoint = max(0.0, (float(first_segment.get("start", 0.0) or 0.0) + float(first_segment.get("end", 0.0) or 0.0)) / 2.0)

        if len(frame_samples) < max_frames:
            frame_path = frames_root / f"{coalesce_text(row.get('episode_id', 'folge'))}_{coalesce_text(row.get('scene_id', str(row_index)))}.png"
            if export_frame(ffmpeg_path, video_file, midpoint, frame_path, width, height, force):
                frame_samples.append(
                    {
                        "episode_id": row.get("episode_id", ""),
                        "scene_id": row.get("scene_id", ""),
                        "path": str(frame_path),
                        "timestamp_seconds": midpoint,
                    }
                )

        if len(video_samples) < max_clips:
            start_seconds = midpoint
            end_seconds = midpoint + min(3.0, max(0.8, float(row.get("duration_seconds", 0.0) or 0.0)))
            clip_path = video_root / f"{coalesce_text(row.get('episode_id', 'folge'))}_{coalesce_text(row.get('scene_id', str(row_index)))}.mp4"
            if export_clip(ffmpeg_path, video_file, start_seconds, end_seconds, clip_path, width, height, force):
                video_samples.append(
                    {
                        "episode_id": row.get("episode_id", ""),
                        "scene_id": row.get("scene_id", ""),
                        "path": str(clip_path),
                        "start_seconds": start_seconds,
                        "end_seconds": end_seconds,
                    }
                )
        if len(frame_samples) >= max_frames and len(video_samples) >= max_clips:
            break

    voice_samples: list[dict] = []
    for index, source in enumerate(voice_reference_candidates(cfg, character["name"])[:max_voice], start=1):
        target = voice_root / f"{character['slug']}_{index:03d}{source.suffix.lower() or '.wav'}"
        if not target.exists() or force:
            shutil.copy2(source, target)
        voice_samples.append({"path": str(target), "source": str(source)})

    return {
        "name": character["name"],
        "slug": character["slug"],
        "priority": bool(character.get("priority", False)),
        "scene_count": int(character.get("scene_count", 0)),
        "line_count": int(character.get("line_count", 0)),
        "frame_samples": frame_samples,
        "video_samples": video_samples,
        "voice_samples": voice_samples,
    }


def write_training_plan(
    cfg: dict,
    manifests: list[dict],
    downloads: list[dict],
    episode_filter: str,
) -> tuple[Path, Path]:
    plans_dir = resolve_project_path(cfg["paths"]["foundation_plans"])
    plans_dir.mkdir(parents=True, exist_ok=True)
    json_path = plans_dir / "foundation_training_plan.json"
    markdown_path = plans_dir / "foundation_training_plan.md"
    summary = {
        "process_version": PROCESS_VERSION,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episode_filter": episode_filter,
        "character_count": len(manifests),
        "characters": manifests,
        "downloads": downloads,
    }
    write_json(json_path, summary)

    lines = [
        "# Foundation-Training Plan",
        "",
        f"- Generiert am: {summary['generated_at']}",
        f"- Figuren mit Trainingsmaterial: {len(manifests)}",
        f"- Quellenfilter: {episode_filter or 'alle verarbeiteten Folgen'}",
        "",
        "## Figuren",
        "",
    ]
    for manifest in manifests:
        lines.extend(
            [
                f"### {manifest['name']}",
                "",
                f"- Prioritaet: {'ja' if manifest.get('priority') else 'nein'}",
                f"- Szenen-Hinweis: {manifest.get('scene_count', 0)}",
                f"- Zeilen-Hinweis: {manifest.get('line_count', 0)}",
                f"- Exportierte Frames: {len(manifest.get('frame_samples', []))}",
                f"- Exportierte Clips: {len(manifest.get('video_samples', []))}",
                f"- Referenzstimmen: {len(manifest.get('voice_samples', []))}",
                "",
            ]
        )
    if downloads:
        lines.extend(["## Basis-Downloads", ""])
        for target in downloads:
            lines.append(f"- {target['kind']}: {target['model_id']} -> {target.get('snapshot_path', target.get('target_dir', ''))}")
        lines.append("")
    write_text(markdown_path, "\n".join(lines).strip() + "\n")
    return json_path, markdown_path


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Foundation-Training vorbereiten")
    cfg = load_config()
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}

    ffmpeg_bin = resolve_project_path("tools/ffmpeg/bin")
    ffmpeg_path = detect_tool(ffmpeg_bin, "ffmpeg")
    char_map = read_json(resolve_project_path(cfg["paths"]["character_map"]), {"clusters": {}, "aliases": {}})
    series_model = read_json(resolve_project_path(cfg["paths"]["series_model"]), {})
    rows = read_dataset_rows(cfg, coalesce_text(args.episode or ""))
    known_clusters = cluster_name_map(char_map)
    candidates = character_training_candidates(char_map, series_model, cfg)
    if args.limit_characters and args.limit_characters > 0:
        candidates = candidates[: args.limit_characters]

    if not candidates:
        info("Keine benannten Hauptfiguren mit genug Material fuer Foundation-Training gefunden.")
        return

    manifests: list[dict] = []
    manifest_root = resolve_project_path(cfg["paths"]["foundation_manifests"])
    manifest_root.mkdir(parents=True, exist_ok=True)
    for character in candidates:
        character_rows = collect_character_rows(rows, character["name"], known_clusters)
        if not character_rows:
            continue
        info(f"Bereite Trainingsdaten vor: {character['name']}")
        manifest = prepare_character_dataset(ffmpeg_path, character, character_rows, cfg, force=args.force)
        manifests.append(manifest)
        write_json(manifest_root / f"{character['slug']}_manifest.json", manifest)

    download_targets = build_download_targets(cfg)
    downloaded: list[dict] = []
    missing_downloads = any(not model_target_ready(target) for target in download_targets)
    wants_downloads = not args.skip_downloads and bool(
        args.download_models or foundation_cfg.get("download_base_models", True) or missing_downloads
    )
    if wants_downloads and download_targets:
        downloaded = download_models(cfg, download_targets)
    elif wants_downloads:
        info("Kein Basis-Modell in der Config gesetzt. Es wurden nur Trainingsdaten vorbereitet.")

    plan_json, plan_md = write_training_plan(cfg, manifests, downloaded or download_targets, coalesce_text(args.episode or ""))
    ok(f"Foundation-Training vorbereitet: {plan_json}")
    ok(f"Trainingsplan geschrieben: {plan_md}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
