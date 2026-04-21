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
    add_shared_worker_arguments,
    distributed_item_lease,
    distributed_step_runtime_root,
    LiveProgressReporter,
    SCRIPT_DIR,
    coalesce_text,
    detect_tool,
    dominant_language,
    merge_language_counts,
    error,
    has_primary_person_name,
    headline,
    info,
    is_background_person_name,
    load_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    pip_install_command,
    read_json,
    rerun_in_runtime,
    normalize_language_code,
    resolve_project_path,
    runtime_python,
    shared_worker_id_for_args,
    shared_workers_enabled_for_args,
    write_json,
    write_text,
)

PROCESS_VERSION = 2
DOWNLOAD_METADATA_FILE = ".foundation_download.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare foundation training for image/video/voice"
    )
    parser.add_argument("--download-models", action="store_true", help="Download configured base models and check for updates.")
    parser.add_argument("--skip-downloads", action="store_true", help="Skip automatic base downloads and update checks for this run.")
    parser.add_argument("--force", action="store_true", help="Recreate existing exports and manifests.")
    parser.add_argument("--episode", help="Optionally include only one specific source episode.")
    parser.add_argument("--limit-characters", type=int, default=0, help="Optionally limit the number of candidates.")
    add_shared_worker_arguments(parser)
    return parser.parse_args()


def slugify(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in coalesce_text(value))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "figur"


def build_download_targets(cfg: dict) -> list[dict]:
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    use_local_voice_models = bool(foundation_cfg.get("use_local_character_voice_models", True))
    targets: list[dict] = []
    for kind, key in (
        ("image", "image_base_model"),
        ("video", "video_base_model"),
        ("voice", "voice_base_model"),
    ):
        if kind == "voice" and use_local_voice_models:
            continue
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


def download_metadata_path(target: dict) -> Path:
    return Path(str(target.get("target_dir", ""))) / DOWNLOAD_METADATA_FILE


def read_download_metadata(target: dict) -> dict:
    metadata_path = download_metadata_path(target)
    return read_json(metadata_path, {})


def write_download_metadata(target: dict, payload: dict) -> None:
    metadata_path = download_metadata_path(target)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(metadata_path, payload)


def model_target_ready(target: dict) -> bool:
    target_dir = Path(str(target.get("target_dir", "")))
    if not target_dir.exists():
        return False
    return any(path.name != DOWNLOAD_METADATA_FILE for path in target_dir.iterdir())


def incomplete_download_files(target: dict) -> list[Path]:
    target_dir = Path(str(target.get("target_dir", "")))
    cache_dir = target_dir / ".cache" / "huggingface" / "download"
    if not cache_dir.exists():
        return []
    return sorted(path for path in cache_dir.rglob("*.incomplete") if path.is_file())


def cleanup_incomplete_download_files(target: dict) -> int:
    removed = 0
    for path in incomplete_download_files(target):
        try:
            path.unlink()
            removed += 1
        except Exception:
            continue
    return removed


def infer_local_revision_from_cache(target: dict) -> str:
    target_dir = Path(str(target.get("target_dir", "")))
    cache_dir = target_dir / ".cache" / "huggingface" / "download"
    if not cache_dir.exists():
        return ""
    for metadata_file in sorted(cache_dir.rglob("*.metadata")):
        try:
            lines = metadata_file.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        if not lines:
            continue
        candidate = coalesce_text(lines[0])
        if len(candidate) >= 12:
            return candidate
    return ""


def local_download_state(target: dict) -> dict:
    metadata = read_download_metadata(target)
    local_revision = coalesce_text(metadata.get("revision", ""))
    local_model_id = coalesce_text(metadata.get("model_id", ""))
    inferred_revision = ""
    if not local_revision and model_target_ready(target):
        inferred_revision = infer_local_revision_from_cache(target)
        if inferred_revision:
            local_revision = inferred_revision
    return {
        "metadata": metadata,
        "local_revision": local_revision,
        "local_model_id": local_model_id,
        "inferred_revision": inferred_revision,
    }


def resolve_download_action(target: dict, remote_revision: str) -> str:
    if not model_target_ready(target):
        return "download"
    state = local_download_state(target)
    local_revision = coalesce_text(state.get("local_revision", ""))
    local_model_id = coalesce_text(state.get("local_model_id", ""))
    has_incomplete_files = bool(incomplete_download_files(target))
    if local_model_id and local_model_id != coalesce_text(target.get("model_id", "")):
        return "update"
    if has_incomplete_files and (not remote_revision or local_revision == remote_revision):
        return "cleanup"
    if has_incomplete_files:
        return "repair"
    if not remote_revision:
        return "skip"
    if local_revision != remote_revision:
        return "update"
    return "skip"


def fetch_remote_revision(api, target: dict, token: str) -> str:
    info_payload = api.model_info(repo_id=target["model_id"], token=token or None)
    return coalesce_text(getattr(info_payload, "sha", ""))


def refresh_target_dir(target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)


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


def original_voice_candidates(rows: list[dict], character_name: str) -> list[dict]:
    candidates: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        episode_id = coalesce_text(row.get("episode_id", ""))
        scene_id = coalesce_text(row.get("scene_id", ""))
        for index, segment in enumerate(row.get("transcript_segments", []) or []):
            if not isinstance(segment, dict):
                continue
            speaker_name = coalesce_text(segment.get("speaker_name", ""))
            if speaker_name.lower() != character_name.lower():
                continue
            audio_path = Path(str(segment.get("audio_file", "")))
            if not audio_path.exists():
                continue
            start_seconds = float(segment.get("start", 0.0) or 0.0)
            end_seconds = float(segment.get("end", start_seconds) or start_seconds)
            duration_seconds = max(0.0, end_seconds - start_seconds)
            candidates.append(
                {
                    "episode_id": episode_id,
                    "scene_id": scene_id,
                    "segment_id": coalesce_text(segment.get("segment_id", "")) or f"{scene_id}_voice_{index + 1:03d}",
                    "audio_path": audio_path,
                    "duration_seconds": duration_seconds,
                    "language": normalize_language_code(segment.get("language", ""), row.get("language", "")),
                    "text": coalesce_text(segment.get("text", "")),
                }
            )
    candidates.sort(
        key=lambda row: (
            -float(row.get("duration_seconds", 0.0) or 0.0),
            row.get("episode_id", ""),
            row.get("scene_id", ""),
            row.get("segment_id", ""),
        )
    )
    return candidates


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
        pip_install_command(runtime_python(), "--upgrade", package_name),
        check=True,
    )


def download_models(cfg: dict, targets: list[dict]) -> list[dict]:
    ensure_runtime_package("huggingface_hub", "huggingface_hub")
    from huggingface_hub import HfApi, snapshot_download

    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    token_env = coalesce_text(foundation_cfg.get("huggingface_token_env", "HF_TOKEN")) or "HF_TOKEN"
    token = coalesce_text(os.environ.get(token_env, ""))
    api = HfApi()
    results: list[dict] = []
    for target in targets:
        target_dir = Path(target["target_dir"])
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        state = local_download_state(target)
        remote_revision = ""
        try:
            remote_revision = fetch_remote_revision(api, target, token)
        except Exception as exc:
            if model_target_ready(target):
                info(
                    f"Could not check the revision for {target['model_id']}. Continuing with the local copy: {exc}"
                )
            else:
                raise

        action = resolve_download_action(target, remote_revision)
        if action == "skip":
            if state.get("inferred_revision") and not read_download_metadata(target):
                write_download_metadata(
                    target,
                    {
                        "model_id": target["model_id"],
                        "kind": target["kind"],
                        "revision": state["local_revision"],
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "snapshot_path": str(target_dir),
                    },
                )
            results.append(
                {
                    **target,
                    "snapshot_path": str(target_dir),
                    "downloaded": False,
                    "updated": False,
                    "revision": remote_revision or coalesce_text(state.get("local_revision", "")),
                }
            )
            continue
        if action == "cleanup":
            removed_incomplete = cleanup_incomplete_download_files(target)
            write_download_metadata(
                target,
                {
                    "model_id": target["model_id"],
                    "kind": target["kind"],
                    "revision": remote_revision or coalesce_text(state.get("local_revision", "")),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "snapshot_path": str(target_dir),
                    "cleaned_incomplete_files": removed_incomplete,
                },
            )
            results.append(
                {
                    **target,
                    "snapshot_path": str(target_dir),
                    "downloaded": False,
                    "updated": False,
                    "cleaned": True,
                    "revision": remote_revision or coalesce_text(state.get("local_revision", "")),
                    "cleaned_incomplete_files": removed_incomplete,
                }
            )
            continue

        if action == "repair":
            info(f"Repariere unvollstaendigen Basis-Download ({target['kind']}): {target['model_id']}")
        elif action == "update":
            info(f"Aktualisiere Basis-Modell ({target['kind']}): {target['model_id']}")
        else:
            info(f"Lade Basis-Modell ({target['kind']}): {target['model_id']}")

        snapshot_path = snapshot_download(
            repo_id=target["model_id"],
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            token=token or None,
            resume_download=True,
            revision=remote_revision or None,
        )
        removed_incomplete = cleanup_incomplete_download_files(target)
        final_revision = remote_revision
        if not final_revision:
            try:
                final_revision = fetch_remote_revision(api, target, token)
            except Exception:
                final_revision = ""
        write_download_metadata(
            target,
            {
                "model_id": target["model_id"],
                "kind": target["kind"],
                "revision": final_revision,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "snapshot_path": str(snapshot_path),
                "repaired_incomplete_files": removed_incomplete,
            },
        )
        results.append(
            {
                **target,
                "snapshot_path": str(snapshot_path),
                "downloaded": action == "download",
                "updated": action in {"update", "repair"},
                "repaired": action == "repair",
                "revision": final_revision,
                "repaired_incomplete_files": removed_incomplete,
            }
        )
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
    seen_sources: set[str] = set()
    original_candidates = original_voice_candidates(rows, character["name"])
    for index, source in enumerate(original_candidates[:max_voice], start=1):
        source_path = Path(str(source.get("audio_path", "")))
        source_key = str(source_path.resolve()) if source_path.exists() else str(source_path)
        if not source_path.exists() or source_key in seen_sources:
            continue
        target = voice_root / f"{character['slug']}_{index:03d}{source_path.suffix.lower() or '.wav'}"
        if not target.exists() or force:
            shutil.copy2(source_path, target)
        voice_samples.append(
            {
                "path": str(target),
                "source": str(source_path),
                "source_type": "original_segment",
                "episode_id": coalesce_text(source.get("episode_id", "")),
                "scene_id": coalesce_text(source.get("scene_id", "")),
                "segment_id": coalesce_text(source.get("segment_id", "")),
                "language": normalize_language_code(source.get("language", "")),
                "duration_seconds": round(float(source.get("duration_seconds", 0.0) or 0.0), 3),
                "text": coalesce_text(source.get("text", "")),
            }
        )
        seen_sources.add(source_key)

    next_index = len(voice_samples) + 1
    if len(voice_samples) < max_voice:
        for source in voice_reference_candidates(cfg, character["name"]):
            if len(voice_samples) >= max_voice:
                break
            source_key = str(source.resolve()) if source.exists() else str(source)
            if not source.exists() or source_key in seen_sources:
                continue
            target = voice_root / f"{character['slug']}_{next_index:03d}{source.suffix.lower() or '.wav'}"
            if not target.exists() or force:
                shutil.copy2(source, target)
            voice_samples.append(
                {
                    "path": str(target),
                    "source": str(source),
                    "source_type": "curated_reference",
                    "language": "",
                    "duration_seconds": 0.0,
                    "text": "",
                }
            )
            seen_sources.add(source_key)
            next_index += 1

    voice_language_counts = merge_language_counts(
        {
            normalize_language_code(sample.get("language", "")): 1
            for sample in voice_samples
            if normalize_language_code(sample.get("language", ""))
        }
    )

    return {
        "name": character["name"],
        "slug": character["slug"],
        "priority": bool(character.get("priority", False)),
        "scene_count": int(character.get("scene_count", 0)),
        "line_count": int(character.get("line_count", 0)),
        "frame_samples": frame_samples,
        "video_samples": video_samples,
        "voice_samples": voice_samples,
        "voice_language_counts": voice_language_counts,
        "dominant_language": dominant_language(voice_language_counts),
        "original_voice_sample_count": sum(1 for sample in voice_samples if sample.get("source_type") == "original_segment"),
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
        "# Foundation Training Plan",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Characters with training material: {len(manifests)}",
        f"- Source filter: {episode_filter or 'all processed episodes'}",
        "",
        "## Characters",
        "",
    ]
    for manifest in manifests:
        lines.extend(
            [
                f"### {manifest['name']}",
                "",
                f"- Priority: {'yes' if manifest.get('priority') else 'no'}",
                f"- Scene count hint: {manifest.get('scene_count', 0)}",
                f"- Line count hint: {manifest.get('line_count', 0)}",
                f"- Exported frames: {len(manifest.get('frame_samples', []))}",
                f"- Exported clips: {len(manifest.get('video_samples', []))}",
                f"- Reference voices: {len(manifest.get('voice_samples', []))}",
                f"- Original voice segments: {int(manifest.get('original_voice_sample_count', 0) or 0)}",
                f"- Dominant voice language: {coalesce_text(manifest.get('dominant_language', '')) or 'auto-detected later'}",
                "",
            ]
        )
    if downloads:
        lines.extend(["## Base Downloads", ""])
        for target in downloads:
            status = "updated" if target.get("updated") else "downloaded" if target.get("downloaded") else "current"
            revision = coalesce_text(target.get("revision", ""))
            revision_text = f" | Revision: {revision}" if revision else ""
            lines.append(
                f"- {target['kind']}: {target['model_id']} -> {target.get('snapshot_path', target.get('target_dir', ''))} "
                f"({status}{revision_text})"
            )
        lines.append("")
    write_text(markdown_path, "\n".join(lines).strip() + "\n")
    return json_path, markdown_path


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Prepare Foundation Training")
    cfg = load_config()
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)
    autosave_target = coalesce_text(args.episode or "global") or "global"
    mark_step_started("09_prepare_foundation_training", autosave_target, {"episode_filter": coalesce_text(args.episode or "")})
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

    try:
        if not candidates:
            info("No named main characters with enough material found for foundation training.")
            return

        manifests: list[dict] = []
        manifest_root = resolve_project_path(cfg["paths"]["foundation_manifests"])
        manifest_root.mkdir(parents=True, exist_ok=True)
        lease_root = distributed_step_runtime_root("09_prepare_foundation_training", autosave_target)
        reporter = LiveProgressReporter(
            script_name="09_prepare_foundation_training.py",
            total=len(candidates),
            phase_label="Prepare Foundation Data",
        )
        if shared_workers:
            info(f"Shared NAS workers: enabled ({worker_id})")
        for index, character in enumerate(candidates, start=1):
            character_rows = collect_character_rows(rows, character["name"], known_clusters)
            if not character_rows:
                continue
            with distributed_item_lease(
                root=lease_root,
                lease_name=str(character.get("slug", "")),
                cfg=cfg,
                worker_id=worker_id,
                enabled=shared_workers,
                meta={"step": "09_prepare_foundation_training", "character": character["name"], "slug": character["slug"], "worker_id": worker_id},
            ) as acquired:
                if not acquired:
                    continue
                info(f"Bereite Trainingsdaten vor: {character['name']}")
                manifest = prepare_character_dataset(ffmpeg_path, character, character_rows, cfg, force=args.force)
                manifests.append(manifest)
                write_json(manifest_root / f"{character['slug']}_manifest.json", manifest)
                reporter.update(
                    index,
                    current_label=str(character.get("name", "")),
                    extra_label=f"Manifests so far: {len(manifests)}",
                )
        reporter.finish(current_label="Foundation Training", extra_label=f"Total manifests: {len(manifests)}")

        download_targets = build_download_targets(cfg)
        downloaded: list[dict] = []
        missing_downloads = any(not model_target_ready(target) for target in download_targets)
        check_updates = bool(foundation_cfg.get("check_model_updates", True))
        wants_downloads = not args.skip_downloads and bool(
            args.download_models or foundation_cfg.get("download_base_models", True) or missing_downloads or check_updates
        )
        if wants_downloads and download_targets:
            downloaded = download_models(cfg, download_targets)
        elif wants_downloads:
            info("No base model configured. Only training data was prepared.")

        plan_json, plan_md = write_training_plan(cfg, manifests, downloaded or download_targets, coalesce_text(args.episode or ""))
        mark_step_completed(
            "09_prepare_foundation_training",
            autosave_target,
            {"manifest_count": len(manifests), "plan_json": str(plan_json), "plan_markdown": str(plan_md)},
        )
        ok(f"Foundation Training vorbereitet: {plan_json}")
        ok(f"Trainingsplan geschrieben: {plan_md}")
    except Exception as exc:
        mark_step_failed("09_prepare_foundation_training", str(exc), autosave_target)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise

