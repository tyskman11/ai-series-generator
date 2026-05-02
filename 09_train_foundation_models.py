#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "ai_series_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse
import math
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from support_scripts.pipeline_common import (
    add_shared_worker_arguments,
    distributed_item_lease,
    distributed_step_runtime_root,
    LiveProgressReporter,
    coalesce_text,
    dominant_language,
    error,
    headline,
    info,
    load_config,
    load_step_autosave,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    merge_language_counts,
    normalize_language_code,
    ok,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
    shared_worker_id_for_args,
    shared_workers_enabled_for_args,
    write_json,
)

PROCESS_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local foundation packs for image, video and voice")
    parser.add_argument("--limit-characters", type=int, default=0, help="Optionally train only the first N characters.")
    parser.add_argument("--character", help="Optionally train only one specific character.")
    parser.add_argument("--force", action="store_true", help="Intentionally retrain existing foundation packs.")
    add_shared_worker_arguments(parser)
    return parser.parse_args()


def load_manifests(cfg: dict, character_filter: str = "", limit_characters: int = 0) -> list[dict]:
    manifest_root = resolve_project_path(cfg["paths"]["foundation_manifests"])
    manifests = []
    for path in sorted(manifest_root.glob("*_manifest.json")):
        payload = read_json(path, {})
        if not payload:
            continue
        if character_filter and coalesce_text(payload.get("name", "")).lower() != character_filter.lower():
            continue
        manifests.append(payload)
    if limit_characters > 0:
        manifests = manifests[:limit_characters]
    return manifests


def image_stats(paths: list[Path]) -> dict:
    sample_count = 0
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    histograms: list[np.ndarray] = []
    for path in paths:
        if not path.exists():
            continue
        with Image.open(path).convert("RGB") as image:
            array = np.asarray(image, dtype=np.float32) / 255.0
        means.append(array.mean(axis=(0, 1)))
        stds.append(array.std(axis=(0, 1)))
        channel_histograms = []
        for channel in range(3):
            hist, _edges = np.histogram(array[:, :, channel], bins=32, range=(0.0, 1.0), density=True)
            channel_histograms.append(hist.astype(np.float32))
        histograms.append(np.concatenate(channel_histograms))
        sample_count += 1
    if not sample_count:
        return {"sample_count": 0}
    return {
        "sample_count": sample_count,
        "rgb_mean": np.mean(means, axis=0).round(6).tolist(),
        "rgb_std": np.mean(stds, axis=0).round(6).tolist(),
        "histogram_32x3": np.mean(histograms, axis=0).round(6).tolist(),
    }


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frame_rate = handle.getframerate() or 24000
        frame_count = handle.getnframes() or 0
    return frame_count / frame_rate if frame_rate else 0.0


def voice_stats(entries: list[dict], cfg: dict) -> dict:
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    durations = []
    language_counts: dict[str, int] = {}
    original_voice_sample_count = 0
    sample_paths: list[str] = []
    reference_segments: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path = Path(str(entry.get("path", "")))
        if not path.exists():
            continue
        durations.append(wav_duration_seconds(path))
        sample_paths.append(str(path))
        if entry.get("source_type") == "original_segment":
            original_voice_sample_count += 1
        language = normalize_language_code(entry.get("language", ""))
        if language:
            language_counts[language] = language_counts.get(language, 0) + 1
        if entry.get("source_type") == "original_segment":
            reference_segments.append(
                {
                    "path": str(path),
                    "episode_id": coalesce_text(entry.get("episode_id", "")),
                    "scene_id": coalesce_text(entry.get("scene_id", "")),
                    "segment_id": coalesce_text(entry.get("segment_id", "")),
                    "language": language,
                    "duration_seconds": round(float(entry.get("duration_seconds", 0.0) or 0.0), 3),
                    "text": coalesce_text(entry.get("text", "")),
                }
            )
    if not durations:
        return {
            "sample_count": 0,
            "duration_seconds_total": 0.0,
            "quality_score": 0.0,
            "clone_ready": False,
            "language_counts": {},
            "dominant_language": "",
            "original_voice_sample_count": 0,
            "sample_paths": [],
            "reference_segments": [],
        }
    total_duration = sum(durations)
    min_total_duration = float(foundation_cfg.get("min_voice_duration_seconds_total", 8.0) or 8.0)
    target_total_duration = float(foundation_cfg.get("target_voice_duration_seconds_total", 18.0) or 18.0)
    min_voice_samples = max(1, int(foundation_cfg.get("min_voice_samples_for_clone", 4) or 4))
    coverage_score = min(1.0, total_duration / max(1.0, target_total_duration))
    sample_score = min(1.0, len(durations) / max(1.0, float(min_voice_samples + 2)))
    max_duration_score = min(1.0, max(durations) / 4.0)
    quality_score = round((coverage_score * 0.55) + (sample_score * 0.30) + (max_duration_score * 0.15), 4)
    clone_ready = len(durations) >= min_voice_samples and total_duration >= min_total_duration
    return {
        "sample_count": len(durations),
        "duration_seconds_total": round(total_duration, 3),
        "duration_seconds_mean": round(total_duration / max(1, len(durations)), 3),
        "duration_seconds_max": round(max(durations), 3),
        "quality_score": quality_score,
        "clone_ready": clone_ready,
        "language_counts": merge_language_counts(language_counts),
        "dominant_language": dominant_language(language_counts),
        "original_voice_sample_count": original_voice_sample_count,
        "sample_paths": sample_paths,
        "reference_segments": reference_segments[: max(1, int(foundation_cfg.get("min_voice_samples_for_clone", 4) or 4) * 2)],
    }


def video_stats(entries: list[dict]) -> dict:
    durations = [
        max(0.0, float(entry.get("end_seconds", 0.0) or 0.0) - float(entry.get("start_seconds", 0.0) or 0.0))
        for entry in entries
    ]
    durations = [duration for duration in durations if duration > 0.0]
    if not durations:
        return {"sample_count": 0}
    total_duration = sum(durations)
    return {
        "sample_count": len(durations),
        "duration_seconds_total": round(total_duration, 3),
        "duration_seconds_mean": round(total_duration / max(1, len(durations)), 3),
        "duration_seconds_max": round(max(durations), 3),
    }


def build_training_pack(manifest: dict, cfg: dict) -> dict:
    frame_paths = [Path(str(entry.get("path", ""))) for entry in manifest.get("frame_samples", [])]
    voice_entries = list(manifest.get("voice_samples", []) or [])
    video_entries = manifest.get("video_samples", []) or []
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    use_local_voice_models = bool(foundation_cfg.get("use_local_character_voice_models", True))
    voice_pack = voice_stats(voice_entries, cfg)
    return {
        "process_version": PROCESS_VERSION,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "character": manifest.get("name", ""),
        "slug": manifest.get("slug", ""),
        "priority": bool(manifest.get("priority", False)),
        "scene_count": int(manifest.get("scene_count", 0)),
        "line_count": int(manifest.get("line_count", 0)),
        "base_models": {
            "image": coalesce_text(foundation_cfg.get("image_base_model", "")),
            "video": coalesce_text(foundation_cfg.get("video_base_model", "")),
            "voice": "local_character_voice_model"
            if use_local_voice_models
            else coalesce_text(foundation_cfg.get("voice_base_model", "")),
        },
        "image_pack": image_stats(frame_paths),
        "video_pack": video_stats(video_entries),
        "voice_pack": voice_pack,
        "training_ready": bool(frame_paths or voice_entries or video_entries),
        "target_inference_minutes": [float(cfg.get("generation", {}).get("target_episode_minutes_min", 20.0)), float(cfg.get("generation", {}).get("target_episode_minutes_max", 24.0))],
    }


def voice_model_path_for_manifest(cfg: dict, manifest: dict) -> Path:
    voice_models_root = resolve_project_path(cfg["paths"]["voice_models"])
    slug = str(manifest.get("slug", "") or "figur")
    return voice_models_root / f"{slug}_voice_model.json"


def build_character_voice_model(manifest: dict, pack_payload: dict) -> dict:
    voice_pack = pack_payload.get("voice_pack", {}) if isinstance(pack_payload.get("voice_pack", {}), dict) else {}
    sample_paths = list(voice_pack.get("sample_paths", []) or [])
    reference_audio = sample_paths[0] if sample_paths else ""
    return {
        "process_version": PROCESS_VERSION,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "character": manifest.get("name", ""),
        "slug": manifest.get("slug", ""),
        "priority": bool(manifest.get("priority", False)),
        "model_kind": "local_character_voice_profile",
        "training_source": "original_character_voices",
        "dominant_language": coalesce_text(voice_pack.get("dominant_language", "")),
        "language_counts": dict(voice_pack.get("language_counts", {}) or {}),
        "sample_count": int(voice_pack.get("sample_count", 0) or 0),
        "original_voice_sample_count": int(voice_pack.get("original_voice_sample_count", 0) or 0),
        "duration_seconds_total": float(voice_pack.get("duration_seconds_total", 0.0) or 0.0),
        "quality_score": float(voice_pack.get("quality_score", 0.0) or 0.0),
        "clone_ready": bool(voice_pack.get("clone_ready", False)),
        "reference_audio": reference_audio,
        "sample_paths": sample_paths,
        "reference_segments": list(voice_pack.get("reference_segments", []) or []),
    }


def pack_path_for_manifest(cfg: dict, manifest: dict) -> Path:
    checkpoint_root = resolve_project_path(cfg["paths"]["foundation_checkpoints"])
    slug = str(manifest.get("slug", "") or "figur")
    return checkpoint_root / slug / "foundation_pack.json"


def foundation_pack_completed(pack_path: Path) -> bool:
    if not pack_path.exists():
        return False
    payload = read_json(pack_path, {})
    if not payload:
        return False
    if int(payload.get("process_version", 0) or 0) != PROCESS_VERSION:
        return False
    return bool(payload.get("training_ready", False))


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Train Foundation Models")
    cfg = load_config()
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)
    manifests = load_manifests(cfg, coalesce_text(args.character or ""), int(args.limit_characters or 0))
    if not manifests:
        info("No foundation manifests found. Run 08_prepare_foundation_training.py first.")
        return

    checkpoint_root = resolve_project_path(cfg["paths"]["foundation_checkpoints"])
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    lease_root = distributed_step_runtime_root("09_train_foundation_models", "characters")
    reporter = LiveProgressReporter(
        script_name="09_train_foundation_models.py",
        total=len(manifests),
        phase_label="Train Foundation Packs",
    )
    if shared_workers:
        info(f"Shared NAS workers: enabled ({worker_id})")
    for index, manifest in enumerate(manifests, start=1):
        character_name = coalesce_text(manifest.get("name", ""))
        slug = str(manifest.get("slug", "") or "figur")
        autosave_target = slug
        pack_path = pack_path_for_manifest(cfg, manifest)
        with distributed_item_lease(
            root=lease_root,
            lease_name=slug,
            cfg=cfg,
            worker_id=worker_id,
            enabled=shared_workers,
            meta={"step": "09_train_foundation_models", "character": character_name, "slug": slug, "worker_id": worker_id},
        ) as acquired:
            if not acquired:
                continue
            if not args.force and foundation_pack_completed(pack_path):
                payload = read_json(pack_path, {})
                info(f"Foundation pack already exists: {character_name}")
            else:
                info(f"Training local foundation pack: {character_name}")
                mark_step_started(
                    "09_train_foundation_models",
                    autosave_target,
                    {"character": character_name, "pack_path": str(pack_path)},
                )
                try:
                    payload = build_training_pack(manifest, cfg)
                    pack_dir = checkpoint_root / str(payload.get("slug", "figur"))
                    pack_dir.mkdir(parents=True, exist_ok=True)
                    voice_model_path = voice_model_path_for_manifest(cfg, manifest)
                    voice_model_path.parent.mkdir(parents=True, exist_ok=True)
                    voice_model_payload = build_character_voice_model(manifest, payload)
                    write_json(voice_model_path, voice_model_payload)
                    payload["voice_model_path"] = str(voice_model_path)
                    payload["voice_languages"] = dict((payload.get("voice_pack", {}) or {}).get("language_counts", {}) or {})
                    payload["dominant_voice_language"] = coalesce_text((payload.get("voice_pack", {}) or {}).get("dominant_language", ""))
                    write_json(pack_path, payload)
                    mark_step_completed(
                        "09_train_foundation_models",
                        autosave_target,
                        {
                            "character": character_name,
                            "pack_path": str(pack_path),
                            "frame_samples": int(payload.get("image_pack", {}).get("sample_count", 0)),
                            "video_samples": int(payload.get("video_pack", {}).get("sample_count", 0)),
                            "voice_samples": int(payload.get("voice_pack", {}).get("sample_count", 0)),
                            "voice_duration_seconds": float(payload.get("voice_pack", {}).get("duration_seconds_total", 0.0) or 0.0),
                            "voice_quality_score": float(payload.get("voice_pack", {}).get("quality_score", 0.0) or 0.0),
                            "voice_clone_ready": bool(payload.get("voice_pack", {}).get("clone_ready", False)),
                            "voice_model_path": str(voice_model_path),
                            "dominant_voice_language": coalesce_text((payload.get("voice_pack", {}) or {}).get("dominant_language", "")),
                        },
                    )
                except Exception as exc:
                    mark_step_failed(
                        "09_train_foundation_models",
                        str(exc),
                        autosave_target,
                        {"character": character_name, "pack_path": str(pack_path)},
                    )
                    raise

            summary.append(
                {
                    "character": payload.get("character", character_name),
                    "pack_path": str(pack_path),
                    "frame_samples": int(payload.get("image_pack", {}).get("sample_count", 0)),
                    "video_samples": int(payload.get("video_pack", {}).get("sample_count", 0)),
                    "voice_samples": int(payload.get("voice_pack", {}).get("sample_count", 0)),
                    "voice_duration_seconds": float(payload.get("voice_pack", {}).get("duration_seconds_total", 0.0) or 0.0),
                    "voice_quality_score": float(payload.get("voice_pack", {}).get("quality_score", 0.0) or 0.0),
                    "voice_clone_ready": bool(payload.get("voice_pack", {}).get("clone_ready", False)),
                    "voice_model_path": coalesce_text(payload.get("voice_model_path", "")),
                    "voice_languages": dict((payload.get("voice_pack", {}) or {}).get("language_counts", {}) or {}),
                    "dominant_voice_language": coalesce_text((payload.get("voice_pack", {}) or {}).get("dominant_language", "")),
                    "original_voice_sample_count": int((payload.get("voice_pack", {}) or {}).get("original_voice_sample_count", 0) or 0),
                    "autosave": load_step_autosave("09_train_foundation_models", autosave_target),
                }
            )
            reporter.update(
                index,
                current_label=character_name,
                extra_label=f"Trained packs so far: {len(summary)}",
            )
    reporter.finish(current_label="Foundation Training", extra_label=f"Total trained packs: {len(summary)}")

    summary_path = checkpoint_root / "foundation_training_summary.json"
    write_json(
        summary_path,
        {
            "process_version": PROCESS_VERSION,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "character_count": len(summary),
            "characters": summary,
        },
    )
    ok(f"Foundation-Packs trainiert: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise


