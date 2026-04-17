#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline_common import (
    coalesce_text,
    error,
    headline,
    info,
    load_config,
    load_step_autosave,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
    write_json,
)

PROCESS_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lokale Foundation-Packs fuer Bild, Video und Stimme trainieren")
    parser.add_argument("--limit-characters", type=int, default=0, help="Optional nur die ersten N Figuren trainieren.")
    parser.add_argument("--character", help="Optional nur eine bestimmte Figur trainieren.")
    parser.add_argument("--force", action="store_true", help="Trainiert vorhandene Foundation-Packs bewusst neu.")
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


def voice_stats(paths: list[Path], cfg: dict) -> dict:
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    durations = [wav_duration_seconds(path) for path in paths if path.exists()]
    if not durations:
        return {"sample_count": 0, "duration_seconds_total": 0.0, "quality_score": 0.0, "clone_ready": False}
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
    voice_paths = [Path(str(entry.get("path", ""))) for entry in manifest.get("voice_samples", [])]
    video_entries = manifest.get("video_samples", []) or []
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
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
            "voice": coalesce_text(foundation_cfg.get("voice_base_model", "")),
        },
        "image_pack": image_stats(frame_paths),
        "video_pack": video_stats(video_entries),
        "voice_pack": voice_stats(voice_paths, cfg),
        "training_ready": bool(frame_paths or voice_paths or video_entries),
        "target_inference_minutes": [float(cfg.get("generation", {}).get("target_episode_minutes_min", 20.0)), float(cfg.get("generation", {}).get("target_episode_minutes_max", 24.0))],
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
    headline("Foundation-Modelle trainieren")
    cfg = load_config()
    manifests = load_manifests(cfg, coalesce_text(args.character or ""), int(args.limit_characters or 0))
    if not manifests:
        info("Keine Foundation-Manifeste gefunden. Fuehre zuerst 09_prepare_foundation_training.py aus.")
        return

    checkpoint_root = resolve_project_path(cfg["paths"]["foundation_checkpoints"])
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    for manifest in manifests:
        character_name = coalesce_text(manifest.get("name", ""))
        slug = str(manifest.get("slug", "") or "figur")
        autosave_target = slug
        pack_path = pack_path_for_manifest(cfg, manifest)
        if not args.force and foundation_pack_completed(pack_path):
            payload = read_json(pack_path, {})
            info(f"Foundation-Pack bereits vorhanden: {character_name}")
        else:
            info(f"Trainiere lokalen Foundation-Pack: {character_name}")
            mark_step_started(
                "10_train_foundation_models",
                autosave_target,
                {"character": character_name, "pack_path": str(pack_path)},
            )
            try:
                payload = build_training_pack(manifest, cfg)
                pack_dir = checkpoint_root / str(payload.get("slug", "figur"))
                pack_dir.mkdir(parents=True, exist_ok=True)
                write_json(pack_path, payload)
                mark_step_completed(
                    "10_train_foundation_models",
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
                    },
                )
            except Exception as exc:
                mark_step_failed(
                    "10_train_foundation_models",
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
                "autosave": load_step_autosave("10_train_foundation_models", autosave_target),
            }
        )

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
