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
    add_shared_worker_arguments,
    distributed_item_lease,
    distributed_step_runtime_root,
    LiveProgressReporter,
    adapter_summary_path,
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
    shared_worker_id_for_args,
    shared_workers_enabled_for_args,
    write_json,
)

PROCESS_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local adapter profiles for image, voice and clip dynamics")
    parser.add_argument("--limit-characters", type=int, default=0, help="Optionally train only the first N characters.")
    parser.add_argument("--character", help="Optionally train only one specific character.")
    parser.add_argument("--force", action="store_true", help="Intentionally retrain existing adapter profiles.")
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


def adapter_profile_path(cfg: dict, manifest: dict) -> Path:
    adapter_root = resolve_project_path(cfg["paths"]["foundation_adapters"])
    slug = str(manifest.get("slug", "") or "figur")
    return adapter_root / slug / "adapter_profile.json"


def adapter_profile_completed(profile_path: Path) -> bool:
    if not profile_path.exists():
        return False
    payload = read_json(profile_path, {})
    if not payload:
        return False
    if int(payload.get("process_version", 0) or 0) != PROCESS_VERSION:
        return False
    return bool(payload.get("training_ready", False))


def normalize_vector(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return []
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0:
        return []
    return (values.astype(np.float32) / norm).round(6).tolist()


def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] <= 0:
        return []
    return normalize_vector(matrix.mean(axis=0))


def image_feature_vector(path: Path, bins: int, thumbnail_size: int) -> list[float]:
    if not path.exists():
        return []
    with Image.open(path).convert("RGB") as image:
        image = image.resize((thumbnail_size, thumbnail_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
    means = array.mean(axis=(0, 1))
    stds = array.std(axis=(0, 1))
    channel_histograms: list[np.ndarray] = []
    for channel in range(3):
        hist, _edges = np.histogram(array[:, :, channel], bins=bins, range=(0.0, 1.0), density=True)
        channel_histograms.append(hist.astype(np.float32))
    feature = np.concatenate([means, stds, *channel_histograms]).astype(np.float32)
    return normalize_vector(feature)


def decode_wave_samples(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = int(handle.getframerate() or 24000)
        frame_count = int(handle.getnframes() or 0)
        sample_width = int(handle.getsampwidth() or 2)
        channel_count = int(handle.getnchannels() or 1)
        raw = handle.readframes(frame_count)
    if not raw:
        return np.asarray([], dtype=np.float32), sample_rate
    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return np.asarray([], dtype=np.float32), sample_rate
    if channel_count > 1:
        audio = audio.reshape(-1, channel_count).mean(axis=1)
    return np.asarray(audio, dtype=np.float32), sample_rate


def voice_feature_vector(path: Path, band_count: int) -> list[float]:
    if not path.exists():
        return []
    audio, sample_rate = decode_wave_samples(path)
    if audio.size <= 0 or sample_rate <= 0:
        return []
    audio = audio[np.isfinite(audio)]
    if audio.size <= 0:
        return []
    max_amplitude = float(np.max(np.abs(audio)))
    if max_amplitude > 0:
        audio = audio / max_amplitude
    zero_crossings = np.mean(np.abs(np.diff(np.sign(audio)))) if audio.size > 1 else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size > 0 else 0.0
    fft = np.abs(np.fft.rfft(audio))
    if fft.size <= 1:
        return []
    frequencies = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)
    magnitude_sum = float(np.sum(fft)) or 1.0
    centroid = float(np.sum(frequencies * fft) / magnitude_sum)
    bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * fft) / magnitude_sum))
    cumulative = np.cumsum(fft)
    rolloff_target = cumulative[-1] * 0.85 if cumulative.size else 0.0
    rolloff_index = int(np.searchsorted(cumulative, rolloff_target)) if cumulative.size else 0
    rolloff = float(frequencies[min(rolloff_index, max(0, len(frequencies) - 1))]) if frequencies.size else 0.0
    spectral_bins = np.array_split(fft.astype(np.float32), max(1, int(band_count)))
    band_energies = np.asarray([float(bin_values.mean()) if bin_values.size else 0.0 for bin_values in spectral_bins], dtype=np.float32)
    duration_seconds = float(audio.size) / float(sample_rate)
    feature = np.concatenate(
        [
            np.asarray([rms, zero_crossings, centroid / sample_rate, bandwidth / sample_rate, rolloff / sample_rate, duration_seconds], dtype=np.float32),
            band_energies,
        ]
    ).astype(np.float32)
    return normalize_vector(feature)


def video_feature_vector(entries: list[dict]) -> list[float]:
    durations = [
        max(0.0, float(entry.get("end_seconds", 0.0) or 0.0) - float(entry.get("start_seconds", 0.0) or 0.0))
        for entry in entries
    ]
    durations = [duration for duration in durations if duration > 0.0]
    if not durations:
        return []
    values = np.asarray(durations, dtype=np.float32)
    feature = np.asarray(
        [
            float(values.mean()),
            float(values.std()),
            float(values.max()),
            float(values.min()),
            float(np.median(values)),
            float(len({coalesce_text(entry.get("episode_id", "")) for entry in entries if coalesce_text(entry.get("episode_id", ""))})),
        ],
        dtype=np.float32,
    )
    return normalize_vector(feature)


def build_image_adapter(manifest: dict, cfg: dict) -> dict:
    adapter_cfg = cfg.get("adapter_training", {}) if isinstance(cfg.get("adapter_training"), dict) else {}
    bins = max(8, int(adapter_cfg.get("image_histogram_bins", 24) or 24))
    thumbnail_size = max(32, int(adapter_cfg.get("image_thumbnail_size", 96) or 96))
    min_samples = max(1, int(adapter_cfg.get("min_image_samples", 8) or 8))
    vectors = []
    for entry in manifest.get("frame_samples", []) or []:
        vector = image_feature_vector(Path(str(entry.get("path", ""))), bins, thumbnail_size)
        if vector:
            vectors.append(vector)
    prototype = mean_vector(vectors)
    return {
        "sample_count": len(vectors),
        "feature_dim": len(prototype),
        "prototype": prototype,
        "ready": len(vectors) >= min_samples and bool(prototype),
    }


def build_voice_adapter(manifest: dict, pack_payload: dict, cfg: dict) -> dict:
    adapter_cfg = cfg.get("adapter_training", {}) if isinstance(cfg.get("adapter_training"), dict) else {}
    band_count = max(8, int(adapter_cfg.get("voice_mfcc_count", 20) or 20))
    min_samples = max(1, int(adapter_cfg.get("min_voice_samples", 4) or 4))
    min_duration_seconds = float(adapter_cfg.get("min_voice_duration_seconds_total", 8.0) or 8.0)
    min_quality_score = float(adapter_cfg.get("min_voice_quality_score", 0.45) or 0.45)
    vectors = []
    for entry in manifest.get("voice_samples", []) or []:
        vector = voice_feature_vector(Path(str(entry.get("path", ""))), band_count)
        if vector:
            vectors.append(vector)
    prototype = mean_vector(vectors)
    source_voice_pack = pack_payload.get("voice_pack", {}) if isinstance(pack_payload.get("voice_pack"), dict) else {}
    duration_seconds_total = float(source_voice_pack.get("duration_seconds_total", 0.0) or 0.0)
    foundation_quality_score = float(source_voice_pack.get("quality_score", 0.0) or 0.0)
    clone_ready = (
        len(vectors) >= min_samples
        and bool(prototype)
        and duration_seconds_total >= min_duration_seconds
        and foundation_quality_score >= min_quality_score
        and bool(source_voice_pack.get("clone_ready", False))
    )
    return {
        "sample_count": len(vectors),
        "feature_dim": len(prototype),
        "prototype": prototype,
        "duration_seconds_total": round(duration_seconds_total, 3),
        "quality_score": round(foundation_quality_score, 4),
        "clone_ready": clone_ready,
        "ready": clone_ready,
        "voice_model_path": coalesce_text(pack_payload.get("voice_model_path", "")),
        "dominant_language": coalesce_text(pack_payload.get("dominant_voice_language", "") or source_voice_pack.get("dominant_language", "")),
        "language_counts": dict(source_voice_pack.get("language_counts", {}) or {}),
        "original_voice_sample_count": int(source_voice_pack.get("original_voice_sample_count", 0) or 0),
    }


def build_video_adapter(manifest: dict, cfg: dict) -> dict:
    adapter_cfg = cfg.get("adapter_training", {}) if isinstance(cfg.get("adapter_training"), dict) else {}
    entries = list(manifest.get("video_samples", []) or [])
    min_samples = max(1, int(adapter_cfg.get("min_video_samples", 4) or 4))
    prototype = video_feature_vector(entries)
    return {
        "sample_count": len(entries),
        "feature_dim": len(prototype),
        "prototype": prototype,
        "ready": len(entries) >= min_samples and bool(prototype),
    }


def build_adapter_profile(manifest: dict, pack_payload: dict, cfg: dict) -> dict:
    modalities = {
        "image": build_image_adapter(manifest, cfg),
        "voice": build_voice_adapter(manifest, pack_payload, cfg),
        "video": build_video_adapter(manifest, cfg),
    }
    modalities_ready = [name for name, payload in modalities.items() if bool(payload.get("ready", False))]
    return {
        "process_version": PROCESS_VERSION,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "character": manifest.get("name", ""),
        "slug": manifest.get("slug", ""),
        "priority": bool(manifest.get("priority", False)),
        "scene_count": int(manifest.get("scene_count", 0) or 0),
        "line_count": int(manifest.get("line_count", 0) or 0),
        "source_manifest": str(manifest.get("_manifest_path", "")),
        "source_foundation_pack": str(manifest.get("_foundation_pack_path", "")),
        "base_models": dict(pack_payload.get("base_models", {}) or {}),
        "modalities": modalities,
        "modalities_ready": modalities_ready,
        "voice_model_path": coalesce_text((modalities.get("voice", {}) or {}).get("voice_model_path", "")),
        "dominant_voice_language": coalesce_text((modalities.get("voice", {}) or {}).get("dominant_language", "")),
        "training_ready": bool(modalities_ready),
    }


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Train Local Adapter Profiles")
    cfg = load_config()
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)
    manifests = load_manifests(cfg, coalesce_text(args.character or ""), int(args.limit_characters or 0))
    if not manifests:
        info("No foundation manifests found. Run 09_prepare_foundation_training.py first.")
        return

    adapter_root = resolve_project_path(cfg["paths"]["foundation_adapters"])
    checkpoint_root = resolve_project_path(cfg["paths"]["foundation_checkpoints"])
    adapter_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    lease_root = distributed_step_runtime_root("11_train_adapter_models", "characters")
    reporter = LiveProgressReporter(
        script_name="11_train_adapter_models.py",
        total=len(manifests),
        phase_label="Train Adapter Profiles",
    )
    if shared_workers:
        info(f"Shared NAS workers: enabled ({worker_id})")
    for index, manifest in enumerate(manifests, start=1):
        character_name = coalesce_text(manifest.get("name", ""))
        slug = str(manifest.get("slug", "") or "figur")
        autosave_target = slug
        profile_path = adapter_profile_path(cfg, manifest)
        foundation_pack_path = checkpoint_root / slug / "foundation_pack.json"
        pack_payload = read_json(foundation_pack_path, {}) if foundation_pack_path.exists() else {}
        manifest["_manifest_path"] = str(resolve_project_path(cfg["paths"]["foundation_manifests"]) / f"{slug}_manifest.json")
        manifest["_foundation_pack_path"] = str(foundation_pack_path)
        with distributed_item_lease(
            root=lease_root,
            lease_name=slug,
            cfg=cfg,
            worker_id=worker_id,
            enabled=shared_workers,
            meta={"step": "11_train_adapter_models", "character": character_name, "slug": slug, "worker_id": worker_id},
        ) as acquired:
            if not acquired:
                continue
            if not args.force and adapter_profile_completed(profile_path):
                payload = read_json(profile_path, {})
                info(f"Adapter profile already exists: {character_name}")
            else:
                info(f"Training local adapter profile: {character_name}")
                mark_step_started(
                    "11_train_adapter_models",
                    autosave_target,
                    {"character": character_name, "profile_path": str(profile_path)},
                )
                try:
                    payload = build_adapter_profile(manifest, pack_payload, cfg)
                    profile_dir = profile_path.parent
                    profile_dir.mkdir(parents=True, exist_ok=True)
                    write_json(profile_path, payload)
                    mark_step_completed(
                        "11_train_adapter_models",
                        autosave_target,
                        {
                            "character": character_name,
                            "profile_path": str(profile_path),
                            "modalities_ready": payload.get("modalities_ready", []),
                            "training_ready": bool(payload.get("training_ready", False)),
                        },
                    )
                except Exception as exc:
                    mark_step_failed(
                        "11_train_adapter_models",
                        str(exc),
                        autosave_target,
                        {"character": character_name, "profile_path": str(profile_path)},
                    )
                    raise

            summary_rows.append(
                {
                    "character": payload.get("character", character_name),
                    "profile_path": str(profile_path),
                    "training_ready": bool(payload.get("training_ready", False)),
                    "modalities_ready": list(payload.get("modalities_ready", []) or []),
                    "image_samples": int((payload.get("modalities", {}).get("image", {}) or {}).get("sample_count", 0) or 0),
                    "voice_samples": int((payload.get("modalities", {}).get("voice", {}) or {}).get("sample_count", 0) or 0),
                    "voice_duration_seconds": float((payload.get("modalities", {}).get("voice", {}) or {}).get("duration_seconds_total", 0.0) or 0.0),
                    "voice_quality_score": float((payload.get("modalities", {}).get("voice", {}) or {}).get("quality_score", 0.0) or 0.0),
                    "voice_clone_ready": bool((payload.get("modalities", {}).get("voice", {}) or {}).get("clone_ready", False)),
                    "voice_model_path": coalesce_text(payload.get("voice_model_path", "")),
                    "dominant_voice_language": coalesce_text(payload.get("dominant_voice_language", "")),
                    "video_samples": int((payload.get("modalities", {}).get("video", {}) or {}).get("sample_count", 0) or 0),
                    "autosave": load_step_autosave("11_train_adapter_models", autosave_target),
                }
            )
            reporter.update(
                index,
                current_label=character_name,
                extra_label=f"Adapter profiles so far: {len(summary_rows)}",
            )
    reporter.finish(current_label="Adapter Training", extra_label=f"Total adapter profiles: {len(summary_rows)}")

    summary_path = adapter_summary_path(cfg)
    write_json(
        summary_path,
        {
            "process_version": PROCESS_VERSION,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "character_count": len(summary_rows),
            "characters": summary_rows,
        },
    )
    ok(f"Adapter-Profile trainiert: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise

