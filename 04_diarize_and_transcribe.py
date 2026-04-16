#!/usr/bin/env python3
from __future__ import annotations

import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline_common import (
    PROJECT_ROOT,
    coalesce_text,
    cosine_similarity,
    detect_tool,
    error,
    first_dir,
    headline,
    info,
    limited_items,
    load_config,
    ok,
    preferred_compute_label,
    preferred_execution_label,
    preferred_torch_device,
    progress,
    rerun_in_runtime,
    resolve_project_path,
    run_command,
    write_json,
)

PROCESS_VERSION = 4


def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as handle:
        frame_rate = handle.getframerate() or 16000
        frame_count = handle.getnframes() or 0
    return frame_count / frame_rate if frame_rate else 0.0


def export_audio(ffmpeg_path: Path, input_video: Path, output_wav: Path) -> None:
    if output_wav.exists() and output_wav.stat().st_size > 0:
        return
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            str(ffmpeg_path),
            "-hide_banner",
            "-y",
            "-i",
            str(input_video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ],
        quiet=True,
    )


def cut_audio_segment(ffmpeg_path: Path, input_wav: Path, output_wav: Path, start_sec: float, end_sec: float) -> None:
    if output_wav.exists() and output_wav.stat().st_size > 0:
        return
    duration = max(0.15, end_sec - start_sec)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            str(ffmpeg_path),
            "-hide_banner",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(input_wav),
            "-t",
            f"{duration:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ],
        quiet=True,
    )


def merge_segments(
    segments: list[dict[str, float | str]],
    merge_gap_seconds: float,
    min_segment_seconds: float,
    scene_duration: float,
) -> list[dict[str, float | str]]:
    merged: list[dict[str, float | str]] = []
    for segment in segments:
        if not merged:
            merged.append(dict(segment))
            continue
        previous = merged[-1]
        current_start = float(segment["start"])
        current_end = float(segment["end"])
        previous_end = float(previous["end"])
        previous_duration = previous_end - float(previous["start"])
        current_duration = current_end - current_start
        if (
            current_start - previous_end <= merge_gap_seconds
            or previous_duration < min_segment_seconds
            or current_duration < min_segment_seconds
        ):
            previous["end"] = max(previous_end, current_end)
            previous["text"] = coalesce_text(f"{previous.get('text', '')} {segment.get('text', '')}")
        else:
            merged.append(dict(segment))

    for segment in merged:
        segment["start"] = max(0.0, float(segment["start"]))
        segment["end"] = min(scene_duration, max(float(segment["end"]), float(segment["start"]) + 0.15))
        segment["text"] = coalesce_text(str(segment.get("text", "")))
    return merged


def transcribe_scene(model, scene_wav: Path, cfg: dict, use_fp16: bool) -> list[dict[str, float | str]]:
    full_duration = wav_duration_seconds(scene_wav)
    result = model.transcribe(
        str(scene_wav),
        language=cfg["transcription"]["language"],
        task=cfg["transcription"]["task"],
        condition_on_previous_text=False,
        temperature=0.0,
        verbose=False,
        fp16=use_fp16,
    )
    raw_segments = result.get("segments") or []
    segments = []
    for item in raw_segments:
        text = coalesce_text(item.get("text", ""))
        start_sec = max(0.0, float(item.get("start", 0.0)))
        end_sec = max(start_sec + 0.15, float(item.get("end", start_sec + 0.15)))
        if not text and end_sec - start_sec < 0.35:
            continue
        segments.append({"start": start_sec, "end": end_sec, "text": text})
    if not segments:
        text = coalesce_text(result.get("text", ""))
        if text or full_duration > 0.1:
            segments = [{"start": 0.0, "end": max(full_duration, 0.15), "text": text}]
    return merge_segments(
        segments,
        float(cfg["transcription"].get("merge_gap_seconds", 0.35)),
        float(cfg["transcription"].get("min_segment_seconds", 0.6)),
        full_duration,
    )


def compute_voice_embedding(segment_wav: Path) -> list[float]:
    import librosa

    audio, sample_rate = librosa.load(str(segment_wav), sr=16000, mono=True)
    if audio.size == 0:
        return []
    trimmed, _ = librosa.effects.trim(audio, top_db=30)
    if trimmed.size >= sample_rate // 5:
        audio = trimmed
    if audio.size < sample_rate // 5:
        return []

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    frame_count = int(mfcc.shape[1]) if mfcc.ndim == 2 else 0
    if frame_count < 3:
        return []
    delta_width = min(9, frame_count if frame_count % 2 == 1 else frame_count - 1)
    if delta_width < 3:
        return []
    delta = librosa.feature.delta(mfcc, width=delta_width, mode="nearest")
    delta2 = librosa.feature.delta(mfcc, order=2, width=delta_width, mode="nearest")
    rms = librosa.feature.rms(y=audio)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)

    features = np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            delta.mean(axis=1),
            delta.std(axis=1),
            delta2.mean(axis=1),
            delta2.std(axis=1),
            rms.mean(axis=1),
            rms.std(axis=1),
            zcr.mean(axis=1),
            zcr.std(axis=1),
            centroid.mean(axis=1),
            centroid.std(axis=1),
            np.array([audio.size / sample_rate], dtype=np.float32),
        ]
    ).astype(np.float32)
    norm = np.linalg.norm(features)
    if not np.isfinite(norm) or norm == 0:
        return []
    return (features / norm).round(6).tolist()


def process_scene(
    model,
    ffmpeg_path: Path,
    episode_name: str,
    scene_file: Path,
    audio_root: Path,
    scene_cache_dir: Path,
    cfg: dict,
    use_fp16: bool,
) -> list[dict]:
    cache_file = scene_cache_dir / f"{scene_file.stem}.json"
    if cache_file.exists():
        rows = resolve_rows(cache_file)
        if rows and rows[0].get("process_version") == PROCESS_VERSION:
            return rows

    scene_wav = audio_root / episode_name / f"{scene_file.stem}.wav"
    export_audio(ffmpeg_path, scene_file, scene_wav)
    segments = transcribe_scene(model, scene_wav, cfg, use_fp16)
    rows = []
    for index, segment in enumerate(segments, start=1):
        start_sec = float(segment["start"])
        end_sec = float(segment["end"])
        segment_wav = audio_root / episode_name / f"{scene_file.stem}_seg_{index:03d}.wav"
        cut_audio_segment(ffmpeg_path, scene_wav, segment_wav, start_sec, end_sec)
        rows.append(
            {
                "process_version": PROCESS_VERSION,
                "scene_id": scene_file.stem,
                "segment_id": f"{scene_file.stem}_seg_{index:03d}",
                "speaker_cluster": "",
                "start": round(start_sec, 3),
                "end": round(end_sec, 3),
                "audio_file": str(segment_wav),
                "text": coalesce_text(str(segment.get("text", ""))),
                "voice_embedding": compute_voice_embedding(segment_wav),
            }
        )
    write_json(cache_file, rows)
    return rows


def resolve_rows(path: Path) -> list[dict]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def assign_speaker_clusters(rows: list[dict], threshold: float) -> tuple[list[dict], list[dict]]:
    clusters: list[dict] = []
    for row in sorted(rows, key=lambda item: (item["scene_id"], float(item["start"]), item["segment_id"])):
        embedding = row.get("voice_embedding") or []
        if not embedding:
            row["speaker_cluster"] = "speaker_unknown"
            continue
        best_index = -1
        best_score = -1.0
        for index, cluster in enumerate(clusters):
            score = cosine_similarity(embedding, cluster["centroid"])
            if score > best_score:
                best_score = score
                best_index = index
        if best_index >= 0 and best_score >= threshold:
            cluster = clusters[best_index]
            count = cluster["count"]
            centroid = cluster["centroid"]
            cluster["centroid"] = [
                round(((centroid[i] * count) + embedding[i]) / (count + 1), 6)
                for i in range(len(embedding))
            ]
            cluster["count"] += 1
            row["speaker_cluster"] = cluster["cluster_id"]
        else:
            cluster_id = f"speaker_{len(clusters) + 1:03d}"
            clusters.append({"cluster_id": cluster_id, "centroid": list(embedding), "count": 1})
            row["speaker_cluster"] = cluster_id
    return rows, clusters


def main() -> None:
    rerun_in_runtime()
    headline("Sprecher segmentieren und transkribieren")
    cfg = load_config()
    ffmpeg = detect_tool(PROJECT_ROOT / "tools" / "ffmpeg" / "bin", "ffmpeg")
    scene_root = resolve_project_path(cfg["paths"]["scene_clips"])
    episode_dir = first_dir(scene_root)
    if episode_dir is None:
        info("Keine Szenenordner gefunden.")
        return

    scenes = limited_items(sorted(episode_dir.glob("*.mp4")))
    if not scenes:
        info("Keine Szene-Dateien gefunden.")
        return

    audio_root = resolve_project_path("data/raw/audio")
    scene_cache_dir = resolve_project_path(cfg["paths"]["speaker_segments"]) / episode_dir.name
    combined_dir = resolve_project_path(cfg["paths"]["speaker_transcripts"])
    scene_cache_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    import whisper

    model_dir = resolve_project_path(cfg["paths"]["whisper_model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    device = preferred_torch_device(cfg)
    use_fp16 = device == "cuda"
    model_name = cfg["transcription"]["model_name"]
    info(f"Whisper-Modell: {model_name}")
    info(f"Ausführungsmodus: {preferred_execution_label(cfg)}")
    info(f"Rechengerät: {preferred_compute_label(cfg)}")
    model = whisper.load_model(model_name, download_root=str(model_dir), device=device)

    all_rows: list[dict] = []
    for index, scene_file in enumerate(scenes, start=1):
        scene_rows = process_scene(
            model,
            ffmpeg,
            episode_dir.name,
            scene_file,
            audio_root,
            scene_cache_dir,
            cfg,
            use_fp16,
        )
        all_rows.extend(scene_rows)
        progress(index, len(scenes), "Audio wird transkribiert")

    threshold = float(cfg["transcription"].get("voice_embedding_threshold", 0.84))
    all_rows, clusters = assign_speaker_clusters(all_rows, threshold)
    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        grouped_rows[row["scene_id"]].append(row)
    for scene_id, scene_rows in grouped_rows.items():
        write_json(scene_cache_dir / f"{scene_id}.json", scene_rows)

    write_json(combined_dir / f"{episode_dir.name}_segments.json", all_rows)
    write_json(scene_cache_dir / "_speaker_clusters.json", {"clusters": clusters, "threshold": threshold})
    ok(f"Segment-Transkripte gespeichert: {len(all_rows)} Segmente")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
