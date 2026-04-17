#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline_common import (
    LiveProgressReporter,
    PROJECT_ROOT,
    completed_step_state,
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
    save_step_autosave,
    mark_step_started,
    mark_step_completed,
    mark_step_failed,
    tokens_from_text,
    warn,
    write_json,
)

PROCESS_VERSION = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprecher segmentieren und transkribieren")
    parser.add_argument("--episode", help="Name des Szenenordners unter data/processed/scene_clips.")
    return parser.parse_args()


def episode_transcription_completed(episode_dir: Path, cfg: dict) -> bool:
    combined_file = resolve_project_path(cfg["paths"]["speaker_transcripts"]) / f"{episode_dir.name}_segments.json"
    cluster_file = resolve_project_path(cfg["paths"]["speaker_segments"]) / episode_dir.name / "_speaker_clusters.json"
    if not combined_file.exists() or not cluster_file.exists():
        return False
    try:
        rows = resolve_rows(combined_file)
        cluster_payload = resolve_rows(cluster_file)
    except Exception:
        return False
    if not rows or not isinstance(cluster_payload, dict):
        return False
    if int(cluster_payload.get("process_version", 0) or 0) != PROCESS_VERSION:
        return False
    if not rows[0].get("process_version") == PROCESS_VERSION:
        return False
    autosave_state = completed_step_state("04_diarize_and_transcribe", episode_dir.name, PROCESS_VERSION)
    if autosave_state:
        expected_scene_count = int(autosave_state.get("scene_count", 0) or 0)
        if expected_scene_count > 0:
            available_scene_count = len([path for path in episode_dir.glob("scene_*.mp4") if path.is_file()])
            if available_scene_count < expected_scene_count:
                return False
    return True


def next_untranscribed_episode_dir(scene_root: Path, cfg: dict) -> Path | None:
    for folder in sorted(scene_root.glob("*")):
        if not folder.is_dir():
            continue
        if not episode_transcription_completed(folder, cfg):
            return folder
    return first_dir(scene_root)


def pending_untranscribed_episode_dirs(scene_root: Path, cfg: dict) -> list[Path]:
    pending = []
    for folder in sorted(scene_root.glob("*")):
        if not folder.is_dir():
            continue
        if not episode_transcription_completed(folder, cfg):
            pending.append(folder)
    return pending


def resolve_episode_dir(scene_root: Path, episode_name: str | None, cfg: dict) -> Path | None:
    if episode_name:
        candidate = scene_root / Path(episode_name).name
        if candidate.is_dir():
            return candidate
        for folder in sorted(scene_root.glob("*")):
            if folder.is_dir() and folder.name == Path(episode_name).stem:
                return folder
        raise FileNotFoundError(f"Szenenordner nicht gefunden: {episode_name}")
    return next_untranscribed_episode_dir(scene_root, cfg)


def resolve_episode_dirs_for_processing(scene_root: Path, episode_name: str | None, cfg: dict) -> list[Path]:
    if episode_name:
        episode_dir = resolve_episode_dir(scene_root, episode_name, cfg)
        return [episode_dir] if episode_dir is not None else []
    return pending_untranscribed_episode_dirs(scene_root, cfg)


def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as handle:
        frame_rate = handle.getframerate() or 16000
        frame_count = handle.getnframes() or 0
    return frame_count / frame_rate if frame_rate else 0.0


def pool_vector(values: np.ndarray, bins: int) -> np.ndarray:
    chunks = np.array_split(values.astype(np.float32), bins)
    pooled = [float(chunk.mean()) if chunk.size else 0.0 for chunk in chunks]
    return np.asarray(pooled, dtype=np.float32)


def load_audio_excerpt(wav_path: Path, start_sec: float, end_sec: float, sample_rate: int = 16000) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    with sf.SoundFile(str(wav_path)) as handle:
        native_rate = int(handle.samplerate or sample_rate)
        frame_start = max(0, int(start_sec * native_rate))
        frame_stop = max(frame_start + 1, int(end_sec * native_rate))
        handle.seek(frame_start)
        audio = handle.read(frames=frame_stop - frame_start, dtype="float32", always_2d=False)

    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if native_rate != sample_rate and audio.size:
        audio = resample_poly(audio, sample_rate, native_rate).astype(np.float32)
    return audio


def trim_silence(audio: np.ndarray, top_db: float = 28.0) -> np.ndarray:
    if audio.size == 0:
        return audio
    max_amplitude = float(np.max(np.abs(audio)))
    if max_amplitude <= 1e-6:
        return audio
    threshold = max_amplitude * (10.0 ** (-top_db / 20.0))
    voiced = np.flatnonzero(np.abs(audio) >= threshold)
    if voiced.size == 0:
        return audio
    return audio[int(voiced[0]) : int(voiced[-1]) + 1]


def prepare_embedding_audio(
    scene_wav: Path,
    start_sec: float,
    end_sec: float,
    cfg: dict,
) -> tuple[np.ndarray, int]:
    sample_rate = 16000
    padding = float(cfg["transcription"].get("voice_embedding_context_padding_seconds", 0.45))
    min_seconds = float(cfg["transcription"].get("voice_embedding_min_seconds", 0.45))
    clip_start = max(0.0, start_sec - padding)
    clip_end = max(clip_start + min_seconds, end_sec + padding)
    audio = load_audio_excerpt(scene_wav, clip_start, clip_end, sample_rate=sample_rate)
    if audio.size == 0:
        return np.asarray([], dtype=np.float32), sample_rate

    trimmed = trim_silence(audio, top_db=28.0)
    if trimmed.size >= int(sample_rate * min_seconds):
        audio = trimmed.astype(np.float32)
    if audio.size < int(sample_rate * min_seconds):
        return np.asarray([], dtype=np.float32), sample_rate
    return audio.astype(np.float32), sample_rate


def compute_mfcc_voice_embedding(audio: np.ndarray, sample_rate: int) -> list[float]:
    import librosa

    if audio.size < int(sample_rate * 0.45):
        return []
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    if mfcc.ndim != 2 or int(mfcc.shape[1]) < 6:
        return []
    delta = librosa.feature.delta(mfcc, mode="nearest")
    delta2 = librosa.feature.delta(mfcc, order=2, mode="nearest")
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    rms = librosa.feature.rms(y=audio)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    flatness = librosa.feature.spectral_flatness(y=audio)
    pitches, voiced_flags, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
    )
    voiced_pitches = pitches[voiced_flags] if pitches is not None and voiced_flags is not None else np.asarray([])
    pitch_stats = np.asarray(
        [
            float(np.nanmean(voiced_pitches)) if voiced_pitches.size else 0.0,
            float(np.nanstd(voiced_pitches)) if voiced_pitches.size else 0.0,
            float(voiced_pitches.size / max(1, pitches.size if pitches is not None else 1)),
        ],
        dtype=np.float32,
    )
    features = np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            delta.mean(axis=1),
            delta.std(axis=1),
            delta2.mean(axis=1),
            delta2.std(axis=1),
            contrast.mean(axis=1),
            contrast.std(axis=1),
            centroid.mean(axis=1),
            centroid.std(axis=1),
            bandwidth.mean(axis=1),
            bandwidth.std(axis=1),
            rolloff.mean(axis=1),
            rolloff.std(axis=1),
            rms.mean(axis=1),
            rms.std(axis=1),
            zcr.mean(axis=1),
            zcr.std(axis=1),
            flatness.mean(axis=1),
            flatness.std(axis=1),
            pitch_stats,
            np.asarray([audio.size / sample_rate], dtype=np.float32),
        ]
    ).astype(np.float32)
    norm = np.linalg.norm(features)
    if not np.isfinite(norm) or norm == 0:
        return []
    return (features / norm).round(6).tolist()


def speechbrain_encoder(model_dir: Path, device: str):
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    source = "speechbrain/spkrec-ecapa-voxceleb"
    run_device = "cuda:0" if device == "cuda" else "cpu"
    return EncoderClassifier.from_hparams(
        source=source,
        savedir=str(model_dir),
        run_opts={"device": run_device},
        local_strategy=LocalStrategy.COPY,
    )


def compute_speechbrain_voice_embedding(audio: np.ndarray, encoder, device: str) -> list[float]:
    import torch

    if audio.size == 0:
        return []
    signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    if device == "cuda":
        signal = signal.cuda()
    with torch.no_grad():
        embedding = encoder.encode_batch(signal).squeeze().detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(embedding)
    if not np.isfinite(norm) or norm == 0:
        return []
    return (embedding / norm).round(6).tolist()


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


def compute_voice_embedding(
    scene_wav: Path,
    start_sec: float,
    end_sec: float,
    cfg: dict,
    backend: str,
    speechbrain_model=None,
    device: str = "cpu",
) -> list[float]:
    audio, sample_rate = prepare_embedding_audio(scene_wav, start_sec, end_sec, cfg)
    if audio.size == 0:
        return []
    if backend == "speechbrain" and speechbrain_model is not None:
        embedding = compute_speechbrain_voice_embedding(audio, speechbrain_model, device=device)
        if embedding:
            return embedding
    return compute_mfcc_voice_embedding(audio, sample_rate)


def process_scene(
    model,
    ffmpeg_path: Path,
    episode_name: str,
    scene_file: Path,
    audio_root: Path,
    scene_cache_dir: Path,
    cfg: dict,
    use_fp16: bool,
    speaker_embedding_backend: str,
    speechbrain_model=None,
    device: str = "cpu",
) -> list[dict]:
    cache_file = scene_cache_dir / f"{scene_file.stem}.json"
    if cache_file.exists():
        rows = resolve_rows(cache_file)
        if rows and rows[0].get("process_version") == PROCESS_VERSION:
            return rows

    scene_wav = audio_root / episode_name / f"{scene_file.stem}.wav"
    export_audio(ffmpeg_path, scene_file, scene_wav)
    cached_rows = resolve_rows(cache_file) if cache_file.exists() else []
    if cached_rows:
        rows = []
        for index, cached in enumerate(cached_rows, start=1):
            start_sec = float(cached.get("start", 0.0))
            end_sec = float(cached.get("end", start_sec + 0.15))
            segment_wav = Path(cached.get("audio_file") or (audio_root / episode_name / f"{scene_file.stem}_seg_{index:03d}.wav"))
            if not segment_wav.exists():
                cut_audio_segment(ffmpeg_path, scene_wav, segment_wav, start_sec, end_sec)
            rows.append(
                {
                    "process_version": PROCESS_VERSION,
                    "scene_id": scene_file.stem,
                    "segment_id": str(cached.get("segment_id") or f"{scene_file.stem}_seg_{index:03d}"),
                    "speaker_cluster": "",
                    "start": round(start_sec, 3),
                    "end": round(end_sec, 3),
                    "audio_file": str(segment_wav),
                    "text": coalesce_text(str(cached.get("text", ""))),
                    "voice_embedding": compute_voice_embedding(
                        scene_wav,
                        start_sec,
                        end_sec,
                        cfg,
                        speaker_embedding_backend,
                        speechbrain_model=speechbrain_model,
                        device=device,
                    ),
                }
            )
        write_json(cache_file, rows)
        return rows

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
                "voice_embedding": compute_voice_embedding(
                    scene_wav,
                    start_sec,
                    end_sec,
                    cfg,
                    speaker_embedding_backend,
                    speechbrain_model=speechbrain_model,
                    device=device,
                ),
            }
        )
    write_json(cache_file, rows)
    return rows


def resolve_rows(path: Path) -> list[dict]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def segment_quality(row: dict, cfg: dict) -> str:
    duration = float(row["end"]) - float(row["start"])
    words = len(tokens_from_text(str(row.get("text", ""))))
    high_duration = float(cfg["transcription"].get("speaker_cluster_high_quality_min_seconds", 1.0))
    if duration >= high_duration and words >= 2:
        return "high"
    if duration >= 0.55 and words >= 1:
        return "medium"
    return "low"


def normalize_embedding(values: list[float]) -> list[float]:
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0:
        return []
    return (vector / norm).round(6).tolist()


def best_cluster_match(embedding: list[float], clusters: list[dict]) -> tuple[int, float]:
    best_index = -1
    best_score = -1.0
    for index, cluster in enumerate(clusters):
        score = cosine_similarity(embedding, cluster["centroid"])
        if score > best_score:
            best_score = score
            best_index = index
    return best_index, best_score


def update_cluster(cluster: dict, embedding: list[float], row: dict) -> None:
    count = cluster["count"]
    centroid = np.asarray(cluster["centroid"], dtype=np.float32)
    current = np.asarray(embedding, dtype=np.float32)
    blended = ((centroid * count) + current) / (count + 1)
    cluster["centroid"] = normalize_embedding(blended.tolist())
    cluster["count"] += 1
    cluster["scene_ids"].add(row["scene_id"])
    cluster["segments"].append(row["segment_id"])


def create_cluster(clusters: list[dict], embedding: list[float], row: dict) -> dict:
    cluster = {
        "cluster_id": f"speaker_{len(clusters) + 1:03d}",
        "centroid": normalize_embedding(list(embedding)),
        "count": 1,
        "scene_ids": {row["scene_id"]},
        "segments": [row["segment_id"]],
    }
    clusters.append(cluster)
    return cluster


def rescue_unknown_speaker_rows(
    ordered_rows: list[dict],
    cluster_by_id: dict[str, dict],
    surviving_ids: set[str],
    threshold: float,
    cfg: dict,
) -> int:
    rescue_margin = float(cfg["transcription"].get("speaker_unknown_rescue_margin", 0.08))
    neighbor_margin = float(cfg["transcription"].get("speaker_unknown_neighbor_margin", 0.12))
    rescued = 0
    rows_by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in ordered_rows:
        rows_by_scene[str(row.get("scene_id", ""))].append(row)

    for scene_rows in rows_by_scene.values():
        for index, row in enumerate(scene_rows):
            if row.get("speaker_cluster") != "speaker_unknown":
                continue
            embedding = row.get("voice_embedding") or []
            if not embedding:
                continue

            neighbor_candidates: list[str] = []
            previous_cluster = scene_rows[index - 1].get("speaker_cluster", "") if index > 0 else ""
            next_cluster = scene_rows[index + 1].get("speaker_cluster", "") if index + 1 < len(scene_rows) else ""
            if previous_cluster in surviving_ids:
                neighbor_candidates.append(previous_cluster)
            if next_cluster in surviving_ids:
                neighbor_candidates.append(next_cluster)
            if len(neighbor_candidates) == 2 and neighbor_candidates[0] != neighbor_candidates[1]:
                neighbor_candidates = []

            assigned_cluster = ""
            if neighbor_candidates:
                candidate_id = neighbor_candidates[0]
                neighbor_score = cosine_similarity(embedding, cluster_by_id[candidate_id]["centroid"])
                if neighbor_score >= max(0.0, threshold - neighbor_margin):
                    assigned_cluster = candidate_id
            if not assigned_cluster:
                best_index, best_score = best_cluster_match(
                    embedding,
                    [cluster_by_id[cluster_id] for cluster_id in sorted(surviving_ids)],
                )
                candidate_ids = sorted(surviving_ids)
                if best_index >= 0 and best_score >= max(0.0, threshold - rescue_margin):
                    assigned_cluster = candidate_ids[best_index]

            if assigned_cluster:
                row["speaker_cluster"] = assigned_cluster
                update_cluster(cluster_by_id[assigned_cluster], embedding, row)
                rescued += 1
    return rescued


def cluster_text_profiles(ordered_rows: list[dict], surviving_ids: set[str]) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    for cluster_id in sorted(surviving_ids):
        profiles[cluster_id] = {
            "token_weights": defaultdict(float),
            "token_rows": 0,
            "scene_ids": set(),
        }

    for row in ordered_rows:
        cluster_id = str(row.get("speaker_cluster", ""))
        if cluster_id not in profiles:
            continue
        tokens = [
            token
            for token in tokens_from_text(str(row.get("text", "")))
            if len(token) >= 4 and token.isalpha()
        ]
        if not tokens:
            continue
        profile = profiles[cluster_id]
        quality = segment_quality(row, {"transcription": {}})
        quality_weight = 1.6 if quality == "high" else 1.2 if quality == "medium" else 1.0
        for token in set(tokens):
            profile["token_weights"][token] += quality_weight
        profile["token_rows"] += 1
        profile["scene_ids"].add(str(row.get("scene_id", "")))

    return profiles


def best_profile_token_match(tokens: list[str], text_profiles: dict[str, dict]) -> tuple[str, float, float]:
    candidate_scores: list[tuple[str, float]] = []
    unique_tokens = [token for token in dict.fromkeys(tokens) if len(token) >= 4 and token.isalpha()]
    if not unique_tokens:
        return "", 0.0, 0.0

    for cluster_id, profile in text_profiles.items():
        token_weights = profile.get("token_weights", {})
        if not token_weights:
            continue
        score = sum(float(token_weights.get(token, 0.0)) for token in unique_tokens)
        if score <= 0:
            continue
        candidate_scores.append((cluster_id, score))

    if not candidate_scores:
        return "", 0.0, 0.0
    candidate_scores.sort(key=lambda item: item[1], reverse=True)
    best_cluster, best_score = candidate_scores[0]
    second_score = candidate_scores[1][1] if len(candidate_scores) > 1 else 0.0
    return best_cluster, float(best_score), float(best_score - second_score)


def rescue_unknown_speaker_rows_with_episode_consensus(
    ordered_rows: list[dict],
    cluster_by_id: dict[str, dict],
    surviving_ids: set[str],
    threshold: float,
    cfg: dict,
) -> int:
    rescue_margin = float(cfg["transcription"].get("speaker_unknown_episode_rescue_margin", 0.11))
    embedding_margin = float(cfg["transcription"].get("speaker_unknown_episode_embedding_margin", 0.04))
    min_token_score = float(cfg["transcription"].get("speaker_unknown_episode_min_token_score", 2.2))
    min_token_margin = float(cfg["transcription"].get("speaker_unknown_episode_min_token_margin", 0.8))
    rescued = 0
    text_profiles = cluster_text_profiles(ordered_rows, surviving_ids)
    candidate_clusters = [cluster_by_id[cluster_id] for cluster_id in sorted(surviving_ids)]
    candidate_ids = [cluster["cluster_id"] for cluster in candidate_clusters]

    for row in ordered_rows:
        if row.get("speaker_cluster") != "speaker_unknown":
            continue
        embedding = row.get("voice_embedding") or []
        if not embedding:
            continue
        row_tokens = [
            token
            for token in tokens_from_text(str(row.get("text", "")))
            if len(token) >= 4 and token.isalpha()
        ]
        if not row_tokens:
            continue

        best_index, best_score = best_cluster_match(embedding, candidate_clusters)
        if best_index < 0:
            continue
        ranked_scores = sorted(
            (
                cosine_similarity(embedding, cluster["centroid"]),
                cluster["cluster_id"],
            )
            for cluster in candidate_clusters
        )
        ranked_scores.sort(reverse=True)
        second_embedding_score = ranked_scores[1][0] if len(ranked_scores) > 1 else -1.0
        best_embedding_cluster = candidate_ids[best_index]
        if best_score < max(0.0, threshold - rescue_margin):
            continue
        if best_score - second_embedding_score < embedding_margin:
            continue

        best_text_cluster, best_text_score, text_margin = best_profile_token_match(row_tokens, text_profiles)
        if not best_text_cluster:
            continue
        if best_text_cluster != best_embedding_cluster:
            continue
        if best_text_score < min_token_score or text_margin < min_token_margin:
            continue

        row["speaker_cluster"] = best_embedding_cluster
        update_cluster(cluster_by_id[best_embedding_cluster], embedding, row)
        rescued += 1

    return rescued


def assign_speaker_clusters(rows: list[dict], threshold: float, cfg: dict) -> tuple[list[dict], list[dict]]:
    clusters: list[dict] = []
    ordered_rows = sorted(rows, key=lambda item: (item["scene_id"], float(item["start"]), item["segment_id"]))
    grouped_rows = {
        "high": [row for row in ordered_rows if (row.get("voice_embedding") and segment_quality(row, cfg) == "high")],
        "medium": [row for row in ordered_rows if (row.get("voice_embedding") and segment_quality(row, cfg) == "medium")],
        "low": [row for row in ordered_rows if (row.get("voice_embedding") and segment_quality(row, cfg) == "low")],
    }

    def assign_row(row: dict, allow_new_cluster: bool, threshold_value: float) -> None:
        embedding = row.get("voice_embedding") or []
        if not embedding:
            row["speaker_cluster"] = "speaker_unknown"
            return
        best_index, best_score = best_cluster_match(embedding, clusters)
        if best_index >= 0 and best_score >= threshold_value:
            cluster = clusters[best_index]
            update_cluster(cluster, embedding, row)
            row["speaker_cluster"] = cluster["cluster_id"]
            return
        if allow_new_cluster:
            cluster = create_cluster(clusters, embedding, row)
            row["speaker_cluster"] = cluster["cluster_id"]
            return
        row["speaker_cluster"] = "speaker_unknown"

    for row in grouped_rows["high"]:
        assign_row(row, allow_new_cluster=True, threshold_value=threshold)
    for row in grouped_rows["medium"]:
        assign_row(row, allow_new_cluster=True, threshold_value=threshold + 0.03)
    for row in grouped_rows["low"]:
        assign_row(row, allow_new_cluster=False, threshold_value=threshold + 0.07)

    minimum_cluster_size = int(cfg["transcription"].get("speaker_cluster_min_segments", 2))
    cluster_by_id = {cluster["cluster_id"]: cluster for cluster in clusters}
    for row in ordered_rows:
        cluster_id = row.get("speaker_cluster", "")
        cluster = cluster_by_id.get(cluster_id)
        if cluster is None:
            continue
        if cluster["count"] < minimum_cluster_size:
            row["speaker_cluster"] = "speaker_unknown"

    surviving_ids = {
        row["speaker_cluster"]
        for row in ordered_rows
        if row.get("speaker_cluster") and row["speaker_cluster"] != "speaker_unknown"
    }
    if surviving_ids:
        rescue_unknown_speaker_rows(ordered_rows, cluster_by_id, surviving_ids, threshold, cfg)
        surviving_ids = {
            row["speaker_cluster"]
            for row in ordered_rows
            if row.get("speaker_cluster") and row["speaker_cluster"] != "speaker_unknown"
        }
        rescue_unknown_speaker_rows_with_episode_consensus(
            ordered_rows,
            cluster_by_id,
            surviving_ids,
            threshold,
            cfg,
        )

    surviving_ids = sorted(
        {
            row["speaker_cluster"]
            for row in ordered_rows
            if row.get("speaker_cluster") and row["speaker_cluster"] != "speaker_unknown"
        },
        key=lambda cluster_id: (-cluster_by_id[cluster_id]["count"], cluster_id),
    )
    cluster_id_map = {cluster_id: f"speaker_{index:03d}" for index, cluster_id in enumerate(surviving_ids, start=1)}
    for row in ordered_rows:
        cluster_id = row.get("speaker_cluster", "")
        if cluster_id in cluster_id_map:
            row["speaker_cluster"] = cluster_id_map[cluster_id]

    cluster_payload = []
    for original_id in surviving_ids:
        cluster = cluster_by_id[original_id]
        cluster_payload.append(
            {
                "cluster_id": cluster_id_map[original_id],
                "centroid": cluster["centroid"],
                "count": cluster["count"],
                "scene_count": len(cluster["scene_ids"]),
                "segments": cluster["segments"][:20],
            }
        )
    return ordered_rows, cluster_payload


def process_episode_dir(
    episode_dir: Path,
    cfg: dict,
    ffmpeg: Path,
    model,
    use_fp16: bool,
    speaker_backend: str,
    speechbrain_model,
    device: str,
    live_reporter: LiveProgressReporter | None = None,
    episode_index: int = 1,
    episode_total: int = 1,
) -> bool:
    autosave_target = episode_dir.name
    if episode_transcription_completed(episode_dir, cfg):
        mark_step_completed(
            "04_diarize_and_transcribe",
            autosave_target,
            {
                "episode_id": episode_dir.name,
                "process_version": PROCESS_VERSION,
                "combined_file": str(resolve_project_path(cfg["paths"]["speaker_transcripts"]) / f"{episode_dir.name}_segments.json"),
                "cluster_file": str(resolve_project_path(cfg["paths"]["speaker_segments"]) / episode_dir.name / "_speaker_clusters.json"),
            },
        )
        ok(f"Transkription bereits erfolgreich vorhanden: {episode_dir.name}")
        return False

    scenes = limited_items(sorted(episode_dir.glob("*.mp4")))
    if not scenes:
        info("Keine Szene-Dateien gefunden.")
        return False

    audio_root = resolve_project_path("data/raw/audio")
    scene_cache_dir = resolve_project_path(cfg["paths"]["speaker_segments"]) / episode_dir.name
    combined_dir = resolve_project_path(cfg["paths"]["speaker_transcripts"])
    scene_cache_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)
    mark_step_started(
        "04_diarize_and_transcribe",
        autosave_target,
        {
            "episode_id": episode_dir.name,
            "process_version": PROCESS_VERSION,
            "scene_count": len(scenes),
            "device": device,
            "model_name": cfg["transcription"]["model_name"],
        },
    )
    completed_scene_ids: list[str] = []
    try:
        all_rows: list[dict] = []
        episode_started_at = time.time()
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
                speaker_backend,
                speechbrain_model=speechbrain_model,
                device=device,
            )
            all_rows.extend(scene_rows)
            completed_scene_ids.append(scene_file.stem)
            save_step_autosave(
                "04_diarize_and_transcribe",
                autosave_target,
                {
                    "status": "in_progress",
                    "episode_id": episode_dir.name,
                    "process_version": PROCESS_VERSION,
                    "scene_count": len(scenes),
                    "completed_scene_ids": completed_scene_ids,
                    "segment_count": len(all_rows),
                    "last_scene_id": scene_file.stem,
                },
            )
            if live_reporter is not None:
                live_reporter.update(
                    (episode_index - 1) + (index / max(1, len(scenes))),
                    current_label=scene_file.name,
                    parent_label=episode_dir.name,
                    extra_label=f"Segmente bisher: {len(all_rows)}",
                    scope_current=index,
                    scope_total=len(scenes),
                    scope_started_at=episode_started_at,
                    scope_label=f"Folge {episode_index}/{episode_total}",
                )

        threshold_key = "voice_embedding_threshold_speechbrain" if speaker_backend == "speechbrain" else "voice_embedding_threshold"
        threshold_default = 0.58 if speaker_backend == "speechbrain" else 0.84
        threshold = float(cfg["transcription"].get(threshold_key, threshold_default))
        all_rows, clusters = assign_speaker_clusters(all_rows, threshold, cfg)
        grouped_rows: dict[str, list[dict]] = defaultdict(list)
        for row in all_rows:
            grouped_rows[row["scene_id"]].append(row)
        for scene_id, scene_rows in grouped_rows.items():
            write_json(scene_cache_dir / f"{scene_id}.json", scene_rows)

        combined_file = combined_dir / f"{episode_dir.name}_segments.json"
        cluster_file = scene_cache_dir / "_speaker_clusters.json"
        write_json(combined_file, all_rows)
        write_json(
            cluster_file,
            {
                "clusters": clusters,
                "threshold": threshold,
                "backend": speaker_backend,
                "process_version": PROCESS_VERSION,
            },
        )
        mark_step_completed(
            "04_diarize_and_transcribe",
            autosave_target,
            {
                "episode_id": episode_dir.name,
                "process_version": PROCESS_VERSION,
                "scene_count": len(scenes),
                "segment_count": len(all_rows),
                "completed_scene_ids": completed_scene_ids,
                "combined_file": str(combined_file),
                "cluster_file": str(cluster_file),
                "backend": speaker_backend,
            },
        )
        ok(f"Segment-Transkripte gespeichert: {len(all_rows)} Segmente")
        return True
    except Exception as exc:
        mark_step_failed(
            "04_diarize_and_transcribe",
            str(exc),
            autosave_target,
            {
                "episode_id": episode_dir.name,
                "process_version": PROCESS_VERSION,
                "scene_count": len(scenes),
                "completed_scene_ids": completed_scene_ids,
            },
        )
        raise


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Sprecher segmentieren und transkribieren")
    cfg = load_config()
    ffmpeg = detect_tool(PROJECT_ROOT / "tools" / "ffmpeg" / "bin", "ffmpeg")
    scene_root = resolve_project_path(cfg["paths"]["scene_clips"])
    episode_dirs = resolve_episode_dirs_for_processing(scene_root, args.episode, cfg)
    if not episode_dirs:
        if args.episode:
            info("Keine passenden Szenenordner gefunden.")
        else:
            info("Keine offenen Folgen für Schritt 04 gefunden.")
        return

    model_name = cfg["transcription"]["model_name"]
    device = preferred_torch_device(cfg)
    use_fp16 = device == "cuda"
    requested_backend = str(cfg["transcription"].get("speaker_embedding_backend", "auto")).strip().lower()
    speaker_backend = "mfcc"
    speechbrain_model = None
    model = None
    info(f"Whisper-Modell: {model_name}")
    info(f"Ausführungsmodus: {preferred_execution_label(cfg)}")
    info(f"Rechengerät: {preferred_compute_label(cfg)}")

    processed_count = 0
    total = len(episode_dirs)
    live_reporter = LiveProgressReporter(
        script_name="04_diarize_and_transcribe.py",
        total=max(1, total),
        phase_label="Sprecher segmentieren und transkribieren",
        parent_label="Batch",
    )
    for index, episode_dir in enumerate(episode_dirs, start=1):
        if model is None:
            import whisper

            model_dir = resolve_project_path(cfg["paths"]["whisper_model_dir"])
            model_dir.mkdir(parents=True, exist_ok=True)
            model = whisper.load_model(model_name, download_root=str(model_dir), device=device)

            if requested_backend in {"auto", "speechbrain"}:
                try:
                    speechbrain_model = speechbrain_encoder(
                        PROJECT_ROOT.parent / "runtime" / "models" / "speechbrain" / "ecapa",
                        device=device,
                    )
                    speaker_backend = "speechbrain"
                except Exception as exc:
                    if requested_backend == "speechbrain":
                        raise
                    warn(f"SpeechBrain-Speaker-Encoder nicht nutzbar, Fallback auf MFCC: {exc}")
            info(f"Sprecher-Embedding-Backend: {speaker_backend}")

        if process_episode_dir(
            episode_dir,
            cfg,
            ffmpeg,
            model,
            use_fp16,
            speaker_backend,
            speechbrain_model,
            device,
            live_reporter=live_reporter,
            episode_index=index,
            episode_total=total,
        ):
            processed_count += 1
            live_reporter.update(
                index,
                current_label=episode_dir.name,
                parent_label=episode_dir.name,
                extra_label=f"Folge abgeschlossen: {episode_dir.name}",
                scope_current=1,
                scope_total=1,
                scope_started_at=time.time(),
                scope_label=f"Folge {index}/{total}",
            )

    live_reporter.finish(current_label="Batch", extra_label=f"Folgen verarbeitet: {processed_count}")
    ok(f"Batch abgeschlossen: {processed_count} Folgen in 04 verarbeitet.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
