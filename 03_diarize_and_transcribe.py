#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "ai_series_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse
import time
import wave
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from support_scripts.pipeline_common import (
    DistributedLeaseHeartbeat,
    LiveProgressReporter,
    PROJECT_ROOT,
    acquire_distributed_lease,
    completed_step_state,
    coalesce_text,
    cosine_similarity,
    detect_tool,
    distributed_heartbeat_interval_seconds,
    distributed_lease_ttl_seconds,
    distributed_poll_interval_seconds,
    distributed_runtime_enabled,
    distributed_step_runtime_root,
    distributed_worker_id,
    distributed_worker_metadata,
    error,
    first_dir,
    headline,
    info,
    language_hint_from_name,
    detect_language_from_text,
    language_text_marker_score,
    limited_items,
    load_distributed_lease,
    load_step_autosave,
    load_config,
    normalize_language_code,
    ok,
    preferred_compute_label,
    preferred_execution_label,
    preferred_torch_device,
    progress,
    release_distributed_lease,
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

PROCESS_VERSION = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment Speakers And Transcribe")
    parser.add_argument("--episode", help="Name of the scene folder under data/processed/scene_clips.")
    parser.add_argument("--no-shared-workers", action="store_true", help="Disable NAS/shared-worker scene leasing.")
    parser.add_argument("--worker-id", help="Optional stable worker id for shared NAS processing.")
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
    if not isinstance(rows, list) or not isinstance(cluster_payload, dict):
        return False
    if int(cluster_payload.get("process_version", 0) or 0) != PROCESS_VERSION:
        return False
    if rows and rows[0].get("process_version") != PROCESS_VERSION:
        return False
    autosave_state = completed_step_state("03_diarize_and_transcribe", episode_dir.name, PROCESS_VERSION)
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
        raise FileNotFoundError(f"Scene folder not found: {episode_name}")
    return next_untranscribed_episode_dir(scene_root, cfg)


def resolve_episode_dirs_for_processing(scene_root: Path, episode_name: str | None, cfg: dict) -> list[Path]:
    if episode_name:
        episode_dir = resolve_episode_dir(scene_root, episode_name, cfg)
        return [episode_dir] if episode_dir is not None else []
    return pending_untranscribed_episode_dirs(scene_root, cfg)


def scene_cache_path(scene_cache_dir: Path, scene_name: str) -> Path:
    return scene_cache_dir / f"{scene_name}.json"


def scene_cache_completed(cache_file: Path) -> bool:
    if not cache_file.exists():
        return False
    try:
        rows = resolve_rows(cache_file)
    except Exception:
        return False
    if not isinstance(rows, list):
        return False
    if not rows:
        return True
    return bool(rows[0].get("process_version") == PROCESS_VERSION)


def completed_scene_cache_ids(scene_cache_dir: Path) -> set[str]:
    completed: set[str] = set()
    if not scene_cache_dir.exists():
        return completed
    for cache_file in scene_cache_dir.glob("scene_*.json"):
        if scene_cache_completed(cache_file):
            completed.add(cache_file.stem)
    return completed


def completed_scene_segment_count(scene_cache_dir: Path) -> int:
    total = 0
    if not scene_cache_dir.exists():
        return total
    for cache_file in scene_cache_dir.glob("scene_*.json"):
        if not scene_cache_completed(cache_file):
            continue
        rows = resolve_rows(cache_file)
        if isinstance(rows, list):
            total += len(rows)
    return total


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


def transcribe_language_hint(cfg: dict, *source_names: object) -> str:
    transcription_cfg = cfg.get("transcription", {}) if isinstance(cfg.get("transcription", {}), dict) else {}
    configured = normalize_language_code(transcription_cfg.get("language", ""))
    if configured not in {"", "auto", "detect"}:
        return configured
    return language_hint_from_name(*source_names)


def configured_transcription_language(cfg: dict) -> str:
    transcription_cfg = cfg.get("transcription", {}) if isinstance(cfg.get("transcription", {}), dict) else {}
    configured = normalize_language_code(transcription_cfg.get("language", ""))
    return "" if configured in {"", "auto", "detect"} else configured


def auto_language_candidates(cfg: dict) -> list[str]:
    transcription_cfg = cfg.get("transcription", {}) if isinstance(cfg.get("transcription", {}), dict) else {}
    configured = transcription_cfg.get("auto_language_candidates", ["de", "en", "es", "fr", "it", "pt", "nl", "tr", "pl"])
    if isinstance(configured, str):
        raw_items = [item.strip() for item in configured.split(",")]
    elif isinstance(configured, list):
        raw_items = configured
    else:
        raw_items = []
    languages: list[str] = []
    for item in raw_items:
        language = normalize_language_code(item)
        if language and language not in languages:
            languages.append(language)
    return languages or ["de", "en", "es", "fr", "it", "pt", "nl", "tr", "pl"]


def config_with_transcription_language(cfg: dict, language: str) -> dict:
    normalized = normalize_language_code(language)
    if not normalized:
        return cfg
    updated = deepcopy(cfg)
    transcription_cfg = updated.setdefault("transcription", {})
    if isinstance(transcription_cfg, dict):
        transcription_cfg["language"] = normalized
    return updated


def representative_language_probe_scenes(scenes: list[Path], cfg: dict) -> list[Path]:
    transcription_cfg = cfg.get("transcription", {}) if isinstance(cfg.get("transcription", {}), dict) else {}
    scene_count = max(1, int(transcription_cfg.get("auto_language_probe_scene_count", 4) or 4))
    min_bytes = max(0, int(transcription_cfg.get("auto_language_probe_min_scene_bytes", 120000) or 0))
    ranked = sorted(
        [scene for scene in scenes if scene.is_file()],
        key=lambda scene: scene.stat().st_size if scene.exists() else 0,
        reverse=True,
    )
    selected = [scene for scene in ranked if scene.stat().st_size >= min_bytes][:scene_count]
    return selected or ranked[:scene_count]


def episode_language_cache_path(scene_cache_dir: Path) -> Path:
    return scene_cache_dir / "_episode_language.json"


def normalize_whisper_language_key(value: object) -> str:
    text = coalesce_text(str(value or "")).replace("<|", "").replace("|>", "").strip("|")
    return normalize_language_code(text)


def rank_episode_language_probability_rows(probability_rows: list[dict[str, float]], cfg: dict) -> tuple[str, dict[str, float]]:
    if not probability_rows:
        return "", {}
    candidates = set(auto_language_candidates(cfg))
    scores: dict[str, float] = {}
    for row in probability_rows:
        for language, probability in row.items():
            normalized = normalize_language_code(language)
            if not normalized:
                continue
            if candidates and normalized not in candidates:
                continue
            scores[normalized] = scores.get(normalized, 0.0) + max(0.0, float(probability or 0.0))
    if not scores:
        return "", {}
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_language, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    row_count = max(1, len(probability_rows))
    transcription_cfg = cfg.get("transcription", {}) if isinstance(cfg.get("transcription", {}), dict) else {}
    min_confidence = float(transcription_cfg.get("auto_language_min_confidence", 0.35) or 0.35)
    min_margin = float(transcription_cfg.get("auto_language_min_margin", 0.08) or 0.08)
    best_average = best_score / row_count
    second_average = second_score / row_count
    if best_average >= min_confidence and best_average - second_average >= min_margin:
        return best_language, scores
    if best_average >= 0.6 and best_average > second_average:
        return best_language, scores
    return "", scores


def whisper_audio_language_probabilities(model, scene_wav: Path, cfg: dict, device: str) -> dict[str, float]:
    if not hasattr(model, "detect_language"):
        return {}
    try:
        import whisper
    except Exception:
        return {}
    transcription_cfg = cfg.get("transcription", {}) if isinstance(cfg.get("transcription", {}), dict) else {}
    max_seconds = max(1.0, float(transcription_cfg.get("auto_language_probe_max_seconds", 30.0) or 30.0))
    try:
        audio = whisper.load_audio(str(scene_wav))
        audio = audio[: int(max_seconds * 16000)]
        audio = whisper.pad_or_trim(audio)
        n_mels = int(getattr(getattr(model, "dims", None), "n_mels", 80) or 80)
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)
        target_device = getattr(model, "device", None) or device
        if target_device:
            mel = mel.to(target_device)
        _detected, probabilities = model.detect_language(mel)
    except Exception as exc:
        warn(f"Episode language audio probe failed for {scene_wav.name}: {exc}")
        return {}
    rows: dict[str, float] = {}
    for language, probability in dict(probabilities).items():
        normalized = normalize_whisper_language_key(language)
        if normalized:
            rows[normalized] = max(rows.get(normalized, 0.0), float(probability or 0.0))
    return rows


def text_from_transcription_payload(payload: dict) -> str:
    raw_segments = payload.get("segments") or []
    pieces = [coalesce_text(payload.get("text", ""))]
    pieces.extend(
        coalesce_text(segment.get("text", ""))
        for segment in raw_segments
        if isinstance(segment, dict)
    )
    return coalesce_text(" ".join(piece for piece in pieces if piece))


def language_probe_text_score(text: object, language: str) -> float:
    normalized = normalize_language_code(language)
    if not normalized:
        return 0.0
    content = coalesce_text(str(text or ""))
    marker_score = language_text_marker_score(content, normalized)
    detected_language = detect_language_from_text(content)
    token_count = len(tokens_from_text(content))
    score = float(marker_score * 4)
    if detected_language == normalized:
        score += 8.0
    score += min(token_count, 40) * 0.05
    return score


def forced_language_probe(
    model,
    scene_wav: Path,
    cfg: dict,
    use_fp16: bool,
    candidates: list[str],
    probability_scores: dict[str, float] | None = None,
) -> tuple[str, dict[str, float]]:
    transcription_cfg = cfg.get("transcription", {}) if isinstance(cfg.get("transcription", {}), dict) else {}
    if not bool(transcription_cfg.get("auto_language_forced_probe", False)):
        return "", {}
    max_candidates = max(1, int(transcription_cfg.get("auto_language_forced_probe_candidates", 4) or 4))
    unique_candidates: list[str] = []
    for language in candidates:
        normalized = normalize_language_code(language)
        if normalized and normalized not in unique_candidates:
            unique_candidates.append(normalized)
    scores: dict[str, float] = {}
    for language in unique_candidates[:max_candidates]:
        transcribe_kwargs = {
            "task": transcription_cfg.get("task", "transcribe"),
            "condition_on_previous_text": False,
            "temperature": 0.0,
            "verbose": False,
            "fp16": use_fp16,
            "language": language,
        }
        try:
            payload = model.transcribe(str(scene_wav), **transcribe_kwargs)
        except Exception as exc:
            warn(f"Forced language probe failed for {language} on {scene_wav.name}: {exc}")
            continue
        text = text_from_transcription_payload(payload if isinstance(payload, dict) else {})
        score = language_probe_text_score(text, language)
        score += float((probability_scores or {}).get(language, 0.0)) * 2.0
        scores[language] = score
    if not scores:
        return "", {}
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_language, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score >= 8.0 and best_score >= second_score + 2.0:
        return best_language, scores
    if best_score >= 12.0 and best_score > second_score:
        return best_language, scores
    return "", scores


def merge_language_score_rows(score_rows: list[dict[str, float]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for row in score_rows:
        if not isinstance(row, dict):
            continue
        for language, score in row.items():
            normalized = normalize_language_code(language)
            if not normalized:
                continue
            merged[normalized] = merged.get(normalized, 0.0) + max(0.0, float(score or 0.0))
    return merged


def rank_forced_language_scores(score_rows: list[dict[str, float]]) -> tuple[str, dict[str, float]]:
    scores = merge_language_score_rows(score_rows)
    if not scores:
        return "", {}
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_language, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score >= 8.0 and best_score >= second_score + 2.0:
        return best_language, scores
    if best_score >= 12.0 and best_score > second_score:
        return best_language, scores
    return "", scores


def detect_episode_language(
    model,
    ffmpeg_path: Path,
    episode_dir: Path,
    scenes: list[Path],
    audio_root: Path,
    scene_cache_dir: Path,
    cfg: dict,
    use_fp16: bool,
    device: str,
) -> str:
    explicit_language = configured_transcription_language(cfg)
    if explicit_language:
        return explicit_language

    cache_file = episode_language_cache_path(scene_cache_dir)
    try:
        cached = resolve_rows(cache_file) if cache_file.exists() else {}
    except Exception:
        cached = {}
    if isinstance(cached, dict) and int(cached.get("process_version", 0) or 0) == PROCESS_VERSION:
        cached_language = normalize_language_code(cached.get("detected_language", ""))
        if cached_language:
            return cached_language

    probe_scenes = representative_language_probe_scenes(scenes, cfg)
    episode_audio_dir = audio_root / episode_dir.name
    episode_audio_dir.mkdir(parents=True, exist_ok=True)
    probability_rows: list[dict[str, float]] = []
    for scene_file in probe_scenes:
        scene_wav = episode_audio_dir / f"{scene_file.stem}.wav"
        if not scene_wav.exists():
            export_audio(ffmpeg_path, scene_file, scene_wav)
        probabilities = whisper_audio_language_probabilities(model, scene_wav, cfg, device)
        if probabilities:
            probability_rows.append(probabilities)

    probability_language, probability_scores = rank_episode_language_probability_rows(probability_rows, cfg)
    ranked_probability_languages = [
        language
        for language, _score in sorted(probability_scores.items(), key=lambda item: (-item[1], item[0]))
    ]
    forced_candidates = ranked_probability_languages + auto_language_candidates(cfg)
    forced_score_rows: list[dict[str, float]] = []
    for probe_scene in probe_scenes:
        probe_wav = episode_audio_dir / f"{probe_scene.stem}.wav"
        _forced_language, scene_forced_scores = forced_language_probe(
            model,
            probe_wav,
            cfg,
            use_fp16,
            forced_candidates,
            probability_scores,
        )
        if scene_forced_scores:
            forced_score_rows.append(scene_forced_scores)
    forced_language, forced_scores = rank_forced_language_scores(forced_score_rows)

    fallback_hint = language_hint_from_name(episode_dir.name)
    detected_language = forced_language or probability_language or fallback_hint
    write_json(
        cache_file,
        {
            "process_version": PROCESS_VERSION,
            "detected_language": detected_language,
            "probability_language": probability_language,
            "forced_language": forced_language,
            "fallback_hint": fallback_hint,
            "probability_scores": probability_scores,
            "forced_scores": forced_scores,
            "probe_scenes": [scene.name for scene in probe_scenes],
        },
    )
    if detected_language:
        info(f"Episode language detected for {episode_dir.name}: {detected_language}")
    else:
        warn(f"Episode language could not be detected confidently for {episode_dir.name}; Whisper auto mode remains active.")
    return detected_language


def language_counts_from_rows(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        text_language = detect_language_from_text(row.get("text", ""))
        language = text_language or normalize_language_code(row.get("language", ""))
        if not language:
            continue
        counts[language] = counts.get(language, 0) + 1
    return counts


def dominant_count_language(language_counts: dict[str, int], fallback: str = "") -> str:
    normalized_fallback = normalize_language_code(fallback)
    if not language_counts:
        return normalized_fallback
    ranked = sorted(language_counts.items(), key=lambda item: (-int(item[1]), item[0]))
    if not ranked:
        return normalized_fallback
    return normalize_language_code(ranked[0][0], normalized_fallback)


def episode_language_consensus(rows: list[dict], cfg: dict) -> tuple[str, dict[str, int]]:
    explicit_language = configured_transcription_language(cfg)
    counts = language_counts_from_rows(rows)
    if explicit_language:
        return explicit_language, counts
    combined_text = " ".join(coalesce_text(str(row.get("text", ""))) for row in rows if isinstance(row, dict))
    text_language = detect_language_from_text(combined_text)
    return text_language or dominant_count_language(counts), counts


def apply_episode_language_consensus(rows: list[dict], cfg: dict) -> tuple[list[dict], str, dict[str, int]]:
    episode_language, counts = episode_language_consensus(rows, cfg)
    if not episode_language:
        return rows, episode_language, counts
    transcription_cfg = cfg.get("transcription", {}) if isinstance(cfg.get("transcription", {}), dict) else {}
    lock_segments = bool(transcription_cfg.get("lock_segments_to_episode_language", True))
    normalized_rows: list[dict] = []
    for row in rows:
        updated = dict(row)
        text_language = detect_language_from_text(updated.get("text", ""))
        updated["language"] = episode_language if lock_segments else text_language or episode_language
        normalized_rows.append(updated)
    return normalized_rows, episode_language, counts


def active_scene_lease_ids(scene_lease_root: Path) -> list[str]:
    if not scene_lease_root.exists():
        return []
    active: list[str] = []
    now = time.time()
    for lease_file in scene_lease_root.glob("*.json"):
        payload = load_distributed_lease(scene_lease_root, lease_file.stem)
        if float(payload.get("expires_at", 0.0) or 0.0) > now:
            active.append(lease_file.stem)
    return sorted(active)


def stale_shared_transcription_run(
    autosave_state: dict,
    *,
    completed_scene_ids: list[str],
    active_scene_leases: list[str],
) -> bool:
    if not isinstance(autosave_state, dict):
        return False
    if str(autosave_state.get("status", "")) != "in_progress":
        return False
    if completed_scene_ids:
        return False
    if active_scene_leases:
        return False
    return True


def transcribe_scene(model, scene_wav: Path, cfg: dict, use_fp16: bool) -> dict[str, object]:
    full_duration = wav_duration_seconds(scene_wav)
    transcribe_kwargs = {
        "task": cfg["transcription"]["task"],
        "condition_on_previous_text": False,
        "temperature": 0.0,
        "verbose": False,
        "fp16": use_fp16,
    }
    explicit_language = configured_transcription_language(cfg)
    language_hint = transcribe_language_hint(cfg, scene_wav.name, scene_wav.stem, scene_wav.parent.name)
    if language_hint:
        transcribe_kwargs["language"] = language_hint
    result = model.transcribe(str(scene_wav), **transcribe_kwargs)
    raw_segments = result.get("segments") or []
    text_for_language = " ".join(
        [coalesce_text(result.get("text", ""))]
        + [
            coalesce_text(item.get("text", ""))
            for item in raw_segments
            if isinstance(item, dict)
        ]
    )
    whisper_language = normalize_language_code(result.get("language", ""), language_hint)
    text_language = detect_language_from_text(text_for_language)
    detected_language = explicit_language or text_language or language_hint or whisper_language
    raw_segments = result.get("segments") or []
    segments = []
    for item in raw_segments:
        text = coalesce_text(item.get("text", ""))
        start_sec = max(0.0, float(item.get("start", 0.0)))
        end_sec = max(start_sec + 0.15, float(item.get("end", start_sec + 0.15)))
        if not text and end_sec - start_sec < 0.35:
            continue
        segments.append(
            {
                "start": start_sec,
                "end": end_sec,
                "text": text,
                "language": detect_language_from_text(text, normalize_language_code(item.get("language", ""), detected_language)),
            }
        )
    if not segments:
        text = coalesce_text(result.get("text", ""))
        if text or full_duration > 0.1:
            segments = [{"start": 0.0, "end": max(full_duration, 0.15), "text": text, "language": detected_language}]
    merged_segments = merge_segments(
        segments,
        float(cfg["transcription"].get("merge_gap_seconds", 0.35)),
        float(cfg["transcription"].get("min_segment_seconds", 0.6)),
        full_duration,
    )
    for segment in merged_segments:
        segment["language"] = normalize_language_code(segment.get("language", ""), detected_language)
    return {"segments": merged_segments, "detected_language": detected_language}


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

    transcription_payload = transcribe_scene(model, scene_wav, cfg, use_fp16)
    segments = transcription_payload.get("segments", []) if isinstance(transcription_payload.get("segments", []), list) else []
    detected_language = normalize_language_code(transcription_payload.get("detected_language", ""))
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
                "language": normalize_language_code(segment.get("language", ""), detected_language),
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


def collect_episode_scene_rows(scene_cache_dir: Path, scenes: list[Path]) -> list[dict]:
    all_rows: list[dict] = []
    for scene_file in scenes:
        cache_file = scene_cache_path(scene_cache_dir, scene_file.stem)
        if not scene_cache_completed(cache_file):
            raise RuntimeError(f"Scene cache is not ready yet: {scene_file.stem}")
        rows = resolve_rows(cache_file)
        if isinstance(rows, list):
            all_rows.extend(rows)
    return all_rows


def finalize_episode_transcription_outputs(
    episode_dir: Path,
    scenes: list[Path],
    cfg: dict,
    scene_cache_dir: Path,
    combined_dir: Path,
    speaker_backend: str,
) -> tuple[Path, Path, int]:
    all_rows = collect_episode_scene_rows(scene_cache_dir, scenes)
    all_rows, episode_language, language_counts = apply_episode_language_consensus(all_rows, cfg)
    threshold_key = "voice_embedding_threshold_speechbrain" if speaker_backend == "speechbrain" else "voice_embedding_threshold"
    threshold_default = 0.58 if speaker_backend == "speechbrain" else 0.84
    threshold = float(cfg["transcription"].get(threshold_key, threshold_default))
    all_rows, clusters = assign_speaker_clusters(all_rows, threshold, cfg)
    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        grouped_rows[str(row.get("scene_id", ""))].append(row)
    for scene_file in scenes:
        write_json(scene_cache_path(scene_cache_dir, scene_file.stem), grouped_rows.get(scene_file.stem, []))

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
            "scene_count": len(scenes),
            "segment_count": len(all_rows),
            "detected_language": episode_language,
            "language_counts": language_counts,
        },
    )
    return combined_file, cluster_file, len(all_rows)


def process_episode_dir(
    episode_dir: Path,
    cfg: dict,
    ffmpeg: Path,
    model,
    use_fp16: bool,
    speaker_backend: str,
    speechbrain_model,
    device: str,
    worker_id: str,
    shared_workers: bool,
    live_reporter: LiveProgressReporter | None = None,
    episode_index: int = 1,
    episode_total: int = 1,
) -> bool:
    autosave_target = episode_dir.name
    if episode_transcription_completed(episode_dir, cfg):
        mark_step_completed(
            "03_diarize_and_transcribe",
            autosave_target,
            {
                "episode_id": episode_dir.name,
                "process_version": PROCESS_VERSION,
                "combined_file": str(resolve_project_path(cfg["paths"]["speaker_transcripts"]) / f"{episode_dir.name}_segments.json"),
                "cluster_file": str(resolve_project_path(cfg["paths"]["speaker_segments"]) / episode_dir.name / "_speaker_clusters.json"),
            },
        )
        ok(f"Transcription already completed successfully: {episode_dir.name}")
        return False

    scenes = limited_items(sorted(episode_dir.glob("*.mp4")))
    if not scenes:
        info("No scene files found.")
        return False

    audio_root = resolve_project_path("data/raw/audio")
    scene_cache_dir = resolve_project_path(cfg["paths"]["speaker_segments"]) / episode_dir.name
    combined_dir = resolve_project_path(cfg["paths"]["speaker_transcripts"])
    scene_cache_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)
    lease_root = distributed_step_runtime_root("03_diarize_and_transcribe", episode_dir.name)
    scene_lease_root = lease_root / "scenes"
    finalize_lease_root = lease_root / "finalize"
    lease_ttl_seconds = distributed_lease_ttl_seconds(cfg)
    heartbeat_interval_seconds = distributed_heartbeat_interval_seconds(cfg)
    poll_interval_seconds = distributed_poll_interval_seconds(cfg)
    completed_scene_ids: list[str] = sorted(completed_scene_cache_ids(scene_cache_dir))
    autosave_state = load_step_autosave("03_diarize_and_transcribe", autosave_target)
    active_leases = active_scene_lease_ids(scene_lease_root) if shared_workers else []
    recovered_stale_run = stale_shared_transcription_run(
        autosave_state,
        completed_scene_ids=completed_scene_ids,
        active_scene_leases=active_leases,
    )
    if recovered_stale_run:
        warn(
            "Recovering an abandoned shared-worker transcription run without scene-cache outputs. "
            "A previous worker stopped before any scene cache was written."
        )
    mark_step_started(
        "03_diarize_and_transcribe",
        autosave_target,
        {
            "episode_id": episode_dir.name,
            "process_version": PROCESS_VERSION,
            "scene_count": len(scenes),
            "device": device,
            "model_name": cfg["transcription"]["model_name"],
            "completed_scene_ids": completed_scene_ids,
            "active_scene_lease_count": len(active_leases),
            "recovered_stale_run": recovered_stale_run,
            "worker_id": worker_id,
            "shared_workers": shared_workers,
        },
    )
    contributed = False
    try:
        episode_started_at = time.time()
        episode_language = detect_episode_language(
            model,
            ffmpeg,
            episode_dir,
            scenes,
            audio_root,
            scene_cache_dir,
            cfg,
            use_fp16,
            device,
        )
        effective_cfg = config_with_transcription_language(cfg, episode_language)
        scene_index_by_id = {scene_file.stem: index for index, scene_file in enumerate(scenes, start=1)}
        while len(completed_scene_ids) < len(scenes):
            claimed_scene = False
            for scene_file in scenes:
                cache_file = scene_cache_path(scene_cache_dir, scene_file.stem)
                if scene_cache_completed(cache_file):
                    continue
                heartbeat = None
                if shared_workers:
                    lease_meta = lambda: distributed_worker_metadata(  # noqa: E731
                        {
                            "step": "03_diarize_and_transcribe",
                            "episode_id": episode_dir.name,
                            "scene_id": scene_file.stem,
                        }
                    )
                    lease = acquire_distributed_lease(
                        scene_lease_root,
                        scene_file.stem,
                        worker_id,
                        lease_ttl_seconds,
                        meta=lease_meta(),
                    )
                    if lease is None:
                        continue
                    heartbeat = DistributedLeaseHeartbeat(
                        root=scene_lease_root,
                        lease_name=scene_file.stem,
                        owner_id=worker_id,
                        ttl_seconds=lease_ttl_seconds,
                        interval_seconds=heartbeat_interval_seconds,
                        meta_factory=lease_meta,
                    )
                    heartbeat.start()
                try:
                    if scene_cache_completed(cache_file):
                        continue
                    scene_rows = process_scene(
                        model,
                        ffmpeg,
                        episode_dir.name,
                        scene_file,
                        audio_root,
                        scene_cache_dir,
                        effective_cfg,
                        use_fp16,
                        speaker_backend,
                        speechbrain_model=speechbrain_model,
                        device=device,
                    )
                    contributed = True
                    completed_scene_ids = sorted(completed_scene_cache_ids(scene_cache_dir))
                    save_step_autosave(
                        "03_diarize_and_transcribe",
                        autosave_target,
                        {
                            "status": "in_progress",
                            "episode_id": episode_dir.name,
                            "process_version": PROCESS_VERSION,
                            "scene_count": len(scenes),
                            "completed_scene_ids": completed_scene_ids,
                            "segment_count": completed_scene_segment_count(scene_cache_dir),
                            "last_scene_id": scene_file.stem,
                            "worker_id": worker_id,
                            "shared_workers": shared_workers,
                        },
                    )
                    if live_reporter is not None:
                        live_reporter.update(
                            (episode_index - 1) + (len(completed_scene_ids) / max(1, len(scenes))),
                            current_label=scene_file.name,
                            parent_label=episode_dir.name,
                            extra_label=f"Completed scenes: {len(completed_scene_ids)}/{len(scenes)} | Segments in scene: {len(scene_rows)}",
                            scope_current=scene_index_by_id.get(scene_file.stem, len(completed_scene_ids)),
                            scope_total=len(scenes),
                            scope_started_at=episode_started_at,
                            scope_label=f"Episode {episode_index}/{episode_total}",
                        )
                    claimed_scene = True
                    break
                finally:
                    if heartbeat is not None:
                        heartbeat.stop()
                        release_distributed_lease(scene_lease_root, scene_file.stem, worker_id)
                if not shared_workers:
                    break

            if claimed_scene:
                continue

            completed_scene_ids = sorted(completed_scene_cache_ids(scene_cache_dir))
            if live_reporter is not None:
                live_reporter.update(
                    (episode_index - 1) + (len(completed_scene_ids) / max(1, len(scenes))),
                    current_label="Waiting for shared workers",
                    parent_label=episode_dir.name,
                    extra_label=f"Completed scenes: {len(completed_scene_ids)}/{len(scenes)}",
                    scope_current=min(len(completed_scene_ids) + 1, len(scenes)),
                    scope_total=len(scenes),
                    scope_started_at=episode_started_at,
                    scope_label=f"Episode {episode_index}/{episode_total}",
                )
            if len(completed_scene_ids) >= len(scenes):
                break
            time.sleep(poll_interval_seconds)

        completed_scene_ids = sorted(completed_scene_cache_ids(scene_cache_dir))
        combined_file = combined_dir / f"{episode_dir.name}_segments.json"
        cluster_file = scene_cache_dir / "_speaker_clusters.json"
        if not episode_transcription_completed(episode_dir, cfg):
            finalize_heartbeat = None
            finalize_acquired = not shared_workers
            if shared_workers:
                finalize_meta = lambda: distributed_worker_metadata(  # noqa: E731
                    {
                        "step": "03_diarize_and_transcribe",
                        "episode_id": episode_dir.name,
                        "phase": "finalize",
                    }
                )
                finalize_acquired = (
                    acquire_distributed_lease(
                        finalize_lease_root,
                        "episode_finalize",
                        worker_id,
                        lease_ttl_seconds,
                        meta=finalize_meta(),
                    )
                    is not None
                )
                if finalize_acquired:
                    finalize_heartbeat = DistributedLeaseHeartbeat(
                        root=finalize_lease_root,
                        lease_name="episode_finalize",
                        owner_id=worker_id,
                        ttl_seconds=lease_ttl_seconds,
                        interval_seconds=heartbeat_interval_seconds,
                        meta_factory=finalize_meta,
                    )
                    finalize_heartbeat.start()
            try:
                if finalize_acquired and not episode_transcription_completed(episode_dir, cfg):
                    combined_file, cluster_file, segment_count = finalize_episode_transcription_outputs(
                        episode_dir,
                        scenes,
                        effective_cfg,
                        scene_cache_dir,
                        combined_dir,
                        speaker_backend,
                    )
                    contributed = True
                    mark_step_completed(
                        "03_diarize_and_transcribe",
                        autosave_target,
                        {
                            "episode_id": episode_dir.name,
                            "process_version": PROCESS_VERSION,
                            "scene_count": len(scenes),
                            "segment_count": segment_count,
                            "completed_scene_ids": completed_scene_ids,
                            "combined_file": str(combined_file),
                            "cluster_file": str(cluster_file),
                            "backend": speaker_backend,
                            "detected_language": episode_language,
                            "worker_id": worker_id,
                            "shared_workers": shared_workers,
                        },
                    )
                    ok(f"Saved segment transcripts: {segment_count} segments")
                elif shared_workers:
                    while not episode_transcription_completed(episode_dir, cfg):
                        time.sleep(poll_interval_seconds)
            finally:
                if finalize_heartbeat is not None:
                    finalize_heartbeat.stop()
                    release_distributed_lease(finalize_lease_root, "episode_finalize", worker_id)

        if episode_transcription_completed(episode_dir, cfg) and not completed_step_state(
            "03_diarize_and_transcribe", autosave_target, PROCESS_VERSION
        ):
            cluster_payload = resolve_rows(cluster_file) if cluster_file.exists() else {}
            rows = resolve_rows(combined_file) if combined_file.exists() else []
            mark_step_completed(
                "03_diarize_and_transcribe",
                autosave_target,
                {
                    "episode_id": episode_dir.name,
                    "process_version": PROCESS_VERSION,
                    "scene_count": len(scenes),
                    "segment_count": len(rows) if isinstance(rows, list) else 0,
                    "completed_scene_ids": completed_scene_ids,
                    "combined_file": str(combined_file),
                    "cluster_file": str(cluster_file),
                    "backend": str(cluster_payload.get("backend", speaker_backend)) if isinstance(cluster_payload, dict) else speaker_backend,
                    "worker_id": worker_id,
                    "shared_workers": shared_workers,
                },
            )
        return contributed
    except Exception as exc:
        mark_step_failed(
            "03_diarize_and_transcribe",
            str(exc),
            autosave_target,
            {
                "episode_id": episode_dir.name,
                "process_version": PROCESS_VERSION,
                "scene_count": len(scenes),
                "completed_scene_ids": completed_scene_ids,
                "worker_id": worker_id,
                "shared_workers": shared_workers,
            },
        )
        raise


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Segment Speakers And Transcribe")
    cfg = load_config()
    ffmpeg = detect_tool(PROJECT_ROOT / "tools" / "ffmpeg" / "bin", "ffmpeg")
    scene_root = resolve_project_path(cfg["paths"]["scene_clips"])
    episode_dirs = resolve_episode_dirs_for_processing(scene_root, args.episode, cfg)
    if not episode_dirs:
        if args.episode:
            info("No matching scene folders found.")
        else:
            info("No pending episodes found for step 04.")
        return

    model_name = cfg["transcription"]["model_name"]
    device = preferred_torch_device(cfg)
    use_fp16 = device == "cuda"
    worker_id = coalesce_text(args.worker_id) or distributed_worker_id()
    shared_workers = distributed_runtime_enabled(cfg) and not args.no_shared_workers
    requested_backend = str(cfg["transcription"].get("speaker_embedding_backend", "auto")).strip().lower()
    speaker_backend = "mfcc"
    speechbrain_model = None
    model = None
    info(f"Whisper model: {model_name}")
    info(f"Execution mode: {preferred_execution_label(cfg)}")
    info(f"Compute device: {preferred_compute_label(cfg)}")
    info(f"Shared NAS workers: {'enabled' if shared_workers else 'disabled'}")
    if shared_workers:
        info(f"Worker id: {worker_id}")

    processed_count = 0
    total = len(episode_dirs)
    live_reporter = LiveProgressReporter(
        script_name="03_diarize_and_transcribe.py",
        total=max(1, total),
        phase_label="Segment Speakers And Transcribe",
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
                        resolve_project_path("runtime/models/speechbrain/ecapa"),
                        device=device,
                    )
                    speaker_backend = "speechbrain"
                except Exception as exc:
                    if requested_backend == "speechbrain":
                        raise
                    warn(f"SpeechBrain speaker encoder unavailable, falling back to MFCC: {exc}")
            info(f"Speaker embedding backend: {speaker_backend}")

        if process_episode_dir(
            episode_dir,
            cfg,
            ffmpeg,
            model,
            use_fp16,
            speaker_backend,
            speechbrain_model,
            device,
            worker_id,
            shared_workers,
            live_reporter=live_reporter,
            episode_index=index,
            episode_total=total,
        ):
            processed_count += 1
            live_reporter.update(
                index,
                current_label=episode_dir.name,
                parent_label=episode_dir.name,
                extra_label=f"Episode completed: {episode_dir.name}",
                scope_current=1,
                scope_total=1,
                scope_started_at=time.time(),
                scope_label=f"Episode {index}/{total}",
            )

    live_reporter.finish(current_label="Batch", extra_label=f"Episodes processed: {processed_count}")
    ok(f"Batch completed: {processed_count} episodes processed in 04.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise


