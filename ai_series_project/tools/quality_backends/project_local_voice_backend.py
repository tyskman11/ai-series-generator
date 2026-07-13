#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
import wave
from datetime import datetime
from pathlib import Path

from backend_common import (
    ensure_parent,
    existing_file_path,
    find_project_local_ffmpeg,
    load_backend_context,
    load_json,
    print_runtime_error,
)

AUDIO_PATTERNS = ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.ogg")
NON_SPEECH_AUDIO_TYPES = {"music", "applause", "laughter", "noise", "ambience", "sfx", "silence", "uncertain_non_speech"}
DEFAULT_XTTS_LINE_TIMEOUT_SECONDS = 600
DEFAULT_MIN_REFERENCE_SECONDS = 6.0
DEFAULT_MIN_REFERENCE_COUNT = 2
DEFAULT_MIN_SINGLE_REFERENCE_SECONDS = 0.35
DEFAULT_MAX_REFERENCE_PATHS = 4
DEFAULT_VOXCPM_INFERENCE_TIMESTEPS = 10
DEFAULT_VOXCPM_CFG_VALUE = 2.0


def clean_text(value: object) -> str:
    return str(value or "").strip()


def normalize_language_code(value: object) -> str:
    text = clean_text(value).lower().replace("_", "-")
    if not text or text == "auto":
        return ""
    if text.startswith("de"):
        return "de"
    if text.startswith("en"):
        return "en"
    return text.split("-", 1)[0]


def env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def runtime_float(runtime_cfg: dict, key: str, default: float, env_name: str = "") -> float:
    if env_name and str(os.environ.get(env_name, "") or "").strip():
        return env_float(env_name, default)
    try:
        return float(runtime_cfg.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def runtime_int(runtime_cfg: dict, key: str, default: int, env_name: str = "") -> int:
    if env_name and str(os.environ.get(env_name, "") or "").strip():
        return env_int(env_name, default)
    try:
        return int(float(runtime_cfg.get(key, default) or default))
    except (TypeError, ValueError):
        return default


def reference_dict_eligible(item: dict) -> bool:
    if not isinstance(item, dict):
        return True
    if item.get("voice_reference_eligible") is False:
        return False
    content_type = clean_text(item.get("audio_content_type", "")).lower()
    if content_type in NON_SPEECH_AUDIO_TYPES:
        return False
    text = clean_text(item.get("text", "")).lower()
    if text in {"music", "musik", "applause", "applaus", "laughter", "lachen", "gelächter", "silence", "stille"}:
        return False
    return True


def load_voice_model_metadata(path_value: object) -> dict:
    candidate = existing_file_path(path_value)
    if candidate is None:
        return {}
    payload = load_json(str(candidate))
    return payload if isinstance(payload, dict) else {}


def collect_sample_dir_audio(sample_dir_value: object) -> list[Path]:
    directory = Path(str(sample_dir_value or "").strip())
    if not str(directory) or not directory.exists() or not directory.is_dir():
        return []
    files: list[Path] = []
    for pattern in AUDIO_PATTERNS:
        files.extend(path for path in directory.glob(pattern) if path.is_file() and path.stat().st_size > 0)
    return sorted(files)


def collect_reference_audio_paths(line: dict) -> list[Path]:
    voice_profile = line.get("voice_profile", {}) if isinstance(line.get("voice_profile", {}), dict) else {}
    original_reference = line.get("original_voice_reference", {}) if isinstance(line.get("original_voice_reference", {}), dict) else {}
    voice_model = load_voice_model_metadata(voice_profile.get("voice_model_path", ""))

    candidates: list[Path] = []
    ordered_values: list[object] = []
    if reference_dict_eligible(original_reference):
        ordered_values.append(original_reference.get("audio_path", ""))
    ordered_values.append(voice_profile.get("reference_audio", ""))
    for item in line.get("reference_segments", []) if isinstance(line.get("reference_segments", []), list) else []:
        if isinstance(item, dict) and reference_dict_eligible(item):
            ordered_values.append(clean_text(item.get("audio_path", "")))
    ordered_values.extend(line.get("reference_audio_candidates", []) if isinstance(line.get("reference_audio_candidates", []), list) else [])
    ordered_values.append(voice_model.get("reference_audio", ""))
    ordered_values.extend(voice_model.get("sample_paths", []) if isinstance(voice_model.get("sample_paths", []), list) else [])
    for item in voice_model.get("reference_segments", []) if isinstance(voice_model.get("reference_segments", []), list) else []:
        if isinstance(item, dict) and reference_dict_eligible(item):
            ordered_values.append(clean_text(item.get("audio_path", "")))
    seen: set[str] = set()
    for value in ordered_values:
        candidate = existing_file_path(value)
        if candidate is None:
            continue
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(candidate)
    for directory in line.get("candidate_sample_dirs", []) if isinstance(line.get("candidate_sample_dirs", []), list) else []:
        for candidate in collect_sample_dir_audio(directory):
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(candidate)
    return candidates


def audio_duration_seconds(path: Path) -> float:
    try:
        if path.suffix.lower() == ".wav":
            with wave.open(str(path), "rb") as handle:
                rate = float(handle.getframerate() or 0)
                if rate <= 0:
                    return 0.0
                return float(handle.getnframes()) / rate
    except Exception:
        return 0.0
    return 0.0


def voice_reference_profile(line: dict, runtime_cfg: dict) -> dict:
    all_paths = collect_reference_audio_paths(line)
    min_single_seconds = max(
        0.0,
        runtime_float(
            runtime_cfg,
            "min_single_voice_reference_seconds",
            DEFAULT_MIN_SINGLE_REFERENCE_SECONDS,
            "SERIES_XTTS_MIN_SINGLE_REFERENCE_SECONDS",
        ),
    )
    rows: list[dict] = []
    usable_paths: list[Path] = []
    for path in all_paths:
        duration = audio_duration_seconds(path)
        usable = duration >= min_single_seconds
        rows.append(
            {
                "path": str(path),
                "duration_seconds": round(duration, 3),
                "usable": usable,
            }
        )
        if usable:
            usable_paths.append(path)
    return {
        "candidate_count": len(all_paths),
        "usable_count": len(usable_paths),
        "usable_duration_seconds": round(sum(audio_duration_seconds(path) for path in usable_paths), 3),
        "usable_paths": usable_paths,
        "rows": rows,
    }


def select_xtts_reference_audio_paths(line: dict, runtime_cfg: dict) -> tuple[list[Path], dict]:
    profile = voice_reference_profile(line, runtime_cfg)
    min_reference_seconds = max(
        0.0,
        runtime_float(
            runtime_cfg,
            "min_voice_reference_seconds",
            DEFAULT_MIN_REFERENCE_SECONDS,
            "SERIES_XTTS_MIN_REFERENCE_SECONDS",
        ),
    )
    min_reference_count = max(
        1,
        runtime_int(
            runtime_cfg,
            "min_voice_reference_count",
            DEFAULT_MIN_REFERENCE_COUNT,
            "SERIES_XTTS_MIN_REFERENCE_COUNT",
        ),
    )
    max_reference_paths = max(
        1,
        runtime_int(
            runtime_cfg,
            "max_voice_reference_paths",
            DEFAULT_MAX_REFERENCE_PATHS,
            "SERIES_XTTS_MAX_REFERENCE_PATHS",
        ),
    )
    voice_profile = line.get("voice_profile", {}) if isinstance(line.get("voice_profile", {}), dict) else {}
    voice_model = load_voice_model_metadata(voice_profile.get("voice_model_path", ""))
    clone_ready = bool(voice_model.get("clone_ready", False))
    total_seconds = float(profile.get("usable_duration_seconds", 0.0) or 0.0)
    usable_count = int(profile.get("usable_count", 0) or 0)
    if usable_count < min_reference_count or total_seconds < min_reference_seconds:
        raise RuntimeError(
            f"{line['speaker_name']}: insufficient usable XTTS reference audio for line "
            f"{int(line.get('line_index', 0) or 0) + 1:03d}. "
            f"usable_refs={usable_count}/{min_reference_count}, "
            f"usable_seconds={total_seconds:.2f}/{min_reference_seconds:.2f}, "
            f"voice_model_clone_ready={clone_ready}. "
            "Rerun 03-10 for this speaker or review/repair voice assignments before rendering."
        )
    selected_paths = list(profile.get("usable_paths", []))[:max_reference_paths]
    profile["selected_count"] = len(selected_paths)
    profile["voice_model_clone_ready"] = clone_ready
    profile["voice_model_quality_score"] = voice_model.get("quality_score", "")
    profile["voice_model_duration_seconds_total"] = voice_model.get("duration_seconds_total", "")
    return selected_paths, profile


def collect_line_specs(scene_package: dict) -> list[dict]:
    voice_clone = scene_package.get("voice_clone", {}) if isinstance(scene_package.get("voice_clone"), dict) else {}
    lines = voice_clone.get("lines", []) if isinstance(voice_clone.get("lines"), list) else []
    prepared: list[dict] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        original_reference = line.get("original_voice_reference", {}) if isinstance(line.get("original_voice_reference"), dict) else {}
        runtime_cfg = line.get("runtime", {}) if isinstance(line.get("runtime", {}), dict) else {}
        force_voice_cloning = bool(runtime_cfg.get("force_voice_cloning", True))
        candidate = None
        if not force_voice_cloning:
            candidate = existing_file_path(line.get("target_output_audio", ""))
        if candidate is None and not force_voice_cloning:
            candidate = existing_file_path(original_reference.get("audio_path", ""))
        prepared.append(
            {
                "line_index": int(line.get("line_index", 0) or 0),
                "speaker_name": clean_text(line.get("speaker_name", "")) or "Narrator",
                "text": clean_text(line.get("text", "")),
                "language": normalize_language_code(line.get("language", "")),
                "audio_path": candidate,
                "voice_profile": line.get("voice_profile", {}) if isinstance(line.get("voice_profile", {}), dict) else {},
                "original_voice_reference": original_reference,
                "reference_segments": line.get("reference_segments", []) if isinstance(line.get("reference_segments", []), list) else [],
                "reference_audio_candidates": line.get("reference_audio_candidates", []) if isinstance(line.get("reference_audio_candidates", []), list) else [],
                "candidate_sample_dirs": line.get("candidate_sample_dirs", []) if isinstance(line.get("candidate_sample_dirs", []), list) else [],
                "target_output_audio": clean_text(line.get("target_output_audio", "")),
                "emotion": clean_text(line.get("emotion", "")),
                "pace": clean_text(line.get("pace", "")),
                "energy": float(line.get("energy", 0.0) or 0.0),
                "pause_after_seconds": max(0.0, float(line.get("pause_after_seconds", 0.0) or 0.0)),
                "delivery_notes": clean_text(line.get("delivery_notes", "")),
                "runtime": runtime_cfg,
                "force_voice_cloning": force_voice_cloning,
            }
        )
    return prepared


def load_xtts_runtime(runtime_cfg: dict):
    if clean_text(runtime_cfg.get("engine", "")) != "xtts":
        return None, ""
    if not bool(runtime_cfg.get("xtts_license_accepted", False)):
        return None, "XTTS license is not accepted in the local project configuration."
    try:
        from TTS.api import TTS
    except Exception as exc:
        return None, f"XTTS runtime is not available: {exc}"
    try:
        import torch

        device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
    except Exception:
        device = "cpu"
    model_name = clean_text(runtime_cfg.get("xtts_model_name", "")) or "tts_models/multilingual/multi-dataset/xtts_v2"
    try:
        synthesizer = TTS(model_name=model_name)
        if hasattr(synthesizer, "to"):
            synthesizer = synthesizer.to(device)
        return synthesizer, ""
    except Exception as exc:
        return None, f"XTTS could not be initialized: {exc}"


def synthesize_xtts_line(
    synthesizer,
    text: str,
    language: str,
    reference_paths: list[Path],
    output_path: Path,
    timeout_seconds: int = 0,
) -> None:
    kwargs = {
        "text": text,
        "file_path": str(output_path),
    }
    if reference_paths:
        kwargs["speaker_wav"] = [str(path) for path in reference_paths[:4]]
    if language:
        kwargs["language"] = language
    if timeout_seconds > 0 and hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer"):
        def timeout_handler(_signum, _frame):
            raise TimeoutError(f"XTTS line synthesis exceeded {timeout_seconds} seconds.")

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
        try:
            synthesizer.tts_to_file(**kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
        return
    synthesizer.tts_to_file(**kwargs)


def load_voxcpm_runtime(runtime_cfg: dict):
    if clean_text(runtime_cfg.get("engine", "")).lower() not in {"voxcpm", "voxcpm2"}:
        return None, ""
    model_dir = clean_text(os.environ.get("SERIES_VOICE_MODEL_DIR", "")) or clean_text(runtime_cfg.get("voice_model_dir", ""))
    if not model_dir:
        return None, "VoxCPM2 model directory is not configured. Run 00_prepare_runtime.py."
    candidate = Path(model_dir)
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parents[2] / candidate
    if not candidate.exists() or not any(candidate.rglob("*.safetensors")):
        return None, f"Project-local VoxCPM2 model is incomplete: {candidate}. Run 00_prepare_runtime.py without --skip-downloads."
    try:
        from voxcpm import VoxCPM
    except Exception as exc:
        return None, f"VoxCPM2 runtime is not available: {exc}"
    try:
        import torch

        device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
    except Exception:
        device = "cpu"
    try:
        # The local_files_only flag is deliberate: inference must never fetch a
        # model or contact a service after step 00 has completed.
        runtime = VoxCPM.from_pretrained(
            str(candidate),
            local_files_only=True,
            load_denoiser=False,
            optimize=device == "cuda",
            device=device,
        )
        return runtime, ""
    except Exception as exc:
        return None, f"VoxCPM2 could not be initialized from {candidate}: {exc}"


def synthesize_missing_lines_voxcpm(
    temp_root: Path,
    line_specs: list[dict],
    progress_path: Path,
) -> tuple[dict[int, Path], list[str]]:
    runtime_cfg = next(
        (line.get("runtime", {}) for line in line_specs if isinstance(line.get("runtime", {}), dict) and line.get("runtime")),
        {},
    )
    synthesizer, runtime_error = load_voxcpm_runtime(runtime_cfg)
    created: dict[int, Path] = {}
    failures: list[str] = []
    if synthesizer is None:
        if runtime_error:
            failures.append(runtime_error)
        return created, failures
    try:
        import soundfile as sf
    except Exception as exc:
        return created, [f"VoxCPM2 requires soundfile: {exc}"]
    inference_steps = max(1, runtime_int(runtime_cfg, "voxcpm_inference_timesteps", DEFAULT_VOXCPM_INFERENCE_TIMESTEPS))
    cfg_value = max(0.1, runtime_float(runtime_cfg, "voxcpm_cfg_value", DEFAULT_VOXCPM_CFG_VALUE))
    for line in line_specs:
        if line.get("audio_path") is not None or not clean_text(line.get("text", "")):
            continue
        try:
            reference_paths, reference_profile = select_xtts_reference_audio_paths(line, runtime_cfg)
        except Exception as exc:
            failures.append(str(exc).replace("XTTS", "VoxCPM2"))
            continue
        line_index = int(line.get("line_index", 0) or 0)
        target_text = clean_text(line.get("target_output_audio", ""))
        target = Path(target_text) if target_text else temp_root / f"line_{line_index:04d}_voxcpm.wav"
        ensure_parent(str(target))
        started_at = time.time()
        write_voice_progress(
            progress_path,
            {
                "status": "synthesizing_line",
                "engine": "voxcpm2",
                "line_index": line_index,
                "line_number": line_index + 1,
                "speaker_name": line["speaker_name"],
                "target_output_audio": str(target),
                "reference_profile": {key: value for key, value in reference_profile.items() if key != "usable_paths"},
            },
        )
        try:
            reference_path = reference_paths[0]
            result = synthesizer.generate(
                text=clean_text(line.get("text", "")),
                prompt_wav_path=str(reference_path),
                cfg_value=cfg_value,
                inference_timesteps=inference_steps,
                normalize=True,
                denoise=True,
            )
            sample_rate = int(getattr(getattr(synthesizer, "tts_model", None), "sample_rate", 48000) or 48000)
            sf.write(str(target), result, sample_rate)
        except Exception as exc:
            failures.append(f"{line['speaker_name']}: VoxCPM2 synthesis failed for line {line_index}: {exc}")
            write_voice_progress(
                progress_path,
                {"status": "line_failed", "engine": "voxcpm2", "line_index": line_index, "error": str(exc)},
            )
            continue
        if not target.exists() or target.stat().st_size <= 0:
            failures.append(f"{line['speaker_name']}: VoxCPM2 produced no audio for line {line_index}.")
            continue
        created[line_index] = target
        write_voice_progress(
            progress_path,
            {
                "status": "line_complete",
                "engine": "voxcpm2",
                "line_index": line_index,
                "elapsed_seconds": round(time.time() - started_at, 3),
                "target_output_audio": str(target),
            },
        )
    return created, failures


def write_voice_progress(progress_path: Path, payload: dict) -> None:
    try:
        ensure_parent(str(progress_path))
        payload = dict(payload)
        payload["updated_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def synthesize_missing_lines_xtts(
    temp_root: Path,
    line_specs: list[dict],
    progress_path: Path,
) -> tuple[dict[int, Path], list[str]]:
    runtime_cfg = {}
    for line in line_specs:
        if isinstance(line.get("runtime", {}), dict):
            runtime_cfg = line.get("runtime", {})
            if runtime_cfg:
                break
    synthesizer, xtts_error = load_xtts_runtime(runtime_cfg)
    created: dict[int, Path] = {}
    failures: list[str] = []
    if synthesizer is None:
        if xtts_error:
            failures.append(xtts_error)
        return created, failures
    line_timeout_seconds = max(
        0,
        runtime_int(
            runtime_cfg,
            "xtts_line_timeout_seconds",
            DEFAULT_XTTS_LINE_TIMEOUT_SECONDS,
            "SERIES_XTTS_LINE_TIMEOUT_SECONDS",
        ),
    )
    for line in line_specs:
        if line.get("audio_path") is not None or not clean_text(line.get("text", "")):
            continue
        try:
            reference_paths, reference_profile = select_xtts_reference_audio_paths(line, runtime_cfg)
        except Exception as exc:
            failures.append(str(exc))
            continue
        line_index = int(line.get("line_index", 0) or 0)
        target_text = clean_text(line.get("target_output_audio", ""))
        target = Path(target_text) if target_text else temp_root / f"line_{line_index:04d}_xtts.wav"
        ensure_parent(str(target))
        started_at = time.time()
        write_voice_progress(
            progress_path,
            {
                "status": "synthesizing_line",
                "line_index": line_index,
                "line_number": line_index + 1,
                "speaker_name": line["speaker_name"],
                "text": clean_text(line.get("text", "")),
                "language": normalize_language_code(line.get("language", "")) or normalize_language_code(runtime_cfg.get("xtts_language", "")),
                "target_output_audio": str(target),
                "reference_profile": {
                    key: value
                    for key, value in reference_profile.items()
                    if key != "usable_paths"
                },
                "line_timeout_seconds": line_timeout_seconds,
            },
        )
        try:
            synthesize_xtts_line(
                synthesizer,
                clean_text(line.get("text", "")),
                normalize_language_code(line.get("language", "")) or normalize_language_code(runtime_cfg.get("xtts_language", "")),
                reference_paths,
                target,
                line_timeout_seconds,
            )
        except Exception as exc:
            failures.append(f"{line['speaker_name']}: XTTS synthesis failed for line {line_index}: {exc}")
            write_voice_progress(
                progress_path,
                {
                    "status": "line_failed",
                    "line_index": line_index,
                    "line_number": line_index + 1,
                    "speaker_name": line["speaker_name"],
                    "target_output_audio": str(target),
                    "elapsed_seconds": round(time.time() - started_at, 3),
                    "error": str(exc),
                },
            )
            continue
        if not target.exists() or target.stat().st_size <= 0:
            failures.append(f"{line['speaker_name']}: XTTS produced no audio for line {line_index}.")
            continue
        created[line_index] = target
        write_voice_progress(
            progress_path,
            {
                "status": "line_completed",
                "line_index": line_index,
                "line_number": line_index + 1,
                "speaker_name": line["speaker_name"],
                "target_output_audio": str(target),
                "elapsed_seconds": round(time.time() - started_at, 3),
                "bytes": target.stat().st_size,
            },
        )
    return created, failures


def synthesize_missing_lines_pyttsx3(temp_root: Path, line_specs: list[dict]) -> dict[int, Path]:
    try:
        import pyttsx3
    except Exception as exc:
        raise RuntimeError(
            "No reusable per-line audio exists yet, XTTS is unavailable, and pyttsx3 is not available for the local voice fallback."
        ) from exc

    engine = pyttsx3.init()
    created: dict[int, Path] = {}
    try:
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 1.0)
        for line in line_specs:
            if line.get("audio_path") is not None or not clean_text(line.get("text", "")):
                continue
            line_index = int(line.get("line_index", 0) or 0)
            target_text = clean_text(line.get("target_output_audio", ""))
            target = Path(target_text) if target_text else temp_root / f"line_{line_index:04d}_tts.wav"
            ensure_parent(str(target))
            engine.save_to_file(clean_text(line["text"]), str(target))
            created[line_index] = target
        if created:
            engine.runAndWait()
    finally:
        try:
            engine.stop()
        except Exception:
            pass
    missing = [index for index, path in created.items() if not path.exists() or path.stat().st_size <= 0]
    if missing:
        raise RuntimeError(f"Project-local voice fallback synthesis failed for {len(missing)} dialogue lines.")
    return created


def synthesize_missing_lines(temp_root: Path, line_specs: list[dict], progress_path: Path) -> tuple[dict[int, Path], str]:
    runtime_cfg = next(
        (line.get("runtime", {}) for line in line_specs if isinstance(line.get("runtime", {}), dict) and line.get("runtime")),
        {},
    )
    engine = clean_text(runtime_cfg.get("engine", "")).lower()
    if engine in {"voxcpm", "voxcpm2"}:
        created, failures = synthesize_missing_lines_voxcpm(temp_root, line_specs, progress_path)
        backend_name = "voxcpm2_voice_clone"
    else:
        created, failures = {}, [
            f"Unsupported local voice clone engine '{engine or 'unset'}'. "
            "This project accepts only the project-local VoxCPM2 model; run 00_prepare_runtime.py to migrate the configuration."
        ]
    if created:
        return created, backend_name
    force_clone = any(bool(line.get("force_voice_cloning", True)) for line in line_specs)
    if force_clone:
        detail = "; ".join(failures) if failures else "No character reference audio is available for local voice-clone synthesis."
        raise RuntimeError(f"Project-local voice cloning could not synthesize the missing lines. {detail}")
    if not bool(runtime_cfg.get("allow_system_tts_fallback", False)):
        detail = "; ".join(failures) if failures else "No character reference audio is available for VoxCPM2 synthesis."
        raise RuntimeError(f"Project-local voice cloning could not synthesize the missing lines. {detail}")
    return synthesize_missing_lines_pyttsx3(temp_root, line_specs), "pyttsx3"


def generate_silence(ffmpeg: str, target: Path, duration_seconds: float) -> Path | None:
    if duration_seconds <= 0.01:
        return None
    ensure_parent(str(target))
    command = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=24000:cl=mono",
        "-t",
        f"{duration_seconds:.3f}",
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return target if completed.returncode == 0 and target.exists() and target.stat().st_size > 0 else None


def write_voice_diagnostics(output_path: Path, line_specs: list[dict], backend_name: str, audio_files: list[Path]) -> None:
    diagnostics_path = output_path.with_suffix(output_path.suffix + ".diagnostics.json")
    payload = {
        "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "backend": backend_name,
        "scene_dialogue_audio": str(output_path),
        "audio_piece_count": len(audio_files),
        "lines": [
            {
                "line_index": int(line.get("line_index", 0) or 0),
                "speaker_name": clean_text(line.get("speaker_name", "")),
                "language": clean_text(line.get("language", "")),
                "emotion": clean_text(line.get("emotion", "")),
                "pace": clean_text(line.get("pace", "")),
                "energy": float(line.get("energy", 0.0) or 0.0),
                "pause_after_seconds": float(line.get("pause_after_seconds", 0.0) or 0.0),
                "delivery_notes": clean_text(line.get("delivery_notes", "")),
                "reference_candidate_count": len(collect_reference_audio_paths(line)),
                "reference_profile": {
                    key: value
                    for key, value in voice_reference_profile(line, line.get("runtime", {}) if isinstance(line.get("runtime", {}), dict) else {}).items()
                    if key != "usable_paths"
                },
            }
            for line in line_specs
        ],
    }
    diagnostics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    context = load_backend_context()
    scene_package = load_json(str(context.get("scene_package", "") or ""))
    if not scene_package:
        raise RuntimeError("Could not load scene package for the project-local voice backend.")

    output_path = Path(str(context.get("scene_dialogue_audio", "") or ""))
    if not str(output_path):
        raise RuntimeError("The project-local voice backend did not receive a scene dialogue output path.")
    ensure_parent(str(output_path))

    line_specs = collect_line_specs(scene_package)
    if not line_specs:
        raise RuntimeError("The project-local voice backend did not receive any voice-clone line definitions.")

    ffmpeg = find_project_local_ffmpeg()
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_root = Path(tmpdir)
        progress_path = output_path.with_suffix(output_path.suffix + ".progress.json")
        synthesized, backend_name = synthesize_missing_lines(temp_root, line_specs, progress_path)
        audio_files: list[Path] = []
        for line in sorted(line_specs, key=lambda item: int(item.get("line_index", 0) or 0)):
            candidate = line.get("audio_path")
            if isinstance(candidate, Path) and candidate.exists():
                audio_files.append(candidate)
            else:
                line_index = int(line.get("line_index", 0) or 0)
                synthesized_path = synthesized.get(line_index)
                if synthesized_path is not None and synthesized_path.exists():
                    audio_files.append(synthesized_path)
            silence = generate_silence(
                ffmpeg,
                temp_root / f"line_{int(line.get('line_index', 0) or 0):04d}_pause.wav",
                float(line.get("pause_after_seconds", 0.0) or 0.0),
            )
            if silence is not None:
                audio_files.append(silence)
        if not audio_files:
            raise RuntimeError(
                "No reusable or synthesized per-line audio exists yet for this scene. Prepare character voice references first."
            )
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
            concat_path = Path(handle.name)
            for path in audio_files:
                escaped = str(path).replace("'", "''")
                handle.write(f"file '{escaped}'\n")
        try:
            command = [
                ffmpeg,
                "-hide_banner",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path),
                "-ar",
                "24000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ]
            completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        finally:
            concat_path.unlink(missing_ok=True)

    if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Project-local voice backend failed with {backend_name}. {(completed.stdout or '')[-1200:]}")
    write_voice_diagnostics(output_path, line_specs, backend_name, audio_files)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(print_runtime_error(exc))
