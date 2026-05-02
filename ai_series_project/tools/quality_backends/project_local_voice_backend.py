#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import tempfile
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
    ordered_values: list[object] = [
        original_reference.get("audio_path", ""),
        voice_profile.get("reference_audio", ""),
        *(
            clean_text(item.get("audio_path", ""))
            for item in (line.get("reference_segments", []) if isinstance(line.get("reference_segments", []), list) else [])
            if isinstance(item, dict)
        ),
        *(line.get("reference_audio_candidates", []) if isinstance(line.get("reference_audio_candidates", []), list) else []),
        voice_model.get("reference_audio", ""),
        *(voice_model.get("sample_paths", []) if isinstance(voice_model.get("sample_paths", []), list) else []),
        *(
            clean_text(item.get("audio_path", ""))
            for item in (voice_model.get("reference_segments", []) if isinstance(voice_model.get("reference_segments", []), list) else [])
            if isinstance(item, dict)
        ),
    ]
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


def collect_line_specs(scene_package: dict) -> list[dict]:
    voice_clone = scene_package.get("voice_clone", {}) if isinstance(scene_package.get("voice_clone"), dict) else {}
    lines = voice_clone.get("lines", []) if isinstance(voice_clone.get("lines"), list) else []
    prepared: list[dict] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        original_reference = line.get("original_voice_reference", {}) if isinstance(line.get("original_voice_reference"), dict) else {}
        candidate = existing_file_path(line.get("target_output_audio", "")) or existing_file_path(original_reference.get("audio_path", ""))
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
                "runtime": line.get("runtime", {}) if isinstance(line.get("runtime", {}), dict) else {},
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


def synthesize_xtts_line(synthesizer, text: str, language: str, reference_paths: list[Path], output_path: Path) -> None:
    kwargs = {
        "text": text,
        "file_path": str(output_path),
    }
    if reference_paths:
        kwargs["speaker_wav"] = [str(path) for path in reference_paths[:4]]
    if language:
        kwargs["language"] = language
    synthesizer.tts_to_file(**kwargs)


def synthesize_missing_lines_xtts(temp_root: Path, line_specs: list[dict]) -> tuple[dict[int, Path], list[str]]:
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
    for line in line_specs:
        if line.get("audio_path") is not None or not clean_text(line.get("text", "")):
            continue
        reference_paths = collect_reference_audio_paths(line)
        if not reference_paths:
            failures.append(f"{line['speaker_name']}: no character reference audio was found.")
            continue
        line_index = int(line.get("line_index", 0) or 0)
        target = temp_root / f"line_{line_index:04d}_xtts.wav"
        try:
            synthesize_xtts_line(
                synthesizer,
                clean_text(line.get("text", "")),
                normalize_language_code(line.get("language", "")) or normalize_language_code(runtime_cfg.get("xtts_language", "")),
                reference_paths,
                target,
            )
        except Exception as exc:
            failures.append(f"{line['speaker_name']}: XTTS synthesis failed for line {line_index}: {exc}")
            continue
        if not target.exists() or target.stat().st_size <= 0:
            failures.append(f"{line['speaker_name']}: XTTS produced no audio for line {line_index}.")
            continue
        created[line_index] = target
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
            target = temp_root / f"line_{line_index:04d}_tts.wav"
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


def synthesize_missing_lines(temp_root: Path, line_specs: list[dict]) -> tuple[dict[int, Path], str]:
    created, failures = synthesize_missing_lines_xtts(temp_root, line_specs)
    if created:
        return created, "xtts"
    runtime_cfg = {}
    for line in line_specs:
        if isinstance(line.get("runtime", {}), dict):
            runtime_cfg = line.get("runtime", {})
            if runtime_cfg:
                break
    if not bool(runtime_cfg.get("allow_system_tts_fallback", False)):
        detail = "; ".join(failures) if failures else "No character reference audio is available for XTTS synthesis."
        raise RuntimeError(f"Project-local voice cloning could not synthesize the missing lines. {detail}")
    return synthesize_missing_lines_pyttsx3(temp_root, line_specs), "pyttsx3"


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
        synthesized, backend_name = synthesize_missing_lines(temp_root, line_specs)
        audio_files: list[Path] = []
        for line in sorted(line_specs, key=lambda item: int(item.get("line_index", 0) or 0)):
            candidate = line.get("audio_path")
            if isinstance(candidate, Path) and candidate.exists():
                audio_files.append(candidate)
                continue
            line_index = int(line.get("line_index", 0) or 0)
            synthesized_path = synthesized.get(line_index)
            if synthesized_path is not None and synthesized_path.exists():
                audio_files.append(synthesized_path)
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
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(print_runtime_error(exc))
