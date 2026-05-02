#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from backend_common import ensure_parent, existing_path, find_project_local_ffmpeg, load_backend_context, load_json, print_runtime_error


def collect_line_specs(scene_package: dict) -> list[dict]:
    voice_clone = scene_package.get("voice_clone", {}) if isinstance(scene_package.get("voice_clone"), dict) else {}
    lines = voice_clone.get("lines", []) if isinstance(voice_clone.get("lines"), list) else []
    prepared: list[dict] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        original_reference = line.get("original_voice_reference", {}) if isinstance(line.get("original_voice_reference"), dict) else {}
        candidate = existing_path(line.get("target_output_audio", "")) or existing_path(original_reference.get("audio_path", ""))
        prepared.append(
            {
                "line_index": int(line.get("line_index", 0) or 0),
                "text": str(line.get("text", "") or "").strip(),
                "audio_path": candidate,
            }
        )
    return prepared


def synthesize_missing_lines(temp_root: Path, line_specs: list[dict]) -> dict[int, Path]:
    try:
        import pyttsx3
    except Exception as exc:
        raise RuntimeError(
            "No reusable per-line audio exists yet, and pyttsx3 is not available for project-local voice fallback synthesis."
        ) from exc

    engine = pyttsx3.init()
    created: dict[int, Path] = {}
    try:
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 1.0)
        for line in line_specs:
            if line.get("audio_path") is not None or not str(line.get("text", "")).strip():
                continue
            line_index = int(line.get("line_index", 0) or 0)
            target = temp_root / f"line_{line_index:04d}_tts.wav"
            engine.save_to_file(str(line["text"]), str(target))
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
        synthesized = synthesize_missing_lines(temp_root, line_specs)
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
                "No reusable or synthesized per-line audio exists yet for this scene. Prepare project-local voice assets first or enable a local TTS fallback."
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
        raise RuntimeError(f"Project-local voice backend failed. {(completed.stdout or '')[-1200:]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(print_runtime_error(exc))
