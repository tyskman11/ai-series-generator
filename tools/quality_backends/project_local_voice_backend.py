#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from backend_common import ensure_parent, existing_path, find_project_local_ffmpeg, load_backend_context, load_json, print_runtime_error


def collect_existing_line_audio(scene_package: dict) -> list[Path]:
    voice_clone = scene_package.get("voice_clone", {}) if isinstance(scene_package.get("voice_clone"), dict) else {}
    lines = voice_clone.get("lines", []) if isinstance(voice_clone.get("lines"), list) else []
    audio_files: list[Path] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        candidate = existing_path(line.get("target_output_audio", ""))
        if candidate is None:
            original_reference = line.get("original_voice_reference", {}) if isinstance(line.get("original_voice_reference"), dict) else {}
            candidate = existing_path(original_reference.get("audio_path", ""))
        if candidate is not None:
            audio_files.append(candidate)
    return audio_files


def main() -> int:
    context = load_backend_context()
    scene_package = load_json(str(context.get("scene_package", "") or ""))
    if not scene_package:
        raise RuntimeError("Could not load scene package for the project-local voice backend.")

    output_path = Path(str(context.get("scene_dialogue_audio", "") or ""))
    if not str(output_path):
        raise RuntimeError("The project-local voice backend did not receive a scene dialogue output path.")
    ensure_parent(str(output_path))

    audio_files = collect_existing_line_audio(scene_package)
    if not audio_files:
        raise RuntimeError(
            "No reusable per-line audio exists yet for this scene. Prepare project-local voice assets first or generate line audio before the final render."
        )

    ffmpeg = find_project_local_ffmpeg()
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
