#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

from backend_common import copy_if_needed, ensure_parent, existing_path, find_project_local_ffmpeg, first_existing_path, load_backend_context, print_runtime_error


def main() -> int:
    context = load_backend_context()
    scene_video = existing_path(context.get("scene_video", ""))
    scene_audio = existing_path(context.get("scene_dialogue_audio", ""))
    if scene_video is None:
        raise RuntimeError("The project-local lip-sync backend needs an existing scene video.")
    if scene_audio is None:
        raise RuntimeError("The project-local lip-sync backend needs an existing dialogue audio track.")

    output_path = Path(str(context.get("lipsync_video", "") or ""))
    if not str(output_path):
        raise RuntimeError("The project-local lip-sync backend did not receive an output path.")
    ensure_parent(str(output_path))

    ffmpeg = find_project_local_ffmpeg()
    command = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-i",
        str(scene_video),
        "-i",
        str(scene_audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Project-local lip-sync backend failed. {(completed.stdout or '')[-1200:]}")

    poster_target = Path(str(context.get("lipsync_poster_frame", "") or ""))
    if str(poster_target):
        poster_source = first_existing_path(context.get("primary_frame", ""), context.get("video_poster_frame", ""))
        if poster_source is not None:
            copy_if_needed(poster_source, poster_target)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(print_runtime_error(exc))
