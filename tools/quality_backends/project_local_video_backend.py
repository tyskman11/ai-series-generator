#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

from backend_common import copy_if_needed, ensure_parent, existing_path, find_project_local_ffmpeg, load_backend_context, load_json, print_runtime_error


def main() -> int:
    context = load_backend_context()
    scene_package = load_json(str(context.get("scene_package", "") or ""))
    if not scene_package:
        raise RuntimeError("Could not load scene package for the project-local video backend.")

    primary_frame = existing_path(context.get("primary_frame", ""))
    if primary_frame is None:
        raise RuntimeError("The project-local video backend needs an existing primary frame.")

    output_path = Path(str(context.get("scene_video", "") or ""))
    if not str(output_path):
        raise RuntimeError("The project-local video backend did not receive a scene video output path.")
    ensure_parent(str(output_path))

    ffmpeg = find_project_local_ffmpeg()
    duration_seconds = max(1.0, float(scene_package.get("duration_seconds", 8.0) or 8.0))
    command = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-loop",
        "1",
        "-i",
        str(primary_frame),
        "-t",
        f"{duration_seconds:.3f}",
        "-vf",
        "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,zoompan=z='min(zoom+0.0008,1.08)':d=1:s=1280x720",
        "-r",
        "30",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Project-local video backend failed. {(completed.stdout or '')[-1200:]}")

    preview_frame = Path(str(context.get("video_preview_frame", "") or ""))
    if str(preview_frame):
        copy_if_needed(primary_frame, preview_frame)
    poster_frame = Path(str(context.get("video_poster_frame", "") or ""))
    if str(poster_frame):
        copy_if_needed(primary_frame, poster_frame)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(print_runtime_error(exc))
