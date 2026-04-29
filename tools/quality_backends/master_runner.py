#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from backend_common import ensure_parent, load_json, print_runtime_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the final master episode from scene master clips.")
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--final-master-episode", required=True)
    return parser.parse_args()


def render_output_ready(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def collect_scene_master_clips(package_payload: dict) -> list[Path]:
    clips: list[Path] = []
    scenes = package_payload.get("scenes", []) if isinstance(package_payload.get("scenes", []), list) else []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        outputs = scene.get("current_generated_outputs", {}) if isinstance(scene.get("current_generated_outputs", {}), dict) else {}
        for key in ("scene_master_clip", "generated_lipsync_video", "video_source_path"):
            value = str(outputs.get(key, "") or "").strip()
            if not value:
                continue
            candidate = Path(value)
            if render_output_ready(candidate):
                clips.append(candidate)
                break
    return clips


def write_concat_file(clips: list[Path], concat_path: Path) -> None:
    lines = []
    for path in clips:
        escaped = str(path).replace("'", "''")
        lines.append(f"file '{escaped}'")
    concat_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    args = parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for the built-in finished-episode master runner.")

    package_payload = load_json(args.package_path)
    if not isinstance(package_payload, dict) or not package_payload:
        raise RuntimeError(f"Could not load production package: {args.package_path}")

    scene_clips = collect_scene_master_clips(package_payload)
    if not scene_clips:
        raise RuntimeError("No scene master clips, lip-sync videos, or generated scene videos are available for mastering.")

    output_path = Path(args.final_master_episode)
    ensure_parent(str(output_path))

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        concat_path = Path(handle.name)
    try:
        write_concat_file(scene_clips, concat_path)
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
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if completed.returncode != 0 or not render_output_ready(output_path):
            log_text = completed.stdout or ""
            raise RuntimeError(f"FFmpeg mastering failed. {log_text[-800:]}")
    finally:
        concat_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(print_runtime_error(exc))
