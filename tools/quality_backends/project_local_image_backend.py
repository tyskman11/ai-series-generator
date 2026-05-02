#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

from backend_common import (
    copy_if_needed,
    ensure_parent,
    existing_path,
    find_project_local_ffmpeg,
    first_existing_path,
    load_backend_context,
    load_json,
    print_runtime_error,
)


def candidate_image_paths(scene_package: dict, context: dict) -> list[Path]:
    preview = scene_package.get("current_preview_assets", {}) if isinstance(scene_package.get("current_preview_assets"), dict) else {}
    image_generation = scene_package.get("image_generation", {}) if isinstance(scene_package.get("image_generation"), dict) else {}
    targets = image_generation.get("target_outputs", {}) if isinstance(image_generation.get("target_outputs"), dict) else {}
    video_generation = scene_package.get("video_generation", {}) if isinstance(scene_package.get("video_generation"), dict) else {}
    local_plan = video_generation.get("local_video_plan", {}) if isinstance(video_generation.get("local_video_plan"), dict) else {}
    beats = local_plan.get("beats", []) if isinstance(local_plan.get("beats"), list) else []

    candidates: list[Path] = []
    for value in [
        context.get("primary_frame"),
        preview.get("preview_frame_path"),
        preview.get("asset_source_path"),
        targets.get("layered_storyboard_frame"),
        context.get("layered_storyboard_frame"),
    ]:
        candidate = existing_path(value)
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)
    for beat in beats:
        if not isinstance(beat, dict):
            continue
        candidate = existing_path(beat.get("reference_image_path", ""))
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def candidate_reference_videos(scene_package: dict) -> list[Path]:
    candidates: list[Path] = []
    for section_name in ("storyboard", "image_generation", "video_generation"):
        section = scene_package.get(section_name, {}) if isinstance(scene_package.get(section_name), dict) else {}
        for slot in section.get("reference_slots", []) if isinstance(section.get("reference_slots", []), list) else []:
            if not isinstance(slot, dict):
                continue
            candidate = existing_path(slot.get("video_file", ""))
            if candidate is not None and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def extract_reference_frame(reference_video: Path, output_path: Path) -> None:
    ensure_parent(str(output_path))
    ffmpeg = find_project_local_ffmpeg()
    command = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-ss",
        "0.8",
        "-i",
        str(reference_video),
        "-frames:v",
        "1",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Could not extract a fallback frame from {reference_video.name}. {(completed.stdout or '')[-1200:]}")


def main() -> int:
    context = load_backend_context()
    scene_package_path = str(context.get("scene_package", "") or "")
    scene_package = load_json(scene_package_path)
    if not scene_package:
        raise RuntimeError(f"Could not load scene package: {scene_package_path}")

    primary_frame = Path(str(context.get("primary_frame", "") or ""))
    if not str(primary_frame):
        raise RuntimeError("The image backend did not receive a primary frame output path.")

    source = first_existing_path(*candidate_image_paths(scene_package, context))
    if source is None:
        reference_video = first_existing_path(*candidate_reference_videos(scene_package))
        if reference_video is None:
            raise RuntimeError("No project-local source image or reference scene video is available for the image backend.")
        extract_reference_frame(reference_video, primary_frame)
        source = primary_frame
    else:
        copy_if_needed(source, primary_frame)

    layered_storyboard_frame = Path(str(context.get("layered_storyboard_frame", "") or ""))
    if str(layered_storyboard_frame):
        copy_if_needed(source, layered_storyboard_frame)

    alternate_dir_text = str(context.get("alternate_frame_dir", "") or "").strip()
    if alternate_dir_text:
        alternate_dir = Path(alternate_dir_text)
        ensure_parent(str(alternate_dir))
        alternate_dir.mkdir(parents=True, exist_ok=True)
        alternate_target = alternate_dir / f"{primary_frame.stem}_alt01{primary_frame.suffix or source.suffix or '.png'}"
        copy_if_needed(source, alternate_target)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(print_runtime_error(exc))
