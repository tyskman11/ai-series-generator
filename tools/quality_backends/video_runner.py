#!/usr/bin/env python3
from __future__ import annotations

import argparse

from backend_common import choose_scene_package_root, ensure_parent, load_json, print_runtime_error, run_delegated_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delegate finished-episode scene video generation to an external backend command.")
    parser.add_argument("--scene-package", required=True)
    parser.add_argument("--scene-video", required=True)
    parser.add_argument("--video-preview-frame", default="")
    parser.add_argument("--video-poster-frame", default="")
    parser.add_argument("--primary-frame", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_package = load_json(args.scene_package)
    ensure_parent(args.scene_video)
    if args.video_preview_frame:
        ensure_parent(args.video_preview_frame)
    if args.video_poster_frame:
        ensure_parent(args.video_poster_frame)
    context = {
        "runner_kind": "video",
        "scene_package": args.scene_package,
        "scene_video": args.scene_video,
        "video_preview_frame": args.video_preview_frame,
        "video_poster_frame": args.video_poster_frame,
        "primary_frame": args.primary_frame,
        "scene_id": scene_package.get("scene_id", ""),
        "episode_id": scene_package.get("episode_id", ""),
        "scene_title": scene_package.get("title", ""),
        "video_generation": scene_package.get("video_generation", {}),
        "continuity": scene_package.get("continuity", {}),
        "reference_slots": scene_package.get("reference_slots", []),
    }
    return run_delegated_backend(
        env_var_name="SERIES_VIDEO_BACKEND_COMMAND",
        context_payload=context,
        context_prefix="series_video_backend",
        cwd=choose_scene_package_root(args.scene_package),
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(print_runtime_error(exc))
