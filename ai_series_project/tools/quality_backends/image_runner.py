#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from backend_common import choose_scene_package_root, ensure_parent, load_json, print_runtime_error, run_delegated_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delegate finished-episode image generation to an external backend command.")
    parser.add_argument("--scene-package", required=True)
    parser.add_argument("--primary-frame", required=True)
    parser.add_argument("--alternate-frame-dir", default="")
    parser.add_argument("--layered-storyboard-frame", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_package = load_json(args.scene_package)
    ensure_parent(args.primary_frame)
    if args.alternate_frame_dir:
        ensure_parent(args.alternate_frame_dir)
    if args.layered_storyboard_frame:
        ensure_parent(args.layered_storyboard_frame)
    context = {
        "runner_kind": "image",
        "scene_package": args.scene_package,
        "primary_frame": args.primary_frame,
        "alternate_frame_dir": args.alternate_frame_dir,
        "layered_storyboard_frame": args.layered_storyboard_frame,
        "scene_id": scene_package.get("scene_id", ""),
        "episode_id": scene_package.get("episode_id", ""),
        "scene_title": scene_package.get("title", ""),
        "image_generation": scene_package.get("image_generation", {}),
        "reference_slots": scene_package.get("reference_slots", []),
        "continuity": scene_package.get("continuity", {}),
    }
    return run_delegated_backend(
        env_var_name="SERIES_IMAGE_BACKEND_COMMAND",
        context_payload=context,
        context_prefix="series_image_backend",
        cwd=choose_scene_package_root(args.scene_package),
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(print_runtime_error(exc))
