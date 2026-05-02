#!/usr/bin/env python3
from __future__ import annotations

import argparse

from backend_common import choose_scene_package_root, ensure_parent, load_json, print_runtime_error, run_delegated_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delegate finished-episode lip-sync generation to an external backend command.")
    parser.add_argument("--scene-package", required=True)
    parser.add_argument("--scene-video", required=True)
    parser.add_argument("--scene-dialogue-audio", required=True)
    parser.add_argument("--lipsync-video", required=True)
    parser.add_argument("--lipsync-poster-frame", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_package = load_json(args.scene_package)
    ensure_parent(args.lipsync_video)
    if args.lipsync_poster_frame:
        ensure_parent(args.lipsync_poster_frame)
    context = {
        "runner_kind": "lipsync",
        "scene_package": args.scene_package,
        "scene_video": args.scene_video,
        "scene_dialogue_audio": args.scene_dialogue_audio,
        "lipsync_video": args.lipsync_video,
        "lipsync_poster_frame": args.lipsync_poster_frame,
        "scene_id": scene_package.get("scene_id", ""),
        "episode_id": scene_package.get("episode_id", ""),
        "scene_title": scene_package.get("title", ""),
        "lip_sync": scene_package.get("lip_sync", {}),
    }
    return run_delegated_backend(
        env_var_name="SERIES_LIPSYNC_BACKEND_COMMAND",
        context_payload=context,
        context_prefix="series_lipsync_backend",
        cwd=choose_scene_package_root(args.scene_package),
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(print_runtime_error(exc))
