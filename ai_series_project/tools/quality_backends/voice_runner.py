#!/usr/bin/env python3
from __future__ import annotations

import argparse

from backend_common import choose_scene_package_root, ensure_parent, load_json, print_runtime_error, run_delegated_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delegate finished-episode dialogue audio generation to an external backend command.")
    parser.add_argument("--scene-package", required=True)
    parser.add_argument("--scene-dialogue-audio", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_package = load_json(args.scene_package)
    ensure_parent(args.scene_dialogue_audio)
    context = {
        "runner_kind": "voice",
        "scene_package": args.scene_package,
        "scene_dialogue_audio": args.scene_dialogue_audio,
        "scene_id": scene_package.get("scene_id", ""),
        "episode_id": scene_package.get("episode_id", ""),
        "scene_title": scene_package.get("title", ""),
        "voice_clone": scene_package.get("voice_clone", {}),
        "continuity": scene_package.get("continuity", {}),
    }
    return run_delegated_backend(
        env_var_name="SERIES_VOICE_BACKEND_COMMAND",
        context_payload=context,
        context_prefix="series_voice_backend",
        cwd=choose_scene_package_root(args.scene_package),
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(print_runtime_error(exc))
