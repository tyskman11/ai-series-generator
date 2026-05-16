#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from backend_common import choose_scene_package_root, ensure_parent, load_json, print_runtime_error, run_delegated_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delegate storyboard keyframe generation to an external backend command.")
    parser.add_argument("--backend-input", required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--preview-frame", default="")
    parser.add_argument("--poster-frame", default="")
    parser.add_argument("--clip", default="")
    parser.add_argument("--alternate-root", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_json(args.backend_input)
    ensure_parent(args.frame)
    for optional_path in (args.preview_frame, args.poster_frame, args.clip):
        if optional_path:
            ensure_parent(optional_path)
    if args.alternate_root:
        Path(args.alternate_root).mkdir(parents=True, exist_ok=True)
    context = {
        "runner_kind": "storyboard",
        "backend_input": args.backend_input,
        "frame": args.frame,
        "preview_frame": args.preview_frame,
        "poster_frame": args.poster_frame,
        "clip": args.clip,
        "alternate_root": args.alternate_root,
        "scene_id": payload.get("scene_id", ""),
        "episode_id": payload.get("episode_id", ""),
        "positive_prompt": payload.get("positive_prompt", ""),
        "negative_prompt": payload.get("negative_prompt", ""),
        "batch_prompt_line": payload.get("batch_prompt_line", ""),
        "control_hints": payload.get("control_hints", {}),
        "camera_plan": payload.get("camera_plan", {}),
        "continuity": payload.get("continuity", {}),
        "backend_candidates": payload.get("backend_candidates", {}),
    }
    return run_delegated_backend(
        env_var_name="SERIES_STORYBOARD_BACKEND_COMMAND",
        context_payload=context,
        context_prefix="series_storyboard_backend",
        cwd=choose_scene_package_root(args.backend_input),
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(print_runtime_error(exc))
