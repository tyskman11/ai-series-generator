#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy

from pipeline_common import headline, info, load_config, ok, quality_first_requirements_report, warn, write_json, CONFIG_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configure portable quality-first backend runner templates for original-episode generation."
    )
    parser.add_argument("--print-only", action="store_true", help="Only print the planned config without writing project.json.")
    return parser.parse_args()


def configured_backends() -> dict:
    return {
        "storyboard_scene_runner": {
            "enabled": True,
            "command_template": [
                "python",
                "54_run_storyboard_backend.py",
                "--episode-id",
                "{episode_id}",
                "--scene-ids",
                "{scene_id}",
            ],
            "working_directory": ".",
            "environment": {},
            "required_commands": ["python"],
            "required_environment_variables": [],
            "shell": False,
            "timeout_seconds": 1800,
            "skip_if_outputs_exist": True,
            "success_mode": "any",
            "success_outputs": ["{frame_path}"],
        },
        "finished_episode_image_runner": {
            "enabled": True,
            "command_template": [
                "python",
                "tools/quality_backends/image_runner.py",
                "--scene-package",
                "{scene_package_path}",
                "--primary-frame",
                "{primary_frame}",
                "--alternate-frame-dir",
                "{alternate_frame_dir}",
                "--layered-storyboard-frame",
                "{layered_storyboard_frame}",
            ],
            "working_directory": ".",
            "environment": {},
            "required_commands": ["python"],
            "required_environment_variables": ["SERIES_IMAGE_BACKEND_COMMAND"],
            "shell": False,
            "timeout_seconds": 3600,
            "skip_if_outputs_exist": True,
            "success_mode": "any",
            "success_outputs": ["{primary_frame}"],
        },
        "finished_episode_video_runner": {
            "enabled": True,
            "command_template": [
                "python",
                "tools/quality_backends/video_runner.py",
                "--scene-package",
                "{scene_package_path}",
                "--scene-video",
                "{scene_video}",
                "--video-preview-frame",
                "{video_preview_frame}",
                "--video-poster-frame",
                "{video_poster_frame}",
                "--primary-frame",
                "{primary_frame}",
            ],
            "working_directory": ".",
            "environment": {},
            "required_commands": ["python"],
            "required_environment_variables": ["SERIES_VIDEO_BACKEND_COMMAND"],
            "shell": False,
            "timeout_seconds": 7200,
            "skip_if_outputs_exist": True,
            "success_mode": "any",
            "success_outputs": ["{scene_video}"],
        },
        "finished_episode_voice_runner": {
            "enabled": True,
            "command_template": [
                "python",
                "tools/quality_backends/voice_runner.py",
                "--scene-package",
                "{scene_package_path}",
                "--scene-dialogue-audio",
                "{scene_dialogue_audio}",
            ],
            "working_directory": ".",
            "environment": {},
            "required_commands": ["python"],
            "required_environment_variables": ["SERIES_VOICE_BACKEND_COMMAND"],
            "shell": False,
            "timeout_seconds": 3600,
            "skip_if_outputs_exist": True,
            "success_mode": "any",
            "success_outputs": ["{scene_dialogue_audio}"],
        },
        "finished_episode_lipsync_runner": {
            "enabled": True,
            "command_template": [
                "python",
                "tools/quality_backends/lipsync_runner.py",
                "--scene-package",
                "{scene_package_path}",
                "--scene-video",
                "{scene_video}",
                "--scene-dialogue-audio",
                "{scene_dialogue_audio}",
                "--lipsync-video",
                "{lipsync_video}",
                "--lipsync-poster-frame",
                "{lipsync_poster_frame}",
            ],
            "working_directory": ".",
            "environment": {},
            "required_commands": ["python"],
            "required_environment_variables": ["SERIES_LIPSYNC_BACKEND_COMMAND"],
            "shell": False,
            "timeout_seconds": 7200,
            "skip_if_outputs_exist": True,
            "success_mode": "any",
            "success_outputs": ["{lipsync_video}"],
        },
        "finished_episode_master_runner": {
            "enabled": True,
            "command_template": [
                "python",
                "tools/quality_backends/master_runner.py",
                "--package-path",
                "{package_path}",
                "--final-master-episode",
                "{final_master_episode}",
            ],
            "working_directory": ".",
            "environment": {},
            "required_commands": ["python", "ffmpeg"],
            "required_environment_variables": [],
            "shell": False,
            "timeout_seconds": 3600,
            "skip_if_outputs_exist": True,
            "success_mode": "any",
            "success_outputs": ["{final_master_episode}"],
        },
    }


def main() -> None:
    args = parse_args()
    headline("Configure Quality Backends")
    cfg = load_config()
    updated = deepcopy(cfg)
    updated["external_backends"] = configured_backends()
    report = quality_first_requirements_report(updated)

    if args.print_only:
        info(f"Config target: {CONFIG_PATH}")
    else:
        write_json(CONFIG_PATH, updated)
        ok(f"Updated {CONFIG_PATH}")

    if report.get("ready", False):
        ok("Quality-first runner prerequisites are fully satisfied.")
    else:
        warn("Quality-first runner setup is configured, but prerequisites are still missing:")
        for entry in report.get("missing", []):
            warn(f"- {entry}")
    for entry in report.get("warnings", []):
        info(f"Hint: {entry}")


if __name__ == "__main__":
    main()
