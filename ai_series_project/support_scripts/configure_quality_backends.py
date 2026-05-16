#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from support_scripts.pipeline_common import (
    CONFIG_PATH,
    CONFIG_TEMPLATE_PATH,
    headline,
    info,
    load_config,
    ok,
    quality_first_requirements_report,
    warn,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configure portable quality-first backend runner templates for original-episode generation."
    )
    parser.add_argument("--print-only", action="store_true", help="Only print the planned config without writing project.json.")
    return parser.parse_args()


def configured_backends() -> dict:
    storyboard_backend_command = '"{python}" "tools/quality_backends/local_diffusion_image_backend.py"'
    image_backend_command = '"{python}" "tools/quality_backends/local_diffusion_image_backend.py"'
    video_backend_command = '"{python}" "tools/quality_backends/local_ltx_video_backend.py"'
    voice_backend_command = '"{python}" "tools/quality_backends/local_xtts_voice_backend.py"'
    lipsync_backend_command = '"{python}" "tools/quality_backends/local_wav2lip_backend.py"'
    return {
        "storyboard_scene_runner": {
            "enabled": True,
            "command_template": [
                "{python}",
                "tools/quality_backends/storyboard_runner.py",
                "--backend-input",
                "{backend_input_path}",
                "--frame",
                "{frame_path}",
                "--preview-frame",
                "{preview_path}",
                "--poster-frame",
                "{poster_path}",
                "--clip",
                "{clip_path}",
                "--alternate-root",
                "{alternate_root}",
            ],
            "working_directory": ".",
            "environment": {"SERIES_STORYBOARD_BACKEND_COMMAND": storyboard_backend_command},
            "required_commands": [],
            "required_python_modules": ["diffusers"],
            "required_environment_variables": ["SERIES_STORYBOARD_BACKEND_COMMAND"],
            "shell": False,
            "timeout_seconds": 1800,
            "skip_if_outputs_exist": True,
            "success_mode": "any",
            "success_outputs": ["{frame_path}"],
        },
        "finished_episode_image_runner": {
            "enabled": True,
            "command_template": [
                "{python}",
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
            "environment": {"SERIES_IMAGE_BACKEND_COMMAND": image_backend_command},
            "required_commands": [],
            "required_python_modules": ["diffusers"],
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
                "{python}",
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
            "environment": {"SERIES_VIDEO_BACKEND_COMMAND": video_backend_command},
            "required_commands": [],
            "required_python_modules": ["diffusers"],
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
                "{python}",
                "tools/quality_backends/voice_runner.py",
                "--scene-package",
                "{scene_package_path}",
                "--scene-dialogue-audio",
                "{scene_dialogue_audio}",
            ],
            "working_directory": ".",
            "environment": {"SERIES_VOICE_BACKEND_COMMAND": voice_backend_command},
            "required_commands": [],
            "required_python_modules": ["TTS"],
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
                "{python}",
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
            "environment": {"SERIES_LIPSYNC_BACKEND_COMMAND": lipsync_backend_command},
            "required_commands": [],
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
                "{python}",
                "tools/quality_backends/master_runner.py",
                "--package-path",
                "{package_path}",
                "--final-master-episode",
                "{final_master_episode}",
            ],
            "working_directory": ".",
            "environment": {},
            "required_commands": [],
            "required_environment_variables": [],
            "shell": False,
            "timeout_seconds": 3600,
            "skip_if_outputs_exist": True,
            "success_mode": "any",
            "success_outputs": ["{final_master_episode}"],
        },
    }


def ensure_quality_asset_targets(config: dict) -> None:
    assets_cfg = config.setdefault("quality_backend_assets", {})
    if not isinstance(assets_cfg, dict):
        assets_cfg = {}
        config["quality_backend_assets"] = assets_cfg
    targets = assets_cfg.setdefault("targets", [])
    if not isinstance(targets, list):
        targets = []
        assets_cfg["targets"] = targets
    existing_names = {str(item.get("name", "")).strip() for item in targets if isinstance(item, dict)}
    if "wav2lip" not in existing_names:
        targets.insert(
            1 if targets else 0,
            {
                "name": "wav2lip",
                "kind": "git",
                "repo_url": "https://github.com/Rudrabha/Wav2Lip.git",
                "ref": "master",
                "target_dir": "tools/quality_backends/wav2lip",
                "required_files": ["inference.py"],
            },
        )


def main() -> None:
    args = parse_args()
    headline("Configure Quality Backends")
    cfg = load_config()
    updated = deepcopy(cfg)
    updated["external_backends"] = configured_backends()
    ensure_quality_asset_targets(updated)
    report = quality_first_requirements_report(updated)

    if args.print_only:
        info(f"Working config target: {CONFIG_PATH}")
        info(f"Template base: {CONFIG_TEMPLATE_PATH}")
    else:
        write_json(CONFIG_PATH, updated)
        ok(f"Updated working config {CONFIG_PATH}")

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
