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

IMAGE_MODEL_ID = "black-forest-labs/FLUX.2-dev"
IMAGE_MODEL_DIR = "tools/quality_models/image/black-forest-labs__FLUX.2-dev"
IMAGE_MODEL_REQUIRED_FILES = [
    "model_index.json",
    "flux2-dev.safetensors",
    "scheduler/scheduler_config.json",
    "text_encoder/model.safetensors.index.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/diffusion_pytorch_model.safetensors",
    "tokenizer/tokenizer_config.json",
]
IMAGE_IDENTITY_FALLBACK_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IMAGE_IDENTITY_FALLBACK_MODEL_DIR = "tools/quality_models/image/stabilityai__stable-diffusion-xl-base-1.0"
IMAGE_IDENTITY_FALLBACK_REQUIRED_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/model.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "tokenizer/tokenizer_config.json",
]
VIDEO_LATEST_MODEL_ID = "Lightricks/LTX-2.3"
VIDEO_LATEST_MODEL_DIR = "tools/quality_models/video/Lightricks__LTX-2.3"
VIDEO_LATEST_REQUIRED_FILES = [
    "ltx-2.3-22b-distilled-1.1.safetensors",
]
VIDEO_DIFFUSERS_MODEL_ID = "Lightricks/LTX-Video-0.9.8-13B-distilled"
VIDEO_DIFFUSERS_MODEL_DIR = "tools/quality_models/video/Lightricks__LTX-Video-0.9.8-13B-distilled"
VIDEO_DIFFUSERS_REQUIRED_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/model.safetensors.index.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/diffusion_pytorch_model.safetensors",
    "tokenizer/tokenizer_config.json",
]


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
            "environment": {
                "SERIES_STORYBOARD_BACKEND_COMMAND": storyboard_backend_command,
                "SERIES_IMAGE_MODEL_ID": IMAGE_MODEL_ID,
                "SERIES_IMAGE_MODEL_DIR": IMAGE_MODEL_DIR,
                "SERIES_IMAGE_IDENTITY_MODEL_ID": IMAGE_IDENTITY_FALLBACK_MODEL_ID,
                "SERIES_IMAGE_IDENTITY_MODEL_DIR": IMAGE_IDENTITY_FALLBACK_MODEL_DIR,
                "SERIES_IMAGE_ALLOW_CPU": "1",
                "SERIES_IMAGE_WIDTH": "1216",
                "SERIES_IMAGE_HEIGHT": "704",
                "SERIES_IMAGE_INFERENCE_STEPS": "28",
                "SERIES_IMAGE_GUIDANCE_SCALE": "3.5",
                "SERIES_IMAGE_QUALITY_PRESET": "flux2_source_series",
                "SERIES_IMAGE_REQUIRE_IDENTITY_REFERENCES": "1",
                "SERIES_IMAGE_REQUIRE_IDENTITY_ADAPTER": "1",
                "PYTHONUNBUFFERED": "1",
            },
            "requires_gpu": False,
            "prefer_gpu": True,
            "allow_cpu_execution": True,
            "required_commands": [],
            "required_python_modules": ["diffusers"],
            "required_environment_variables": ["SERIES_STORYBOARD_BACKEND_COMMAND"],
            "shell": False,
            "timeout_seconds": 0,
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
            "environment": {
                "SERIES_IMAGE_BACKEND_COMMAND": image_backend_command,
                "SERIES_IMAGE_MODEL_ID": IMAGE_MODEL_ID,
                "SERIES_IMAGE_MODEL_DIR": IMAGE_MODEL_DIR,
                "SERIES_IMAGE_IDENTITY_MODEL_ID": IMAGE_IDENTITY_FALLBACK_MODEL_ID,
                "SERIES_IMAGE_IDENTITY_MODEL_DIR": IMAGE_IDENTITY_FALLBACK_MODEL_DIR,
                "SERIES_IMAGE_ALLOW_CPU": "1",
                "SERIES_IMAGE_WIDTH": "1216",
                "SERIES_IMAGE_HEIGHT": "704",
                "SERIES_IMAGE_INFERENCE_STEPS": "28",
                "SERIES_IMAGE_GUIDANCE_SCALE": "3.5",
                "SERIES_IMAGE_QUALITY_PRESET": "flux2_source_series",
                "SERIES_IMAGE_REQUIRE_IDENTITY_REFERENCES": "1",
                "SERIES_IMAGE_REQUIRE_IDENTITY_ADAPTER": "1",
                "SERIES_IMAGE_RESUME_SHOTS": "1",
            },
            "requires_gpu": False,
            "prefer_gpu": True,
            "allow_cpu_execution": True,
            "required_commands": [],
            "required_python_modules": ["diffusers"],
            "required_environment_variables": ["SERIES_IMAGE_BACKEND_COMMAND"],
            "shell": False,
            "timeout_seconds": 0,
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
            "environment": {
                "SERIES_VIDEO_BACKEND_COMMAND": video_backend_command,
                "SERIES_VIDEO_LATEST_MODEL_ID": VIDEO_LATEST_MODEL_ID,
                "SERIES_VIDEO_LATEST_MODEL_DIR": VIDEO_LATEST_MODEL_DIR,
                "SERIES_VIDEO_MODEL_ID": VIDEO_DIFFUSERS_MODEL_ID,
                "SERIES_VIDEO_MODEL_DIR": VIDEO_DIFFUSERS_MODEL_DIR,
                "SERIES_VIDEO_COMPATIBILITY_MODE": "ltx_diffusers_fallback_until_ltx2_runner",
                "SERIES_VIDEO_WIDTH": "1216",
                "SERIES_VIDEO_HEIGHT": "704",
                "SERIES_VIDEO_FPS": "30",
                "SERIES_VIDEO_INFERENCE_STEPS": "30",
                "SERIES_VIDEO_GUIDANCE_SCALE": "3.0",
                "SERIES_VIDEO_QUALITY_PRESET": "source_series_high",
                "SERIES_VIDEO_RESUME_SHOTS": "1",
            },
            "required_commands": [],
            "required_python_modules": ["diffusers"],
            "required_environment_variables": ["SERIES_VIDEO_BACKEND_COMMAND"],
            "shell": False,
            "timeout_seconds": 0,
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
            "timeout_seconds": 0,
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
            "timeout_seconds": 0,
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
            "timeout_seconds": 0,
            "skip_if_outputs_exist": True,
            "success_mode": "any",
            "success_outputs": ["{final_master_episode}"],
        },
    }


def ensure_quality_asset_targets(config: dict) -> None:
    foundation_cfg = config.setdefault("foundation_training", {})
    if isinstance(foundation_cfg, dict):
        foundation_cfg["image_base_model"] = IMAGE_MODEL_ID
        foundation_cfg["image_identity_fallback_model"] = IMAGE_IDENTITY_FALLBACK_MODEL_ID
        foundation_cfg["video_base_model"] = VIDEO_LATEST_MODEL_ID
        foundation_cfg["video_diffusers_fallback_model"] = VIDEO_DIFFUSERS_MODEL_ID
    assets_cfg = config.setdefault("quality_backend_assets", {})
    if not isinstance(assets_cfg, dict):
        assets_cfg = {}
        config["quality_backend_assets"] = assets_cfg
    targets = assets_cfg.setdefault("targets", [])
    if not isinstance(targets, list):
        targets = []
        assets_cfg["targets"] = targets
    existing_names = {str(item.get("name", "")).strip() for item in targets if isinstance(item, dict)}
    model_targets = [
        {
            "name": "image_base_model",
            "kind": "huggingface",
            "repo_id": IMAGE_MODEL_ID,
            "target_dir": IMAGE_MODEL_DIR,
            "required_files": IMAGE_MODEL_REQUIRED_FILES,
        },
        {
            "name": "image_identity_fallback_model",
            "kind": "huggingface",
            "repo_id": IMAGE_IDENTITY_FALLBACK_MODEL_ID,
            "target_dir": IMAGE_IDENTITY_FALLBACK_MODEL_DIR,
            "required_files": IMAGE_IDENTITY_FALLBACK_REQUIRED_FILES,
        },
        {
            "name": "video_base_model",
            "kind": "huggingface",
            "repo_id": VIDEO_LATEST_MODEL_ID,
            "target_dir": VIDEO_LATEST_MODEL_DIR,
            "required_files": VIDEO_LATEST_REQUIRED_FILES,
        },
        {
            "name": "video_diffusers_fallback_model",
            "kind": "huggingface",
            "repo_id": VIDEO_DIFFUSERS_MODEL_ID,
            "target_dir": VIDEO_DIFFUSERS_MODEL_DIR,
            "required_files": VIDEO_DIFFUSERS_REQUIRED_FILES,
        },
    ]
    for model_target in model_targets:
        for item in targets:
            if isinstance(item, dict) and str(item.get("name", "")).strip() == model_target["name"]:
                item.pop("required_patterns", None)
                item.update(model_target)
                break
        else:
            targets.append(model_target)
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
