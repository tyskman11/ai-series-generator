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

IMAGE_MODEL_ID = "Qwen/Qwen-Image"
IMAGE_MODEL_DIR = "tools/quality_models/image/Qwen__Qwen-Image"
IMAGE_MODEL_REQUIRED_FILES = [
    "model_index.json",
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
IMAGE_IDENTITY_ADAPTER_MODEL_ID = "h94/IP-Adapter"
IMAGE_IDENTITY_ADAPTER_MODEL_DIR = "tools/quality_models/image/h94__IP-Adapter"
IMAGE_IDENTITY_ADAPTER_REQUIRED_FILES = [
    "models/image_encoder/config.json",
    "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
]
ANIME_IMAGE_MODEL_ID = "cagliostrolab/animagine-xl-4.0"
ANIME_IMAGE_MODEL_DIR = "tools/quality_models/image/cagliostrolab__animagine-xl-4.0"
ANIME_IMAGE_MODEL_REQUIRED_FILES = [
    "model_index.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
]
VIDEO_LATEST_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
VIDEO_LATEST_MODEL_DIR = "tools/quality_models/video/Wan-AI__Wan2.1-T2V-1.3B"
VIDEO_LATEST_REQUIRED_FILES = ["model_index.json"]
SCRIPTWRITER_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SCRIPTWRITER_MODEL_DIR = "tools/quality_models/text/Qwen__Qwen2.5-7B-Instruct"
SCRIPTWRITER_MODEL_REQUIRED_FILES = ["config.json", "tokenizer_config.json", "model.safetensors.index.json"]
VOICE_MODEL_ID = "openbmb/VoxCPM2"
VOICE_MODEL_DIR = "tools/quality_models/voice/openbmb__VoxCPM2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configure portable quality-first backend runner templates for original-episode generation."
    )
    parser.add_argument("--print-only", action="store_true", help="Only print the planned config without writing project.json.")
    return parser.parse_args()


def configured_backends() -> dict:
    storyboard_backend_command = '"{python}" "tools/quality_backends/local_diffusion_image_backend.py"'
    image_backend_command = '"{python}" "tools/quality_backends/local_diffusion_image_backend.py"'
    video_backend_command = '"{python}" "tools/quality_backends/local_wan_video_backend.py"'
    voice_backend_command = '"{python}" "tools/quality_backends/local_voxcpm_voice_backend.py"'
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
                "SERIES_IMAGE_CPU_MODEL_ID": IMAGE_IDENTITY_FALLBACK_MODEL_ID,
                "SERIES_IMAGE_CPU_MODEL_DIR": IMAGE_IDENTITY_FALLBACK_MODEL_DIR,
                "SERIES_IMAGE_CPU_MAX_WIDTH": "896",
                "SERIES_IMAGE_CPU_MAX_HEIGHT": "512",
                "SERIES_IMAGE_CPU_MAX_STEPS": "24",
                "SERIES_IMAGE_WIDTH": "1216",
                "SERIES_IMAGE_HEIGHT": "704",
                "SERIES_IMAGE_INFERENCE_STEPS": "50",
                "SERIES_IMAGE_GUIDANCE_SCALE": "4.0",
                "SERIES_IMAGE_QUALITY_PRESET": "qwen_image_source_series",
                "SERIES_IMAGE_REQUIRE_IDENTITY_REFERENCES": "1",
                "SERIES_IMAGE_REQUIRE_IDENTITY_ADAPTER": "1",
                "SERIES_IMAGE_ALLOW_UNVERIFIED_MULTI_IDENTITY": "0",
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
                "SERIES_IMAGE_CPU_MODEL_ID": IMAGE_IDENTITY_FALLBACK_MODEL_ID,
                "SERIES_IMAGE_CPU_MODEL_DIR": IMAGE_IDENTITY_FALLBACK_MODEL_DIR,
                "SERIES_IMAGE_CPU_MAX_WIDTH": "896",
                "SERIES_IMAGE_CPU_MAX_HEIGHT": "512",
                "SERIES_IMAGE_CPU_MAX_STEPS": "24",
                "SERIES_IMAGE_WIDTH": "1216",
                "SERIES_IMAGE_HEIGHT": "704",
                "SERIES_IMAGE_INFERENCE_STEPS": "50",
                "SERIES_IMAGE_GUIDANCE_SCALE": "4.0",
                "SERIES_IMAGE_QUALITY_PRESET": "qwen_image_source_series",
                "SERIES_IMAGE_REQUIRE_IDENTITY_REFERENCES": "1",
                "SERIES_IMAGE_REQUIRE_IDENTITY_ADAPTER": "1",
                "SERIES_IMAGE_ALLOW_UNVERIFIED_MULTI_IDENTITY": "0",
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
                "SERIES_VIDEO_MODEL_ID": VIDEO_LATEST_MODEL_ID,
                "SERIES_VIDEO_MODEL_DIR": VIDEO_LATEST_MODEL_DIR,
                "SERIES_VIDEO_MODEL_FAMILY": "wan",
                "SERIES_VIDEO_COMPATIBILITY_MODE": "local_wan_diffusers",
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
            "environment": {
                "SERIES_VOICE_BACKEND_COMMAND": voice_backend_command,
                "SERIES_VOICE_BACKEND_TIMEOUT_SECONDS": "0",
                "SERIES_VOICE_MODEL_ID": VOICE_MODEL_ID,
                "SERIES_VOICE_MODEL_DIR": VOICE_MODEL_DIR,
                "SERIES_VOICE_LOCAL_FILES_ONLY": "1",
                "SERIES_VOICE_MIN_REFERENCE_SECONDS": "6.0",
                "SERIES_VOICE_MIN_REFERENCE_COUNT": "2",
            },
            "required_commands": [],
            "required_python_modules": ["voxcpm", "soundfile"],
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
        foundation_cfg["identity_adapter_model"] = IMAGE_IDENTITY_ADAPTER_MODEL_ID
        foundation_cfg["video_base_model"] = VIDEO_LATEST_MODEL_ID
        foundation_cfg["video_diffusers_fallback_model"] = ""
        foundation_cfg["voice_base_model"] = VOICE_MODEL_ID
    assets_cfg = config.setdefault("quality_backend_assets", {})
    if not isinstance(assets_cfg, dict):
        assets_cfg = {}
        config["quality_backend_assets"] = assets_cfg
    targets = assets_cfg.setdefault("targets", [])
    if not isinstance(targets, list):
        targets = []
        assets_cfg["targets"] = targets
    obsolete_target_names = {
        "video_diffusers_fallback_model",
        "video_latest_model",
        "anime_video_model",
        "xtts_model_name_record",
    }
    targets[:] = [
        item
        for item in targets
        if not isinstance(item, dict) or str(item.get("name", "") or "").strip() not in obsolete_target_names
    ]
    existing_names = {str(item.get("name", "")).strip() for item in targets if isinstance(item, dict)}
    model_targets = [
        {
            "name": "image_base_model",
            "kind": "huggingface",
            "repo_id": IMAGE_MODEL_ID,
            "target_dir": IMAGE_MODEL_DIR,
            "public_no_login": True,
            "required_files": IMAGE_MODEL_REQUIRED_FILES,
        },
        {
            "name": "image_identity_fallback_model",
            "kind": "huggingface",
            "repo_id": IMAGE_IDENTITY_FALLBACK_MODEL_ID,
            "target_dir": IMAGE_IDENTITY_FALLBACK_MODEL_DIR,
            "public_no_login": True,
            "required_files": IMAGE_IDENTITY_FALLBACK_REQUIRED_FILES,
        },
        {
            "name": "image_identity_adapter",
            "kind": "huggingface",
            "repo_id": IMAGE_IDENTITY_ADAPTER_MODEL_ID,
            "target_dir": IMAGE_IDENTITY_ADAPTER_MODEL_DIR,
            "public_no_login": True,
            "allow_patterns": [
                "models/image_encoder/**",
                "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
            ],
            "required_files": IMAGE_IDENTITY_ADAPTER_REQUIRED_FILES,
        },
        {
            "name": "video_base_model",
            "kind": "huggingface",
            "repo_id": VIDEO_LATEST_MODEL_ID,
            "target_dir": VIDEO_LATEST_MODEL_DIR,
            "public_no_login": True,
            "required_files": VIDEO_LATEST_REQUIRED_FILES,
        },
        {
            "name": "anime_image_model",
            "kind": "huggingface",
            "repo_id": ANIME_IMAGE_MODEL_ID,
            "target_dir": ANIME_IMAGE_MODEL_DIR,
            "public_no_login": True,
            "license_spdx": "openrail++",
            "required_files": ANIME_IMAGE_MODEL_REQUIRED_FILES,
        },
        {
            "name": "local_scriptwriter_model",
            "kind": "huggingface",
            "repo_id": SCRIPTWRITER_MODEL_ID,
            "target_dir": SCRIPTWRITER_MODEL_DIR,
            "public_no_login": True,
            "license_spdx": "Apache-2.0",
            "required_files": SCRIPTWRITER_MODEL_REQUIRED_FILES,
        },
        {
            "name": "voice_base_model",
            "kind": "huggingface",
            "repo_id": VOICE_MODEL_ID,
            "target_dir": VOICE_MODEL_DIR,
            "public_no_login": True,
            "required_patterns": [],
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


def ensure_release_retry_policy(config: dict) -> None:
    """Migrate the old finite retry default without overriding deliberate custom limits."""
    release_cfg = config.setdefault("release_mode", {})
    if not isinstance(release_cfg, dict):
        release_cfg = {}
        config["release_mode"] = release_cfg
    release_cfg["retry_until_pass"] = True
    release_cfg["auto_retry_failed_gate"] = True
    old_default_limit = 12
    configured_limit = release_cfg.get("max_auto_retry_cycles")
    try:
        configured_limit_number = int(configured_limit or 0)
    except (TypeError, ValueError):
        configured_limit_number = old_default_limit
    if configured_limit is None or configured_limit_number == old_default_limit:
        # Zero is intentionally unlimited. Blocked causes still stop immediately in 18.
        release_cfg["max_auto_retry_cycles"] = 0


def ensure_local_generation_config(config: dict) -> None:
    generation_cfg = config.setdefault("generation", {})
    if not isinstance(generation_cfg, dict):
        generation_cfg = {}
        config["generation"] = generation_cfg
    generation_cfg["model_profile"] = str(generation_cfg.get("model_profile", "") or "live_action").strip() or "live_action"
    local_cfg = config.setdefault("local_generation", {})
    if not isinstance(local_cfg, dict):
        local_cfg = {}
        config["local_generation"] = local_cfg
    local_cfg.update(
        {
            "enabled": True,
            "local_models_only": True,
            "allow_runtime_model_downloads": False,
            "require_public_non_gated_models": True,
            "active_profile": str(local_cfg.get("active_profile", "") or "live_action").strip() or "live_action",
        }
    )
    local_cfg["scriptwriter"] = {
        "enabled": True,
        "engine": "transformers",
        "model_id": SCRIPTWRITER_MODEL_ID,
        "model_dir": SCRIPTWRITER_MODEL_DIR,
        "local_files_only": True,
        "max_new_tokens": 768,
        "temperature": 0.75,
    }
    local_cfg["profiles"] = {
        "live_action": {
            "label": "Live action",
            "image_model_id": IMAGE_MODEL_ID,
            "image_model_dir": IMAGE_MODEL_DIR,
            "identity_model_id": IMAGE_IDENTITY_FALLBACK_MODEL_ID,
            "identity_model_dir": IMAGE_IDENTITY_FALLBACK_MODEL_DIR,
            "video_model_id": VIDEO_LATEST_MODEL_ID,
            "video_model_dir": VIDEO_LATEST_MODEL_DIR,
            "video_model_family": "wan",
            "style_prompt": "live-action episodic television, natural production lighting",
            "negative_prompt": "anime, illustration, cartoon, cel shading",
        },
        "anime": {
            "label": "Anime",
            "image_model_id": ANIME_IMAGE_MODEL_ID,
            "image_model_dir": ANIME_IMAGE_MODEL_DIR,
            "identity_model_id": ANIME_IMAGE_MODEL_ID,
            "identity_model_dir": ANIME_IMAGE_MODEL_DIR,
            # Wan is shared with the live-action profile; the style changes in
            # the image/keyframe profile and prompts, so it downloads only once.
            "video_model_id": VIDEO_LATEST_MODEL_ID,
            "video_model_dir": VIDEO_LATEST_MODEL_DIR,
            "video_model_family": "wan",
            "style_prompt": "high-quality serialized anime, clean line art, consistent cel shading, expressive animation",
            "negative_prompt": "photorealistic live action, live-action skin texture, realistic camera footage",
        },
    }
    cloning_cfg = config.setdefault("cloning", {})
    if not isinstance(cloning_cfg, dict):
        cloning_cfg = {}
        config["cloning"] = cloning_cfg
    cloning_cfg.update(
        {
            "voice_clone_engine": "voxcpm2",
            "voice_model_id": VOICE_MODEL_ID,
            "voice_model_dir": VOICE_MODEL_DIR,
            "voice_model_local_files_only": True,
            "voxcpm_inference_timesteps": int(cloning_cfg.get("voxcpm_inference_timesteps", 10) or 10),
            "voxcpm_cfg_value": float(cloning_cfg.get("voxcpm_cfg_value", 2.0) or 2.0),
        }
    )
def main() -> None:
    args = parse_args()
    headline("Configure Quality Backends")
    cfg = load_config()
    updated = deepcopy(cfg)
    updated["external_backends"] = configured_backends()
    ensure_local_generation_config(updated)
    ensure_quality_asset_targets(updated)
    ensure_release_retry_policy(updated)
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
