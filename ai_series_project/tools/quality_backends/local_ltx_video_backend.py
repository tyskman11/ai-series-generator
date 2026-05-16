#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image

from backend_common import (
    PROJECT_DIR,
    copy_if_needed,
    ensure_parent,
    existing_path,
    find_project_local_ffmpeg,
    load_backend_context,
    load_json,
    print_runtime_error,
)


DEFAULT_VIDEO_MODEL_DIR = PROJECT_DIR / "tools" / "quality_models" / "video" / "Lightricks__LTX-Video"


def clean_text(value: object) -> str:
    return str(value or "").strip()


def resolve_model_dir() -> Path:
    configured = clean_text(os.environ.get("SERIES_VIDEO_MODEL_DIR", ""))
    candidate = Path(configured) if configured else DEFAULT_VIDEO_MODEL_DIR
    if not candidate.is_absolute():
        candidate = PROJECT_DIR / candidate
    candidate = candidate.resolve(strict=False)
    if not candidate.exists() or not any(candidate.glob("*.safetensors")):
        raise RuntimeError(
            "Local LTX video model is not ready. Run 00_prepare_runtime.py without --skip-downloads "
            f"or set SERIES_VIDEO_MODEL_DIR. Expected safetensors files in {candidate}."
        )
    return candidate


def prompt_from_package(context: dict[str, Any], scene_package: dict[str, Any]) -> str:
    video_generation = scene_package.get("video_generation", {}) if isinstance(scene_package.get("video_generation"), dict) else {}
    image_generation = scene_package.get("image_generation", {}) if isinstance(scene_package.get("image_generation"), dict) else {}
    parts = [
        clean_text(video_generation.get("positive_prompt", "")),
        clean_text(image_generation.get("positive_prompt", "")),
        clean_text(context.get("scene_title", "")),
        clean_text(scene_package.get("summary", "")),
        "faithful original TV episode shot, stable character identity, natural motion, cinematic continuity",
    ]
    return ", ".join(part for part in parts if part)


def deterministic_seed(context: dict[str, Any]) -> int:
    raw = "|".join(clean_text(context.get(key, "")) for key in ("episode_id", "scene_id", "scene_title", "runner_kind"))
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16)


def target_size() -> tuple[int, int]:
    width = int(float(os.environ.get("SERIES_VIDEO_WIDTH", "704") or "704"))
    height = int(float(os.environ.get("SERIES_VIDEO_HEIGHT", "480") or "480"))
    width = max(256, (width // 32) * 32)
    height = max(256, (height // 32) * 32)
    return width, height


def frame_count(scene_package: dict[str, Any], fps: int) -> int:
    duration = float(scene_package.get("duration_seconds", 5.0) or 5.0)
    requested = int(duration * fps)
    configured = int(float(os.environ.get("SERIES_VIDEO_NUM_FRAMES", "0") or "0"))
    count = configured if configured > 0 else max(49, min(145, requested))
    if count % 8 != 1:
        count = ((count // 8) * 8) + 1
    return max(49, count)


def export_preview_frame(video_path: Path, output_path: Path) -> None:
    if not str(output_path):
        return
    ensure_parent(str(output_path))
    command = [
        find_project_local_ffmpeg(),
        "-hide_banner",
        "-y",
        "-ss",
        "0.3",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Could not extract video preview frame. {(completed.stdout or '')[-1200:]}")


def generate_ltx_video(context: dict[str, Any], scene_package: dict[str, Any], output_path: Path) -> None:
    try:
        import torch
        import diffusers
        from diffusers.utils import export_to_video
    except Exception as exc:
        raise RuntimeError(
            "The real local video backend requires diffusers and torch. Run 00_prepare_runtime.py first."
        ) from exc

    pipeline_cls = getattr(diffusers, "LTXImageToVideoPipeline", None) or getattr(diffusers, "LTXPipeline", None)
    if pipeline_cls is None:
        raise RuntimeError(
            "The installed diffusers version does not expose LTXPipeline/LTXImageToVideoPipeline. "
            "Run 00_prepare_runtime.py to upgrade the project runtime."
        )

    model_dir = resolve_model_dir()
    cuda_ready = bool(torch.cuda.is_available())
    dtype = torch.float16 if cuda_ready else torch.float32
    device = "cuda" if cuda_ready else "cpu"
    width, height = target_size()
    fps = int(float(os.environ.get("SERIES_VIDEO_FPS", "24") or "24"))
    steps = int(float(os.environ.get("SERIES_VIDEO_INFERENCE_STEPS", "30") or "30"))
    guidance = float(os.environ.get("SERIES_VIDEO_GUIDANCE_SCALE", "3.0") or "3.0")

    pipeline = pipeline_cls.from_pretrained(str(model_dir), local_files_only=True, torch_dtype=dtype)
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    pipeline = pipeline.to(device)
    generator = torch.Generator(device=device).manual_seed(deterministic_seed(context))

    kwargs: dict[str, Any] = {
        "prompt": prompt_from_package(context, scene_package),
        "width": width,
        "height": height,
        "num_frames": frame_count(scene_package, fps),
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "generator": generator,
    }
    primary_frame = existing_path(context.get("primary_frame", ""))
    if primary_frame is not None and "ImageToVideo" in getattr(pipeline_cls, "__name__", ""):
        kwargs["image"] = Image.open(primary_frame).convert("RGB").resize((width, height))

    result = pipeline(**kwargs)
    frames = getattr(result, "frames", None)
    if isinstance(frames, list) and frames and isinstance(frames[0], list):
        frames = frames[0]
    if not frames:
        videos = getattr(result, "videos", None)
        if videos:
            frames = videos[0]
    if not frames:
        raise RuntimeError("LTX did not return video frames.")

    ensure_parent(str(output_path))
    export_to_video(frames, str(output_path), fps=fps)
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError("LTX video export produced no output.")


def main() -> int:
    context = load_backend_context()
    scene_package = load_json(clean_text(context.get("scene_package", "")))
    if not scene_package:
        raise RuntimeError("Could not load scene package for the local LTX video backend.")
    output_text = clean_text(context.get("scene_video", ""))
    if not output_text:
        raise RuntimeError("The local LTX video backend did not receive a scene video output path.")
    output_path = Path(output_text)

    generate_ltx_video(context, scene_package, output_path)
    preview = Path(clean_text(context.get("video_preview_frame", "")))
    poster = Path(clean_text(context.get("video_poster_frame", "")))
    if str(preview):
        export_preview_frame(output_path, preview)
    if str(poster):
        if str(preview) and preview.exists():
            copy_if_needed(preview, poster)
        else:
            export_preview_frame(output_path, poster)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(print_runtime_error(exc))
