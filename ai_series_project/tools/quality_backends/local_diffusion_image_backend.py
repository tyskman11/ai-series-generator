#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from PIL import Image

from backend_common import PROJECT_DIR, copy_if_needed, ensure_parent, load_backend_context, load_json, print_runtime_error


DEFAULT_IMAGE_MODEL_DIR = PROJECT_DIR / "tools" / "quality_models" / "image" / "stabilityai__stable-diffusion-xl-base-1.0"


def clean_text(value: object) -> str:
    return str(value or "").strip()


def resolve_model_dir() -> Path:
    configured = clean_text(os.environ.get("SERIES_IMAGE_MODEL_DIR", ""))
    candidate = Path(configured) if configured else DEFAULT_IMAGE_MODEL_DIR
    if not candidate.is_absolute():
        candidate = PROJECT_DIR / candidate
    candidate = candidate.resolve(strict=False)
    if not (candidate / "model_index.json").exists():
        raise RuntimeError(
            "Local SDXL image model is not ready. Run 00_prepare_runtime.py without --skip-downloads "
            f"or set SERIES_IMAGE_MODEL_DIR. Expected model_index.json in {candidate}."
        )
    return candidate


def prompt_from_context(context: dict[str, Any], scene_package: dict[str, Any]) -> tuple[str, str]:
    positive = clean_text(context.get("positive_prompt", ""))
    negative = clean_text(context.get("negative_prompt", ""))
    if not positive:
        image_generation = scene_package.get("image_generation", {}) if isinstance(scene_package.get("image_generation"), dict) else {}
        video_generation = scene_package.get("video_generation", {}) if isinstance(scene_package.get("video_generation"), dict) else {}
        continuity = scene_package.get("continuity", {}) if isinstance(scene_package.get("continuity"), dict) else {}
        prompt_parts = [
            clean_text(image_generation.get("positive_prompt", "")),
            clean_text(video_generation.get("positive_prompt", "")),
            clean_text(context.get("scene_title", "")),
            clean_text(scene_package.get("summary", "")),
            clean_text(continuity.get("style_prompt", "")),
        ]
        positive = ", ".join(part for part in prompt_parts if part)
    if not positive:
        positive = "faithful original TV episode still, cinematic scene, coherent characters, production lighting"
    if not negative:
        negative = (
            "cropped faces, distorted hands, unreadable text, watermark, duplicate characters, "
            "low quality, blurry, unfinished, blue placeholder frame"
        )
    return positive, negative


def deterministic_seed(context: dict[str, Any]) -> int:
    raw = "|".join(clean_text(context.get(key, "")) for key in ("episode_id", "scene_id", "scene_title", "runner_kind"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def target_size() -> tuple[int, int]:
    width = int(float(os.environ.get("SERIES_IMAGE_WIDTH", "1024") or "1024"))
    height = int(float(os.environ.get("SERIES_IMAGE_HEIGHT", "576") or "576"))
    width = max(512, (width // 8) * 8)
    height = max(512, (height // 8) * 8)
    return width, height


def generate_image(prompt: str, negative_prompt: str, output_path: Path, seed: int) -> None:
    try:
        import torch
        from diffusers import StableDiffusionXLPipeline
    except Exception as exc:
        raise RuntimeError(
            "The real local image backend requires diffusers and torch. Run 00_prepare_runtime.py first."
        ) from exc

    model_dir = resolve_model_dir()
    cuda_ready = bool(torch.cuda.is_available())
    dtype = torch.float16 if cuda_ready else torch.float32
    device = "cuda" if cuda_ready else "cpu"
    width, height = target_size()
    steps = int(float(os.environ.get("SERIES_IMAGE_INFERENCE_STEPS", "28") or "28"))
    guidance = float(os.environ.get("SERIES_IMAGE_GUIDANCE_SCALE", "6.5") or "6.5")

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    pipeline = pipeline.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )
    image = result.images[0]
    ensure_parent(str(output_path))
    image.save(output_path)


def output_paths(context: dict[str, Any]) -> list[Path]:
    runner_kind = clean_text(context.get("runner_kind", ""))
    if runner_kind == "storyboard":
        paths = [
            clean_text(context.get("frame", "")),
            clean_text(context.get("preview_frame", "")),
            clean_text(context.get("poster_frame", "")),
        ]
    else:
        paths = [
            clean_text(context.get("primary_frame", "")),
            clean_text(context.get("layered_storyboard_frame", "")),
        ]
    return [Path(path) for path in paths if path]


def write_alternate_image(context: dict[str, Any], source: Path) -> None:
    alternate_dir = clean_text(context.get("alternate_frame_dir", ""))
    if not alternate_dir:
        return
    target_dir = Path(alternate_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{source.stem}_sdxl_alt01{source.suffix or '.png'}"
    copy_if_needed(source, target)


def main() -> int:
    context = load_backend_context()
    scene_package_path = clean_text(context.get("scene_package", "")) or clean_text(context.get("backend_input", ""))
    scene_package = load_json(scene_package_path) if scene_package_path else {}
    paths = output_paths(context)
    if not paths:
        raise RuntimeError("The local diffusion backend did not receive an output image path.")
    primary_output = paths[0]
    prompt, negative_prompt = prompt_from_context(context, scene_package)
    generate_image(prompt, negative_prompt, primary_output, deterministic_seed(context))

    for extra_path in paths[1:]:
        copy_if_needed(primary_output, extra_path)
    write_alternate_image(context, primary_output)
    with Image.open(primary_output) as image:
        image.verify()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(print_runtime_error(exc))
