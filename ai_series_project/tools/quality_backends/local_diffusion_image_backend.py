#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from backend_common import PROJECT_DIR, copy_if_needed, ensure_parent, load_backend_context, load_json, print_runtime_error


DEFAULT_IMAGE_MODEL_DIR = PROJECT_DIR / "tools" / "quality_models" / "image" / "stabilityai__stable-diffusion-xl-base-1.0"


def clean_text(value: object) -> str:
    return str(value or "").strip()


def compact_visual_prompt(context: dict[str, Any], scene_package: dict[str, Any], original_prompt: str) -> str:
    characters = [
        clean_text(value)
        for value in scene_package.get("characters", [])
        if clean_text(value)
    ] if isinstance(scene_package.get("characters", []), list) else []
    camera = scene_package.get("camera_plan", {}) if isinstance(scene_package.get("camera_plan", {}), dict) else {}
    visible_count = len(characters)
    subject = (
        f"{visible_count} people clearly visible, characters {', '.join(characters[:3])}"
        if visible_count
        else "people clearly visible with expressive faces"
    )
    parts = [
        "live-action TV sitcom frame",
        "source-series visual style",
        clean_text(camera.get("shot_type", "") or camera.get("camera", "")),
        clean_text(camera.get("composition", "") or camera.get("focus", "")),
        clean_text(camera.get("camera_move", "") or camera.get("movement", "")),
        clean_text(camera.get("lens_hint", "") or camera.get("lens", "")),
        subject,
        clean_text(camera.get("pose_hint", "")),
        clean_text(scene_package.get("title", "")),
        "visible faces",
        "consistent wardrobe and set",
        "production lighting",
    ]
    compact = ", ".join(part for part in parts if part)
    if visible_count:
        return compact
    original_parts = [part.strip() for part in original_prompt.split(",") if part.strip()]
    return ", ".join([*original_parts[:4], compact])


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
    positive = compact_visual_prompt(context, scene_package, positive)
    if not negative:
        negative = (
            "cropped faces, distorted hands, unreadable text, watermark, duplicate characters, "
            "low quality, blurry, unfinished, blue placeholder frame"
        )
    return positive, negative


def deterministic_seed(context: dict[str, Any]) -> int:
    raw = "|".join(clean_text(context.get(key, "")) for key in ("episode_id", "scene_id", "shot_id", "scene_title", "runner_kind"))
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
    allow_cpu = clean_text(os.environ.get("SERIES_IMAGE_ALLOW_CPU", "")).lower() in {"1", "true", "yes", "on"}
    if not cuda_ready and not allow_cpu:
        raise RuntimeError(
            "The local SDXL backend requires a CUDA GPU. This worker is CPU-only, so it must not claim "
            "shot-image tasks. Run step 16 on a CUDA worker, or explicitly set SERIES_IMAGE_ALLOW_CPU=1 "
            "for a very slow diagnostic CPU render."
        )
    dtype = torch.float16 if cuda_ready else torch.float32
    device = "cuda" if cuda_ready else "cpu"
    width, height = target_size()
    steps = int(float(os.environ.get("SERIES_IMAGE_INFERENCE_STEPS", "28") or "28"))
    guidance = float(os.environ.get("SERIES_IMAGE_GUIDANCE_SCALE", "6.5") or "6.5")
    print(
        f"[INFO] SDXL backend: device={device}, size={width}x{height}, steps={steps}, model={model_dir}",
        flush=True,
    )

    print("[INFO] Loading local SDXL pipeline ...", flush=True)
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    if hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
    if cuda_ready:
        gpu_memory_gb = float(torch.cuda.get_device_properties(0).total_memory) / float(1024**3)
        if gpu_memory_gb < 10.0 and hasattr(pipeline, "enable_model_cpu_offload"):
            print(f"[INFO] Enabling model CPU offload for {gpu_memory_gb:.1f} GB GPU memory.", flush=True)
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to(device)
    else:
        pipeline = pipeline.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"[INFO] Generating image: {output_path}", flush=True)
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
    print(f"[OK] Saved generated image: {output_path}", flush=True)


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


def shot_prompt(prompt: str, shot: dict[str, Any]) -> str:
    parts = [
        prompt,
        clean_text(shot.get("shot_type", "")),
        clean_text(shot.get("camera_angle", "")),
        clean_text(shot.get("camera_movement", "")),
        clean_text(shot.get("purpose", "")),
        f"visible characters {', '.join(shot.get('characters_visible', []))}"
        if isinstance(shot.get("characters_visible", []), list) and shot.get("characters_visible", [])
        else "",
    ]
    return ", ".join(part for part in parts if part)


def shot_manifest(shot: dict[str, Any], output_path: Path, prompt: str, seed: int) -> None:
    outputs = shot.get("target_outputs", {}) if isinstance(shot.get("target_outputs", {}), dict) else {}
    manifest_text = clean_text(outputs.get("manifest", ""))
    if not manifest_text:
        return
    manifest_path = Path(manifest_text)
    ensure_parent(str(manifest_path))
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest() if output_path.exists() else ""
    payload = {
        "task_id": f"{clean_text(shot.get('shot_id', 'shot'))}_local_sdxl_image",
        "scene_id": clean_text(shot.get("scene_id", "")),
        "shot_id": clean_text(shot.get("shot_id", "")),
        "task_type": "image",
        "backend": "local_diffusion_image_backend",
        "inputs": {"prompt": prompt, "seed": seed},
        "outputs": {"primary_frame": str(output_path)},
        "output_hashes": {str(output_path): digest} if digest else {},
        "status": "success" if digest else "failed",
        "fallback_used": False,
        "placeholder_used": False,
        "finished_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def shot_packages(scene_package: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in scene_package.get("shot_packages", []) if isinstance(scene_package.get("shot_packages", []), list)
        if isinstance(row, dict)
        and clean_text((row.get("target_outputs", {}) if isinstance(row.get("target_outputs", {}), dict) else {}).get("primary_frame", ""))
    ]


def generate_shot_images(context: dict[str, Any], scene_package: dict[str, Any], prompt: str, negative_prompt: str) -> list[Path]:
    paths: list[Path] = []
    for shot in shot_packages(scene_package):
        outputs = shot.get("target_outputs", {}) if isinstance(shot.get("target_outputs", {}), dict) else {}
        output_path = Path(clean_text(outputs.get("primary_frame", "")))
        shot_context = {**context, "shot_id": clean_text(shot.get("shot_id", ""))}
        seed = deterministic_seed(shot_context)
        rendered_prompt = shot_prompt(prompt, shot)
        generate_image(rendered_prompt, negative_prompt, output_path, seed)
        shot_manifest(shot, output_path, rendered_prompt, seed)
        paths.append(output_path)
    return paths


def main() -> int:
    context = load_backend_context()
    scene_package_path = clean_text(context.get("scene_package", "")) or clean_text(context.get("backend_input", ""))
    scene_package = load_json(scene_package_path) if scene_package_path else {}
    paths = output_paths(context)
    if not paths:
        raise RuntimeError("The local diffusion backend did not receive an output image path.")
    primary_output = paths[0]
    prompt, negative_prompt = prompt_from_context(context, scene_package)
    generated_shots = generate_shot_images(context, scene_package, prompt, negative_prompt)
    if generated_shots:
        copy_if_needed(generated_shots[0], primary_output)
    else:
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
