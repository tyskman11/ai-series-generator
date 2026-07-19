#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import inspect
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from backend_common import (
    PROJECT_DIR,
    apply_torch_resource_limits,
    copy_if_needed,
    ensure_parent,
    load_backend_context,
    load_json,
    print_runtime_error,
    resolve_stored_project_path,
)


DEFAULT_IMAGE_MODEL_ID = "Qwen/Qwen-Image"
DEFAULT_IMAGE_MODEL_DIR = PROJECT_DIR / "tools" / "quality_models" / "image" / "Qwen__Qwen-Image"
SDXL_IDENTITY_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_IDENTITY_MODEL_DIR = PROJECT_DIR / "tools" / "quality_models" / "image" / "stabilityai__stable-diffusion-xl-base-1.0"
DEFAULT_IDENTITY_ADAPTER_DIR = PROJECT_DIR / "tools" / "quality_models" / "image" / "h94__IP-Adapter"
IDENTITY_ADAPTER_WEIGHT = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
FALLBACK_IMAGE_MODEL_DIRS = [DEFAULT_IMAGE_MODEL_DIR, SDXL_IDENTITY_MODEL_DIR]
_PIPELINE: Any = None
_PIPELINE_META: dict[str, Any] = {}


def clean_text(value: object) -> str:
    return str(value or "").strip()


def truthy_env(name: str, default: bool) -> bool:
    value = clean_text(os.environ.get(name, ""))
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def clean_text_list(value: object) -> list[str]:
    return [clean_text(item) for item in value if clean_text(item)] if isinstance(value, list) else []


def scene_character_names(scene_package: dict[str, Any]) -> list[str]:
    return clean_text_list(scene_package.get("characters", []))


def identity_lock(scene_package: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = scene_package.get("character_continuity_lock", {})
    if not isinstance(value, dict) or not value:
        image_generation = scene_package.get("image_generation", {}) if isinstance(scene_package.get("image_generation", {}), dict) else {}
        value = image_generation.get("character_continuity_lock", {})
    return {
        clean_text(name): dict(payload)
        for name, payload in value.items()
        if clean_text(name) and isinstance(payload, dict)
    } if isinstance(value, dict) else {}


def reference_slots(scene_package: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        scene_package.get("reference_slots", []),
        (scene_package.get("storyboard", {}) if isinstance(scene_package.get("storyboard", {}), dict) else {}).get("reference_slots", []),
        (scene_package.get("image_generation", {}) if isinstance(scene_package.get("image_generation", {}), dict) else {}).get("reference_slots", []),
    ]
    for value in candidates:
        if isinstance(value, list) and value:
            return [dict(item) for item in value if isinstance(item, dict)]
    return []


def character_identity_descriptors(scene_package: dict[str, Any], names: list[str]) -> list[str]:
    locks = identity_lock(scene_package)
    descriptors: list[str] = []
    for name in names:
        row = locks.get(name, {})
        details = [
            clean_text(row.get("identity_attribute_policy", "")),
            clean_text(row.get("gender_presentation", "")),
            clean_text(row.get("age_group", "")),
            clean_text(row.get("canonical_hairstyle", "")),
            clean_text(row.get("canonical_outfit", "")),
            clean_text(row.get("canonical_body_shape", "")),
        ]
        useful = [
            detail
            for detail in details
            if detail and not detail.lower().startswith(("match reviewed", "preserve source"))
        ]
        descriptors.append(f"{name}: {', '.join(useful)}" if useful else f"{name}: exact identity from supplied face references")
    return descriptors


def compact_visual_prompt(context: dict[str, Any], scene_package: dict[str, Any], original_prompt: str) -> str:
    characters = scene_character_names(scene_package)
    camera = scene_package.get("camera_plan", {}) if isinstance(scene_package.get("camera_plan", {}), dict) else {}
    image_generation = (
        scene_package.get("image_generation", {})
        if isinstance(scene_package.get("image_generation", {}), dict)
        else {}
    )
    generation_mode = clean_text(image_generation.get("mode", ""))
    people_free_intro = generation_mode.startswith("generated_season_") and generation_mode.endswith("_keyframes") and not characters
    subject = (
        f"canonical cast identities available for {', '.join(characters[:3])}"
        if characters
        else "empty recurring set and signature props, no people visible"
        if people_free_intro
        else "people clearly visible with expressive faces"
    )
    profile = local_model_profile(scene_package)
    parts = [
        "serialized anime TV keyframe" if clean_text(profile.get("profile_id", "")) == "anime" else "live-action TV sitcom frame",
        "source-series visual style",
        clean_text(camera.get("shot_type", "") or camera.get("camera", "")),
        clean_text(camera.get("composition", "") or camera.get("focus", "")),
        clean_text(camera.get("camera_move", "") or camera.get("movement", "")),
        clean_text(camera.get("lens_hint", "") or camera.get("lens", "")),
        subject,
        clean_text(camera.get("pose_hint", "")),
        clean_text(scene_package.get("title", "")),
        "visible faces" if not people_free_intro else "",
        "exact same facial identity, age, gender presentation, hairstyle, body proportions and wardrobe in every shot"
        if not people_free_intro
        else "",
        "natural undistorted symmetrical faces and eyes" if not people_free_intro else "",
        "consistent wardrobe and set geography",
        "production lighting",
        *character_identity_descriptors(scene_package, characters),
    ]
    compact = ", ".join(part for part in parts if part)
    if characters:
        return compact
    original_parts = [part.strip() for part in original_prompt.split(",") if part.strip()]
    return ", ".join([*original_parts[:4], compact])


def model_dir_ready(candidate: Path) -> bool:
    return (candidate / "model_index.json").is_file() and any(candidate.rglob("*.safetensors"))


def normalize_model_dir(candidate: Path) -> Path:
    if not candidate.is_absolute():
        candidate = PROJECT_DIR / candidate
    return candidate.resolve(strict=False)


def image_model_family(model_id: str, model_dir: Path) -> str:
    text = f"{model_id} {model_dir}".lower()
    if "animagine" in text:
        return "sdxl"
    if "qwen" in text:
        return "qwen"
    if "flux" in text:
        return "flux"
    if "stable-diffusion-xl" in text or "sdxl" in text:
        return "sdxl"
    return "diffusers"


def local_model_profile(scene_package: dict[str, Any]) -> dict[str, Any]:
    plan = scene_package.get("generation_plan", {}) if isinstance(scene_package.get("generation_plan", {}), dict) else {}
    profile = plan.get("local_model_profile", {}) if isinstance(plan.get("local_model_profile", {}), dict) else {}
    return dict(profile)


def apply_local_model_profile(scene_package: dict[str, Any]) -> str:
    profile = local_model_profile(scene_package)
    profile_id = clean_text(profile.get("profile_id", "")) or "live_action"
    for key, value in {
        "SERIES_IMAGE_MODEL_ID": clean_text(profile.get("image_model_id", "")),
        "SERIES_IMAGE_MODEL_DIR": clean_text(profile.get("image_model_dir", "")),
        "SERIES_IMAGE_IDENTITY_MODEL_ID": clean_text(profile.get("identity_model_id", "")),
        "SERIES_IMAGE_IDENTITY_MODEL_DIR": clean_text(profile.get("identity_model_dir", "")),
    }.items():
        if value:
            os.environ[key] = value
    return profile_id


def configured_model_id(candidate: Path, *, require_identity_adapter: bool, cpu_safe: bool = False) -> str:
    if cpu_safe:
        configured_cpu = clean_text(os.environ.get("SERIES_IMAGE_CPU_MODEL_ID", ""))
        if configured_cpu:
            return configured_cpu
        resolved = candidate.resolve(strict=False)
        if resolved == SDXL_IDENTITY_MODEL_DIR.resolve(strict=False):
            return SDXL_IDENTITY_MODEL_ID
        if resolved == DEFAULT_IMAGE_MODEL_DIR.resolve(strict=False):
            return DEFAULT_IMAGE_MODEL_ID
        return candidate.name.replace("__", "/")
    env_name = "SERIES_IMAGE_IDENTITY_MODEL_ID" if require_identity_adapter else "SERIES_IMAGE_MODEL_ID"
    configured = clean_text(os.environ.get(env_name, ""))
    if configured:
        return configured
    resolved = candidate.resolve(strict=False)
    if resolved == DEFAULT_IMAGE_MODEL_DIR.resolve(strict=False):
        return DEFAULT_IMAGE_MODEL_ID
    if resolved == SDXL_IDENTITY_MODEL_DIR.resolve(strict=False):
        return SDXL_IDENTITY_MODEL_ID
    return candidate.name.replace("__", "/")


def image_model_candidates(*, require_identity_adapter: bool, cpu_safe: bool = False) -> list[Path]:
    if require_identity_adapter:
        configured = (
            clean_text(os.environ.get("SERIES_IMAGE_IDENTITY_MODEL_DIR", ""))
            or clean_text(os.environ.get("SERIES_IMAGE_IDENTITY_FALLBACK_MODEL_DIR", ""))
        )
        if configured:
            return [normalize_model_dir(Path(configured))]
        return [SDXL_IDENTITY_MODEL_DIR.resolve(strict=False)]
    if cpu_safe:
        configured_cpu = clean_text(os.environ.get("SERIES_IMAGE_CPU_MODEL_DIR", ""))
        if configured_cpu:
            return [normalize_model_dir(Path(configured_cpu))]
        return [SDXL_IDENTITY_MODEL_DIR.resolve(strict=False)]
    configured = clean_text(os.environ.get("SERIES_IMAGE_MODEL_DIR", ""))
    if configured:
        return [normalize_model_dir(Path(configured))]
    return [path.resolve(strict=False) for path in FALLBACK_IMAGE_MODEL_DIRS]


def resolve_model_dir(*, require_identity_adapter: bool = False, cpu_safe: bool = False) -> Path:
    candidates = image_model_candidates(require_identity_adapter=require_identity_adapter, cpu_safe=cpu_safe)
    for candidate in candidates:
        if model_dir_ready(candidate):
            return candidate
    expected = ", ".join(str(candidate) for candidate in candidates)
    if require_identity_adapter:
        raise RuntimeError(
            "Canonical face references require the project-local SDXL identity model because the current "
            "IP-Adapter face workflow is SDXL-based. Run 00_prepare_runtime.py without --skip-downloads or set "
            f"SERIES_IMAGE_IDENTITY_MODEL_DIR. Expected model_index.json and safetensors files in {expected}."
        )
    if cpu_safe:
        raise RuntimeError(
            "The CPU-safe local image model is not ready. Run 00_prepare_runtime.py without --skip-downloads "
            f"or set SERIES_IMAGE_CPU_MODEL_DIR. Expected model_index.json and safetensors files in {expected}."
        )
    raise RuntimeError(
        "Local Qwen-Image model is not ready. Run 00_prepare_runtime.py without --skip-downloads "
        f"or set SERIES_IMAGE_MODEL_DIR. Expected model_index.json and safetensors files in {expected}."
    )


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
            "cropped faces, deformed face, warped face, melted face, asymmetrical eyes, crossed eyes, "
            "wrong person, identity drift, face swap, gender swap, age swap, wrong hairstyle, wrong body proportions, "
            "merged people, duplicate characters, extra people, unnamed extras, background crowd, random children, "
            "missing people, distorted hands, unreadable text, watermark, low quality, blurry, unfinished, "
            "blue placeholder frame"
        )
    else:
        negative = ", ".join(
            part
            for part in [
                negative,
                "unnamed extras, background crowd, random children, extra faces, duplicate cast members, identity drift",
            ]
            if part
        )
    return positive, negative


def deterministic_seed(context: dict[str, Any], scene_package: dict[str, Any] | None = None) -> int:
    scene_package = scene_package if isinstance(scene_package, dict) else {}
    characters = sorted(scene_character_names(scene_package))
    raw = "|".join(
        [
            clean_text(context.get("episode_id", "")),
            clean_text(context.get("scene_id", "")),
            clean_text(context.get("scene_title", "")),
            clean_text(context.get("runner_kind", "")),
            ",".join(characters),
        ]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def target_size() -> tuple[int, int]:
    width = int(float(os.environ.get("SERIES_IMAGE_WIDTH", "1024") or "1024"))
    height = int(float(os.environ.get("SERIES_IMAGE_HEIGHT", "576") or "576"))
    width = max(512, (width // 8) * 8)
    height = max(512, (height // 8) * 8)
    return width, height


def resolve_identity_adapter_dir() -> Path:
    configured = clean_text(os.environ.get("SERIES_IMAGE_IDENTITY_ADAPTER_DIR", ""))
    candidate = Path(configured) if configured else DEFAULT_IDENTITY_ADAPTER_DIR
    if not candidate.is_absolute():
        candidate = PROJECT_DIR / candidate
    return candidate.resolve(strict=False)


def identity_adapter_ready(adapter_dir: Path) -> bool:
    return (
        (adapter_dir / "sdxl_models" / IDENTITY_ADAPTER_WEIGHT).is_file()
        and (adapter_dir / "models" / "image_encoder" / "config.json").is_file()
    )


def enable_pipeline_memory_optimizations(pipeline: Any, *, identity_adapter_loaded: bool) -> None:
    # Attention slicing replaces attention processors in some diffusers releases.
    # Preserve the IP-Adapter processors once identity conditioning is active.
    if not identity_adapter_loaded and hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    elif identity_adapter_loaded:
        print("[INFO] Preserving IP-Adapter attention processors; attention slicing is disabled.", flush=True)

    vae = getattr(pipeline, "vae", None)
    if vae is not None and hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    elif hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    if vae is not None and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    elif hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()


def pipeline_accepts_argument(pipeline: Any, name: str) -> bool:
    try:
        signature = inspect.signature(pipeline.__call__)
    except (TypeError, ValueError):
        return True
    return name in signature.parameters


def load_pipeline(require_identity_adapter: bool) -> tuple[Any, dict[str, Any]]:
    global _PIPELINE, _PIPELINE_META
    try:
        import torch
        from diffusers import DiffusionPipeline
    except Exception as exc:
        raise RuntimeError(
            "The real local image backend requires diffusers and torch. Run 00_prepare_runtime.py first."
        ) from exc

    resource_limits = apply_torch_resource_limits(torch)
    cuda_ready = bool(torch.cuda.is_available())
    allow_cpu = clean_text(os.environ.get("SERIES_IMAGE_ALLOW_CPU", "")).lower() in {"1", "true", "yes", "on"}
    allow_heavy_cpu = truthy_env("SERIES_IMAGE_ALLOW_HEAVY_CPU", False)
    if not cuda_ready and not allow_cpu:
        raise RuntimeError(
            "The local image backend requires a CUDA GPU for production speed. This worker is CPU-only, so it must not claim "
            "shot-image tasks. Run step 16 on a CUDA worker, or explicitly set SERIES_IMAGE_ALLOW_CPU=1 "
            "for a very slow diagnostic CPU render."
        )
    cpu_safe = bool(not cuda_ready and not require_identity_adapter and not allow_heavy_cpu)
    model_dir = resolve_model_dir(require_identity_adapter=require_identity_adapter, cpu_safe=cpu_safe)
    model_id = configured_model_id(model_dir, require_identity_adapter=require_identity_adapter, cpu_safe=cpu_safe)
    model_family = image_model_family(model_id, model_dir)
    if _PIPELINE is not None and clean_text(_PIPELINE_META.get("model_dir", "")) == str(model_dir):
        cached_adapter_loaded = bool(_PIPELINE_META.get("identity_adapter_loaded", False))
        if cached_adapter_loaded == require_identity_adapter:
            return _PIPELINE, dict(_PIPELINE_META)
        print(
            "[INFO] Reloading image pipeline because the shot identity-adapter requirement changed.",
            flush=True,
        )
        _PIPELINE = None
        _PIPELINE_META = {}
    dtype = torch.bfloat16 if cuda_ready and model_family in {"flux", "qwen"} else torch.float16 if cuda_ready else torch.float32
    device = "cuda" if cuda_ready else "cpu"
    width, height = target_size()
    default_steps = "50" if model_family == "qwen" else "28" if model_family == "flux" else "36"
    default_guidance = "4.0" if model_family == "qwen" else "3.5" if model_family == "flux" else "6.0"
    steps = int(float(os.environ.get("SERIES_IMAGE_INFERENCE_STEPS", default_steps) or default_steps))
    guidance = float(os.environ.get("SERIES_IMAGE_GUIDANCE_SCALE", default_guidance) or default_guidance)
    if cpu_safe:
        max_cpu_width = int(float(os.environ.get("SERIES_IMAGE_CPU_MAX_WIDTH", "896") or "896"))
        max_cpu_height = int(float(os.environ.get("SERIES_IMAGE_CPU_MAX_HEIGHT", "512") or "512"))
        max_cpu_steps = int(float(os.environ.get("SERIES_IMAGE_CPU_MAX_STEPS", "24") or "24"))
        original_size = (width, height, steps)
        width = max(512, (min(width, max_cpu_width) // 8) * 8)
        height = max(512, (min(height, max_cpu_height) // 8) * 8)
        steps = max(12, min(steps, max_cpu_steps))
        print(
            "[INFO] CPU-safe image generation active: "
            f"using {model_id} at {width}x{height}, {steps} steps instead of "
            f"{original_size[0]}x{original_size[1]}, {original_size[2]} steps. "
            "Set SERIES_IMAGE_ALLOW_HEAVY_CPU=1 to force the configured heavy model on CPU.",
            flush=True,
        )
    print(
        f"[INFO] {model_family.upper()} image backend: device={device}, size={width}x{height}, steps={steps}, model={model_id}",
        flush=True,
    )

    print("[INFO] Loading local Diffusers image pipeline ...", flush=True)
    pipeline = DiffusionPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    adapter_dir = resolve_identity_adapter_dir()
    adapter_loaded = False
    if model_family != "sdxl" and require_identity_adapter:
        raise RuntimeError(
            "Identity-conditioned shots currently use the SDXL IP-Adapter face workflow. "
            f"The selected model {model_id} is not SDXL; set SERIES_IMAGE_IDENTITY_MODEL_DIR to the project-local SDXL model."
        )
    if model_family == "sdxl" and require_identity_adapter and identity_adapter_ready(adapter_dir):
        try:
            pipeline.load_ip_adapter(
                str(adapter_dir),
                subfolder="sdxl_models",
                weight_name=IDENTITY_ADAPTER_WEIGHT,
                image_encoder_folder="models/image_encoder",
                local_files_only=True,
            )
            if hasattr(pipeline, "set_ip_adapter_scale"):
                pipeline.set_ip_adapter_scale(
                    float(os.environ.get("SERIES_IMAGE_IDENTITY_SCALE", "0.82") or "0.82")
                )
            adapter_loaded = True
            print(f"[OK] Canonical face identity adapter loaded: {adapter_dir}", flush=True)
        except Exception as exc:
            if require_identity_adapter:
                raise RuntimeError(
                    f"Could not load the project-local SDXL identity adapter from {adapter_dir}: {exc}"
                ) from exc
            print(f"[WARN] SDXL identity adapter unavailable: {exc}", flush=True)
    elif model_family == "sdxl" and require_identity_adapter:
        raise RuntimeError(
            "The project-local SDXL identity adapter is incomplete. Run 00_prepare_runtime.py without "
            f"--skip-downloads. Expected {adapter_dir / 'sdxl_models' / IDENTITY_ADAPTER_WEIGHT} and "
            f"{adapter_dir / 'models' / 'image_encoder' / 'config.json'}."
        )
    elif model_family == "sdxl":
        print("[INFO] Identity adapter not loaded for this no-character/text-only shot.", flush=True)
    enable_pipeline_memory_optimizations(
        pipeline,
        identity_adapter_loaded=adapter_loaded,
    )
    if cuda_ready:
        gpu_memory_gb = float(torch.cuda.get_device_properties(0).total_memory) / float(1024**3)
        if gpu_memory_gb < 10.0 and hasattr(pipeline, "enable_model_cpu_offload"):
            print(f"[INFO] Enabling model CPU offload for {gpu_memory_gb:.1f} GB GPU memory.", flush=True)
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to(device)
    else:
        pipeline = pipeline.to(device)
    _PIPELINE = pipeline
    _PIPELINE_META = {
        "device": device,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": guidance,
        "model_id": model_id,
        "model_dir": str(model_dir),
        "model_family": model_family,
        "pipeline_class": type(pipeline).__name__,
        "identity_adapter_dir": str(adapter_dir),
        "identity_adapter_loaded": adapter_loaded,
        "cpu_safe_mode": cpu_safe,
        "resource_limits": resource_limits,
    }
    return pipeline, dict(_PIPELINE_META)


def build_identity_reference_board(reference_paths: list[Path]) -> Image.Image | None:
    usable: list[Image.Image] = []
    for path in reference_paths[:3]:
        try:
            with Image.open(path) as source:
                usable.append(ImageOps.fit(source.convert("RGB"), (256, 256), method=Image.Resampling.LANCZOS))
        except Exception:
            continue
    if not usable:
        return None
    # IP-Adapter Face is stable with one canonical portrait. A contact sheet of
    # different poses makes the adapter blend identity attributes instead of
    # keeping the person from the reviewed source material.
    return ImageOps.fit(usable[0], (512, 512), method=Image.Resampling.LANCZOS)


UNSAFE_IDENTITY_REFERENCE_TOKENS = {
    "montage",
    "contact",
    "sheet",
    "context",
    "storyboard",
    "preview",
    "poster",
    "panel",
    "collage",
    "grid",
    "speaker_frame",
    "full_frame",
    "scene_frame",
}


def identity_reference_is_safe(path: Path) -> bool:
    name = path.name.lower()
    if any(token in name for token in UNSAFE_IDENTITY_REFERENCE_TOKENS):
        return False
    stem = path.stem.lower()
    if "_crop" in stem or "portrait" in stem or stem.startswith("face_"):
        return True
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}


def unsafe_identity_reference_images(paths: list[Path]) -> list[str]:
    return [str(path) for path in paths if not identity_reference_is_safe(path)]


def manifest_identity_references_are_safe(manifest_path: Path) -> bool:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    inputs = payload.get("inputs", {}) if isinstance(payload.get("inputs"), dict) else {}
    references = inputs.get("identity_reference_images", [])
    if not isinstance(references, list):
        return False
    return not unsafe_identity_reference_images([Path(str(value)) for value in references])


def generate_image(
    prompt: str,
    negative_prompt: str,
    output_path: Path,
    seed: int,
    *,
    identity_reference: Image.Image | None = None,
) -> dict[str, Any]:
    require_identity = identity_reference is not None and truthy_env("SERIES_IMAGE_REQUIRE_IDENTITY_ADAPTER", True)
    pipeline, pipeline_meta = load_pipeline(require_identity)
    import torch

    device = clean_text(pipeline_meta.get("device", "")) or "cpu"
    width = int(pipeline_meta.get("width", 1024) or 1024)
    height = int(pipeline_meta.get("height", 576) or 576)
    steps = int(pipeline_meta.get("steps", 36) or 36)
    guidance = float(pipeline_meta.get("guidance", 6.0) or 6.0)
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"[INFO] Generating image: {output_path}", flush=True)
    arguments: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "generator": generator,
    }
    if negative_prompt and pipeline_accepts_argument(pipeline, "negative_prompt"):
        arguments["negative_prompt"] = negative_prompt
    if pipeline_accepts_argument(pipeline, "true_cfg_scale"):
        arguments["true_cfg_scale"] = guidance
    elif pipeline_accepts_argument(pipeline, "guidance_scale"):
        arguments["guidance_scale"] = guidance
    if identity_reference is not None and pipeline_accepts_argument(pipeline, "ip_adapter_image"):
        arguments["ip_adapter_image"] = identity_reference
    result = pipeline(
        **arguments,
    )
    image = result.images[0]
    ensure_parent(str(output_path))
    image.save(output_path)
    print(f"[OK] Saved generated image: {output_path}", flush=True)
    return pipeline_meta


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
    target = target_dir / f"{source.stem}_image_alt01{source.suffix or '.png'}"
    copy_if_needed(source, target)


def shot_prompt(prompt: str, shot: dict[str, Any]) -> str:
    visible_characters = clean_text_list(shot.get("characters_visible", []))
    explicit_people_free = isinstance(shot.get("characters_visible"), list) and not visible_characters
    visible_count = len(visible_characters)
    exact_people_contract = ""
    if visible_characters:
        exact_people_contract = (
            f"STRICT CAST COUNT: exactly {visible_count} person{'s' if visible_count != 1 else ''} total in frame, "
            f"only {', '.join(visible_characters)}, no extras, no background people, no crowd, no duplicated faces, "
            "do not add any other person"
        )
    parts = [
        prompt,
        clean_text(shot.get("shot_type", "")),
        clean_text(shot.get("camera_angle", "")),
        clean_text(shot.get("camera_movement", "")),
        clean_text(shot.get("purpose", "")),
        exact_people_contract,
        f"exactly {len(visible_characters)} visible named characters: {', '.join(visible_characters)}; preserve each supplied identity separately"
        if visible_characters
        else "STRICT EMPTY SET SHOT: no people visible, no faces, environment and props only"
        if explicit_people_free
        else "",
    ]
    return ", ".join(part for part in parts if part)


def shot_identity_contract(
    visible_characters: list[str],
    reference_paths: list[Path],
    conditioned_characters: list[str],
    missing_characters: list[str],
) -> dict[str, Any]:
    visible_count = len(visible_characters)
    multi_character = visible_count > 1
    return {
        "expected_visible_characters": visible_characters,
        "expected_visible_character_count": visible_count,
        "maximum_allowed_visible_people": visible_count,
        "forbid_unnamed_extras": True,
        "forbid_duplicate_characters": True,
        "forbid_gender_or_age_swap": True,
        "reference_mode": "multi_character_reference_board" if multi_character else "single_character_reference",
        "regional_identity_control": False,
        "verification_status": "unverified_multi_character_identity_board" if multi_character else "single_character_identity_conditioned",
        "identity_risk": "high" if multi_character else "normal",
        "generation_allowed": not multi_character or truthy_env("SERIES_IMAGE_ALLOW_UNVERIFIED_MULTI_IDENTITY", False),
        "reference_count": len(reference_paths),
        "characters_conditioned": conditioned_characters,
        "missing_characters": missing_characters,
    }


def reference_images_by_character(scene_package: dict[str, Any], names: list[str]) -> dict[str, list[Path]]:
    locks = identity_lock(scene_package)
    slots = reference_slots(scene_package)
    slot_index = {
        clean_text(slot.get("name", "")): slot
        for slot in slots
        if clean_text(slot.get("name", "")) and clean_text(slot.get("type", "")) == "character"
    }
    results: dict[str, list[Path]] = {}

    def append(target: list[Path], seen: set[str], value: object) -> None:
        candidate = resolve_stored_project_path(value)
        if not candidate.is_file():
            return
        if not identity_reference_is_safe(candidate):
            return
        key = str(candidate.resolve(strict=False)).lower()
        if key not in seen:
            seen.add(key)
            target.append(candidate)

    for name in names:
        lock = locks.get(name, {})
        slot = slot_index.get(name, {})
        portrait_values = clean_text_list(slot.get("portrait_images", []))
        context_values = clean_text_list(slot.get("context_images", []))
        lock_values = clean_text_list(lock.get("reference_images", []))
        character_paths: list[Path] = []
        seen: set[str] = set()
        for value in [*portrait_values, *lock_values, *context_values]:
            append(character_paths, seen, value)
        results[name] = character_paths
    return results


def reference_image_paths(scene_package: dict[str, Any], names: list[str]) -> list[Path]:
    references = reference_images_by_character(scene_package, names)
    results: list[Path] = []
    max_references = 4
    depth = 0
    while len(results) < max_references:
        appended = False
        for name in names:
            candidates = references.get(name, [])
            if depth < len(candidates):
                results.append(candidates[depth])
                appended = True
                if len(results) >= max_references:
                    break
        if not appended:
            break
        depth += 1
    return results


def require_identity_references(names: list[str], references: dict[str, list[Path]]) -> None:
    missing = [name for name in names if not references.get(name)]
    if missing and truthy_env("SERIES_IMAGE_REQUIRE_IDENTITY_REFERENCES", True):
        raise RuntimeError(
            "No canonical face reference images were resolved for "
            f"{', '.join(missing)}. Rerun 08_train_series_model.py and 14_generate_episode.py before generating frames."
        )


def shot_manifest(
    shot: dict[str, Any],
    output_path: Path,
    prompt: str,
    seed: int,
    reference_paths: list[Path],
    reference_characters: list[str],
    missing_characters: list[str],
    pipeline_meta: dict[str, Any],
) -> None:
    outputs = shot.get("target_outputs", {}) if isinstance(shot.get("target_outputs", {}), dict) else {}
    manifest_text = clean_text(outputs.get("image_manifest", "")) or clean_text(outputs.get("manifest", ""))
    if not manifest_text:
        return
    manifest_path = Path(manifest_text)
    ensure_parent(str(manifest_path))
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest() if output_path.exists() else ""
    unsafe_references = unsafe_identity_reference_images(reference_paths)
    payload = {
        "task_id": f"{clean_text(shot.get('shot_id', 'shot'))}_local_image",
        "scene_id": clean_text(shot.get("scene_id", "")),
        "shot_id": clean_text(shot.get("shot_id", "")),
        "task_type": "image",
        "backend": "local_diffusion_image_backend",
        "inputs": {
            "prompt": prompt,
            "seed": seed,
            "identity_reference_images": [str(path) for path in reference_paths],
            "identity_seed_scope": "episode_scene_character_set",
            "model_id": clean_text(pipeline_meta.get("model_id", "")),
            "model_family": clean_text(pipeline_meta.get("model_family", "")),
        },
        "outputs": {"primary_frame": str(output_path)},
        "output_hashes": {str(output_path): digest} if digest else {},
        "status": "success" if digest else "failed",
        "fallback_used": False,
        "placeholder_used": False,
        "identity_conditioning": {
            "mode": "sdxl_ip_adapter_plus_face" if reference_paths else "none",
            "adapter_loaded": bool(pipeline_meta.get("identity_adapter_loaded", False)),
            "adapter_dir": clean_text(pipeline_meta.get("identity_adapter_dir", "")),
            "reference_count": len(reference_paths),
            "characters_conditioned": reference_characters,
            "missing_characters": missing_characters,
            "reference_safety": not unsafe_references,
            "unsafe_reference_images": unsafe_references,
            "identity_model_id": clean_text(pipeline_meta.get("model_id", "")),
        },
        "identity_contract": shot_identity_contract(
            clean_text_list(shot.get("characters_visible", [])),
            reference_paths,
            reference_characters,
            missing_characters,
        ),
        "model": {
            "id": clean_text(pipeline_meta.get("model_id", "")),
            "family": clean_text(pipeline_meta.get("model_family", "")),
            "dir": clean_text(pipeline_meta.get("model_dir", "")),
            "pipeline_class": clean_text(pipeline_meta.get("pipeline_class", "")),
        },
        "resource_limits": pipeline_meta.get("resource_limits", {}),
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
    shared_seed = deterministic_seed(context, scene_package)
    resume_existing = truthy_env("SERIES_IMAGE_RESUME_SHOTS", False)
    for shot in shot_packages(scene_package):
        outputs = shot.get("target_outputs", {}) if isinstance(shot.get("target_outputs", {}), dict) else {}
        output_path = Path(clean_text(outputs.get("primary_frame", "")))
        manifest_text = clean_text(outputs.get("image_manifest", "")) or clean_text(outputs.get("manifest", ""))
        manifest_path = Path(manifest_text) if manifest_text else Path()
        if (
            resume_existing
            and output_path.is_file()
            and output_path.stat().st_size > 0
            and manifest_text
            and manifest_path.is_file()
            and manifest_path.stat().st_size > 0
            and manifest_identity_references_are_safe(manifest_path)
        ):
            print(f"[INFO] Resuming image package: keeping completed shot {shot.get('shot_id', output_path.stem)}", flush=True)
            paths.append(output_path)
            continue
        visible_value = shot.get("characters_visible")
        visible_characters = (
            clean_text_list(visible_value)
            if isinstance(visible_value, list)
            else scene_character_names(scene_package)
        )
        if len(visible_characters) > 1 and not truthy_env("SERIES_IMAGE_ALLOW_UNVERIFIED_MULTI_IDENTITY", False):
            raise RuntimeError(
                f"{shot.get('shot_id', 'shot')} asks for {len(visible_characters)} visible named characters, but the local "
                "SDXL IP-Adapter backend cannot verify independent multi-person identity conditioning. "
                "Regenerate the shot plan with single-character coverage, or configure a backend with regional identity control."
            )
        references_by_character = reference_images_by_character(scene_package, visible_characters)
        require_identity_references(visible_characters, references_by_character)
        reference_paths = reference_image_paths(scene_package, visible_characters)
        conditioned_characters = [name for name in visible_characters if references_by_character.get(name)]
        missing_characters = [name for name in visible_characters if not references_by_character.get(name)]
        identity_reference = build_identity_reference_board(reference_paths)
        rendered_prompt = shot_prompt(prompt, shot)
        pipeline_meta = generate_image(
            rendered_prompt,
            negative_prompt,
            output_path,
            shared_seed,
            identity_reference=identity_reference,
        )
        shot_manifest(
            shot,
            output_path,
            rendered_prompt,
            shared_seed,
            reference_paths,
            conditioned_characters,
            missing_characters,
            pipeline_meta,
        )
        paths.append(output_path)
    return paths


def main() -> int:
    context = load_backend_context()
    scene_package_path = clean_text(context.get("scene_package", "")) or clean_text(context.get("backend_input", ""))
    scene_package = load_json(scene_package_path) if scene_package_path else {}
    profile_id = apply_local_model_profile(scene_package)
    print(f"[INFO] Local image model profile: {profile_id}", flush=True)
    paths = output_paths(context)
    if not paths:
        raise RuntimeError("The local diffusion backend did not receive an output image path.")
    primary_output = paths[0]
    prompt, negative_prompt = prompt_from_context(context, scene_package)
    generated_shots = generate_shot_images(context, scene_package, prompt, negative_prompt)
    if generated_shots:
        copy_if_needed(generated_shots[0], primary_output)
    else:
        names = scene_character_names(scene_package)
        references_by_character = reference_images_by_character(scene_package, names)
        reference_paths = reference_image_paths(scene_package, names)
        require_identity_references(names, references_by_character)
        generate_image(
            prompt,
            negative_prompt,
            primary_output,
            deterministic_seed(context, scene_package),
            identity_reference=build_identity_reference_board(reference_paths),
        )

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
