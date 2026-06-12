#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from backend_common import (
    PROJECT_DIR,
    copy_if_needed,
    ensure_parent,
    load_backend_context,
    load_json,
    print_runtime_error,
    resolve_stored_project_path,
)


DEFAULT_IMAGE_MODEL_DIR = PROJECT_DIR / "tools" / "quality_models" / "image" / "stabilityai__stable-diffusion-xl-base-1.0"
DEFAULT_IDENTITY_ADAPTER_DIR = PROJECT_DIR / "tools" / "quality_models" / "image" / "h94__IP-Adapter"
IDENTITY_ADAPTER_WEIGHT = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
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
    people_free_intro = (
        clean_text(image_generation.get("mode", "")) == "generated_season_intro_keyframes"
        and not characters
    )
    subject = (
        f"canonical cast identities available for {', '.join(characters[:3])}"
        if characters
        else "empty recurring set and signature props, no people visible"
        if people_free_intro
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
            "cropped faces, deformed face, warped face, melted face, asymmetrical eyes, crossed eyes, "
            "wrong person, identity drift, face swap, gender swap, age swap, wrong hairstyle, wrong body proportions, "
            "merged people, duplicate characters, extra people, missing people, distorted hands, unreadable text, "
            "watermark, low quality, blurry, unfinished, blue placeholder frame"
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


def load_pipeline(require_identity_adapter: bool) -> tuple[Any, dict[str, Any]]:
    global _PIPELINE, _PIPELINE_META
    if _PIPELINE is not None:
        if require_identity_adapter and not bool(_PIPELINE_META.get("identity_adapter_loaded", False)):
            raise RuntimeError(
                "Canonical face references are present, but the SDXL identity adapter is not loaded. "
                "Run 00_prepare_runtime.py without --skip-downloads."
            )
        return _PIPELINE, dict(_PIPELINE_META)
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
    steps = int(float(os.environ.get("SERIES_IMAGE_INFERENCE_STEPS", "36") or "36"))
    guidance = float(os.environ.get("SERIES_IMAGE_GUIDANCE_SCALE", "6.0") or "6.0")
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
    adapter_dir = resolve_identity_adapter_dir()
    adapter_loaded = False
    if identity_adapter_ready(adapter_dir):
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
    elif require_identity_adapter:
        raise RuntimeError(
            "The project-local SDXL identity adapter is incomplete. Run 00_prepare_runtime.py without "
            f"--skip-downloads. Expected {adapter_dir / 'sdxl_models' / IDENTITY_ADAPTER_WEIGHT} and "
            f"{adapter_dir / 'models' / 'image_encoder' / 'config.json'}."
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
        "model_dir": str(model_dir),
        "identity_adapter_dir": str(adapter_dir),
        "identity_adapter_loaded": adapter_loaded,
    }
    return pipeline, dict(_PIPELINE_META)


def build_identity_reference_board(reference_paths: list[Path]) -> Image.Image | None:
    usable: list[Image.Image] = []
    for path in reference_paths[:4]:
        try:
            with Image.open(path) as source:
                usable.append(ImageOps.fit(source.convert("RGB"), (256, 256), method=Image.Resampling.LANCZOS))
        except Exception:
            continue
    if not usable:
        return None
    board = Image.new("RGB", (512, 512), (127, 127, 127))
    positions = [(0, 0), (256, 0), (0, 256), (256, 256)]
    for image, position in zip(usable, positions):
        board.paste(image, position)
    if len(usable) == 1:
        board.paste(usable[0], (256, 0))
        board.paste(usable[0], (0, 256))
        board.paste(usable[0], (256, 256))
    return board


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
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "generator": generator,
    }
    if identity_reference is not None:
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
    target = target_dir / f"{source.stem}_sdxl_alt01{source.suffix or '.png'}"
    copy_if_needed(source, target)


def shot_prompt(prompt: str, shot: dict[str, Any]) -> str:
    visible_characters = clean_text_list(shot.get("characters_visible", []))
    explicit_people_free = isinstance(shot.get("characters_visible"), list) and not visible_characters
    parts = [
        prompt,
        clean_text(shot.get("shot_type", "")),
        clean_text(shot.get("camera_angle", "")),
        clean_text(shot.get("camera_movement", "")),
        clean_text(shot.get("purpose", "")),
        f"exactly {len(visible_characters)} visible named characters: {', '.join(visible_characters)}; preserve each supplied identity separately"
        if visible_characters
        else "no people visible, environment and props only"
        if explicit_people_free
        else "",
    ]
    return ", ".join(part for part in parts if part)


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
        for value in [*portrait_values[:2], *lock_values[:2], *context_values[:1]]:
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
    payload = {
        "task_id": f"{clean_text(shot.get('shot_id', 'shot'))}_local_sdxl_image",
        "scene_id": clean_text(shot.get("scene_id", "")),
        "shot_id": clean_text(shot.get("shot_id", "")),
        "task_type": "image",
        "backend": "local_diffusion_image_backend",
        "inputs": {
            "prompt": prompt,
            "seed": seed,
            "identity_reference_images": [str(path) for path in reference_paths],
            "identity_seed_scope": "episode_scene_character_set",
        },
        "outputs": {"primary_frame": str(output_path)},
        "output_hashes": {str(output_path): digest} if digest else {},
        "status": "success" if digest else "failed",
        "fallback_used": False,
        "placeholder_used": False,
        "identity_conditioning": {
            "mode": "sdxl_ip_adapter_plus_face",
            "adapter_loaded": bool(pipeline_meta.get("identity_adapter_loaded", False)),
            "adapter_dir": clean_text(pipeline_meta.get("identity_adapter_dir", "")),
            "reference_count": len(reference_paths),
            "characters_conditioned": reference_characters,
            "missing_characters": missing_characters,
        },
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
    for shot in shot_packages(scene_package):
        outputs = shot.get("target_outputs", {}) if isinstance(shot.get("target_outputs", {}), dict) else {}
        output_path = Path(clean_text(outputs.get("primary_frame", "")))
        visible_value = shot.get("characters_visible")
        visible_characters = (
            clean_text_list(visible_value)
            if isinstance(visible_value, list)
            else scene_character_names(scene_package)
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
