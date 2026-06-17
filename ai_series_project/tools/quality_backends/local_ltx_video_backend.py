#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import tempfile
from datetime import datetime
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


DEFAULT_VIDEO_MODEL_ID = "Lightricks/LTX-Video-0.9.8-13B-distilled"
LEGACY_VIDEO_MODEL_ID = "Lightricks/LTX-Video"
DEFAULT_VIDEO_MODEL_DIR = PROJECT_DIR / "tools" / "quality_models" / "video" / "Lightricks__LTX-Video-0.9.8-13B-distilled"
LEGACY_VIDEO_MODEL_DIR = PROJECT_DIR / "tools" / "quality_models" / "video" / "Lightricks__LTX-Video"
FALLBACK_VIDEO_MODEL_DIRS = [DEFAULT_VIDEO_MODEL_DIR, LEGACY_VIDEO_MODEL_DIR]
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low resolution, blurry, jittery, inconsistent motion, slideshow, still image, "
    "color-grade-only effect, blue filter, warped face, distorted eyes, deformed mouth, extra limbs, "
    "wrong gender, wrong age, different actor, identity drift, inconsistent outfit, inconsistent hairstyle, "
    "duplicate character, missing character, cartoon, anime, watermark, subtitles, text overlay"
)


def clean_text(value: object) -> str:
    return str(value or "").strip()


def truthy_env(name: str, default: bool = False) -> bool:
    value = clean_text(os.environ.get(name, ""))
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def model_dir_ready(candidate: Path) -> bool:
    return candidate.exists() and (candidate / "model_index.json").is_file() and any(candidate.rglob("*.safetensors"))


def normalize_model_dir(candidate: Path) -> Path:
    if not candidate.is_absolute():
        candidate = PROJECT_DIR / candidate
    return candidate.resolve(strict=False)


def configured_model_id(candidate: Path) -> str:
    configured = clean_text(os.environ.get("SERIES_VIDEO_MODEL_ID", ""))
    if configured:
        return configured
    resolved = candidate.resolve(strict=False)
    if resolved == DEFAULT_VIDEO_MODEL_DIR.resolve(strict=False):
        return DEFAULT_VIDEO_MODEL_ID
    if resolved == LEGACY_VIDEO_MODEL_DIR.resolve(strict=False):
        return LEGACY_VIDEO_MODEL_ID
    return candidate.name.replace("__", "/")


def resolve_model_dir() -> Path:
    configured = clean_text(os.environ.get("SERIES_VIDEO_MODEL_DIR", ""))
    candidates = [normalize_model_dir(Path(configured))] if configured else [path.resolve(strict=False) for path in FALLBACK_VIDEO_MODEL_DIRS]
    for candidate in candidates:
        if model_dir_ready(candidate):
            return candidate
    expected = ", ".join(str(candidate) for candidate in candidates)
    if any("LTX-2" in str(candidate) and not (candidate / "model_index.json").exists() for candidate in candidates):
        raise RuntimeError(
            "The selected LTX-2/LTX-2.3 checkpoint directory is not a Diffusers-ready model. "
            "Use an external LTX-2/ComfyUI runner for that model, or use the project default "
            f"{DEFAULT_VIDEO_MODEL_ID}. Checked: {expected}"
        )
    raise RuntimeError(
        "Local LTX video model is not ready. Run 00_prepare_runtime.py without --skip-downloads "
        f"or set SERIES_VIDEO_MODEL_DIR. Expected model_index.json and safetensors files in {expected}."
    )


def compact(value: object, limit: int = 420) -> str:
    text = " ".join(clean_text(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def text_list(value: object, limit: int = 5) -> list[str]:
    if isinstance(value, list):
        return [compact(item, 160) for item in value if clean_text(item)][:limit]
    if isinstance(value, tuple):
        return [compact(item, 160) for item in value if clean_text(item)][:limit]
    text = compact(value, 160)
    return [text] if text else []


def current_shot_package(context: dict[str, Any], scene_package: dict[str, Any]) -> dict[str, Any]:
    shot_id = clean_text(context.get("shot_id", ""))
    shots = scene_package.get("shot_packages", []) if isinstance(scene_package.get("shot_packages", []), list) else []
    for shot in shots:
        if isinstance(shot, dict) and clean_text(shot.get("shot_id", "")) == shot_id:
            return shot
    return {}


def visible_characters(context: dict[str, Any], scene_package: dict[str, Any]) -> list[str]:
    shot = current_shot_package(context, scene_package)
    visible = text_list(shot.get("characters_visible", []), 6)
    if visible:
        return visible
    return text_list(scene_package.get("characters", []), 6)


def set_description(scene_package: dict[str, Any]) -> str:
    set_context = scene_package.get("set_context", {}) if isinstance(scene_package.get("set_context", {}), dict) else {}
    parts = [
        set_context.get("name", ""),
        set_context.get("visual_description", ""),
        set_context.get("lighting", ""),
        set_context.get("camera_axis", ""),
    ]
    key_props = text_list(set_context.get("key_props", []), 6)
    if key_props:
        parts.append("key props: " + ", ".join(key_props))
    return compact(", ".join(part for part in (clean_text(value) for value in parts) if part), 520)


def character_identity_prompt(scene_package: dict[str, Any], characters: list[str]) -> str:
    lock = scene_package.get("character_continuity_lock", {})
    if not isinstance(lock, dict) or not characters:
        return ""
    rows: list[str] = []
    for name in characters[:5]:
        meta = lock.get(name, {}) if isinstance(lock.get(name, {}), dict) else {}
        if not meta:
            continue
        constraints = ["same canonical face from references"]
        if bool(meta.get("outfit_lock", False)):
            constraints.append("same outfit")
        if bool(meta.get("hair_lock", False)):
            constraints.append("same hairstyle")
        if bool(meta.get("voice_lock", False)):
            constraints.append("voice continuity")
        variations = text_list(meta.get("allowed_variations", []), 2)
        if variations:
            constraints.append("allowed variation: " + ", ".join(variations))
        forbidden = text_list(meta.get("forbidden_variations", []), 2)
        if forbidden:
            constraints.append("forbidden: " + ", ".join(forbidden))
        rows.append(f"{name}: {', '.join(constraints)}")
    return "; ".join(rows)


def writer_room_prompt(scene_package: dict[str, Any]) -> str:
    plan = scene_package.get("writer_room_plan", {}) if isinstance(scene_package.get("writer_room_plan", {}), dict) else {}
    if not plan:
        return ""
    keys = [
        "scene_function",
        "emotional_goal",
        "conflict_source",
        "comedy_engine",
        "scene_button",
        "who_drives_scene",
        "who_resists_scene",
        "who_gets_punchline",
        "next_scene_hook",
    ]
    parts = [f"{key.replace('_', ' ')}: {compact(plan.get(key, ''), 120)}" for key in keys if clean_text(plan.get(key, ""))]
    return "; ".join(parts)


def prompt_from_package(context: dict[str, Any], scene_package: dict[str, Any]) -> str:
    video_generation = scene_package.get("video_generation", {}) if isinstance(scene_package.get("video_generation"), dict) else {}
    image_generation = scene_package.get("image_generation", {}) if isinstance(scene_package.get("image_generation"), dict) else {}
    shot = current_shot_package(context, scene_package)
    characters = visible_characters(context, scene_package)
    behavior_constraints = text_list(scene_package.get("behavior_constraints", []), 4)
    dialogue_style = text_list(scene_package.get("dialogue_style_constraints", []), 4)
    shot_dialogue_indices = shot.get("dialogue_line_indices", []) if isinstance(shot.get("dialogue_line_indices", []), list) else []
    dialogue_meta = scene_package.get("dialogue_line_metadata", []) if isinstance(scene_package.get("dialogue_line_metadata", []), list) else []
    shot_dialogue: list[str] = []
    for item in dialogue_meta:
        if not isinstance(item, dict):
            continue
        index = item.get("line_index", item.get("index", None))
        if shot_dialogue_indices and index not in shot_dialogue_indices:
            continue
        speaker = clean_text(item.get("speaker", "") or item.get("speaker_name", ""))
        function = clean_text(item.get("dialogue_function", ""))
        action = clean_text(item.get("physical_action", ""))
        expression = clean_text(item.get("facial_expression", ""))
        if speaker or function or action or expression:
            shot_dialogue.append(compact(f"{speaker} {function} {action} {expression}", 180))
        if len(shot_dialogue) >= 4:
            break
    parts = [
        clean_text(video_generation.get("positive_prompt", "")),
        clean_text(video_generation.get("prompt", "")),
        clean_text(image_generation.get("positive_prompt", "")),
        clean_text(image_generation.get("prompt", "")),
        clean_text(context.get("scene_title", "")),
        clean_text(context.get("shot_type", "")),
        clean_text(context.get("shot_purpose", "")),
        clean_text(context.get("camera_angle", "")),
        clean_text(context.get("camera_movement", "")),
        f"visible characters: {', '.join(characters)}" if characters else "",
        character_identity_prompt(scene_package, characters),
        set_description(scene_package),
        writer_room_prompt(scene_package),
        clean_text(scene_package.get("scene_function", "")),
        clean_text(scene_package.get("scene_purpose", "")),
        clean_text(scene_package.get("conflict", "")),
        "behavior: " + "; ".join(behavior_constraints) if behavior_constraints else "",
        "dialogue style: " + "; ".join(dialogue_style) if dialogue_style else "",
        "shot acting beats: " + "; ".join(shot_dialogue) if shot_dialogue else "",
        clean_text(scene_package.get("summary", "")),
        (
            "source-series faithful live-action TV episode shot, match the trained show's lighting, lensing, "
            "blocking, set design, color palette and camera grammar; stable canonical character identities from "
            "the provided references; no gender swaps; natural actor motion; dialogue-timed reactions; "
            "multi-camera sitcom continuity; realistic faces and hands; production-ready motion video"
        ),
    ]
    return ", ".join(compact(part, 520) for part in parts if clean_text(part))


def negative_prompt_from_package(scene_package: dict[str, Any]) -> str:
    video_generation = scene_package.get("video_generation", {}) if isinstance(scene_package.get("video_generation"), dict) else {}
    image_generation = scene_package.get("image_generation", {}) if isinstance(scene_package.get("image_generation"), dict) else {}
    negatives = [
        clean_text(video_generation.get("negative_prompt", "")),
        clean_text(image_generation.get("negative_prompt", "")),
    ]
    style_constraints = video_generation.get("style_constraints", {}) if isinstance(video_generation.get("style_constraints", {}), dict) else {}
    negatives.extend(text_list(style_constraints.get("negative", []), 8))
    negatives.append(DEFAULT_NEGATIVE_PROMPT)
    seen: set[str] = set()
    unique: list[str] = []
    for item in negatives:
        text = clean_text(item)
        if text and text.lower() not in seen:
            seen.add(text.lower())
            unique.append(text)
    return ", ".join(unique)


def pipeline_accepts_argument(pipeline: Any, name: str) -> bool:
    try:
        signature = inspect.signature(pipeline.__call__)
    except Exception:
        return False
    if name in signature.parameters:
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def deterministic_seed(context: dict[str, Any]) -> int:
    raw = "|".join(clean_text(context.get(key, "")) for key in ("episode_id", "scene_id", "shot_id", "scene_title", "runner_kind"))
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16)


def target_size() -> tuple[int, int]:
    width = int(float(os.environ.get("SERIES_VIDEO_WIDTH", "1216") or "1216"))
    height = int(float(os.environ.get("SERIES_VIDEO_HEIGHT", "704") or "704"))
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


def torch_dtype_from_env(torch: Any, cuda_ready: bool) -> Any:
    requested = clean_text(os.environ.get("SERIES_VIDEO_DTYPE", "")).lower()
    if not requested:
        if cuda_ready and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            requested = "bfloat16"
        elif cuda_ready:
            requested = "float16"
        else:
            requested = "float32"
    mapping = {
        "bf16": getattr(torch, "bfloat16", torch.float32),
        "bfloat16": getattr(torch, "bfloat16", torch.float32),
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping.get(requested, torch.float16 if cuda_ready else torch.float32)


def generate_ltx_video(context: dict[str, Any], scene_package: dict[str, Any], output_path: Path) -> dict[str, Any]:
    try:
        import torch
        import diffusers
        from diffusers.utils import export_to_video
    except Exception as exc:
        raise RuntimeError(
            "The real local video backend requires diffusers and torch. Run 00_prepare_runtime.py first."
        ) from exc

    pipeline_cls = (
        getattr(diffusers, "LTXImageToVideoPipeline", None)
        or getattr(diffusers, "LTXPipeline", None)
        or getattr(diffusers, "DiffusionPipeline", None)
    )
    if pipeline_cls is None:
        raise RuntimeError(
            "The installed diffusers version does not expose an LTX-compatible DiffusionPipeline. "
            "Run 00_prepare_runtime.py to upgrade the project runtime."
        )

    model_dir = resolve_model_dir()
    cuda_ready = bool(torch.cuda.is_available())
    dtype = torch_dtype_from_env(torch, cuda_ready)
    device = "cuda" if cuda_ready else "cpu"
    width, height = target_size()
    fps = int(float(os.environ.get("SERIES_VIDEO_FPS", "30") or "30"))
    steps = int(float(os.environ.get("SERIES_VIDEO_INFERENCE_STEPS", "30") or "30"))
    guidance = float(os.environ.get("SERIES_VIDEO_GUIDANCE_SCALE", "3.0") or "3.0")

    pipeline = pipeline_cls.from_pretrained(str(model_dir), local_files_only=True, torch_dtype=dtype)
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    if hasattr(getattr(pipeline, "vae", None), "enable_tiling"):
        pipeline.vae.enable_tiling()
    if hasattr(getattr(pipeline, "vae", None), "enable_slicing"):
        pipeline.vae.enable_slicing()
    pipeline = pipeline.to(device)
    generator = torch.Generator(device=device).manual_seed(deterministic_seed(context))
    prompt = prompt_from_package(context, scene_package)
    negative_prompt = negative_prompt_from_package(scene_package)
    frames_requested = frame_count(scene_package, fps)

    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_frames": frames_requested,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "generator": generator,
    }
    if negative_prompt and pipeline_accepts_argument(pipeline, "negative_prompt"):
        kwargs["negative_prompt"] = negative_prompt
    primary_frame = existing_path(context.get("primary_frame", ""))
    if primary_frame is not None and pipeline_accepts_argument(pipeline, "image"):
        kwargs["image"] = Image.open(primary_frame).convert("RGB").resize((width, height))
    if not pipeline_accepts_argument(pipeline, "guidance_scale"):
        kwargs.pop("guidance_scale", None)

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
    return {
        "model_id": configured_model_id(model_dir),
        "model_dir": str(model_dir),
        "pipeline_class": pipeline.__class__.__name__,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "width": width,
        "height": height,
        "fps": fps,
        "num_frames": kwargs.get("num_frames", frames_requested),
        "num_inference_steps": steps,
        "guidance_scale": guidance if "guidance_scale" in kwargs else None,
        "prompt": prompt,
        "negative_prompt": negative_prompt if "negative_prompt" in kwargs else "",
        "primary_frame": str(primary_frame) if primary_frame else "",
    }


def shot_packages(scene_package: dict[str, Any]) -> list[dict[str, Any]]:
    raw = scene_package.get("shot_packages", []) if isinstance(scene_package.get("shot_packages", []), list) else []
    return [
        row
        for row in raw
        if isinstance(row, dict)
        and clean_text((row.get("target_outputs", {}) if isinstance(row.get("target_outputs", {}), dict) else {}).get("video_clip", ""))
    ]


def shot_frame_path(shot: dict[str, Any]) -> str:
    outputs = shot.get("target_outputs", {}) if isinstance(shot.get("target_outputs", {}), dict) else {}
    return clean_text(outputs.get("primary_frame", ""))


def shot_context(context: dict[str, Any], shot: dict[str, Any]) -> dict[str, Any]:
    return {
        **context,
        "shot_id": clean_text(shot.get("shot_id", "")),
        "shot_type": clean_text(shot.get("shot_type", "")),
        "shot_purpose": clean_text(shot.get("purpose", "")),
        "camera_angle": clean_text(shot.get("camera_angle", "")),
        "camera_movement": clean_text(shot.get("camera_movement", "")),
        "primary_frame": shot_frame_path(shot) or context.get("primary_frame", ""),
    }


def shot_scene_package(scene_package: dict[str, Any], shot: dict[str, Any]) -> dict[str, Any]:
    package = dict(scene_package)
    package["duration_seconds"] = float(shot.get("duration_seconds", scene_package.get("duration_seconds", 5.0)) or 5.0)
    package["summary"] = ", ".join(
        part
        for part in [
            clean_text(scene_package.get("summary", "")),
            clean_text(shot.get("purpose", "")),
            clean_text(shot.get("shot_type", "")),
        ]
        if part
    )
    return package


def write_shot_manifest(shot: dict[str, Any], output_path: Path, generation_metadata: dict[str, Any] | None = None) -> None:
    outputs = shot.get("target_outputs", {}) if isinstance(shot.get("target_outputs", {}), dict) else {}
    manifest_text = clean_text(outputs.get("video_manifest", "")) or clean_text(outputs.get("manifest", ""))
    if not manifest_text:
        return
    manifest_path = Path(manifest_text)
    ensure_parent(str(manifest_path))
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest() if output_path.exists() else ""
    generation_metadata = generation_metadata if isinstance(generation_metadata, dict) else {}
    payload = {
        "task_id": f"{clean_text(shot.get('shot_id', 'shot'))}_local_ltx_video",
        "scene_id": clean_text(shot.get("scene_id", "")),
        "shot_id": clean_text(shot.get("shot_id", "")),
        "task_type": "video",
        "backend": "local_ltx_video_backend",
        "inputs": {
            "shot_type": clean_text(shot.get("shot_type", "")),
            "camera_angle": clean_text(shot.get("camera_angle", "")),
            "camera_movement": clean_text(shot.get("camera_movement", "")),
            "characters_visible": text_list(shot.get("characters_visible", []), 8),
            "model_id": clean_text(generation_metadata.get("model_id", "")),
            "model_dir": clean_text(generation_metadata.get("model_dir", "")),
            "pipeline_class": clean_text(generation_metadata.get("pipeline_class", "")),
            "device": clean_text(generation_metadata.get("device", "")),
            "dtype": clean_text(generation_metadata.get("dtype", "")),
            "width": generation_metadata.get("width", 0),
            "height": generation_metadata.get("height", 0),
            "fps": generation_metadata.get("fps", 0),
            "num_frames": generation_metadata.get("num_frames", 0),
            "num_inference_steps": generation_metadata.get("num_inference_steps", 0),
            "guidance_scale": generation_metadata.get("guidance_scale", None),
            "positive_prompt": clean_text(generation_metadata.get("prompt", "")),
            "negative_prompt": clean_text(generation_metadata.get("negative_prompt", "")),
            "primary_frame": clean_text(generation_metadata.get("primary_frame", "")),
        },
        "outputs": {"video_clip": str(output_path)},
        "output_hashes": {str(output_path): digest} if digest else {},
        "status": "success" if digest else "failed",
        "fallback_used": False,
        "placeholder_used": False,
        "finished_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def concat_video_clips(clips: list[Path], output_path: Path) -> None:
    if not clips:
        raise RuntimeError("No generated shot clips are available for scene assembly.")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        concat_path = Path(handle.name)
        for clip in clips:
            escaped = str(clip).replace("'", "''")
            handle.write(f"file '{escaped}'\n")
    try:
        ensure_parent(str(output_path))
        command = [
            find_project_local_ffmpeg(),
            "-hide_banner",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"Shot clip concat failed. {(completed.stdout or '')[-1400:]}")
    finally:
        concat_path.unlink(missing_ok=True)


def generate_ltx_shots(context: dict[str, Any], scene_package: dict[str, Any], scene_output: Path) -> bool:
    clips: list[Path] = []
    resume_existing = truthy_env("SERIES_VIDEO_RESUME_SHOTS")
    for shot in shot_packages(scene_package):
        outputs = shot.get("target_outputs", {}) if isinstance(shot.get("target_outputs", {}), dict) else {}
        clip = Path(clean_text(outputs.get("video_clip", "")))
        manifest_text = clean_text(outputs.get("video_manifest", "")) or clean_text(outputs.get("manifest", ""))
        manifest_path = Path(manifest_text) if manifest_text else Path()
        completed_shot = (
            resume_existing
            and clip.is_file()
            and clip.stat().st_size > 0
            and manifest_text
            and manifest_path.is_file()
            and manifest_path.stat().st_size > 0
        )
        if completed_shot:
            print(f"[INFO] Resuming video package: keeping completed shot {shot.get('shot_id', clip.stem)}", flush=True)
        else:
            metadata = generate_ltx_video(shot_context(context, shot), shot_scene_package(scene_package, shot), clip)
            write_shot_manifest(shot, clip, metadata)
        clips.append(clip)
    if not clips:
        return False
    concat_video_clips(clips, scene_output)
    return True


def main() -> int:
    context = load_backend_context()
    scene_package = load_json(clean_text(context.get("scene_package", "")))
    if not scene_package:
        raise RuntimeError("Could not load scene package for the local LTX video backend.")
    output_text = clean_text(context.get("scene_video", ""))
    if not output_text:
        raise RuntimeError("The local LTX video backend did not receive a scene video output path.")
    output_path = Path(output_text)

    if not generate_ltx_shots(context, scene_package, output_path):
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
