#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "ai_series_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse
import difflib
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from support_scripts.pipeline_common import (
    DistributedLeaseHeartbeat,
    acquire_distributed_lease,
    add_shared_worker_arguments,
    distributed_heartbeat_interval_seconds,
    distributed_lease_ttl_seconds,
    distributed_step_runtime_root,
    LiveProgressReporter,
    PROJECT_ROOT,
    detect_tool,
    prepare_quality_backend_assets_runtime,
    ensure_quality_first_ready,
    episode_quality_assessment,
    error,
    external_tool_command,
    generated_episode_completion_summary,
    headline,
    info,
    latest_matching_file,
    load_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    normalize_language_code,
    ok,
    read_json,
    rerun_in_runtime,
    run_external_backend_runner,
    scene_quality_assessment,
    resolve_project_path,
    resolve_stored_project_path,
    release_distributed_lease,
    shared_worker_id_for_args,
    shared_workers_enabled_for_args,
    write_json,
    write_text,
    write_realtime_preview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a draft preview, a final voiced storyboard episode, and a backend-ready full-episode production package."
    )
    parser.add_argument("--episode-id", help="Target a specific episode ID such as episode_09 or folge_09.")
    parser.add_argument("--force", action="store_true", help="Recreate existing draft and final renders.")
    add_shared_worker_arguments(parser)
    return parser.parse_args()


def find_latest_shotlist(shotlist_dir: Path) -> Path | None:
    episode_id = os.environ.get("SERIES_RENDER_EPISODE", "").strip()
    if episode_id:
        candidate = shotlist_dir / f"{episode_id}.json"
        if candidate.exists():
            return candidate
    return latest_matching_file(shotlist_dir, "*.json")


def storyboard_assets_root(cfg: dict, episode_id: str) -> Path:
    raw_root = str((cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}).get("storyboard_assets", "generation/storyboard_assets"))
    candidate = Path(raw_root)
    base_root = candidate if candidate.is_absolute() else resolve_project_path(raw_root)
    return base_root / episode_id


def episode_production_package_root(cfg: dict, episode_id: str) -> Path:
    raw_root = str((cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}).get("final_episode_packages", "generation/final_episode_packages"))
    candidate = Path(raw_root)
    base_root = candidate if candidate.is_absolute() else resolve_project_path(raw_root)
    return base_root / episode_id


def episode_delivery_bundle_root(cfg: dict, episode_id: str) -> Path:
    raw_root = str((cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}).get("episode_deliveries", "generation/renders/deliveries"))
    candidate = Path(raw_root)
    base_root = candidate if candidate.is_absolute() else resolve_project_path(raw_root)
    return base_root / episode_id


def latest_episode_delivery_bundle_root(cfg: dict) -> Path:
    raw_root = str((cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}).get("episode_deliveries", "generation/renders/deliveries"))
    candidate = Path(raw_root)
    base_root = candidate if candidate.is_absolute() else resolve_project_path(raw_root)
    return base_root / "latest"


def production_scene_frame_candidates(package_root: Path, scene_id: str) -> list[tuple[str, Path]]:
    scene_slug = production_scene_slug(scene_id)
    return [
        ("generated_episode_frame", package_root / "images" / scene_slug / "frame_0001.png"),
        ("generated_episode_frame", package_root / "images" / scene_slug / "frame.png"),
        ("generated_episode_frame", package_root / "images" / scene_slug / "keyframe.png"),
        ("generated_episode_frame", package_root / "videos" / scene_slug / "poster.png"),
        ("generated_episode_frame", package_root / "videos" / scene_slug / "preview.jpg"),
        ("generated_episode_frame", package_root / "lipsync" / scene_slug / "poster.png"),
    ]


def production_scene_video_candidates(package_root: Path, scene_id: str) -> list[tuple[str, Path]]:
    scene_slug = production_scene_slug(scene_id)
    return [
        ("generated_lipsync_video", package_root / "lipsync" / scene_slug / f"{scene_slug}_lipsync.mp4"),
        ("generated_lipsync_video", package_root / "lipsync" / scene_slug / "lipsync.mp4"),
        ("generated_scene_video", package_root / "videos" / scene_slug / f"{scene_slug}.mp4"),
        ("generated_scene_video", package_root / "videos" / scene_slug / "scene.mp4"),
        ("generated_scene_video", package_root / "videos" / scene_slug / "clip.mp4"),
    ]


def scene_frame_candidates(assets_root: Path, scene_id: str) -> list[tuple[str, Path]]:
    return [
        ("backend_frame", assets_root / scene_id / "frame.png"),
        ("backend_preview", assets_root / scene_id / "preview.jpg"),
        ("storyboard_asset", assets_root / f"{scene_id}.png"),
        ("storyboard_asset", assets_root / f"{scene_id}.jpg"),
    ]


def first_existing_scene_frame(assets_root: Path, scene_id: str) -> tuple[str, Path] | None:
    for source_type, candidate in scene_frame_candidates(assets_root, scene_id):
        if candidate.exists() and candidate.stat().st_size > 0:
            return source_type, candidate
    return None


def first_existing_production_scene_video(package_root: Path, scene_id: str) -> tuple[str, Path] | None:
    for source_type, candidate in production_scene_video_candidates(package_root, scene_id):
        if candidate.exists() and candidate.stat().st_size > 0:
            return source_type, candidate
    return None


def storyboard_backend_scene_video_candidates(assets_root: Path, scene_id: str) -> list[tuple[str, Path]]:
    return [
        ("storyboard_backend_scene_video", assets_root / scene_id / "clip.mp4"),
        ("storyboard_backend_scene_video", assets_root / scene_id / f"{scene_id}.mp4"),
        ("storyboard_backend_scene_video", assets_root / scene_id / "scene.mp4"),
    ]


def first_existing_storyboard_backend_scene_video(assets_root: Path, scene_id: str) -> tuple[str, Path] | None:
    for source_type, candidate in storyboard_backend_scene_video_candidates(assets_root, scene_id):
        if candidate.exists() and candidate.stat().st_size > 0:
            return source_type, candidate
    return None


def first_existing_scene_video_source(package_root: Path, assets_root: Path, scene_id: str) -> tuple[str, Path] | None:
    return first_existing_production_scene_video(package_root, scene_id) or first_existing_storyboard_backend_scene_video(
        assets_root,
        scene_id,
    )


def generated_storyboard_scene_frame(cfg: dict, episode_id: str, scene_id: str) -> Path | None:
    assets_root = storyboard_assets_root(cfg, episode_id)
    match = first_existing_scene_frame(assets_root, scene_id)
    return match[1] if match else None


def fit_image(path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)


def theme_color(scene: dict, scene_index: int) -> tuple[int, int, int]:
    source = " ".join(
        [
            str(scene.get("title", "")),
            str(scene.get("beat", "")),
            str(scene.get("mood", "")),
            str(scene.get("summary", "")),
        ]
    ).lower()
    if not source:
        source = f"scene-{scene_index}"
    seed = sum(ord(char) for char in source)
    return (56 + (seed % 120), 72 + ((seed // 5) % 112), 96 + ((seed // 9) % 104))


def draw_multiline_block(
    draw: ImageDraw.ImageDraw,
    *,
    lines: list[str],
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    spacing: int,
) -> int:
    current_y = y
    for line in lines:
        draw.text((x, current_y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, current_y), line, font=font)
        current_y = bbox[3] + spacing
    return current_y


def placeholder_scene_frame(scene: dict, width: int, height: int, scene_index: int) -> Image.Image:
    accent = theme_color(scene, scene_index)
    image = Image.new("RGB", (width, height), (18, 24, 32))
    draw = ImageDraw.Draw(image)
    overlay = Image.new("RGBA", (width, height), accent + (0,))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle((0, 0, width, height), fill=accent + (42,))
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()
    draw.rectangle((60, 60, width - 60, height - 60), outline=(230, 232, 236), width=2)
    draw.rectangle((60, height - 230, width - 60, height - 60), fill=(10, 12, 16))
    draw.text((90, 90), f"{scene.get('scene_id', 'scene')} | {scene.get('title', 'Storyboard Scene')}", font=title_font, fill=(245, 245, 245))

    characters = ", ".join(scene.get("characters", []) or []) or "No characters assigned"
    summary = str(scene.get("summary", "")).strip() or "No scene summary available."
    mood = str(scene.get("mood", "")).strip() or "n/a"
    location = str(scene.get("location", "")).strip() or "n/a"
    dialogue_lines = scene.get("dialogue_lines", []) if isinstance(scene.get("dialogue_lines", []), list) else []
    excerpt = dialogue_lines[:3] or ["No dialogue lines available yet."]
    wrapped_summary = textwrap.wrap(summary, width=78)[:3]
    wrapped_dialogue: list[str] = []
    for line in excerpt:
        wrapped_dialogue.extend(textwrap.wrap(str(line), width=72)[:2] or [str(line)])
    content_lines = [
        f"Characters: {characters}",
        f"Location: {location}",
        f"Mood: {mood}",
        "",
        "Summary:",
        *wrapped_summary,
        "",
        "Dialogue Preview:",
        *wrapped_dialogue,
    ]
    draw_multiline_block(
        draw,
        lines=content_lines,
        x=90,
        y=height - 210,
        font=body_font,
        fill=(240, 240, 240),
        spacing=5,
    )
    return image


def build_scene_frame(scene: dict, scene_index: int, cfg: dict, episode_id: str, assets_root: Path, width: int, height: int) -> tuple[Image.Image, dict]:
    scene_id = str(scene.get("scene_id", "")).strip() or f"scene_{scene_index + 1:04d}"
    package_root = episode_production_package_root(cfg, episode_id)
    source = None
    for source_type, candidate in production_scene_frame_candidates(package_root, scene_id):
        if candidate.exists() and candidate.stat().st_size > 0:
            source = (source_type, candidate)
            break
    if source is None:
        source = first_existing_scene_frame(assets_root, scene_id)
    meta = {
        "scene_id": scene_id,
        "asset_source_type": "placeholder",
        "asset_source_path": "",
    }
    if source is None:
        return placeholder_scene_frame(scene, width, height, scene_index), meta
    source_type, source_path = source
    meta["asset_source_type"] = source_type
    meta["asset_source_path"] = str(source_path)
    return fit_image(source_path, width, height), meta


def compose_scene_card_from_base(scene: dict, scene_index: int, base: Image.Image, meta: dict, width: int, height: int) -> Image.Image:
    image = base.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, height - 190, width, height), fill=(8, 10, 14, 196))
    draw.rectangle((0, 0, width, 92), fill=(8, 10, 14, 182))
    accent = theme_color(scene, scene_index)
    draw.rectangle((0, 0, width, 8), fill=accent + (255,))
    merged = Image.alpha_composite(image, overlay).convert("RGB")

    draw = ImageDraw.Draw(merged)
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()
    scene_id = meta["scene_id"]
    title = str(scene.get("title", "")).strip() or "Storyboard Scene"
    characters = ", ".join(scene.get("characters", []) or []) or "No characters"
    summary = str(scene.get("summary", "")).strip()
    dialogue_lines = scene.get("dialogue_lines", []) if isinstance(scene.get("dialogue_lines", []), list) else []
    dialogue_preview = " | ".join(str(line) for line in dialogue_lines[:2]) or "No dialogue preview"

    draw.text((40, 26), f"{scene_id} | {title}", font=title_font, fill=(248, 248, 248))
    footer_lines = [
        f"Characters: {characters}",
        f"Summary: {textwrap.shorten(summary, width=120, placeholder='...') if summary else 'n/a'}",
        f"Dialogue: {textwrap.shorten(dialogue_preview, width=120, placeholder='...')}",
        f"Source: {meta['asset_source_type']}",
    ]
    draw_multiline_block(
        draw,
        lines=footer_lines,
        x=40,
        y=height - 165,
        font=body_font,
        fill=(242, 242, 242),
        spacing=5,
    )
    return merged


def compose_scene_card(scene: dict, scene_index: int, cfg: dict, episode_id: str, assets_root: Path, width: int, height: int) -> tuple[Image.Image, dict]:
    base, meta = build_scene_frame(scene, scene_index, cfg, episode_id, assets_root, width, height)
    return compose_scene_card_from_base(scene, scene_index, base, meta, width, height), meta


def title_card(width: int, height: int, episode_id: str, display_title: str, episode_title: str) -> Image.Image:
    image = Image.new("RGB", (width, height), ImageColor.getrgb("#0f1720"))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, height), fill=(15, 23, 32))
    draw.rectangle((80, 80, width - 80, height - 80), outline=(240, 244, 248), width=2)
    font = ImageFont.load_default()
    lines = [
        "AI Series Episode Render",
        "",
        display_title or episode_id,
        episode_title or "",
        "",
        "Storyboard-driven local final episode render",
    ]
    draw_multiline_block(draw, lines=lines, x=120, y=150, font=font, fill=(248, 248, 248), spacing=10)
    return image


def closing_card(width: int, height: int, display_title: str, scene_count: int) -> Image.Image:
    image = Image.new("RGB", (width, height), ImageColor.getrgb("#121212"))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, height), fill=(18, 18, 18))
    font = ImageFont.load_default()
    lines = [
        display_title or "Final Episode",
        "",
        f"Rendered storyboard scenes: {scene_count}",
        "Draft and final preview written locally",
    ]
    draw_multiline_block(draw, lines=lines, x=130, y=220, font=font, fill=(244, 244, 244), spacing=12)
    return image


def safe_duration_seconds(scene: dict) -> float:
    duration = float(scene.get("estimated_runtime_seconds", 0.0) or 0.0)
    dialogue_lines = scene.get("dialogue_lines", []) if isinstance(scene.get("dialogue_lines", []), list) else []
    dialogue_text = " ".join(clean_text(line) for line in dialogue_lines if clean_text(line))
    dialogue_tokens = re.findall(r"\w+", dialogue_text, flags=re.UNICODE)
    dialogue_runtime = 0.0
    if dialogue_tokens:
        dialogue_runtime = max(
            0.0,
            (len(dialogue_tokens) / 150.0) * 60.0 + max(0.8, len(dialogue_lines) * 0.35),
        )
    target_duration = max(duration if duration > 0 else 0.0, dialogue_runtime, 6.0)
    return max(6.0, min(75.0, target_duration))


def render_output_ready(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def run_media_command(command: list[object]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        external_tool_command(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def scene_dialogue_source(scene: dict, line_index: int) -> dict:
    sources = scene.get("dialogue_sources", []) if isinstance(scene.get("dialogue_sources", []), list) else []
    if 0 <= line_index < len(sources) and isinstance(sources[line_index], dict):
        return sources[line_index]
    return {}


def text_similarity_score(expected_text: str, candidate_text: str) -> tuple[float, float]:
    left_tokens = {token.lower() for token in str(expected_text).replace(".", " ").replace(",", " ").split() if token.strip()}
    right_tokens = {token.lower() for token in str(candidate_text).replace(".", " ").replace(",", " ").split() if token.strip()}
    overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    ratio = difflib.SequenceMatcher(None, str(expected_text).lower(), str(candidate_text).lower()).ratio()
    return (ratio * 0.65) + (overlap * 0.35), overlap


def select_retrieval_segment(character_name: str, line_text: str, library: dict, cfg: dict) -> dict | None:
    if not bool(cfg.get("enable_original_line_reuse", False)):
        return None
    threshold = float(cfg.get("original_line_similarity_threshold", 0.74) or 0.74)
    min_overlap = float(cfg.get("original_line_min_token_overlap", 0.34) or 0.34)
    best_match: dict | None = None
    best_score = -1.0
    for entry in library.get(character_name, []) if isinstance(library.get(character_name, []), list) else []:
        score, overlap = text_similarity_score(line_text, str(entry.get("text", "")))
        if score < threshold or overlap < min_overlap:
            continue
        if score > best_score:
            best_score = score
            best_match = dict(entry)
            best_match["match_score"] = round(score, 4)
            best_match["token_overlap"] = round(overlap, 4)
    return best_match


def clean_text(value: object) -> str:
    return str(value or "").strip()


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


STRICT_GENERATED_VIDEO_TYPES = {"generated_scene_video", "generated_lipsync_video"}
STRICT_VOICE_BACKENDS = {"xtts", "xtts_voice_clone", "voice_clone"}
LOCAL_COMPOSED_VIDEO_TYPES = {
    "auto_generated_multishot_video",
    "local_motion_fallback",
    "storyboard_motion_fallback",
    "generated_episode_frame",
}

SCENE_RUNNER_OUTPUT_KEYS = {
    "image_generation": ("primary_frame", "layered_storyboard_frame"),
    "video_generation": ("scene_video", "video_preview_frame", "video_poster_frame"),
    "voice_clone": ("scene_dialogue_audio",),
    "lip_sync": ("lipsync_video", "lipsync_poster_frame"),
}


def strict_original_episode_outputs_required(cfg: dict) -> bool:
    release_cfg = cfg.get("release_mode", {}) if isinstance(cfg.get("release_mode"), dict) else {}
    generation_cfg = cfg.get("generation", {}) if isinstance(cfg.get("generation"), dict) else {}
    quality_mode = clean_text(generation_cfg.get("quality_mode", "")).lower()
    return bool(release_cfg.get("force_finished_episode_generation", True)) or quality_mode == "original_episode_quality_first"


def local_motion_fallback_allowed(cfg: dict, render_cfg: dict) -> bool:
    if "allow_local_motion_fallback" in render_cfg:
        return bool(render_cfg.get("allow_local_motion_fallback", False))
    return not strict_original_episode_outputs_required(cfg)


def audio_backend_is_voice_clone(audio_backend: object) -> bool:
    value = clean_text(audio_backend).lower()
    return value in STRICT_VOICE_BACKENDS or value.endswith("_voice_clone")


def clean_text_list(value: object, limit: int | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    rows = [clean_text(item) for item in value if clean_text(item)]
    return rows if limit is None else rows[:limit]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path_value: object) -> str:
    path = resolve_stored_project_path(path_value)
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lipsync_backend_profile(cfg: dict) -> dict:
    lipsync_cfg = cfg.get("lipsync_backends", {}) if isinstance(cfg.get("lipsync_backends", {}), dict) else {}
    preferred_order = clean_text_list(lipsync_cfg.get("preferred_order", [])) or ["musetalk", "latentsync", "wav2lip"]
    min_sync_score = max(0.0, min(1.0, safe_float(lipsync_cfg.get("min_sync_score", 0.75), 0.75)))
    candidates: list[dict] = []
    for backend in preferred_order:
        backend_key = backend.lower()
        env_name = f"SERIES_{backend_key.upper().replace('-', '_')}_COMMAND"
        tool_dir = resolve_project_path(f"tools/quality_backends/{backend_key}")
        model_dir = resolve_project_path(f"tools/quality_models/lipsync/{backend_key}")
        checkpoint = resolve_project_path("tools/quality_models/lipsync/wav2lip_gan.pth")
        env_command = clean_text(os.environ.get(env_name, ""))
        if backend_key == "wav2lip":
            available = bool(env_command or checkpoint.exists() or tool_dir.exists())
            reason = "wav2lip checkpoint/tooling ready" if available else "wav2lip checkpoint or command missing"
        else:
            available = bool(env_command or tool_dir.exists() or model_dir.exists())
            reason = f"{backend_key} command/tooling ready" if available else f"{backend_key} command/tooling not installed"
        candidates.append(
            {
                "name": backend_key,
                "available": available,
                "reason": reason,
                "command_env": env_name,
                "tool_dir": str(tool_dir),
                "model_dir": str(model_dir),
            }
        )
    selected = next((row for row in candidates if row.get("available")), candidates[0] if candidates else {"name": "wav2lip", "reason": "default"})
    return {
        "backend_interface_version": 2,
        "preferred_order": preferred_order,
        "selected_backend": clean_text(selected.get("name", "")) or "wav2lip",
        "backend_candidates": candidates,
        "backend_reason": clean_text(selected.get("reason", "")) or "default backend selection",
        "allow_fallback": bool(lipsync_cfg.get("allow_fallback", False)),
        "min_sync_score": round(min_sync_score, 3),
        "backend_requirements": {
            "musetalk": {
                "status": "optional_configured_backend",
                "input_contract": "source video/frame sequence + cloned speech wav + speaker target",
            },
            "latentsync": {
                "status": "optional_configured_backend",
                "input_contract": "source video/frame sequence + cloned speech wav + speaker target",
            },
            "wav2lip": {
                "status": "existing_configured_backend",
                "input_contract": "face video/image + cloned speech wav + checkpoint",
            },
        },
    }


def scene_runner_statuses(package_payload: dict) -> dict[str, dict[str, str]]:
    backend_summary = package_payload.get("backend_runner_summary", {}) if isinstance(package_payload.get("backend_runner_summary", {}), dict) else {}
    scene_summary = backend_summary.get("scene_runners", {}) if isinstance(backend_summary.get("scene_runners", {}), dict) else {}
    statuses: dict[str, dict[str, str]] = {}
    for scene_result in scene_summary.get("scene_results", []) if isinstance(scene_summary.get("scene_results", []), list) else []:
        if not isinstance(scene_result, dict):
            continue
        scene_id = clean_text(scene_result.get("scene_id", ""))
        if not scene_id:
            continue
        scene_statuses: dict[str, str] = {}
        for runner_result in scene_result.get("runner_results", []) if isinstance(scene_result.get("runner_results", []), list) else []:
            if not isinstance(runner_result, dict):
                continue
            runner_name = clean_text(runner_result.get("runner_name", ""))
            if runner_name:
                scene_statuses[runner_name] = clean_text(runner_result.get("status", ""))
        statuses[scene_id] = scene_statuses
    return statuses


def runner_completed_for_strict_gate(status: str) -> bool:
    return clean_text(status) == "completed"


def strict_generation_output_gaps(cfg: dict, package_payload: dict) -> list[str]:
    if not strict_original_episode_outputs_required(cfg):
        return []
    render_cfg = cfg.get("render", {}) if isinstance(cfg.get("render"), dict) else {}
    require_video = bool(render_cfg.get("require_generated_scene_video", True))
    require_voice = bool(render_cfg.get("require_voice_clone_audio", True))
    require_lipsync = bool(render_cfg.get("require_lipsync_video", True))
    backend_summary = package_payload.get("backend_runner_summary", {}) if isinstance(package_payload.get("backend_runner_summary", {}), dict) else {}
    master_summary = backend_summary.get("master_runner", {}) if isinstance(backend_summary.get("master_runner", {}), dict) else {}
    runner_statuses = scene_runner_statuses(package_payload)
    gaps: list[str] = []
    scenes = package_payload.get("scenes", []) if isinstance(package_payload.get("scenes", []), list) else []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_id = clean_text(scene.get("scene_id", "")) or "scene"
        current_outputs = scene.get("current_generated_outputs", {}) if isinstance(scene.get("current_generated_outputs", {}), dict) else {}
        voice_clone = scene.get("voice_clone", {}) if isinstance(scene.get("voice_clone", {}), dict) else {}
        lip_sync = scene.get("lip_sync", {}) if isinstance(scene.get("lip_sync", {}), dict) else {}
        statuses = runner_statuses.get(scene_id, {})
        video_source_type = clean_text(current_outputs.get("video_source_type", ""))
        if require_video:
            if video_source_type not in STRICT_GENERATED_VIDEO_TYPES:
                gaps.append(f"{scene_id}: generated scene video missing (current source: {video_source_type or 'none'})")
            if not runner_completed_for_strict_gate(statuses.get("finished_episode_video_runner", "")):
                gaps.append(f"{scene_id}: finished_episode_video_runner did not complete")
        if require_voice and bool(voice_clone.get("required", False)):
            scene_audio_path = resolve_stored_project_path(current_outputs.get("scene_dialogue_audio", ""))
            audio_backend = clean_text(current_outputs.get("audio_backend", ""))
            if not render_output_ready(scene_audio_path):
                gaps.append(f"{scene_id}: cloned scene dialogue audio missing")
            elif not audio_backend_is_voice_clone(audio_backend):
                gaps.append(f"{scene_id}: scene dialogue audio is not marked as voice clone ({audio_backend or 'unknown backend'})")
            if not runner_completed_for_strict_gate(statuses.get("finished_episode_voice_runner", "")):
                gaps.append(f"{scene_id}: finished_episode_voice_runner did not complete")
        if require_lipsync and bool(lip_sync.get("required", False)):
            if video_source_type != "generated_lipsync_video":
                gaps.append(f"{scene_id}: generated lip-sync video missing")
            if not runner_completed_for_strict_gate(statuses.get("finished_episode_lipsync_runner", "")):
                gaps.append(f"{scene_id}: finished_episode_lipsync_runner did not complete")
    if not runner_completed_for_strict_gate(master_summary.get("status", "")):
        gaps.append("finished_episode_master_runner did not complete")
    return gaps


def ensure_strict_generation_outputs(cfg: dict, package_payload: dict, *, context_label: str) -> None:
    gaps = strict_generation_output_gaps(cfg, package_payload)
    if not gaps:
        return
    preview = "\n".join(f"- {gap}" for gap in gaps[:24])
    if len(gaps) > 24:
        preview += f"\n- ... plus {len(gaps) - 24} more gap(s)"
    raise RuntimeError(
        f"{context_label} did not produce real generated episode outputs. "
        "Local frame/video/voice fallbacks are disabled for quality-first mode.\n"
        f"{preview}"
    )


def remove_stale_generated_output(path_value: object) -> None:
    path = resolve_stored_project_path(path_value)
    try:
        if render_output_ready(path):
            path.unlink()
    except OSError:
        pass


def remove_stale_scene_runner_outputs(section_name: str, context: dict[str, object]) -> None:
    for key in SCENE_RUNNER_OUTPUT_KEYS.get(section_name, ()):
        remove_stale_generated_output(context.get(key, ""))


def backend_runner_fallback_used(runner_result: dict) -> bool:
    command_text = clean_text(runner_result.get("command_text", "")).lower()
    command_parts = (
        " ".join(str(part).lower() for part in runner_result.get("command", []) if str(part).strip())
        if isinstance(runner_result.get("command", []), list)
        else ""
    )
    combined = f"{command_text} {command_parts}"
    return any(marker in combined for marker in ("project_local_", "fallback", "pyttsx3"))


def write_backend_task_manifest(
    *,
    package_root: Path,
    scene_package: dict,
    scene_package_path: Path,
    runner_name: str,
    section_name: str,
    runner_result: dict,
) -> str:
    scene_id = clean_text(scene_package.get("scene_id", "")) or "scene"
    section_payload = scene_package.get(section_name, {}) if isinstance(scene_package.get(section_name, {}), dict) else {}
    target_outputs = section_payload.get("target_outputs", {}) if isinstance(section_payload.get("target_outputs", {}), dict) else {}
    manifest_dir = package_root / "manifests" / production_scene_slug(scene_id)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{runner_name}.json"
    produced_outputs = (
        [clean_text(path) for path in runner_result.get("produced_outputs", []) if clean_text(path)]
        if isinstance(runner_result.get("produced_outputs", []), list)
        else []
    )
    output_hashes: dict[str, str] = {}
    for output_path in produced_outputs:
        digest = file_sha256(output_path)
        if digest:
            output_hashes[output_path] = digest
    runner_status = clean_text(runner_result.get("status", "")) or "skipped"
    manifest = {
        "task_id": f"{scene_id}_{runner_name}",
        "scene_id": scene_id,
        "shot_id": "",
        "task_type": {
            "finished_episode_image_runner": "image",
            "finished_episode_video_runner": "video",
            "finished_episode_voice_runner": "voice",
            "finished_episode_lipsync_runner": "lipsync",
        }.get(runner_name, section_name),
        "backend": runner_name,
        "command": runner_result.get("command", []) if isinstance(runner_result.get("command", []), list) else runner_result.get("command_text", ""),
        "inputs": {"scene_package": str(scene_package_path), "section": section_name},
        "outputs": dict(target_outputs),
        "produced_outputs": produced_outputs,
        "output_hashes": output_hashes,
        "started_at": utc_now_iso(),
        "finished_at": utc_now_iso(),
        "duration_seconds": 0,
        "exit_code": int(runner_result.get("returncode", 0) or 0),
        "status": "success" if runner_status == "completed" else "skipped" if runner_status in {"disabled", "existing_outputs"} else "failed",
        "runner_status": runner_status,
        "fallback_used": backend_runner_fallback_used(runner_result),
        "placeholder_used": bool(runner_status not in {"completed", "existing_outputs"}),
        "stale_output": bool(runner_status == "existing_outputs"),
        "log_path": clean_text(runner_result.get("log_path", "")),
    }
    write_json(manifest_path, manifest)
    return str(manifest_path)


def usable_library_speaker_name(name: str) -> bool:
    lowered = clean_text(name).lower()
    if not lowered:
        return False
    if lowered in {"unknown", "speaker_unknown", "no face", "noface", "ignore", "ignored"}:
        return False
    if lowered.startswith("speaker_"):
        return False
    return True


def transcript_episode_id(path: Path) -> str:
    suffix = "_segments"
    stem = path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def linked_segment_index(cfg: dict) -> dict[str, dict[str, dict]]:
    linked_root = resolve_project_path(cfg["paths"]["linked_segments"])
    index: dict[str, dict[str, dict]] = {}
    for linked_path in sorted(linked_root.glob("*_linked_segments.json")):
        episode_id = linked_path.stem.replace("_linked_segments", "")
        rows = read_json(linked_path, [])
        episode_index: dict[str, dict] = {}
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            segment_id = clean_text(row.get("segment_id", ""))
            if segment_id:
                episode_index[segment_id] = row
        if episode_index:
            index[episode_id] = episode_index
    return index


def build_original_line_library(cfg: dict) -> dict[str, list[dict]]:
    transcript_root = resolve_project_path(cfg["paths"]["speaker_transcripts"])
    scene_root = resolve_project_path(cfg["paths"]["scene_clips"])
    linked_rows = linked_segment_index(cfg)
    library: dict[str, list[dict]] = {}
    for transcript_path in sorted(transcript_root.glob("*_segments.json")):
        episode_id = transcript_episode_id(transcript_path)
        episode_linked = linked_rows.get(episode_id, {})
        rows = read_json(transcript_path, [])
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            segment_id = clean_text(row.get("segment_id", ""))
            linked = episode_linked.get(segment_id, {})
            speaker_name = clean_text(linked.get("speaker_name", ""))
            text = clean_text(linked.get("text", "") or row.get("text", ""))
            if not usable_library_speaker_name(speaker_name) or not text:
                continue
            scene_id = clean_text(row.get("scene_id", "") or linked.get("scene_id", ""))
            scene_clip_path = scene_root / episode_id / f"{scene_id}.mp4" if scene_id else Path()
            audio_path = clean_text(row.get("audio_file", ""))
            resolved_audio_path = resolve_stored_project_path(audio_path) if audio_path else Path()
            library.setdefault(speaker_name, []).append(
                {
                    "episode_id": episode_id,
                    "scene_id": scene_id,
                    "segment_id": segment_id,
                    "text": text,
                    "audio_path": str(resolved_audio_path) if audio_path else "",
                    "scene_clip_path": str(scene_clip_path) if scene_id else "",
                    "language": normalize_language_code(row.get("language", "")),
                    "start": float(row.get("start", 0.0) or 0.0),
                    "end": float(row.get("end", 0.0) or 0.0),
                }
            )
    return library


def collect_speaker_reference_segments(original_line_library: dict[str, list[dict]], speaker_name: str, max_segments: int) -> list[dict]:
    entries = original_line_library.get(speaker_name, []) if isinstance(original_line_library.get(speaker_name, []), list) else []
    ranked_entries: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        audio_path = clean_text(entry.get("audio_path", ""))
        if not audio_path:
            continue
        resolved_audio_path = resolve_stored_project_path(audio_path)
        if not resolved_audio_path.exists() or not resolved_audio_path.is_file():
            continue
        ranked_entries.append(
            {
                **entry,
                "audio_path": str(resolved_audio_path),
                "_duration": max(0.0, float(entry.get("end", 0.0) or 0.0) - float(entry.get("start", 0.0) or 0.0)),
            }
        )
    ranked_entries.sort(key=lambda row: (float(row.get("_duration", 0.0) or 0.0), len(clean_text(row.get("text", "")))), reverse=True)
    prepared: list[dict] = []
    seen_paths: set[str] = set()
    for entry in ranked_entries:
        path_text = clean_text(entry.get("audio_path", ""))
        if not path_text or path_text in seen_paths:
            continue
        seen_paths.add(path_text)
        prepared.append(
            {
                "episode_id": clean_text(entry.get("episode_id", "")),
                "scene_id": clean_text(entry.get("scene_id", "")),
                "segment_id": clean_text(entry.get("segment_id", "")),
                "text": clean_text(entry.get("text", "")),
                "audio_path": path_text,
                "language": normalize_language_code(entry.get("language", "")),
                "start": float(entry.get("start", 0.0) or 0.0),
                "end": float(entry.get("end", 0.0) or 0.0),
            }
        )
        if len(prepared) >= max(1, int(max_segments or 1)):
            break
    return prepared


def build_voice_lookup(cfg: dict) -> dict[str, dict]:
    voice_map = read_json(resolve_project_path(cfg["paths"]["voice_map"]), {"clusters": {}, "aliases": {}})
    voice_models_root = resolve_project_path(str((cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}).get("voice_models", "characters/voice_models")))
    lookup: dict[str, dict] = {}
    for cluster_id, payload in (voice_map.get("clusters", {}) or {}).items():
        if not isinstance(payload, dict):
            continue
        speaker_name = clean_text(payload.get("name", ""))
        if not speaker_name:
            continue
        speaker_slug = production_scene_slug(speaker_name.lower())
        voice_model_path = voice_models_root / f"{speaker_slug}_voice_model.json"
        voice_model = read_json(voice_model_path, {}) if voice_model_path.exists() else {}
        lookup.setdefault(
            speaker_name,
            {
                "cluster_id": cluster_id,
                "name": speaker_name,
                "linked_face_cluster": clean_text(payload.get("linked_face_cluster", "")),
                "auto_named": bool(payload.get("auto_named", False)),
                "voice_model_path": str(voice_model_path) if voice_model_path.exists() else "",
                "dominant_language": normalize_language_code(voice_model.get("dominant_language", "")),
                "language_counts": dict(voice_model.get("language_counts", {}) or {}),
                "reference_audio": clean_text(voice_model.get("reference_audio", "")),
            },
        )
    return lookup


def voice_profile_for_speaker_name(voice_lookup: dict[str, dict], speaker_name: str, cfg: dict) -> dict:
    direct = voice_lookup.get(speaker_name, {})
    if isinstance(direct, dict) and direct:
        return direct
    clean_name = clean_text(speaker_name)
    if not clean_name:
        return {}
    speaker_slug = production_scene_slug(clean_name.lower())
    voice_models_root = resolve_project_path(str((cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}).get("voice_models", "characters/voice_models")))
    voice_model_path = voice_models_root / f"{speaker_slug}_voice_model.json"
    if not voice_model_path.exists():
        return {}
    voice_model = read_json(voice_model_path, {})
    return {
        "cluster_id": "",
        "name": clean_name,
        "linked_face_cluster": "",
        "auto_named": False,
        "voice_model_path": str(voice_model_path),
        "dominant_language": normalize_language_code(voice_model.get("dominant_language", "")),
        "language_counts": dict(voice_model.get("language_counts", {}) or {}),
        "reference_audio": clean_text(voice_model.get("reference_audio", "")),
    }


def parse_dialogue_line(raw_line: str, source: dict) -> tuple[str, str]:
    line = clean_text(raw_line)
    fallback_speaker = clean_text(source.get("speaker", ""))
    if ":" in line:
        speaker, text = line.split(":", 1)
        return clean_text(speaker) or fallback_speaker or "Narrator", clean_text(text)
    return fallback_speaker or "Narrator", line


def estimate_dialogue_duration_seconds(text: str, voice_rate: int, audio_pad_seconds: float) -> float:
    tokens = re.findall(r"\w+", clean_text(text), flags=re.UNICODE)
    word_count = max(1, len(tokens))
    punctuation_pauses = sum(clean_text(text).count(mark) for mark in ".!?;:")
    words_per_minute = max(90, int(voice_rate or 175))
    spoken_seconds = (word_count / words_per_minute) * 60.0
    pause_seconds = min(1.6, punctuation_pauses * 0.12)
    return max(0.9, spoken_seconds + pause_seconds + max(0.12, float(audio_pad_seconds or 0.0) * 0.5))


def build_scene_voice_plan(
    scene: dict,
    scene_duration_seconds: float,
    scene_start_seconds: float,
    original_line_library: dict[str, list[dict]],
    voice_lookup: dict[str, dict],
    cloning_cfg: dict,
    render_cfg: dict,
    cfg: dict,
) -> list[dict]:
    dialogue_lines = scene.get("dialogue_lines", []) if isinstance(scene.get("dialogue_lines", []), list) else []
    if not dialogue_lines:
        return []
    dialogue_voice_metadata = (
        scene.get("dialogue_voice_metadata", [])
        if isinstance(scene.get("dialogue_voice_metadata", []), list)
        else []
    )
    voice_rate = int(render_cfg.get("voice_rate", 175) or 175)
    audio_pad_seconds = float(render_cfg.get("audio_pad_seconds", 0.35) or 0.35)
    scene_language = normalize_language_code(scene.get("language", "") or scene.get("series_language", ""))
    prepared: list[dict] = []
    for line_index, raw_line in enumerate(dialogue_lines):
        source = scene_dialogue_source(scene, line_index)
        speaker_name, line_text = parse_dialogue_line(str(raw_line), source)
        if not line_text:
            continue
        voice_meta = (
            dialogue_voice_metadata[line_index]
            if line_index < len(dialogue_voice_metadata) and isinstance(dialogue_voice_metadata[line_index], dict)
            else {}
        )
        emotion = clean_text(voice_meta.get("emotion", "")) or ("curious" if "?" in line_text else "excited" if "!" in line_text else "focused")
        pace = clean_text(voice_meta.get("pace", "")) or "natural"
        energy = max(0.0, min(1.0, safe_float(voice_meta.get("energy", 0.52), 0.52)))
        target_duration_seconds = safe_float(voice_meta.get("target_duration_seconds", 0.0), 0.0)
        voice_reference_priority = (
            [clean_text(value) for value in voice_meta.get("voice_reference_priority", []) if clean_text(value)]
            if isinstance(voice_meta.get("voice_reference_priority", []), list)
            else []
        )
        pause_after_seconds = safe_float(voice_meta.get("pause_after_seconds", 0.18), 0.18)
        overlap_with_next_seconds = safe_float(voice_meta.get("overlap_with_next_seconds", 0.0), 0.0)
        delivery_notes = clean_text(voice_meta.get("delivery_notes", ""))
        retrieval_segment: dict | None = None
        if clean_text(source.get("type", "")) == "original_line" and clean_text(source.get("segment_id", "")):
            retrieval_segment = {
                "segment_id": clean_text(source.get("segment_id", "")),
                "text": clean_text(source.get("text", "") or line_text),
                "audio_path": clean_text(source.get("audio_file", "")),
                "scene_clip_path": clean_text(source.get("video_file", "")),
                "start": float(source.get("start", 0.0) or 0.0),
                "end": float(source.get("end", 0.0) or 0.0),
                "match_score": 1.0,
                "token_overlap": 1.0,
            }
        elif usable_library_speaker_name(speaker_name):
            retrieval_segment = select_retrieval_segment(speaker_name, line_text, original_line_library, cloning_cfg)
        reference_segments = collect_speaker_reference_segments(
            original_line_library,
            speaker_name,
            int(cloning_cfg.get("voice_reference_max_segments", 4) or 4),
        )
        prepared.append(
            {
                "line_index": line_index,
                "speaker_name": speaker_name,
                "text": line_text,
                "source": source,
                "retrieval_segment": retrieval_segment or {},
                "reference_segments": reference_segments,
                "voice_profile": voice_profile_for_speaker_name(voice_lookup, speaker_name, cfg),
                "raw_duration_seconds": max(
                    estimate_dialogue_duration_seconds(line_text, voice_rate, audio_pad_seconds),
                    target_duration_seconds,
                ),
                "emotion": emotion,
                "pace": pace,
                "energy": energy,
                "target_duration_seconds": target_duration_seconds,
                "pause_after_seconds": pause_after_seconds,
                "overlap_with_next_seconds": overlap_with_next_seconds,
                "delivery_notes": delivery_notes,
                "voice_reference_priority": voice_reference_priority or [
                    "matched_original_segment",
                    "trained_character_voice_model",
                    "speaker_reference_samples",
                ],
            }
        )

    raw_total = sum(safe_float(row.get("raw_duration_seconds", 0.0), 0.0) for row in prepared)
    line_gap_seconds = 0.18 if len(prepared) > 1 else 0.0
    raw_total_with_gaps = raw_total + max(0, len(prepared) - 1) * line_gap_seconds
    effective_scene_duration = max(safe_float(scene_duration_seconds, 0.0), raw_total_with_gaps, len(prepared) * 1.15)
    scale = 1.0
    if raw_total_with_gaps > effective_scene_duration:
        scale = max(0.82, effective_scene_duration / max(raw_total_with_gaps, 0.001))

    cursor = safe_float(scene_start_seconds, 0.0)
    scene_end = safe_float(scene_start_seconds, 0.0) + max(0.0, safe_float(effective_scene_duration, 0.0))
    plan: list[dict] = []
    for row_index, row in enumerate(prepared):
        duration = round(safe_float(row.get("raw_duration_seconds", 0.0), 0.0) * scale, 3)
        start_seconds = round(cursor, 3)
        end_seconds = round(min(scene_end, cursor + duration), 3)
        retrieval_segment = row["retrieval_segment"] if isinstance(row["retrieval_segment"], dict) else {}
        voice_profile = row["voice_profile"] if isinstance(row["voice_profile"], dict) else {}
        line_language = normalize_language_code(
            clean_text((row["source"] or {}).get("language", ""))
            or retrieval_segment.get("language", "")
            or voice_profile.get("dominant_language", "")
            or scene_language
        )
        plan.append(
            {
                "line_index": int(row["line_index"]),
                "speaker_name": row["speaker_name"],
                "text": row["text"],
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "estimated_duration_seconds": round(max(0.0, end_seconds - start_seconds), 3),
                "target_duration_seconds": round(safe_float(row.get("target_duration_seconds", 0.0), 0.0), 3),
                "pause_after_seconds": round(safe_float(row.get("pause_after_seconds", 0.18), 0.18), 3),
                "overlap_with_next_seconds": round(safe_float(row.get("overlap_with_next_seconds", 0.0), 0.0), 3),
                "delivery_notes": clean_text(row.get("delivery_notes", "")),
                "emotion": clean_text(row.get("emotion", "")) or "focused",
                "pace": clean_text(row.get("pace", "")) or "natural",
                "energy": round(max(0.0, min(1.0, safe_float(row.get("energy", 0.52), 0.52))), 3),
                "voice_reference_priority": row.get("voice_reference_priority", []) if isinstance(row.get("voice_reference_priority", []), list) else [],
                "audio_strategy": "reuse_original_segment" if retrieval_segment else "synthesize_preview_tts",
                "source_type": clean_text((row["source"] or {}).get("type", "")) or "generated",
                "source_segment_id": clean_text((row["source"] or {}).get("segment_id", "")),
                "language": line_language,
                "voice_profile": {
                    "cluster_id": clean_text(voice_profile.get("cluster_id", "")),
                    "name": clean_text(voice_profile.get("name", "")),
                    "linked_face_cluster": clean_text(voice_profile.get("linked_face_cluster", "")),
                    "auto_named": bool(voice_profile.get("auto_named", False)),
                    "voice_model_path": clean_text(voice_profile.get("voice_model_path", "")),
                    "dominant_language": normalize_language_code(voice_profile.get("dominant_language", "")),
                    "language_counts": dict(voice_profile.get("language_counts", {}) or {}),
                    "reference_audio": clean_text(voice_profile.get("reference_audio", "")),
                },
                "retrieval_segment": retrieval_segment,
                "reference_segments": row["reference_segments"] if isinstance(row.get("reference_segments", []), list) else [],
            }
        )
        cursor += duration
        if row_index < len(prepared) - 1:
            cursor = min(scene_end, cursor + line_gap_seconds)
    return plan


def format_srt_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, int(round(float(seconds or 0.0) * 1000.0)))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def render_subtitle_preview_srt(voice_plan_lines: list[dict]) -> str:
    blocks: list[str] = []
    for index, line in enumerate(voice_plan_lines, start=1):
        speaker_name = clean_text(line.get("speaker_name", "")) or "Narrator"
        text = clean_text(line.get("text", ""))
        if not text:
            continue
        start_seconds = float(line.get("start_seconds", 0.0) or 0.0)
        token_count = max(1, len(re.findall(r"\w+", text, flags=re.UNICODE)))
        minimum_read_time = min(6.5, max(1.6, token_count * 0.28))
        planned_end = float(line.get("end_seconds", start_seconds + minimum_read_time) or (start_seconds + minimum_read_time))
        end_seconds = max(start_seconds + minimum_read_time, planned_end)
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_timestamp(start_seconds)} --> {format_srt_timestamp(end_seconds)}",
                    f"{speaker_name}: {text}",
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def production_scene_slug(scene_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", clean_text(scene_id))
    return slug or "scene"


def prompt_preview_text(text: str, fallback: str) -> str:
    cleaned = clean_text(text)
    return cleaned or fallback


def voice_line_output_audio_path(episode_package_root: Path, line: dict) -> Path:
    speaker_name = clean_text(line.get("speaker_name", "")) or "Narrator"
    speaker_slug = production_scene_slug(speaker_name.lower())
    return episode_package_root / "audio" / speaker_slug / f"line_{int(line.get('line_index', 0) or 0):04d}.wav"


def scene_dialogue_output_audio_path(episode_package_root: Path, scene_id: str) -> Path:
    scene_slug = production_scene_slug(scene_id)
    return episode_package_root / "audio" / scene_slug / f"{scene_slug}_dialogue.wav"


def scene_master_clip_output_path(episode_package_root: Path, scene_id: str) -> Path:
    scene_slug = production_scene_slug(scene_id)
    return episode_package_root / "master" / "scenes" / f"{scene_slug}_master.mp4"


def scene_video_output_path(episode_package_root: Path, scene_id: str) -> Path:
    scene_slug = production_scene_slug(scene_id)
    return episode_package_root / "videos" / scene_slug / f"{scene_slug}.mp4"


def scene_video_preview_output_path(episode_package_root: Path, scene_id: str) -> Path:
    scene_slug = production_scene_slug(scene_id)
    return episode_package_root / "videos" / scene_slug / "preview.jpg"


def scene_video_poster_output_path(episode_package_root: Path, scene_id: str) -> Path:
    scene_slug = production_scene_slug(scene_id)
    return episode_package_root / "videos" / scene_slug / "poster.png"


def valid_media_source_path(path: Path) -> bool:
    return bool(str(path).strip()) and path.exists() and path.is_file()


def renderable_image_path(path: Path) -> bool:
    if not valid_media_source_path(path):
        return False
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def collect_scene_motion_frames(package_root: Path, scene_id: str, fallback_frame_path: Path) -> list[tuple[str, Path]]:
    frames: list[tuple[str, Path]] = []
    seen: set[str] = set()

    def add_frame(source_type: str, candidate: Path) -> None:
        if not renderable_image_path(candidate):
            return
        resolved = str(candidate.resolve())
        if resolved in seen:
            return
        seen.add(resolved)
        frames.append((source_type, candidate))

    for source_type, candidate in production_scene_frame_candidates(package_root, scene_id):
        add_frame(source_type, candidate)
    alternate_root = package_root / "images" / production_scene_slug(scene_id) / "alternates"
    if alternate_root.exists():
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
            for candidate in sorted(alternate_root.glob(pattern)):
                add_frame("generated_alternate_frame", candidate)
    add_frame("fallback_clean_frame", fallback_frame_path)
    return frames


def write_scene_video_reference_images(source_image_path: Path, preview_path: Path, poster_path: Path, width: int, height: int) -> None:
    image = fit_image(source_image_path, width, height)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    poster_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(poster_path)
    image.convert("RGB").save(preview_path, quality=92)


def scene_asset_sidecar_frames(asset_source_path: Path) -> list[tuple[str, Path]]:
    if not renderable_image_path(asset_source_path):
        return []
    parent = asset_source_path.parent
    candidates: list[tuple[str, Path]] = []
    candidate_roots = [parent]
    sibling_scene_root = parent / asset_source_path.stem
    if sibling_scene_root not in candidate_roots:
        candidate_roots.append(sibling_scene_root)
    for root in candidate_roots:
        for name in ("frame.png", "frame.jpg", "preview.jpg", "preview.png", "poster.png", "poster.jpg"):
            candidate = root / name
            if candidate != asset_source_path and renderable_image_path(candidate):
                candidates.append(("asset_sidecar_frame", candidate))
        alternates_root = root / "alternates"
        if alternates_root.exists():
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
                for candidate in sorted(alternates_root.glob(pattern)):
                    if renderable_image_path(candidate):
                        candidates.append(("asset_alternate_frame", candidate))
    return candidates


def derive_scene_visual_beats(
    scene: dict,
    scene_voice_plan: list[dict],
    generation_plan: dict,
    scene_duration_seconds: float,
    scene_start_seconds: float = 0.0,
) -> list[dict]:
    scene_start = float(scene_start_seconds or 0.0)
    scene_duration = max(0.25, float(scene_duration_seconds or 0.0))
    scene_end = scene_start + scene_duration
    camera_plan_raw = generation_plan.get("camera_plan", {})
    if isinstance(camera_plan_raw, dict):
        camera_plan = [camera_plan_raw]
    elif isinstance(camera_plan_raw, list):
        camera_plan = [row for row in camera_plan_raw if isinstance(row, dict)]
    else:
        camera_plan = []
    beats: list[dict] = []

    def append_beat(
        beat_name: str,
        start_seconds: float,
        end_seconds: float,
        *,
        speaker_name: str = "",
        focus: str = "",
        camera_hint: str = "",
        source: str = "",
        dialogue_text: str = "",
    ) -> None:
        start_value = max(scene_start, float(start_seconds or scene_start))
        end_value = min(scene_end, max(start_value + 0.01, float(end_seconds or start_value + 0.01)))
        duration_value = round(max(0.01, end_value - start_value), 3)
        beat_index = len(beats) + 1
        beats.append(
            {
                "beat_index": beat_index,
                "beat_name": beat_name,
                "start_seconds": round(start_value, 3),
                "end_seconds": round(end_value, 3),
                "duration_seconds": duration_value,
                "relative_start_seconds": round(start_value - scene_start, 3),
                "relative_end_seconds": round(end_value - scene_start, 3),
                "speaker_name": clean_text(speaker_name),
                "focus": clean_text(focus),
                "camera_hint": clean_text(camera_hint),
                "source": clean_text(source),
                "dialogue_text": clean_text(dialogue_text)[:160],
            }
        )

    prepared_lines = [line for line in scene_voice_plan if isinstance(line, dict)]
    if not prepared_lines:
        first_split = scene_start + (scene_duration * 0.55)
        if scene_duration <= 1.2:
            append_beat("wide_intro", scene_start, scene_end, focus=clean_text(scene.get("title", "")), source="scene_summary")
            return beats
        append_beat("wide_intro", scene_start, first_split, focus=clean_text(scene.get("title", "")), source="scene_summary")
        append_beat("reaction", first_split, scene_end, focus=clean_text(scene.get("summary", "")), source="scene_summary")
        return beats

    cursor = scene_start
    for index, line in enumerate(prepared_lines):
        line_start = max(scene_start, min(scene_end, float(line.get("start_seconds", cursor) or cursor)))
        line_end = max(line_start + 0.01, min(scene_end, float(line.get("end_seconds", line_start + 0.01) or (line_start + 0.01))))
        gap = line_start - cursor
        camera_entry = camera_plan[index] if index < len(camera_plan) and isinstance(camera_plan[index], dict) else {}
        focus = clean_text(camera_entry.get("focus", "")) or clean_text(line.get("speaker_name", ""))
        camera_hint = clean_text(camera_entry.get("camera", "")) or ("wide" if index == 0 else "dialogue")
        if gap >= 0.18:
            append_beat(
                "wide_intro" if not beats else ("push_in" if index % 2 == 0 else "reaction"),
                cursor,
                line_start,
                focus=focus or clean_text(scene.get("title", "")),
                camera_hint=camera_hint,
                source="dialogue_gap",
            )
        speaker_name = clean_text(line.get("speaker_name", ""))
        beat_name = "speaker_left" if index % 2 == 0 else "speaker_right"
        if not speaker_name:
            beat_name = "push_in" if index % 2 == 0 else "reaction"
        append_beat(
            "wide_intro" if not beats and line_start <= scene_start + 0.05 else beat_name,
            line_start,
            line_end,
            speaker_name=speaker_name,
            focus=focus or speaker_name,
            camera_hint=camera_hint,
            source="dialogue_line",
            dialogue_text=clean_text(line.get("text", "")),
        )
        cursor = line_end

    if scene_end - cursor >= 0.16:
        append_beat(
            "reaction",
            cursor,
            scene_end,
            focus=clean_text(scene.get("summary", "")) or clean_text(scene.get("title", "")),
            camera_hint=clean_text((camera_plan[-1] if camera_plan and isinstance(camera_plan[-1], dict) else {}).get("camera", "")),
            source="scene_outro",
        )

    return beats


def image_variant_for_beat(source_image: Image.Image, beat_name: str, accent: tuple[int, int, int], index: int) -> Image.Image:
    width, height = source_image.size
    crop_box = (0, 0, width, height)
    if beat_name == "speaker_left":
        crop_box = (0, 0, int(width * 0.84), height)
    elif beat_name == "speaker_right":
        crop_box = (int(width * 0.16), 0, width, height)
    elif beat_name == "push_in":
        crop_box = (int(width * 0.08), int(height * 0.08), int(width * 0.92), int(height * 0.92))
    elif beat_name == "reaction":
        crop_box = (int(width * 0.06), int(height * 0.02), int(width * 0.94), int(height * 0.90))
    elif beat_name == "detail":
        crop_box = (int(width * 0.14), int(height * 0.12), int(width * 0.86), int(height * 0.84))
    variant = ImageOps.fit(source_image.crop(crop_box), (width, height), method=Image.Resampling.LANCZOS)
    if beat_name == "speaker_right" and index % 2 == 1:
        variant = ImageOps.mirror(variant)
    variant = ImageEnhance.Contrast(variant).enhance(1.04 + (0.03 * (index % 3)))
    variant = ImageEnhance.Color(variant).enhance(1.02 + (0.02 * ((index + 1) % 3)))
    overlay = Image.new("RGBA", variant.size, accent + (24 + min(28, index * 6),))
    mask = Image.new("L", variant.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    if beat_name in {"speaker_left", "wide_intro"}:
        mask_draw.rectangle((0, 0, int(width * 0.58), height), fill=190)
    elif beat_name == "speaker_right":
        mask_draw.rectangle((int(width * 0.42), 0, width, height), fill=190)
    else:
        mask_draw.ellipse((int(width * 0.08), int(height * 0.06), int(width * 0.92), int(height * 0.94)), fill=180)
    overlay.putalpha(mask.filter(ImageFilter.GaussianBlur(max(14, int(min(width, height) * 0.025)))))
    variant = Image.alpha_composite(variant.convert("RGBA"), overlay).convert("RGB")
    return variant


def build_video_generation_prompt(scene: dict, generation_plan: dict) -> str:
    prompt_parts: list[str] = []
    positive_prompt = clean_text(generation_plan.get("positive_prompt", ""))
    if positive_prompt:
        prompt_parts.append(positive_prompt)
    characters = ", ".join(clean_text(value) for value in (scene.get("characters", []) or []) if clean_text(value))
    if characters:
        prompt_parts.append(f"characters: {characters}")
    location = clean_text(scene.get("location", ""))
    if location:
        prompt_parts.append(f"location: {location}")
    mood = clean_text(scene.get("mood", ""))
    if mood:
        prompt_parts.append(f"mood: {mood}")
    summary = clean_text(scene.get("summary", ""))
    if summary:
        prompt_parts.append(f"scene summary: {summary}")
    camera_plan_raw = generation_plan.get("camera_plan", {})
    if isinstance(camera_plan_raw, dict):
        camera_plan = [camera_plan_raw]
    elif isinstance(camera_plan_raw, list):
        camera_plan = [row for row in camera_plan_raw if isinstance(row, dict)]
    else:
        camera_plan = []
    for step in camera_plan[:3]:
        camera = clean_text(step.get("camera", "") or step.get("shot_type", ""))
        focus = clean_text(step.get("focus", "") or step.get("composition", ""))
        prompt_parts.append(f"camera: {camera or 'shot'} / focus: {focus or 'characters'}")
    control_hints_raw = generation_plan.get("control_hints", {})
    if isinstance(control_hints_raw, dict):
        control_hints = [control_hints_raw]
    elif isinstance(control_hints_raw, list):
        control_hints = [row for row in control_hints_raw if isinstance(row, dict)]
    else:
        control_hints = []
    for hint in control_hints[:3]:
        label = (
            clean_text(hint.get("hint", ""))
            or clean_text(hint.get("pose_emphasis", ""))
            or clean_text(hint.get("composition_emphasis", ""))
            or clean_text(hint.get("motion_emphasis", ""))
            or clean_text(hint.get("style_camera", ""))
        )
        value = (
            clean_text(hint.get("value", ""))
            or clean_text(hint.get("pose_guidance", ""))
            or clean_text(hint.get("composition_guidance", ""))
            or clean_text(hint.get("motion_guidance", ""))
            or clean_text(hint.get("style_angle", ""))
        )
        if label or value:
            prompt_parts.append(f"{label or 'control'}: {value}")
    style_constraints = generation_plan.get("style_constraints", {}) if isinstance(generation_plan.get("style_constraints", {}), dict) else {}
    style_positive = [clean_text(value) for value in style_constraints.get("positive", []) if clean_text(value)]
    if style_positive:
        prompt_parts.append(f"style: {', '.join(style_positive[:3])}")
    style_negative = [clean_text(value) for value in style_constraints.get("negative", []) if clean_text(value)]
    if style_negative:
        prompt_parts.append(f"avoid: {', '.join(style_negative[:4])}")
    style_guidance = style_constraints.get("guidance", {}) if isinstance(style_constraints.get("guidance", {}), dict) else {}
    for key in ("camera", "angle", "lighting", "palette", "framing"):
        value = clean_text(style_guidance.get(key, ""))
        if value:
            prompt_parts.append(f"{key}: {value}")
    character_continuity = generation_plan.get("character_continuity", []) if isinstance(generation_plan.get("character_continuity", []), list) else []
    for hint in character_continuity[:2]:
        if not isinstance(hint, dict):
            continue
        character = clean_text(hint.get("character", ""))
        outfit = clean_text(hint.get("outfit", ""))
        accessories = clean_text(hint.get("accessories", ""))
        details = ", ".join(value for value in (outfit, accessories) if value)
        if character and details:
            prompt_parts.append(f"keep {character}: {details}")
    continuity = generation_plan.get("continuity", {}) if isinstance(generation_plan.get("continuity", {}), dict) else {}
    previous_scene_id = clean_text(continuity.get("previous_scene_id", ""))
    if previous_scene_id:
        prompt_parts.append(f"continuity anchor from {previous_scene_id}")
    quality_targets = generation_plan.get("quality_targets", {}) if isinstance(generation_plan.get("quality_targets", {}), dict) else {}
    if quality_targets:
        desired = [clean_text(value) for value in quality_targets.get("desired_visual_traits", []) if clean_text(value)]
        if desired:
            prompt_parts.append(f"quality targets: {', '.join(desired[:4])}")
        if bool(quality_targets.get("preserve_character_identity", False)):
            prompt_parts.append("preserve character identity")
        if bool(quality_targets.get("preserve_series_style", False)):
            prompt_parts.append("preserve series style")
    return " | ".join(part for part in prompt_parts if part)


def build_voice_clone_line_package(cfg: dict, episode_package_root: Path, line: dict) -> dict:
    speaker_name = clean_text(line.get("speaker_name", "")) or "Narrator"
    speaker_slug = production_scene_slug(speaker_name.lower())
    voice_profile = line.get("voice_profile", {}) if isinstance(line.get("voice_profile", {}), dict) else {}
    retrieval_segment = line.get("retrieval_segment", {}) if isinstance(line.get("retrieval_segment", {}), dict) else {}
    reference_segments = line.get("reference_segments", []) if isinstance(line.get("reference_segments", []), list) else []
    cloning_cfg = cfg.get("cloning", {}) if isinstance(cfg.get("cloning", {}), dict) else {}
    voice_reference_priority = clean_text_list(line.get("voice_reference_priority", [])) or [
        "matched_original_segment",
        "trained_character_voice_model",
        "speaker_reference_samples",
    ]
    target_duration_seconds = safe_float(
        line.get("target_duration_seconds", line.get("estimated_duration_seconds", 0.0)),
        safe_float(line.get("estimated_duration_seconds", 0.0), 0.0),
    )
    energy = max(0.0, min(1.0, safe_float(line.get("energy", 0.52), 0.52)))
    voice_samples_root = resolve_project_path(str((cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}).get("voice_samples", "characters/voice_samples")))
    voice_models_root = resolve_project_path(str((cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}).get("voice_models", "characters/voice_models")))
    cluster_id = clean_text(voice_profile.get("cluster_id", ""))
    candidate_sample_dirs = [
        str(voice_samples_root / speaker_slug),
        str(voice_samples_root / cluster_id) if cluster_id else "",
    ]
    candidate_model_dirs = [
        str(voice_models_root / speaker_slug),
        str(voice_models_root / cluster_id) if cluster_id else "",
    ]
    return {
        "line_index": int(line.get("line_index", 0) or 0),
        "speaker_name": speaker_name,
        "text": clean_text(line.get("text", "")),
        "language": normalize_language_code(line.get("language", "") or voice_profile.get("dominant_language", "")),
        "emotion": clean_text(line.get("emotion", "")) or "focused",
        "pace": clean_text(line.get("pace", "")) or "natural",
        "energy": round(energy, 3),
        "target_duration_seconds": round(target_duration_seconds, 3),
        "pause_after_seconds": round(safe_float(line.get("pause_after_seconds", 0.18), 0.18), 3),
        "overlap_with_next_seconds": round(safe_float(line.get("overlap_with_next_seconds", 0.0), 0.0), 3),
        "delivery_notes": clean_text(line.get("delivery_notes", "")),
        "dialogue_function": clean_text(line.get("dialogue_function", "")),
        "voice_reference_priority": voice_reference_priority,
        "start_seconds": safe_float(line.get("start_seconds", 0.0), 0.0),
        "end_seconds": safe_float(line.get("end_seconds", 0.0), 0.0),
        "estimated_duration_seconds": safe_float(line.get("estimated_duration_seconds", 0.0), 0.0),
        "voice_profile": {
            "cluster_id": cluster_id,
            "linked_face_cluster": clean_text(voice_profile.get("linked_face_cluster", "")),
            "auto_named": bool(voice_profile.get("auto_named", False)),
            "voice_model_path": clean_text(voice_profile.get("voice_model_path", "")),
            "dominant_language": normalize_language_code(voice_profile.get("dominant_language", "")),
            "language_counts": dict(voice_profile.get("language_counts", {}) or {}),
            "reference_audio": clean_text(voice_profile.get("reference_audio", "")),
        },
        "original_voice_reference": {
            "audio_path": clean_text(retrieval_segment.get("audio_path", "")),
            "scene_clip_path": clean_text(retrieval_segment.get("scene_clip_path", "")),
            "segment_id": clean_text(retrieval_segment.get("segment_id", "")),
            "match_score": safe_float(retrieval_segment.get("match_score", 0.0), 0.0),
            "language": normalize_language_code(retrieval_segment.get("language", "")),
        },
        "reference_segments": [
            {
                "episode_id": clean_text(item.get("episode_id", "")),
                "scene_id": clean_text(item.get("scene_id", "")),
                "segment_id": clean_text(item.get("segment_id", "")),
                "text": clean_text(item.get("text", "")),
                "audio_path": clean_text(item.get("audio_path", "")),
                "language": normalize_language_code(item.get("language", "")),
            }
            for item in reference_segments
            if isinstance(item, dict) and clean_text(item.get("audio_path", ""))
        ],
        "reference_audio_candidates": [
            value
            for value in [
                clean_text(retrieval_segment.get("audio_path", "")),
                clean_text(voice_profile.get("reference_audio", "")),
                *[
                    clean_text(item.get("audio_path", ""))
                    for item in reference_segments
                    if isinstance(item, dict)
                ],
            ]
            if value
        ],
        "candidate_sample_dirs": [path for path in candidate_sample_dirs if path],
        "candidate_model_dirs": [path for path in candidate_model_dirs if path],
        "runtime": {
            "engine": clean_text(cloning_cfg.get("voice_clone_engine", "")),
            "xtts_model_name": clean_text(cloning_cfg.get("xtts_model_name", "")),
            "xtts_language": clean_text(cloning_cfg.get("xtts_language", "")),
            "xtts_license_accepted": bool(cloning_cfg.get("xtts_license_accepted", False)),
            "force_voice_cloning": bool(cloning_cfg.get("force_voice_cloning", True)),
            "allow_system_tts_fallback": bool(cloning_cfg.get("allow_system_tts_fallback", False)),
        },
        "target_output_audio": str(voice_line_output_audio_path(episode_package_root, line)),
    }


def build_shot_packages(episode_package_root: Path, scene_id: str, shot_plan: list[dict]) -> list[dict]:
    scene_slug = production_scene_slug(scene_id)
    rows: list[dict] = []
    for index, shot in enumerate(shot_plan, start=1):
        if not isinstance(shot, dict):
            continue
        shot_id = clean_text(shot.get("shot_id", "")) or f"{scene_id}_shot_{index:03d}"
        shot_slug = production_scene_slug(shot_id)
        shot_root = episode_package_root / "shots" / scene_slug / shot_slug
        rows.append(
            {
                "shot_id": shot_id,
                "scene_id": scene_id,
                "shot_type": clean_text(shot.get("shot_type", "")),
                "duration_seconds": safe_float(shot.get("duration_seconds", 0.0), 0.0),
                "camera_angle": clean_text(shot.get("camera_angle", "")),
                "camera_movement": clean_text(shot.get("camera_movement", "")),
                "characters_visible": clean_text_list(shot.get("characters_visible", [])),
                "dialogue_line_indices": [
                    int(value)
                    for value in (shot.get("dialogue_line_indices", []) if isinstance(shot.get("dialogue_line_indices", []), list) else [])
                    if str(value).strip().lstrip("-").isdigit()
                ],
                "purpose": clean_text(shot.get("purpose", "")),
                "target_outputs": {
                    "primary_frame": str(shot_root / "primary_frame.png"),
                    "video_clip": str(shot_root / f"{shot_slug}_video.mp4"),
                    "lipsync_clip": str(shot_root / f"{shot_slug}_lipsync.mp4"),
                    "manifest": str(shot_root / f"{shot_slug}_manifest.json"),
                },
                "backend_metadata_required": True,
            }
        )
    return rows


def scene_audio_mix_package(episode_package_root: Path, scene_id: str, audio_mix: dict) -> dict:
    scene_slug = production_scene_slug(scene_id)
    mix_root = episode_package_root / "audio" / scene_slug / "mix"
    return {
        "required": bool(audio_mix.get("required", True)) if isinstance(audio_mix, dict) else True,
        "mode": "dialogue_ambience_music_sfx_scene_mix",
        "target_lufs": safe_float(audio_mix.get("loudness_target_lufs", -16.0), -16.0) if isinstance(audio_mix, dict) else -16.0,
        "stems": {
            "dialogue": str(mix_root / f"{scene_slug}_dialogue_stem.wav"),
            "ambience": str(mix_root / f"{scene_slug}_ambience_stem.wav"),
            "music": str(mix_root / f"{scene_slug}_music_stem.wav"),
            "sfx": str(mix_root / f"{scene_slug}_sfx_stem.wav"),
            "final_mix": str(mix_root / f"{scene_slug}_final_mix.wav"),
        },
        "placeholder_policy": {
            "allow_music_placeholder": False,
            "allow_sfx_placeholder": False,
        },
    }


def merge_scene_regeneration_metadata(
    quality_assessment: dict,
    previous_quality: dict | None = None,
) -> dict:
    merged = dict(quality_assessment) if isinstance(quality_assessment, dict) else {}
    previous = previous_quality if isinstance(previous_quality, dict) else {}
    retries = int(previous.get("regeneration_retries", 0) or 0)
    retry_limit = int(
        previous.get("regeneration_retry_limit", previous.get("max_regeneration_retries", 0)) or 0
    )
    if retries > 0:
        merged["regeneration_retries"] = retries
    if retry_limit > 0:
        merged["regeneration_retry_limit"] = retry_limit
        merged["max_regeneration_retries"] = retry_limit
    for key in (
        "last_regeneration_requested_at",
        "last_regeneration_applied_at",
        "last_regeneration_reason",
        "last_regeneration_request_source",
        "last_regeneration_queue_manifest",
        "last_regeneration_apply_mode",
    ):
        value = clean_text(previous.get(key, ""))
        if value:
            merged[key] = value
    if isinstance(previous.get("last_regeneration_queue_entry"), dict) and previous.get("last_regeneration_queue_entry"):
        merged["last_regeneration_queue_entry"] = dict(previous.get("last_regeneration_queue_entry", {}))
    if "queued_for_regeneration" in previous:
        merged["queued_for_regeneration"] = bool(previous.get("queued_for_regeneration", False))
    return merged


def build_scene_production_package(
    cfg: dict,
    episode_id: str,
    episode_package_root: Path,
    scene: dict,
    scene_manifest: dict,
    scene_voice_plan: dict,
    previous_scene_package: dict | None = None,
) -> dict:
    scene_id = clean_text(scene.get("scene_id", "") or scene_manifest.get("scene_id", "")) or "scene"
    scene_slug = production_scene_slug(scene_id)
    generation_plan = scene.get("generation_plan", {}) if isinstance(scene.get("generation_plan", {}), dict) else {}
    reference_slots = generation_plan.get("reference_slots", []) if isinstance(generation_plan.get("reference_slots", []), list) else []
    continuity = generation_plan.get("continuity", {}) if isinstance(generation_plan.get("continuity", {}), dict) else {}
    style_constraints = generation_plan.get("style_constraints", {}) if isinstance(generation_plan.get("style_constraints", {}), dict) else {}
    character_continuity = generation_plan.get("character_continuity", []) if isinstance(generation_plan.get("character_continuity", []), list) else []
    quality_targets = generation_plan.get("quality_targets", {}) if isinstance(generation_plan.get("quality_targets", {}), dict) else {}
    camera_plan_raw = generation_plan.get("camera_plan", {})
    control_hints_raw = generation_plan.get("control_hints", {})
    camera_plan = camera_plan_raw if isinstance(camera_plan_raw, dict) else {}
    control_hints = control_hints_raw if isinstance(control_hints_raw, dict) else {}
    image_output_root = episode_package_root / "images" / scene_slug
    video_output_root = episode_package_root / "videos" / scene_slug
    lipsync_output_root = episode_package_root / "lipsync" / scene_slug
    current_generated_outputs = scene_manifest.get("current_generated_outputs", {}) if isinstance(scene_manifest.get("current_generated_outputs", {}), dict) else {}
    manifest_visual_beats = scene_manifest.get("scene_visual_beats", []) if isinstance(scene_manifest.get("scene_visual_beats", []), list) else []
    visual_beats: list[dict] = []
    for beat in manifest_visual_beats:
        if not isinstance(beat, dict):
            continue
        visual_beats.append(
            {
                "beat_index": int(beat.get("beat_index", len(visual_beats) + 1) or (len(visual_beats) + 1)),
                "beat_name": clean_text(beat.get("beat_name", "")),
                "start_seconds": round(float(beat.get("start_seconds", 0.0) or 0.0), 3),
                "end_seconds": round(float(beat.get("end_seconds", 0.0) or 0.0), 3),
                "duration_seconds": round(float(beat.get("duration_seconds", 0.0) or 0.0), 3),
                "relative_start_seconds": round(float(beat.get("relative_start_seconds", 0.0) or 0.0), 3),
                "relative_end_seconds": round(float(beat.get("relative_end_seconds", 0.0) or 0.0), 3),
                "speaker_name": clean_text(beat.get("speaker_name", "")),
                "focus": clean_text(beat.get("focus", "")),
                "camera_hint": clean_text(beat.get("camera_hint", "")),
                "source": clean_text(beat.get("source", "")),
                "dialogue_text": clean_text(beat.get("dialogue_text", "")),
                "reference_image_path": clean_text(beat.get("reference_image_path", "")),
                "frame_source_type": clean_text(beat.get("frame_source_type", "")),
                "frame_source_path": clean_text(beat.get("frame_source_path", "")),
            }
        )
    beat_reference_images = [clean_text(beat.get("reference_image_path", "")) for beat in visual_beats if clean_text(beat.get("reference_image_path", ""))]
    current_video_source_type = clean_text(scene_manifest.get("video_source_type", "")) or clean_text(current_generated_outputs.get("video_source_type", ""))
    local_composed_scene_video = current_video_source_type in LOCAL_COMPOSED_VIDEO_TYPES
    compose_strategy = "reuse_generated_scene_video" if current_video_source_type in {"generated_scene_video", "generated_lipsync_video"} else "compose_local_scene_video_from_visual_beats"
    voice_lines = [
        build_voice_clone_line_package(cfg, episode_package_root, line)
        for line in (scene_voice_plan.get("lines", []) if isinstance(scene_voice_plan.get("lines", []), list) else [])
        if isinstance(line, dict)
    ]
    image_prompt = prompt_preview_text(
        generation_plan.get("positive_prompt", ""),
        f"Generate a new clean keyframe for {scene_id}: {clean_text(scene.get('summary', '')) or clean_text(scene.get('title', ''))}",
    )
    video_prompt = prompt_preview_text(
        build_video_generation_prompt(scene, generation_plan),
        f"Generate a new video shot for {scene_id} with the listed characters and continuity.",
    )
    lipsync_profile = lipsync_backend_profile(cfg)
    behavior_constraints = clean_text_list(scene.get("behavior_constraints", []))
    dialogue_style_constraints = clean_text_list(scene.get("dialogue_style_constraints", []))
    callback_targets = clean_text_list(scene.get("callback_targets", []))
    character_intents = scene.get("character_intents", {}) if isinstance(scene.get("character_intents", {}), dict) else {}
    writer_room_plan = scene.get("writer_room_plan", {}) if isinstance(scene.get("writer_room_plan", {}), dict) else {}
    relationship_context = scene.get("relationship_context", []) if isinstance(scene.get("relationship_context", []), list) else []
    dialogue_voice_metadata = [
        dict(item)
        for item in (scene.get("dialogue_voice_metadata", []) if isinstance(scene.get("dialogue_voice_metadata", []), list) else [])
        if isinstance(item, dict)
    ]
    dialogue_sources = scene.get("dialogue_sources", []) if isinstance(scene.get("dialogue_sources", []), list) else []
    dialogue_line_metadata = [
        dict(item)
        for item in (scene.get("dialogue_line_metadata", []) if isinstance(scene.get("dialogue_line_metadata", []), list) else [])
        if isinstance(item, dict)
    ]
    shot_plan = [
        dict(item)
        for item in (scene.get("shot_plan", []) if isinstance(scene.get("shot_plan", []), list) else [])
        if isinstance(item, dict)
    ]
    character_continuity_lock = scene.get("character_continuity_lock", {}) if isinstance(scene.get("character_continuity_lock", {}), dict) else {}
    set_context = scene.get("set_context", {}) if isinstance(scene.get("set_context", {}), dict) else {}
    audio_mix = scene_audio_mix_package(
        episode_package_root,
        scene_id,
        scene.get("audio_mix", {}) if isinstance(scene.get("audio_mix", {}), dict) else {},
    )
    shot_packages = build_shot_packages(episode_package_root, scene_id, shot_plan)
    generic_source_count = sum(
        1
        for item in dialogue_sources
        if isinstance(item, dict) and clean_text(item.get("type", "")) == "generated_template"
    )
    generic_template_line_ratio = generic_source_count / max(1, len(dialogue_sources))
    voice_metadata_available = bool(voice_lines) and all(
        clean_text(line.get("emotion", ""))
        and clean_text(line.get("pace", ""))
        and safe_float(line.get("target_duration_seconds", 0.0), 0.0) > 0.0
        and clean_text(line.get("delivery_notes", ""))
        and isinstance(line.get("voice_reference_priority", []), list)
        for line in voice_lines
    )
    scene_package = {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "title": clean_text(scene.get("title", "")),
        "summary": clean_text(scene.get("summary", "")),
        "location": clean_text(scene.get("location", "")),
        "location_id": clean_text(scene.get("location_id", "")),
        "set_context": set_context,
        "mood": clean_text(scene.get("mood", "")),
        "scene_function": clean_text(scene.get("scene_function", "")),
        "scene_purpose": clean_text(scene.get("scene_purpose", "")),
        "conflict": clean_text(scene.get("conflict", "")),
        "character_intents": character_intents,
        "behavior_constraints": behavior_constraints,
        "dialogue_style_constraints": dialogue_style_constraints,
        "comedy_pattern": clean_text(scene.get("comedy_pattern", "")),
        "emotional_arc": clean_text(scene.get("emotional_arc", "")),
        "callback_targets": callback_targets,
        "writer_room_plan": writer_room_plan,
        "relationship_context": relationship_context,
        "dialogue_voice_metadata": dialogue_voice_metadata,
        "dialogue_line_metadata": dialogue_line_metadata,
        "dialogue_sources": dialogue_sources,
        "characters": scene.get("characters", []) if isinstance(scene.get("characters", []), list) else [],
        "character_continuity_lock": character_continuity_lock,
        "shot_plan": shot_plan,
        "shot_packages": shot_packages,
        "audio_mix": audio_mix,
        "duration_seconds": float(scene_manifest.get("duration_seconds", scene_voice_plan.get("duration_seconds", 0.0)) or 0.0),
        "current_preview_assets": {
            "asset_source_type": clean_text(scene_manifest.get("asset_source_type", "")),
            "asset_source_path": clean_text(scene_manifest.get("asset_source_path", "")),
            "preview_frame_path": clean_text(scene_manifest.get("frame_path", "")),
        },
        "current_generated_outputs": {
            "asset_source_type": clean_text(scene_manifest.get("asset_source_type", "")),
            "video_source_type": current_video_source_type,
            "video_source_path": clean_text(scene_manifest.get("video_source_path", "")) or clean_text(current_generated_outputs.get("video_source_path", "")),
            "final_clip_path": clean_text(scene_manifest.get("final_clip_path", "")),
            "scene_dialogue_audio": clean_text(scene_manifest.get("scene_dialogue_audio", "")),
            "scene_master_clip": clean_text(scene_manifest.get("scene_master_clip", "")),
            "audio_backend": clean_text(scene_manifest.get("audio_backend", "")),
            "has_generated_scene_video": bool(current_video_source_type),
            "has_scene_dialogue_audio": bool(clean_text(scene_manifest.get("scene_dialogue_audio", ""))),
            "has_scene_master_clip": bool(clean_text(scene_manifest.get("scene_master_clip", ""))),
            "scene_visual_beat_count": len(visual_beats),
            "beat_reference_images": beat_reference_images,
            "has_visual_beat_reference_images": bool(beat_reference_images),
            "local_composed_scene_video": local_composed_scene_video,
        },
        "storyboard": {
            "requires_new_storyboard_frames": True,
            "behavior": {
                "scene_purpose": clean_text(scene.get("scene_purpose", "")),
                "conflict": clean_text(scene.get("conflict", "")),
                "character_intents": character_intents,
                "behavior_constraints": behavior_constraints,
                "dialogue_style_constraints": dialogue_style_constraints,
                "comedy_pattern": clean_text(scene.get("comedy_pattern", "")),
                "emotional_arc": clean_text(scene.get("emotional_arc", "")),
                "callback_targets": callback_targets,
                "writer_room_plan": writer_room_plan,
            },
            "reference_slots": reference_slots,
            "camera_plan": camera_plan,
            "control_hints": control_hints,
            "continuity": continuity,
            "style_constraints": style_constraints,
            "character_continuity": character_continuity,
            "character_continuity_lock": character_continuity_lock,
            "shot_plan": shot_plan,
            "shot_packages": shot_packages,
            "set_context": set_context,
            "quality_targets": quality_targets,
            "scene_package_path": str(episode_package_root / "scenes" / f"{scene_slug}_production.json"),
        },
        "image_generation": {
            "required": True,
            "mode": "new_scene_keyframes",
            "prompt": image_prompt,
            "negative_prompt": clean_text(generation_plan.get("negative_prompt", "")),
            "batch_prompt_line": clean_text(generation_plan.get("batch_prompt_line", "")),
            "reference_slots": reference_slots,
            "style_constraints": style_constraints,
            "character_continuity": character_continuity,
            "character_continuity_lock": character_continuity_lock,
            "shot_plan": shot_plan,
            "shot_packages": shot_packages,
            "set_context": set_context,
            "target_outputs": {
                "primary_frame": str(image_output_root / "frame_0001.png"),
                "alternate_frame_dir": str(image_output_root / "alternates"),
                "layered_storyboard_frame": str(image_output_root / "storyboard_frame.png"),
                "backend_manifest": str(episode_package_root / "manifests" / scene_slug / "finished_episode_image_runner.json"),
            },
        },
        "video_generation": {
            "required": True,
            "mode": "new_scene_video_clip",
            "prompt": video_prompt,
            "camera_plan": camera_plan,
            "control_hints": control_hints,
            "continuity": continuity,
            "style_constraints": style_constraints,
            "character_continuity": character_continuity,
            "character_continuity_lock": character_continuity_lock,
            "shot_plan": shot_plan,
            "shot_packages": shot_packages,
            "set_context": set_context,
            "quality_targets": quality_targets,
            "compose_strategy": compose_strategy,
            "local_video_plan": {
                "mode": "dialogue_timed_multi_shot_scene_video",
                "fallback_active": not current_video_source_type or local_composed_scene_video,
                "beat_reference_root": str(video_output_root / "beat_references"),
                "beat_count": len(visual_beats),
                "beats": visual_beats,
            },
            "target_outputs": {
                "scene_video": str(scene_video_output_path(episode_package_root, scene_id)),
                "preview_frame": str(scene_video_preview_output_path(episode_package_root, scene_id)),
                "poster_frame": str(scene_video_poster_output_path(episode_package_root, scene_id)),
                "backend_manifest": str(episode_package_root / "manifests" / scene_slug / "finished_episode_video_runner.json"),
            },
        },
        "voice_clone": {
            "required": bool(voice_lines),
            "metadata_schema_version": 2,
            "mode": "original_character_voice_clone",
            "target_outputs": {
                "scene_dialogue_audio": str(scene_dialogue_output_audio_path(episode_package_root, scene_id)),
                "backend_manifest": str(episode_package_root / "manifests" / scene_slug / "finished_episode_voice_runner.json"),
            },
            "lines": voice_lines,
        },
        "lip_sync": {
            "required": bool(voice_lines),
            **lipsync_profile,
            "mode": "character_lip_sync_composite",
            "target_outputs": {
                "lipsync_video": str(lipsync_output_root / f"{scene_slug}_lipsync.mp4"),
                "poster_frame": str(lipsync_output_root / "poster.png"),
                "backend_manifest": str(episode_package_root / "manifests" / scene_slug / "finished_episode_lipsync_runner.json"),
            },
            "speaker_targets": sorted({clean_text(line.get("speaker_name", "")) for line in voice_lines if clean_text(line.get("speaker_name", ""))}),
            "audio_dependencies": [clean_text(line.get("target_output_audio", "")) for line in voice_lines if clean_text(line.get("target_output_audio", ""))],
        },
        "mastering": {
            "required": True,
            "mode": "scene_master_clip",
            "target_outputs": {
                "scene_master_clip": str(scene_master_clip_output_path(episode_package_root, scene_id)),
                "backend_manifest": str(episode_package_root / "manifests" / scene_slug / "scene_master.json"),
            },
        },
    }
    previous_quality = (
        previous_scene_package.get("quality_assessment", {})
        if isinstance(previous_scene_package, dict) and isinstance(previous_scene_package.get("quality_assessment", {}), dict)
        else {}
    )
    scene_package["quality_assessment"] = merge_scene_regeneration_metadata(
        scene_quality_assessment(
            scene_id=scene_id,
            current_outputs=scene_package.get("current_generated_outputs", {}),
            voice_required=bool(voice_lines),
            lipsync_required=bool(voice_lines),
            reference_slot_count=len(reference_slots),
            continuity_active=bool(continuity),
            continuity_character_count=len(character_continuity),
            style_guidance_available=bool(style_constraints.get("positive") or style_constraints.get("guidance")),
            quality_targets_available=bool(quality_targets),
            behavior_constraints_available=bool(behavior_constraints and dialogue_style_constraints),
            voice_metadata_available=voice_metadata_available,
            relationship_context_available=bool(relationship_context),
            scene_conflict_available=bool(clean_text(scene.get("conflict", ""))),
            generic_template_line_ratio=generic_template_line_ratio,
        ),
        previous_quality,
    )
    return scene_package


def build_episode_production_package_payload(
    cfg: dict,
    episode_id: str,
    shotlist: dict,
    manifest: dict,
    voice_plan_payload: dict,
    package_root: Path,
    existing_scene_packages: dict[str, dict] | None = None,
) -> dict:
    scenes = shotlist.get("scenes", []) if isinstance(shotlist.get("scenes", []), list) else []
    manifest_scenes = manifest.get("scenes", []) if isinstance(manifest.get("scenes", []), list) else []
    voice_plan_scenes = voice_plan_payload.get("scenes", []) if isinstance(voice_plan_payload.get("scenes", []), list) else []
    manifest_index = {clean_text(scene.get("scene_id", "")): scene for scene in manifest_scenes if isinstance(scene, dict)}
    voice_plan_index = {clean_text(scene.get("scene_id", "")): scene for scene in voice_plan_scenes if isinstance(scene, dict)}
    existing_scene_index = existing_scene_packages if isinstance(existing_scene_packages, dict) else {}
    scene_packages: list[dict] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_id = clean_text(scene.get("scene_id", ""))
        scene_packages.append(
            build_scene_production_package(
                cfg,
                episode_id,
                package_root,
                scene,
                manifest_index.get(scene_id, {}),
                voice_plan_index.get(scene_id, {}),
                existing_scene_index.get(scene_id, {}),
            )
        )
    total_line_count = sum(len(scene.get("voice_clone", {}).get("lines", [])) for scene in scene_packages if isinstance(scene.get("voice_clone", {}), dict))
    edit_decision_list = shotlist.get("edit_decision_list", []) if isinstance(shotlist.get("edit_decision_list", []), list) else []
    audio_mix_plan = shotlist.get("audio_mix_plan", {}) if isinstance(shotlist.get("audio_mix_plan", {}), dict) else {}
    audio_mastering_cfg = cfg.get("audio_mastering", {}) if isinstance(cfg.get("audio_mastering", {}), dict) else {}
    generated_scene_video_count = sum(
        1
        for scene in scene_packages
        if isinstance(scene, dict)
        and isinstance(scene.get("current_generated_outputs", {}), dict)
        and bool(scene["current_generated_outputs"].get("has_generated_scene_video", False))
    )
    scene_dialogue_audio_count = sum(
        1
        for scene in scene_packages
        if isinstance(scene, dict)
        and isinstance(scene.get("current_generated_outputs", {}), dict)
        and bool(scene["current_generated_outputs"].get("has_scene_dialogue_audio", False))
    )
    scene_master_clip_count = sum(
        1
        for scene in scene_packages
        if isinstance(scene, dict)
        and isinstance(scene.get("current_generated_outputs", {}), dict)
        and bool(scene["current_generated_outputs"].get("has_scene_master_clip", False))
    )
    local_composed_scene_video_count = sum(
        1
        for scene in scene_packages
        if isinstance(scene, dict)
        and isinstance(scene.get("current_generated_outputs", {}), dict)
        and bool(scene["current_generated_outputs"].get("local_composed_scene_video", False))
    )
    visual_beat_count = sum(
        int(scene.get("current_generated_outputs", {}).get("scene_visual_beat_count", 0) or 0)
        for scene in scene_packages
        if isinstance(scene, dict) and isinstance(scene.get("current_generated_outputs", {}), dict)
    )
    completion_status = generated_episode_completion_summary(
        scene_count=len(scene_packages),
        generated_scene_video_count=generated_scene_video_count,
        scene_dialogue_audio_count=scene_dialogue_audio_count,
        scene_master_clip_count=scene_master_clip_count,
        render_mode=clean_text(manifest.get("render_mode", "")),
        final_render=clean_text(manifest.get("final_render", "")),
        full_generated_episode=str(package_root / "master" / f"{episode_id}_full_generated_episode.mp4"),
    )
    quality_assessment = episode_quality_assessment(
        [scene.get("quality_assessment", {}) for scene in scene_packages if isinstance(scene, dict)],
        scene_count=len(scene_packages),
    )
    return {
        "episode_id": episode_id,
        "package_kind": "full_generated_episode_backend_package",
        "render_mode": clean_text(manifest.get("render_mode", "")),
        "display_title": clean_text(shotlist.get("display_title", episode_id)),
        "episode_title": clean_text(shotlist.get("episode_title", "")),
        "source_shotlist": clean_text(manifest.get("shotlist_path", "")),
        "source_storyboard_request_dir": clean_text(shotlist.get("storyboard_request_dir", "")),
        "source_render_manifest": clean_text(manifest.get("render_manifest_path", "")),
        "source_voice_plan": clean_text(manifest.get("voice_plan", "")),
        "episode_blueprint": shotlist.get("episode_blueprint", {}) if isinstance(shotlist.get("episode_blueprint", {}), dict) else {},
        "set_bible": shotlist.get("set_bible", {}) if isinstance(shotlist.get("set_bible", {}), dict) else {},
        "edit_decision_list": edit_decision_list,
        "audio_mastering": {
            "config": audio_mastering_cfg,
            "mix_plan": audio_mix_plan,
            "required": bool(audio_mastering_cfg.get("enabled", True)),
            "target_outputs": {
                "dialogue_stem": str(package_root / "master" / f"{episode_id}_dialogue_stem.wav"),
                "ambience_stem": str(package_root / "master" / f"{episode_id}_ambience_stem.wav"),
                "music_stem": str(package_root / "master" / f"{episode_id}_music_stem.wav"),
                "sfx_stem": str(package_root / "master" / f"{episode_id}_sfx_stem.wav"),
                "final_mix": str(package_root / "master" / f"{episode_id}_final_mix.wav"),
            },
        },
        "current_preview_outputs": {
            "draft_render": clean_text(manifest.get("draft_render", "")),
            "final_storyboard_render": clean_text(manifest.get("final_render", "")),
            "dialogue_audio": clean_text(manifest.get("dialogue_audio", "")),
            "subtitle_preview": clean_text(manifest.get("subtitle_preview", "")),
        },
        "production_goal": "fully_generated_episode_with_new_storyboard_new_frames_original_voices_and_lip_sync",
        "backend_requirements": {
            "image_generation": True,
            "video_generation": True,
            "voice_clone": total_line_count > 0,
            "lip_sync": total_line_count > 0,
        },
        "completion_status": completion_status,
        "quality_assessment": quality_assessment,
        "local_composition_status": {
            "scene_visual_beat_count": visual_beat_count,
            "scene_count_with_local_composed_video": local_composed_scene_video_count,
            "scene_count_with_generated_or_lipsync_video": generated_scene_video_count,
        },
        "target_master_outputs": {
            "image_root": str(package_root / "images"),
            "video_root": str(package_root / "videos"),
            "audio_root": str(package_root / "audio"),
            "lipsync_root": str(package_root / "lipsync"),
            "scene_package_root": str(package_root / "scenes"),
            "scene_master_root": str(package_root / "master" / "scenes"),
            "manifest_root": str(package_root / "manifests"),
            "final_audio_mix": str(package_root / "master" / f"{episode_id}_final_mix.wav"),
            "final_master_episode": str(package_root / "master" / f"{episode_id}_full_generated_episode.mp4"),
        },
        "scene_count": len(scene_packages),
        "line_count": total_line_count,
        "scenes": scene_packages,
    }


def render_production_prompt_preview(scene_packages: list[dict]) -> str:
    lines: list[str] = []
    for scene in scene_packages:
        if not isinstance(scene, dict):
            continue
        scene_id = clean_text(scene.get("scene_id", ""))
        image_generation = scene.get("image_generation", {}) if isinstance(scene.get("image_generation", {}), dict) else {}
        video_generation = scene.get("video_generation", {}) if isinstance(scene.get("video_generation", {}), dict) else {}
        local_video_plan = video_generation.get("local_video_plan", {}) if isinstance(video_generation.get("local_video_plan", {}), dict) else {}
        beat_names = [
            clean_text(beat.get("beat_name", ""))
            for beat in local_video_plan.get("beats", [])
            if isinstance(beat, dict) and clean_text(beat.get("beat_name", ""))
        ]
        lines.extend(
            [
                f"[{scene_id}] {clean_text(scene.get('title', ''))}",
                f"image_prompt = {clean_text(image_generation.get('prompt', ''))}",
                f"video_prompt = {clean_text(video_generation.get('prompt', ''))}",
                f"compose_strategy = {clean_text(video_generation.get('compose_strategy', ''))}",
                f"visual_beats = {', '.join(beat_names[:8]) if beat_names else 'n/a'}",
                "",
            ]
        )
    return "\n".join(lines).strip() + ("\n" if lines else "")


def copy_delivery_file(source_path: Path, target_path: Path) -> str:
    if not render_output_ready(source_path):
        return ""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)
    return str(target_path)


def write_delivery_summary(
    target_path: Path,
    *,
    delivery_payload: dict,
    latest_delivery_root: Path,
) -> str:
    lines = [
        "# Finished Episode Delivery Bundle",
        "",
        f"- Episode ID: {clean_text(delivery_payload.get('episode_id', '')) or '-'}",
        f"- Display title: {clean_text(delivery_payload.get('display_title', '')) or '-'}",
        f"- Episode title: {clean_text(delivery_payload.get('episode_title', '')) or '-'}",
        f"- Render mode: {clean_text(delivery_payload.get('render_mode', '')) or '-'}",
        f"- Production readiness: {clean_text(delivery_payload.get('production_readiness', '')) or '-'}",
        "",
        "## Main Outputs",
        "",
        f"- Watch episode: {clean_text(delivery_payload.get('watch_episode', '')) or '-'}",
        f"- Storyboard render copy: {clean_text(delivery_payload.get('storyboard_render_copy', '')) or '-'}",
        f"- Dialogue audio: {clean_text(delivery_payload.get('dialogue_audio', '')) or '-'}",
        f"- Subtitle preview: {clean_text(delivery_payload.get('subtitle_preview', '')) or '-'}",
        f"- Voice plan: {clean_text(delivery_payload.get('voice_plan', '')) or '-'}",
        f"- Render manifest: {clean_text(delivery_payload.get('render_manifest', '')) or '-'}",
        f"- Production package: {clean_text(delivery_payload.get('production_package', '')) or '-'}",
        f"- Backend prompt preview: {clean_text(delivery_payload.get('production_prompt_preview', '')) or '-'}",
        "",
        "## Delivery Paths",
        "",
        f"- Episode delivery folder: {clean_text(delivery_payload.get('delivery_root', '')) or '-'}",
        f"- Stable latest delivery folder: {latest_delivery_root}",
        "",
        "## Remaining Backend Tasks",
        "",
    ]
    remaining_backend_tasks = delivery_payload.get("remaining_backend_tasks", [])
    if isinstance(remaining_backend_tasks, list) and remaining_backend_tasks:
        lines.extend(f"- {clean_text(task)}" for task in remaining_backend_tasks if clean_text(task))
    else:
        lines.append("- none")
    lines.append("")
    write_text(target_path, "\n".join(lines))
    return str(target_path)


def write_latest_episode_delivery_bundle(cfg: dict, episode_id: str, delivery_payload: dict) -> dict:
    latest_root = latest_episode_delivery_bundle_root(cfg)
    latest_root.mkdir(parents=True, exist_ok=True)
    latest_episode_path = latest_root / "latest_finished_episode.mp4"
    latest_storyboard_path = latest_root / "latest_storyboard_render.mp4"
    latest_dialogue_audio_path = latest_root / "latest_dialogue_audio.wav"
    latest_subtitle_path = latest_root / "latest_finished_episode.srt"
    latest_voice_plan_path = latest_root / "latest_voice_plan.json"
    latest_render_manifest_path = latest_root / "latest_render_manifest.json"
    latest_production_package_path = latest_root / "latest_production_package.json"
    latest_prompt_preview_path = latest_root / "latest_backend_prompts.txt"
    latest_delivery_manifest_path = latest_root / "latest_delivery_manifest.json"
    latest_delivery_summary_path = latest_root / "README_finished_episode.md"
    latest_payload = {
        "episode_id": episode_id,
        "display_title": clean_text(delivery_payload.get("display_title", "")),
        "episode_title": clean_text(delivery_payload.get("episode_title", "")),
        "render_mode": clean_text(delivery_payload.get("render_mode", "")),
        "production_readiness": clean_text(delivery_payload.get("production_readiness", "")),
        "delivery_root": str(latest_root),
        "watch_episode": copy_delivery_file(resolve_stored_project_path(delivery_payload.get("watch_episode", "")), latest_episode_path),
        "storyboard_render_copy": copy_delivery_file(resolve_stored_project_path(delivery_payload.get("storyboard_render_copy", "")), latest_storyboard_path),
        "dialogue_audio": copy_delivery_file(resolve_stored_project_path(delivery_payload.get("dialogue_audio", "")), latest_dialogue_audio_path),
        "subtitle_preview": copy_delivery_file(resolve_stored_project_path(delivery_payload.get("subtitle_preview", "")), latest_subtitle_path),
        "voice_plan": copy_delivery_file(resolve_stored_project_path(delivery_payload.get("voice_plan", "")), latest_voice_plan_path),
        "render_manifest": copy_delivery_file(resolve_stored_project_path(delivery_payload.get("render_manifest", "")), latest_render_manifest_path),
        "production_package": copy_delivery_file(resolve_stored_project_path(delivery_payload.get("production_package", "")), latest_production_package_path),
        "production_prompt_preview": copy_delivery_file(resolve_stored_project_path(delivery_payload.get("production_prompt_preview", "")), latest_prompt_preview_path),
        "source_outputs": dict(delivery_payload.get("source_outputs", {}) or {}),
        "scene_master_root": clean_text(delivery_payload.get("scene_master_root", "")),
        "remaining_backend_tasks": list(delivery_payload.get("remaining_backend_tasks", []) or []),
    }
    latest_payload["delivery_summary"] = write_delivery_summary(
        latest_delivery_summary_path,
        delivery_payload=latest_payload,
        latest_delivery_root=latest_root,
    )
    write_json(latest_delivery_manifest_path, latest_payload)
    latest_payload["delivery_manifest"] = str(latest_delivery_manifest_path)
    return latest_payload


def write_episode_delivery_bundle(
    cfg: dict,
    episode_id: str,
    shotlist: dict,
    manifest: dict,
    production_package: dict,
    final_render_path: Path,
    full_generated_episode_path: Path,
    dialogue_audio_path: Path,
    subtitle_preview_path: Path,
    voice_plan_path: Path,
) -> dict:
    delivery_root = episode_delivery_bundle_root(cfg, episode_id)
    delivery_root.mkdir(parents=True, exist_ok=True)
    package_path = resolve_stored_project_path(production_package.get("package_path", ""))
    prompt_preview_path = resolve_stored_project_path(production_package.get("prompt_preview_path", ""))
    render_manifest_path = resolve_stored_project_path(manifest.get("render_manifest_path", ""))
    preferred_episode_video = full_generated_episode_path if render_output_ready(full_generated_episode_path) else final_render_path
    delivery_episode_path = delivery_root / f"{episode_id}_finished_episode.mp4"
    delivery_storyboard_fallback_path = delivery_root / f"{episode_id}_storyboard_render.mp4"
    delivery_dialogue_audio_path = delivery_root / f"{episode_id}_dialogue_audio.wav"
    delivery_subtitle_path = delivery_root / f"{episode_id}.srt"
    delivery_voice_plan_path = delivery_root / f"{episode_id}_voice_plan.json"
    delivery_manifest_path = delivery_root / f"{episode_id}_delivery_manifest.json"
    delivery_production_package_path = delivery_root / f"{episode_id}_production_package.json"
    delivery_prompt_preview_path = delivery_root / f"{episode_id}_backend_prompts.txt"
    delivery_summary_path = delivery_root / "README_finished_episode.md"

    copied_episode_path = copy_delivery_file(preferred_episode_video, delivery_episode_path)
    copied_storyboard_path = ""
    if render_output_ready(final_render_path) and final_render_path.resolve() != preferred_episode_video.resolve():
        copied_storyboard_path = copy_delivery_file(final_render_path, delivery_storyboard_fallback_path)
    elif render_output_ready(final_render_path):
        copied_storyboard_path = copy_delivery_file(final_render_path, delivery_storyboard_fallback_path)

    delivery_payload = {
        "episode_id": episode_id,
        "display_title": clean_text(shotlist.get("display_title", episode_id)),
        "episode_title": clean_text(shotlist.get("episode_title", "")),
        "render_mode": clean_text(manifest.get("render_mode", "")),
        "production_readiness": clean_text((production_package.get("completion_status", {}) if isinstance(production_package.get("completion_status", {}), dict) else {}).get("production_readiness", "")),
        "delivery_root": str(delivery_root),
        "watch_episode": copied_episode_path,
        "storyboard_render_copy": copied_storyboard_path,
        "dialogue_audio": copy_delivery_file(dialogue_audio_path, delivery_dialogue_audio_path),
        "subtitle_preview": copy_delivery_file(subtitle_preview_path, delivery_subtitle_path),
        "voice_plan": copy_delivery_file(voice_plan_path, delivery_voice_plan_path),
        "render_manifest": copy_delivery_file(render_manifest_path, delivery_root / f"{episode_id}_render_manifest.json"),
        "production_package": copy_delivery_file(package_path, delivery_production_package_path),
        "production_prompt_preview": copy_delivery_file(prompt_preview_path, delivery_prompt_preview_path),
        "source_outputs": {
            "final_render": str(final_render_path) if render_output_ready(final_render_path) else "",
            "full_generated_episode": str(full_generated_episode_path) if render_output_ready(full_generated_episode_path) else "",
            "render_manifest": str(render_manifest_path) if render_output_ready(render_manifest_path) else "",
            "production_package": str(package_path) if render_output_ready(package_path) else "",
        },
        "scene_master_root": clean_text(((production_package.get("target_master_outputs", {}) if isinstance(production_package.get("target_master_outputs", {}), dict) else {}).get("scene_master_root", ""))),
        "remaining_backend_tasks": (production_package.get("completion_status", {}) if isinstance(production_package.get("completion_status", {}), dict) else {}).get("remaining_backend_tasks", []),
    }
    latest_delivery_payload = write_latest_episode_delivery_bundle(cfg, episode_id, delivery_payload)
    delivery_payload["delivery_summary"] = write_delivery_summary(
        delivery_summary_path,
        delivery_payload=delivery_payload,
        latest_delivery_root=latest_episode_delivery_bundle_root(cfg),
    )
    delivery_payload["latest_delivery_root"] = latest_delivery_payload.get("delivery_root", "")
    delivery_payload["latest_delivery_manifest"] = latest_delivery_payload.get("delivery_manifest", "")
    delivery_payload["latest_watch_episode"] = latest_delivery_payload.get("watch_episode", "")
    write_json(delivery_manifest_path, delivery_payload)
    delivery_payload["delivery_manifest"] = str(delivery_manifest_path)
    return delivery_payload


def write_episode_production_package(
    cfg: dict,
    episode_id: str,
    shotlist: dict,
    manifest: dict,
    voice_plan_payload: dict,
) -> dict:
    package_root = episode_production_package_root(cfg, episode_id)
    scene_root = package_root / "scenes"
    master_root = package_root / "master"
    package_root.mkdir(parents=True, exist_ok=True)
    scene_root.mkdir(parents=True, exist_ok=True)
    master_root.mkdir(parents=True, exist_ok=True)
    existing_scene_packages: dict[str, dict] = {}
    for existing_scene_path in scene_root.glob("*_production.json"):
        existing_payload = read_json(existing_scene_path, {})
        if not isinstance(existing_payload, dict):
            continue
        existing_scene_id = clean_text(existing_payload.get("scene_id", "")) or clean_text(
            existing_scene_path.stem.removesuffix("_production")
        )
        if existing_scene_id:
            existing_scene_packages[existing_scene_id] = existing_payload
    payload = build_episode_production_package_payload(
        cfg,
        episode_id,
        shotlist,
        manifest,
        voice_plan_payload,
        package_root,
        existing_scene_packages=existing_scene_packages,
    )
    scene_package_paths: list[str] = []
    for scene in payload.get("scenes", []) if isinstance(payload.get("scenes", []), list) else []:
        if not isinstance(scene, dict):
            continue
        scene_id = production_scene_slug(clean_text(scene.get("scene_id", "")))
        scene_path = scene_root / f"{scene_id}_production.json"
        storyboard = scene.get("storyboard", {})
        if isinstance(storyboard, dict):
            storyboard["scene_package_path"] = str(scene_path)
        write_json(scene_path, scene)
        scene_package_paths.append(str(scene_path))
    prompt_preview_path = master_root / f"{episode_id}_backend_prompts.txt"
    package_path = master_root / f"{episode_id}_production_package.json"
    payload["package_root"] = str(package_root)
    payload["package_path"] = str(package_path)
    payload["prompt_preview_path"] = str(prompt_preview_path)
    payload["scene_package_paths"] = scene_package_paths
    payload["prompt_preview"] = str(prompt_preview_path)
    write_text(prompt_preview_path, render_production_prompt_preview(payload.get("scenes", []) if isinstance(payload.get("scenes", []), list) else []))
    write_json(package_path, payload)
    return {
        "package_root": str(package_root),
        "package_path": str(package_path),
        "prompt_preview_path": str(prompt_preview_path),
        "scene_package_paths": scene_package_paths,
    }


def scene_package_target_paths(scene_package: dict) -> dict[str, str]:
    image_generation = scene_package.get("image_generation", {}) if isinstance(scene_package.get("image_generation", {}), dict) else {}
    image_outputs = image_generation.get("target_outputs", {}) if isinstance(image_generation.get("target_outputs", {}), dict) else {}
    video_generation = scene_package.get("video_generation", {}) if isinstance(scene_package.get("video_generation", {}), dict) else {}
    video_outputs = video_generation.get("target_outputs", {}) if isinstance(video_generation.get("target_outputs", {}), dict) else {}
    voice_clone = scene_package.get("voice_clone", {}) if isinstance(scene_package.get("voice_clone", {}), dict) else {}
    voice_outputs = voice_clone.get("target_outputs", {}) if isinstance(voice_clone.get("target_outputs", {}), dict) else {}
    lip_sync = scene_package.get("lip_sync", {}) if isinstance(scene_package.get("lip_sync", {}), dict) else {}
    lipsync_outputs = lip_sync.get("target_outputs", {}) if isinstance(lip_sync.get("target_outputs", {}), dict) else {}
    mastering = scene_package.get("mastering", {}) if isinstance(scene_package.get("mastering", {}), dict) else {}
    mastering_outputs = mastering.get("target_outputs", {}) if isinstance(mastering.get("target_outputs", {}), dict) else {}
    return {
        "primary_frame": clean_text(image_outputs.get("primary_frame", "")),
        "alternate_frame_dir": clean_text(image_outputs.get("alternate_frame_dir", "")),
        "layered_storyboard_frame": clean_text(image_outputs.get("layered_storyboard_frame", "")),
        "scene_video": clean_text(video_outputs.get("scene_video", "")),
        "video_preview_frame": clean_text(video_outputs.get("preview_frame", "")),
        "video_poster_frame": clean_text(video_outputs.get("poster_frame", "")),
        "scene_dialogue_audio": clean_text(voice_outputs.get("scene_dialogue_audio", "")),
        "lipsync_video": clean_text(lipsync_outputs.get("lipsync_video", "")),
        "lipsync_poster_frame": clean_text(lipsync_outputs.get("poster_frame", "")),
        "scene_master_clip": clean_text(mastering_outputs.get("scene_master_clip", "")),
    }


def refresh_scene_package_outputs(scene_package: dict, package_root: Path) -> dict:
    refreshed = dict(scene_package)
    scene_id = clean_text(refreshed.get("scene_id", "")) or "scene"
    current_outputs = refreshed.get("current_generated_outputs", {}) if isinstance(refreshed.get("current_generated_outputs", {}), dict) else {}
    storyboard = refreshed.get("storyboard", {}) if isinstance(refreshed.get("storyboard", {}), dict) else {}
    voice_clone = refreshed.get("voice_clone", {}) if isinstance(refreshed.get("voice_clone", {}), dict) else {}
    lip_sync = refreshed.get("lip_sync", {}) if isinstance(refreshed.get("lip_sync", {}), dict) else {}
    previous_quality = refreshed.get("quality_assessment", {}) if isinstance(refreshed.get("quality_assessment", {}), dict) else {}
    target_paths = scene_package_target_paths(refreshed)
    primary_frame_text = clean_text(target_paths.get("primary_frame", ""))
    alternate_dir_text = clean_text(target_paths.get("alternate_frame_dir", ""))
    layered_storyboard_text = clean_text(target_paths.get("layered_storyboard_frame", ""))
    scene_dialogue_audio_text = clean_text(target_paths.get("scene_dialogue_audio", ""))
    scene_master_clip_text = clean_text(target_paths.get("scene_master_clip", ""))
    lipsync_video_text = clean_text(target_paths.get("lipsync_video", ""))
    primary_frame_path = resolve_stored_project_path(primary_frame_text) if primary_frame_text else Path()
    alternate_dir_path = resolve_stored_project_path(alternate_dir_text) if alternate_dir_text else Path()
    layered_storyboard_path = resolve_stored_project_path(layered_storyboard_text) if layered_storyboard_text else Path()
    scene_dialogue_audio_path = resolve_stored_project_path(scene_dialogue_audio_text) if scene_dialogue_audio_text else Path()
    scene_master_clip_path = resolve_stored_project_path(scene_master_clip_text) if scene_master_clip_text else Path()
    lipsync_video_path = resolve_stored_project_path(lipsync_video_text) if lipsync_video_text else Path()
    voice_lines = voice_clone.get("lines", []) if isinstance(voice_clone.get("lines", []), list) else []
    behavior_constraints = clean_text_list(refreshed.get("behavior_constraints", []))
    dialogue_style_constraints = clean_text_list(refreshed.get("dialogue_style_constraints", []))
    relationship_context = refreshed.get("relationship_context", []) if isinstance(refreshed.get("relationship_context", []), list) else []
    dialogue_sources = refreshed.get("dialogue_sources", []) if isinstance(refreshed.get("dialogue_sources", []), list) else []
    generic_source_count = sum(
        1
        for item in dialogue_sources
        if isinstance(item, dict) and clean_text(item.get("type", "")) == "generated_template"
    )
    generic_template_line_ratio = generic_source_count / max(1, len(dialogue_sources))
    voice_metadata_available = bool(voice_lines) and all(
        isinstance(line, dict)
        and clean_text(line.get("emotion", ""))
        and clean_text(line.get("pace", ""))
        and safe_float(line.get("target_duration_seconds", 0.0), 0.0) > 0.0
        and clean_text(line.get("delivery_notes", ""))
        and isinstance(line.get("voice_reference_priority", []), list)
        for line in voice_lines
    )
    voice_runtime = {}
    for line in voice_lines:
        if isinstance(line, dict) and isinstance(line.get("runtime", {}), dict):
            voice_runtime = line.get("runtime", {})
            if voice_runtime:
                break
    scene_video_source = first_existing_production_scene_video(package_root, scene_id)
    beat_reference_images = current_outputs.get("beat_reference_images", []) if isinstance(current_outputs.get("beat_reference_images", []), list) else []
    local_video_plan = refreshed.get("video_generation", {}).get("local_video_plan", {}) if isinstance(refreshed.get("video_generation", {}), dict) else {}
    if not beat_reference_images and isinstance(local_video_plan, dict):
        beat_reference_images = [
            clean_text(beat.get("reference_image_path", ""))
            for beat in local_video_plan.get("beats", [])
            if isinstance(beat, dict) and clean_text(beat.get("reference_image_path", ""))
        ]
    scene_visual_beat_count = int(current_outputs.get("scene_visual_beat_count", 0) or 0)
    if scene_visual_beat_count <= 0 and isinstance(local_video_plan, dict):
        scene_visual_beat_count = int(local_video_plan.get("beat_count", 0) or 0)
    audio_backend = clean_text(current_outputs.get("audio_backend", ""))
    if render_output_ready(scene_dialogue_audio_path) and bool(voice_clone.get("required", False)):
        if clean_text(voice_runtime.get("engine", "")).lower() == "xtts" and bool(voice_runtime.get("force_voice_cloning", True)):
            audio_backend = "xtts_voice_clone"
        elif not audio_backend:
            audio_backend = "scene_dialogue_audio"
    local_composed_scene_video = bool(current_outputs.get("local_composed_scene_video", False))
    if scene_video_source and scene_video_source[0] in {"generated_lipsync_video", "generated_scene_video"}:
        local_composed_scene_video = False
    refreshed["current_generated_outputs"] = {
        **current_outputs,
        "video_source_type": scene_video_source[0] if scene_video_source else "",
        "video_source_path": str(scene_video_source[1]) if scene_video_source else "",
        "scene_dialogue_audio": str(scene_dialogue_audio_path) if render_output_ready(scene_dialogue_audio_path) else "",
        "scene_master_clip": str(scene_master_clip_path) if render_output_ready(scene_master_clip_path) else "",
        "generated_primary_frame": str(primary_frame_path) if render_output_ready(primary_frame_path) else "",
        "generated_layered_storyboard_frame": str(layered_storyboard_path) if render_output_ready(layered_storyboard_path) else "",
        "generated_lipsync_video": str(lipsync_video_path) if render_output_ready(lipsync_video_path) else "",
        "alternate_frame_dir": str(alternate_dir_path) if alternate_dir_text and alternate_dir_path.exists() else "",
        "has_generated_scene_video": bool(scene_video_source),
        "has_generated_primary_frame": render_output_ready(primary_frame_path),
        "has_scene_dialogue_audio": render_output_ready(scene_dialogue_audio_path),
        "has_scene_master_clip": render_output_ready(scene_master_clip_path),
        "scene_visual_beat_count": scene_visual_beat_count,
        "beat_reference_images": beat_reference_images,
        "has_visual_beat_reference_images": bool(beat_reference_images),
        "local_composed_scene_video": local_composed_scene_video,
        "audio_backend": audio_backend,
    }
    refreshed["quality_assessment"] = merge_scene_regeneration_metadata(
        scene_quality_assessment(
            scene_id=scene_id,
            current_outputs=refreshed.get("current_generated_outputs", {}),
            voice_required=bool(voice_clone.get("required", False)),
            lipsync_required=bool(lip_sync.get("required", False)),
            reference_slot_count=len(storyboard.get("reference_slots", [])) if isinstance(storyboard.get("reference_slots", []), list) else 0,
            continuity_active=bool(storyboard.get("continuity", {})),
            continuity_character_count=len(storyboard.get("character_continuity", [])) if isinstance(storyboard.get("character_continuity", []), list) else 0,
            style_guidance_available=bool(
                isinstance(storyboard.get("style_constraints", {}), dict)
                and (
                    (storyboard.get("style_constraints", {}) or {}).get("positive")
                    or (storyboard.get("style_constraints", {}) or {}).get("guidance")
                )
            ),
            quality_targets_available=bool(storyboard.get("quality_targets", {})),
            behavior_constraints_available=bool(behavior_constraints and dialogue_style_constraints),
            voice_metadata_available=voice_metadata_available,
            relationship_context_available=bool(relationship_context),
            scene_conflict_available=bool(clean_text(refreshed.get("conflict", ""))),
            generic_template_line_ratio=generic_template_line_ratio,
        ),
        previous_quality,
    )
    return refreshed


def refresh_episode_production_package(
    package_path: Path,
    *,
    full_generated_episode_path: Path | None = None,
    backend_runner_summary: dict | None = None,
) -> dict:
    payload = read_json(package_path, {})
    if not isinstance(payload, dict):
        return {}
    package_root_text = clean_text(payload.get("package_root", ""))
    package_root = resolve_stored_project_path(package_root_text) if package_root_text else package_path.parent.parent
    scene_package_paths = payload.get("scene_package_paths", []) if isinstance(payload.get("scene_package_paths", []), list) else []
    refreshed_scenes: list[dict] = []
    resolved_scene_paths: list[str] = []
    for index, scene_package_path_raw in enumerate(scene_package_paths):
        scene_package_path = resolve_stored_project_path(scene_package_path_raw)
        if scene_package_path is None or not scene_package_path.exists():
            scene_payload = payload.get("scenes", [])[index] if index < len(payload.get("scenes", [])) and isinstance(payload.get("scenes", []), list) else {}
            if not isinstance(scene_payload, dict):
                continue
            refreshed_scene = refresh_scene_package_outputs(scene_payload, package_root)
            refreshed_scenes.append(refreshed_scene)
            continue
        scene_payload = read_json(scene_package_path, {})
        if not isinstance(scene_payload, dict):
            continue
        refreshed_scene = refresh_scene_package_outputs(scene_payload, package_root)
        write_json(scene_package_path, refreshed_scene)
        refreshed_scenes.append(refreshed_scene)
        resolved_scene_paths.append(str(scene_package_path))

    completion_status = generated_episode_completion_summary(
        scene_count=len(refreshed_scenes),
        generated_scene_video_count=sum(
            1
            for scene in refreshed_scenes
            if isinstance(scene.get("current_generated_outputs", {}), dict)
            and bool(scene["current_generated_outputs"].get("has_generated_scene_video", False))
        ),
        scene_dialogue_audio_count=sum(
            1
            for scene in refreshed_scenes
            if isinstance(scene.get("current_generated_outputs", {}), dict)
            and bool(scene["current_generated_outputs"].get("has_scene_dialogue_audio", False))
        ),
        scene_master_clip_count=sum(
            1
            for scene in refreshed_scenes
            if isinstance(scene.get("current_generated_outputs", {}), dict)
            and bool(scene["current_generated_outputs"].get("has_scene_master_clip", False))
        ),
        render_mode=clean_text(payload.get("render_mode", "")),
        final_render=clean_text((payload.get("current_preview_outputs", {}) if isinstance(payload.get("current_preview_outputs", {}), dict) else {}).get("final_storyboard_render", "")),
        full_generated_episode=str(full_generated_episode_path) if full_generated_episode_path and render_output_ready(full_generated_episode_path) else "",
    )
    quality_assessment = episode_quality_assessment(
        [scene.get("quality_assessment", {}) for scene in refreshed_scenes if isinstance(scene, dict)],
        scene_count=len(refreshed_scenes),
    )
    payload["scenes"] = refreshed_scenes
    payload["scene_count"] = len(refreshed_scenes)
    payload["scene_package_paths"] = resolved_scene_paths or scene_package_paths
    payload["line_count"] = sum(
        len(scene.get("voice_clone", {}).get("lines", []))
        for scene in refreshed_scenes
        if isinstance(scene.get("voice_clone", {}), dict)
    )
    payload["completion_status"] = completion_status
    payload["quality_assessment"] = quality_assessment
    payload["package_root"] = str(package_root)
    payload["package_path"] = str(package_path)
    if backend_runner_summary:
        payload["backend_runner_summary"] = backend_runner_summary
    write_json(package_path, payload)
    return payload


def build_scene_runner_context(
    scene_package: dict,
    scene_package_path: Path,
    package_path: Path,
    prompt_preview_path: Path,
    package_root: Path,
) -> dict[str, object]:
    target_paths = scene_package_target_paths(scene_package)
    scene_id = clean_text(scene_package.get("scene_id", "")) or scene_package_path.stem
    current_outputs = scene_package.get("current_generated_outputs", {}) if isinstance(scene_package.get("current_generated_outputs", {}), dict) else {}
    current_preview_assets = scene_package.get("current_preview_assets", {}) if isinstance(scene_package.get("current_preview_assets", {}), dict) else {}
    voice_clone = scene_package.get("voice_clone", {}) if isinstance(scene_package.get("voice_clone", {}), dict) else {}
    lip_sync = scene_package.get("lip_sync", {}) if isinstance(scene_package.get("lip_sync", {}), dict) else {}
    video_generation = scene_package.get("video_generation", {}) if isinstance(scene_package.get("video_generation", {}), dict) else {}
    image_generation = scene_package.get("image_generation", {}) if isinstance(scene_package.get("image_generation", {}), dict) else {}
    return {
        "episode_id": clean_text(scene_package.get("episode_id", "")),
        "scene_id": scene_id,
        "scene_title": clean_text(scene_package.get("title", "")),
        "scene_summary": clean_text(scene_package.get("summary", "")),
        "scene_package_path": scene_package_path,
        "scene_dir": scene_package_path.parent,
        "package_path": package_path,
        "package_root": package_root,
        "prompt_preview_path": prompt_preview_path,
        "preview_frame_path": clean_text(current_preview_assets.get("preview_frame_path", "")),
        "preview_asset_path": clean_text(current_preview_assets.get("asset_source_path", "")),
        "current_video_source_path": clean_text(current_outputs.get("video_source_path", "")),
        "image_prompt": clean_text(image_generation.get("prompt", "")),
        "video_prompt": clean_text(video_generation.get("prompt", "")),
        "speaker_targets": ",".join(lip_sync.get("speaker_targets", [])) if isinstance(lip_sync.get("speaker_targets", []), list) else "",
        "voice_line_count": len(voice_clone.get("lines", [])) if isinstance(voice_clone.get("lines", []), list) else 0,
        **target_paths,
    }


def run_finished_episode_scene_runners(
    cfg: dict,
    package_path: Path,
    prompt_preview_path: Path,
    *,
    force: bool,
) -> dict[str, object]:
    package_payload = read_json(package_path, {})
    if not isinstance(package_payload, dict):
        return {"scene_results": []}
    package_root_text = clean_text(package_payload.get("package_root", ""))
    package_root = resolve_stored_project_path(package_root_text) if package_root_text else package_path.parent.parent
    scene_results: list[dict] = []
    scene_package_paths = package_payload.get("scene_package_paths", []) if isinstance(package_payload.get("scene_package_paths", []), list) else []
    runner_specs = [
        ("finished_episode_image_runner", "image_generation"),
        ("finished_episode_video_runner", "video_generation"),
        ("finished_episode_voice_runner", "voice_clone"),
        ("finished_episode_lipsync_runner", "lip_sync"),
    ]
    for scene_package_path_raw in scene_package_paths:
        scene_package_path = resolve_stored_project_path(scene_package_path_raw)
        if scene_package_path is None or not scene_package_path.exists():
            continue
        scene_package = read_json(scene_package_path, {})
        if not isinstance(scene_package, dict):
            continue
        context = build_scene_runner_context(scene_package, scene_package_path, package_path, prompt_preview_path, package_root)
        runner_rows: list[dict] = []
        backend_manifests = scene_package.get("backend_manifests", {}) if isinstance(scene_package.get("backend_manifests", {}), dict) else {}
        backend_manifest_rows = scene_package.get("backend_manifest_rows", []) if isinstance(scene_package.get("backend_manifest_rows", []), list) else []
        for runner_name, section_name in runner_specs:
            section_payload = scene_package.get(section_name, {}) if isinstance(scene_package.get(section_name, {}), dict) else {}
            if not bool(section_payload.get("required", False)):
                continue
            if force:
                remove_stale_scene_runner_outputs(section_name, context)
            runner_rows.append(
                run_external_backend_runner(
                    cfg,
                    runner_name,
                    context=context,
                    force=force,
                    fallback_cwd=scene_package_path.parent,
                    log_dir=resolve_project_path("logs") / "external_backends" / runner_name / clean_text(scene_package.get("episode_id", "")) / clean_text(scene_package.get("scene_id", "")),
                )
            )
            manifest_path = write_backend_task_manifest(
                package_root=package_root,
                scene_package=scene_package,
                scene_package_path=scene_package_path,
                runner_name=runner_name,
                section_name=section_name,
                runner_result=runner_rows[-1],
            )
            backend_manifests[runner_name] = manifest_path
            backend_manifest_rows.append(
                {
                    "runner_name": runner_name,
                    "section": section_name,
                    "manifest_path": manifest_path,
                    "status": clean_text(runner_rows[-1].get("status", "")),
                }
            )
        if backend_manifests:
            scene_package["backend_manifests"] = backend_manifests
            scene_package["backend_manifest_rows"] = backend_manifest_rows
            write_json(scene_package_path, scene_package)
        scene_results.append(
            {
                "scene_id": clean_text(scene_package.get("scene_id", "")),
                "scene_package_path": str(scene_package_path),
                "runner_results": runner_rows,
            }
        )
    summary_path = package_path.parent / f"{package_path.stem}_backend_runners.json"
    summary = {
        "package_path": str(package_path),
        "prompt_preview_path": str(prompt_preview_path),
        "force": bool(force),
        "scene_results": scene_results,
    }
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def run_finished_episode_master_runner(
    cfg: dict,
    package_payload: dict,
    *,
    force: bool,
) -> dict[str, object]:
    package_path_text = clean_text(package_payload.get("package_path", ""))
    if not package_path_text:
        return {"status": "missing_package"}
    package_path = resolve_stored_project_path(package_path_text)
    package_root_text = clean_text(package_payload.get("package_root", ""))
    package_root = resolve_stored_project_path(package_root_text) if package_root_text else package_path.parent.parent
    prompt_preview_text = clean_text(package_payload.get("prompt_preview_path", ""))
    prompt_preview_path = resolve_stored_project_path(prompt_preview_text) if prompt_preview_text else Path()
    target_master_outputs = package_payload.get("target_master_outputs", {}) if isinstance(package_payload.get("target_master_outputs", {}), dict) else {}
    if force:
        remove_stale_generated_output(target_master_outputs.get("final_master_episode", ""))
    result = run_external_backend_runner(
        cfg,
        "finished_episode_master_runner",
        context={
            "episode_id": clean_text(package_payload.get("episode_id", "")),
            "package_path": package_path,
            "package_root": package_root,
            "prompt_preview_path": prompt_preview_path or "",
            "final_master_episode": clean_text(target_master_outputs.get("final_master_episode", "")),
            "video_root": clean_text(target_master_outputs.get("video_root", "")),
            "audio_root": clean_text(target_master_outputs.get("audio_root", "")),
            "lipsync_root": clean_text(target_master_outputs.get("lipsync_root", "")),
            "scene_master_root": clean_text(target_master_outputs.get("scene_master_root", "")),
        },
        force=force,
        fallback_cwd=package_root,
        log_dir=resolve_project_path("logs") / "external_backends" / "finished_episode_master_runner" / clean_text(package_payload.get("episode_id", "")),
    )
    summary_path = package_path.parent / f"{package_path.stem}_master_runner.json"
    manifest_root = package_root / "manifests" / "master"
    manifest_root.mkdir(parents=True, exist_ok=True)
    master_manifest_path = manifest_root / "finished_episode_master_runner.json"
    produced_outputs = [clean_text(path) for path in result.get("produced_outputs", []) if clean_text(path)] if isinstance(result.get("produced_outputs", []), list) else []
    write_json(
        master_manifest_path,
        {
            "task_id": f"{clean_text(package_payload.get('episode_id', 'episode'))}_finished_episode_master_runner",
            "scene_id": "",
            "shot_id": "",
            "task_type": "master",
            "backend": "finished_episode_master_runner",
            "command": result.get("command", []) if isinstance(result.get("command", []), list) else result.get("command_text", ""),
            "inputs": {"package_path": str(package_path)},
            "outputs": {"final_master_episode": clean_text(target_master_outputs.get("final_master_episode", ""))},
            "produced_outputs": produced_outputs,
            "output_hashes": {path: file_sha256(path) for path in produced_outputs if file_sha256(path)},
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "duration_seconds": 0,
            "exit_code": int(result.get("returncode", 0) or 0),
            "status": "success" if clean_text(result.get("status", "")) == "completed" else "skipped" if clean_text(result.get("status", "")) == "existing_outputs" else "failed",
            "runner_status": clean_text(result.get("status", "")),
            "fallback_used": backend_runner_fallback_used(result),
            "placeholder_used": clean_text(result.get("status", "")) not in {"completed", "existing_outputs"},
            "stale_output": clean_text(result.get("status", "")) == "existing_outputs",
            "log_path": clean_text(result.get("log_path", "")),
        },
    )
    write_json(summary_path, result)
    result["summary_path"] = str(summary_path)
    result["backend_manifest"] = str(master_manifest_path)
    return result


def collect_scene_dialogue_outputs_from_package(package_payload: dict) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for scene in package_payload.get("scenes", []) if isinstance(package_payload.get("scenes", []), list) else []:
        if not isinstance(scene, dict):
            continue
        scene_id = clean_text(scene.get("scene_id", ""))
        current_outputs = scene.get("current_generated_outputs", {}) if isinstance(scene.get("current_generated_outputs", {}), dict) else {}
        scene_dialogue_audio = clean_text(current_outputs.get("scene_dialogue_audio", ""))
        if scene_id and scene_dialogue_audio:
            outputs[scene_id] = scene_dialogue_audio
    return outputs


def normalized_scene_dialogue_outputs(audio_track_meta: object) -> dict[str, str]:
    if not isinstance(audio_track_meta, dict):
        return {}
    outputs = audio_track_meta.get("scene_dialogue_outputs", {})
    if not isinstance(outputs, dict):
        return {}
    return {
        clean_text(scene_id): clean_text(path)
        for scene_id, path in outputs.items()
        if clean_text(scene_id) and clean_text(path)
    }


def voice_supports_language(voice: dict, language_code: str) -> bool:
    normalized = normalize_language_code(language_code)
    if not normalized:
        return False
    languages = [normalize_language_code(value) for value in voice.get("languages", [])] if isinstance(voice.get("languages", []), list) else []
    if any(language == normalized or language.startswith(f"{normalized}-") for language in languages):
        return True
    voice_name = str(voice.get("name", "") or "").lower()
    aliases = {
        "de": ("german", "deutsch"),
        "en": ("english", "englisch"),
        "fr": ("french", "francais"),
        "es": ("spanish", "espanol"),
        "it": ("italian",),
        "pt": ("portuguese",),
    }
    return any(alias in voice_name for alias in aliases.get(normalized, ()))


def resolve_system_voice_id(default_voice_id: str, voices: list[dict], preferred_language: str = "", require_german: bool = False) -> str:
    target_language = normalize_language_code(preferred_language)
    if not target_language and require_german:
        target_language = "de"
    if not target_language:
        return default_voice_id
    for voice in voices:
        if str(voice.get("id", "")) == default_voice_id and voice_supports_language(voice, target_language):
            return default_voice_id
    for voice in voices:
        if voice_supports_language(voice, target_language):
            return str(voice.get("id", default_voice_id))
    return default_voice_id


def describe_system_voices(engine) -> list[dict]:
    payloads: list[dict] = []
    for voice in engine.getProperty("voices") or []:
        languages = []
        for language in getattr(voice, "languages", []) or []:
            text = str(language)
            text = text.replace("b'\\x05", "").replace("b'\\x02", "").replace("'", "")
            languages.append(text)
        lowered_languages = [value.lower() for value in languages]
        voice_name = str(getattr(voice, "name", "") or "")
        payloads.append(
            {
                "id": str(getattr(voice, "id", "") or ""),
                "name": voice_name,
                "languages": languages,
                "is_german": "german" in voice_name.lower() or any(language.startswith("de") for language in lowered_languages),
            }
        )
    return payloads


def build_audio_segment_plan(voice_plan_lines: list[dict], total_duration: float) -> list[dict]:
    duration_limit = max(0.0, float(total_duration or 0.0))
    segments: list[dict] = []
    cursor = 0.0
    for line in sorted(voice_plan_lines, key=lambda item: float(item.get("start_seconds", 0.0) or 0.0)):
        start_seconds = max(0.0, min(duration_limit, float(line.get("start_seconds", 0.0) or 0.0)))
        end_seconds = max(start_seconds, min(duration_limit, float(line.get("end_seconds", start_seconds) or start_seconds)))
        line_duration = max(0.0, end_seconds - start_seconds)
        gap_duration = max(0.0, start_seconds - cursor)
        if gap_duration > 0.01:
            segments.append({"kind": "silence", "duration_seconds": round(gap_duration, 3)})
        if line_duration > 0.01:
            segments.append(
                {
                    "kind": "line",
                    "line_index": int(line.get("line_index", 0) or 0),
                    "duration_seconds": round(line_duration, 3),
                }
            )
        cursor = max(cursor, end_seconds)
    tail_duration = max(0.0, duration_limit - cursor)
    if tail_duration > 0.01 or not segments:
        segments.append({"kind": "silence", "duration_seconds": round(tail_duration if segments else duration_limit, 3)})
    return segments


def synthesize_voice_lines(temp_root: Path, voice_plan_lines: list[dict], render_cfg: dict) -> tuple[dict[int, Path], dict]:
    try:
        import pyttsx3
    except Exception as exc:
        raise RuntimeError("pyttsx3 is not available for voiced final episode rendering.") from exc

    sample_rate = int(render_cfg.get("audio_sample_rate", 22050) or 22050)
    engine = pyttsx3.init()
    voices = describe_system_voices(engine)
    configured_voice_id = clean_text(render_cfg.get("voice_id", ""))
    default_voice_id = configured_voice_id or clean_text(engine.getProperty("voice"))
    voice_rate = int(render_cfg.get("voice_rate", 175) or 175)
    voice_volume = float(render_cfg.get("voice_volume", 1.0) or 1.0)
    output_map: dict[int, Path] = {}
    line_voice_ids: dict[int, str] = {}
    line_languages: dict[int, str] = {}
    prefer_character_language = bool(render_cfg.get("prefer_detected_character_language", True))
    try:
        engine.setProperty("rate", voice_rate)
        engine.setProperty("volume", max(0.0, min(1.0, voice_volume)))
        for line in voice_plan_lines:
            line_index = int(line.get("line_index", 0) or 0)
            text = dialogue_line_text(line)
            if not text:
                continue
            voice_profile = line.get("voice_profile", {}) if isinstance(line.get("voice_profile", {}), dict) else {}
            preferred_language = ""
            if prefer_character_language:
                preferred_language = normalize_language_code(
                    line.get("language", "") or voice_profile.get("dominant_language", "")
                )
            voice_id = resolve_system_voice_id(
                default_voice_id,
                voices,
                preferred_language=preferred_language,
                require_german=bool(render_cfg.get("prefer_german_voice", False)),
            )
            if voice_id:
                engine.setProperty("voice", voice_id)
            target = temp_root / f"line_{line_index:04d}_raw.wav"
            engine.save_to_file(text, str(target))
            output_map[line_index] = target
            line_voice_ids[line_index] = voice_id
            line_languages[line_index] = preferred_language
        engine.runAndWait()
    finally:
        try:
            engine.stop()
        except Exception:
            pass

    missing = [index for index, path in output_map.items() if not path.exists() or path.stat().st_size <= 0]
    if missing:
        line_lookup = {
            int(line.get("line_index", 0) or 0): line
            for line in voice_plan_lines
            if isinstance(line, dict)
        }
        for line_index in list(missing):
            line = line_lookup.get(line_index, {})
            target = output_map.get(line_index, temp_root / f"line_{line_index:04d}_raw.wav")
            retry_voice_id = line_voice_ids.get(line_index, default_voice_id)
            retry_language = line_languages.get(line_index, "")
            if synthesize_single_voice_line(target, line, render_cfg, voice_id=retry_voice_id, preferred_language=retry_language):
                output_map[line_index] = target
                missing.remove(line_index)
        if missing:
            raise RuntimeError(f"Voice synthesis failed for {len(missing)} dialogue lines.")
    return {
        **output_map,
    }, {
        "backend": "pyttsx3",
        "voice_id": line_voice_ids.get(next(iter(line_voice_ids), -1), default_voice_id) if line_voice_ids else default_voice_id,
        "sample_rate": sample_rate,
        "voices": voices,
        "line_voice_ids": line_voice_ids,
        "line_languages": line_languages,
    }


def dialogue_line_text(line: dict) -> str:
    if not isinstance(line, dict):
        return ""
    retrieval_segment = line.get("retrieval_segment", {}) if isinstance(line.get("retrieval_segment", {}), dict) else {}
    source_payload = line.get("source", {}) if isinstance(line.get("source", {}), dict) else {}
    return (
        clean_text(line.get("text", ""))
        or clean_text(source_payload.get("text", ""))
        or clean_text(retrieval_segment.get("text", ""))
    )


def synthesize_single_voice_line(
    target: Path,
    line: dict,
    render_cfg: dict,
    *,
    voice_id: str = "",
    preferred_language: str = "",
) -> bool:
    text = dialogue_line_text(line)
    if not text:
        return False
    try:
        import pyttsx3
    except Exception:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    engine = None
    try:
        engine = pyttsx3.init()
        voices = describe_system_voices(engine)
        configured_voice_id = clean_text(render_cfg.get("voice_id", ""))
        default_voice_id = voice_id or configured_voice_id or clean_text(engine.getProperty("voice"))
        voice_profile = line.get("voice_profile", {}) if isinstance(line.get("voice_profile", {}), dict) else {}
        effective_language = preferred_language
        if not effective_language and bool(render_cfg.get("prefer_detected_character_language", True)):
            effective_language = normalize_language_code(
                line.get("language", "") or voice_profile.get("dominant_language", "")
            )
        resolved_voice_id = resolve_system_voice_id(
            default_voice_id,
            voices,
            preferred_language=effective_language,
            require_german=bool(render_cfg.get("prefer_german_voice", False)),
        )
        engine.setProperty("rate", int(render_cfg.get("voice_rate", 175) or 175))
        engine.setProperty("volume", max(0.0, min(1.0, float(render_cfg.get("voice_volume", 1.0) or 1.0))))
        if resolved_voice_id:
            engine.setProperty("voice", resolved_voice_id)
        engine.save_to_file(text, str(target))
        engine.runAndWait()
        return target.exists() and target.stat().st_size > 0
    except Exception:
        return False
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass


def create_silence_audio(ffmpeg: Path, duration_seconds: float, output_path: Path, sample_rate: int) -> None:
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={sample_rate}:cl=mono",
        "-t",
        f"{max(0.01, float(duration_seconds or 0.0)):.3f}",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_path),
    ]
    result = run_media_command(command)
    if result.returncode != 0 or not render_output_ready(output_path):
        raise RuntimeError(f"Could not create silence audio segment {output_path.name}.")


def dialogue_audio_filter(duration_seconds: float) -> str:
    duration = max(0.01, float(duration_seconds or 0.0))
    fade_duration = min(0.045, max(0.0, duration / 3.0))
    filters = [
        "apad",
        f"atrim=0:{duration:.3f}",
        "asetpts=N/SR/TB",
        "loudnorm=I=-18:TP=-2:LRA=11",
    ]
    if fade_duration >= 0.005:
        filters.append(f"afade=t=in:st=0:d={fade_duration:.3f}")
        filters.append(f"afade=t=out:st={max(0.0, duration - fade_duration):.3f}:d={fade_duration:.3f}")
    return ",".join(filters)


def normalize_line_audio(ffmpeg: Path, input_path: Path, duration_seconds: float, output_path: Path, sample_rate: int) -> None:
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-af",
        dialogue_audio_filter(duration_seconds),
        "-t",
        f"{max(0.01, float(duration_seconds or 0.0)):.3f}",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_path),
    ]
    result = run_media_command(command)
    if result.returncode != 0 or not render_output_ready(output_path):
        raise RuntimeError(f"Could not normalize dialogue audio {input_path.name}.")


def extract_clip_audio(
    ffmpeg: Path,
    input_path: Path,
    start_seconds: float,
    duration_seconds: float,
    output_path: Path,
    sample_rate: int,
) -> None:
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-ss",
        f"{max(0.0, float(start_seconds or 0.0)):.3f}",
        "-i",
        str(input_path),
        "-af",
        dialogue_audio_filter(duration_seconds),
        "-t",
        f"{max(0.01, float(duration_seconds or 0.0)):.3f}",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_path),
    ]
    result = run_media_command(command)
    if result.returncode != 0 or not render_output_ready(output_path):
        raise RuntimeError(f"Could not extract dialogue audio from {input_path.name}.")


def materialize_original_segment_audio(
    ffmpeg: Path,
    retrieval_segment: dict,
    duration_seconds: float,
    output_path: Path,
    sample_rate: int,
) -> str:
    source_audio_path = resolve_stored_project_path(retrieval_segment.get("audio_path", ""))
    if valid_media_source_path(source_audio_path):
        normalize_line_audio(ffmpeg, source_audio_path, duration_seconds, output_path, sample_rate)
        return "original_segment_audio"

    scene_clip_path = resolve_stored_project_path(retrieval_segment.get("scene_clip_path", ""))
    if valid_media_source_path(scene_clip_path):
        extract_clip_audio(
            ffmpeg,
            scene_clip_path,
            float(retrieval_segment.get("start", 0.0) or 0.0),
            duration_seconds,
            output_path,
            sample_rate,
        )
        return "scene_clip_extract"
    return ""


def concat_audio_segments(ffmpeg: Path, segment_paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        concat_path = Path(handle.name)
        for segment_path in segment_paths:
            escaped = str(segment_path).replace("'", "''")
            handle.write(f"file '{escaped}'\n")
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-c",
        "copy",
        str(output_path),
    ]
    try:
        result = run_media_command(command)
        if result.returncode != 0 or not render_output_ready(output_path):
            raise RuntimeError(f"Could not assemble final episode audio {output_path.name}.")
    finally:
        try:
            concat_path.unlink()
        except FileNotFoundError:
            pass


def mux_episode_audio(ffmpeg: Path, video_path: Path, audio_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        str(output_path),
    ]
    result = run_media_command(command)
    if result.returncode != 0 or not render_output_ready(output_path):
        raise RuntimeError(f"Could not mux final voiced episode {output_path.name}.")


def materialize_scene_master_clips(
    ffmpeg: Path,
    manifest_scenes: list[dict],
    scene_dialogue_outputs: dict[str, str],
    package_root: Path,
) -> dict[str, str]:
    master_outputs: dict[str, str] = {}
    for scene in manifest_scenes:
        if not isinstance(scene, dict):
            continue
        scene_id = clean_text(scene.get("scene_id", ""))
        if not scene_id:
            continue
        production_scene_video = first_existing_production_scene_video(package_root, scene_id)
        final_clip_path = (
            production_scene_video[1]
            if production_scene_video is not None and render_output_ready(production_scene_video[1])
            else resolve_stored_project_path(scene.get("final_clip_path", ""))
        )
        if not render_output_ready(final_clip_path):
            continue
        scene_master_path = scene_master_clip_output_path(package_root, scene_id)
        scene_master_path.parent.mkdir(parents=True, exist_ok=True)
        scene_audio_path = resolve_stored_project_path(scene_dialogue_outputs.get(scene_id, ""))
        if render_output_ready(scene_audio_path):
            mux_episode_audio(ffmpeg, final_clip_path, scene_audio_path, scene_master_path)
        else:
            shutil.copyfile(final_clip_path, scene_master_path)
        master_outputs[scene_id] = str(scene_master_path)
    return master_outputs


def materialize_scene_dialogue_tracks(
    ffmpeg: Path,
    voice_plan_scenes: list[dict],
    line_output_map: dict[int, Path],
    sample_rate: int,
    package_root: Path,
) -> dict[str, str]:
    scene_outputs: dict[str, str] = {}
    for scene in voice_plan_scenes:
        if not isinstance(scene, dict):
            continue
        scene_id = clean_text(scene.get("scene_id", ""))
        if not scene_id:
            continue
        scene_duration = float(scene.get("duration_seconds", 0.0) or 0.0)
        scene_start = float(scene.get("scene_start_seconds", 0.0) or 0.0)
        local_lines: list[dict] = []
        for line in scene.get("lines", []) if isinstance(scene.get("lines", []), list) else []:
            if not isinstance(line, dict):
                continue
            line_index = int(line.get("line_index", 0) or 0)
            line_output = line_output_map.get(line_index)
            if line_output is None or not line_output.exists():
                continue
            start_seconds = max(0.0, float(line.get("start_seconds", 0.0) or 0.0) - scene_start)
            end_seconds = max(start_seconds, float(line.get("end_seconds", start_seconds) or start_seconds) - scene_start)
            local_lines.append(
                {
                    "line_index": line_index,
                    "start_seconds": round(start_seconds, 3),
                    "end_seconds": round(min(scene_duration, end_seconds), 3),
                }
            )
        if not local_lines and scene_duration <= 0.0:
            continue
        segment_plan = build_audio_segment_plan(local_lines, scene_duration)
        output_path = scene_dialogue_output_audio_path(package_root, scene_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        concat_segments: list[Path] = []
        silence_index = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            for segment in segment_plan:
                if segment.get("kind") == "silence":
                    duration_seconds = float(segment.get("duration_seconds", 0.0) or 0.0)
                    if duration_seconds <= 0.01:
                        continue
                    silence_path = temp_root / f"silence_{silence_index:04d}.wav"
                    create_silence_audio(ffmpeg, duration_seconds, silence_path, sample_rate)
                    concat_segments.append(silence_path)
                    silence_index += 1
                    continue
                line_index = int(segment.get("line_index", 0) or 0)
                line_output = line_output_map.get(line_index)
                if line_output is not None and line_output.exists():
                    concat_segments.append(line_output)
            if not concat_segments:
                silence_path = temp_root / "silence_full.wav"
                create_silence_audio(ffmpeg, max(scene_duration, 0.25), silence_path, sample_rate)
                concat_segments.append(silence_path)
            concat_audio_segments(ffmpeg, concat_segments, output_path)
        scene_outputs[scene_id] = str(output_path)
    return scene_outputs


def materialize_episode_audio_track_from_scene_outputs(
    ffmpeg: Path,
    voice_plan_scenes: list[dict],
    total_duration: float,
    scene_output_map: dict[str, str],
    sample_rate: int,
    output_path: Path,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_segments: list[Path] = []
    cursor = 0.0
    scene_audio_count = 0
    silence_index = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_root = Path(tmpdir)
        for scene in sorted(
            [row for row in voice_plan_scenes if isinstance(row, dict)],
            key=lambda item: float(item.get("scene_start_seconds", 0.0) or 0.0),
        ):
            scene_id = clean_text(scene.get("scene_id", ""))
            scene_start = max(0.0, float(scene.get("scene_start_seconds", 0.0) or 0.0))
            scene_duration = max(0.0, float(scene.get("duration_seconds", 0.0) or 0.0))
            if scene_start > cursor + 0.01:
                silence_path = temp_root / f"gap_{silence_index:04d}.wav"
                create_silence_audio(ffmpeg, scene_start - cursor, silence_path, sample_rate)
                concat_segments.append(silence_path)
                silence_index += 1
            scene_audio_path = resolve_stored_project_path(scene_output_map.get(scene_id, ""))
            if render_output_ready(scene_audio_path) and scene_duration > 0.01:
                normalized_scene_path = temp_root / f"scene_{len(concat_segments):04d}.wav"
                normalize_line_audio(ffmpeg, scene_audio_path, scene_duration, normalized_scene_path, sample_rate)
                concat_segments.append(normalized_scene_path)
                scene_audio_count += 1
            elif scene_duration > 0.01:
                silence_path = temp_root / f"scene_silence_{silence_index:04d}.wav"
                create_silence_audio(ffmpeg, scene_duration, silence_path, sample_rate)
                concat_segments.append(silence_path)
                silence_index += 1
            cursor = max(cursor, scene_start + scene_duration)
        tail_duration = max(0.0, float(total_duration or 0.0) - cursor)
        if tail_duration > 0.01:
            silence_path = temp_root / f"tail_{silence_index:04d}.wav"
            create_silence_audio(ffmpeg, tail_duration, silence_path, sample_rate)
            concat_segments.append(silence_path)
        if not concat_segments:
            create_silence_audio(ffmpeg, max(float(total_duration or 0.0), 0.25), output_path, sample_rate)
        else:
            concat_audio_segments(ffmpeg, concat_segments, output_path)
    return {
        "audio_path": str(output_path),
        "scene_audio_count": scene_audio_count,
        "scene_count": len([row for row in voice_plan_scenes if isinstance(row, dict)]),
    }


def render_episode_audio_track(
    ffmpeg: Path,
    voice_plan_lines: list[dict],
    total_duration: float,
    render_cfg: dict,
    temp_root: Path,
    output_path: Path,
    package_root: Path | None = None,
    voice_plan_scenes: list[dict] | None = None,
) -> dict:
    audio_root = temp_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    sample_rate = int(render_cfg.get("audio_sample_rate", 22050) or 22050)
    reusable_indexes: set[int] = set()
    for line in voice_plan_lines:
        if not isinstance(line, dict):
            continue
        retrieval_segment = line.get("retrieval_segment", {}) if isinstance(line.get("retrieval_segment", {}), dict) else {}
        source_audio_path = resolve_stored_project_path(retrieval_segment.get("audio_path", ""))
        scene_clip_path = resolve_stored_project_path(retrieval_segment.get("scene_clip_path", ""))
        if valid_media_source_path(source_audio_path) or valid_media_source_path(scene_clip_path):
            reusable_indexes.add(int(line.get("line_index", 0) or 0))
    tts_lines = [line for line in voice_plan_lines if int(line.get("line_index", 0) or 0) not in reusable_indexes]
    synthesized_map: dict[int, Path] = {}
    voice_meta: dict[str, object] = {"backend": "", "voice_id": "", "voices": []}
    if tts_lines:
        if bool(render_cfg.get("force_voice_cloning", False)) and not bool(render_cfg.get("allow_system_tts_fallback", False)):
            raise RuntimeError(
                f"Voice cloning is required, but {len(tts_lines)} dialogue line(s) do not have cloned audio yet. "
                "System TTS fallback is disabled; finished_episode_voice_runner must create the scene dialogue tracks."
            )
        synthesized_map, voice_meta = synthesize_voice_lines(audio_root, tts_lines, render_cfg)
    segment_plan = build_audio_segment_plan(voice_plan_lines, total_duration)
    normalized_map: dict[int, Path] = {}
    packaged_line_map: dict[int, Path] = {}
    line_materializations: list[dict[str, object]] = []
    reused_original_lines = 0
    synthesized_lines = 0
    for line in voice_plan_lines:
        line_index = int(line.get("line_index", 0) or 0)
        if line_index in normalized_map:
            continue
        normalized_path = audio_root / f"line_{line_index:04d}.wav"
        line_duration = float(line.get("estimated_duration_seconds", 0.0) or 0.0)
        retrieval_segment = line.get("retrieval_segment", {}) if isinstance(line.get("retrieval_segment", {}), dict) else {}
        source_backend = materialize_original_segment_audio(
            ffmpeg,
            retrieval_segment,
            line_duration,
            normalized_path,
            sample_rate,
        )
        if source_backend:
            reused_original_lines += 1
        else:
            synthesized_source = synthesized_map.get(line_index)
            if synthesized_source is None:
                retry_source = audio_root / f"line_{line_index:04d}_retry.wav"
                if synthesize_single_voice_line(retry_source, line, render_cfg):
                    synthesized_source = retry_source
                    synthesized_map[line_index] = retry_source
                else:
                    line_text = dialogue_line_text(line)
                    if line_text:
                        raise RuntimeError(f"Voice synthesis input is missing for dialogue line {line_index}.")
                    if bool(render_cfg.get("force_voice_cloning", False)) and not bool(render_cfg.get("allow_system_tts_fallback", False)):
                        raise RuntimeError(f"Voice cloning is required, but dialogue line {line_index} has no text to synthesize.")
                    create_silence_audio(ffmpeg, max(line_duration, 0.25), normalized_path, sample_rate)
                    source_backend = "missing_dialogue_text_silence"
            if not source_backend:
                normalize_line_audio(ffmpeg, synthesized_source, line_duration, normalized_path, sample_rate)
                source_backend = "pyttsx3"
                synthesized_lines += 1
        normalized_map[line_index] = normalized_path
        package_line_path = normalized_path
        if package_root is not None:
            package_line_path = voice_line_output_audio_path(package_root, line)
            package_line_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(normalized_path, package_line_path)
        packaged_line_map[line_index] = package_line_path
        line_materializations.append(
            {
                "line_index": line_index,
                "speaker_name": clean_text(line.get("speaker_name", "")) or "Narrator",
                "audio_backend": source_backend,
                "audio_path": str(package_line_path),
            }
        )
    concat_segments: list[Path] = []
    silence_index = 0
    for segment in segment_plan:
        if segment.get("kind") == "silence":
            duration_seconds = float(segment.get("duration_seconds", 0.0) or 0.0)
            if duration_seconds <= 0.01:
                continue
            silence_path = audio_root / f"silence_{silence_index:04d}.wav"
            create_silence_audio(ffmpeg, duration_seconds, silence_path, sample_rate)
            concat_segments.append(silence_path)
            silence_index += 1
            continue
        line_index = int(segment.get("line_index", 0) or 0)
        concat_segments.append(normalized_map[line_index])
    if not concat_segments:
        silence_path = audio_root / "silence_full.wav"
        create_silence_audio(ffmpeg, total_duration, silence_path, sample_rate)
        concat_segments.append(silence_path)
    concat_audio_segments(ffmpeg, concat_segments, output_path)
    scene_dialogue_outputs: dict[str, str] = {}
    if package_root is not None and voice_plan_scenes:
        scene_dialogue_outputs = materialize_scene_dialogue_tracks(
            ffmpeg,
            voice_plan_scenes,
            packaged_line_map,
            sample_rate,
            package_root,
        )
    audio_backend = str(voice_meta.get("backend", "") or "")
    if reused_original_lines and synthesized_lines:
        audio_backend = "mixed_original_segment_and_pyttsx3"
    elif reused_original_lines and not synthesized_lines:
        audio_backend = "original_segment_reuse"
    elif not audio_backend:
        audio_backend = "pyttsx3"
    return {
        "audio_track": str(output_path),
        "audio_backend": audio_backend,
        "voice_id": voice_meta.get("voice_id", ""),
        "sample_rate": sample_rate,
        "segment_count": len(concat_segments),
        "reused_original_lines": reused_original_lines,
        "synthesized_lines": synthesized_lines,
        "scene_dialogue_outputs": scene_dialogue_outputs,
        "line_materializations": line_materializations,
    }


def run_ffmpeg_with_codec_fallback(command_factory, video_codec: str, output_path: Path) -> str:
    attempted: list[str] = []
    for codec in [video_codec, "libx264", "mpeg4"]:
        if codec in attempted:
            continue
        attempted.append(codec)
        command = command_factory(codec)
        result = run_media_command(command)
        if result.returncode == 0 and render_output_ready(output_path):
            return codec
    raise RuntimeError(f"FFmpeg render failed for {output_path.name}")


def create_final_render(ffmpeg_path: Path, segment_paths: list[Path], output_path: Path, video_codec: str) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        concat_path = Path(handle.name)
        for segment_path in segment_paths:
            escaped = str(segment_path).replace("'", "''")
            handle.write(f"file '{escaped}'\n")

    def command_factory(codec: str) -> list[str]:
        return [
            str(ffmpeg_path),
            "-hide_banner",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-an",
            "-c:v",
            codec,
            str(output_path),
        ]

    try:
        return run_ffmpeg_with_codec_fallback(command_factory, video_codec, output_path)
    finally:
        try:
            concat_path.unlink()
        except FileNotFoundError:
            pass


def choose_render_mode(scene_count: int, generated_scene_video_count: int, audio_available: bool) -> str:
    if generated_scene_video_count >= max(1, int(scene_count or 0)):
        return "fully_generated_scene_video_episode" if audio_available else "silent_generated_scene_video_fallback"
    if generated_scene_video_count > 0:
        return "hybrid_generated_scene_video_episode" if audio_available else "silent_hybrid_generated_scene_video_fallback"
    return "voiced_storyboard_episode" if audio_available else "silent_storyboard_preview_fallback"


def encode_motion_still_clip(
    ffmpeg: Path,
    image_path: Path,
    duration_seconds: float,
    output_path: Path,
    fps: int,
    width: int,
    height: int,
    crf: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, int(round(max(0.01, float(duration_seconds or 0.0)) * max(1, int(fps or 1)))))
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-frames:v",
        str(frame_count),
        "-vf",
        (
            "zoompan="
            f"z='if(eq(on,0),1.0,min(zoom+0.0012,1.12))':"
            "x='iw/2-(iw/zoom/2)':"
            "y='ih/2-(ih/zoom/2)':"
            f"d={frame_count}:s={width}x{height}:fps={fps},"
            "format=yuv420p"
        ),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = run_media_command(command)
    if result.returncode != 0 or not render_output_ready(output_path):
        raise RuntimeError(f"Could not encode motion clip {output_path.name}: {(result.stdout or '').strip()[-600:]}")


def encode_still_clip(ffmpeg: Path, image_path: Path, duration_seconds: float, output_path: Path, fps: int, width: int, height: int, crf: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-t",
        f"{max(0.01, float(duration_seconds or 0.0)):.3f}",
        "-vf",
        f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = run_media_command(command)
    if result.returncode != 0 or not render_output_ready(output_path):
        raise RuntimeError(f"Could not encode still clip {output_path.name}: {(result.stdout or '').strip()[-600:]}")


def normalize_scene_video_clip(
    ffmpeg: Path,
    input_path: Path,
    duration_seconds: float,
    output_path: Path,
    fps: int,
    width: int,
    height: int,
    crf: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_duration = max(0.01, float(duration_seconds or 0.0))
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-t",
        f"{target_duration:.3f}",
        "-vf",
        f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,tpad=stop_mode=clone:stop_duration={target_duration:.3f},format=yuv420p",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = run_media_command(command)
    if result.returncode != 0 or not render_output_ready(output_path):
        raise RuntimeError(f"Could not normalize generated scene clip {input_path.name}: {(result.stdout or '').strip()[-600:]}")


def materialize_scene_motion_video(
    ffmpeg: Path,
    package_root: Path,
    scene_id: str,
    fallback_frame_path: Path,
    asset_source_path: Path | None,
    scene: dict,
    visual_beats: list[dict],
    fps: int,
    width: int,
    height: int,
    crf: int,
) -> tuple[str, Path] | None:
    frame_sources = collect_scene_motion_frames(package_root, scene_id, fallback_frame_path)
    if asset_source_path is not None:
        for source_type, candidate in scene_asset_sidecar_frames(asset_source_path):
            if renderable_image_path(candidate):
                frame_sources.append((source_type, candidate))
        if renderable_image_path(asset_source_path):
            frame_sources.insert(0, ("asset_source_frame", asset_source_path))
    if not frame_sources:
        return None
    deduped_frame_sources: list[tuple[str, Path]] = []
    seen_paths: set[str] = set()
    for source_type, candidate in frame_sources:
        resolved = str(candidate.resolve())
        if resolved in seen_paths or not renderable_image_path(candidate):
            continue
        seen_paths.add(resolved)
        deduped_frame_sources.append((source_type, candidate))
    frame_sources = deduped_frame_sources
    if not frame_sources:
        return None
    output_path = scene_video_output_path(package_root, scene_id)
    preview_path = scene_video_preview_output_path(package_root, scene_id)
    poster_path = scene_video_poster_output_path(package_root, scene_id)
    beat_reference_root = output_path.parent / "beat_references"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    beat_reference_root.mkdir(parents=True, exist_ok=True)
    write_scene_video_reference_images(frame_sources[0][1], preview_path, poster_path, width, height)
    if not visual_beats:
        generation_plan = scene.get("generation_plan", {}) if isinstance(scene.get("generation_plan", {}), dict) else {}
        visual_beats = derive_scene_visual_beats(scene, [], generation_plan, safe_duration_seconds(scene))
    with tempfile.TemporaryDirectory(prefix=f"scene_motion_{production_scene_slug(scene_id)}_", dir=str(output_path.parent)) as temp_dir:
        temp_root = Path(temp_dir)
        clip_paths: list[Path] = []
        accent = theme_color(scene, max(0, len(visual_beats)))
        for index, beat in enumerate(visual_beats, start=1):
            beat_name = clean_text(beat.get("beat_name", "")) or ("speaker_left" if index % 2 else "speaker_right")
            source_path = frame_sources[(index - 1) % len(frame_sources)][1]
            source_image = fit_image(source_path, width, height)
            variant_image = image_variant_for_beat(source_image, beat_name, accent, index - 1)
            variant_path = temp_root / f"variant_{index:04d}.png"
            variant_image.save(variant_path, quality=95)
            reference_path = beat_reference_root / f"beat_{index:04d}_{beat_name}.png"
            variant_image.save(reference_path, quality=95)
            beat["reference_image_path"] = str(reference_path)
            beat["frame_source_type"] = frame_sources[(index - 1) % len(frame_sources)][0]
            beat["frame_source_path"] = str(source_path)
            clip_duration = max(0.01, float(beat.get("duration_seconds", 0.01) or 0.01))
            clip_path = temp_root / f"{index:04d}.mp4"
            try:
                encode_motion_still_clip(ffmpeg, variant_path, clip_duration, clip_path, fps, width, height, crf)
            except RuntimeError:
                encode_still_clip(ffmpeg, variant_path, clip_duration, clip_path, fps, width, height, crf)
            clip_paths.append(clip_path)
        concat_path = temp_root / "concat.txt"
        build_clip_concat_file(clip_paths, concat_path)
        encode_clip_sequence(ffmpeg, concat_path, output_path, crf)
    return "auto_generated_multishot_video", output_path


def encode_video(ffmpeg: Path, concat_path: Path, output_path: Path, fps: int, width: int, height: int, crf: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-vf",
        f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = run_media_command(command)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg render failed for {output_path.name}: {(result.stdout or '').strip()[-600:]}")


def encode_clip_sequence(ffmpeg: Path, concat_path: Path, output_path: Path, crf: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = run_media_command(command)
    if result.returncode != 0 or not render_output_ready(output_path):
        raise RuntimeError(f"FFmpeg clip concat failed for {output_path.name}: {(result.stdout or '').strip()[-600:]}")


def encode_full_generated_episode_master(
    ffmpeg: Path,
    concat_path: Path,
    output_path: Path,
    dialogue_audio_path: Path,
    crf: int,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if render_output_ready(dialogue_audio_path):
        video_only_path = output_path.with_name(f"{output_path.stem}_video_only{output_path.suffix}")
        encode_clip_sequence(ffmpeg, concat_path, video_only_path, crf)
        mux_episode_audio(ffmpeg, video_only_path, dialogue_audio_path, output_path)
        return {
            "audio_muxed": True,
            "video_only_path": str(video_only_path),
            "audio_source": str(dialogue_audio_path),
        }
    encode_clip_sequence(ffmpeg, concat_path, output_path, crf)
    return {
        "audio_muxed": False,
        "video_only_path": "",
        "audio_source": "",
    }


def build_concat_file(entries: list[tuple[Path, float]], output_path: Path) -> None:
    lines: list[str] = []
    for frame_path, duration in entries:
        escaped = str(frame_path).replace("'", "''")
        lines.append(f"file '{escaped}'")
        lines.append(f"duration {duration:.3f}")
    if entries:
        escaped = str(entries[-1][0]).replace("'", "''")
        lines.append(f"file '{escaped}'")
    write_text(output_path, "\n".join(lines) + "\n")


def build_clip_concat_file(clip_paths: list[Path], output_path: Path) -> None:
    lines = []
    for path in clip_paths:
        escaped = str(path).replace("'", "''")
        lines.append(f"file '{escaped}'")
    write_text(output_path, "\n".join(lines) + ("\n" if lines else ""))


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Render Episode")
    cfg = load_config()
    prepare_quality_backend_assets_runtime()
    ensure_quality_first_ready(cfg, context_label="17_render_episode.py")
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)

    write_realtime_preview(
        PROJECT_ROOT,
        args.episode_id or "pending",
        {"status": "started", "progress": 0, "message": "Render started"},
    )

    shotlist_dir = resolve_project_path("generation/shotlists")
    shotlist_path = (shotlist_dir / f"{args.episode_id}.json") if args.episode_id else find_latest_shotlist(shotlist_dir)
    if shotlist_path is None or not shotlist_path.exists():
        info("No shotlist found for render.")
        return

    shotlist = read_json(shotlist_path, {})
    episode_id = str(shotlist.get("episode_id", shotlist_path.stem))
    scenes = shotlist.get("scenes", []) if isinstance(shotlist.get("scenes", []), list) else []
    if not scenes:
        info("No scenes found in the shotlist. Run 14_generate_episode.py first.")
        return

    render_cfg = cfg.get("render", {}) if isinstance(cfg.get("render"), dict) else {}
    cloning_cfg = cfg.get("cloning", {}) if isinstance(cfg.get("cloning"), dict) else {}
    strict_outputs_required = strict_original_episode_outputs_required(cfg)
    render_cfg.setdefault(
        "prefer_detected_character_language",
        bool(cloning_cfg.get("prefer_detected_character_language", True)),
    )
    render_cfg.setdefault("allow_system_tts_fallback", bool(cloning_cfg.get("allow_system_tts_fallback", False)))
    render_cfg.setdefault("force_voice_cloning", bool(cloning_cfg.get("force_voice_cloning", True)))
    render_cfg.setdefault("voice_clone_engine", clean_text(cloning_cfg.get("voice_clone_engine", "")))
    width = int(render_cfg.get("width", 1280))
    height = int(render_cfg.get("height", 720))
    fps = max(12, int(render_cfg.get("fps", 30)))
    title_card_seconds = max(0.0, float(render_cfg.get("title_card_seconds", 2.5) or 0.0))
    closing_card_seconds = max(0.0, float(render_cfg.get("closing_card_seconds", 2.0) or 0.0))
    include_title_cards = bool(render_cfg.get("include_title_cards", False))

    ffmpeg = detect_tool(PROJECT_ROOT / "tools" / "ffmpeg" / "bin", "ffmpeg")
    assets_root = Path(str(shotlist.get("storyboard_assets_root", ""))).resolve() if shotlist.get("storyboard_assets_root") else storyboard_assets_root(cfg, episode_id)
    render_root = resolve_project_path("generation/renders")
    temp_frame_root = render_root / "tmp" / episode_id
    temp_clip_root = temp_frame_root / "clips"
    draft_path = render_root / "drafts" / f"{episode_id}.mp4"
    final_path = render_root / "final" / f"{episode_id}.mp4"
    final_video_only_path = temp_frame_root / f"{episode_id}_video_only.mp4"
    manifest_path = render_root / "final" / f"{episode_id}_render_manifest.json"
    voice_plan_path = render_root / "final" / f"{episode_id}_voice_plan.json"
    subtitle_preview_path = render_root / "final" / f"{episode_id}_dialogue_preview.srt"
    dialogue_audio_path = render_root / "final" / f"{episode_id}_dialogue_audio.wav"
    concat_path = temp_frame_root / "frames.txt"
    clip_concat_path = temp_frame_root / "clips.txt"
    production_package_path = episode_production_package_root(cfg, episode_id) / "master" / f"{episode_id}_production_package.json"
    full_generated_episode_path = episode_production_package_root(cfg, episode_id) / "master" / f"{episode_id}_full_generated_episode.mp4"
    if args.force and strict_outputs_required:
        for stale_output in (final_path, dialogue_audio_path, full_generated_episode_path):
            remove_stale_generated_output(stale_output)

    autosave_target = episode_id
    mark_step_started("17_render_episode", autosave_target, {"episode_id": episode_id, "shotlist": str(shotlist_path)})
    if shared_workers:
        info(f"Shared NAS workers: enabled ({worker_id})")
    reporter = LiveProgressReporter(
        script_name="17_render_episode.py",
        total=len(scenes) + 3,
        phase_label="Render Episode",
        parent_label=episode_id,
    )
    lease_root = distributed_step_runtime_root("17_render_episode", "episodes")
    lease_heartbeat = None
    try:
        if shared_workers:
            lease_meta = {"step": "17_render_episode", "episode_id": episode_id, "worker_id": worker_id}
            acquired = acquire_distributed_lease(
                lease_root,
                episode_id,
                worker_id,
                distributed_lease_ttl_seconds(cfg),
                meta=lease_meta,
            )
            if acquired is None:
                info("This episode is already being rendered by another worker.")
                return
            lease_heartbeat = DistributedLeaseHeartbeat(
                root=lease_root,
                lease_name=episode_id,
                owner_id=worker_id,
                ttl_seconds=distributed_lease_ttl_seconds(cfg),
                interval_seconds=distributed_heartbeat_interval_seconds(cfg),
                meta_factory=lambda: lease_meta,
            )
            lease_heartbeat.start()
        existing_render_ready = render_output_ready(full_generated_episode_path) if strict_outputs_required else render_output_ready(final_path)
        if render_output_ready(draft_path) and existing_render_ready and production_package_path.exists() and not args.force:
            reporter.finish(current_label=episode_id, extra_label="Draft, final render, and production package already exist")
            mark_step_completed(
                "17_render_episode",
                autosave_target,
                {
                    "episode_id": episode_id,
                    "draft_render": str(draft_path),
                    "final_render": str(final_path) if render_output_ready(final_path) else "",
                    "manifest": str(manifest_path),
                    "production_package": str(production_package_path),
                },
            )
            ok(f"Render already available: {episode_id}")
            return

        temp_frame_root.mkdir(parents=True, exist_ok=True)
        temp_clip_root.mkdir(parents=True, exist_ok=True)
        entries: list[tuple[Path, float]] = []
        final_clip_paths: list[Path] = []
        opening_clip_paths: list[Path] = []
        scene_clip_paths: list[Path] = []
        closing_clip_paths: list[Path] = []
        manifest_scenes: list[dict] = []
        voice_plan_scenes: list[dict] = []
        voice_plan_lines: list[dict] = []
        original_line_library = build_original_line_library(cfg)
        voice_lookup = build_voice_lookup(cfg)
        timeline_cursor = 0.0
        generated_scene_video_count = 0
        render_mode = "voiced_storyboard_episode"
        audio_track_meta: dict[str, object] = {}
        audio_render_error = ""
        full_generated_episode_master_meta: dict[str, object] = {}
        package_root = episode_production_package_root(cfg, episode_id)

        if include_title_cards and title_card_seconds > 0.0:
            reporter.update(0, current_label="Title Card", extra_label="Running now: render opening title card", force=True)
            title_path = temp_frame_root / "0000_title.png"
            title_image = title_card(
                width,
                height,
                episode_id,
                str(shotlist.get("display_title", episode_id)),
                str(shotlist.get("episode_title", "")),
            )
            title_image.save(title_path, quality=95)
            entries.append((title_path, title_card_seconds))
            title_clip_path = temp_clip_root / "0000_title.mp4"
            encode_still_clip(ffmpeg, title_path, title_card_seconds, title_clip_path, fps, width, height, crf=20)
            opening_clip_paths.append(title_clip_path)
            final_clip_paths.append(title_clip_path)
            timeline_cursor += title_card_seconds

        for index, scene in enumerate(scenes, start=1):
            scene_id = str(scene.get("scene_id", "")).strip() or f"scene_{index:04d}"
            reporter.update(index - 1, current_label=scene_id, extra_label="Running now: compose storyboard render frame", force=True)
            scene_base_image, scene_meta = build_scene_frame(scene, index - 1, cfg, episode_id, assets_root, width, height)
            scene_image = compose_scene_card_from_base(scene, index - 1, scene_base_image, scene_meta, width, height)
            frame_path = temp_frame_root / f"{index:04d}_{scene_id}.png"
            clean_frame_path = temp_frame_root / f"{index:04d}_{scene_id}_clean.png"
            scene_image.save(frame_path, quality=95)
            scene_base_image.save(clean_frame_path, quality=95)
            duration = safe_duration_seconds(scene)
            entries.append((frame_path, duration))
            scene_voice_plan = build_scene_voice_plan(
                scene,
                duration,
                timeline_cursor,
                original_line_library,
                voice_lookup,
                cloning_cfg,
                render_cfg,
                cfg,
            )
            scene_generation_plan = scene.get("generation_plan", {}) if isinstance(scene.get("generation_plan", {}), dict) else {}
            scene_visual_beats = derive_scene_visual_beats(
                scene,
                scene_voice_plan,
                scene_generation_plan,
                duration,
                timeline_cursor,
            )
            final_clip_path = temp_clip_root / f"{index:04d}_{scene_id}.mp4"
            scene_video_source = first_existing_scene_video_source(package_root, assets_root, scene_id)
            if scene_video_source is None and local_motion_fallback_allowed(cfg, render_cfg):
                scene_video_source = materialize_scene_motion_video(
                    ffmpeg,
                    package_root,
                    scene_id,
                    clean_frame_path,
                    resolve_stored_project_path(scene_meta.get("asset_source_path", "")),
                    scene,
                    scene_visual_beats,
                    fps,
                    width,
                    height,
                    crf=20,
                )
            if scene_video_source is not None:
                generated_scene_video_count += 1
                scene_video_source_type, scene_video_source_path = scene_video_source
                normalize_scene_video_clip(ffmpeg, scene_video_source_path, duration, final_clip_path, fps, width, height, crf=20)
            else:
                scene_video_source_type = ""
                scene_video_source_path = Path()
                encode_still_clip(ffmpeg, frame_path, duration, final_clip_path, fps, width, height, crf=20)
            scene_clip_paths.append(final_clip_path)
            final_clip_paths.append(final_clip_path)
            voice_plan_scenes.append(
                {
                    "scene_id": scene_id,
                    "scene_start_seconds": round(timeline_cursor, 3),
                    "scene_end_seconds": round(timeline_cursor + duration, 3),
                    "duration_seconds": duration,
                    "line_count": len(scene_voice_plan),
                    "lines": scene_voice_plan,
                }
            )
            voice_plan_lines.extend(scene_voice_plan)
            manifest_scenes.append(
                {
                    "scene_id": scene_id,
                    "title": scene.get("title", ""),
                    "duration_seconds": duration,
                    "frame_path": str(frame_path),
                    "clean_frame_path": str(clean_frame_path),
                    "asset_source_type": scene_meta.get("asset_source_type", ""),
                    "asset_source_path": scene_meta.get("asset_source_path", ""),
                    "video_source_type": scene_video_source_type,
                    "video_source_path": str(scene_video_source_path) if scene_video_source is not None else "",
                    "final_clip_path": str(final_clip_path),
                    "characters": scene.get("characters", []) if isinstance(scene.get("characters", []), list) else [],
                    "summary": scene.get("summary", ""),
                    "generation_plan": scene_generation_plan,
                    "scene_visual_beats": scene_visual_beats,
                    "voice_line_count": len(scene_voice_plan),
                }
            )
            reporter.update(
                index,
                current_label=scene_id,
                extra_label=(
                    f"Frame source: {scene_meta.get('asset_source_type', 'placeholder')} | "
                    f"Final clip: {scene_video_source_type or 'storyboard_card'}"
                ),
            )
            timeline_cursor += duration

        if closing_card_seconds > 0.0:
            closing_index = len(entries) + 1
            reporter.update(len(scenes), current_label="Closing Card", extra_label="Running now: render closing card", force=True)
            closing_path = temp_frame_root / f"{closing_index:04d}_closing.png"
            closing_image = closing_card(width, height, str(shotlist.get("display_title", episode_id)), len(scenes))
            closing_image.save(closing_path, quality=95)
            entries.append((closing_path, closing_card_seconds))
            closing_clip_path = temp_clip_root / f"{closing_index:04d}_closing.mp4"
            encode_still_clip(ffmpeg, closing_path, closing_card_seconds, closing_clip_path, fps, width, height, crf=20)
            closing_clip_paths.append(closing_clip_path)
            final_clip_paths.append(closing_clip_path)
            timeline_cursor += closing_card_seconds

        reporter.update(len(scenes) + 1, current_label="Concat List", extra_label="Running now: assemble FFmpeg concat list")
        build_concat_file(entries, concat_path)
        build_clip_concat_file(final_clip_paths, clip_concat_path)

        reporter.update(len(scenes) + 2, current_label="Draft Render", extra_label="Running now: encode draft and final video", force=True)
        encode_video(ffmpeg, concat_path, draft_path, fps, min(width, 960), min(height, 540), crf=28)

        write_realtime_preview(
            PROJECT_ROOT,
            episode_id,
            {
                "status": "rendering",
                "progress": 50,
                "scenes_rendered": len(scenes),
                "draft_ready": draft_path.exists(),
            },
        )

        encode_clip_sequence(ffmpeg, clip_concat_path, final_video_only_path, crf=20)

        if strict_outputs_required:
            audio_render_error = "Voice cloning is delegated to finished_episode_voice_runner; local TTS/audio fallback is disabled."
        else:
            try:
                audio_track_meta = render_episode_audio_track(
                    ffmpeg,
                    voice_plan_lines,
                    timeline_cursor,
                    render_cfg,
                    temp_frame_root,
                    dialogue_audio_path,
                    episode_production_package_root(cfg, episode_id),
                    voice_plan_scenes,
                )
                mux_episode_audio(ffmpeg, final_video_only_path, dialogue_audio_path, final_path)
            except Exception as audio_exc:
                audio_render_error = str(audio_exc)
                shutil.copyfile(final_video_only_path, final_path)
                info(f"Audio render fallback active for {episode_id}: {audio_render_error}")
        scene_dialogue_outputs = normalized_scene_dialogue_outputs(audio_track_meta)
        audio_backend_name = clean_text(audio_track_meta.get("audio_backend", "")) or clean_text(audio_track_meta.get("backend", ""))
        render_mode = choose_render_mode(len(scenes), generated_scene_video_count, bool(audio_track_meta))
        scene_master_outputs = materialize_scene_master_clips(
            ffmpeg,
            manifest_scenes,
            scene_dialogue_outputs,
            package_root,
        )
        for scene_meta in manifest_scenes:
            if not isinstance(scene_meta, dict):
                continue
            scene_id = clean_text(scene_meta.get("scene_id", ""))
            scene_meta["scene_dialogue_audio"] = ""
            scene_meta["scene_master_clip"] = ""
            scene_meta["audio_backend"] = audio_backend_name
            if scene_id:
                scene_meta["scene_dialogue_audio"] = clean_text(scene_dialogue_outputs.get(scene_id, ""))
            scene_meta["scene_master_clip"] = clean_text(scene_master_outputs.get(scene_id, ""))
        package_master_clip_paths = [*opening_clip_paths]
        for scene_meta, scene_clip_path in zip(manifest_scenes, scene_clip_paths):
            if not isinstance(scene_meta, dict):
                continue
            scene_id = clean_text(scene_meta.get("scene_id", ""))
            scene_master_path = resolve_stored_project_path(scene_master_outputs.get(scene_id, ""))
            package_master_clip_paths.append(scene_master_path if render_output_ready(scene_master_path) else scene_clip_path)
        package_master_clip_paths.extend(closing_clip_paths)
        build_clip_concat_file(package_master_clip_paths, clip_concat_path)
        full_generated_episode_master_meta = encode_full_generated_episode_master(
            ffmpeg,
            clip_concat_path,
            full_generated_episode_path,
            dialogue_audio_path if audio_track_meta else Path(),
            crf=20,
        )

        voice_plan_payload = {
            "episode_id": episode_id,
            "render_mode": render_mode,
            "voice_plan_mode": "timed_dialogue_preview",
            "subtitle_preview": str(subtitle_preview_path),
            "audio_track": str(dialogue_audio_path) if audio_track_meta else "",
            "audio_track_available": bool(audio_track_meta),
            "audio_render_error": audio_render_error,
            "scene_count": len(voice_plan_scenes),
            "line_count": len(voice_plan_lines),
            "generated_scene_video_count": generated_scene_video_count,
            "scene_master_clip_count": len(scene_master_outputs),
            "full_generated_episode_audio_muxed": bool(full_generated_episode_master_meta.get("audio_muxed", False)),
            "scenes": voice_plan_scenes,
        }
        write_json(voice_plan_path, voice_plan_payload)
        write_text(subtitle_preview_path, render_subtitle_preview_srt(voice_plan_lines))

        manifest = {
            "episode_id": episode_id,
            "display_title": shotlist.get("display_title", episode_id),
            "episode_title": shotlist.get("episode_title", ""),
            "shotlist_path": str(shotlist_path),
            "storyboard_assets_root": str(assets_root),
            "render_mode": render_mode,
            "voice_plan_mode": "timed_dialogue_preview",
            "draft_render": str(draft_path),
            "final_render": str(final_path) if render_output_ready(final_path) else "",
            "final_video_only_render": str(final_video_only_path),
            "voice_plan": str(voice_plan_path),
            "subtitle_preview": str(subtitle_preview_path),
            "dialogue_audio": str(dialogue_audio_path) if audio_track_meta else "",
            "audio_track_available": bool(audio_track_meta),
            "audio_render_error": audio_render_error,
            "audio_track_meta": audio_track_meta,
            "ffmpeg": str(ffmpeg),
            "fps": fps,
            "width": width,
            "height": height,
            "include_title_cards": include_title_cards,
            "generated_scene_video_count": generated_scene_video_count,
            "scene_master_clip_count": len(scene_master_outputs),
            "full_generated_episode": str(full_generated_episode_path),
            "full_generated_episode_master_meta": full_generated_episode_master_meta,
            "scene_count": len(scenes),
            "scenes": manifest_scenes,
        }
        manifest["render_manifest_path"] = str(manifest_path)
        write_json(manifest_path, manifest)
        production_package = write_episode_production_package(cfg, episode_id, shotlist, manifest, voice_plan_payload)
        production_package_path_text = clean_text(production_package.get("package_path", ""))
        prompt_preview_path_text = clean_text(production_package.get("prompt_preview_path", ""))
        production_package_path_resolved = (
            resolve_stored_project_path(production_package_path_text) if production_package_path_text else Path(production_package["package_path"])
        )
        prompt_preview_path_resolved = (
            resolve_stored_project_path(prompt_preview_path_text) if prompt_preview_path_text else Path(production_package["prompt_preview_path"])
        )
        scene_runner_summary = run_finished_episode_scene_runners(
            cfg,
            production_package_path_resolved,
            prompt_preview_path_resolved,
            force=args.force,
        )
        production_package_payload = refresh_episode_production_package(
            production_package_path_resolved,
            full_generated_episode_path=full_generated_episode_path if render_output_ready(full_generated_episode_path) else None,
            backend_runner_summary={"scene_runners": scene_runner_summary},
        )
        package_scene_dialogue_outputs = collect_scene_dialogue_outputs_from_package(production_package_payload)
        generated_scene_dialogue_meta: dict[str, object] = {}
        preferred_dialogue_audio_path = dialogue_audio_path if audio_track_meta else Path()
        if package_scene_dialogue_outputs:
            generated_scene_dialogue_path = package_root / "master" / f"{episode_id}_generated_scene_dialogue.wav"
            generated_scene_dialogue_meta = materialize_episode_audio_track_from_scene_outputs(
                ffmpeg,
                voice_plan_scenes,
                timeline_cursor,
                package_scene_dialogue_outputs,
                int(render_cfg.get("audio_sample_rate", 22050) or 22050),
                generated_scene_dialogue_path,
            )
            generated_scene_dialogue_audio_path = resolve_stored_project_path(generated_scene_dialogue_meta.get("audio_path", ""))
            if render_output_ready(generated_scene_dialogue_audio_path):
                preferred_dialogue_audio_path = generated_scene_dialogue_audio_path
                generated_scene_dialogue_meta["audio_backend"] = "xtts_voice_clone"
        scene_master_audio_outputs = (
            package_scene_dialogue_outputs
            if package_scene_dialogue_outputs
            else (
                audio_track_meta.get("scene_dialogue_outputs", {})
                if isinstance(audio_track_meta.get("scene_dialogue_outputs", {}), dict)
                else {}
            )
        )
        scene_master_outputs = materialize_scene_master_clips(
            ffmpeg,
            manifest_scenes,
            scene_master_audio_outputs,
            package_root,
        )
        for scene_meta in manifest_scenes:
            if not isinstance(scene_meta, dict):
                continue
            scene_id = clean_text(scene_meta.get("scene_id", ""))
            if scene_id and isinstance(scene_master_audio_outputs, dict):
                scene_meta["scene_dialogue_audio"] = clean_text(scene_master_audio_outputs.get(scene_id, ""))
            scene_meta["scene_master_clip"] = clean_text(scene_master_outputs.get(scene_id, ""))
            scene_meta["audio_backend"] = clean_text(generated_scene_dialogue_meta.get("audio_backend", "")) or clean_text(audio_track_meta.get("audio_backend", "")) or clean_text(audio_track_meta.get("backend", ""))
        package_master_clip_paths = [*opening_clip_paths]
        for scene_meta, scene_clip_path in zip(manifest_scenes, scene_clip_paths):
            if not isinstance(scene_meta, dict):
                continue
            scene_id = clean_text(scene_meta.get("scene_id", ""))
            scene_master_path = resolve_stored_project_path(scene_master_outputs.get(scene_id, ""))
            package_master_clip_paths.append(scene_master_path if render_output_ready(scene_master_path) else scene_clip_path)
        package_master_clip_paths.extend(closing_clip_paths)
        build_clip_concat_file(package_master_clip_paths, clip_concat_path)
        master_runner_result = run_finished_episode_master_runner(
            cfg,
            production_package_payload,
            force=args.force,
        )
        master_runner_ready = (
            str(master_runner_result.get("status", "")).strip() in {"completed", "existing_outputs"}
            and render_output_ready(full_generated_episode_path)
        )
        if master_runner_ready:
            full_generated_episode_master_meta = {
                "audio_muxed": bool(render_output_ready(preferred_dialogue_audio_path)),
                "video_only_path": "",
                "audio_source": str(preferred_dialogue_audio_path) if render_output_ready(preferred_dialogue_audio_path) else "",
                "external_master_runner": True,
            }
        else:
            full_generated_episode_master_meta = encode_full_generated_episode_master(
                ffmpeg,
                clip_concat_path,
                full_generated_episode_path,
                preferred_dialogue_audio_path if render_output_ready(preferred_dialogue_audio_path) else Path(),
                crf=20,
            )
        production_package_payload = refresh_episode_production_package(
            production_package_path_resolved,
            full_generated_episode_path=full_generated_episode_path,
            backend_runner_summary={
                "scene_runners": scene_runner_summary,
                "master_runner": master_runner_result,
            },
        )
        production_package = production_package_payload
        ensure_strict_generation_outputs(cfg, production_package_payload, context_label=f"{episode_id} render")
        manifest["dialogue_audio"] = str(preferred_dialogue_audio_path) if render_output_ready(preferred_dialogue_audio_path) else ""
        if render_output_ready(preferred_dialogue_audio_path) and clean_text(generated_scene_dialogue_meta.get("audio_backend", "")):
            manifest["audio_track_meta"] = {
                **(audio_track_meta if isinstance(audio_track_meta, dict) else {}),
                "audio_backend": clean_text(generated_scene_dialogue_meta.get("audio_backend", "")),
                "scene_dialogue_outputs": scene_master_audio_outputs,
            }
        manifest["full_generated_episode"] = str(full_generated_episode_path)
        manifest["full_generated_episode_master_meta"] = full_generated_episode_master_meta
        manifest["scene_master_clip_count"] = len(scene_master_outputs)
        manifest["generated_scene_dialogue_audio"] = str(preferred_dialogue_audio_path) if render_output_ready(preferred_dialogue_audio_path) else ""
        manifest["generated_scene_dialogue_meta"] = generated_scene_dialogue_meta
        manifest["production_backend_runner_manifest"] = clean_text(
            (
                production_package_payload.get("backend_runner_summary", {})
                if isinstance(production_package_payload.get("backend_runner_summary", {}), dict)
                else {}
            ).get("scene_runners", {}).get("summary_path", "")
        )
        delivery_bundle = write_episode_delivery_bundle(
            cfg,
            episode_id,
            shotlist,
            manifest,
            production_package,
            final_path,
            full_generated_episode_path,
            preferred_dialogue_audio_path,
            subtitle_preview_path,
            voice_plan_path,
        )
        manifest["production_package_root"] = production_package["package_root"]
        manifest["production_package"] = production_package["package_path"]
        manifest["production_prompt_preview"] = production_package["prompt_preview_path"]
        manifest["scene_production_packages"] = production_package["scene_package_paths"]
        manifest["delivery_bundle_root"] = delivery_bundle["delivery_root"]
        manifest["delivery_manifest"] = delivery_bundle["delivery_manifest"]
        manifest["delivery_episode"] = delivery_bundle["watch_episode"]
        manifest["delivery_summary"] = delivery_bundle.get("delivery_summary", "")
        manifest["latest_delivery_root"] = delivery_bundle.get("latest_delivery_root", "")
        manifest["latest_delivery_manifest"] = delivery_bundle.get("latest_delivery_manifest", "")
        manifest["latest_delivery_episode"] = delivery_bundle.get("latest_watch_episode", "")
        write_json(manifest_path, manifest)
        shotlist["render_manifest"] = str(manifest_path)
        shotlist["draft_render"] = str(draft_path)
        shotlist["final_render"] = str(final_path) if render_output_ready(final_path) else ""
        shotlist["dialogue_audio"] = str(preferred_dialogue_audio_path) if render_output_ready(preferred_dialogue_audio_path) else ""
        shotlist["voice_plan"] = str(voice_plan_path)
        shotlist["subtitle_preview"] = str(subtitle_preview_path)
        shotlist["full_generated_episode"] = str(full_generated_episode_path)
        shotlist["production_package_root"] = production_package["package_root"]
        shotlist["production_package"] = production_package["package_path"]
        shotlist["production_prompt_preview"] = production_package["prompt_preview_path"]
        shotlist["production_backend_runner_manifest"] = manifest.get("production_backend_runner_manifest", "")
        shotlist["delivery_bundle_root"] = delivery_bundle["delivery_root"]
        shotlist["delivery_manifest"] = delivery_bundle["delivery_manifest"]
        shotlist["delivery_episode"] = delivery_bundle["watch_episode"]
        shotlist["delivery_summary"] = delivery_bundle.get("delivery_summary", "")
        shotlist["latest_delivery_root"] = delivery_bundle.get("latest_delivery_root", "")
        shotlist["latest_delivery_manifest"] = delivery_bundle.get("latest_delivery_manifest", "")
        shotlist["latest_delivery_episode"] = delivery_bundle.get("latest_watch_episode", "")
        write_json(shotlist_path, shotlist)

        write_realtime_preview(
            PROJECT_ROOT,
            episode_id,
            {
                "status": "completed",
                "progress": 100,
                "final_render": str(final_path) if render_output_ready(final_path) else "",
                "full_generated_episode": str(full_generated_episode_path),
                "delivery_bundle": delivery_bundle.get("delivery_root", ""),
            },
        )

        reporter.finish(current_label=episode_id, extra_label=f"Rendered {len(scenes)} scenes and exported the full-episode production package")
        mark_step_completed(
            "17_render_episode",
            autosave_target,
            {
                "episode_id": episode_id,
                "draft_render": str(draft_path),
                "final_render": str(final_path) if render_output_ready(final_path) else "",
                "dialogue_audio": str(preferred_dialogue_audio_path) if render_output_ready(preferred_dialogue_audio_path) else "",
                "render_mode": render_mode,
                "manifest": str(manifest_path),
                "production_package": production_package["package_path"],
                "delivery_manifest": delivery_bundle["delivery_manifest"],
                "delivery_episode": delivery_bundle["watch_episode"],
                "delivery_summary": delivery_bundle.get("delivery_summary", ""),
                "latest_delivery_manifest": delivery_bundle.get("latest_delivery_manifest", ""),
                "latest_delivery_episode": delivery_bundle.get("latest_watch_episode", ""),
            },
        )
        rendered_title = clean_text(shotlist.get("display_title", "")) or episode_id
        ok(f"Episode rendered: {rendered_title} ({episode_id})")
    except Exception as exc:
        mark_step_failed("17_render_episode", str(exc), autosave_target, {"episode_id": episode_id})
        raise
    finally:
        if lease_heartbeat is not None:
            lease_heartbeat.stop()
            release_distributed_lease(lease_root, episode_id, worker_id)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise

