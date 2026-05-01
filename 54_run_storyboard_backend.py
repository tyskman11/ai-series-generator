#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from support_scripts.pipeline_common import (
    add_shared_worker_arguments,
    detect_tool,
    distributed_item_lease,
    distributed_step_runtime_root,
    LiveProgressReporter,
    error,
    headline,
    info,
    latest_matching_file,
    load_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    read_json,
    rerun_in_runtime,
    run_external_backend_runner,
    resolve_project_path,
    shared_worker_id_for_args,
    shared_workers_enabled_for_args,
    warn,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize local storyboard backend frames from per-scene backend input payloads"
    )
    parser.add_argument("--episode-id", help="Target a specific episode ID such as episode_09 or folge_09.")
    parser.add_argument("--force", action="store_true", help="Recreate already materialized backend frames.")
    parser.add_argument("--scene-ids", nargs="*", help="Only process these scene IDs (scene-selective regeneration).")
    add_shared_worker_arguments(parser)
    return parser.parse_args()


def find_latest_shotlist(shotlist_dir: Path) -> Path | None:
    episode_id = os.environ.get("SERIES_STORYBOARD_EPISODE", "").strip()
    if episode_id:
        candidate = shotlist_dir / f"{episode_id}.json"
        if candidate.exists():
            return candidate
    return latest_matching_file(shotlist_dir, "*.json")


def candidate_image_paths(assets_root: Path, scene_id: str, payload: dict) -> list[Path]:
    source_hints = payload.get("source_hints", {}) if isinstance(payload.get("source_hints"), dict) else {}
    candidates = [
        assets_root / f"{scene_id}.png",
        assets_root / f"{scene_id}.jpg",
        Path(str(source_hints.get("reference_image", ""))),
        Path(str(source_hints.get("continuity_asset", ""))),
    ]
    generation_plan = payload.get("reference_slots", []) if isinstance(payload.get("reference_slots"), list) else []
    for slot in generation_plan:
        if not isinstance(slot, dict):
            continue
        for key in ("context_images", "portrait_images"):
            values = slot.get(key, [])
            if not isinstance(values, list):
                continue
            for value in values:
                candidates.append(Path(str(value)))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def preferred_previous_frame(assets_root: Path, payload: dict) -> Path | None:
    continuity = payload.get("continuity", {}) if isinstance(payload.get("continuity"), dict) else {}
    previous_scene_id = str(continuity.get("previous_scene_id", "")).strip()
    if not previous_scene_id:
        return None
    candidates = [
        assets_root / previous_scene_id / "frame.png",
        assets_root / previous_scene_id / "frame.jpg",
        assets_root / previous_scene_id / "preview.png",
        assets_root / previous_scene_id / "preview.jpg",
        assets_root / f"{previous_scene_id}.png",
        assets_root / f"{previous_scene_id}.jpg",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def first_existing_path(paths: list[Path]) -> Path | None:
    for candidate in paths:
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def accent_from_payload(payload: dict) -> tuple[int, int, int]:
    text_parts = [
        str(payload.get("positive_prompt", "")),
        str(payload.get("negative_prompt", "")),
        str(payload.get("batch_prompt_line", "")),
        str(payload.get("scene_id", "")),
        str(payload.get("title", "")),
    ]
    seed_text = " ".join(part for part in text_parts if part).lower()
    if not seed_text:
        return (104, 148, 196)
    value = sum(ord(char) for char in seed_text)
    return (
        72 + (value % 120),
        84 + ((value // 3) % 116),
        96 + ((value // 7) % 108),
    )


def fit_rgb_image(path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)


def scene_output_root(assets_root: Path, scene_id: str) -> Path:
    return assets_root / scene_id


def scene_frame_output_path(assets_root: Path, scene_id: str) -> Path:
    return scene_output_root(assets_root, scene_id) / "frame.png"


def scene_preview_output_path(assets_root: Path, scene_id: str) -> Path:
    return scene_output_root(assets_root, scene_id) / "preview.jpg"


def scene_poster_output_path(assets_root: Path, scene_id: str) -> Path:
    return scene_output_root(assets_root, scene_id) / "poster.png"


def scene_clip_output_path(assets_root: Path, scene_id: str) -> Path:
    return scene_output_root(assets_root, scene_id) / "clip.mp4"


def scene_alternates_root(assets_root: Path, scene_id: str) -> Path:
    return scene_output_root(assets_root, scene_id) / "alternates"


def backend_output_ready(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def apply_backend_grade(
    image: Image.Image,
    payload: dict,
    accent: tuple[int, int, int],
    previous_frame: Path | None,
    backend_strength: float,
) -> Image.Image:
    working = image.convert("RGB")
    if previous_frame is not None and previous_frame.exists():
        continuity_image = fit_rgb_image(previous_frame, working.width, working.height)
        working = Image.blend(working, continuity_image, min(0.28, 0.12 + (backend_strength * 0.12)))

    control_hints = payload.get("control_hints", {}) if isinstance(payload.get("control_hints"), dict) else {}
    camera_plan = payload.get("camera_plan", {}) if isinstance(payload.get("camera_plan"), dict) else {}
    contrast_boost = 1.06 + (backend_strength * 0.08)
    color_boost = 1.04 + (backend_strength * 0.06)
    brightness_boost = 1.01 + (backend_strength * 0.04)
    if str(control_hints.get("pose_emphasis", "")).strip():
        contrast_boost += 0.03
    if str(camera_plan.get("lens", "")).strip():
        color_boost += 0.02
    working = ImageEnhance.Contrast(working).enhance(contrast_boost)
    working = ImageEnhance.Color(working).enhance(color_boost)
    working = ImageEnhance.Brightness(working).enhance(brightness_boost)
    working = working.filter(ImageFilter.UnsharpMask(radius=1.5, percent=130, threshold=3))

    overlay = Image.new("RGBA", working.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    overlay_color = accent + (34 + int(backend_strength * 30),)
    draw.rectangle((0, 0, working.width, working.height), fill=overlay_color)
    vignette = Image.new("L", working.size, 0)
    vignette_draw = ImageDraw.Draw(vignette)
    vignette_draw.ellipse(
        (-working.width * 0.15, -working.height * 0.12, working.width * 1.15, working.height * 1.12),
        fill=215,
    )
    vignette = vignette.filter(ImageFilter.GaussianBlur(max(16, int(min(working.size) * 0.05))))
    overlay.putalpha(vignette)
    merged = Image.alpha_composite(working.convert("RGBA"), overlay)
    return merged.convert("RGB")


def backend_variant_labels(payload: dict) -> list[str]:
    labels: list[str] = []
    camera_plan = payload.get("camera_plan", {}) if isinstance(payload.get("camera_plan", {}), dict) else {}
    for raw_value in (camera_plan.get("lens", ""), camera_plan.get("movement", "")):
        cleaned = str(raw_value or "").strip().lower()
        if cleaned:
            labels.append(cleaned)
    labels.extend(["hero", "close", "wide", "reaction"])
    deduped: list[str] = []
    seen: set[str] = set()
    for label in labels:
        safe = "".join(char if char.isalnum() else "_" for char in label).strip("_") or "variant"
        if safe in seen:
            continue
        seen.add(safe)
        deduped.append(safe)
    return deduped[:4]


def variant_image(image: Image.Image, accent: tuple[int, int, int], index: int) -> Image.Image:
    working = image.convert("RGB")
    if index % 4 == 1:
        working = ImageEnhance.Contrast(working).enhance(1.08)
        working = ImageEnhance.Color(working).enhance(1.05)
    elif index % 4 == 2:
        working = ImageOps.mirror(working)
        working = ImageEnhance.Sharpness(working).enhance(1.18)
    elif index % 4 == 3:
        bordered = ImageOps.expand(working, border=max(12, working.width // 24), fill=accent)
        working = ImageOps.fit(bordered, working.size, method=Image.Resampling.LANCZOS)
        working = ImageEnhance.Brightness(working).enhance(1.03)
    else:
        overlay = Image.new("RGBA", working.size, accent + (24,))
        working = Image.alpha_composite(working.convert("RGBA"), overlay).convert("RGB")
    return working


def build_concat_file(entries: list[tuple[Path, float]], output_path: Path) -> None:
    lines: list[str] = []
    for frame_path, duration in entries:
        escaped = str(frame_path).replace("'", "''")
        lines.append(f"file '{escaped}'")
        lines.append(f"duration {duration:.3f}")
    if entries:
        escaped = str(entries[-1][0]).replace("'", "''")
        lines.append(f"file '{escaped}'")
    write_text(output_path, "\n".join(lines) + ("\n" if lines else ""))


def encode_scene_video(
    ffmpeg_path: Path,
    concat_path: Path,
    output_path: Path,
    fps: int,
    width: int,
    height: int,
    crf: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg_path),
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
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Could not encode local backend scene clip {output_path.name}: {(result.stdout or '').strip()[-600:]}")


def scene_backend_row(
    *,
    scene_id: str,
    frame_path: Path,
    preview_path: Path,
    poster_path: Path,
    clip_path: Path,
    alternates_root: Path,
    manifest_path: Path,
    status: str,
    source_type: str,
    backend_mode: str,
    clip_status: str = "",
    external_runner: dict | None = None,
) -> dict:
    row = {
        "scene_id": scene_id,
        "output_path": str(frame_path),
        "preview_path": str(preview_path) if backend_output_ready(preview_path) else "",
        "poster_path": str(poster_path) if backend_output_ready(poster_path) else "",
        "clip_path": str(clip_path) if backend_output_ready(clip_path) else "",
        "alternate_root": str(alternates_root) if alternates_root.exists() else "",
        "manifest_path": str(manifest_path) if manifest_path.exists() else "",
        "status": status,
        "source_type": source_type,
        "backend_mode": backend_mode,
        "clip_status": clip_status or ("completed" if backend_output_ready(clip_path) else ""),
    }
    if external_runner:
        row["external_runner"] = external_runner
    return row


def ensure_scene_backend_derivatives(
    frame_path: Path,
    preview_path: Path,
    poster_path: Path,
    alternates_root: Path,
    payload: dict,
    width: int,
    height: int,
    ffmpeg_path: Path | None,
    clip_path: Path,
    fps: int,
    crf: int,
) -> tuple[list[Path], str, float]:
    rendered = fit_rgb_image(frame_path, width, height)
    if not backend_output_ready(poster_path):
        poster_path.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(poster_path, quality=95)
    if not backend_output_ready(preview_path):
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview = rendered.copy()
        preview.thumbnail((960, 540), Image.Resampling.LANCZOS)
        preview.save(preview_path, quality=90)
    alternates_root.mkdir(parents=True, exist_ok=True)
    variant_paths = sorted(path for path in alternates_root.glob("*.png") if backend_output_ready(path))
    if not variant_paths:
        accent = accent_from_payload(payload)
        for index, label in enumerate(backend_variant_labels(payload), start=1):
            variant_path = alternates_root / f"{index:02d}_{label}.png"
            variant_image(rendered, accent, index).save(variant_path, quality=95)
            variant_paths.append(variant_path)
    clip_status = "completed" if backend_output_ready(clip_path) else "not_requested"
    clip_duration_seconds = max(3.2, min(8.0, 1.2 * max(1, len(variant_paths))))
    if ffmpeg_path is not None and variant_paths and not backend_output_ready(clip_path):
        try:
            with tempfile.TemporaryDirectory(prefix=f"backend_scene_{frame_path.stem}_", dir=str(frame_path.parent)) as temp_dir:
                concat_path = Path(temp_dir) / "concat.txt"
                per_frame_duration = clip_duration_seconds / max(1, len(variant_paths))
                build_concat_file([(path, per_frame_duration) for path in variant_paths], concat_path)
                encode_scene_video(ffmpeg_path, concat_path, clip_path, fps, width, height, crf)
            clip_status = "completed"
        except Exception:
            clip_status = "failed"
    return variant_paths, clip_status, clip_duration_seconds


def materialize_scene_backend_frame(
    assets_root: Path,
    payload: dict,
    width: int,
    height: int,
    fps: int,
    crf: int,
    ffmpeg_path: Path | None = None,
    force: bool = False,
    cfg: dict | None = None,
    backend_input_path: Path | None = None,
) -> dict:
    cfg = cfg or {}
    scene_id = str(payload.get("scene_id", "")).strip() or "scene"
    scene_root = scene_output_root(assets_root, scene_id)
    backend_input_path = backend_input_path or (assets_root / f"{scene_id}_backend_input.json")
    frame_path = scene_frame_output_path(assets_root, scene_id)
    preview_path = scene_preview_output_path(assets_root, scene_id)
    poster_path = scene_poster_output_path(assets_root, scene_id)
    clip_path = scene_clip_output_path(assets_root, scene_id)
    alternates_root = scene_alternates_root(assets_root, scene_id)
    manifest_path = scene_root / "backend_frame_manifest.json"

    existing_manifest = read_json(manifest_path, {}) if manifest_path.exists() else {}
    existing_backend_mode = str(existing_manifest.get("backend_mode", "materialized_local_backend_scene_pack") or "materialized_local_backend_scene_pack")
    if backend_output_ready(frame_path) and not force:
        return scene_backend_row(
            scene_id=scene_id,
            frame_path=frame_path,
            preview_path=preview_path,
            poster_path=poster_path,
            clip_path=clip_path,
            alternates_root=alternates_root,
            manifest_path=manifest_path,
            status="existing",
            source_type="existing_backend_frame",
            backend_mode=existing_backend_mode,
            clip_status=str(existing_manifest.get("clip_status", "")),
            external_runner=existing_manifest.get("external_runner") if isinstance(existing_manifest.get("external_runner"), dict) else None,
        )

    backend_candidates = payload.get("backend_candidates", {}) if isinstance(payload.get("backend_candidates"), dict) else {}
    image_candidates = backend_candidates.get("image", []) if isinstance(backend_candidates.get("image"), list) else []
    video_candidates = backend_candidates.get("video", []) if isinstance(backend_candidates.get("video"), list) else []
    scene_root.mkdir(parents=True, exist_ok=True)
    external_runner = run_external_backend_runner(
        cfg,
        "storyboard_scene_runner",
        context={
            "scene_id": scene_id,
            "scene_root": scene_root,
            "assets_root": assets_root,
            "backend_input_path": backend_input_path,
            "frame_path": frame_path,
            "preview_path": preview_path,
            "poster_path": poster_path,
            "clip_path": clip_path,
            "alternate_root": alternates_root,
        },
        force=force,
        fallback_cwd=scene_root,
        log_dir=resolve_project_path("logs") / "external_backends" / "storyboard_scene_runner" / scene_id,
    )
    if backend_output_ready(frame_path):
        variant_paths, clip_status, clip_duration_seconds = ensure_scene_backend_derivatives(
            frame_path,
            preview_path,
            poster_path,
            alternates_root,
            payload,
            width,
            height,
            ffmpeg_path,
            clip_path,
            fps,
            crf,
        )
        manifest = {
            "scene_id": scene_id,
            "backend_mode": "configured_external_storyboard_runner",
            "status": "completed",
            "output_path": str(frame_path),
            "preview_path": str(preview_path) if backend_output_ready(preview_path) else "",
            "poster_path": str(poster_path) if backend_output_ready(poster_path) else "",
            "clip_path": str(clip_path) if backend_output_ready(clip_path) else "",
            "clip_status": clip_status,
            "alternate_root": str(alternates_root),
            "alternate_frames": [str(path) for path in variant_paths],
            "scene_video_duration_seconds": round(clip_duration_seconds, 3),
            "source_path": str(frame_path),
            "source_type": "configured_external_runner",
            "backend_image_candidates": image_candidates,
            "backend_video_candidates": video_candidates,
            "positive_prompt": payload.get("positive_prompt", ""),
            "negative_prompt": payload.get("negative_prompt", ""),
            "batch_prompt_line": payload.get("batch_prompt_line", ""),
            "control_hints": payload.get("control_hints", {}),
            "camera_plan": payload.get("camera_plan", {}),
            "continuity": payload.get("continuity", {}),
            "external_runner": external_runner,
        }
        write_json(manifest_path, manifest)
        return manifest

    source_path = first_existing_path(candidate_image_paths(assets_root, scene_id, payload))
    if source_path is None:
        return scene_backend_row(
            scene_id=scene_id,
            frame_path=frame_path,
            preview_path=preview_path,
            poster_path=poster_path,
            clip_path=clip_path,
            alternates_root=alternates_root,
            manifest_path=manifest_path,
            status="missing_source",
            source_type="missing",
            backend_mode="materialized_local_backend_scene_pack",
            external_runner=external_runner,
        )

    backend_strength = min(1.0, (len(image_candidates) * 0.22) + (len(video_candidates) * 0.16))
    accent = accent_from_payload(payload)
    previous_frame = preferred_previous_frame(assets_root, payload)

    alternates_root.mkdir(parents=True, exist_ok=True)
    working = fit_rgb_image(source_path, width, height)
    rendered = apply_backend_grade(working, payload, accent, previous_frame, backend_strength)
    rendered.save(frame_path, quality=95)
    rendered.save(poster_path, quality=95)

    preview = rendered.copy()
    preview.thumbnail((960, 540), Image.Resampling.LANCZOS)
    preview.save(preview_path, quality=90)

    variant_paths: list[Path] = []
    for index, label in enumerate(backend_variant_labels(payload), start=1):
        variant_path = alternates_root / f"{index:02d}_{label}.png"
        variant_image(rendered, accent, index).save(variant_path, quality=95)
        variant_paths.append(variant_path)

    clip_status = "not_requested"
    clip_duration_seconds = max(3.2, min(8.0, 1.2 * max(1, len(variant_paths))))
    if ffmpeg_path is not None and variant_paths:
        try:
            with tempfile.TemporaryDirectory(prefix=f"backend_scene_{scene_id}_", dir=str(scene_root)) as temp_dir:
                concat_path = Path(temp_dir) / "concat.txt"
                per_frame_duration = clip_duration_seconds / max(1, len(variant_paths))
                build_concat_file([(path, per_frame_duration) for path in variant_paths], concat_path)
                encode_scene_video(ffmpeg_path, concat_path, clip_path, fps, width, height, crf)
            clip_status = "completed"
        except Exception:
            clip_status = "failed"

    manifest = {
        "scene_id": scene_id,
        "backend_mode": "materialized_local_backend_scene_pack",
        "status": "completed",
        "output_path": str(frame_path),
        "preview_path": str(preview_path),
        "poster_path": str(poster_path),
        "clip_path": str(clip_path) if clip_path.exists() else "",
        "clip_status": clip_status,
        "alternate_root": str(alternates_root),
        "alternate_frames": [str(path) for path in variant_paths],
        "scene_video_duration_seconds": round(clip_duration_seconds, 3),
        "source_path": str(source_path),
        "source_type": "seed_asset" if source_path.parent == assets_root else "reference_image",
        "previous_frame": str(previous_frame) if previous_frame else "",
        "backend_image_candidates": image_candidates,
        "backend_video_candidates": video_candidates,
        "positive_prompt": payload.get("positive_prompt", ""),
        "negative_prompt": payload.get("negative_prompt", ""),
        "batch_prompt_line": payload.get("batch_prompt_line", ""),
        "control_hints": payload.get("control_hints", {}),
        "camera_plan": payload.get("camera_plan", {}),
        "continuity": payload.get("continuity", {}),
        "accent_rgb": list(accent),
        "external_runner": external_runner,
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Run Storyboard Backend")
    cfg = load_config()
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)
    shotlist_dir = resolve_project_path("generation/shotlists")
    shotlist_path = (shotlist_dir / f"{args.episode_id}.json") if args.episode_id else find_latest_shotlist(shotlist_dir)
    if shotlist_path is None or not shotlist_path.exists():
        info("No shotlist found for storyboard backend materialization.")
        return

    shotlist = read_json(shotlist_path, {})
    episode_id = str(shotlist.get("episode_id", shotlist_path.stem))
    render_cfg = cfg.get("render", {}) if isinstance(cfg.get("render"), dict) else {}
    width = int(render_cfg.get("width", 1280))
    height = int(render_cfg.get("height", 720))
    fps = max(12, int(render_cfg.get("fps", 30)))
    crf = int(render_cfg.get("crf", 23))
    assets_root = resolve_project_path(cfg["paths"].get("storyboard_assets", "generation/storyboard_assets")) / episode_id
    if not assets_root.exists():
        info("No storyboard asset root found yet. Run 14_generate_storyboard_assets.py first.")
        return

    ffmpeg: Path | None = None
    try:
        ffmpeg = detect_tool(resolve_project_path("tools/ffmpeg/bin"), "ffmpeg")
    except Exception:
        warn("No FFmpeg found. The storyboard backend will still write frames and alternates, but no local backend scene videos.")

    backend_inputs = sorted(assets_root.glob("*_backend_input.json"))
    if not backend_inputs:
        info("No storyboard backend input payloads found. Run 14_generate_storyboard_assets.py first.")
        return

    total_backend_inputs = len(backend_inputs)
    requested_scene_ids = set(args.scene_ids) if args.scene_ids else None
    if requested_scene_ids:
        filtered_inputs: list[Path] = []
        for bi in backend_inputs:
            payload = read_json(bi, {})
            sid = str(payload.get("scene_id", bi.stem.replace("_backend_input", ""))).strip()
            if sid in requested_scene_ids:
                filtered_inputs.append(bi)
        if not filtered_inputs:
            info(f"None of the requested scene IDs were found: {', '.join(sorted(requested_scene_ids))}")
            return
        backend_inputs = filtered_inputs
        info(f"Scene-selective mode: processing {len(backend_inputs)} of {total_backend_inputs} available scene payloads.")

    autosave_target = episode_id
    mark_step_started("54_run_storyboard_backend", autosave_target, {"episode_id": episode_id, "shotlist": str(shotlist_path)})
    reporter = LiveProgressReporter(
        script_name="54_run_storyboard_backend.py",
        total=max(1, len(backend_inputs)),
        phase_label="Storyboard Backend",
        parent_label=episode_id,
    )
    scene_lease_root = distributed_step_runtime_root("54_run_storyboard_backend", episode_id) / "scenes"
    if shared_workers:
        info(f"Shared NAS workers: enabled ({worker_id})")
    try:
        rows: list[dict] = []
        completed_count = 0
        for index, backend_input_path in enumerate(backend_inputs, start=1):
            payload = read_json(backend_input_path, {})
            scene_id = str(payload.get("scene_id", backend_input_path.stem.replace("_backend_input", ""))).strip() or f"scene_{index:04d}"
            with distributed_item_lease(
                root=scene_lease_root,
                lease_name=scene_id,
                cfg=cfg,
                worker_id=worker_id,
                enabled=shared_workers,
                        meta={"step": "54_run_storyboard_backend", "episode_id": episode_id, "scene_id": scene_id, "worker_id": worker_id},
            ) as acquired:
                if not acquired:
                    continue
                reporter.update(index - 1, current_label=scene_id, extra_label="Running now: materialize backend scene pack", force=True)
                row = materialize_scene_backend_frame(assets_root, payload, width, height, fps, crf, ffmpeg, args.force, cfg, backend_input_path)
                row["backend_input_path"] = str(backend_input_path)
                rows.append(row)
                if row.get("status") == "completed":
                    completed_count += 1
                reporter.update(index, current_label=scene_id, extra_label=f"Status: {row.get('status', 'unknown')}")
        if shared_workers and len(rows) < len(backend_inputs):
            rows = []
            completed_count = 0
            for backend_input_path in backend_inputs:
                payload = read_json(backend_input_path, {})
                row = materialize_scene_backend_frame(assets_root, payload, width, height, fps, crf, ffmpeg, False, cfg, backend_input_path)
                row["backend_input_path"] = str(backend_input_path)
                rows.append(row)
                if row.get("status") == "completed":
                    completed_count += 1

        completed_scene_video_count = sum(1 for row in rows if str(row.get("clip_status", "")) == "completed")

        manifest_path = assets_root / f"{episode_id}_storyboard_backend_manifest.json"
        write_json(
            manifest_path,
            {
                "episode_id": episode_id,
                "asset_root": str(assets_root),
                "backend_mode": "materialized_local_backend_scene_pack",
                "scene_runs": rows,
                "completed_scene_count": completed_count,
                "completed_scene_video_count": completed_scene_video_count,
                "requested_scene_count": len(backend_inputs),
            },
        )
        shotlist["storyboard_backend_manifest"] = str(manifest_path)
        shotlist["storyboard_backend_mode"] = "materialized_local_backend_scene_pack"
        write_json(shotlist_path, shotlist)
        reporter.finish(
            current_label=episode_id,
            extra_label=f"Storyboard backend ready: {completed_count}/{len(backend_inputs)} scenes | local scene videos: {completed_scene_video_count}",
        )
        mark_step_completed(
                "54_run_storyboard_backend",
            autosave_target,
            {
                "episode_id": episode_id,
                "manifest": str(manifest_path),
                "completed_scene_count": completed_count,
                "completed_scene_video_count": completed_scene_video_count,
            },
        )
        ok(f"Storyboard backend materialized: {episode_id}")
    except Exception as exc:
        mark_step_failed("54_run_storyboard_backend", str(exc), autosave_target, {"episode_id": episode_id})
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise

