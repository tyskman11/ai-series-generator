#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from pipeline_common import (
    LiveProgressReporter,
    error,
    headline,
    info,
    load_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize local storyboard backend frames from per-scene backend input payloads"
    )
    parser.add_argument("--episode-id", help="Target a specific episode ID such as folge_09.")
    parser.add_argument("--force", action="store_true", help="Recreate already materialized backend frames.")
    return parser.parse_args()


def find_latest_shotlist(shotlist_dir: Path) -> Path | None:
    episode_id = os.environ.get("SERIES_STORYBOARD_EPISODE", "").strip()
    if episode_id:
        candidate = shotlist_dir / f"{episode_id}.json"
        if candidate.exists():
            return candidate
    files = sorted(shotlist_dir.glob("folge_*.json"))
    return files[-1] if files else None


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
        if candidate.exists() and candidate.stat().st_size > 0:
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


def materialize_scene_backend_frame(
    assets_root: Path,
    payload: dict,
    width: int,
    height: int,
    force: bool,
) -> dict:
    scene_id = str(payload.get("scene_id", "")).strip() or "scene"
    scene_root = assets_root / scene_id
    frame_path = scene_root / "frame.png"
    preview_path = scene_root / "preview.jpg"
    manifest_path = scene_root / "backend_frame_manifest.json"

    if frame_path.exists() and frame_path.stat().st_size > 0 and not force:
        return {
            "scene_id": scene_id,
            "output_path": str(frame_path),
            "preview_path": str(preview_path) if preview_path.exists() else "",
            "manifest_path": str(manifest_path) if manifest_path.exists() else "",
            "status": "existing",
            "source_type": "existing_backend_frame",
            "backend_mode": "materialized_local_backend_frame",
        }

    source_path = first_existing_path(candidate_image_paths(assets_root, scene_id, payload))
    if source_path is None:
        return {
            "scene_id": scene_id,
            "output_path": str(frame_path),
            "preview_path": str(preview_path),
            "manifest_path": str(manifest_path),
            "status": "missing_source",
            "source_type": "missing",
            "backend_mode": "materialized_local_backend_frame",
        }

    backend_candidates = payload.get("backend_candidates", {}) if isinstance(payload.get("backend_candidates"), dict) else {}
    image_candidates = backend_candidates.get("image", []) if isinstance(backend_candidates.get("image"), list) else []
    video_candidates = backend_candidates.get("video", []) if isinstance(backend_candidates.get("video"), list) else []
    backend_strength = min(1.0, (len(image_candidates) * 0.22) + (len(video_candidates) * 0.16))
    accent = accent_from_payload(payload)
    previous_frame = preferred_previous_frame(assets_root, payload)

    scene_root.mkdir(parents=True, exist_ok=True)
    working = fit_rgb_image(source_path, width, height)
    rendered = apply_backend_grade(working, payload, accent, previous_frame, backend_strength)
    rendered.save(frame_path, quality=95)

    preview = rendered.copy()
    preview.thumbnail((960, 540), Image.Resampling.LANCZOS)
    preview.save(preview_path, quality=90)

    manifest = {
        "scene_id": scene_id,
        "backend_mode": "materialized_local_backend_frame",
        "status": "completed",
        "output_path": str(frame_path),
        "preview_path": str(preview_path),
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
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Run Storyboard Backend")
    cfg = load_config()
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
    assets_root = resolve_project_path(cfg["paths"].get("storyboard_assets", "generation/storyboard_assets")) / episode_id
    if not assets_root.exists():
        info("No storyboard asset root found yet. Run 15_generate_storyboard_assets.py first.")
        return

    backend_inputs = sorted(assets_root.glob("*_backend_input.json"))
    if not backend_inputs:
        info("No storyboard backend input payloads found. Run 15_generate_storyboard_assets.py first.")
        return

    autosave_target = episode_id
    mark_step_started("16_run_storyboard_backend", autosave_target, {"episode_id": episode_id, "shotlist": str(shotlist_path)})
    reporter = LiveProgressReporter(
        script_name="16_run_storyboard_backend.py",
        total=max(1, len(backend_inputs)),
        phase_label="Storyboard Backend",
        parent_label=episode_id,
    )
    try:
        rows: list[dict] = []
        completed_count = 0
        for index, backend_input_path in enumerate(backend_inputs, start=1):
            payload = read_json(backend_input_path, {})
            scene_id = str(payload.get("scene_id", backend_input_path.stem.replace("_backend_input", ""))).strip() or f"scene_{index:04d}"
            reporter.update(index - 1, current_label=scene_id, extra_label="Running now: materialize backend frame", force=True)
            row = materialize_scene_backend_frame(assets_root, payload, width, height, args.force)
            row["backend_input_path"] = str(backend_input_path)
            rows.append(row)
            if row.get("status") == "completed":
                completed_count += 1
            reporter.update(index, current_label=scene_id, extra_label=f"Status: {row.get('status', 'unknown')}")

        manifest_path = assets_root / f"{episode_id}_storyboard_backend_manifest.json"
        write_json(
            manifest_path,
            {
                "episode_id": episode_id,
                "asset_root": str(assets_root),
                "backend_mode": "materialized_local_backend_frame",
                "scene_runs": rows,
                "completed_scene_count": completed_count,
                "requested_scene_count": len(backend_inputs),
            },
        )
        shotlist["storyboard_backend_manifest"] = str(manifest_path)
        shotlist["storyboard_backend_mode"] = "materialized_local_backend_frame"
        write_json(shotlist_path, shotlist)
        reporter.finish(
            current_label=episode_id,
            extra_label=f"Storyboard backend ready: {completed_count}/{len(backend_inputs)} scenes",
        )
        mark_step_completed(
            "16_run_storyboard_backend",
            autosave_target,
            {"episode_id": episode_id, "manifest": str(manifest_path), "completed_scene_count": completed_count},
        )
        ok(f"Storyboard backend materialized: {episode_id}")
    except Exception as exc:
        mark_step_failed("16_run_storyboard_backend", str(exc), autosave_target, {"episode_id": episode_id})
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
