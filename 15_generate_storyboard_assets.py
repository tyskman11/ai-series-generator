#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from pipeline_common import (
    LiveProgressReporter,
    PROJECT_ROOT,
    detect_tool,
    error,
    headline,
    info,
    load_backend_run_index,
    load_config,
    latest_matching_file,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    read_json,
    resolve_stored_project_path,
    rerun_in_runtime,
    resolve_project_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate storyboard scene assets from exported storyboard requests")
    parser.add_argument("--episode-id", help="Target a specific episode ID such as episode_09 or folge_09.")
    parser.add_argument("--force", action="store_true", help="Recreate existing storyboard scene assets.")
    return parser.parse_args()


def find_latest_shotlist(shotlist_dir: Path) -> Path | None:
    episode_id = os.environ.get("SERIES_STORYBOARD_EPISODE", "").strip()
    if episode_id:
        candidate = shotlist_dir / f"{episode_id}.json"
        if candidate.exists():
            return candidate
    return latest_matching_file(shotlist_dir, "*.json")


def fit_image_to_frame(source_path: Path, output_path: Path, width: int, height: int) -> bool:
    if not source_path.exists():
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(source_path).convert("RGB")
    image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    image.save(output_path, quality=95)
    return output_path.exists()


def build_seed_asset(
    source_path: Path,
    output_path: Path,
    width: int,
    height: int,
    *,
    continuity_path: Path | None = None,
    backend_ready: bool = False,
) -> bool:
    if not source_path.exists():
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(source_path).convert("RGB")
    image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    if continuity_path is not None and continuity_path.exists():
        continuity = Image.open(continuity_path).convert("RGB")
        continuity = ImageOps.fit(continuity, (width, height), method=Image.Resampling.LANCZOS)
        image = Image.blend(image, continuity, 0.22)
    if backend_ready:
        image = ImageEnhance.Contrast(image).enhance(1.08)
        image = ImageEnhance.Color(image).enhance(1.04)
        image = image.filter(ImageFilter.UnsharpMask(radius=1.4, percent=115, threshold=3))
    image.save(output_path, quality=95)
    return output_path.exists()


def extract_video_frame(
    ffmpeg: Path,
    video_file: Path,
    output_path: Path,
    width: int,
    height: int,
    force: bool,
) -> bool:
    if output_path.exists() and not force:
        return True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y" if force else "-n",
        "-ss",
        "1.500",
        "-i",
        str(video_file),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
        str(output_path),
    ]
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return result.returncode == 0 and output_path.exists()


def best_reference_image(scene_request: dict) -> Path | None:
    generation_plan = scene_request.get("generation_plan", {}) if isinstance(scene_request.get("generation_plan", {}), dict) else {}
    reference_slots = generation_plan.get("reference_slots", []) if isinstance(generation_plan.get("reference_slots", []), list) else []
    for slot in reference_slots:
        if not isinstance(slot, dict):
            continue
        for key in ("context_images", "portrait_images"):
            values = slot.get(key, [])
            if isinstance(values, list):
                for value in values:
                    candidate = Path(str(value))
                    if candidate.exists():
                        return candidate
    return None


def best_environment_video(scene_request: dict) -> Path | None:
    generation_plan = scene_request.get("generation_plan", {}) if isinstance(scene_request.get("generation_plan", {}), dict) else {}
    reference_slots = generation_plan.get("reference_slots", []) if isinstance(generation_plan.get("reference_slots", []), list) else []
    for slot in reference_slots:
        if not isinstance(slot, dict):
            continue
        if slot.get("type") != "environment":
            continue
        candidate = Path(str(slot.get("video_file", "")))
        if candidate.exists():
            return candidate
    return None


def previous_scene_asset(assets_root: Path, scene_request: dict) -> Path | None:
    generation_plan = scene_request.get("generation_plan", {}) if isinstance(scene_request.get("generation_plan", {}), dict) else {}
    continuity = generation_plan.get("continuity", {}) if isinstance(generation_plan.get("continuity", {}), dict) else {}
    previous_scene_id = str(continuity.get("previous_scene_id", "")).strip()
    if not previous_scene_id:
        return None
    for suffix in (".png", ".jpg"):
        candidate = assets_root / f"{previous_scene_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def scene_backend_candidates(scene_request: dict, backend_index: dict[str, dict]) -> dict[str, list[dict]]:
    scene_characters = scene_request.get("characters", []) if isinstance(scene_request.get("characters", []), list) else []
    image_candidates: list[dict] = []
    video_candidates: list[dict] = []
    for name in scene_characters:
        normalized = str(name or "").strip().lower()
        row = backend_index.get(normalized)
        if not row or not bool(row.get("training_ready", False)):
            continue
        backends = row.get("backends", {}) if isinstance(row.get("backends"), dict) else {}
        for modality, target in (("image", image_candidates), ("video", video_candidates)):
            backend_payload = backends.get(modality, {}) if isinstance(backends.get(modality), dict) else {}
            if not bool(backend_payload.get("ready", False)):
                continue
            artifacts = backend_payload.get("artifacts", {}) if isinstance(backend_payload.get("artifacts"), dict) else {}
            bundle_path = resolve_stored_project_path(artifacts.get("bundle_path", ""))
            weights_path = resolve_stored_project_path(artifacts.get("weights_path", ""))
            job_path = resolve_stored_project_path(artifacts.get("job_path", ""))
            if not bundle_path.exists() or not weights_path.exists() or not job_path.exists():
                continue
            target.append(
                {
                    "character": row.get("character", name),
                    "backend": backend_payload.get("backend", ""),
                    "job_path": str(job_path),
                    "bundle_path": str(bundle_path),
                    "weights_path": str(weights_path),
                }
            )
    return {"image": image_candidates, "video": video_candidates}


def write_scene_backend_input(
    assets_root: Path,
    episode_id: str,
    scene_request: dict,
    backend_candidates: dict[str, list[dict]],
    source_hints: dict[str, str],
) -> Path:
    scene_id = str(scene_request.get("scene_id", "")).strip() or "scene"
    generation_plan = scene_request.get("generation_plan", {}) if isinstance(scene_request.get("generation_plan", {}), dict) else {}
    payload = {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "title": scene_request.get("title", ""),
        "characters": scene_request.get("characters", []) if isinstance(scene_request.get("characters", []), list) else [],
        "backend_mode": "local_storyboard_seed_package",
        "positive_prompt": generation_plan.get("positive_prompt", ""),
        "negative_prompt": generation_plan.get("negative_prompt", ""),
        "batch_prompt_line": generation_plan.get("batch_prompt_line", ""),
        "camera_plan": generation_plan.get("camera_plan", {}),
        "control_hints": generation_plan.get("control_hints", {}),
        "continuity": generation_plan.get("continuity", {}),
        "reference_slots": generation_plan.get("reference_slots", []),
        "backend_candidates": backend_candidates,
        "source_hints": source_hints,
    }
    backend_input_path = assets_root / f"{scene_id}_backend_input.json"
    write_json(backend_input_path, payload)
    return backend_input_path


def build_scene_asset(
    ffmpeg: Path,
    assets_root: Path,
    episode_id: str,
    scene_request: dict,
    backend_index: dict[str, dict],
    width: int,
    height: int,
    force: bool,
) -> dict:
    scene_id = str(scene_request.get("scene_id", "")).strip() or "scene"
    output_path = assets_root / f"{scene_id}.png"
    continuity_asset = previous_scene_asset(assets_root, scene_request)
    backend_candidates = scene_backend_candidates(scene_request, backend_index)
    backend_ready = bool(backend_candidates["image"] or backend_candidates["video"])
    source_hints = {
        "environment_video": "",
        "reference_image": "",
        "continuity_asset": str(continuity_asset) if continuity_asset and continuity_asset.exists() else "",
    }
    backend_input_path = write_scene_backend_input(assets_root, episode_id, scene_request, backend_candidates, source_hints)
    if output_path.exists() and not force:
        return {
            "scene_id": scene_id,
            "asset_path": str(output_path),
            "source_type": "existing",
            "source_path": str(output_path),
            "backend_input_path": str(backend_input_path),
            "backend_ready_image_characters": [row["character"] for row in backend_candidates["image"]],
            "backend_ready_video_characters": [row["character"] for row in backend_candidates["video"]],
        }

    environment_video = best_environment_video(scene_request)
    if environment_video and extract_video_frame(ffmpeg, environment_video, output_path, width, height, force):
        source_hints["environment_video"] = str(environment_video)
        write_scene_backend_input(assets_root, episode_id, scene_request, backend_candidates, source_hints)
        return {
            "scene_id": scene_id,
            "asset_path": str(output_path),
            "source_type": "environment_video",
            "source_path": str(environment_video),
            "backend_input_path": str(backend_input_path),
            "backend_ready_image_characters": [row["character"] for row in backend_candidates["image"]],
            "backend_ready_video_characters": [row["character"] for row in backend_candidates["video"]],
        }

    reference_image = best_reference_image(scene_request)
    if reference_image and build_seed_asset(
        reference_image,
        output_path,
        width,
        height,
        continuity_path=continuity_asset,
        backend_ready=backend_ready,
    ):
        source_hints["reference_image"] = str(reference_image)
        write_scene_backend_input(assets_root, episode_id, scene_request, backend_candidates, source_hints)
        return {
            "scene_id": scene_id,
            "asset_path": str(output_path),
            "source_type": "reference_seed",
            "source_path": str(reference_image),
            "backend_input_path": str(backend_input_path),
            "backend_ready_image_characters": [row["character"] for row in backend_candidates["image"]],
            "backend_ready_video_characters": [row["character"] for row in backend_candidates["video"]],
        }

    if continuity_asset and continuity_asset.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(continuity_asset, output_path)
        write_scene_backend_input(assets_root, episode_id, scene_request, backend_candidates, source_hints)
        return {
            "scene_id": scene_id,
            "asset_path": str(output_path),
            "source_type": "continuity_reuse",
            "source_path": str(continuity_asset),
            "backend_input_path": str(backend_input_path),
            "backend_ready_image_characters": [row["character"] for row in backend_candidates["image"]],
            "backend_ready_video_characters": [row["character"] for row in backend_candidates["video"]],
        }

    return {
        "scene_id": scene_id,
        "asset_path": str(output_path),
        "source_type": "missing",
        "source_path": "",
        "backend_input_path": str(backend_input_path),
        "backend_ready_image_characters": [row["character"] for row in backend_candidates["image"]],
        "backend_ready_video_characters": [row["character"] for row in backend_candidates["video"]],
    }


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Generate Storyboard Assets")
    cfg = load_config()
    shotlist_dir = resolve_project_path("generation/shotlists")
    shotlist_path = (shotlist_dir / f"{args.episode_id}.json") if args.episode_id else find_latest_shotlist(shotlist_dir)
    if shotlist_path is None or not shotlist_path.exists():
        info("No shotlist found for storyboard asset generation.")
        return

    shotlist = read_json(shotlist_path, {})
    episode_id = str(shotlist.get("episode_id", shotlist_path.stem))
    request_dir = Path(str(shotlist.get("storyboard_request_dir", "")))
    episode_request_path = Path(str(shotlist.get("storyboard_request", "")))
    if not episode_request_path.exists():
        info("No storyboard request export found. Run 14_generate_episode_from_trained_model.py first.")
        return

    render_cfg = cfg.get("render", {})
    width = int(render_cfg.get("width", 1280))
    height = int(render_cfg.get("height", 720))
    ffmpeg = detect_tool(PROJECT_ROOT / "tools" / "ffmpeg" / "bin", "ffmpeg")
    assets_root = resolve_project_path(cfg["paths"].get("storyboard_assets", "generation/storyboard_assets")) / episode_id
    assets_root.mkdir(parents=True, exist_ok=True)
    backend_index = load_backend_run_index(cfg)

    episode_request = read_json(episode_request_path, {})
    scene_requests = episode_request.get("scene_requests", []) if isinstance(episode_request.get("scene_requests", []), list) else []
    autosave_target = episode_id
    mark_step_started("15_generate_storyboard_assets", autosave_target, {"episode_id": episode_id, "shotlist": str(shotlist_path)})
    reporter = LiveProgressReporter(
        script_name="15_generate_storyboard_assets.py",
        total=max(1, len(scene_requests)),
        phase_label="Generate Storyboard Assets",
        parent_label=episode_id,
    )
    try:
        generated_assets: list[dict] = []
        for index, scene_request in enumerate(scene_requests, start=1):
            scene_id = str(scene_request.get("scene_id", "")).strip() or f"scene_{index:04d}"
            reporter.update(index - 1, current_label=scene_id, extra_label="Running now: build or reuse storyboard scene frame", force=True)
            asset_row = build_scene_asset(ffmpeg, assets_root, episode_id, scene_request, backend_index, width, height, args.force)
            generated_assets.append(asset_row)
            backend_label = asset_row.get("backend_ready_image_characters", []) or asset_row.get("backend_ready_video_characters", [])
            if backend_label:
                reporter.update(index, current_label=scene_id, extra_label=f"Source: {asset_row['source_type']} | Backends: {', '.join(backend_label)}")
            else:
                reporter.update(index, current_label=scene_id, extra_label=f"Source: {asset_row['source_type']}")

        manifest_path = assets_root / f"{episode_id}_storyboard_assets_manifest.json"
        write_json(
            manifest_path,
            {
                "episode_id": episode_id,
                "shotlist": str(shotlist_path),
                "request_dir": str(request_dir),
                "episode_request": str(episode_request_path),
                "asset_root": str(assets_root),
                "scene_assets": generated_assets,
            },
        )
        shotlist["storyboard_assets_root"] = str(assets_root)
        shotlist["storyboard_assets_manifest"] = str(manifest_path)
        write_json(shotlist_path, shotlist)
        reporter.finish(current_label=episode_id, extra_label=f"Storyboard assets ready: {len(generated_assets)} scenes")
        mark_step_completed(
            "15_generate_storyboard_assets",
            autosave_target,
            {"episode_id": episode_id, "asset_root": str(assets_root), "manifest": str(manifest_path)},
        )
        ok(f"Storyboard assets generated: {episode_id}")
    except Exception as exc:
        mark_step_failed("15_generate_storyboard_assets", str(exc), autosave_target, {"episode_id": episode_id})
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
