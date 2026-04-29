#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from pipeline_common import (
    PROJECT_ROOT,
    add_shared_worker_arguments,
    LiveProgressReporter,
    adapter_training_status,
    backend_fine_tune_status,
    distributed_item_lease,
    distributed_step_runtime_root,
    ensure_adapter_training_ready,
    ensure_backend_fine_tune_ready,
    ensure_fine_tune_training_ready,
    ensure_foundation_training_ready,
    error,
    fine_tune_training_status,
    headline,
    info,
    load_config,
    load_multi_series_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
    shared_worker_id_for_args,
    shared_workers_enabled_for_args,
    write_json,
    write_text,
)


def load_step08():
    path = Path(__file__).resolve().parent / "08_train_series_model.py"
    spec = importlib.util.spec_from_file_location("step08_generate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


STEP08 = load_step08()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a new episode from the trained series model")
    parser.add_argument("--episode-id", help="Target a specific episode ID such as episode_09 or folge_09.")
    add_shared_worker_arguments(parser)
    return parser.parse_args()


def storyboard_episode_request(episode_id: str, shotlist_payload: dict) -> dict:
    scenes = shotlist_payload.get("scenes", []) or []
    return {
        "episode_id": episode_id,
        "storyboard_plan_mode": shotlist_payload.get("storyboard_plan_mode", ""),
        "display_title": shotlist_payload.get("display_title", episode_id),
        "episode_title": shotlist_payload.get("episode_title", ""),
        "focus_characters": shotlist_payload.get("focus_characters", []) or [],
        "keywords": shotlist_payload.get("keywords", []) or [],
        "scene_requests": [
            {
                "scene_id": scene.get("scene_id", ""),
                "title": scene.get("title", ""),
                "characters": scene.get("characters", []) or [],
                "summary": scene.get("summary", ""),
                "location": scene.get("location", ""),
                "mood": scene.get("mood", ""),
                "generation_plan": scene.get("generation_plan", {}) if isinstance(scene.get("generation_plan", {}), dict) else {},
            }
            for scene in scenes
        ],
    }


def write_storyboard_backend_requests(cfg: dict, episode_id: str, shotlist_payload: dict) -> dict:
    requests_root = resolve_project_path(cfg["paths"].get("storyboard_requests", "generation/storyboard_requests")) / episode_id
    requests_root.mkdir(parents=True, exist_ok=True)
    episode_request = storyboard_episode_request(episode_id, shotlist_payload)
    episode_request_path = requests_root / f"{episode_id}_storyboard_request.json"
    write_json(episode_request_path, episode_request)

    scene_request_paths: list[str] = []
    prompt_preview_lines: list[str] = []
    for scene in episode_request.get("scene_requests", []):
        scene_id = str(scene.get("scene_id", "")).strip() or "scene"
        scene_request_path = requests_root / f"{scene_id}_request.json"
        write_json(scene_request_path, scene)
        scene_request_paths.append(str(scene_request_path))
        generation_plan = scene.get("generation_plan", {}) if isinstance(scene.get("generation_plan", {}), dict) else {}
        prompt_preview_lines.extend(
            [
                f"[{scene_id}] {scene.get('title', '')}",
                f"batch_prompt_line = {generation_plan.get('batch_prompt_line', '')}",
                f"positive_prompt = {generation_plan.get('positive_prompt', '')}",
                f"negative_prompt = {generation_plan.get('negative_prompt', '')}",
                "",
            ]
        )

    prompt_preview_path = requests_root / f"{episode_id}_storyboard_prompts.txt"
    write_text(prompt_preview_path, "\n".join(prompt_preview_lines).strip() + "\n")
    return {
        "episode_request_path": str(episode_request_path),
        "scene_request_paths": scene_request_paths,
        "prompt_preview_path": str(prompt_preview_path),
        "request_dir": str(requests_root),
    }


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Generate New Episode From Trained Model")
    cfg = load_config()
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)

    multi_series_cfg = load_multi_series_config(PROJECT_ROOT)
    active_series = multi_series_cfg.get("active_series", "default")
    if active_series != "default":
        info(f"Active series: {active_series}")

    model_path = resolve_project_path(cfg["paths"]["series_model"])
    if not model_path.exists():
        info("No trained series model found. Run 08_train_series_model.py first.")
        return

    autosave_target = (args.episode_id or "").strip() or "auto_next"
    mark_step_started("13_generate_episode", autosave_target, {"series_model": str(model_path)})
    if shared_workers:
        info(f"Shared NAS workers: enabled ({worker_id})")
    lease_manager = distributed_item_lease(
        root=distributed_step_runtime_root("13_generate_episode", autosave_target),
        lease_name=autosave_target,
        cfg=cfg,
        worker_id=worker_id,
        enabled=shared_workers,
        meta={"step": "13_generate_episode", "scope": autosave_target, "worker_id": worker_id},
    )
    acquired = lease_manager.__enter__()
    if not acquired:
        info(f"Episode generation for '{autosave_target}' is already running on another worker.")
        lease_manager.__exit__(None, None, None)
        return
    reporter = LiveProgressReporter(
        script_name="13_generate_episode.py",
        total=5,
        phase_label="Generate New Episode",
        parent_label=autosave_target,
    )
    try:
        reporter.update(0, current_label="Load Series Model", extra_label="Running now: load the trained model and validate training status", force=True)
        model = read_json(model_path, {})
        if not model:
            reporter.finish(current_label="Series Model", extra_label="Stopped: model is empty")
            info("The series model is empty. Run 08_train_series_model.py first.")
            return
        reporter.update(1, current_label="Validate Training Status", extra_label="Checking foundation, adapter, fine-tune and backend status")
        ensure_foundation_training_ready(cfg, model_path=model_path)
        ensure_adapter_training_ready(cfg)
        adapter_status = adapter_training_status(cfg)
        ensure_fine_tune_training_ready(cfg)
        fine_tune_status = fine_tune_training_status(cfg)
        ensure_backend_fine_tune_ready(cfg)
        backend_status = backend_fine_tune_status(cfg)

        story_dir = resolve_project_path("generation/story_prompts")
        shotlist_dir = resolve_project_path("generation/shotlists")
        story_dir.mkdir(parents=True, exist_ok=True)
        shotlist_dir.mkdir(parents=True, exist_ok=True)

        episode_id = (args.episode_id or "").strip() or STEP08.next_episode_id(story_dir)
        reporter.update(2, current_label=episode_id, extra_label="Running now: generate episode package from trained model")
        episode_package, markdown = STEP08.generate_episode_package(model, cfg, STEP08.parse_episode_index(episode_id))
        story_path = story_dir / f"{episode_id}.md"
        shotlist_path = shotlist_dir / f"{episode_id}.json"
        reporter.update(3, current_label=episode_id, extra_label="Running now: write story and shotlist")
        write_text(story_path, markdown)
        shotlist_payload = {
            "episode_id": episode_id,
            "trained_model": str(model_path),
            "adapter_training_summary": str(adapter_status.get("summary_path", "")) if adapter_status.get("summary_exists") else "",
            "fine_tune_training_summary": str(fine_tune_status.get("summary_path", "")) if fine_tune_status.get("summary_exists") else "",
            "backend_fine_tune_summary": str(backend_status.get("summary_path", "")) if backend_status.get("summary_exists") else "",
            **episode_package,
        }
        write_json(shotlist_path, shotlist_payload)
        reporter.update(4, current_label=episode_id, extra_label="Running now: export model-side storyboard backend requests")
        storyboard_request_paths = write_storyboard_backend_requests(cfg, episode_id, shotlist_payload)
        shotlist_payload.update(
            {
                "storyboard_request": storyboard_request_paths["episode_request_path"],
                "storyboard_request_dir": storyboard_request_paths["request_dir"],
                "storyboard_scene_requests": storyboard_request_paths["scene_request_paths"],
                "storyboard_prompt_preview": storyboard_request_paths["prompt_preview_path"],
            }
        )
        write_json(shotlist_path, shotlist_payload)
        reporter.finish(current_label=episode_id, extra_label=f"Episode geschrieben: {story_path.name}, {shotlist_path.name} und Storyboard-Requests")
        mark_step_completed(
            "13_generate_episode",
            episode_id,
            {
                "series_model": str(model_path),
                "story_path": str(story_path),
                "shotlist_path": str(shotlist_path),
                "storyboard_request": storyboard_request_paths["episode_request_path"],
            },
        )
        ok(f"Neue Episode aus trainiertem Modell erzeugt: {episode_id}")
    except Exception as exc:
        mark_step_failed("13_generate_episode", str(exc), autosave_target, {"series_model": str(model_path)})
        raise
    finally:
        lease_manager.__exit__(None, None, None)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise

