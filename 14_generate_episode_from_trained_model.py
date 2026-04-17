#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from pipeline_common import (
    adapter_training_status,
    backend_fine_tune_status,
    ensure_adapter_training_ready,
    ensure_backend_fine_tune_ready,
    ensure_fine_tune_training_ready,
    ensure_foundation_training_ready,
    error,
    fine_tune_training_status,
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
    parser = argparse.ArgumentParser(description="Neue Episode aus dem trainierten Serienmodell erzeugen")
    parser.add_argument("--episode-id", help="Zielt auf eine konkrete Folge-ID wie folge_09.")
    return parser.parse_args()


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Neue Episode aus trainiertem Modell erzeugen")
    cfg = load_config()
    model_path = resolve_project_path(cfg["paths"]["series_model"])
    if not model_path.exists():
        info("Kein trainiertes Serienmodell gefunden. Fuehre zuerst 08_train_series_model.py aus.")
        return

    autosave_target = (args.episode_id or "").strip() or "auto_next"
    mark_step_started("14_generate_episode_from_trained_model", autosave_target, {"series_model": str(model_path)})
    try:
        model = read_json(model_path, {})
        if not model:
            info("Das Serienmodell ist leer. Fuehre zuerst 08_train_series_model.py aus.")
            return
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
        episode_package, markdown = STEP08.generate_episode_package(model, cfg, STEP08.parse_episode_index(episode_id))
        story_path = story_dir / f"{episode_id}.md"
        shotlist_path = shotlist_dir / f"{episode_id}.json"
        write_text(story_path, markdown)
        write_json(
            shotlist_path,
            {
                "episode_id": episode_id,
                "trained_model": str(model_path),
                "adapter_training_summary": str(adapter_status.get("summary_path", "")) if adapter_status.get("summary_exists") else "",
                "fine_tune_training_summary": str(fine_tune_status.get("summary_path", "")) if fine_tune_status.get("summary_exists") else "",
                "backend_fine_tune_summary": str(backend_status.get("summary_path", "")) if backend_status.get("summary_exists") else "",
                **episode_package,
            },
        )
        mark_step_completed(
            "14_generate_episode_from_trained_model",
            episode_id,
            {"series_model": str(model_path), "story_path": str(story_path), "shotlist_path": str(shotlist_path)},
        )
        ok(f"Neue Episode aus trainiertem Modell erzeugt: {episode_id}")
    except Exception as exc:
        mark_step_failed("14_generate_episode_from_trained_model", str(exc), autosave_target, {"series_model": str(model_path)})
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
