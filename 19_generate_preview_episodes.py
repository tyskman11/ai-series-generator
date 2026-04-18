#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from pipeline_common import (
    LiveProgressReporter,
    SCRIPT_DIR,
    error,
    headline,
    load_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    open_review_item_count,
    ok,
    rerun_in_runtime,
    runtime_python,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multiple visible preview episodes in one run.")
    parser.add_argument("--count", type=int, default=2, help="Number of new episodes. Default: 2")
    return parser.parse_args()


def story_dir() -> Path:
    return SCRIPT_DIR / "ai_series_project" / "generation" / "story_prompts"


def latest_episode_id() -> str | None:
    files = sorted(story_dir().glob("folge_*.md"))
    return files[-1].stem if files else None


def run_step(script_name: str, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(
        [str(runtime_python()), str(SCRIPT_DIR / script_name)],
        env=env,
        cwd=str(SCRIPT_DIR),
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    count = max(1, int(args.count))
    cfg = load_config()
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    adapter_cfg = cfg.get("adapter_training", {}) if isinstance(cfg.get("adapter_training"), dict) else {}
    fine_tune_cfg = cfg.get("fine_tune_training", {}) if isinstance(cfg.get("fine_tune_training"), dict) else {}
    backend_cfg = cfg.get("backend_fine_tune", {}) if isinstance(cfg.get("backend_fine_tune"), dict) else {}

    headline("Generate Multiple Visible Preview Episodes")
    generated: list[str] = []
    autosave_target = f"count_{count}"
    mark_step_started("19_generate_preview_episodes", autosave_target, {"requested_count": count})
    try:
        review_count = open_review_item_count(cfg)
        if review_count > 0:
            raise RuntimeError(
                f"Es gibt noch {review_count} offene Review-Faelle. "
                "Run 06_review_unknowns.py first before training, generation, or render can start."
            )
        planned_steps = ["07_build_dataset.py", "08_train_series_model.py"]
        if bool(foundation_cfg.get("required_before_generate", True)) or bool(foundation_cfg.get("required_before_render", True)):
            planned_steps.extend(["09_prepare_foundation_training.py", "10_train_foundation_models.py"])
            if bool(adapter_cfg.get("auto_train_after_foundation", True)):
                planned_steps.append("11_train_adapter_models.py")
                if bool(fine_tune_cfg.get("auto_train_after_adapter", True)):
                    planned_steps.append("12_train_fine_tune_models.py")
                    if bool(backend_cfg.get("auto_run_after_fine_tune", True)):
                        planned_steps.append("13_run_backend_finetunes.py")
        planned_steps.extend(
            [
                "14_generate_episode_from_trained_model.py",
                "15_generate_storyboard_assets.py",
                "16_run_storyboard_backend.py",
                "17_build_series_bible.py",
                "18_render_episode.py",
            ]
            * count
        )
        reporter = LiveProgressReporter(
            script_name="19_generate_preview_episodes.py",
            total=len(planned_steps),
            phase_label="Generate Preview Episodes",
            parent_label=f"Anzahl: {count}",
        )
        completed_steps = 0

        reporter.update(completed_steps, current_label="Build Datasets", extra_label="Running now: 07_build_dataset.py", force=True)
        run_step("07_build_dataset.py")
        completed_steps += 1
        reporter.update(completed_steps, current_label="Build Datasets", extra_label="Completed: 07_build_dataset.py")

        reporter.update(completed_steps, current_label="Train Series Model", extra_label="Running now: 08_train_series_model.py", force=True)
        run_step("08_train_series_model.py")
        completed_steps += 1
        reporter.update(completed_steps, current_label="Train Series Model", extra_label="Completed: 08_train_series_model.py")

        if bool(foundation_cfg.get("required_before_generate", True)) or bool(foundation_cfg.get("required_before_render", True)):
            reporter.update(completed_steps, current_label="Prepare Foundation Data", extra_label="Running now: 09_prepare_foundation_training.py", force=True)
            run_step("09_prepare_foundation_training.py")
            completed_steps += 1
            reporter.update(completed_steps, current_label="Prepare Foundation Data", extra_label="Completed: 09_prepare_foundation_training.py")

            reporter.update(completed_steps, current_label="Train Foundation Packs", extra_label="Running now: 10_train_foundation_models.py", force=True)
            run_step("10_train_foundation_models.py")
            completed_steps += 1
            reporter.update(completed_steps, current_label="Train Foundation Packs", extra_label="Completed: 10_train_foundation_models.py")

            if bool(adapter_cfg.get("auto_train_after_foundation", True)):
                reporter.update(completed_steps, current_label="Train Adapter Profiles", extra_label="Running now: 11_train_adapter_models.py", force=True)
                run_step("11_train_adapter_models.py")
                completed_steps += 1
                reporter.update(completed_steps, current_label="Train Adapter Profiles", extra_label="Completed: 11_train_adapter_models.py")
                if bool(fine_tune_cfg.get("auto_train_after_adapter", True)):
                    reporter.update(completed_steps, current_label="Train Fine-Tune Profiles", extra_label="Running now: 12_train_fine_tune_models.py", force=True)
                    run_step("12_train_fine_tune_models.py")
                    completed_steps += 1
                    reporter.update(completed_steps, current_label="Train Fine-Tune Profiles", extra_label="Completed: 12_train_fine_tune_models.py")
                    if bool(backend_cfg.get("auto_run_after_fine_tune", True)):
                        reporter.update(completed_steps, current_label="Prepare Backend Fine-Tunes", extra_label="Running now: 13_run_backend_finetunes.py", force=True)
                        run_step("13_run_backend_finetunes.py")
                        completed_steps += 1
                        reporter.update(completed_steps, current_label="Prepare Backend Fine-Tunes", extra_label="Completed: 13_run_backend_finetunes.py")
        for index in range(count):
            before = latest_episode_id()
            reporter.update(completed_steps, current_label=f"Generate episode {index + 1}", extra_label="Running now: 14_generate_episode_from_trained_model.py", force=True)
            run_step("14_generate_episode_from_trained_model.py")
            completed_steps += 1
            episode_id = latest_episode_id()
            if not episode_id or episode_id == before:
                raise RuntimeError("Could not determine the new episode.")
            reporter.update(completed_steps, current_label=episode_id, extra_label=f"Episode generated: {episode_id}")
            env = os.environ.copy()
            env["SERIES_STORYBOARD_EPISODE"] = episode_id
            reporter.update(completed_steps, current_label=f"{episode_id} storyboard assets", extra_label="Running now: 15_generate_storyboard_assets.py", force=True)
            run_step("15_generate_storyboard_assets.py", env=env)
            completed_steps += 1
            reporter.update(completed_steps, current_label=episode_id, extra_label=f"Storyboard assets ready: {episode_id}")
            reporter.update(completed_steps, current_label=f"{episode_id} backend frames", extra_label="Running now: 16_run_storyboard_backend.py", force=True)
            run_step("16_run_storyboard_backend.py", env=env)
            completed_steps += 1
            reporter.update(completed_steps, current_label=episode_id, extra_label=f"Storyboard backend frames ready: {episode_id}")
            reporter.update(completed_steps, current_label="Update series bible", extra_label="Running now: 17_build_series_bible.py", force=True)
            run_step("17_build_series_bible.py")
            completed_steps += 1
            reporter.update(completed_steps, current_label="Update series bible", extra_label="Completed: 17_build_series_bible.py")
            env = os.environ.copy()
            env["SERIES_RENDER_EPISODE"] = episode_id
            reporter.update(completed_steps, current_label=f"{episode_id} render", extra_label="Running now: 18_render_episode.py", force=True)
            run_step("18_render_episode.py", env=env)
            completed_steps += 1
            generated.append(episode_id)
            reporter.update(completed_steps, current_label=episode_id, extra_label=f"Episode render complete: {episode_id}")
            ok(f"{index + 1}/{count}: {episode_id} generated and rendered.")
        reporter.finish(current_label="Preview Episodes", extra_label=f"Total episodes: {len(generated)}")
        mark_step_completed(
            "19_generate_preview_episodes",
            autosave_target,
            {"requested_count": count, "generated_episodes": generated, "generated_count": len(generated)},
        )
        ok(f"Done. New visible episodes: {', '.join(generated)}")
    except Exception as exc:
        mark_step_failed(
            "19_generate_preview_episodes",
            str(exc),
            autosave_target,
            {"requested_count": count, "generated_episodes": generated, "generated_count": len(generated)},
        )
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise

