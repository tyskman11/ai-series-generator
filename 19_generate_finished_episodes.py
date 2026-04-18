#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from pipeline_common import (
    open_face_review_item_count,
    LiveProgressReporter,
    SCRIPT_DIR,
    error,
    generated_episode_artifacts,
    headline,
    latest_matching_file,
    load_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    rerun_in_runtime,
    runtime_python,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multiple finished episodes in one run.")
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of new episodes. Use 0 for endless generation. Default: 0 (endless)",
    )
    parser.add_argument(
        "--endless",
        action="store_true",
        help="Ignore --count and keep generating episodes until the run is stopped manually.",
    )
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Skip model downloads in 09 and only use existing downloads/updates.",
    )
    return parser.parse_args()


def story_dir() -> Path:
    return SCRIPT_DIR / "ai_series_project" / "generation" / "story_prompts"


def latest_episode_id() -> str | None:
    latest = latest_matching_file(story_dir(), "*.md")
    return latest.stem if latest else None


def run_step(script_name: str, env: dict[str, str] | None = None, extra_args: list[str] | None = None) -> None:
    result = subprocess.run(
        [str(runtime_python()), str(SCRIPT_DIR / script_name), *(extra_args or [])],
        env=env,
        cwd=str(SCRIPT_DIR),
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def training_stage_flags(cfg: dict) -> tuple[bool, bool, bool, bool, bool]:
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    adapter_cfg = cfg.get("adapter_training", {}) if isinstance(cfg.get("adapter_training"), dict) else {}
    fine_tune_cfg = cfg.get("fine_tune_training", {}) if isinstance(cfg.get("fine_tune_training"), dict) else {}
    backend_cfg = cfg.get("backend_fine_tune", {}) if isinstance(cfg.get("backend_fine_tune"), dict) else {}
    needs_foundation_training = bool(foundation_cfg.get("required_before_generate", True)) or bool(
        foundation_cfg.get("required_before_render", True)
    )
    should_prepare_foundation = bool(foundation_cfg.get("prepare_after_batch", False)) or needs_foundation_training
    should_train_foundation = bool(foundation_cfg.get("auto_train_after_prepare", False)) or needs_foundation_training
    should_train_adapter = bool(adapter_cfg.get("auto_train_after_foundation", True))
    should_train_fine_tune = bool(fine_tune_cfg.get("auto_train_after_adapter", True))
    should_run_backend = bool(backend_cfg.get("auto_run_after_fine_tune", True))
    return (
        should_prepare_foundation,
        should_train_foundation,
        should_train_adapter,
        should_train_fine_tune,
        should_run_backend,
    )


def training_plan_rows(cfg: dict, skip_downloads: bool = False) -> list[tuple[str, str, list[str]]]:
    (
        should_prepare_foundation,
        should_train_foundation,
        should_train_adapter,
        should_train_fine_tune,
        should_run_backend,
    ) = training_stage_flags(cfg)
    rows: list[tuple[str, str, list[str]]] = [
        ("07_build_dataset.py", "Build Datasets", []),
        ("08_train_series_model.py", "Train Series Model", []),
    ]
    if should_prepare_foundation:
        prepare_args: list[str] = ["--skip-downloads"] if skip_downloads else []
        rows.append(("09_prepare_foundation_training.py", "Prepare Foundation Data", prepare_args))
        if should_train_foundation:
            rows.append(("10_train_foundation_models.py", "Train Foundation Packs", []))
            if should_train_adapter:
                rows.append(("11_train_adapter_models.py", "Train Adapter Profiles", []))
                if should_train_fine_tune:
                    rows.append(("12_train_fine_tune_models.py", "Train Fine-Tune Profiles", []))
                    if should_run_backend:
                        rows.append(("13_run_backend_finetunes.py", "Prepare Backend Fine-Tunes", []))
    return rows


def planned_preview_steps(cfg: dict, count: int) -> list[str]:
    return planned_preview_steps_for_mode(cfg, count, endless=False)


def planned_preview_steps_for_mode(cfg: dict, count: int, *, endless: bool) -> list[str]:
    steps = [row[0] for row in training_plan_rows(cfg, skip_downloads=False)]
    generation_cycle = [
        "14_generate_episode_from_trained_model.py",
        "15_generate_storyboard_assets.py",
        "16_run_storyboard_backend.py",
        "17_render_episode.py",
    ]
    if endless:
        steps.extend(generation_cycle)
        steps.append("18_build_series_bible.py")
        return steps
    steps.extend(generation_cycle * max(1, int(count)))
    steps.append("18_build_series_bible.py")
    return steps


def preview_endless_mode(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "endless", False)) or int(getattr(args, "count", 0) or 0) <= 0


def preview_parent_label(count: int, endless: bool) -> str:
    return "Endless" if endless else f"Count: {count}"


def preview_scope_total(count: int, endless: bool, generated_count: int) -> int:
    if endless:
        return max(1, int(generated_count) + 1)
    return max(1, int(count or 1))


def preview_reporter_total(training_rows: list[tuple[str, str, list[str]]], planned_steps: list[str], endless: bool) -> int:
    if endless:
        return max(1, len(training_rows) + 5)
    return max(1, len(planned_steps))


def preview_reporter_current(reporter: LiveProgressReporter, completed_steps: int) -> int:
    reporter_total = getattr(reporter, "total", None)
    if isinstance(reporter_total, (int, float)):
        return min(int(completed_steps), max(0, int(reporter_total) - 1))
    return max(0, int(completed_steps))


def completed_step_labels(planned_steps: list[str], completed_count: int) -> list[str]:
    limit = max(0, min(int(completed_count or 0), len(planned_steps)))
    return list(planned_steps[:limit])


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    endless = preview_endless_mode(args)
    count = 0 if endless else max(1, int(args.count))
    cfg = load_config()
    training_rows = training_plan_rows(cfg, skip_downloads=bool(args.skip_downloads))

    headline("Generate Multiple Finished Episodes")
    generated: list[str] = []
    generated_outputs: list[dict[str, object]] = []
    autosave_target = "endless" if endless else f"count_{count}"
    mark_step_started(
        "19_generate_finished_episodes",
        autosave_target,
        {"requested_count": count, "endless": endless, "skip_downloads": bool(args.skip_downloads)},
    )
    try:
        review_count = open_face_review_item_count(cfg)
        if review_count > 0:
            raise RuntimeError(
                f"There are still {review_count} open face review cases. "
                "Run 06_review_unknowns.py first before training, generation, or render can start."
            )
        planned_steps = planned_preview_steps_for_mode(cfg, count, endless=endless)
        reporter = LiveProgressReporter(
            script_name="19_generate_finished_episodes.py",
            total=preview_reporter_total(training_rows, planned_steps, endless),
            phase_label="Generate Finished Episodes",
            parent_label=preview_parent_label(count, endless),
        )
        completed_steps = 0

        for script_name, current_label, extra_args in training_rows:
            reporter.update(completed_steps, current_label=current_label, extra_label=f"Running now: {script_name}", force=True)
            run_step(script_name, extra_args=extra_args)
            completed_steps += 1
            reporter.update(completed_steps, current_label=current_label, extra_label=f"Completed: {script_name}")
        index = 0
        while endless or index < count:
            before = latest_episode_id()
            reporter.update(
                preview_reporter_current(reporter, completed_steps),
                current_label=f"Generate episode {index + 1}",
                extra_label="Running now: 14_generate_episode_from_trained_model.py",
                force=True,
                scope_current=len(generated),
                scope_total=preview_scope_total(count, endless, len(generated)),
                scope_label="Episodes",
            )
            run_step("14_generate_episode_from_trained_model.py")
            completed_steps += 1
            episode_id = latest_episode_id()
            if not episode_id or episode_id == before:
                raise RuntimeError("Could not determine the new episode.")
            reporter.update(
                preview_reporter_current(reporter, completed_steps),
                current_label=episode_id,
                extra_label=f"Episode generated: {episode_id}",
                scope_current=len(generated) + 1,
                scope_total=preview_scope_total(count, endless, len(generated)),
                scope_label="Episodes",
            )
            env = os.environ.copy()
            env["SERIES_STORYBOARD_EPISODE"] = episode_id
            reporter.update(
                preview_reporter_current(reporter, completed_steps),
                current_label=f"{episode_id} storyboard assets",
                extra_label="Running now: 15_generate_storyboard_assets.py",
                force=True,
                scope_current=len(generated) + 1,
                scope_total=preview_scope_total(count, endless, len(generated)),
                scope_label="Episodes",
            )
            run_step("15_generate_storyboard_assets.py", env=env)
            completed_steps += 1
            reporter.update(
                preview_reporter_current(reporter, completed_steps),
                current_label=episode_id,
                extra_label=f"Storyboard assets ready: {episode_id}",
                scope_current=len(generated) + 1,
                scope_total=preview_scope_total(count, endless, len(generated)),
                scope_label="Episodes",
            )
            reporter.update(
                preview_reporter_current(reporter, completed_steps),
                current_label=f"{episode_id} backend frames",
                extra_label="Running now: 16_run_storyboard_backend.py",
                force=True,
                scope_current=len(generated) + 1,
                scope_total=preview_scope_total(count, endless, len(generated)),
                scope_label="Episodes",
            )
            run_step("16_run_storyboard_backend.py", env=env)
            completed_steps += 1
            reporter.update(
                preview_reporter_current(reporter, completed_steps),
                current_label=episode_id,
                extra_label=f"Storyboard backend frames ready: {episode_id}",
                scope_current=len(generated) + 1,
                scope_total=preview_scope_total(count, endless, len(generated)),
                scope_label="Episodes",
            )
            env = os.environ.copy()
            env["SERIES_RENDER_EPISODE"] = episode_id
            reporter.update(
                preview_reporter_current(reporter, completed_steps),
                current_label=f"{episode_id} render",
                extra_label="Running now: 17_render_episode.py",
                force=True,
                scope_current=len(generated) + 1,
                scope_total=preview_scope_total(count, endless, len(generated)),
                scope_label="Episodes",
            )
            run_step("17_render_episode.py", env=env)
            completed_steps += 1
            episode_outputs = generated_episode_artifacts(cfg, episode_id)
            if episode_outputs:
                generated_outputs.append(episode_outputs)
            reporter.update(
                preview_reporter_current(reporter, completed_steps),
                current_label=episode_id,
                extra_label=f"Episode render complete: {episode_id}",
                scope_current=len(generated) + 1,
                scope_total=preview_scope_total(count, endless, len(generated)),
                scope_label="Episodes",
            )
            generated.append(episode_id)
            if endless:
                reporter.update(
                    preview_reporter_current(reporter, completed_steps),
                    current_label="Update series bible",
                    extra_label="Running now: 18_build_series_bible.py",
                    force=True,
                    scope_current=len(generated),
                    scope_total=preview_scope_total(count, endless, len(generated)),
                    scope_label="Episodes",
                )
                run_step("18_build_series_bible.py")
                completed_steps += 1
                reporter.update(
                    preview_reporter_current(reporter, completed_steps),
                    current_label="Series bible updated",
                    extra_label=f"Endless mode active. Episodes generated so far: {len(generated)}",
                    scope_current=len(generated),
                    scope_total=preview_scope_total(count, endless, len(generated)),
                    scope_label="Episodes",
                )
                ok(f"{len(generated)} episodes generated so far. Latest: {episode_id}")
            else:
                ok(f"{index + 1}/{count}: {episode_id} generated and rendered.")
            index += 1
        if not endless:
            reporter.update(completed_steps, current_label="Update series bible", extra_label="Running now: 18_build_series_bible.py", force=True)
            run_step("18_build_series_bible.py")
            completed_steps += 1
            reporter.update(completed_steps, current_label="Update series bible", extra_label="Completed: 18_build_series_bible.py")
            reporter.finish(current_label="Finished Episodes", extra_label=f"Total episodes: {len(generated)}")
        mark_step_completed(
            "19_generate_finished_episodes",
            autosave_target,
            {
                "requested_count": count,
                "endless": endless,
                "skip_downloads": bool(args.skip_downloads),
                "planned_steps": planned_steps,
                "completed_steps": completed_step_labels(planned_steps, completed_steps),
                "generated_episodes": generated,
                "generated_episode_outputs": generated_outputs,
                "latest_generated_episode": generated_outputs[-1] if generated_outputs else {},
                "generated_count": len(generated),
                "bible_updated_once_after_batch": not endless,
                "bible_updated_after_each_episode": endless,
            },
        )
        ok(f"Done. New visible episodes: {', '.join(generated)}")
    except KeyboardInterrupt:
        if generated and endless:
            info_message = f"Endless generation interrupted after {len(generated)} episodes. Updating the series bible one last time."
            headline(info_message)
            run_step("18_build_series_bible.py")
        mark_step_failed(
            "19_generate_finished_episodes",
            "Interrupted by user",
            autosave_target,
            {
                "requested_count": count,
                "endless": endless,
                "skip_downloads": bool(args.skip_downloads),
                "planned_steps": planned_steps if "planned_steps" in locals() else [],
                "completed_steps": completed_step_labels(planned_steps, completed_steps)
                if "planned_steps" in locals()
                else [],
                "generated_episodes": generated,
                "generated_episode_outputs": generated_outputs,
                "latest_generated_episode": generated_outputs[-1] if generated_outputs else {},
                "generated_count": len(generated),
            },
        )
        raise
    except Exception as exc:
        mark_step_failed(
            "19_generate_finished_episodes",
            str(exc),
            autosave_target,
            {
                "requested_count": count,
                "endless": endless,
                "skip_downloads": bool(args.skip_downloads),
                "planned_steps": planned_steps if "planned_steps" in locals() else [],
                "completed_steps": completed_step_labels(planned_steps, completed_steps)
                if "planned_steps" in locals()
                else [],
                "generated_episodes": generated,
                "generated_episode_outputs": generated_outputs,
                "latest_generated_episode": generated_outputs[-1] if generated_outputs else {},
                "generated_count": len(generated),
            },
        )
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise

