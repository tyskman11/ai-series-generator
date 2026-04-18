#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess

from pipeline_common import (
    LiveProgressReporter,
    SCRIPT_DIR,
    error,
    headline,
    info,
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
    parser = argparse.ArgumentParser(
        description="Rebuild the pipeline after manual character review"
    )
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Skip model downloads in 09 and use only existing downloads/updates.",
    )
    parser.add_argument(
        "--stop-after-training",
        action="store_true",
        help="Stop after the complete training block through 13 and skip episode generation and rendering.",
    )
    parser.add_argument(
        "--allow-open-review",
        action="store_true",
        help="Allow the rebuild even if step 06 still has open review cases.",
    )
    return parser.parse_args()


def run_step(script_name: str, title: str, extra_args: list[str] | None = None) -> None:
    headline(title)
    command = [str(runtime_python()), str(SCRIPT_DIR / script_name), *(extra_args or [])]
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Rebuild After Manual Character Review")
    autosave_target = "global"
    mark_step_started(
        "19_refresh_after_manual_review",
        autosave_target,
        {
            "allow_open_review": bool(args.allow_open_review),
            "skip_downloads": bool(args.skip_downloads),
            "stop_after_training": bool(args.stop_after_training),
        },
    )
    try:
        cfg = load_config()
        if not args.allow_open_review:
            review_count = open_review_item_count(cfg)
            if review_count > 0:
                raise RuntimeError(
                    f"Es gibt noch {review_count} offene Review-Faelle. "
                    "Run 06_review_unknowns.py first or intentionally start with --allow-open-review."
                )
        planned_steps: list[tuple[str, str, list[str]]] = [
            ("07_build_dataset.py", "Datensaetze mit aktuellen Figurennamen neu aufbauen", ["--force"]),
            ("08_train_series_model.py", "Series Model mit aktuellen Namen neu trainieren", []),
        ]
        prepare_args = ["--force"]
        if args.skip_downloads:
            prepare_args.append("--skip-downloads")
        planned_steps.extend(
            [
                ("09_prepare_foundation_training.py", "Foundation Training mit aktuellem Figurenstand vorbereiten", prepare_args),
                ("10_train_foundation_models.py", "Foundation-Packs mit aktuellem Figurenstand neu trainieren", ["--force"]),
                ("11_train_adapter_models.py", "Lokale Adapter-Profile mit aktuellem Figurenstand neu trainieren", ["--force"]),
                ("12_train_fine_tune_models.py", "Lokale Fine-Tune-Profile mit aktuellem Figurenstand neu trainieren", ["--force"]),
                ("13_run_backend_finetunes.py", "Konkrete Backend-Fine-Tune-Laeufe mit aktuellem Figurenstand erzeugen", ["--force"]),
            ]
        )
        if not args.stop_after_training:
            planned_steps.extend(
                [
                    ("14_generate_episode_from_trained_model.py", "Neue Folge aus aktualisiertem Modell erzeugen", []),
                    ("15_generate_storyboard_assets.py", "Storyboard-Assets fuer die neue Folge erzeugen", []),
                    ("16_build_series_bible.py", "Series Bible mit aktuellem Stand aktualisieren", []),
                    ("17_render_episode.py", "Aktualisierte Folge render", []),
                ]
            )
        reporter = LiveProgressReporter(
            script_name="19_refresh_after_manual_review.py",
            total=len(planned_steps),
            phase_label="Rebuild After Review",
            parent_label="global",
        )
        completed_count = 0
        for script_name, title, extra_args in planned_steps:
            reporter.update(completed_count, current_label=title, extra_label=f"Running now: {script_name}", force=True)
            run_step(script_name, title, extra_args)
            completed_count += 1
            reporter.update(completed_count, current_label=title, extra_label=f"Completed: {script_name}")
        reporter.finish(current_label="Rebuild", extra_label=f"Completed steps: {completed_count}")

        completed_steps = [
            "07_build_dataset.py --force",
            "08_train_series_model.py",
            "09_prepare_foundation_training.py",
            "10_train_foundation_models.py --force",
            "11_train_adapter_models.py --force",
            "12_train_fine_tune_models.py --force",
            "13_run_backend_finetunes.py --force",
        ]

        if not args.stop_after_training:
            completed_steps.extend(
                [
                    "14_generate_episode_from_trained_model.py",
                    "15_generate_storyboard_assets.py",
                    "16_build_series_bible.py",
                    "17_render_episode.py",
                ]
            )

        mark_step_completed(
            "19_refresh_after_manual_review",
            autosave_target,
            {
                "allow_open_review": bool(args.allow_open_review),
                "skip_downloads": bool(args.skip_downloads),
                "stop_after_training": bool(args.stop_after_training),
                "completed_steps": completed_steps,
            },
        )
        ok("Rebuild After Manual Character Review abgeschlossen.")
    except Exception as exc:
        mark_step_failed(
            "19_refresh_after_manual_review",
            str(exc),
            autosave_target,
            {
                "allow_open_review": bool(args.allow_open_review),
                "skip_downloads": bool(args.skip_downloads),
                "stop_after_training": bool(args.stop_after_training),
            },
        )
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise

