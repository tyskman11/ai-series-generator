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
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    rerun_in_runtime,
    runtime_python,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline nach manueller Figuren-Review mit echtem Rebuild neu aufbauen"
    )
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Ueberspringt in 09 die Modell-Downloads und nutzt nur vorhandene Downloads/Updates.",
    )
    parser.add_argument(
        "--stop-after-training",
        action="store_true",
        help="Beendet nach dem kompletten Trainingsblock bis 13 und erzeugt noch keine neue Folge und keinen Render.",
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
    headline("Rebuild nach manueller Figuren-Review")
    autosave_target = "global"
    mark_step_started(
        "18_refresh_after_manual_review",
        autosave_target,
        {
            "skip_downloads": bool(args.skip_downloads),
            "stop_after_training": bool(args.stop_after_training),
        },
    )
    try:
        planned_steps: list[tuple[str, str, list[str]]] = [
            ("07_build_dataset.py", "Datensaetze mit aktuellen Figurennamen neu aufbauen", ["--force"]),
            ("08_train_series_model.py", "Serienmodell mit aktuellen Namen neu trainieren", []),
        ]
        prepare_args = ["--force"]
        if args.skip_downloads:
            prepare_args.append("--skip-downloads")
        planned_steps.extend(
            [
                ("09_prepare_foundation_training.py", "Foundation-Training mit aktuellem Figurenstand vorbereiten", prepare_args),
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
                    ("15_build_series_bible.py", "Serienbibel mit aktuellem Stand aktualisieren", []),
                    ("16_render_episode.py", "Aktualisierte Folge rendern", []),
                ]
            )
        reporter = LiveProgressReporter(
            script_name="18_refresh_after_manual_review.py",
            total=len(planned_steps),
            phase_label="Rebuild nach Review",
            parent_label="global",
        )
        completed_count = 0
        for script_name, title, extra_args in planned_steps:
            reporter.update(completed_count, current_label=title, extra_label=f"Laeuft jetzt: {script_name}", force=True)
            run_step(script_name, title, extra_args)
            completed_count += 1
            reporter.update(completed_count, current_label=title, extra_label=f"Abgeschlossen: {script_name}")
        reporter.finish(current_label="Rebuild", extra_label=f"Abgeschlossene Schritte: {completed_count}")

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
                    "15_build_series_bible.py",
                    "16_render_episode.py",
                ]
            )

        mark_step_completed(
            "18_refresh_after_manual_review",
            autosave_target,
            {
                "skip_downloads": bool(args.skip_downloads),
                "stop_after_training": bool(args.stop_after_training),
                "completed_steps": completed_steps,
            },
        )
        ok("Rebuild nach manueller Figuren-Review abgeschlossen.")
    except Exception as exc:
        mark_step_failed(
            "18_refresh_after_manual_review",
            str(exc),
            autosave_target,
            {
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
