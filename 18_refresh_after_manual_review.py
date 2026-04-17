#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess

from pipeline_common import (
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
        run_step("07_build_dataset.py", "Datensaetze mit aktuellen Figurennamen neu aufbauen", ["--force"])
        run_step("08_train_series_model.py", "Serienmodell mit aktuellen Namen neu trainieren")
        prepare_args = ["--force"]
        if args.skip_downloads:
            prepare_args.append("--skip-downloads")
        run_step("09_prepare_foundation_training.py", "Foundation-Training mit aktuellem Figurenstand vorbereiten", prepare_args)
        run_step("10_train_foundation_models.py", "Foundation-Packs mit aktuellem Figurenstand neu trainieren", ["--force"])
        run_step("11_train_adapter_models.py", "Lokale Adapter-Profile mit aktuellem Figurenstand neu trainieren", ["--force"])
        run_step("12_train_fine_tune_models.py", "Lokale Fine-Tune-Profile mit aktuellem Figurenstand neu trainieren", ["--force"])
        run_step("13_run_backend_finetunes.py", "Konkrete Backend-Fine-Tune-Laeufe mit aktuellem Figurenstand erzeugen", ["--force"])

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
            run_step("14_generate_episode_from_trained_model.py", "Neue Folge aus aktualisiertem Modell erzeugen")
            run_step("15_build_series_bible.py", "Serienbibel mit aktuellem Stand aktualisieren")
            run_step("16_render_episode.py", "Aktualisierte Folge rendern")
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
