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
    parser = argparse.ArgumentParser(description="Erzeugt mehrere neue sichtbare Preview-Episoden am Stück.")
    parser.add_argument("--count", type=int, default=2, help="Anzahl neuer Episoden. Standard: 2")
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

    headline("Mehrere sichtbare Preview-Episoden erzeugen")
    generated: list[str] = []
    autosave_target = f"count_{count}"
    mark_step_started("17_generate_preview_episodes", autosave_target, {"requested_count": count})
    try:
        review_count = open_review_item_count(cfg)
        if review_count > 0:
            raise RuntimeError(
                f"Es gibt noch {review_count} offene Review-Faelle. "
                "Fuehre zuerst 06_review_unknowns.py aus, bevor Training, Generierung oder Render starten."
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
        planned_steps.extend(["14_generate_episode_from_trained_model.py", "16_render_episode.py"] * count)
        planned_steps.append("15_build_series_bible.py")
        reporter = LiveProgressReporter(
            script_name="17_generate_preview_episodes.py",
            total=len(planned_steps),
            phase_label="Preview-Episoden erzeugen",
            parent_label=f"Anzahl: {count}",
        )
        completed_steps = 0

        reporter.update(completed_steps, current_label="Datensaetze aufbauen", extra_label="Laeuft jetzt: 07_build_dataset.py", force=True)
        run_step("07_build_dataset.py")
        completed_steps += 1
        reporter.update(completed_steps, current_label="Datensaetze aufbauen", extra_label="Abgeschlossen: 07_build_dataset.py")

        reporter.update(completed_steps, current_label="Serienmodell trainieren", extra_label="Laeuft jetzt: 08_train_series_model.py", force=True)
        run_step("08_train_series_model.py")
        completed_steps += 1
        reporter.update(completed_steps, current_label="Serienmodell trainieren", extra_label="Abgeschlossen: 08_train_series_model.py")

        if bool(foundation_cfg.get("required_before_generate", True)) or bool(foundation_cfg.get("required_before_render", True)):
            reporter.update(completed_steps, current_label="Foundation-Daten vorbereiten", extra_label="Laeuft jetzt: 09_prepare_foundation_training.py", force=True)
            run_step("09_prepare_foundation_training.py")
            completed_steps += 1
            reporter.update(completed_steps, current_label="Foundation-Daten vorbereiten", extra_label="Abgeschlossen: 09_prepare_foundation_training.py")

            reporter.update(completed_steps, current_label="Foundation-Packs trainieren", extra_label="Laeuft jetzt: 10_train_foundation_models.py", force=True)
            run_step("10_train_foundation_models.py")
            completed_steps += 1
            reporter.update(completed_steps, current_label="Foundation-Packs trainieren", extra_label="Abgeschlossen: 10_train_foundation_models.py")

            if bool(adapter_cfg.get("auto_train_after_foundation", True)):
                reporter.update(completed_steps, current_label="Adapter-Profile trainieren", extra_label="Laeuft jetzt: 11_train_adapter_models.py", force=True)
                run_step("11_train_adapter_models.py")
                completed_steps += 1
                reporter.update(completed_steps, current_label="Adapter-Profile trainieren", extra_label="Abgeschlossen: 11_train_adapter_models.py")
                if bool(fine_tune_cfg.get("auto_train_after_adapter", True)):
                    reporter.update(completed_steps, current_label="Fine-Tune-Profile trainieren", extra_label="Laeuft jetzt: 12_train_fine_tune_models.py", force=True)
                    run_step("12_train_fine_tune_models.py")
                    completed_steps += 1
                    reporter.update(completed_steps, current_label="Fine-Tune-Profile trainieren", extra_label="Abgeschlossen: 12_train_fine_tune_models.py")
                    if bool(backend_cfg.get("auto_run_after_fine_tune", True)):
                        reporter.update(completed_steps, current_label="Backend-Fine-Tunes vorbereiten", extra_label="Laeuft jetzt: 13_run_backend_finetunes.py", force=True)
                        run_step("13_run_backend_finetunes.py")
                        completed_steps += 1
                        reporter.update(completed_steps, current_label="Backend-Fine-Tunes vorbereiten", extra_label="Abgeschlossen: 13_run_backend_finetunes.py")
        for index in range(count):
            before = latest_episode_id()
            reporter.update(completed_steps, current_label=f"Episode {index + 1} generieren", extra_label="Laeuft jetzt: 14_generate_episode_from_trained_model.py", force=True)
            run_step("14_generate_episode_from_trained_model.py")
            completed_steps += 1
            episode_id = latest_episode_id()
            if not episode_id or episode_id == before:
                raise RuntimeError("Neue Episode konnte nicht ermittelt werden.")
            reporter.update(completed_steps, current_label=episode_id, extra_label=f"Episode erzeugt: {episode_id}")
            env = os.environ.copy()
            env["SERIES_RENDER_EPISODE"] = episode_id
            reporter.update(completed_steps, current_label=f"{episode_id} rendern", extra_label="Laeuft jetzt: 16_render_episode.py", force=True)
            run_step("16_render_episode.py", env=env)
            completed_steps += 1
            generated.append(episode_id)
            reporter.update(completed_steps, current_label=episode_id, extra_label=f"Episoden-Render fertig: {episode_id}")
            ok(f"{index + 1}/{count}: {episode_id} erzeugt und gerendert.")

        reporter.update(completed_steps, current_label="Serienbibel aktualisieren", extra_label="Laeuft jetzt: 15_build_series_bible.py", force=True)
        run_step("15_build_series_bible.py")
        completed_steps += 1
        reporter.update(completed_steps, current_label="Serienbibel aktualisieren", extra_label="Abgeschlossen: 15_build_series_bible.py")
        reporter.finish(current_label="Preview-Episoden", extra_label=f"Episoden gesamt: {len(generated)}")
        mark_step_completed(
            "17_generate_preview_episodes",
            autosave_target,
            {"requested_count": count, "generated_episodes": generated, "generated_count": len(generated)},
        )
        ok(f"Fertig. Neue sichtbare Episoden: {', '.join(generated)}")
    except Exception as exc:
        mark_step_failed(
            "17_generate_preview_episodes",
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
