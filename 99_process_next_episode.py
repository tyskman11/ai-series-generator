#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

from pipeline_common import (
    SCRIPT_DIR,
    error,
    headline,
    info,
    load_config,
    next_unprocessed_video,
    ok,
    rerun_in_runtime,
    resolve_project_path,
    runtime_python,
)


def run_step(script_name: str, title: str, extra_args: list[str] | None = None) -> None:
    headline(title)
    command = [str(runtime_python()), str(SCRIPT_DIR / script_name), *(extra_args or [])]
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    rerun_in_runtime()
    headline("Komplette Serien-Pipeline ausführen")
    cfg = load_config()
    inbox_dir = resolve_project_path(cfg["paths"]["inbox_episodes"])

    run_step("01_setup_project.py", "Projektstruktur")

    processed_count = 0
    while True:
        next_video = next_unprocessed_video(inbox_dir)
        if next_video is None:
            break

        episode_name = next_video.stem
        info(f"Starte Batch-Folge: {next_video.name}")
        run_step("02_import_episode.py", f"Import: {next_video.name}")
        run_step("03_split_scenes.py", f"Szenenerkennung: {episode_name}", ["--episode-file", next_video.name])
        run_step(
            "04_diarize_and_transcribe.py",
            f"Audiosegmentierung und Transkription: {episode_name}",
            ["--episode", episode_name],
        )
        run_step(
            "05_link_faces_and_speakers.py",
            f"Gesichter und Stimmen verknüpfen: {episode_name}",
            ["--episode", episode_name],
        )
        run_step("06_build_dataset.py", f"Trainingsdatensatz: {episode_name}", ["--episode", episode_name])
        processed_count += 1

    if processed_count == 0:
        info("Keine neuen Folgen im Inbox-Ordner gefunden.")
        return

    run_step("07_train_series_model.py", "Serienmodell trainieren")

    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    if bool(foundation_cfg.get("prepare_after_batch", False)):
        run_step("13_prepare_foundation_training.py", "Foundation-Training vorbereiten")
        if bool(foundation_cfg.get("auto_train_after_prepare", False)):
            run_step("14_train_foundation_models.py", "Foundation-Modelle trainieren")

    run_step("09_generate_episode_from_trained_model.py", "Neue Folge aus trainiertem Modell erzeugen")
    run_step("10_build_series_bible.py", "Serienbibel aktualisieren")
    run_step("11_render_episode.py", "Storyboard-Video rendern")
    ok(f"Die Pipeline ist komplett durchgelaufen. Neue Quellfolgen verarbeitet: {processed_count}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
