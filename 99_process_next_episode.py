#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

from pipeline_common import SCRIPT_DIR, error, headline, ok, rerun_in_runtime, runtime_python


def run_step(script_name: str, title: str) -> None:
    headline(title)
    result = subprocess.run([str(runtime_python()), str(SCRIPT_DIR / script_name)])
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    rerun_in_runtime()
    headline("Komplette Serien-Pipeline ausführen")
    steps = [
        ("01_setup_project.py", "Projektstruktur"),
        ("02_import_episode.py", "Import"),
        ("03_split_scenes.py", "Szenenerkennung"),
        ("04_diarize_and_transcribe.py", "Audiosegmentierung und Transkription"),
        ("05_link_faces_and_speakers.py", "Gesichter und Stimmen verknüpfen"),
        ("06_build_dataset.py", "Trainingsdatensatz"),
        ("07_generate_episode.py", "Serienmodell trainieren und Folge erzeugen"),
        ("09_build_series_bible.py", "Serienbibel aktualisieren"),
        ("10_render_episode.py", "Storyboard-Video rendern"),
    ]
    for script_name, title in steps:
        run_step(script_name, title)
    ok("Die Pipeline ist komplett durchgelaufen.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
