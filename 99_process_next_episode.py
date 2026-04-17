#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from datetime import datetime
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

AUTOSAVE_VERSION = 1
AUTOSAVE_KEEP_COUNT = 2
SETUP_STEP = "01_setup_project.py"
EPISODE_STEPS = [
    "02_import_episode.py",
    "03_split_scenes.py",
    "04_diarize_and_transcribe.py",
    "05_link_faces_and_speakers.py",
    "06_build_dataset.py",
]
GLOBAL_STEPS = [
    "07_train_series_model.py",
    "09_prepare_foundation_training.py",
    "10_train_foundation_models.py",
    "11_generate_episode_from_trained_model.py",
    "12_build_series_bible.py",
    "13_render_episode.py",
]


def run_step(script_name: str, title: str, extra_args: list[str] | None = None) -> None:
    headline(title)
    command = [str(runtime_python()), str(SCRIPT_DIR / script_name), *(extra_args or [])]
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def cleanup_processed_inbox_episode(path: Path) -> bool:
    if not path.exists():
        return False
    path.unlink()
    return True


def autosave_dir(cfg: dict) -> Path:
    return resolve_project_path("runtime/autosaves/99_process_next_episode")


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def autosave_filename() -> str:
    return f"autosave_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.json"


def autosave_files(cfg: dict) -> list[Path]:
    root = autosave_dir(cfg)
    if not root.exists():
        return []
    return sorted(root.glob("autosave_*.json"))


def default_state() -> dict:
    return {
        "version": AUTOSAVE_VERSION,
        "status": "running",
        "updated_at": utc_timestamp(),
        "setup_completed": False,
        "processed_count": 0,
        "current_phase": None,
        "current_episode_file": None,
        "current_episode_name": None,
        "episode_steps_completed": {},
        "processed_episodes": [],
        "global_steps_completed": [],
    }


def load_latest_autosave(cfg: dict) -> dict | None:
    for candidate in reversed(autosave_files(cfg)):
        try:
            state = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(state, dict):
            continue
        if int(state.get("version", 0) or 0) != AUTOSAVE_VERSION:
            continue
        if str(state.get("status", "")).strip().lower() == "completed":
            return None
        return state
    return None


def prune_autosaves(cfg: dict) -> None:
    snapshots = autosave_files(cfg)
    while len(snapshots) > AUTOSAVE_KEEP_COUNT:
        oldest = snapshots.pop(0)
        try:
            oldest.unlink()
        except FileNotFoundError:
            pass


def save_autosave(cfg: dict, state: dict, reason: str) -> Path:
    root = autosave_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_timestamp()
    state["autosave_reason"] = reason
    target = root / autosave_filename()
    target.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    prune_autosaves(cfg)
    return target


def episode_step_args(script_name: str, episode_file: str, episode_name: str) -> list[str]:
    if script_name == "02_import_episode.py":
        return []
    if script_name == "03_split_scenes.py":
        return ["--episode-file", episode_file]
    if script_name in {"04_diarize_and_transcribe.py", "05_link_faces_and_speakers.py", "06_build_dataset.py"}:
        return ["--episode", episode_name]
    return []


def episode_step_title(script_name: str, episode_file: str, episode_name: str) -> str:
    titles = {
        "02_import_episode.py": f"Import: {episode_file}",
        "03_split_scenes.py": f"Szenenerkennung: {episode_name}",
        "04_diarize_and_transcribe.py": f"Audiosegmentierung und Transkription: {episode_name}",
        "05_link_faces_and_speakers.py": f"Gesichter und Stimmen verknüpfen: {episode_name}",
        "06_build_dataset.py": f"Trainingsdatensatz: {episode_name}",
    }
    return titles.get(script_name, episode_name)


def completed_episode_steps(state: dict, episode_name: str) -> list[str]:
    steps = state.setdefault("episode_steps_completed", {}).setdefault(episode_name, [])
    return [str(step) for step in steps]


def mark_episode_step_completed(state: dict, episode_name: str, script_name: str) -> None:
    steps = completed_episode_steps(state, episode_name)
    if script_name not in steps:
        steps.append(script_name)
    state.setdefault("episode_steps_completed", {})[episode_name] = steps


def mark_global_step_completed(state: dict, script_name: str) -> None:
    steps = list(state.get("global_steps_completed", []) or [])
    if script_name not in steps:
        steps.append(script_name)
    state["global_steps_completed"] = steps


def mark_episode_completed(state: dict, episode_name: str) -> None:
    processed = list(state.get("processed_episodes", []) or [])
    if episode_name not in processed:
        processed.append(episode_name)
        state["processed_count"] = int(state.get("processed_count", 0) or 0) + 1
    state["processed_episodes"] = processed
    state["current_phase"] = None
    state["current_episode_file"] = None
    state["current_episode_name"] = None


def global_steps_to_run(cfg: dict) -> list[str]:
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    needs_foundation_training = bool(foundation_cfg.get("required_before_generate", True)) or bool(
        foundation_cfg.get("required_before_render", True)
    )
    steps = ["07_train_series_model.py"]
    if bool(foundation_cfg.get("prepare_after_batch", False)) or needs_foundation_training:
        steps.append("09_prepare_foundation_training.py")
        if bool(foundation_cfg.get("auto_train_after_prepare", False)) or needs_foundation_training:
            steps.append("10_train_foundation_models.py")
    steps.extend(
        [
            "11_generate_episode_from_trained_model.py",
            "12_build_series_bible.py",
            "13_render_episode.py",
        ]
    )
    return steps


def global_step_title(script_name: str) -> str:
    titles = {
        "07_train_series_model.py": "Serienmodell trainieren",
        "09_prepare_foundation_training.py": "Foundation-Training vorbereiten",
        "10_train_foundation_models.py": "Foundation-Modelle trainieren",
        "11_generate_episode_from_trained_model.py": "Neue Folge aus trainiertem Modell erzeugen",
        "12_build_series_bible.py": "Serienbibel aktualisieren",
        "13_render_episode.py": "Storyboard-Video rendern",
    }
    return titles[script_name]


def main() -> None:
    rerun_in_runtime()
    headline("Komplette Serien-Pipeline ausführen")
    cfg = load_config()
    inbox_dir = resolve_project_path(cfg["paths"]["inbox_episodes"])
    state = load_latest_autosave(cfg) or default_state()

    if load_latest_autosave(cfg):
        info("Vorhandenen Autosave gefunden. Pipeline setzt am letzten erfolgreich abgeschlossenen Schritt fort.")

    if not bool(state.get("setup_completed", False)):
        run_step(SETUP_STEP, "Projektstruktur")
        state["setup_completed"] = True
        save_autosave(cfg, state, "setup_completed")

    while True:
        current_episode_file = str(state.get("current_episode_file") or "").strip()
        current_episode_name = str(state.get("current_episode_name") or "").strip()
        if current_episode_file and current_episode_name:
            next_video_name = current_episode_file
            episode_name = current_episode_name
        else:
            next_video = next_unprocessed_video(inbox_dir)
            if next_video is None:
                break
            next_video_name = next_video.name
            episode_name = next_video.stem
            state["current_phase"] = "episode"
            state["current_episode_file"] = next_video_name
            state["current_episode_name"] = episode_name
            state.setdefault("episode_steps_completed", {}).setdefault(episode_name, [])
            save_autosave(cfg, state, f"episode_selected:{episode_name}")

        info(f"Starte Batch-Folge: {next_video_name}")
        finished_steps = set(completed_episode_steps(state, episode_name))
        for script_name in EPISODE_STEPS:
            if script_name in finished_steps:
                continue
            run_step(
                script_name,
                episode_step_title(script_name, next_video_name, episode_name),
                episode_step_args(script_name, next_video_name, episode_name),
            )
            mark_episode_step_completed(state, episode_name, script_name)
            save_autosave(cfg, state, f"{episode_name}:{script_name}")

        inbox_file = inbox_dir / next_video_name
        if cleanup_processed_inbox_episode(inbox_file):
            info(f"Inbox-Datei entfernt: {next_video_name}")
        mark_episode_completed(state, episode_name)
        save_autosave(cfg, state, f"episode_completed:{episode_name}")

    if int(state.get("processed_count", 0) or 0) == 0:
        info("Keine neuen Folgen im Inbox-Ordner gefunden.")
        return

    state["current_phase"] = "global"
    save_autosave(cfg, state, "global_phase_started")
    completed_global_steps = set(state.get("global_steps_completed", []) or [])
    for script_name in global_steps_to_run(cfg):
        if script_name in completed_global_steps:
            continue
        run_step(script_name, global_step_title(script_name))
        mark_global_step_completed(state, script_name)
        save_autosave(cfg, state, f"global:{script_name}")

    state["status"] = "completed"
    state["current_phase"] = None
    save_autosave(cfg, state, "pipeline_completed")
    ok(f"Die Pipeline ist komplett durchgelaufen. Neue Quellfolgen verarbeitet: {state['processed_count']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
