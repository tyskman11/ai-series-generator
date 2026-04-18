#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from pipeline_common import (
    LiveProgressReporter,
    SCRIPT_DIR,
    error,
    headline,
    info,
    load_config,
    next_unprocessed_video,
    open_review_item_count,
    ok,
    read_json,
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
]
GLOBAL_STEPS = [
    "07_build_dataset.py",
    "08_train_series_model.py",
    "09_prepare_foundation_training.py",
    "10_train_foundation_models.py",
    "11_train_adapter_models.py",
    "12_train_fine_tune_models.py",
    "13_run_backend_finetunes.py",
    "14_generate_episode_from_trained_model.py",
    "15_generate_storyboard_assets.py",
    "16_run_storyboard_backend.py",
    "17_build_series_bible.py",
    "18_render_episode.py",
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


def status_json_path(cfg: dict) -> Path:
    return autosave_dir(cfg) / "current_status.json"


def status_markdown_path(cfg: dict) -> Path:
    return autosave_dir(cfg) / "current_status.md"


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
        "current_step": None,
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


def known_episode_names(state: dict) -> list[str]:
    names = set(state.get("processed_episodes", []) or [])
    names.update((state.get("episode_steps_completed", {}) or {}).keys())
    current_episode = str(state.get("current_episode_name") or "").strip()
    if current_episode:
        names.add(current_episode)
    return sorted(name for name in names if str(name).strip())


def build_status_snapshot(cfg: dict, state: dict, inbox_dir: Path | None = None) -> dict:
    pending_inbox_count = 0
    if inbox_dir is not None and inbox_dir.exists():
        pending_inbox_count = len([path for path in inbox_dir.iterdir() if path.is_file()])

    episode_rows = []
    processed = set(state.get("processed_episodes", []) or [])
    current_phase = str(state.get("current_phase") or "").strip().lower()
    current_episode = str(state.get("current_episode_name") or "").strip()
    current_step = str(state.get("current_step") or "").strip()
    for episode_name in known_episode_names(state):
        completed = completed_episode_steps(state, episode_name)
        remaining = [step for step in EPISODE_STEPS if step not in completed]
        if episode_name in processed:
            status = "completed"
        elif current_phase == "episode" and episode_name == current_episode:
            status = "running"
        elif completed:
            status = "partial"
        else:
            status = "pending"
        episode_rows.append(
            {
                "episode": episode_name,
                "status": status,
                "completed_steps": completed,
                "remaining_steps": remaining,
                "current_step": current_step if status == "running" else None,
            }
        )

    global_rows = []
    completed_global_steps = set(state.get("global_steps_completed", []) or [])
    for step_name in global_steps_to_run(cfg):
        if step_name in completed_global_steps:
            status = "completed"
        elif current_phase == "global" and current_step == step_name:
            status = "running"
        else:
            status = "pending"
        global_rows.append({"step": step_name, "status": status})

    return {
        "status": str(state.get("status", "running")),
        "updated_at": str(state.get("updated_at", "")),
        "autosave_reason": str(state.get("autosave_reason", "")),
        "setup_completed": bool(state.get("setup_completed", False)),
        "processed_count": int(state.get("processed_count", 0) or 0),
        "pending_inbox_count": pending_inbox_count,
        "current_phase": state.get("current_phase"),
        "current_episode_file": state.get("current_episode_file"),
        "current_episode_name": state.get("current_episode_name"),
        "current_step": state.get("current_step"),
        "episode_progress": episode_rows,
        "global_progress": global_rows,
    }


def render_status_markdown(snapshot: dict) -> str:
    lines = [
        "# 99 Process Status",
        "",
        f"- Status: {snapshot.get('status', '-')}",
        f"- Updated: {snapshot.get('updated_at', '-')}",
        f"- Reason: {snapshot.get('autosave_reason', '-')}",
        f"- Setup completed: {'yes' if snapshot.get('setup_completed') else 'no'}",
        f"- Processed source episodes: {snapshot.get('processed_count', 0)}",
        f"- Files currently in inbox: {snapshot.get('pending_inbox_count', 0)}",
        f"- Phase: {snapshot.get('current_phase') or '-'}",
        f"- Current episode: {snapshot.get('current_episode_name') or '-'}",
        f"- Current step: {snapshot.get('current_step') or '-'}",
        "",
        "## Episode Status",
        "",
    ]
    episode_progress = snapshot.get("episode_progress", []) or []
    if episode_progress:
        for row in episode_progress:
            remaining = ", ".join(row.get("remaining_steps", []) or []) or "-"
            completed = ", ".join(row.get("completed_steps", []) or []) or "-"
            lines.extend(
                [
                    f"### {row.get('episode')}",
                    f"- Status: {row.get('status')}",
                    f"- Completed: {completed}",
                    f"- Remaining: {remaining}",
                    f"- Current step: {row.get('current_step') or '-'}",
                    "",
                ]
            )
    else:
        lines.append("No episodes are currently listed in the status snapshot.")
        lines.append("")

    lines.append("## Global Status")
    lines.append("")
    global_progress = snapshot.get("global_progress", []) or []
    if global_progress:
        for row in global_progress:
            lines.append(f"- {row.get('step')}: {row.get('status')}")
    else:
        lines.append("- No global steps are currently active.")
    lines.append("")
    return "\n".join(lines)


def write_status_files(cfg: dict, snapshot: dict) -> None:
    root = autosave_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    status_json_path(cfg).write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    status_markdown_path(cfg).write_text(render_status_markdown(snapshot), encoding="utf-8")


def save_autosave(cfg: dict, state: dict, reason: str, inbox_dir: Path | None = None) -> Path:
    root = autosave_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_timestamp()
    state["autosave_reason"] = reason
    target = root / autosave_filename()
    target.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    prune_autosaves(cfg)
    write_status_files(cfg, build_status_snapshot(cfg, state, inbox_dir))
    return target


def episode_step_args(script_name: str, episode_file: str, episode_name: str) -> list[str]:
    if script_name == "02_import_episode.py":
        return []
    if script_name == "03_split_scenes.py":
        return ["--episode-file", episode_file]
    if script_name in {"04_diarize_and_transcribe.py", "05_link_faces_and_speakers.py"}:
        return ["--episode", episode_name]
    return []


def episode_step_title(script_name: str, episode_file: str, episode_name: str) -> str:
    titles = {
        "02_import_episode.py": f"Import: {episode_file}",
        "03_split_scenes.py": f"Scene Detection: {episode_name}",
        "04_diarize_and_transcribe.py": f"Audio Segmentation And Transcription: {episode_name}",
        "05_link_faces_and_speakers.py": f"Link Faces And Voices: {episode_name}",
        "07_build_dataset.py": "Build training dataset from reviewed data",
        "08_train_series_model.py": "Series Model aus reviewtem Datensatz trainieren",
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
    state["current_step"] = None


def global_steps_to_run(cfg: dict) -> list[str]:
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    adapter_cfg = cfg.get("adapter_training", {}) if isinstance(cfg.get("adapter_training"), dict) else {}
    fine_tune_cfg = cfg.get("fine_tune_training", {}) if isinstance(cfg.get("fine_tune_training"), dict) else {}
    backend_cfg = cfg.get("backend_fine_tune", {}) if isinstance(cfg.get("backend_fine_tune"), dict) else {}
    needs_foundation_training = bool(foundation_cfg.get("required_before_generate", True)) or bool(
        foundation_cfg.get("required_before_render", True)
    )
    steps = ["07_build_dataset.py", "08_train_series_model.py"]
    if bool(foundation_cfg.get("prepare_after_batch", False)) or needs_foundation_training:
        steps.append("09_prepare_foundation_training.py")
        if bool(foundation_cfg.get("auto_train_after_prepare", False)) or needs_foundation_training:
            steps.append("10_train_foundation_models.py")
            if bool(adapter_cfg.get("auto_train_after_foundation", True)):
                steps.append("11_train_adapter_models.py")
                if bool(fine_tune_cfg.get("auto_train_after_adapter", True)):
                    steps.append("12_train_fine_tune_models.py")
                    if bool(backend_cfg.get("auto_run_after_fine_tune", True)):
                        steps.append("13_run_backend_finetunes.py")
    steps.extend(
        [
            "14_generate_episode_from_trained_model.py",
            "15_generate_storyboard_assets.py",
            "16_run_storyboard_backend.py",
            "17_build_series_bible.py",
            "18_render_episode.py",
        ]
    )
    return steps


def global_step_title(script_name: str) -> str:
    titles = {
        "07_build_dataset.py": "Build training dataset from reviewed data",
        "08_train_series_model.py": "Train Series Model",
        "09_prepare_foundation_training.py": "Prepare Foundation Training",
        "10_train_foundation_models.py": "Train Foundation Models",
        "11_train_adapter_models.py": "Train Local Adapter Profiles",
        "12_train_fine_tune_models.py": "Train Local Fine-Tune Profiles",
        "13_run_backend_finetunes.py": "Create Concrete Backend Fine-Tune Runs",
        "14_generate_episode_from_trained_model.py": "Generate New Episode From Trained Model",
        "15_generate_storyboard_assets.py": "Generate Storyboard Scene Assets",
        "16_run_storyboard_backend.py": "Materialize Storyboard Backend Frames",
        "17_build_series_bible.py": "Update Series Bible",
        "18_render_episode.py": "Render Storyboard Video",
    }
    return titles[script_name]


def main() -> None:
    rerun_in_runtime()
    headline("Run Full Series Pipeline")
    cfg = load_config()
    inbox_dir = resolve_project_path(cfg["paths"]["inbox_episodes"])
    state = load_latest_autosave(cfg) or default_state()
    initial_inbox_files = sorted([path for path in inbox_dir.iterdir() if path.is_file()]) if inbox_dir.exists() else []
    initial_episode_total = len(initial_inbox_files)
    resume_episode = str(state.get("current_episode_name") or "").strip()
    if resume_episode and resume_episode not in set(state.get("processed_episodes", []) or []):
        initial_episode_total += 1
    episode_batch_reporter = LiveProgressReporter(
        script_name="99_process_next_episode.py",
        total=max(1, initial_episode_total),
        phase_label="Process Source Episodes",
        parent_label="Inbox Batch",
    )
    processed_in_this_run = 0

    if load_latest_autosave(cfg):
        info("Found an existing autosave. The pipeline will resume from the last successfully completed step.")

    if not bool(state.get("setup_completed", False)):
        state["current_phase"] = "setup"
        state["current_step"] = SETUP_STEP
        save_autosave(cfg, state, "setup_started", inbox_dir)
        run_step(SETUP_STEP, "Set Up Project Structure")
        state["setup_completed"] = True
        state["current_phase"] = None
        state["current_step"] = None
        save_autosave(cfg, state, "setup_completed", inbox_dir)

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
            state["current_step"] = None
            state.setdefault("episode_steps_completed", {}).setdefault(episode_name, [])
            save_autosave(cfg, state, f"episode_selected:{episode_name}", inbox_dir)

        info(f"Starting batch episode: {next_video_name}")
        finished_steps = set(completed_episode_steps(state, episode_name))
        episode_step_reporter = LiveProgressReporter(
            script_name="99_process_next_episode.py",
            total=len(EPISODE_STEPS),
            phase_label="Steps Per Source Episode",
            parent_label=episode_name,
        )
        for script_name in EPISODE_STEPS:
            if script_name in finished_steps:
                continue
            step_title = episode_step_title(script_name, next_video_name, episode_name)
            step_started_at = time.time()
            state["current_phase"] = "episode"
            state["current_episode_file"] = next_video_name
            state["current_episode_name"] = episode_name
            state["current_step"] = script_name
            save_autosave(cfg, state, f"episode_step_started:{episode_name}:{script_name}", inbox_dir)
            episode_step_reporter.update(
                len(finished_steps),
                current_label=step_title,
                extra_label=f"Running now: {script_name}",
                force=True,
                scope_current=len(finished_steps),
                scope_total=len(EPISODE_STEPS),
                scope_started_at=step_started_at,
                scope_label="Current Episode Steps",
            )
            run_step(
                script_name,
                step_title,
                episode_step_args(script_name, next_video_name, episode_name),
            )
            mark_episode_step_completed(state, episode_name, script_name)
            finished_steps.add(script_name)
            episode_step_reporter.update(
                len(finished_steps),
                current_label=step_title,
                extra_label=f"Completed: {script_name}",
                scope_current=len(finished_steps),
                scope_total=len(EPISODE_STEPS),
                scope_started_at=step_started_at,
                scope_label="Current Episode Steps",
            )
            save_autosave(cfg, state, f"{episode_name}:{script_name}", inbox_dir)

        inbox_file = inbox_dir / next_video_name
        if cleanup_processed_inbox_episode(inbox_file):
            info(f"Inbox file removed: {next_video_name}")
        episode_step_reporter.finish(current_label=episode_name, extra_label=f"Total completed steps: {len(finished_steps)}")
        mark_episode_completed(state, episode_name)
        processed_in_this_run += 1
        episode_batch_reporter.update(
            processed_in_this_run,
            current_label=episode_name,
            extra_label=f"Komplett verarbeitet: {episode_name}",
        )
        save_autosave(cfg, state, f"episode_completed:{episode_name}", inbox_dir)

    review_count = open_review_item_count(cfg)
    if review_count > 0:
        info(
            f"Es gibt noch {review_count} offene Review-Faelle. "
            "Run 06_review_unknowns.py first before dataset rebuild, training, generation, or render can continue."
        )
        state["current_phase"] = "review"
        state["current_step"] = "06_review_unknowns.py"
        save_autosave(cfg, state, "review_pending", inbox_dir)
        return

    if int(state.get("processed_count", 0) or 0) == 0:
        write_status_files(cfg, build_status_snapshot(cfg, state, inbox_dir))
        info("No new episodes found in the inbox folder.")
        return

    state["current_phase"] = "global"
    state["current_step"] = None
    save_autosave(cfg, state, "global_phase_started", inbox_dir)
    steps_to_run = global_steps_to_run(cfg)
    completed_global_steps = set(state.get("global_steps_completed", []) or [])
    global_reporter = LiveProgressReporter(
        script_name="99_process_next_episode.py",
        total=max(1, len(steps_to_run)),
        phase_label="Global Pipeline Steps",
        parent_label="Training to Render",
    )
    for script_name in steps_to_run:
        if script_name in completed_global_steps:
            continue
        global_step_started_at = time.time()
        state["current_phase"] = "global"
        state["current_step"] = script_name
        save_autosave(cfg, state, f"global_step_started:{script_name}", inbox_dir)
        global_reporter.update(
            len(completed_global_steps),
            current_label=global_step_title(script_name),
            extra_label=f"Running now: {script_name}",
            force=True,
            scope_current=len(completed_global_steps),
            scope_total=len(steps_to_run),
            scope_started_at=global_step_started_at,
            scope_label="Global Steps",
        )
        run_step(script_name, global_step_title(script_name))
        mark_global_step_completed(state, script_name)
        completed_global_steps.add(script_name)
        global_reporter.update(
            len(completed_global_steps),
            current_label=global_step_title(script_name),
            extra_label=f"Completed: {script_name}",
            scope_current=len(completed_global_steps),
            scope_total=len(steps_to_run),
            scope_started_at=global_step_started_at,
            scope_label="Global Steps",
        )
        save_autosave(cfg, state, f"global:{script_name}", inbox_dir)
    global_reporter.finish(current_label="Global Phase", extra_label=f"Total global steps: {len(completed_global_steps)}")
    episode_batch_reporter.finish(current_label="Inbox Batch", extra_label=f"Source episodes in this run: {processed_in_this_run}")

    state["status"] = "completed"
    state["current_phase"] = None
    state["current_step"] = None
    save_autosave(cfg, state, "pipeline_completed", inbox_dir)
    ok(f"The pipeline completed successfully. New source episodes processed: {state['processed_count']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise

