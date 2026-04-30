#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from pipeline_common import (
    PROJECT_ROOT,
    add_batch_job,
    add_shared_worker_arguments,
    distributed_item_lease,
    distributed_step_runtime_root,
    ensure_quality_first_ready,
    prepare_quality_backend_assets_runtime,
    open_face_review_item_count,
    LiveProgressReporter,
    SCRIPT_DIR,
    error,
    latest_generated_episode_artifacts,
    headline,
    info,
    load_config,
    load_batch_jobs,
    next_unprocessed_video,
    ok,
    read_json,
    release_mode_enabled,
    rerun_in_runtime,
    resolve_project_path,
    runtime_python,
    shared_worker_cli_args,
    shared_worker_id_for_args,
    shared_workers_enabled_for_args,
    update_batch_job_status,
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
    "50_run_backend_finetunes.py",
    "13_generate_episode.py",
    "14_generate_storyboard_assets.py",
    "54_run_storyboard_backend.py",
    "15_render_episode.py",
    "16_build_series_bible.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full end-to-end series pipeline.")
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Skip model downloads in 09 and only use existing downloads/updates.",
    )
    add_shared_worker_arguments(parser)
    return parser.parse_args()


def format_quality_summary(label: object, percent: object) -> str:
    quality_label = str(label or "").strip()
    if not quality_label:
        return "-"
    return f"{quality_label} ({int(round(float(percent or 0.0)))}%)"


def run_step(
    script_name: str,
    title: str,
    extra_args: list[str] | None = None,
    shared_args: list[str] | None = None,
) -> None:
    headline(title)
    command = [str(runtime_python()), str(SCRIPT_DIR / script_name), *(extra_args or []), *(shared_args or [])]
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


def next_autosave_path(root: Path) -> Path:
    base_name = autosave_filename()
    target = root / base_name
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    for index in range(1, 1000):
        candidate = root / f"{stem}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not create a unique autosave filename.")


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
        "skip_downloads": False,
        "processed_count": 0,
        "current_phase": None,
        "current_episode_file": None,
        "current_episode_name": None,
        "current_step": None,
        "episode_steps_completed": {},
        "processed_episodes": [],
        "global_planned_steps": [],
        "global_completed_step_labels": [],
        "global_steps_completed": [],
        "latest_generated_episode": {},
        "global_step_outputs": {},
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
    for row in global_step_payloads(state, cfg):
        step_name = str(row.get("script_name") or "").strip()
        step_label = str(row.get("label") or step_name).strip() or step_name
        if step_name in completed_global_steps:
            status = "completed"
        elif current_phase == "global" and current_step == step_name:
            status = "running"
        else:
            status = "pending"
        global_rows.append({"step": step_label, "script_name": step_name, "status": status})

    return {
        "status": str(state.get("status", "running")),
        "updated_at": str(state.get("updated_at", "")),
        "autosave_reason": str(state.get("autosave_reason", "")),
        "setup_completed": bool(state.get("setup_completed", False)),
        "skip_downloads": bool(state.get("skip_downloads", False)),
        "processed_count": int(state.get("processed_count", 0) or 0),
        "pending_inbox_count": pending_inbox_count,
        "current_phase": state.get("current_phase"),
        "current_episode_file": state.get("current_episode_file"),
        "current_episode_name": state.get("current_episode_name"),
        "current_step": state.get("current_step"),
        "global_planned_steps": list(state.get("global_planned_steps", []) or []),
        "global_completed_step_labels": list(state.get("global_completed_step_labels", []) or []),
        "latest_generated_episode": dict(state.get("latest_generated_episode", {}) or {}),
        "global_step_outputs": dict(state.get("global_step_outputs", {}) or {}),
        "episode_progress": episode_rows,
        "global_progress": global_rows,
    }


def format_completion_ratio(value: object) -> str:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return "-"
    if ratio < 0:
        ratio = 0.0
    if ratio > 1:
        ratio = 1.0
    return f"{ratio * 100:.1f}%"


def format_backend_tasks(value: object) -> str:
    if not isinstance(value, list):
        return "-"
    tasks = [str(task).strip() for task in value if str(task).strip()]
    return ", ".join(tasks) if tasks else "-"


def render_status_markdown(snapshot: dict) -> str:
    lines = [
        "# 99 Process Status",
        "",
        f"- Status: {snapshot.get('status', '-')}",
        f"- Updated: {snapshot.get('updated_at', '-')}",
        f"- Reason: {snapshot.get('autosave_reason', '-')}",
        f"- Setup completed: {'yes' if snapshot.get('setup_completed') else 'no'}",
        f"- Skip downloads: {'yes' if snapshot.get('skip_downloads') else 'no'}",
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

    latest_generated_episode = snapshot.get("latest_generated_episode", {}) or {}
    if latest_generated_episode:
        lines.extend(
            [
                "## Latest Generated Episode",
                "",
                f"- Episode: {latest_generated_episode.get('episode_id') or '-'}",
                f"- Display title: {latest_generated_episode.get('display_title') or '-'}",
                f"- Render mode: {latest_generated_episode.get('render_mode') or '-'}",
                f"- Production readiness: {latest_generated_episode.get('production_readiness') or '-'}",
                f"- Scene count: {latest_generated_episode.get('scene_count') or '-'}",
                f"- Scene video coverage: {format_completion_ratio(latest_generated_episode.get('scene_video_completion_ratio'))}",
                f"- Scene dialogue coverage: {format_completion_ratio(latest_generated_episode.get('scene_dialogue_completion_ratio'))}",
                f"- Scene master coverage: {format_completion_ratio(latest_generated_episode.get('scene_master_completion_ratio'))}",
                f"- Episode quality: {format_quality_summary(latest_generated_episode.get('quality_label'), latest_generated_episode.get('quality_percent'))}",
                f"- Minimum scene quality: {format_quality_summary(latest_generated_episode.get('minimum_scene_quality_label'), latest_generated_episode.get('minimum_scene_quality_percent'))}",
                f"- Scenes below watch threshold: {latest_generated_episode.get('scenes_below_watch_threshold') or 0}",
                f"- Scenes below release threshold: {latest_generated_episode.get('scenes_below_release_threshold') or 0}",
                f"- Release gate passed: {'yes' if latest_generated_episode.get('release_gate_passed') else 'no'}",
                f"- Quality gate warnings: {len(latest_generated_episode.get('quality_gate_warnings', []) or [])}",
                f"- Quality gate report: {latest_generated_episode.get('quality_gate_report') or '-'}",
                f"- Regeneration queue count: {latest_generated_episode.get('regeneration_queue_count') or 0}",
                f"- Regeneration queue manifest: {latest_generated_episode.get('regeneration_queue_manifest') or '-'}",
                f"- Regeneration apply requested: {'yes' if latest_generated_episode.get('regeneration_apply_requested') else 'no'}",
                f"- Last regeneration request: {latest_generated_episode.get('regeneration_last_requested_at') or '-'}",
                f"- Last regeneration apply: {latest_generated_episode.get('regeneration_last_applied_at') or '-'}",
                f"- Backend runner status: {latest_generated_episode.get('backend_runner_status') or '-'}",
                f"- Backend runner coverage: {format_completion_ratio(latest_generated_episode.get('backend_runner_coverage_ratio'))}",
                f"- Backend runners ready: {latest_generated_episode.get('backend_runner_ready_count') or 0}/{latest_generated_episode.get('backend_runner_expected_count') or 0}",
                f"- Master backend runner: {latest_generated_episode.get('master_backend_runner_status') or '-'}",
                f"- Remaining backend tasks: {format_backend_tasks(latest_generated_episode.get('remaining_backend_tasks'))}",
                f"- Delivery bundle: {latest_generated_episode.get('delivery_bundle_root') or '-'}",
                f"- Delivery manifest: {latest_generated_episode.get('delivery_manifest') or '-'}",
                f"- Delivery watch episode: {latest_generated_episode.get('delivery_episode') or '-'}",
                f"- Stable latest delivery episode: {latest_generated_episode.get('latest_delivery_episode') or '-'}",
                f"- Final render: {latest_generated_episode.get('final_render') or '-'}",
                f"- Full generated episode: {latest_generated_episode.get('full_generated_episode') or '-'}",
                f"- Production package: {latest_generated_episode.get('production_package') or '-'}",
                f"- Render manifest: {latest_generated_episode.get('render_manifest') or '-'}",
                "",
            ]
        )
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
    target = next_autosave_path(root)
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
        "08_train_series_model.py": "Train series model from reviewed data",
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
    return [row[0] for row in global_step_rows(cfg)]


def global_step_rows(cfg: dict, skip_downloads: bool = False) -> list[tuple[str, str, list[str]]]:
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    adapter_cfg = cfg.get("adapter_training", {}) if isinstance(cfg.get("adapter_training"), dict) else {}
    fine_tune_cfg = cfg.get("fine_tune_training", {}) if isinstance(cfg.get("fine_tune_training"), dict) else {}
    backend_cfg = cfg.get("backend_fine_tune", {}) if isinstance(cfg.get("backend_fine_tune"), dict) else {}
    needs_foundation_training = bool(foundation_cfg.get("required_before_generate", True)) or bool(
        foundation_cfg.get("required_before_render", True)
    )
    steps: list[tuple[str, str, list[str]]] = [
        ("07_build_dataset.py", "Build training dataset from reviewed data", []),
        ("08_train_series_model.py", "Train Series Model", []),
    ]
    if bool(foundation_cfg.get("prepare_after_batch", False)) or needs_foundation_training:
        prepare_args = ["--skip-downloads"] if skip_downloads else []
        steps.append(("09_prepare_foundation_training.py", "Prepare Foundation Training", prepare_args))
        if bool(foundation_cfg.get("auto_train_after_prepare", False)) or needs_foundation_training:
            steps.append(("10_train_foundation_models.py", "Train Foundation Models", []))
            if bool(adapter_cfg.get("auto_train_after_foundation", True)):
                steps.append(("11_train_adapter_models.py", "Train Local Adapter Profiles", []))
                if bool(fine_tune_cfg.get("auto_train_after_adapter", True)):
                    steps.append(("12_train_fine_tune_models.py", "Train Local Fine-Tune Profiles", []))
                    if bool(backend_cfg.get("auto_run_after_fine_tune", True)):
                        steps.append(("50_run_backend_finetunes.py", "Create Concrete Backend Fine-Tune Runs", []))
    steps.extend(
        [
            ("13_generate_episode.py", "Generate New Episode From Trained Model", []),
            ("14_generate_storyboard_assets.py", "Generate Storyboard Scene Assets", []),
            ("54_run_storyboard_backend.py", "Materialize Storyboard Backend Frames", []),
            ("15_render_episode.py", "Render Finished Episode", []),
        ]
    )
    if release_mode_enabled(cfg):
        release_cfg = cfg.get("release_mode", {}) if isinstance(cfg.get("release_mode"), dict) else {}
        quality_gate_args = ["--strict"] if bool(release_cfg.get("strict_warnings", False)) else []
        steps.append(("52_quality_gate.py", "Run Release-Style Quality Gate", quality_gate_args))
    steps.append(("16_build_series_bible.py", "Update Series Bible", []))
    return steps


def global_step_label(script_name: str, extra_args: list[str] | None = None) -> str:
    suffix = " ".join(str(arg) for arg in (extra_args or []) if str(arg).strip())
    return f"{script_name} {suffix}".strip()


def global_step_payloads(state: dict, cfg: dict) -> list[dict[str, object]]:
    stored = state.get("global_planned_steps", [])
    if isinstance(stored, list) and stored:
        payloads: list[dict[str, object]] = []
        for row in stored:
            if not isinstance(row, dict):
                continue
            script_name = str(row.get("script_name") or "").strip()
            title = str(row.get("title") or "").strip()
            args = row.get("args", [])
            if not script_name:
                continue
            if not isinstance(args, list):
                args = []
            payloads.append(
                {
                    "script_name": script_name,
                    "title": title,
                    "args": [str(arg) for arg in args],
                    "label": str(row.get("label") or global_step_label(script_name, args)),
                }
            )
        if payloads:
            return payloads
    return serialize_global_step_rows(global_step_rows(cfg, skip_downloads=bool(state.get("skip_downloads", False))))


def serialize_global_step_rows(step_rows: list[tuple[str, str, list[str]]]) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for script_name, title, extra_args in step_rows:
        payloads.append(
            {
                "script_name": script_name,
                "title": title,
                "args": list(extra_args),
                "label": global_step_label(script_name, extra_args),
            }
        )
    return payloads


def completed_global_step_labels(planned_steps: list[dict[str, object]], completed_scripts: set[str]) -> list[str]:
    labels: list[str] = []
    for row in planned_steps:
        script_name = str(row.get("script_name") or "").strip()
        if script_name in completed_scripts:
            labels.append(str(row.get("label") or script_name))
    return labels


def global_step_title(script_name: str) -> str:
    titles = {
        "07_build_dataset.py": "Build training dataset from reviewed data",
        "08_train_series_model.py": "Train Series Model",
        "09_prepare_foundation_training.py": "Prepare Foundation Training",
        "10_train_foundation_models.py": "Train Foundation Models",
        "11_train_adapter_models.py": "Train Local Adapter Profiles",
        "12_train_fine_tune_models.py": "Train Local Fine-Tune Profiles",
        "50_run_backend_finetunes.py": "Create Concrete Backend Fine-Tune Runs",
        "13_generate_episode.py": "Generate New Episode From Trained Model",
        "14_generate_storyboard_assets.py": "Generate Storyboard Scene Assets",
        "54_run_storyboard_backend.py": "Materialize Storyboard Backend Frames",
        "15_render_episode.py": "Render Finished Episode",
        "52_quality_gate.py": "Run Release-Style Quality Gate",
        "16_build_series_bible.py": "Update Series Bible",
    }
    return titles[script_name]


def record_global_generated_episode_outputs(state: dict, cfg: dict, script_name: str) -> None:
    if script_name not in {
        "13_generate_episode.py",
        "14_generate_storyboard_assets.py",
        "54_run_storyboard_backend.py",
        "15_render_episode.py",
        "52_quality_gate.py",
        "16_build_series_bible.py",
    }:
        return
    outputs = latest_generated_episode_artifacts(cfg)
    if not outputs:
        return
    state["latest_generated_episode"] = outputs
    global_step_outputs = state.get("global_step_outputs", {})
    if not isinstance(global_step_outputs, dict):
        global_step_outputs = {}
    global_step_outputs[script_name] = outputs
    state["global_step_outputs"] = global_step_outputs


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Run Full Series Pipeline")
    cfg = load_config()
    prepare_quality_backend_assets_runtime()
    ensure_quality_first_ready(cfg, context_label="99_process_next_episode.py")
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)
    child_shared_args = shared_worker_cli_args(cfg, args)
    if shared_workers:
        info(f"Shared NAS workers: enabled ({worker_id})")
    inbox_dir = resolve_project_path(cfg["paths"]["inbox_episodes"])
    lease_manager = distributed_item_lease(
        root=distributed_step_runtime_root("99_process_next_episode", "global"),
        lease_name="global",
        cfg=cfg,
        worker_id=worker_id,
        enabled=shared_workers,
        meta={"step": "99_process_next_episode", "scope": "global", "worker_id": worker_id},
    )
    acquired = lease_manager.__enter__()
    if not acquired:
        info("The full pipeline orchestrator is already running on another worker.")
        lease_manager.__exit__(None, None, None)
        return
    state = load_latest_autosave(cfg) or default_state()
    try:
        state["skip_downloads"] = bool(args.skip_downloads)
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
            run_step(SETUP_STEP, "Set Up Project Structure", shared_args=child_shared_args)
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
                    shared_args=child_shared_args,
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

            jobs = load_batch_jobs(PROJECT_ROOT)
            for job in jobs:
                if job.get("type") == "process_episode" and job.get("status") == "running":
                    if job.get("config", {}).get("episode_name") == episode_name:
                        update_batch_job_status(PROJECT_ROOT, job.get("id", ""), "completed")
                        break

            processed_in_this_run += 1
            episode_batch_reporter.update(
                processed_in_this_run,
                current_label=episode_name,
                extra_label=f"Completed: {episode_name}",
            )
            save_autosave(cfg, state, f"episode_completed:{episode_name}", inbox_dir)

        review_count = open_face_review_item_count(cfg)
        if review_count > 0:
            info(
                f"There are still {review_count} open face review cases. "
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
        step_rows = global_step_rows(cfg, skip_downloads=bool(args.skip_downloads))
        state["global_planned_steps"] = serialize_global_step_rows(step_rows)
        steps_to_run = [row[0] for row in step_rows]
        completed_global_steps = set(state.get("global_steps_completed", []) or [])
        state["global_completed_step_labels"] = completed_global_step_labels(state["global_planned_steps"], completed_global_steps)
        save_autosave(cfg, state, "global_phase_started", inbox_dir)
        global_reporter = LiveProgressReporter(
            script_name="99_process_next_episode.py",
            total=max(1, len(steps_to_run)),
            phase_label="Global Pipeline Steps",
            parent_label="Training to Render",
        )
        for script_name, step_title, step_args in step_rows:
            if script_name in completed_global_steps:
                continue
            global_step_started_at = time.time()
            state["current_phase"] = "global"
            state["current_step"] = script_name
            save_autosave(cfg, state, f"global_step_started:{script_name}", inbox_dir)
            global_reporter.update(
                len(completed_global_steps),
                current_label=step_title,
                extra_label=f"Running now: {script_name}",
                force=True,
                scope_current=len(completed_global_steps),
                scope_total=len(steps_to_run),
                scope_started_at=global_step_started_at,
                scope_label="Global Steps",
            )
            run_step(script_name, step_title, step_args, shared_args=child_shared_args)
            mark_global_step_completed(state, script_name)
            completed_global_steps.add(script_name)
            state["global_completed_step_labels"] = completed_global_step_labels(state["global_planned_steps"], completed_global_steps)
            record_global_generated_episode_outputs(state, cfg, script_name)
            global_reporter.update(
                len(completed_global_steps),
                current_label=step_title,
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
    finally:
        lease_manager.__exit__(None, None, None)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise

