#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "ai_series_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse
import subprocess

from support_scripts.pipeline_common import (
    add_shared_worker_arguments,
    distributed_item_lease,
    distributed_step_runtime_root,
    ensure_quality_first_ready,
    open_face_review_item_count,
    LiveProgressReporter,
    SCRIPT_DIR,
    WORKSPACE_ROOT,
    error,
    headline,
    info,
    load_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    release_mode_enabled,
    rerun_in_runtime,
    runtime_python,
    shared_worker_cli_args,
    shared_worker_id_for_args,
    shared_workers_enabled_for_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the pipeline after manual character review"
    )
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Skip setup and foundation downloads in 00/08 and use only existing local assets.",
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
    add_shared_worker_arguments(parser)
    return parser.parse_args()


def run_step(
    script_name: str,
    title: str,
    extra_args: list[str] | None = None,
    shared_args: list[str] | None = None,
) -> None:
    headline(title)
    command = [str(runtime_python()), str(WORKSPACE_ROOT / script_name), *(extra_args or []), *(shared_args or [])]
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def planned_refresh_steps(cfg: dict, skip_downloads: bool = False, stop_after_training: bool = False) -> list[tuple[str, str, list[str]]]:
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    adapter_cfg = cfg.get("adapter_training", {}) if isinstance(cfg.get("adapter_training"), dict) else {}
    fine_tune_cfg = cfg.get("fine_tune_training", {}) if isinstance(cfg.get("fine_tune_training"), dict) else {}
    backend_cfg = cfg.get("backend_fine_tune", {}) if isinstance(cfg.get("backend_fine_tune"), dict) else {}
    needs_foundation_training = bool(foundation_cfg.get("required_before_generate", True)) or bool(
        foundation_cfg.get("required_before_render", True)
    )
    planned_steps: list[tuple[str, str, list[str]]] = [
        ("06_manage_character_relationships.py", "Review character groups and relationships", []),
        ("07_build_dataset.py", "Rebuild datasets with the latest reviewed character names", ["--force"]),
        ("08_train_series_model.py", "Retrain the series model with the latest reviewed names", []),
    ]
    prepare_args = ["--force"]
    if skip_downloads:
        prepare_args.append("--skip-downloads")
    if bool(foundation_cfg.get("prepare_after_batch", False)) or needs_foundation_training:
        planned_steps.append(
            ("09_prepare_foundation_training.py", "Prepare foundation training with the latest reviewed character state", prepare_args)
        )
        if bool(foundation_cfg.get("auto_train_after_prepare", False)) or needs_foundation_training:
            planned_steps.append(
                ("10_train_foundation_models.py", "Retrain foundation packs with the latest reviewed character state", ["--force"])
            )
            if bool(adapter_cfg.get("auto_train_after_foundation", True)):
                planned_steps.append(
                    ("11_train_adapter_models.py", "Retrain local adapter profiles with the latest reviewed character state", ["--force"])
                )
                if bool(fine_tune_cfg.get("auto_train_after_adapter", True)):
                    planned_steps.append(
                        ("12_train_fine_tune_models.py", "Retrain local fine-tune profiles with the latest reviewed character state", ["--force"])
                    )
                    if bool(backend_cfg.get("auto_run_after_fine_tune", True)):
                        planned_steps.append(
                            ("13_run_backend_finetunes.py", "Create backend fine-tune runs with the latest reviewed character state", ["--force"])
                        )
    if not stop_after_training:
        planned_steps.extend(
            [
                ("14_generate_episode.py", "Generate a new episode from the refreshed model", []),
                ("15_generate_storyboard_assets.py", "Generate storyboard assets for the refreshed episode", []),
                ("16_run_storyboard_backend.py", "Materialize local storyboard backend frames for the refreshed episode", []),
                ("17_render_episode.py", "Render the refreshed episode", []),
            ]
        )
        if release_mode_enabled(cfg):
            release_cfg = cfg.get("release_mode", {}) if isinstance(cfg.get("release_mode"), dict) else {}
            quality_gate_args = ["--strict"] if bool(release_cfg.get("strict_warnings", False)) else []
            planned_steps.append(("18_quality_gate.py", "Run the release-style quality gate for the refreshed episode", quality_gate_args))
        planned_steps.append(("20_build_series_bible.py", "Update the series bible with the refreshed state", []))
        planned_steps.append(("21_export_package.py", "Export the refreshed finished-episode package", []))
    return planned_steps


def completed_step_labels(planned_steps: list[tuple[str, str, list[str]]]) -> list[str]:
    labels: list[str] = []
    for script_name, _title, extra_args in planned_steps:
        suffix = " ".join(str(arg) for arg in extra_args if str(arg).strip())
        labels.append(f"{script_name} {suffix}".strip())
    return labels


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Rebuild After Manual Character Review")
    cfg = load_config()
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)
    child_shared_args = shared_worker_cli_args(cfg, args)
    setup_args = ["--skip-downloads"] if bool(args.skip_downloads) else []
    run_step("00_prepare_runtime.py", "Prepare Runtime And Project Setup", setup_args, shared_args=child_shared_args)
    cfg = load_config()
    if not args.stop_after_training:
        ensure_quality_first_ready(cfg, context_label="22_refresh_after_manual_review.py")
    autosave_target = "global"
    if shared_workers:
        info(f"Shared NAS workers: enabled ({worker_id})")
    mark_step_started(
        "22_refresh_after_manual_review",
        autosave_target,
        {
            "allow_open_review": bool(args.allow_open_review),
            "skip_downloads": bool(args.skip_downloads),
            "stop_after_training": bool(args.stop_after_training),
        },
    )
    lease_manager = distributed_item_lease(
        root=distributed_step_runtime_root("22_refresh_after_manual_review", autosave_target),
        lease_name=autosave_target,
        cfg=cfg,
        worker_id=worker_id,
        enabled=shared_workers,
        meta={"step": "22_refresh_after_manual_review", "scope": autosave_target, "worker_id": worker_id},
    )
    acquired = lease_manager.__enter__()
    if not acquired:
        info("This rebuild run is already active on another worker.")
        lease_manager.__exit__(None, None, None)
        return
    try:
        if not args.allow_open_review:
            review_count = open_face_review_item_count(cfg)
            if review_count > 0:
                raise RuntimeError(
                    f"There are still {review_count} open face review cases. "
                    "Run 05_review_unknowns.py first or intentionally continue with --allow-open-review."
                )
        planned_steps = planned_refresh_steps(
            cfg,
            skip_downloads=bool(args.skip_downloads),
            stop_after_training=bool(args.stop_after_training),
        )
        reporter = LiveProgressReporter(
            script_name="22_refresh_after_manual_review.py",
            total=len(planned_steps),
            phase_label="Rebuild After Review",
            parent_label="global",
        )
        completed_count = 0
        for script_name, title, extra_args in planned_steps:
            reporter.update(completed_count, current_label=title, extra_label=f"Running now: {script_name}", force=True)
            run_step(script_name, title, extra_args, shared_args=child_shared_args)
            completed_count += 1
            reporter.update(completed_count, current_label=title, extra_label=f"Completed: {script_name}")
        reporter.finish(current_label="Rebuild", extra_label=f"Completed steps: {completed_count}")

        mark_step_completed(
            "22_refresh_after_manual_review",
            autosave_target,
            {
                "allow_open_review": bool(args.allow_open_review),
                "skip_downloads": bool(args.skip_downloads),
                "stop_after_training": bool(args.stop_after_training),
                "completed_steps": completed_step_labels(planned_steps),
            },
        )
        ok("Rebuild after manual character review completed.")
    except Exception as exc:
        mark_step_failed(
            "22_refresh_after_manual_review",
            str(exc),
            autosave_target,
            {
                "allow_open_review": bool(args.allow_open_review),
                "skip_downloads": bool(args.skip_downloads),
                "stop_after_training": bool(args.stop_after_training),
            },
        )
        raise
    finally:
        lease_manager.__exit__(None, None, None)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise


