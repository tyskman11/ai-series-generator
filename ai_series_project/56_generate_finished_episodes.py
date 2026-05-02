#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from support_scripts.pipeline_common import (
    add_batch_job,
    add_shared_worker_arguments,
    distributed_item_lease,
    distributed_step_runtime_root,
    ensure_quality_first_ready,
    open_face_review_item_count,
    LiveProgressReporter,
    PROJECT_ROOT,
    SCRIPT_DIR,
    error,
    generated_episode_artifacts,
    headline,
    info,
    latest_matching_file,
    load_config,
    load_batch_jobs,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    read_json,
    rerun_in_runtime,
    release_mode_enabled,
    runtime_python,
    shared_worker_cli_args,
    shared_worker_id_for_args,
    shared_workers_enabled_for_args,
    update_batch_job_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate finished episodes in one run.")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of new episodes. Default: 1. Use 0 for endless generation.",
    )
    parser.add_argument(
        "--endless",
        action="store_true",
        help="Ignore --count and keep generating episodes until the run is stopped manually.",
    )
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Skip setup and foundation downloads in 00/08 and only use existing local assets.",
    )
    add_shared_worker_arguments(parser)
    return parser.parse_args()


def story_dir() -> Path:
    return PROJECT_ROOT / "generation" / "story_prompts"


def latest_episode_id() -> str | None:
    latest = latest_matching_file(story_dir(), "*.md")
    return latest.stem if latest else None


def run_step(
    script_name: str,
    env: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
    shared_args: list[str] | None = None,
) -> None:
    result = subprocess.run(
        [str(runtime_python()), str(SCRIPT_DIR / script_name), *(extra_args or []), *(shared_args or [])],
        env=env,
        cwd=str(SCRIPT_DIR),
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def training_stage_flags(cfg: dict) -> tuple[bool, bool, bool, bool, bool]:
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    adapter_cfg = cfg.get("adapter_training", {}) if isinstance(cfg.get("adapter_training"), dict) else {}
    fine_tune_cfg = cfg.get("fine_tune_training", {}) if isinstance(cfg.get("fine_tune_training"), dict) else {}
    backend_cfg = cfg.get("backend_fine_tune", {}) if isinstance(cfg.get("backend_fine_tune"), dict) else {}
    needs_foundation_training = bool(foundation_cfg.get("required_before_generate", True)) or bool(
        foundation_cfg.get("required_before_render", True)
    )
    should_prepare_foundation = bool(foundation_cfg.get("prepare_after_batch", False)) or needs_foundation_training
    should_train_foundation = bool(foundation_cfg.get("auto_train_after_prepare", False)) or needs_foundation_training
    should_train_adapter = bool(adapter_cfg.get("auto_train_after_foundation", True))
    should_train_fine_tune = bool(fine_tune_cfg.get("auto_train_after_adapter", True))
    should_run_backend = bool(backend_cfg.get("auto_run_after_fine_tune", True))
    return (
        should_prepare_foundation,
        should_train_foundation,
        should_train_adapter,
        should_train_fine_tune,
        should_run_backend,
    )


def training_plan_rows(cfg: dict, skip_downloads: bool = False) -> list[tuple[str, str, list[str]]]:
    (
        should_prepare_foundation,
        should_train_foundation,
        should_train_adapter,
        should_train_fine_tune,
        should_run_backend,
    ) = training_stage_flags(cfg)
    rows: list[tuple[str, str, list[str]]] = [
        ("06_build_dataset.py", "Build Datasets", []),
        ("07_train_series_model.py", "Train Series Model", []),
    ]
    if should_prepare_foundation:
        prepare_args: list[str] = ["--skip-downloads"] if skip_downloads else []
        rows.append(("08_prepare_foundation_training.py", "Prepare Foundation Data", prepare_args))
        if should_train_foundation:
            rows.append(("09_train_foundation_models.py", "Train Foundation Packs", []))
            if should_train_adapter:
                rows.append(("10_train_adapter_models.py", "Train Adapter Profiles", []))
                if should_train_fine_tune:
                    rows.append(("11_train_fine_tune_models.py", "Train Fine-Tune Profiles", []))
                    if should_run_backend:
                        rows.append(("49_run_backend_finetunes.py", "Prepare Backend Fine-Tunes", []))
    return rows


def quality_gate_extra_args(cfg: dict, episode_id: str | None = None) -> list[str]:
    release_cfg = cfg.get("release_mode", {}) if isinstance(cfg.get("release_mode"), dict) else {}
    args: list[str] = []
    if episode_id:
        args.extend(["--episode-id", episode_id])
    if bool(release_cfg.get("strict_warnings", False)):
        args.append("--strict")
    return args


def planned_preview_steps(cfg: dict, count: int) -> list[str]:
    return planned_preview_steps_for_mode(cfg, count, endless=False)


def planned_preview_steps_for_mode(cfg: dict, count: int, *, endless: bool) -> list[str]:
    steps = [row[0] for row in training_plan_rows(cfg, skip_downloads=False)]
    generation_cycle = [
        "12_generate_episode.py",
        "13_generate_storyboard_assets.py",
        "53_run_storyboard_backend.py",
        "14_render_episode.py",
    ]
    if release_mode_enabled(cfg):
        generation_cycle.append("51_quality_gate.py")
    if endless:
        steps.extend(generation_cycle)
        steps.append("15_build_series_bible.py")
        return steps
    steps.extend(generation_cycle * max(1, int(count)))
    steps.append("15_build_series_bible.py")
    return steps


def preview_endless_mode(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "endless", False)) or int(getattr(args, "count", 0) or 0) <= 0


def preview_parent_label(count: int, endless: bool) -> str:
    return "Endless" if endless else f"Count: {count}"


def preview_scope_total(count: int, endless: bool, generated_count: int) -> int:
    if endless:
        return max(1, int(generated_count) + 1)
    return max(1, int(count or 1))


def preview_reporter_total(training_rows: list[tuple[str, str, list[str]]], planned_steps: list[str], endless: bool) -> int:
    if endless:
        return max(1, len(training_rows) + 5)
    return max(1, len(planned_steps))


def preview_reporter_current(reporter: LiveProgressReporter, completed_steps: int) -> int:
    reporter_total = getattr(reporter, "total", None)
    if isinstance(reporter_total, (int, float)):
        return min(int(completed_steps), max(0, int(reporter_total) - 1))
    return max(0, int(completed_steps))


def completed_step_labels(planned_steps: list[str], completed_count: int) -> list[str]:
    limit = max(0, min(int(completed_count or 0), len(planned_steps)))
    return list(planned_steps[:limit])


def output_path_ready(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    candidate = Path(text)
    if candidate.exists() and candidate.is_file():
        return True
    try:
        candidate.resolve(strict=False).relative_to(PROJECT_ROOT.resolve(strict=False))
    except ValueError:
        return True
    except OSError:
        return False
    return False


def ensure_finished_episode_outputs(cfg: dict, episode_id: str, episode_outputs: dict[str, object]) -> dict[str, object]:
    core_required_fields = (
        "final_render",
        "full_generated_episode",
        "production_package",
        "render_manifest",
    )
    missing = [field for field in core_required_fields if not output_path_ready(episode_outputs.get(field))]
    delivery_required_fields = (
        "delivery_manifest",
        "delivery_episode",
    )
    missing_delivery = [field for field in delivery_required_fields if not output_path_ready(episode_outputs.get(field))]
    delivery_metadata_keys = (
        "delivery_bundle_root",
        "delivery_manifest",
        "delivery_episode",
        "delivery_summary",
        "latest_delivery_root",
        "latest_delivery_manifest",
        "latest_delivery_episode",
    )
    has_any_delivery_metadata = any(str(episode_outputs.get(field) or "").strip() for field in delivery_metadata_keys)
    if has_any_delivery_metadata:
        missing.extend(missing_delivery)
    if missing:
        raise RuntimeError(
            f"Episode {episode_id} did not produce a complete finished-episode bundle. Missing outputs: {', '.join(missing)}"
        )
    backend_runner_expected_count = int(episode_outputs.get("backend_runner_expected_count", 0) or 0)
    backend_runner_ready_count = int(episode_outputs.get("backend_runner_ready_count", 0) or 0)
    backend_runner_failed_count = int(episode_outputs.get("backend_runner_failed_count", 0) or 0)
    backend_runner_pending_count = int(episode_outputs.get("backend_runner_pending_count", 0) or 0)
    backend_runner_status = str(episode_outputs.get("backend_runner_status", "") or "").strip()
    if backend_runner_failed_count > 0:
        failure_scenes = episode_outputs.get("runner_failure_scenes", [])
        failure_hint = f" Failed scenes: {', '.join(failure_scenes)}." if isinstance(failure_scenes, list) and failure_scenes else ""
        raise RuntimeError(
            f"Episode {episode_id} finished with failed external backend runners.{failure_hint}"
        )
    if backend_runner_expected_count > 0 and backend_runner_ready_count < backend_runner_expected_count:
        pending_scenes = episode_outputs.get("runner_pending_scenes", [])
        pending_hint = f" Pending scenes: {', '.join(pending_scenes)}." if isinstance(pending_scenes, list) and pending_scenes else ""
        raise RuntimeError(
            f"Episode {episode_id} still has incomplete external backend runners "
            f"({backend_runner_ready_count}/{backend_runner_expected_count}, status: {backend_runner_status or 'unknown'}).{pending_hint}"
        )
    if backend_runner_expected_count > 0 and backend_runner_pending_count > 0:
        raise RuntimeError(
            f"Episode {episode_id} still has pending external backend runner tasks "
            f"({backend_runner_pending_count} remaining)."
        )
    if not bool(episode_outputs.get("release_gate_passed", False)):
        raise RuntimeError(
            f"Episode {episode_id} finished without a passing release gate."
        )
    if str(episode_outputs.get("production_readiness", "") or "").strip().lower() != "fully_generated_episode_ready":
        raise RuntimeError(
            f"Episode {episode_id} is not fully generated yet "
            f"(readiness: {episode_outputs.get('production_readiness') or 'unknown'})."
        )
    if int(episode_outputs.get("scenes_below_release_threshold", 0) or 0) > 0:
        raise RuntimeError(
            f"Episode {episode_id} still has scenes below release threshold "
            f"({int(episode_outputs.get('scenes_below_release_threshold', 0) or 0)})."
        )
    if list(episode_outputs.get("remaining_backend_tasks", []) or []):
        raise RuntimeError(
            f"Episode {episode_id} still lists remaining backend tasks: "
            f"{', '.join(str(task) for task in (episode_outputs.get('remaining_backend_tasks', []) or []))}"
        )
    render_manifest_path = Path(str(episode_outputs.get("render_manifest", "") or "").strip())
    production_package_path = Path(str(episode_outputs.get("production_package", "") or "").strip())
    render_manifest = read_json(render_manifest_path, {}) if render_manifest_path.exists() else {}
    production_package = read_json(production_package_path, {}) if production_package_path.exists() else {}
    audio_track_meta = render_manifest.get("audio_track_meta", {}) if isinstance(render_manifest.get("audio_track_meta", {}), dict) else {}
    audio_backend = str(
        audio_track_meta.get("audio_backend", "") or audio_track_meta.get("backend", "") or ""
    ).strip().lower()
    if audio_backend in {"pyttsx3", "mixed_original_segment_and_pyttsx3"}:
        raise RuntimeError(
            f"Episode {episode_id} still uses fallback dialogue audio backend '{audio_backend}'."
        )
    scenes = production_package.get("scenes", []) if isinstance(production_package.get("scenes", []), list) else []
    placeholder_scenes: list[str] = []
    local_fallback_scenes: list[str] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene.get("scene_id", "") or "").strip() or "scene"
        preview_assets = scene.get("current_preview_assets", {}) if isinstance(scene.get("current_preview_assets", {}), dict) else {}
        current_outputs = scene.get("current_generated_outputs", {}) if isinstance(scene.get("current_generated_outputs", {}), dict) else {}
        if str(preview_assets.get("asset_source_type", "") or "").strip().lower() == "placeholder":
            placeholder_scenes.append(scene_id)
        if bool(current_outputs.get("local_composed_scene_video", False)):
            local_fallback_scenes.append(scene_id)
    if placeholder_scenes:
        raise RuntimeError(
            f"Episode {episode_id} still contains placeholder source scenes: {', '.join(placeholder_scenes[:8])}"
        )
    if local_fallback_scenes:
        raise RuntimeError(
            f"Episode {episode_id} still contains locally composed fallback scene videos: {', '.join(local_fallback_scenes[:8])}"
        )
    return episode_outputs


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    endless = preview_endless_mode(args)
    count = 0 if endless else max(1, int(args.count))
    cfg = load_config()
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)
    child_shared_args = shared_worker_cli_args(cfg, args)
    training_rows = training_plan_rows(cfg, skip_downloads=bool(args.skip_downloads))

    headline("Generate Multiple Finished Episodes")
    generated: list[str] = []
    generated_outputs: list[dict[str, object]] = []
    autosave_target = "endless" if endless else f"count_{count}"
    if shared_workers:
        info(f"Shared NAS workers: enabled ({worker_id})")
    mark_step_started(
        "56_generate_finished_episodes",
        autosave_target,
        {"requested_count": count, "endless": endless, "skip_downloads": bool(args.skip_downloads)},
    )
    lease_manager = distributed_item_lease(
        root=distributed_step_runtime_root("56_generate_finished_episodes", autosave_target),
        lease_name=autosave_target,
        cfg=cfg,
        worker_id=worker_id,
        enabled=shared_workers,
        meta={"step": "56_generate_finished_episodes", "scope": autosave_target, "worker_id": worker_id},
    )
    acquired = lease_manager.__enter__()
    if not acquired:
        info("This finished-episode batch is already being processed by another worker.")
        lease_manager.__exit__(None, None, None)
        return
    try:
        review_count = open_face_review_item_count(cfg)
        if review_count > 0:
            raise RuntimeError(
                f"There are still {review_count} open face review cases. "
                "Run 05_review_unknowns.py first before training, generation, or render can start."
            )
        setup_args = ["--skip-downloads"] if bool(args.skip_downloads) else []
        run_step("00_prepare_runtime.py", extra_args=setup_args, shared_args=child_shared_args)
        cfg = load_config()
        ensure_quality_first_ready(cfg, context_label="56_generate_finished_episodes.py")
        planned_steps = planned_preview_steps_for_mode(cfg, count, endless=endless)
        reporter = LiveProgressReporter(
            script_name="56_generate_finished_episodes.py",
            total=preview_reporter_total(training_rows, planned_steps, endless),
            phase_label="Generate Finished Episodes",
            parent_label=preview_parent_label(count, endless),
        )
        completed_steps = 0

        for script_name, current_label, extra_args in training_rows:
            reporter.update(completed_steps, current_label=current_label, extra_label=f"Running now: {script_name}", force=True)
            run_step(script_name, extra_args=extra_args, shared_args=child_shared_args)
            completed_steps += 1
            reporter.update(completed_steps, current_label=current_label, extra_label=f"Completed: {script_name}")
        index = 0
        while endless or index < count:
            before = latest_episode_id()
            reporter.update(
                preview_reporter_current(reporter, completed_steps),
                current_label=f"Generate episode {index + 1}",
                extra_label="Running now: 12_generate_episode.py",
                force=True,
                scope_current=len(generated),
                scope_total=preview_scope_total(count, endless, len(generated)),
                scope_label="Episodes",
            )
            run_step("12_generate_episode.py", shared_args=child_shared_args)
            completed_steps += 1
            episode_id = latest_episode_id()
            if not episode_id or episode_id == before:
                raise RuntimeError("Could not determine the new episode.")
            reporter.update(
                preview_reporter_current(reporter, completed_steps),
                current_label=episode_id,
                extra_label=f"Episode generated: {episode_id}",
                scope_current=len(generated) + 1,
                scope_total=preview_scope_total(count, endless, len(generated)),
                scope_label="Episodes",
            )
            env = os.environ.copy()
            env["SERIES_STORYBOARD_EPISODE"] = episode_id
            reporter.update(
                preview_reporter_current(reporter, completed_steps),
                current_label=f"{episode_id} storyboard assets",
                extra_label="Running now: 13_generate_storyboard_assets.py",
                force=True,
                scope_current=len(generated) + 1,
                scope_total=preview_scope_total(count, endless, len(generated)),
                scope_label="Episodes",
            )
            run_step("13_generate_storyboard_assets.py", env=env, shared_args=child_shared_args)
            completed_steps += 1
            reporter.update(
                preview_reporter_current(reporter, completed_steps),
                current_label=episode_id,
                extra_label=f"Storyboard assets ready: {episode_id}",
                scope_current=len(generated) + 1,
                scope_total=preview_scope_total(count, endless, len(generated)),
                scope_label="Episodes",
            )
            reporter.update(
                preview_reporter_current(reporter, completed_steps),
                current_label=f"{episode_id} backend frames",
                extra_label="Running now: 53_run_storyboard_backend.py",
                force=True,
                scope_current=len(generated) + 1,
                scope_total=preview_scope_total(count, endless, len(generated)),
                scope_label="Episodes",
            )
            run_step("53_run_storyboard_backend.py", env=env, shared_args=child_shared_args)
            completed_steps += 1
            reporter.update(
                preview_reporter_current(reporter, completed_steps),
                current_label=episode_id,
                extra_label=f"Storyboard backend frames ready: {episode_id}",
                scope_current=len(generated) + 1,
                scope_total=preview_scope_total(count, endless, len(generated)),
                scope_label="Episodes",
            )
            env = os.environ.copy()
            env["SERIES_RENDER_EPISODE"] = episode_id
            reporter.update(
                preview_reporter_current(reporter, completed_steps),
                current_label=f"{episode_id} render",
                extra_label="Running now: 14_render_episode.py",
                force=True,
                scope_current=len(generated) + 1,
                scope_total=preview_scope_total(count, endless, len(generated)),
                scope_label="Episodes",
            )
            run_step("14_render_episode.py", env=env, shared_args=child_shared_args)
            completed_steps += 1
            reporter.update(
                preview_reporter_current(reporter, completed_steps),
                current_label=f"{episode_id} quality gate",
                extra_label="Running now: 51_quality_gate.py",
                force=True,
                scope_current=len(generated) + 1,
                scope_total=preview_scope_total(count, endless, len(generated)),
                scope_label="Episodes",
            )
            run_step(
                "51_quality_gate.py",
                extra_args=quality_gate_extra_args(cfg, episode_id),
                shared_args=child_shared_args,
            )
            completed_steps += 1
            episode_outputs = generated_episode_artifacts(cfg, episode_id)
            if episode_outputs:
                generated_outputs.append(ensure_finished_episode_outputs(cfg, episode_id, episode_outputs))
            else:
                raise RuntimeError(f"Episode {episode_id} finished rendering without a recorded output bundle.")
            reporter.update(
                preview_reporter_current(reporter, completed_steps),
                current_label=episode_id,
                extra_label=(
                    f"Finished episode ready: {episode_id} | "
                    f"readiness={episode_outputs.get('production_readiness', '-') or '-'} | "
                    f"quality={episode_outputs.get('quality_label', '-') or '-'}:"
                    f"{int(round(float(episode_outputs.get('quality_percent', 0.0) or 0.0)))}% | "
                    f"gate={'pass' if episode_outputs.get('release_gate_passed') else 'fail'} | "
                    f"runners={int(episode_outputs.get('backend_runner_ready_count', 0) or 0)}/"
                    f"{int(episode_outputs.get('backend_runner_expected_count', 0) or 0)}"
                ),
                scope_current=len(generated) + 1,
                scope_total=preview_scope_total(count, endless, len(generated)),
                scope_label="Episodes",
            )
            generated.append(episode_id)
            jobs = load_batch_jobs(PROJECT_ROOT)
            for job in jobs:
                if job.get("type") == "generate_episode" and job.get("status") == "running":
                    if job.get("config", {}).get("episode_id") == episode_id:
                        update_batch_job_status(PROJECT_ROOT, job.get("id", ""), "completed")
                        break
            if endless:
                reporter.update(
                    preview_reporter_current(reporter, completed_steps),
                    current_label="Update series bible",
                    extra_label="Running now: 15_build_series_bible.py",
                    force=True,
                    scope_current=len(generated),
                    scope_total=preview_scope_total(count, endless, len(generated)),
                    scope_label="Episodes",
                )
                run_step("15_build_series_bible.py", shared_args=child_shared_args)
                completed_steps += 1
                reporter.update(
                    preview_reporter_current(reporter, completed_steps),
                    current_label="Series bible updated",
                    extra_label=f"Endless mode active. Episodes generated so far: {len(generated)}",
                    scope_current=len(generated),
                    scope_total=preview_scope_total(count, endless, len(generated)),
                    scope_label="Episodes",
                )
                ok(f"{len(generated)} episodes generated so far. Latest: {episode_id}")
            else:
                ok(f"{index + 1}/{count}: {episode_id} generated and rendered.")
            index += 1
        if not endless:
            reporter.update(completed_steps, current_label="Update series bible", extra_label="Running now: 15_build_series_bible.py", force=True)
            run_step("15_build_series_bible.py", shared_args=child_shared_args)
            completed_steps += 1
            reporter.update(completed_steps, current_label="Update series bible", extra_label="Completed: 15_build_series_bible.py")
            reporter.finish(current_label="Finished Episodes", extra_label=f"Total episodes: {len(generated)}")
        mark_step_completed(
                "56_generate_finished_episodes",
            autosave_target,
            {
                "requested_count": count,
                "endless": endless,
                "skip_downloads": bool(args.skip_downloads),
                "planned_steps": planned_steps,
                "completed_steps": completed_step_labels(planned_steps, completed_steps),
                "generated_episodes": generated,
                "generated_episode_outputs": generated_outputs,
                "latest_generated_episode": generated_outputs[-1] if generated_outputs else {},
                "generated_count": len(generated),
                "bible_updated_once_after_batch": not endless,
                "bible_updated_after_each_episode": endless,
            },
        )
        ok(f"Done. New visible episodes: {', '.join(generated)}")
    except KeyboardInterrupt:
        if generated and endless:
            info_message = f"Endless generation interrupted after {len(generated)} episodes. Updating the series bible one last time."
            headline(info_message)
            run_step("15_build_series_bible.py", shared_args=child_shared_args)
        mark_step_failed(
                "56_generate_finished_episodes",
            "Interrupted by user",
            autosave_target,
            {
                "requested_count": count,
                "endless": endless,
                "skip_downloads": bool(args.skip_downloads),
                "planned_steps": planned_steps if "planned_steps" in locals() else [],
                "completed_steps": completed_step_labels(planned_steps, completed_steps)
                if "planned_steps" in locals()
                else [],
                "generated_episodes": generated,
                "generated_episode_outputs": generated_outputs,
                "latest_generated_episode": generated_outputs[-1] if generated_outputs else {},
                "generated_count": len(generated),
            },
        )
        raise
    except Exception as exc:
        mark_step_failed(
                "56_generate_finished_episodes",
            str(exc),
            autosave_target,
            {
                "requested_count": count,
                "endless": endless,
                "skip_downloads": bool(args.skip_downloads),
                "planned_steps": planned_steps if "planned_steps" in locals() else [],
                "completed_steps": completed_step_labels(planned_steps, completed_steps)
                if "planned_steps" in locals()
                else [],
                "generated_episodes": generated,
                "generated_episode_outputs": generated_outputs,
                "latest_generated_episode": generated_outputs[-1] if generated_outputs else {},
                "generated_count": len(generated),
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


