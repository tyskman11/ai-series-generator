#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

from pipeline_common import (
    add_shared_worker_arguments,
    distributed_item_lease,
    distributed_step_runtime_root,
    LiveProgressReporter,
    completed_step_state,
    error,
    extract_keywords,
    first_dir,
    headline,
    info,
    load_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    progress,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
    save_step_autosave,
    shared_worker_id_for_args,
    shared_workers_enabled_for_args,
    write_json,
)

PROCESS_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Training Dataset")
    parser.add_argument("--episode", help="Name of the scene folder under data/processed/scene_clips.")
    parser.add_argument("--force", action="store_true", help="Rebuild existing datasets intentionally.")
    add_shared_worker_arguments(parser)
    return parser.parse_args()


def episode_dataset_completed(episode_dir: Path, cfg: dict) -> bool:
    dataset_file = resolve_project_path(cfg["paths"]["datasets_video_training"]) / f"{episode_dir.name}_dataset.json"
    manifest_file = resolve_project_path(cfg["paths"]["datasets_video_training"]) / f"{episode_dir.name}_dataset_manifest.json"
    if not dataset_file.exists():
        return False
    try:
        rows = read_json(dataset_file, [])
        manifest = read_json(manifest_file, {}) if manifest_file.exists() else {}
    except Exception:
        return False
    if not isinstance(rows, list) or len(rows) <= 0:
        return False
    autosave_state = completed_step_state("07_build_dataset", episode_dir.name, PROCESS_VERSION)
    if manifest:
        if int(manifest.get("process_version", 0) or 0) != PROCESS_VERSION:
            return False
        if int(manifest.get("scene_count", 0) or 0) != len(rows):
            return False
    if autosave_state:
        if int(autosave_state.get("scene_count", 0) or 0) not in {0, len(rows)}:
            return False
    return True


def next_undataseted_episode_dir(scene_root: Path, cfg: dict) -> Path | None:
    for folder in sorted(scene_root.glob("*")):
        if not folder.is_dir():
            continue
        if not episode_dataset_completed(folder, cfg):
            return folder
    return first_dir(scene_root)


def pending_undataseted_episode_dirs(scene_root: Path, cfg: dict) -> list[Path]:
    pending = []
    for folder in sorted(scene_root.glob("*")):
        if not folder.is_dir():
            continue
        if not episode_dataset_completed(folder, cfg):
            pending.append(folder)
    return pending


def resolve_episode_dir(scene_root: Path, episode_name: str | None, cfg: dict) -> Path | None:
    if episode_name:
        candidate = scene_root / Path(episode_name).name
        if candidate.is_dir():
            return candidate
        for folder in sorted(scene_root.glob("*")):
            if folder.is_dir() and folder.name == Path(episode_name).stem:
                return folder
        raise FileNotFoundError(f"Scene folder not found: {episode_name}")
    return next_undataseted_episode_dir(scene_root, cfg)


def resolve_episode_dirs_for_processing(scene_root: Path, episode_name: str | None, cfg: dict, force: bool = False) -> list[Path]:
    if episode_name:
        episode_dir = resolve_episode_dir(scene_root, episode_name, cfg)
        return [episode_dir] if episode_dir is not None else []
    if force:
        return [folder for folder in sorted(scene_root.glob("*")) if folder.is_dir()]
    return pending_undataseted_episode_dirs(scene_root, cfg)


def unique_preserve(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def process_episode_dir(
    episode_dir: Path,
    cfg: dict,
    force: bool = False,
    live_reporter: LiveProgressReporter | None = None,
    episode_index: int = 1,
    episode_total: int = 1,
) -> bool:
    autosave_target = episode_dir.name
    if not force and episode_dataset_completed(episode_dir, cfg):
        mark_step_completed(
            "07_build_dataset",
            autosave_target,
            {
                "episode_id": episode_dir.name,
                "process_version": PROCESS_VERSION,
                "dataset_file": str(resolve_project_path(cfg["paths"]["datasets_video_training"]) / f"{episode_dir.name}_dataset.json"),
                "dataset_manifest": str(resolve_project_path(cfg["paths"]["datasets_video_training"]) / f"{episode_dir.name}_dataset_manifest.json"),
            },
        )
        ok(f"Dataset already exists: {episode_dir.name}")
        return False

    linked_rows = read_json(
        resolve_project_path(cfg["paths"]["linked_segments"]) / f"{episode_dir.name}_linked_segments.json",
        [],
    )
    if not linked_rows:
        info("No linked segments found.")
        return False

    faces_summary = {
        row["scene_id"]: row.get("face_clusters", [])
        for row in read_json(
            resolve_project_path(cfg["paths"]["faces"]) / episode_dir.name / f"{episode_dir.name}_face_summary.json",
            [],
        )
    }

    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in linked_rows:
        grouped_rows[row["scene_id"]].append(row)

    dataset_rows = []
    scene_ids = sorted(grouped_rows)
    mark_step_started(
        "07_build_dataset",
        autosave_target,
        {
            "episode_id": episode_dir.name,
            "process_version": PROCESS_VERSION,
            "scene_count": len(scene_ids),
            "linked_segment_count": len(linked_rows),
        },
    )
    processed_scene_ids: list[str] = []
    try:
        episode_started_at = time.time()
        for index, scene_id in enumerate(scene_ids, start=1):
            scene_segments = sorted(grouped_rows[scene_id], key=lambda row: (float(row["start"]), row["segment_id"]))
            transcript = " ".join(segment["text"] for segment in scene_segments if segment.get("text"))
            speaker_names = unique_preserve([segment.get("speaker_name", "") for segment in scene_segments])
            character_names = unique_preserve(
                [
                    name
                    for segment in scene_segments
                    for name in segment.get("visible_character_names", [])
                ]
            )
            dataset_rows.append(
                {
                    "episode_id": episode_dir.name,
                    "scene_id": scene_id,
                    "video_file": str(episode_dir / f"{scene_id}.mp4"),
                    "duration_seconds": round(
                        max(float(segment["end"]) for segment in scene_segments)
                        - min(float(segment["start"]) for segment in scene_segments),
                        3,
                    ),
                    "transcript": transcript.strip(),
                    "transcript_segments": scene_segments,
                    "speaker_names": speaker_names,
                    "speaker_clusters": unique_preserve([segment.get("speaker_cluster", "") for segment in scene_segments]),
                    "characters_visible": character_names,
                    "face_clusters": faces_summary.get(scene_id, []),
                    "scene_keywords": extract_keywords([transcript], limit=8),
                }
            )
            processed_scene_ids.append(scene_id)
            save_step_autosave(
                "07_build_dataset",
                autosave_target,
                {
                    "status": "in_progress",
                    "episode_id": episode_dir.name,
                    "process_version": PROCESS_VERSION,
                    "scene_count": len(scene_ids),
                    "linked_segment_count": len(linked_rows),
                    "processed_scene_ids": processed_scene_ids,
                    "dataset_row_count": len(dataset_rows),
                    "last_scene_id": scene_id,
                },
            )
            if live_reporter is not None:
                live_reporter.update(
                    (episode_index - 1) + (index / max(1, len(scene_ids))),
                    current_label=scene_id,
                    parent_label=episode_dir.name,
                    extra_label=f"Dataset rows so far: {len(dataset_rows)}",
                    scope_current=index,
                    scope_total=len(scene_ids),
                    scope_started_at=episode_started_at,
                    scope_label=f"Episode {episode_index}/{episode_total}",
                )

        dataset_root = resolve_project_path(cfg["paths"]["datasets_video_training"])
        dataset_root.mkdir(parents=True, exist_ok=True)
        dataset_file = dataset_root / f"{episode_dir.name}_dataset.json"
        manifest_file = dataset_root / f"{episode_dir.name}_dataset_manifest.json"
        write_json(dataset_file, dataset_rows)
        write_json(
            manifest_file,
            {
                "episode_id": episode_dir.name,
                "process_version": PROCESS_VERSION,
                "scene_count": len(dataset_rows),
                "speaker_count": len({speaker for row in dataset_rows for speaker in row["speaker_names"]}),
                "character_count": len({name for row in dataset_rows for name in row["characters_visible"]}),
            },
        )
        mark_step_completed(
            "07_build_dataset",
            autosave_target,
            {
                "episode_id": episode_dir.name,
                "process_version": PROCESS_VERSION,
                "scene_count": len(dataset_rows),
                "linked_segment_count": len(linked_rows),
                "processed_scene_ids": processed_scene_ids,
                "dataset_file": str(dataset_file),
                "dataset_manifest": str(manifest_file),
            },
        )
        ok(f"Dataset created: {len(dataset_rows)} scenes")
        return True
    except Exception as exc:
        mark_step_failed(
            "07_build_dataset",
            str(exc),
            autosave_target,
            {
                "episode_id": episode_dir.name,
                "process_version": PROCESS_VERSION,
                "scene_count": len(scene_ids),
                "linked_segment_count": len(linked_rows),
                "processed_scene_ids": processed_scene_ids,
            },
        )
        raise


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Build Training Dataset")
    cfg = load_config()
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)
    scene_root = resolve_project_path(cfg["paths"]["scene_clips"])
    episode_dirs = resolve_episode_dirs_for_processing(scene_root, args.episode, cfg, args.force)
    if not episode_dirs:
        if args.episode:
            info("No matching scene folders found.")
        else:
            info("No pending episodes found for step 07.")
        return

    processed_count = 0
    total = len(episode_dirs)
    lease_root = distributed_step_runtime_root("07_build_dataset", "episodes")
    if shared_workers:
        info(f"Shared NAS workers: enabled ({worker_id})")
    live_reporter = LiveProgressReporter(
        script_name="07_build_dataset.py",
        total=max(1, total),
        phase_label="Build Datasets",
        parent_label="Batch",
    )
    for index, episode_dir in enumerate(episode_dirs, start=1):
        with distributed_item_lease(
            root=lease_root,
            lease_name=episode_dir.name,
            cfg=cfg,
            worker_id=worker_id,
            enabled=shared_workers,
            meta={"step": "07_build_dataset", "episode_id": episode_dir.name, "worker_id": worker_id},
        ) as acquired:
            if not acquired:
                continue
            if process_episode_dir(
                episode_dir,
                cfg,
                force=args.force,
                live_reporter=live_reporter,
                episode_index=index,
                episode_total=total,
            ):
                processed_count += 1
                live_reporter.update(
                    index,
                    current_label=episode_dir.name,
                    parent_label=episode_dir.name,
                    extra_label=f"Episode completed: {episode_dir.name}",
                    scope_current=1,
                    scope_total=1,
                    scope_started_at=time.time(),
                    scope_label=f"Episode {index}/{total}",
                )

    live_reporter.finish(current_label="Batch", extra_label=f"Episodes processed: {processed_count}")
    ok(f"Batch completed: {processed_count} episodes processed in 07.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise

