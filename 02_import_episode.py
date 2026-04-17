#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pipeline_common import (
    error,
    headline,
    info,
    list_videos,
    load_config,
    load_registry,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    mark_status,
    next_unprocessed_video,
    ok,
    rerun_in_runtime,
    resolve_project_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neue Folgen aus dem Inbox-Ordner importieren")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Importiert alle aktuell im Inbox-Ordner liegenden Folgen statt nur der naechsten.",
    )
    return parser.parse_args()


def delete_inbox_after_verified_copy(source: Path, destination: Path) -> bool:
    if not source.exists() or not destination.exists():
        return False
    if source.stat().st_size != destination.stat().st_size:
        raise RuntimeError(f"Kopie unvollstaendig, Inbox-Datei bleibt erhalten: {source.name}")
    source.unlink()
    return True


def inbox_video_is_already_covered(
    video: Path,
    episodes_dir: Path,
    scene_root: Path,
    metadata_dir: Path,
    registry: dict,
) -> bool:
    working_copy = episodes_dir / video.name
    if working_copy.exists() and working_copy.stat().st_size == video.stat().st_size:
        return True
    if (scene_root / video.stem).exists():
        return True
    if (metadata_dir / f"{video.stem}.json").exists():
        return True
    for payload in (registry.get("files", {}) or {}).values():
        if str(payload.get("filename", "")).strip() == video.name and str(payload.get("status", "")).strip():
            return True
    return False


def purge_already_processed_inbox_videos(
    inbox_dir: Path,
    episodes_dir: Path,
    scene_root: Path,
    metadata_dir: Path,
) -> int:
    registry = load_registry()
    removed = 0
    for video in sorted(inbox_dir.glob("*")):
        if not video.is_file():
            continue
        if not inbox_video_is_already_covered(video, episodes_dir, scene_root, metadata_dir, registry):
            continue
        video.unlink()
        removed += 1
    return removed


def import_single_episode(video: Path, episodes_dir: Path, metadata_dir: Path) -> bool:
    source_file_inbox = str(video)
    destination = episodes_dir / video.name
    autosave_target = video.stem
    mark_step_started(
        "02_import_episode",
        autosave_target,
        {"episode_id": video.stem, "source_file_inbox": source_file_inbox, "working_file": str(destination)},
    )
    try:
        shutil.copy2(video, destination)
        mark_status(video, "importiert", {"episode_id": video.stem})
        removed_inbox = delete_inbox_after_verified_copy(video, destination)
        write_json(
            metadata_dir / f"{video.stem}.json",
            {
                "episode_id": video.stem,
                "source_file_inbox": source_file_inbox,
                "working_file": str(destination),
            },
        )
        mark_step_completed(
            "02_import_episode",
            autosave_target,
            {
                "episode_id": video.stem,
                "source_file_inbox": source_file_inbox,
                "working_file": str(destination),
                "inbox_deleted": bool(removed_inbox),
            },
        )
        ok(f"Episode importiert: {video.name}")
        if removed_inbox:
            info(f"Inbox-Datei geloescht: {video.name}")
        return True
    except Exception as exc:
        mark_step_failed(
            "02_import_episode",
            str(exc),
            autosave_target,
            {"episode_id": video.stem, "source_file_inbox": source_file_inbox, "working_file": str(destination)},
        )
        raise


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Neue Folge importieren")
    cfg = load_config()
    inbox_dir = resolve_project_path(cfg["paths"]["inbox_episodes"])
    episodes_dir = resolve_project_path(cfg["paths"]["episodes"])
    metadata_dir = resolve_project_path(cfg["paths"]["metadata"])
    scene_root = resolve_project_path(cfg["paths"]["scene_clips"])
    inbox_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    scene_root.mkdir(parents=True, exist_ok=True)

    removed_duplicates = purge_already_processed_inbox_videos(inbox_dir, episodes_dir, scene_root, metadata_dir)
    if removed_duplicates:
        info(f"{removed_duplicates} bereits bearbeitete Inbox-Dateien entfernt.")

    imported = 0
    if args.all:
        pending_videos = list_videos(inbox_dir)
        if not pending_videos:
            info("Keine neue Folge im Inbox-Ordner gefunden.")
            return
        for video in pending_videos:
            if not video.exists():
                continue
            import_single_episode(video, episodes_dir, metadata_dir)
            imported += 1
        ok(f"Import abgeschlossen: {imported} Folgen importiert.")
        return

    video = next_unprocessed_video(inbox_dir)
    if video is None:
        info("Keine neue Folge im Inbox-Ordner gefunden.")
        return
    import_single_episode(video, episodes_dir, metadata_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
