#!/usr/bin/env python3
from __future__ import annotations

import shutil

from pipeline_common import (
    error,
    headline,
    info,
    load_config,
    mark_status,
    next_unprocessed_video,
    ok,
    rerun_in_runtime,
    resolve_project_path,
    write_json,
)


def main() -> None:
    rerun_in_runtime()
    headline("Neue Folge importieren")
    cfg = load_config()
    inbox_dir = resolve_project_path(cfg["paths"]["inbox_episodes"])
    episodes_dir = resolve_project_path(cfg["paths"]["episodes"])
    metadata_dir = resolve_project_path(cfg["paths"]["metadata"])
    inbox_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    video = next_unprocessed_video(inbox_dir)
    if video is None:
        info("Keine neue Folge im Inbox-Ordner gefunden.")
        return

    destination = episodes_dir / video.name
    shutil.copy2(video, destination)
    mark_status(video, "importiert", {"episode_id": video.stem})
    write_json(
        metadata_dir / f"{video.stem}.json",
        {
            "episode_id": video.stem,
            "source_file_inbox": str(video),
            "working_file": str(destination),
        },
    )
    ok(f"Episode importiert: {video.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
