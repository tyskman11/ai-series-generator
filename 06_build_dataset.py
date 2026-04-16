#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from pipeline_common import (
    error,
    extract_keywords,
    first_dir,
    headline,
    info,
    load_config,
    ok,
    progress,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trainingsdatensatz bauen")
    parser.add_argument("--episode", help="Name des Szenenordners unter data/processed/scene_clips.")
    return parser.parse_args()


def resolve_episode_dir(scene_root: Path, episode_name: str | None) -> Path | None:
    if episode_name:
        candidate = scene_root / Path(episode_name).name
        if candidate.is_dir():
            return candidate
        for folder in sorted(scene_root.glob("*")):
            if folder.is_dir() and folder.name == Path(episode_name).stem:
                return folder
        raise FileNotFoundError(f"Szenenordner nicht gefunden: {episode_name}")
    return first_dir(scene_root)


def unique_preserve(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Trainingsdatensatz bauen")
    cfg = load_config()
    scene_root = resolve_project_path(cfg["paths"]["scene_clips"])
    episode_dir = resolve_episode_dir(scene_root, args.episode)
    if episode_dir is None:
        info("Keine Szenenordner gefunden.")
        return

    linked_rows = read_json(
        resolve_project_path(cfg["paths"]["linked_segments"]) / f"{episode_dir.name}_linked_segments.json",
        [],
    )
    if not linked_rows:
        info("Keine verknüpften Segmente gefunden.")
        return

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
        progress(index, len(scene_ids), "Datensatz wird aufgebaut")

    dataset_root = resolve_project_path(cfg["paths"]["datasets_video_training"])
    dataset_root.mkdir(parents=True, exist_ok=True)
    write_json(dataset_root / f"{episode_dir.name}_dataset.json", dataset_rows)
    write_json(
        dataset_root / f"{episode_dir.name}_dataset_manifest.json",
        {
            "episode_id": episode_dir.name,
            "scene_count": len(dataset_rows),
            "speaker_count": len({speaker for row in dataset_rows for speaker in row["speaker_names"]}),
            "character_count": len({name for row in dataset_rows for name in row["characters_visible"]}),
        },
    )
    ok(f"Datensatz erstellt: {len(dataset_rows)} Szenen")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
