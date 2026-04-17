#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pipeline_common import (
    LiveProgressReporter,
    PROJECT_ROOT,
    detect_tool,
    error,
    ffmpeg_video_encode_args,
    headline,
    info,
    list_videos,
    limited_items,
    load_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    mark_status,
    ok,
    preferred_ffmpeg_video_codec,
    progress,
    rerun_in_runtime,
    resolve_project_path,
    run_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Folge in Szenen zerlegen")
    parser.add_argument("--episode-file", help="Dateiname oder Pfad der importierten Arbeitsdatei unter data/raw/episodes.")
    return parser.parse_args()


def pending_episode_files(episodes_dir: Path) -> list[Path]:
    return [video for video in list_videos(episodes_dir) if video.is_file()]


def resolve_episode_file(episodes_dir: Path, scene_root: Path, episode_file: str | None) -> Path | None:
    if episode_file:
        requested_name = Path(episode_file).name
        candidate = Path(episode_file)
        if not candidate.is_absolute():
            candidate = episodes_dir / requested_name
        if candidate.is_file():
            return candidate
        for video in sorted(episodes_dir.glob("*")):
            if not video.is_file():
                continue
            if requested_name in {video.name, video.stem}:
                return video
        raise FileNotFoundError(f"Importierte Arbeitsdatei nicht gefunden: {episode_file}")
    pending = pending_episode_files(episodes_dir)
    return pending[0] if pending else None


def export_scene(
    ffmpeg: Path,
    episode: Path,
    out_dir: Path,
    start_sec: float,
    end_sec: float,
    index: int,
    video_codec: str,
) -> None:
    duration = max(0.1, end_sec - start_sec)
    run_command(
        [
            str(ffmpeg),
            "-hide_banner",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(episode),
            "-t",
            f"{duration:.3f}",
            *ffmpeg_video_encode_args(video_codec, quality=18),
            "-c:a",
            "aac",
            str(out_dir / f"scene_{index:04d}.mp4"),
        ],
        quiet=True,
    )


def detect_scenes_with_py_scene_detect(episode: Path, threshold: float) -> list[tuple[float, float]]:
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except Exception:
        return []

    video = open_video(str(episode))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold * 100.0))
    manager.detect_scenes(video=video)
    scenes = []
    for start_time, end_time in manager.get_scene_list():
        scenes.append((start_time.get_seconds(), end_time.get_seconds()))
    return scenes


def cleanup_split_working_file(episode: Path) -> bool:
    if not episode.exists():
        return False
    episode.unlink()
    return True


def split_success_marker_path(scene_index_root: Path, episode_stem: str) -> Path:
    return scene_index_root / f"{episode_stem}_split_success.json"


def write_split_success_marker(
    marker_path: Path,
    episode: Path,
    clip_count: int,
    scene_csv: Path | None,
    mode: str,
) -> None:
    payload = {
        "episode_id": episode.stem,
        "episode_file": str(episode),
        "clip_count": int(clip_count),
        "scene_csv": str(scene_csv) if scene_csv else "",
        "mode": mode,
    }
    marker_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_split_success_marker(marker_path: Path) -> dict:
    if not marker_path.exists():
        return {}
    try:
        return json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def csv_scene_count(scene_csv: Path) -> int:
    if not scene_csv.exists():
        return 0
    try:
        lines = [line.strip() for line in scene_csv.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return 0
    if len(lines) <= 1:
        return 0
    return max(0, len(lines) - 1)


def split_was_completed_successfully(out_dir: Path, scene_csv: Path, marker_path: Path) -> tuple[bool, int, str]:
    existing = sorted(out_dir.glob("scene_*.mp4")) if out_dir.exists() else []
    clip_count = len(existing)
    if clip_count <= 0:
        return False, 0, ""

    marker = load_split_success_marker(marker_path)
    marker_count = int(marker.get("clip_count", 0) or 0)
    if marker_count > 0 and clip_count >= marker_count:
        return True, clip_count, "marker"

    csv_count = csv_scene_count(scene_csv)
    if csv_count > 0 and clip_count >= csv_count:
        return True, clip_count, "csv"

    return False, clip_count, ""


def split_single_episode(
    episode: Path,
    cfg: dict,
    ffmpeg: Path,
    video_codec: str,
    scene_root: Path,
    scene_index_root: Path,
    live_reporter: LiveProgressReporter | None = None,
    episode_index: int = 1,
    episode_total: int = 1,
) -> bool:
    autosave_target = episode.stem
    out_dir = scene_root / episode.stem
    scene_csv = scene_index_root / f"{episode.stem}_scenes.csv"
    marker_path = split_success_marker_path(scene_index_root, episode.stem)
    already_done, existing_count, completion_source = split_was_completed_successfully(out_dir, scene_csv, marker_path)
    if already_done:
        if episode.exists():
            mark_status(episode, "szenen_erstellt", {"episode_id": episode.stem, "scene_count": existing_count})
        if bool(cfg.get("delete_input_after_split", True)):
            cleanup_split_working_file(episode)
        mark_step_completed(
            "03_split_scenes",
            autosave_target,
            {
                "episode_id": episode.stem,
                "clip_count": existing_count,
                "scene_csv": str(scene_csv) if scene_csv.exists() else "",
                "marker_path": str(marker_path) if marker_path.exists() else "",
                "completion_source": completion_source,
            },
        )
        ok(f"Szenenschnitt bereits erfolgreich vorhanden: {existing_count} Clips ({completion_source})")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    scene_index_root.mkdir(parents=True, exist_ok=True)
    info(f"FFmpeg-Videoencoder: {video_codec}")
    mark_step_started(
        "03_split_scenes",
        autosave_target,
        {"episode_id": episode.stem, "episode_file": str(episode), "video_codec": video_codec},
    )
    try:
        scenes = detect_scenes_with_py_scene_detect(episode, float(cfg.get("scene_detection_threshold", 0.35)))
        scenes = limited_items(scenes)

        if scenes:
            episode_started_at = time.time()
            lines = ["scene_number,start_seconds,end_seconds"]
            for index, (start_sec, end_sec) in enumerate(scenes, start=1):
                lines.append(f"{index},{start_sec:.3f},{end_sec:.3f}")
                export_scene(ffmpeg, episode, out_dir, start_sec, end_sec, index, video_codec)
                if live_reporter is not None:
                    live_reporter.update(
                        (episode_index - 1) + (index / max(1, len(scenes))),
                        current_label=f"scene_{index:04d}.mp4",
                        parent_label=episode.name,
                        extra_label=f"Zeitfenster: {start_sec:.3f}s bis {end_sec:.3f}s",
                        scope_current=index,
                        scope_total=len(scenes),
                        scope_started_at=episode_started_at,
                        scope_label=f"Folge {episode_index}/{episode_total}",
                    )
            scene_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
            created = len(scenes)
            write_split_success_marker(marker_path, episode, created, scene_csv, "scenedetect")
            completion_mode = "scenedetect"
        else:
            fallback_seconds = int(cfg.get("default_scene_seconds_fallback", 8))
            run_command(
                [
                    str(ffmpeg),
                    "-hide_banner",
                    "-y",
                    "-i",
                    str(episode),
                    "-map",
                    "0",
                    "-c",
                    "copy",
                    "-f",
                    "segment",
                    "-segment_time",
                    str(fallback_seconds),
                    "-reset_timestamps",
                    "1",
                    str(out_dir / "scene_%04d.mp4"),
                ],
                quiet=True,
            )
            created = len(sorted(out_dir.glob("*.mp4")))
            write_split_success_marker(marker_path, episode, created, None, "segment_fallback")
            completion_mode = "segment_fallback"

        if created:
            mark_status(episode, "szenen_erstellt", {"episode_id": episode.stem, "scene_count": created})
        if created and bool(cfg.get("delete_input_after_split", True)):
            cleanup_split_working_file(episode)
        mark_step_completed(
            "03_split_scenes",
            autosave_target,
            {
                "episode_id": episode.stem,
                "clip_count": created,
                "scene_csv": str(scene_csv) if scene_csv.exists() else "",
                "marker_path": str(marker_path),
                "completion_source": completion_mode,
            },
        )
        ok(f"Szenen erstellt: {created}")
        return True
    except Exception as exc:
        mark_step_failed(
            "03_split_scenes",
            str(exc),
            autosave_target,
            {"episode_id": episode.stem, "episode_file": str(episode)},
        )
        raise


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Folge in Szenen zerlegen")
    cfg = load_config()
    ffmpeg = detect_tool(PROJECT_ROOT / "tools" / "ffmpeg" / "bin", "ffmpeg")
    video_codec = preferred_ffmpeg_video_codec(ffmpeg, cfg)
    episodes_dir = resolve_project_path(cfg["paths"]["episodes"])
    scene_root = resolve_project_path(cfg["paths"]["scene_clips"])
    scene_index_root = resolve_project_path(cfg["paths"]["scene_index"])

    requested_stem = Path(args.episode_file).stem if args.episode_file else ""
    try:
        episode = resolve_episode_file(episodes_dir, scene_root, args.episode_file)
    except FileNotFoundError:
        if not requested_stem:
            raise
        out_dir = scene_root / requested_stem
        scene_csv = scene_index_root / f"{requested_stem}_scenes.csv"
        marker_path = split_success_marker_path(scene_index_root, requested_stem)
        already_done, existing_count, completion_source = split_was_completed_successfully(out_dir, scene_csv, marker_path)
        if already_done:
            mark_step_completed(
                "03_split_scenes",
                requested_stem,
                {
                    "episode_id": requested_stem,
                    "clip_count": existing_count,
                    "scene_csv": str(scene_csv) if scene_csv.exists() else "",
                    "marker_path": str(marker_path) if marker_path.exists() else "",
                    "completion_source": completion_source,
                },
            )
            ok(f"Szenenschnitt bereits erfolgreich vorhanden: {existing_count} Clips ({completion_source})")
            return
        raise
    if args.episode_file:
        if episode is None:
            info("Keine importierte Arbeitsdatei gefunden.")
            return
        split_single_episode(episode, cfg, ffmpeg, video_codec, scene_root, scene_index_root)
        return

    pending = pending_episode_files(episodes_dir)
    if not pending:
        info("Keine importierte Arbeitsdatei gefunden.")
        return

    processed = 0
    live_reporter = LiveProgressReporter(
        script_name="03_split_scenes.py",
        total=max(1, len(pending)),
        phase_label="Folgen in Szenen zerlegen",
        parent_label="Batch",
    )
    for index, current_episode in enumerate(pending, start=1):
        split_single_episode(
            current_episode,
            cfg,
            ffmpeg,
            video_codec,
            scene_root,
            scene_index_root,
            live_reporter=live_reporter,
            episode_index=index,
            episode_total=len(pending),
        )
        processed += 1
        live_reporter.update(
            index,
            current_label=current_episode.name,
            parent_label=current_episode.name,
            extra_label=f"Folge abgeschlossen: {current_episode.name}",
            scope_current=1,
            scope_total=1,
            scope_started_at=time.time(),
            scope_label=f"Folge {index}/{len(pending)}",
        )
    live_reporter.finish(current_label="Batch", extra_label=f"Folgen verarbeitet: {processed}")
    ok(f"Batch abgeschlossen: {processed} Folgen in 03 verarbeitet.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
