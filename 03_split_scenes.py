#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from pipeline_common import (
    PROJECT_ROOT,
    detect_tool,
    error,
    ffmpeg_video_encode_args,
    first_video,
    headline,
    info,
    limited_items,
    load_config,
    ok,
    preferred_ffmpeg_video_codec,
    progress,
    rerun_in_runtime,
    resolve_project_path,
    run_command,
)


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


def main() -> None:
    rerun_in_runtime()
    headline("Folge in Szenen zerlegen")
    cfg = load_config()
    ffmpeg = detect_tool(PROJECT_ROOT / "tools" / "ffmpeg" / "bin", "ffmpeg")
    video_codec = preferred_ffmpeg_video_codec(ffmpeg, cfg)
    episodes_dir = resolve_project_path(cfg["paths"]["episodes"])
    scene_root = resolve_project_path(cfg["paths"]["scene_clips"])
    scene_index_root = resolve_project_path(cfg["paths"]["scene_index"])
    inbox_dir = resolve_project_path(cfg["paths"]["inbox_episodes"])

    episode = first_video(episodes_dir)
    if episode is None:
        info("Keine importierte Arbeitsdatei gefunden.")
        return

    out_dir = scene_root / episode.stem
    existing = sorted(out_dir.glob("scene_*.mp4")) if out_dir.exists() else []
    if existing:
        ok(f"Szenenordner existiert bereits: {len(existing)} Clips")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    scene_index_root.mkdir(parents=True, exist_ok=True)
    info(f"FFmpeg-Videoencoder: {video_codec}")
    scenes = detect_scenes_with_py_scene_detect(episode, float(cfg.get("scene_detection_threshold", 0.35)))
    scenes = limited_items(scenes)

    if scenes:
        scene_csv = scene_index_root / f"{episode.stem}_scenes.csv"
        lines = ["scene_number,start_seconds,end_seconds"]
        for index, (start_sec, end_sec) in enumerate(scenes, start=1):
            lines.append(f"{index},{start_sec:.3f},{end_sec:.3f}")
            export_scene(ffmpeg, episode, out_dir, start_sec, end_sec, index, video_codec)
            progress(index, len(scenes), "Szenen werden exportiert")
        scene_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
        created = len(scenes)
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

    if created and bool(cfg.get("delete_input_after_split", True)):
        source_copy = inbox_dir / episode.name
        if source_copy.exists():
            source_copy.unlink()
        if episode.exists():
            episode.unlink()
    ok(f"Szenen erstellt: {created}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
