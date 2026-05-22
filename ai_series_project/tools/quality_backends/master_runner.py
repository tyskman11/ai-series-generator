#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import wave
from datetime import datetime
from pathlib import Path

from backend_common import ensure_parent, find_project_local_ffmpeg, load_json, print_runtime_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the final master episode from scene master clips.")
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--final-master-episode", required=True)
    return parser.parse_args()


def render_output_ready(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def collect_scene_master_clips(package_payload: dict) -> list[Path]:
    clips: list[Path] = []
    scenes = package_payload.get("scenes", []) if isinstance(package_payload.get("scenes", []), list) else []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        outputs = scene.get("current_generated_outputs", {}) if isinstance(scene.get("current_generated_outputs", {}), dict) else {}
        for key in ("scene_master_clip", "generated_lipsync_video", "video_source_path"):
            value = str(outputs.get(key, "") or "").strip()
            if not value:
                continue
            candidate = Path(value)
            if render_output_ready(candidate):
                clips.append(candidate)
                break
    return clips


def shot_clip_index(package_payload: dict) -> dict[str, Path]:
    clips: dict[str, Path] = {}
    scenes = package_payload.get("scenes", []) if isinstance(package_payload.get("scenes", []), list) else []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        for shot in scene.get("shot_packages", []) if isinstance(scene.get("shot_packages", []), list) else []:
            if not isinstance(shot, dict):
                continue
            outputs = shot.get("target_outputs", {}) if isinstance(shot.get("target_outputs", {}), dict) else {}
            shot_id = str(shot.get("shot_id", "") or "").strip()
            for key in ("lipsync_clip", "video_clip"):
                candidate = Path(str(outputs.get(key, "") or "").strip())
                if shot_id and render_output_ready(candidate):
                    clips[shot_id] = candidate
                    break
    return clips


def collect_edit_decision_clips(package_payload: dict) -> list[Path]:
    rows = package_payload.get("edit_decision_list", []) if isinstance(package_payload.get("edit_decision_list", []), list) else []
    indexed = shot_clip_index(package_payload)
    clips = [
        indexed[str(row.get("shot_id", "") or "").strip()]
        for row in rows
        if isinstance(row, dict) and str(row.get("shot_id", "") or "").strip() in indexed
    ]
    return clips if rows and len(clips) == len(rows) else []


def write_concat_file(clips: list[Path], concat_path: Path) -> None:
    lines = []
    for path in clips:
        escaped = str(path).replace("'", "''")
        lines.append(f"file '{escaped}'")
    concat_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def ffmpeg_audio_to_wav(ffmpeg: str, source: Path, target: Path, target_lufs: float | None = None) -> bool:
    if not render_output_ready(source):
        return False
    ensure_parent(str(target))
    command = [ffmpeg, "-hide_banner", "-y", "-i", str(source), "-vn"]
    if target_lufs is not None:
        command.extend(["-af", f"loudnorm=I={target_lufs}:TP=-1.0:LRA=11"])
    command.extend(["-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(target)])
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return completed.returncode == 0 and render_output_ready(target)


def concat_audio_files(ffmpeg: str, sources: list[Path], target: Path, target_lufs: float | None = None) -> bool:
    ready = [path for path in sources if render_output_ready(path)]
    if not ready:
        return False
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        concat_path = Path(handle.name)
    try:
        write_concat_file(ready, concat_path)
        ensure_parent(str(target))
        command = [ffmpeg, "-hide_banner", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path)]
        if target_lufs is not None:
            command.extend(["-af", f"loudnorm=I={target_lufs}:TP=-1.0:LRA=11"])
        command.extend(["-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(target)])
        completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return completed.returncode == 0 and render_output_ready(target)
    finally:
        concat_path.unlink(missing_ok=True)


def wav_duration_seconds(path: Path) -> float:
    if not render_output_ready(path) or path.suffix.lower() != ".wav":
        return 0.0
    try:
        with wave.open(str(path), "rb") as handle:
            rate = int(handle.getframerate() or 0)
            return handle.getnframes() / rate if rate > 0 else 0.0
    except Exception:
        return 0.0


def scene_mix_outputs(scene: dict, ffmpeg: str) -> Path | None:
    audio_mix = scene.get("audio_mix", {}) if isinstance(scene.get("audio_mix", {}), dict) else {}
    stems = audio_mix.get("stems", {}) if isinstance(audio_mix.get("stems", {}), dict) else {}
    current = scene.get("current_generated_outputs", {}) if isinstance(scene.get("current_generated_outputs", {}), dict) else {}
    dialogue_source_text = str(current.get("scene_dialogue_audio", "") or "").strip()
    dialogue_target_text = str(stems.get("dialogue", "") or "").strip()
    final_target_text = str(stems.get("final_mix", "") or "").strip()
    target_lufs = float(audio_mix.get("target_lufs", -16.0) or -16.0)
    if not dialogue_source_text or not dialogue_target_text or not final_target_text:
        return None
    dialogue_source = Path(dialogue_source_text)
    dialogue_target = Path(dialogue_target_text)
    final_target = Path(final_target_text)
    if not ffmpeg_audio_to_wav(ffmpeg, dialogue_source, dialogue_target):
        return None
    return final_target if ffmpeg_audio_to_wav(ffmpeg, dialogue_target, final_target, target_lufs) else None


def materialize_audio_mastering(package_payload: dict, ffmpeg: str) -> tuple[Path | None, dict]:
    audio_mastering = package_payload.get("audio_mastering", {}) if isinstance(package_payload.get("audio_mastering", {}), dict) else {}
    outputs = audio_mastering.get("target_outputs", {}) if isinstance(audio_mastering.get("target_outputs", {}), dict) else {}
    config = audio_mastering.get("config", {}) if isinstance(audio_mastering.get("config", {}), dict) else {}
    target_lufs = float(config.get("target_lufs", -16.0) or -16.0)
    scene_mixes: list[Path] = []
    scenes = package_payload.get("scenes", []) if isinstance(package_payload.get("scenes", []), list) else []
    for scene in scenes:
        if isinstance(scene, dict):
            mix = scene_mix_outputs(scene, ffmpeg)
            if mix is not None:
                scene_mixes.append(mix)
    dialogue_stem_text = str(outputs.get("dialogue_stem", "") or "").strip()
    final_mix_text = str(outputs.get("final_mix", "") or "").strip()
    dialogue_stem = Path(dialogue_stem_text) if dialogue_stem_text else Path()
    final_mix = Path(final_mix_text) if final_mix_text else Path()
    dialogue_ready = bool(dialogue_stem_text) and concat_audio_files(ffmpeg, scene_mixes, dialogue_stem)
    final_ready = dialogue_ready and bool(final_mix_text) and ffmpeg_audio_to_wav(ffmpeg, dialogue_stem, final_mix, target_lufs)
    metrics = {
        "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "scene_mix_count": len(scene_mixes),
        "metrics": [
            {
                "metric": "audio_mix_score",
                "status": "measured" if final_ready else "failed",
                "score": 1.0 if final_ready and len(scene_mixes) == len(scenes) else len(scene_mixes) / max(1, len(scenes)),
                "reason": "dialogue scene mixes normalized into episode final mix" if final_ready else "scene dialogue mix outputs missing",
                "inputs": {
                    "scene_count": len(scenes),
                    "scene_mix_count": len(scene_mixes),
                    "final_mix_duration_seconds": round(wav_duration_seconds(final_mix), 4) if final_ready else 0.0,
                },
                "tool": "master_runner.audio_mastering",
            }
        ],
    }
    metrics_text = str(outputs.get("mix_metrics", "") or "").strip()
    if metrics_text:
        metrics_path = Path(metrics_text)
        ensure_parent(str(metrics_path))
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return final_mix if final_ready else None, metrics


def main() -> int:
    args = parse_args()
    ffmpeg = find_project_local_ffmpeg()

    package_payload = load_json(args.package_path)
    if not isinstance(package_payload, dict) or not package_payload:
        raise RuntimeError(f"Could not load production package: {args.package_path}")

    scene_clips = collect_edit_decision_clips(package_payload) or collect_scene_master_clips(package_payload)
    if not scene_clips:
        raise RuntimeError("No scene master clips, lip-sync videos, or generated scene videos are available for mastering.")

    output_path = Path(args.final_master_episode)
    ensure_parent(str(output_path))
    final_audio_mix, _audio_metrics = materialize_audio_mastering(package_payload, ffmpeg)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        concat_path = Path(handle.name)
    try:
        write_concat_file(scene_clips, concat_path)
        command = [
            ffmpeg,
            "-hide_banner",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
        ]
        if final_audio_mix is not None and render_output_ready(final_audio_mix):
            command.extend(["-i", str(final_audio_mix)])
        command.extend([
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
        ])
        if final_audio_mix is not None and render_output_ready(final_audio_mix):
            command.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
        command.extend([
            "-movflags",
            "+faststart",
            str(output_path),
        ])
        completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if completed.returncode != 0 or not render_output_ready(output_path):
            log_text = completed.stdout or ""
            raise RuntimeError(f"FFmpeg mastering failed. {log_text[-800:]}")
    finally:
        concat_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(print_runtime_error(exc))
