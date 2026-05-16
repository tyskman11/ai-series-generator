#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from backend_common import PROJECT_DIR, ensure_parent, existing_path, find_project_local_ffmpeg, load_backend_context, print_runtime_error


def clean_text(value: object) -> str:
    return str(value or "").strip()


def resolve_wav2lip_root() -> Path:
    configured = clean_text(os.environ.get("SERIES_WAV2LIP_ROOT", ""))
    candidate = Path(configured) if configured else PROJECT_DIR / "tools" / "quality_backends" / "wav2lip"
    if not candidate.is_absolute():
        candidate = PROJECT_DIR / candidate
    candidate = candidate.resolve(strict=False)
    if not (candidate / "inference.py").exists():
        raise RuntimeError(
            "Local Wav2Lip backend is not ready. Run 00_prepare_runtime.py without --skip-downloads "
            f"or set SERIES_WAV2LIP_ROOT. Expected inference.py in {candidate}."
        )
    return candidate


def resolve_checkpoint() -> Path:
    configured = clean_text(os.environ.get("SERIES_WAV2LIP_CHECKPOINT", ""))
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            PROJECT_DIR / "tools" / "quality_models" / "lipsync" / "wav2lip_gan.pth",
            PROJECT_DIR / "tools" / "quality_models" / "lipsync" / "wav2lip.pth",
            PROJECT_DIR / "tools" / "quality_models" / "lipsync" / "runtime" / "wav2lip_gan.pth",
            PROJECT_DIR / "tools" / "quality_models" / "lipsync" / "runtime" / "wav2lip.pth",
        ]
    )
    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = PROJECT_DIR / candidate
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve(strict=False)
    raise RuntimeError(
        "Local Wav2Lip checkpoint is missing. Place wav2lip_gan.pth in "
        "ai_series_project/tools/quality_models/lipsync or set SERIES_WAV2LIP_CHECKPOINT."
    )


def export_poster(video_path: Path, poster_path: Path) -> None:
    if not str(poster_path):
        return
    ensure_parent(str(poster_path))
    command = [
        find_project_local_ffmpeg(),
        "-hide_banner",
        "-y",
        "-ss",
        "0.3",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(poster_path),
    ]
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0 or not poster_path.exists() or poster_path.stat().st_size <= 0:
        raise RuntimeError(f"Could not extract Wav2Lip poster frame. {(completed.stdout or '')[-1200:]}")


def main() -> int:
    context = load_backend_context()
    scene_video = existing_path(context.get("scene_video", ""))
    scene_audio = existing_path(context.get("scene_dialogue_audio", ""))
    if scene_video is None:
        raise RuntimeError("The Wav2Lip backend needs an existing generated scene video.")
    if scene_audio is None:
        raise RuntimeError("The Wav2Lip backend needs an existing voice-cloned dialogue audio track.")

    output_text = clean_text(context.get("lipsync_video", ""))
    if not output_text:
        raise RuntimeError("The Wav2Lip backend did not receive an output path.")
    output_path = Path(output_text)
    ensure_parent(str(output_path))

    wav2lip_root = resolve_wav2lip_root()
    checkpoint = resolve_checkpoint()
    command = [
        sys.executable,
        str(wav2lip_root / "inference.py"),
        "--checkpoint_path",
        str(checkpoint),
        "--face",
        str(scene_video),
        "--audio",
        str(scene_audio),
        "--outfile",
        str(output_path),
    ]
    completed = subprocess.run(command, cwd=str(wav2lip_root), check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Wav2Lip generation failed. {(completed.stdout or '')[-2000:]}")

    poster_text = clean_text(context.get("lipsync_poster_frame", ""))
    if poster_text:
        export_poster(output_path, Path(poster_text))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(print_runtime_error(exc))
