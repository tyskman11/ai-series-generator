#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR / "ai_series_project"
CONFIG_PATH = PROJECT_ROOT / "configs" / "project.json"
VIDEO_PATTERNS = ("*.mp4", "*.mkv", "*.mov", "*.avi")

try:
    from console_colors import enable_ansi, error, headline, info, ok, warn

    enable_ansi()
except Exception:
    def info(text: str) -> None:
        print(f"[INFO] {text}")

    def ok(text: str) -> None:
        print(f"[OK]   {text}")

    def warn(text: str) -> None:
        print(f"[WARN] {text}")

    def error(text: str) -> None:
        print(f"[ERROR] {text}")

    def headline(text: str) -> None:
        print()
        print("=" * 72)
        print(text)
        print("=" * 72)


DEFAULT_STRUCTURE = {
    "configs": {},
    "data": {
        "inbox": {"episodes": {}},
        "raw": {"episodes": {}, "audio": {}},
        "processed": {
            "metadata": {},
            "scene_clips": {},
            "scene_index": {},
            "faces": {},
            "speaker_segments": {},
            "speaker_transcripts": {},
            "linked_segments": {},
        },
        "datasets": {"video_training": {}},
    },
    "characters": {"maps": {}, "previews": {}, "review": {}, "voice_samples": {}, "voice_models": {}},
    "generation": {
        "model": {},
        "story_prompts": {},
        "shotlists": {},
        "renders": {"drafts": {}, "final": {}},
    },
    "training": {
        "foundation": {
            "datasets": {"frames": {}, "video": {}, "voice": {}},
            "downloads": {},
            "manifests": {},
            "plans": {},
            "checkpoints": {},
            "adapters": {},
            "finetunes": {},
            "backend_runs": {},
            "logs": {},
        }
    },
    "series_bible": {"episode_summaries": {}},
    "logs": {},
    "tmp": {},
    "tools": {"ffmpeg": {"bin": {}}, "whisper": {"models": {}}},
}

DEFAULT_CONFIG = {
    "delete_input_after_split": True,
    "preview_open_automatically": False,
    "scene_detection_threshold": 0.35,
    "default_scene_seconds_fallback": 8,
    "runtime": {
        "device": "auto",
        "prefer_gpu": True,
        "prefer_ffmpeg_gpu": True,
        "torch_cuda_index_url": "https://download.pytorch.org/whl/cu128",
    },
    "paths": {
        "inbox_episodes": "data/inbox/episodes",
        "episodes": "data/raw/episodes",
        "metadata": "data/processed/metadata",
        "scene_clips": "data/processed/scene_clips",
        "scene_index": "data/processed/scene_index",
        "faces": "data/processed/faces",
        "speaker_segments": "data/processed/speaker_segments",
        "speaker_transcripts": "data/processed/speaker_transcripts",
        "linked_segments": "data/processed/linked_segments",
        "datasets_video_training": "data/datasets/video_training",
        "whisper_model_dir": "tools/whisper/models",
        "character_map": "characters/maps/character_map.json",
        "voice_map": "characters/maps/voice_map.json",
        "review_queue": "characters/review/review_queue.json",
        "voice_samples": "characters/voice_samples",
        "voice_models": "characters/voice_models",
        "series_model": "generation/model/series_model.json",
        "series_bible_json": "series_bible/episode_summaries/auto_series_bible.json",
        "series_bible_markdown": "series_bible/episode_summaries/auto_series_bible.md",
        "foundation_frames": "training/foundation/datasets/frames",
        "foundation_video": "training/foundation/datasets/video",
        "foundation_voice": "training/foundation/datasets/voice",
        "foundation_downloads": "training/foundation/downloads",
        "foundation_manifests": "training/foundation/manifests",
        "foundation_plans": "training/foundation/plans",
        "foundation_checkpoints": "training/foundation/checkpoints",
        "foundation_adapters": "training/foundation/adapters",
        "foundation_finetunes": "training/foundation/finetunes",
        "foundation_backend_runs": "training/foundation/backend_runs",
        "foundation_logs": "training/foundation/logs",
    },
    "transcription": {
        "model_name": "large-v3",
        "cpu_model_name": "large-v3",
        "language": "de",
        "task": "transcribe",
        "merge_gap_seconds": 0.35,
        "min_segment_seconds": 0.6,
        "voice_embedding_backend": "auto",
        "voice_embedding_min_seconds": 0.45,
        "voice_embedding_context_padding_seconds": 0.45,
        "voice_embedding_threshold": 0.84,
        "voice_embedding_threshold_speechbrain": 0.44,
        "speaker_cluster_high_quality_min_seconds": 1.0,
        "speaker_cluster_min_segments": 2,
        "speaker_unknown_rescue_margin": 0.08,
        "speaker_unknown_neighbor_margin": 0.12,
        "speaker_unknown_episode_rescue_margin": 0.11,
        "speaker_unknown_episode_embedding_margin": 0.04,
        "speaker_unknown_episode_min_token_score": 2.2,
        "speaker_unknown_episode_min_token_margin": 0.8,
    },
    "diarization": {
        "model_name": "pyannote/speaker-diarization-community-1",
        "hf_token_env": "HF_TOKEN",
        "enabled": False,
    },
    "character_detection": {
        "sample_every_n_frames": 6,
        "interactive_assignment": False,
        "embedding_threshold": 0.80,
        "scene_embedding_threshold": 0.76,
        "detection_confidence_threshold": 0.80,
        "max_faces_per_frame": 2,
        "max_scene_clusters": 6,
        "max_visible_faces_per_segment": 3,
        "segment_visibility_padding_seconds": 0.35,
        "min_face_size": 32,
        "face_cluster_min_scenes": 2,
        "face_cluster_min_detections": 3,
        "voice_face_match_threshold": 0.6,
        "review_known_face_threshold": 0.72,
        "review_known_face_margin": 0.05,
        "review_known_face_reference_count": 8,
        "review_known_face_top_k": 3,
        "review_known_face_consensus_threshold": 0.66,
        "review_known_face_min_consensus": 2,
        "review_known_face_strong_match_threshold": 0.84,
        "review_known_face_min_reference_quality": 5.0,
        "review_known_face_identity_relaxed_consensus_strength": 5.0,
        "review_known_face_identity_weak_strength": 2.0,
        "review_known_face_identity_threshold_bonus_max": 0.03,
        "review_known_face_identity_margin_bonus_max": 0.02,
        "review_known_face_identity_weak_threshold_penalty": 0.02,
        "review_known_face_identity_weak_margin_penalty": 0.01,
    },
    "generation": {
        "default_scene_count": 6,
        "min_dialogue_lines_per_scene": 4,
        "max_dialogue_lines_per_scene": 7,
        "seed": 42,
        "match_source_episode_runtime": True,
        "target_episode_minutes_fallback": 22.0,
        "target_scene_duration_seconds": 42.0,
        "estimated_dialogue_line_seconds": 2.7,
        "prefer_original_dialogue_remix": False,
    },
    "foundation_training": {
        "prepare_after_batch": True,
        "auto_train_after_prepare": True,
        "required_before_generate": True,
        "required_before_render": True,
        "download_base_models": True,
        "check_model_updates": True,
        "huggingface_token_env": "HF_TOKEN",
        "min_character_scene_count": 3,
        "min_character_line_count": 3,
        "max_frame_samples_per_character": 48,
        "max_video_clips_per_character": 18,
        "max_voice_segments_per_character": 48,
        "frame_width": 1280,
        "frame_height": 720,
        "image_base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "video_base_model": "Lightricks/LTX-Video",
        "voice_base_model": "openbmb/VoxCPM2",
        "min_voice_duration_seconds_total": 8.0,
        "target_voice_duration_seconds_total": 18.0,
        "min_voice_samples_for_clone": 4,
    },
    "adapter_training": {
        "auto_train_after_foundation": True,
        "required_before_generate": True,
        "required_before_render": True,
        "min_image_samples": 8,
        "min_video_samples": 4,
        "min_voice_samples": 4,
        "min_voice_duration_seconds_total": 8.0,
        "min_voice_quality_score": 0.45,
        "image_histogram_bins": 24,
        "image_thumbnail_size": 96,
        "voice_mfcc_count": 20,
    },
    "fine_tune_training": {
        "auto_train_after_adapter": True,
        "required_before_generate": True,
        "required_before_render": True,
        "min_modalities_ready": 1,
        "target_steps_image": 1200,
        "target_steps_voice": 800,
        "target_steps_video": 600,
    },
    "backend_fine_tune": {
        "auto_run_after_fine_tune": True,
        "required_before_generate": True,
        "required_before_render": True,
        "image_backend": "lora-image",
        "video_backend": "motion-adapter",
        "voice_backend": "speaker-adapter",
    },
    "cloning": {
        "enable_voice_cloning": True,
        "enable_face_clone": True,
        "enable_lipsync": True,
        "voice_clone_engine": "pyttsx3",
        "require_trained_voice_models": True,
        "allow_system_tts_fallback": True,
        "xtts_model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "xtts_language": "de",
        "xtts_license_accepted": False,
        "voice_reference_max_segments": 4,
        "voice_reference_target_seconds": 16.0,
        "reference_audio_sample_rate": 24000,
        "enable_original_line_reuse": False,
        "original_line_similarity_threshold": 0.74,
        "original_line_min_token_overlap": 0.34,
        "max_voice_model_samples": 48,
        "portrait_width": 330,
        "portrait_height": 230,
        "portrait_zoom": 1.08,
        "portrait_motion_strength": 4.0,
        "portrait_crossfade_span_frames": 18,
        "mouth_sensitivity": 1.6,
        "jaw_pixels": 10,
        "blink_interval_frames": 90,
    },
    "render": {
        "width": 1280,
        "height": 720,
        "fps": 30,
        "title_card_seconds": 2.5,
        "closing_card_seconds": 2.0,
        "audio_pad_seconds": 0.35,
        "voice_rate": 175,
    },
}

GERMAN_STOPWORDS = {
    "aber",
    "alle",
    "alles",
    "auch",
    "auf",
    "aus",
    "bei",
    "bin",
    "bist",
    "damit",
    "dann",
    "das",
    "dass",
    "dein",
    "deine",
    "dem",
    "den",
    "der",
    "des",
    "deshalb",
    "die",
    "dies",
    "dir",
    "doch",
    "dort",
    "du",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "eines",
    "er",
    "es",
    "euch",
    "euer",
    "für",
    "ganz",
    "hab",
    "habe",
    "haben",
    "hat",
    "hier",
    "ich",
    "ihr",
    "ihre",
    "ihren",
    "im",
    "in",
    "ist",
    "ja",
    "jede",
    "jeder",
    "jetzt",
    "kein",
    "keine",
    "können",
    "könnte",
    "mal",
    "mehr",
    "mein",
    "meine",
    "mich",
    "mir",
    "mit",
    "nach",
    "nicht",
    "noch",
    "nur",
    "oder",
    "schon",
    "sehr",
    "sein",
    "seine",
    "sich",
    "sie",
    "sind",
    "so",
    "super",
    "und",
    "uns",
    "unser",
    "unter",
    "vom",
    "von",
    "vor",
    "war",
    "was",
    "weil",
    "wenn",
    "wer",
    "wie",
    "wir",
    "wird",
    "wirst",
    "wo",
    "zu",
    "zum",
    "zur",
}

PLACEHOLDER_PERSON_NAME_PATTERN = re.compile(r"^(speaker|stimme|figur|face|rolle)_[0-9a-z]+$", re.IGNORECASE)
NON_MANUAL_PERSON_NAMES = {
    "",
    "unbekannt",
    "unknown",
    "speaker_unknown",
    "noface",
    "no face",
    "ignore",
    "ignored",
    "erzähler",
    "erzaehler",
}
BACKGROUND_PERSON_NAMES = {
    "statist",
    "statisten",
    "extra",
    "extras",
    "nebenfigur",
    "nebenfiguren",
    "hintergrundfigur",
    "hintergrundfiguren",
    "background",
    "crowd",
}

KEYWORD_BLACKLIST = {
    "also",
    "bitte",
    "dich",
    "danke",
    "dank",
    "dieses",
    "genau",
    "gesagt",
    "heißt",
    "immer",
    "kann",
    "komm",
    "kommt",
    "machen",
    "macht",
    "nächsten",
    "tschüss",
    "vielen",
    "warte",
    "war's",
    "wieso",
    "würdest",
    "okay",
    "ok",
    "nein",
    "ja",
    "hast",
    "habt",
    "habe",
    "haben",
    "untertitel",
    "untertitelung",
    "copyright",
    "youtube",
    "www",
    "http",
    "https",
    "com",
    "net",
}


def current_os() -> str:
    system_name = platform.system().lower()
    if "windows" in system_name:
        return "windows"
    if "linux" in system_name:
        return "linux"
    raise RuntimeError(f"Nicht unterstütztes Betriebssystem: {platform.system()}")


def current_architecture() -> str:
    machine = re.sub(r"[^a-z0-9]+", "_", platform.machine().strip().lower())
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86_64": "x86_64",
        "aarch64": "arm64",
        "arm64": "arm64",
        "armv8": "arm64",
    }
    return aliases.get(machine, machine or "unknown")


def current_bitness() -> str:
    return f"{struct.calcsize('P') * 8}bit"


def runtime_environment_tag() -> str:
    return f"{current_os()}_{current_architecture()}_{current_bitness()}"


def runtime_venv_dir() -> Path:
    return SCRIPT_DIR / "runtime" / f"venv_{runtime_environment_tag()}"


def runtime_python() -> Path:
    if current_os() == "windows":
        candidate = runtime_venv_dir() / "Scripts" / "python.exe"
    else:
        candidate = runtime_venv_dir() / "bin" / "python3"
    if candidate.exists():
        return candidate
    return Path(sys.executable).resolve()


def rerun_in_runtime(script_path: str | Path | None = None) -> None:
    target = runtime_python().resolve()
    current = Path(sys.executable).resolve()
    if current != target:
        script = Path(script_path or sys.argv[0]).resolve()
        raise SystemExit(subprocess.run([str(target), str(script), *sys.argv[1:]]).returncode)


def run_command(
    cmd: list[str],
    quiet: bool = False,
    check: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if quiet:
        return subprocess.run(
            cmd,
            check=check,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return subprocess.run(cmd, check=check, cwd=str(cwd) if cwd else None)


def create_tree(base: Path, tree: dict[str, Any]) -> None:
    for name, subtree in tree.items():
        path = base / name
        path.mkdir(parents=True, exist_ok=True)
        if isinstance(subtree, dict):
            create_tree(path, subtree)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _atomic_write_text(path: Path, text: str, *, retries: int = 8, delay_seconds: float = 0.2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        temp_path = path.with_name(f"{path.name}.tmp_{os.getpid()}_{int(time.time() * 1000)}_{attempt}")
        try:
            temp_path.write_text(text, encoding="utf-8")
            os.replace(str(temp_path), str(path))
            return
        except PermissionError as exc:
            last_error = exc
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            if attempt >= retries - 1:
                raise
            time.sleep(delay_seconds)
        except Exception:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            raise
    if last_error is not None:
        raise last_error


def write_json(path: Path, data: Any) -> None:
    _atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False))


def read_json(path: Path, default: Any) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return deepcopy(default)


def write_text(path: Path, text: str) -> None:
    _atomic_write_text(path, text)


def step_autosave_root() -> Path:
    return resolve_project_path("runtime/autosaves/steps")


def step_autosave_path(step_name: str, target_name: str = "global") -> Path:
    safe_step = "".join(char.lower() if char.isalnum() else "_" for char in coalesce_text(step_name)) or "step"
    safe_target = "".join(char.lower() if char.isalnum() else "_" for char in coalesce_text(target_name)) or "global"
    return step_autosave_root() / safe_step / f"{safe_target}.json"


def load_step_autosave(step_name: str, target_name: str = "global") -> dict[str, Any]:
    return read_json(step_autosave_path(step_name, target_name), {})


def completed_step_state(
    step_name: str,
    target_name: str = "global",
    process_version: int | None = None,
) -> dict[str, Any]:
    state = load_step_autosave(step_name, target_name)
    if not isinstance(state, dict):
        return {}
    if coalesce_text(state.get("status")) != "completed":
        return {}
    if process_version is not None and int(state.get("process_version", 0) or 0) != int(process_version):
        return {}
    return state


def save_step_autosave(step_name: str, target_name: str = "global", payload: dict[str, Any] | None = None) -> Path:
    state = dict(payload or {})
    state.setdefault("step", step_name)
    state.setdefault("target", target_name)
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = step_autosave_path(step_name, target_name)
    write_json(path, state)
    return path


def mark_step_started(step_name: str, target_name: str = "global", extra: dict[str, Any] | None = None) -> Path:
    payload = {"status": "in_progress"}
    if extra:
        payload.update(extra)
    return save_step_autosave(step_name, target_name, payload)


def mark_step_completed(step_name: str, target_name: str = "global", extra: dict[str, Any] | None = None) -> Path:
    payload = {"status": "completed"}
    if extra:
        payload.update(extra)
    return save_step_autosave(step_name, target_name, payload)


def mark_step_failed(
    step_name: str,
    message: str,
    target_name: str = "global",
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = {"status": "failed", "error": coalesce_text(message)}
    if extra:
        payload.update(extra)
    return save_step_autosave(step_name, target_name, payload)


def resolve_project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def ensure_project_structure(config: dict[str, Any] | None = None, write_config_file: bool = False) -> dict[str, Any]:
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    create_tree(PROJECT_ROOT, DEFAULT_STRUCTURE)
    existing = read_json(CONFIG_PATH, {})
    merged = deep_merge(DEFAULT_CONFIG, existing)
    if config:
        merged = deep_merge(merged, config)
    if write_config_file or not CONFIG_PATH.exists():
        write_json(CONFIG_PATH, merged)
    char_map_path = resolve_project_path(merged["paths"]["character_map"])
    voice_map_path = resolve_project_path(merged["paths"]["voice_map"])
    review_queue_path = resolve_project_path(merged["paths"]["review_queue"])
    if not char_map_path.exists():
        write_json(char_map_path, {"clusters": {}, "aliases": {}})
    if not voice_map_path.exists():
        write_json(voice_map_path, {"clusters": {}, "aliases": {}})
    if not review_queue_path.exists():
        write_json(review_queue_path, {"items": []})
    return merged


def load_config() -> dict[str, Any]:
    return ensure_project_structure(write_config_file=False)


def open_review_item_count(cfg: dict[str, Any]) -> int:
    queue = read_json(resolve_project_path(cfg["paths"]["review_queue"]), {"items": []})
    items = queue.get("items", []) if isinstance(queue, dict) else []
    return len(items) if isinstance(items, list) else 0


def tool_on_path(tool_name: str) -> Path | None:
    found = shutil.which(tool_name)
    return Path(found).resolve() if found else None


def detect_tool(bin_dir: Path, tool_name: str) -> Path:
    fallback_dirs = [bin_dir]
    try:
        relative = bin_dir.relative_to(PROJECT_ROOT)
        fallback_dirs.append(SCRIPT_DIR / relative)
    except ValueError:
        pass

    checked = []
    for candidate_dir in fallback_dirs:
        checked.append(str(candidate_dir))
        if current_os() == "windows":
            candidates = [candidate_dir / f"{tool_name}.exe", candidate_dir / tool_name]
        else:
            candidates = [candidate_dir / tool_name, candidate_dir / f"{tool_name}.exe"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    path_candidate = tool_on_path(tool_name)
    if path_candidate is not None:
        return path_candidate
    raise FileNotFoundError(f"{tool_name} nicht gefunden in: {', '.join(checked)}")


@lru_cache(maxsize=1)
def nvidia_smi_path() -> Path | None:
    found = shutil.which("nvidia-smi")
    return Path(found).resolve() if found else None


@lru_cache(maxsize=1)
def nvidia_gpu_available() -> bool:
    path = nvidia_smi_path()
    if path is None:
        return False
    result = subprocess.run(
        [str(path), "--query-gpu=name", "--format=csv,noheader"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode == 0 and bool((result.stdout or "").strip())


@lru_cache(maxsize=1)
def torch_runtime_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "available": False,
            "cuda_available": False,
            "device_count": 0,
            "device_names": [],
            "torch_version": "",
            "cuda_version": "",
            "error": str(exc),
        }

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    device_names = [torch.cuda.get_device_name(index) for index in range(device_count)] if cuda_available else []
    return {
        "available": True,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_names": device_names,
        "torch_version": str(getattr(torch, "__version__", "")),
        "cuda_version": str(getattr(getattr(torch, "version", None), "cuda", "") or ""),
        "error": "",
    }


def runtime_settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    base = deepcopy(DEFAULT_CONFIG.get("runtime", {}))
    if config and isinstance(config.get("runtime"), dict):
        base = deep_merge(base, config["runtime"])
    env_device = os.environ.get("SERIES_DEVICE", "").strip().lower()
    if env_device in {"auto", "cpu", "cuda", "gpu"}:
        base["device"] = env_device
    return base


def preferred_torch_device(config: dict[str, Any] | None = None) -> str:
    runtime_cfg = runtime_settings(config)
    requested = str(runtime_cfg.get("device", "auto")).strip().lower()
    requested = "cuda" if requested == "gpu" else requested
    torch_info = torch_runtime_info()
    cuda_ready = bool(torch_info.get("cuda_available"))
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if cuda_ready else "cpu"
    if bool(runtime_cfg.get("prefer_gpu", True)) and cuda_ready:
        return "cuda"
    return "cpu"


def preferred_compute_label(config: dict[str, Any] | None = None) -> str:
    device = preferred_torch_device(config)
    if device != "cuda":
        return "cpu"
    names = torch_runtime_info().get("device_names") or []
    return f"cuda ({names[0]})" if names else "cuda"


def preferred_execution_label(config: dict[str, Any] | None = None) -> str:
    return "hybrid (cpu + gpu)" if preferred_torch_device(config) == "cuda" else "cpu-only"


@lru_cache(maxsize=8)
def ffmpeg_supports_encoder(ffmpeg_path: str, encoder: str) -> bool:
    result = subprocess.run(
        [ffmpeg_path, "-hide_banner", "-encoders"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.returncode == 0 and encoder in (result.stdout or "")


def preferred_ffmpeg_video_codec(ffmpeg_path: Path, config: dict[str, Any] | None = None) -> str:
    runtime_cfg = runtime_settings(config)
    if not bool(runtime_cfg.get("prefer_ffmpeg_gpu", True)):
        return "libx264"
    if preferred_torch_device(config) != "cuda":
        return "libx264"
    if ffmpeg_supports_encoder(str(ffmpeg_path), "h264_nvenc"):
        return "h264_nvenc"
    return "libx264"


def ffmpeg_video_encode_args(codec: str, quality: int, preset: str | None = None) -> list[str]:
    if codec.endswith("_nvenc"):
        return [
            "-c:v",
            codec,
            "-preset",
            preset or "fast",
            "-rc",
            "vbr",
            "-cq",
            str(quality),
            "-b:v",
            "0",
        ]
    return [
        "-c:v",
        codec,
        "-preset",
        preset or "veryfast",
        "-crf",
        str(quality),
    ]


def first_dir(root: Path) -> Path | None:
    directories = sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []
    return directories[0] if directories else None


def list_videos(folder: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in VIDEO_PATTERNS:
        files.extend(folder.glob(pattern))
    return sorted(files)


def first_video(folder: Path) -> Path | None:
    files = list_videos(folder)
    return files[0] if files else None


def progress(current: int, total: int, prefix: str = "Fortschritt") -> None:
    width = 28
    total = max(total, 1)
    done = int(width * current / total)
    percent = int(100 * current / total)
    print(f"{prefix}: [{'#' * done}{'-' * (width - done)}] {percent:3d}% ({current}/{total})")


def format_duration_hms(seconds: float | int) -> str:
    total_seconds = max(0, int(round(float(seconds or 0))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timestamp_seconds(ts: float | int | None = None) -> str:
    if ts is None:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")


def _progress_eta(started_at: float, current: float, total: float) -> tuple[float, float | None]:
    current_value = max(0.0, float(current or 0.0))
    total_value = max(1.0, float(total or 1.0))
    now = time.time()
    elapsed = max(0.0, now - float(started_at or now))
    if current_value <= 0:
        return 0.0, None
    rate = elapsed / current_value
    remaining_seconds = rate * max(0.0, total_value - current_value)
    return remaining_seconds, now + remaining_seconds


def format_progress_value(value: float | int) -> str:
    numeric = float(value or 0.0)
    if abs(numeric - round(numeric)) < 0.001:
        return str(int(round(numeric)))
    return f"{numeric:.2f}"


def truncate_display(text: str, max_length: int) -> str:
    value = coalesce_text(text)
    if len(value) <= max_length:
        return value
    if max_length <= 3:
        return value[:max_length]
    head = max(1, (max_length - 3) // 2)
    tail = max(1, max_length - 3 - head)
    return f"{value[:head]}...{value[-tail:]}"


def dashboard_bar(current: float | int, total: float | int, width: int = 26) -> str:
    total_value = max(1.0, float(total or 1.0))
    current_value = max(0.0, min(float(current or 0.0), total_value))
    filled = int(round((current_value / total_value) * width))
    filled = max(0, min(width, filled))
    return f"[{'#' * filled}{'-' * (width - filled)}]"


class LiveProgressReporter:
    def __init__(
        self,
        *,
        script_name: str,
        total: int,
        phase_label: str,
        parent_label: str = "",
    ) -> None:
        self.script_name = script_name
        self.total = max(1, int(total or 1))
        self.phase_label = coalesce_text(phase_label)
        self.parent_label = coalesce_text(parent_label)
        self.started_at = time.time()
        self.last_print_at = 0.0
        self.last_rendered_lines = 0
        self.inline_enabled = os.environ.get("SERIES_DISABLE_INLINE_PROGRESS", "").strip().lower() not in {"1", "true", "yes"}

    def _render_lines(
        self,
        current: float | int,
        *,
        current_label: str = "",
        parent_label: str | None = None,
        extra_label: str = "",
        scope_current: float | int | None = None,
        scope_total: float | int | None = None,
        scope_started_at: float | None = None,
        scope_label: str = "",
    ) -> list[str]:
        current_value = max(0.0, min(float(current or 0.0), float(self.total)))
        overall_remaining_seconds, overall_eta_timestamp = _progress_eta(self.started_at, current_value, float(self.total))
        percent = int((100 * current_value) / float(self.total))
        active_parent = coalesce_text(parent_label) if parent_label is not None else self.parent_label
        width = 116
        inner_width = width - 4

        def line(label: str, value: str) -> str:
            body = f" {label:<11}: {truncate_display(value, inner_width - 15)}"
            return f"|{body.ljust(inner_width)}|"

        lines = [
            "+" + ("=" * (width - 2)) + "+",
            line("LIVE", self.script_name),
            line("Schritt", self.phase_label),
        ]
        if active_parent:
            lines.append(line("Datei", active_parent))
        if current_label:
            lines.append(line("Aktuell", current_label))
        if scope_current is not None and scope_total is not None:
            scope_total_value = max(1.0, float(scope_total or 1.0))
            scope_current_value = max(0.0, min(float(scope_current or 0.0), scope_total_value))
            scope_percent = int((100 * scope_current_value) / scope_total_value)
            scope_remaining_seconds, scope_eta_timestamp = _progress_eta(scope_started_at or self.started_at, scope_current_value, scope_total_value)
            active_scope_label = coalesce_text(scope_label) or "Aktuell"
            lines.append(
                line(
                    active_scope_label,
                    f"{dashboard_bar(scope_current_value, scope_total_value)} "
                    f"{format_progress_value(scope_current_value)}/{format_progress_value(scope_total_value)} ({scope_percent}%)",
                )
            )
            lines.append(
                line(
                    "Ende aktuell",
                    format_timestamp_seconds(scope_eta_timestamp) if scope_eta_timestamp else "wird berechnet",
                )
            )
        else:
            lines.append(
                line(
                    "Ende aktuell",
                    format_timestamp_seconds(overall_eta_timestamp) if overall_eta_timestamp else "wird berechnet",
                )
            )
        lines.append(
            line(
                "Gesamt",
                f"{dashboard_bar(current_value, self.total)} "
                f"{format_progress_value(current_value)}/{format_progress_value(self.total)} ({percent}%)",
            )
        )
        lines.append(line("Rest gesamt", format_duration_hms(overall_remaining_seconds)))
        lines.append(
            line(
                "Ende gesamt",
                format_timestamp_seconds(overall_eta_timestamp) if overall_eta_timestamp else "wird berechnet",
            )
        )
        if extra_label:
            lines.append(line("Info", extra_label))
        lines.append("+" + ("=" * (width - 2)) + "+")
        return lines

    def _emit(self, lines: list[str], *, final: bool = False) -> None:
        if self.inline_enabled:
            line_count = len(lines)
            if self.last_rendered_lines > 0:
                sys.stdout.write("\r")
                if self.last_rendered_lines > 1:
                    sys.stdout.write(f"\x1b[{self.last_rendered_lines - 1}A")
                for index in range(self.last_rendered_lines):
                    sys.stdout.write("\x1b[2K")
                    if index < self.last_rendered_lines - 1:
                        sys.stdout.write("\x1b[1B")
                if self.last_rendered_lines > 1:
                    sys.stdout.write(f"\x1b[{self.last_rendered_lines - 1}A")
                sys.stdout.write("\r")
            sys.stdout.write("\n".join(lines))
            if final:
                sys.stdout.write("\n")
                self.last_rendered_lines = 0
            else:
                self.last_rendered_lines = line_count
            sys.stdout.flush()
            return
        for entry in lines:
            info(entry)

    def update(
        self,
        current: float | int,
        *,
        current_label: str = "",
        parent_label: str | None = None,
        extra_label: str = "",
        force: bool = False,
        scope_current: float | int | None = None,
        scope_total: float | int | None = None,
        scope_started_at: float | None = None,
        scope_label: str = "",
    ) -> None:
        now = time.time()
        if not force and current < self.total and self.last_print_at and (now - self.last_print_at) < 0.15:
            return
        self.last_print_at = now
        self._emit(
            self._render_lines(
                current,
                current_label=current_label,
                parent_label=parent_label,
                extra_label=extra_label,
                scope_current=scope_current,
                scope_total=scope_total,
                scope_started_at=scope_started_at,
                scope_label=scope_label,
            ),
            final=False,
        )

    def finish(
        self,
        *,
        current_label: str = "",
        parent_label: str | None = None,
        extra_label: str = "",
        scope_current: float | int | None = None,
        scope_total: float | int | None = None,
        scope_started_at: float | None = None,
        scope_label: str = "",
    ) -> None:
        self.last_print_at = time.time()
        self._emit(
            self._render_lines(
                self.total,
                current_label=current_label,
                parent_label=parent_label,
                extra_label=extra_label,
                scope_current=scope_current,
                scope_total=scope_total,
                scope_started_at=scope_started_at,
                scope_label=scope_label,
            ),
            final=True,
        )


def registry_path() -> Path:
    return PROJECT_ROOT / "logs" / "processing_registry.json"


def load_registry() -> dict[str, Any]:
    return read_json(registry_path(), {"files": {}})


def save_registry(registry: dict[str, Any]) -> None:
    write_json(registry_path(), registry)


def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{path.name}|{stat.st_size}|{int(stat.st_mtime)}"


def next_unprocessed_video(folder: Path) -> Path | None:
    registry = load_registry()
    for video in list_videos(folder):
        if file_fingerprint(video) not in registry.get("files", {}):
            return video
    return None


def mark_status(path: Path, status: str, extra: dict[str, Any] | None = None) -> None:
    registry = load_registry()
    fingerprint = file_fingerprint(path)
    entry = registry.setdefault("files", {}).setdefault(fingerprint, {})
    entry.update(
        {
            "filename": path.name,
            "full_path": str(path),
            "status": status,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    if extra:
        entry.update(extra)
    save_registry(registry)


def open_file_default(path: Path) -> None:
    try:
        if current_os() == "windows":
            os.startfile(str(path))
        elif current_os() == "linux":
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    a_list = list(a)
    b_list = list(b)
    if not a_list or not b_list:
        return -1.0
    dot_product = sum(x * y for x, y in zip(a_list, b_list))
    norm_a = math.sqrt(sum(x * x for x in a_list))
    norm_b = math.sqrt(sum(y * y for y in b_list))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot_product / (norm_a * norm_b)


def get_env_limit(name: str = "SERIES_MAX_SCENES") -> int | None:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    return value if value > 0 else None


def limited_items(items: list[Any], env_name: str = "SERIES_MAX_SCENES") -> list[Any]:
    limit = get_env_limit(env_name)
    if limit is None:
        return items
    return items[:limit]


def slugify_label(text: str, fallback_prefix: str, index: int) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return cleaned or f"{fallback_prefix}_{index:03d}"


def is_interactive_session() -> bool:
    return bool(getattr(sys.stdin, "isatty", lambda: False)())


def tokens_from_text(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß\-']+", text)


def is_placeholder_person_name(name: str) -> bool:
    cleaned = coalesce_text(name)
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if lowered in NON_MANUAL_PERSON_NAMES:
        return True
    return bool(PLACEHOLDER_PERSON_NAME_PATTERN.match(lowered))


def has_manual_person_name(name: str) -> bool:
    return not is_placeholder_person_name(name)


def is_background_person_name(name: str) -> bool:
    cleaned = coalesce_text(name)
    if not cleaned:
        return False
    return cleaned.lower() in BACKGROUND_PERSON_NAMES


def has_primary_person_name(name: str) -> bool:
    return has_manual_person_name(name) and not is_background_person_name(name)


def canonical_person_name(name: str) -> str:
    cleaned = coalesce_text(name)
    if not cleaned:
        return ""
    if is_background_person_name(cleaned):
        return "statist"
    return cleaned


def display_person_name(name: str, fallback: str) -> str:
    canonical = canonical_person_name(name)
    return canonical if has_manual_person_name(canonical) else fallback


def keyword_token_allowed(token: str) -> bool:
    lower = token.lower().strip("-'")
    if len(lower) < 4:
        return False
    if lower in GERMAN_STOPWORDS or lower in KEYWORD_BLACKLIST:
        return False
    if any(char.isdigit() for char in lower):
        return False
    if lower.count("-") > 1:
        return False
    return True


def extract_keywords(texts: Iterable[str], limit: int = 20) -> list[str]:
    counts: dict[str, int] = {}
    document_frequency: dict[str, int] = {}
    prepared_texts = [text for text in texts if text]
    document_count = max(1, len(prepared_texts))
    for text in prepared_texts:
        seen_in_document = set()
        for token in tokens_from_text(text):
            lower = token.lower()
            if not keyword_token_allowed(lower):
                continue
            counts[lower] = counts.get(lower, 0) + 1
            seen_in_document.add(lower)
        for token in seen_in_document:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    def token_score(item: tuple[str, int]) -> tuple[float, int, str]:
        token, tf = item
        df = document_frequency.get(token, 1)
        inverse_document_weight = math.log((document_count + 1) / df) + 1.0
        score = tf * inverse_document_weight
        return (-score, -tf, token)

    sorted_tokens = sorted(counts.items(), key=token_score)
    return [token for token, _ in sorted_tokens[:limit]]


def coalesce_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def foundation_summary_path(cfg: dict[str, Any]) -> Path:
    checkpoint_root = resolve_project_path(cfg["paths"].get("foundation_checkpoints", "training/foundation/checkpoints"))
    return checkpoint_root / "foundation_training_summary.json"


def adapter_summary_path(cfg: dict[str, Any]) -> Path:
    adapter_root = resolve_project_path(cfg["paths"].get("foundation_adapters", "training/foundation/adapters"))
    return adapter_root / "adapter_training_summary.json"


def fine_tune_summary_path(cfg: dict[str, Any]) -> Path:
    finetune_root = resolve_project_path(cfg["paths"].get("foundation_finetunes", "training/foundation/finetunes"))
    return finetune_root / "fine_tune_training_summary.json"


def backend_run_summary_path(cfg: dict[str, Any]) -> Path:
    backend_root = resolve_project_path(cfg["paths"].get("foundation_backend_runs", "training/foundation/backend_runs"))
    return backend_root / "backend_fine_tune_summary.json"


def foundation_training_required(cfg: dict[str, Any]) -> bool:
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    return bool(foundation_cfg.get("required_before_generate", True))


def foundation_render_required(cfg: dict[str, Any]) -> bool:
    foundation_cfg = cfg.get("foundation_training", {}) if isinstance(cfg.get("foundation_training"), dict) else {}
    clone_cfg = cfg.get("cloning", {}) if isinstance(cfg.get("cloning"), dict) else {}
    if bool(clone_cfg.get("require_trained_voice_models", True)):
        return True
    return bool(foundation_cfg.get("required_before_render", True))


def adapter_training_required(cfg: dict[str, Any]) -> bool:
    adapter_cfg = cfg.get("adapter_training", {}) if isinstance(cfg.get("adapter_training"), dict) else {}
    return bool(adapter_cfg.get("required_before_generate", True))


def adapter_render_required(cfg: dict[str, Any]) -> bool:
    adapter_cfg = cfg.get("adapter_training", {}) if isinstance(cfg.get("adapter_training"), dict) else {}
    return bool(adapter_cfg.get("required_before_render", True))


def fine_tune_training_required(cfg: dict[str, Any]) -> bool:
    fine_tune_cfg = cfg.get("fine_tune_training", {}) if isinstance(cfg.get("fine_tune_training"), dict) else {}
    return bool(fine_tune_cfg.get("required_before_generate", True))


def fine_tune_render_required(cfg: dict[str, Any]) -> bool:
    fine_tune_cfg = cfg.get("fine_tune_training", {}) if isinstance(cfg.get("fine_tune_training"), dict) else {}
    return bool(fine_tune_cfg.get("required_before_render", True))


def backend_fine_tune_required(cfg: dict[str, Any]) -> bool:
    backend_cfg = cfg.get("backend_fine_tune", {}) if isinstance(cfg.get("backend_fine_tune"), dict) else {}
    return bool(backend_cfg.get("required_before_generate", True))


def backend_fine_tune_render_required(cfg: dict[str, Any]) -> bool:
    backend_cfg = cfg.get("backend_fine_tune", {}) if isinstance(cfg.get("backend_fine_tune"), dict) else {}
    return bool(backend_cfg.get("required_before_render", True))


def _normalized_training_name(name: str) -> str:
    return coalesce_text(name).lower()


def load_foundation_training_index(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary_path = foundation_summary_path(cfg)
    summary_payload = read_json(summary_path, {})
    index: dict[str, dict[str, Any]] = {}
    for row in summary_payload.get("characters", []) or []:
        character_name = coalesce_text(row.get("character", ""))
        if not character_name:
            continue
        pack_path = Path(str(row.get("pack_path", "") or ""))
        pack_payload = read_json(pack_path, {}) if pack_path.exists() else {}
        voice_pack = pack_payload.get("voice_pack", {}) if isinstance(pack_payload.get("voice_pack"), dict) else {}
        video_pack = pack_payload.get("video_pack", {}) if isinstance(pack_payload.get("video_pack"), dict) else {}
        image_pack = pack_payload.get("image_pack", {}) if isinstance(pack_payload.get("image_pack"), dict) else {}
        index[_normalized_training_name(character_name)] = {
            "character": character_name,
            "pack_path": pack_path,
            "pack_exists": pack_path.exists(),
            "voice_samples": max(int(row.get("voice_samples", 0) or 0), int(voice_pack.get("sample_count", 0) or 0)),
            "voice_duration_seconds": max(
                float(row.get("voice_duration_seconds", 0.0) or 0.0),
                float(voice_pack.get("duration_seconds_total", 0.0) or 0.0),
            ),
            "voice_quality_score": max(
                float(row.get("voice_quality_score", 0.0) or 0.0),
                float(voice_pack.get("quality_score", 0.0) or 0.0),
            ),
            "voice_clone_ready": bool(row.get("voice_clone_ready", False) or voice_pack.get("clone_ready", False)),
            "video_samples": max(int(row.get("video_samples", 0) or 0), int(video_pack.get("sample_count", 0) or 0)),
            "frame_samples": max(int(row.get("frame_samples", 0) or 0), int(image_pack.get("sample_count", 0) or 0)),
        }
    return index


def foundation_training_status(
    cfg: dict[str, Any],
    characters: Iterable[str] | None = None,
    model_path: Path | None = None,
    require_voice_clone: bool = False,
) -> dict[str, Any]:
    summary_path = foundation_summary_path(cfg)
    series_model_path = model_path or resolve_project_path(cfg["paths"].get("series_model", "generation/model/series_model.json"))
    summary_exists = summary_path.exists()
    model_exists = series_model_path.exists()
    summary_mtime = summary_path.stat().st_mtime if summary_exists else 0.0
    model_mtime = series_model_path.stat().st_mtime if model_exists else 0.0
    index = load_foundation_training_index(cfg) if summary_exists else {}
    missing_characters: list[str] = []
    weak_characters: list[str] = []
    for character in characters or []:
        display_name = coalesce_text(character)
        if not has_primary_person_name(display_name):
            continue
        row = index.get(_normalized_training_name(display_name))
        if row is None or not row.get("pack_exists"):
            missing_characters.append(display_name)
            continue
        if int(row.get("voice_samples", 0) or 0) <= 0:
            weak_characters.append(display_name)
            continue
        if require_voice_clone and not bool(row.get("voice_clone_ready", False)):
            weak_characters.append(display_name)
    summary_new_enough = summary_exists and (not model_exists or summary_mtime >= model_mtime)
    return {
        "summary_path": summary_path,
        "summary_exists": summary_exists,
        "summary_new_enough": summary_new_enough,
        "model_path": series_model_path,
        "model_exists": model_exists,
        "character_index": index,
        "missing_characters": missing_characters,
        "weak_characters": weak_characters,
    }


def ensure_foundation_training_ready(
    cfg: dict[str, Any],
    *,
    characters: Iterable[str] | None = None,
    model_path: Path | None = None,
    for_render: bool = False,
) -> dict[str, Any]:
    required = foundation_render_required(cfg) if for_render else foundation_training_required(cfg)
    status = foundation_training_status(
        cfg,
        characters=characters,
        model_path=model_path,
        require_voice_clone=for_render,
    )
    if not required:
        return status
    if not status["summary_exists"]:
        raise RuntimeError(
            "Foundation-Training fehlt. Fuehre vor Generierung/Render zuerst 09_prepare_foundation_training.py und 10_train_foundation_models.py aus."
        )
    if not status["summary_new_enough"]:
        raise RuntimeError(
        "Foundation-Training ist aelter als das aktuelle Serienmodell. Fuehre nach 08_train_series_model.py erneut 09_prepare_foundation_training.py und 10_train_foundation_models.py aus."
        )
    if status["missing_characters"]:
        missing = ", ".join(status["missing_characters"])
        raise RuntimeError(
            f"Fuer diese Figuren fehlen trainierte Foundation-Packs: {missing}. Fuehre 09_prepare_foundation_training.py und 10_train_foundation_models.py erneut aus."
        )
    if status["weak_characters"]:
        weak = ", ".join(status["weak_characters"])
        raise RuntimeError(
            f"Fuer diese Figuren fehlen brauchbare Voice-Samples im Training: {weak}. Pruefe 05/08 und trainiere danach 09_prepare_foundation_training.py und 10_train_foundation_models.py erneut."
        )
    return status


def load_adapter_training_index(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary_payload = read_json(adapter_summary_path(cfg), {})
    index: dict[str, dict[str, Any]] = {}
    for row in summary_payload.get("characters", []) or []:
        character_name = coalesce_text(row.get("character", ""))
        if not character_name:
            continue
        profile_path = Path(str(row.get("profile_path", "") or ""))
        profile_payload = read_json(profile_path, {}) if profile_path.exists() else {}
        modalities = profile_payload.get("modalities", {}) if isinstance(profile_payload.get("modalities"), dict) else {}
        index[_normalized_training_name(character_name)] = {
            "character": character_name,
            "profile_path": profile_path,
            "profile_exists": profile_path.exists(),
            "training_ready": bool(row.get("training_ready", False) or profile_payload.get("training_ready", False)),
            "modalities_ready": list(row.get("modalities_ready", []) or profile_payload.get("modalities_ready", []) or []),
            "image_samples": max(
                int(row.get("image_samples", 0) or 0),
                int((modalities.get("image", {}) or {}).get("sample_count", 0) or 0),
            ),
            "voice_samples": max(
                int(row.get("voice_samples", 0) or 0),
                int((modalities.get("voice", {}) or {}).get("sample_count", 0) or 0),
            ),
            "voice_duration_seconds": max(
                float(row.get("voice_duration_seconds", 0.0) or 0.0),
                float((modalities.get("voice", {}) or {}).get("duration_seconds_total", 0.0) or 0.0),
            ),
            "voice_quality_score": max(
                float(row.get("voice_quality_score", 0.0) or 0.0),
                float((modalities.get("voice", {}) or {}).get("quality_score", 0.0) or 0.0),
            ),
            "voice_clone_ready": bool(
                row.get("voice_clone_ready", False) or (modalities.get("voice", {}) or {}).get("clone_ready", False)
            ),
            "video_samples": max(
                int(row.get("video_samples", 0) or 0),
                int((modalities.get("video", {}) or {}).get("sample_count", 0) or 0),
            ),
        }
    return index


def adapter_training_status(
    cfg: dict[str, Any],
    characters: Iterable[str] | None = None,
    require_voice_clone: bool = False,
) -> dict[str, Any]:
    summary_path = adapter_summary_path(cfg)
    foundation_path = foundation_summary_path(cfg)
    summary_exists = summary_path.exists()
    foundation_exists = foundation_path.exists()
    summary_mtime = summary_path.stat().st_mtime if summary_exists else 0.0
    foundation_mtime = foundation_path.stat().st_mtime if foundation_exists else 0.0
    index = load_adapter_training_index(cfg) if summary_exists else {}
    missing_characters: list[str] = []
    weak_characters: list[str] = []
    for character in characters or []:
        display_name = coalesce_text(character)
        if not has_primary_person_name(display_name):
            continue
        row = index.get(_normalized_training_name(display_name))
        if row is None or not row.get("profile_exists") or not row.get("training_ready"):
            missing_characters.append(display_name)
            continue
        if int(row.get("image_samples", 0) or 0) <= 0 and int(row.get("voice_samples", 0) or 0) <= 0:
            weak_characters.append(display_name)
            continue
        if require_voice_clone and int(row.get("voice_samples", 0) or 0) > 0 and not bool(row.get("voice_clone_ready", False)):
            weak_characters.append(display_name)
    summary_new_enough = summary_exists and (not foundation_exists or summary_mtime >= foundation_mtime)
    return {
        "summary_path": summary_path,
        "summary_exists": summary_exists,
        "summary_new_enough": summary_new_enough,
        "foundation_summary_path": foundation_path,
        "foundation_summary_exists": foundation_exists,
        "character_index": index,
        "missing_characters": missing_characters,
        "weak_characters": weak_characters,
    }


def ensure_adapter_training_ready(
    cfg: dict[str, Any],
    *,
    characters: Iterable[str] | None = None,
    for_render: bool = False,
) -> dict[str, Any]:
    required = adapter_render_required(cfg) if for_render else adapter_training_required(cfg)
    status = adapter_training_status(cfg, characters=characters, require_voice_clone=for_render)
    if not required:
        return status
    if not status["summary_exists"]:
        raise RuntimeError(
        "Adapter-Training fehlt. Fuehre vor Generierung/Render zuerst 11_train_adapter_models.py aus."
        )
    if not status["summary_new_enough"]:
        raise RuntimeError(
        "Adapter-Training ist aelter als das aktuelle Foundation-Training. Fuehre nach 10_train_foundation_models.py erneut 11_train_adapter_models.py aus."
        )
    if status["missing_characters"]:
        missing = ", ".join(status["missing_characters"])
        raise RuntimeError(
        f"Fuer diese Figuren fehlen trainierte Adapter-Profile: {missing}. Fuehre 11_train_adapter_models.py erneut aus."
        )
    if status["weak_characters"]:
        weak = ", ".join(status["weak_characters"])
        raise RuntimeError(
        f"Fuer diese Figuren sind die Adapter-Profile noch zu schwach: {weak}. Pruefe 09/10 und trainiere danach 11_train_adapter_models.py erneut."
        )
    return status


def load_fine_tune_training_index(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary_payload = read_json(fine_tune_summary_path(cfg), {})
    index: dict[str, dict[str, Any]] = {}
    for row in summary_payload.get("characters", []) or []:
        character_name = coalesce_text(row.get("character", ""))
        if not character_name:
            continue
        profile_path = Path(str(row.get("fine_tune_path", "") or ""))
        profile_payload = read_json(profile_path, {}) if profile_path.exists() else {}
        index[_normalized_training_name(character_name)] = {
            "character": character_name,
            "fine_tune_path": profile_path,
            "profile_exists": profile_path.exists(),
            "training_ready": bool(row.get("training_ready", False) or profile_payload.get("training_ready", False)),
            "modalities_ready": list(row.get("modalities_ready", []) or profile_payload.get("modalities_ready", []) or []),
            "target_steps": dict(row.get("target_steps", {}) or profile_payload.get("target_steps", {}) or {}),
            "completed_steps": dict(row.get("completed_steps", {}) or profile_payload.get("completed_steps", {}) or {}),
            "voice_duration_seconds": max(
                float(row.get("voice_duration_seconds", 0.0) or 0.0),
                float(profile_payload.get("voice_duration_seconds", 0.0) or 0.0),
            ),
            "voice_quality_score": max(
                float(row.get("voice_quality_score", 0.0) or 0.0),
                float(profile_payload.get("voice_quality_score", 0.0) or 0.0),
            ),
            "voice_clone_ready": bool(row.get("voice_clone_ready", False) or profile_payload.get("voice_clone_ready", False)),
        }
    return index


def fine_tune_training_status(
    cfg: dict[str, Any],
    characters: Iterable[str] | None = None,
    require_voice_clone: bool = False,
) -> dict[str, Any]:
    summary_path = fine_tune_summary_path(cfg)
    adapter_path = adapter_summary_path(cfg)
    summary_exists = summary_path.exists()
    adapter_exists = adapter_path.exists()
    summary_mtime = summary_path.stat().st_mtime if summary_exists else 0.0
    adapter_mtime = adapter_path.stat().st_mtime if adapter_exists else 0.0
    index = load_fine_tune_training_index(cfg) if summary_exists else {}
    missing_characters: list[str] = []
    weak_characters: list[str] = []
    for character in characters or []:
        display_name = coalesce_text(character)
        if not has_primary_person_name(display_name):
            continue
        row = index.get(_normalized_training_name(display_name))
        if row is None or not row.get("profile_exists") or not row.get("training_ready"):
            missing_characters.append(display_name)
            continue
        modalities_ready = list(row.get("modalities_ready", []) or [])
        if not modalities_ready:
            weak_characters.append(display_name)
            continue
        if require_voice_clone and "voice" in modalities_ready and not bool(row.get("voice_clone_ready", False)):
            weak_characters.append(display_name)
    summary_new_enough = summary_exists and (not adapter_exists or summary_mtime >= adapter_mtime)
    return {
        "summary_path": summary_path,
        "summary_exists": summary_exists,
        "summary_new_enough": summary_new_enough,
        "adapter_summary_path": adapter_path,
        "adapter_summary_exists": adapter_exists,
        "character_index": index,
        "missing_characters": missing_characters,
        "weak_characters": weak_characters,
    }


def ensure_fine_tune_training_ready(
    cfg: dict[str, Any],
    *,
    characters: Iterable[str] | None = None,
    for_render: bool = False,
) -> dict[str, Any]:
    required = fine_tune_render_required(cfg) if for_render else fine_tune_training_required(cfg)
    status = fine_tune_training_status(cfg, characters=characters, require_voice_clone=for_render)
    if not required:
        return status
    if not status["summary_exists"]:
        raise RuntimeError(
        "Fine-Tune-Training fehlt. Fuehre vor Generierung/Render zuerst 12_train_fine_tune_models.py aus."
        )
    if not status["summary_new_enough"]:
        raise RuntimeError(
        "Fine-Tune-Training ist aelter als das aktuelle Adapter-Training. Fuehre nach 11_train_adapter_models.py erneut 12_train_fine_tune_models.py aus."
        )
    if status["missing_characters"]:
        missing = ", ".join(status["missing_characters"])
        raise RuntimeError(
        f"Fuer diese Figuren fehlen trainierte Fine-Tune-Profile: {missing}. Fuehre 12_train_fine_tune_models.py erneut aus."
        )
    if status["weak_characters"]:
        weak = ", ".join(status["weak_characters"])
        raise RuntimeError(
        f"Fuer diese Figuren sind die Fine-Tune-Profile noch zu schwach: {weak}. Fuehre 12_train_fine_tune_models.py erneut aus."
        )
    return status


def load_backend_run_index(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary_payload = read_json(backend_run_summary_path(cfg), {})
    index: dict[str, dict[str, Any]] = {}
    for row in summary_payload.get("characters", []) or []:
        character_name = coalesce_text(row.get("character", ""))
        if not character_name:
            continue
        run_path = Path(str(row.get("backend_run_path", "") or ""))
        run_payload = read_json(run_path, {}) if run_path.exists() else {}
        index[_normalized_training_name(character_name)] = {
            "character": character_name,
            "backend_run_path": run_path,
            "run_exists": run_path.exists(),
            "training_ready": bool(row.get("training_ready", False) or run_payload.get("training_ready", False)),
            "modalities_ready": list(row.get("modalities_ready", []) or run_payload.get("modalities_ready", []) or []),
            "voice_duration_seconds": max(
                float(row.get("voice_duration_seconds", 0.0) or 0.0),
                float(run_payload.get("voice_duration_seconds", 0.0) or 0.0),
            ),
            "voice_quality_score": max(
                float(row.get("voice_quality_score", 0.0) or 0.0),
                float(run_payload.get("voice_quality_score", 0.0) or 0.0),
            ),
            "voice_clone_ready": bool(row.get("voice_clone_ready", False) or run_payload.get("voice_clone_ready", False)),
            "backends": dict(row.get("backends", {}) or run_payload.get("backends", {}) or {}),
        }
    return index


def backend_fine_tune_status(
    cfg: dict[str, Any],
    characters: Iterable[str] | None = None,
    require_voice_clone: bool = False,
) -> dict[str, Any]:
    summary_path = backend_run_summary_path(cfg)
    fine_tune_path = fine_tune_summary_path(cfg)
    summary_exists = summary_path.exists()
    fine_tune_exists = fine_tune_path.exists()
    summary_mtime = summary_path.stat().st_mtime if summary_exists else 0.0
    fine_tune_mtime = fine_tune_path.stat().st_mtime if fine_tune_exists else 0.0
    index = load_backend_run_index(cfg) if summary_exists else {}
    missing_characters: list[str] = []
    weak_characters: list[str] = []
    for character in characters or []:
        display_name = coalesce_text(character)
        if not has_primary_person_name(display_name):
            continue
        row = index.get(_normalized_training_name(display_name))
        if row is None or not row.get("run_exists") or not row.get("training_ready"):
            missing_characters.append(display_name)
            continue
        if not list(row.get("modalities_ready", []) or []):
            weak_characters.append(display_name)
            continue
        if require_voice_clone and "voice" in list(row.get("modalities_ready", []) or []) and not bool(row.get("voice_clone_ready", False)):
            weak_characters.append(display_name)
            continue
        backends = row.get("backends", {}) if isinstance(row.get("backends"), dict) else {}
        artifact_missing = False
        for backend_payload in backends.values():
            if not isinstance(backend_payload, dict):
                artifact_missing = True
                break
            artifacts = backend_payload.get("artifacts", {}) if isinstance(backend_payload.get("artifacts"), dict) else {}
            required_paths = [
                str(artifacts.get("job_path", "") or "").strip(),
                str(artifacts.get("bundle_path", "") or "").strip(),
                str(artifacts.get("weights_path", "") or "").strip(),
            ]
            if not all(required_paths) or not all(Path(path).exists() for path in required_paths):
                artifact_missing = True
                break
        if artifact_missing:
            weak_characters.append(display_name)
    summary_new_enough = summary_exists and (not fine_tune_exists or summary_mtime >= fine_tune_mtime)
    return {
        "summary_path": summary_path,
        "summary_exists": summary_exists,
        "summary_new_enough": summary_new_enough,
        "fine_tune_summary_path": fine_tune_path,
        "fine_tune_summary_exists": fine_tune_exists,
        "character_index": index,
        "missing_characters": missing_characters,
        "weak_characters": weak_characters,
    }


def ensure_backend_fine_tune_ready(
    cfg: dict[str, Any],
    *,
    characters: Iterable[str] | None = None,
    for_render: bool = False,
) -> dict[str, Any]:
    required = backend_fine_tune_render_required(cfg) if for_render else backend_fine_tune_required(cfg)
    status = backend_fine_tune_status(cfg, characters=characters, require_voice_clone=for_render)
    if not required:
        return status
    if not status["summary_exists"]:
        raise RuntimeError(
        "Backend-Fine-Tune-Laeufe fehlen. Fuehre vor Generierung/Render zuerst 13_run_backend_finetunes.py aus."
        )
    if not status["summary_new_enough"]:
        raise RuntimeError(
        "Backend-Fine-Tune-Laeufe sind aelter als das aktuelle Fine-Tune-Training. Fuehre nach 12_train_fine_tune_models.py erneut 13_run_backend_finetunes.py aus."
        )
    if status["missing_characters"]:
        missing = ", ".join(status["missing_characters"])
        raise RuntimeError(
        f"Fuer diese Figuren fehlen Backend-Fine-Tune-Laeufe: {missing}. Fuehre 13_run_backend_finetunes.py erneut aus."
        )
    if status["weak_characters"]:
        weak = ", ".join(status["weak_characters"])
        raise RuntimeError(
        f"Fuer diese Figuren sind die Backend-Fine-Tune-Laeufe noch zu schwach: {weak}. Fuehre 13_run_backend_finetunes.py erneut aus."
        )
    return status
