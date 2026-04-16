#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy
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
        "prepare_after_batch": False,
        "auto_train_after_prepare": False,
        "download_base_models": True,
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
    },
    "cloning": {
        "enable_voice_cloning": True,
        "enable_face_clone": True,
        "enable_lipsync": True,
        "voice_clone_engine": "pyttsx3",
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


def runtime_python() -> Path:
    if current_os() == "windows":
        candidate = SCRIPT_DIR / "runtime" / "venv" / "Scripts" / "python.exe"
    else:
        candidate = SCRIPT_DIR / "runtime" / "venv" / "bin" / "python3"
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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return deepcopy(default)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
