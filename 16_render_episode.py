#!/usr/bin/env python3
from __future__ import annotations

import os
import math
import re
import shutil
import wave
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from pipeline_common import (
    PROJECT_ROOT,
    adapter_training_status,
    backend_fine_tune_status,
    coalesce_text,
    detect_tool,
    ensure_adapter_training_ready,
    ensure_backend_fine_tune_ready,
    ensure_fine_tune_training_ready,
    ensure_foundation_training_ready,
    error,
    ffmpeg_video_encode_args,
    has_primary_person_name,
    headline,
    info,
    load_step_autosave,
    load_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    preferred_ffmpeg_video_codec,
    preferred_torch_device,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
    run_command,
    save_step_autosave,
    fine_tune_training_status,
    warn,
    write_json,
    write_text,
)

PORTRAIT_PANEL_X = 860
PORTRAIT_PANEL_Y = 196
PORTRAIT_PANEL_W = 330
PORTRAIT_PANEL_H = 230


def render_output_ready(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def find_latest_shotlist(shotlist_dir: Path) -> Path | None:
    episode_id = os.environ.get("SERIES_RENDER_EPISODE", "").strip()
    if episode_id:
        candidate = shotlist_dir / f"{episode_id}.json"
        if candidate.exists():
            return candidate
    files = sorted(shotlist_dir.glob("folge_*.json"))
    return files[-1] if files else None


def parse_speaker_line(line: str) -> tuple[str, str]:
    if ":" not in line:
        return "erzähler", line.strip()
    speaker, text = line.split(":", 1)
    return speaker.strip() or "erzähler", text.strip()


def scene_dialogue_source(scene: dict, line_index: int) -> dict:
    sources = scene.get("dialogue_sources", []) or []
    if 0 <= line_index < len(sources) and isinstance(sources[line_index], dict):
        return dict(sources[line_index])
    return {}


def useful_character_name(name: str) -> bool:
    return has_primary_person_name(name)


def safe_filename_slug(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return cleaned or fallback


def xtts_license_accepted(clone_cfg: dict) -> bool:
    env_value = os.environ.get("SERIES_ACCEPT_COQUI_LICENSE", "").strip().lower()
    if env_value in {"1", "true", "yes", "y"}:
        return True
    return bool(clone_cfg.get("xtts_license_accepted", False))


def requested_voice_clone_engine(clone_cfg: dict) -> str:
    return str(clone_cfg.get("voice_clone_engine", "pyttsx3") or "pyttsx3").strip().lower()


def find_linked_segment_files(cfg: dict) -> list[Path]:
    linked_root = resolve_project_path(cfg["paths"]["linked_segments"])
    return sorted(linked_root.glob("*_linked_segments.json"))


def build_audio_segment_index(audio_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    if not audio_root.exists():
        return index
    for candidate in audio_root.rglob("scene_*_seg_*.wav"):
        index.setdefault(candidate.stem, []).append(candidate)
    return index


def build_scene_clip_index(scene_root: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    if not scene_root.exists():
        return index
    for episode_dir in scene_root.iterdir():
        if not episode_dir.is_dir():
            continue
        for clip_path in episode_dir.glob("scene_*.mp4"):
            index[(episode_dir.name, clip_path.stem)] = clip_path
    return index


def text_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-ZäöüÄÖÜß0-9]+", (text or "").lower()) if len(token) >= 2}


def text_similarity_score(left: str, right: str) -> tuple[float, float]:
    left_text = (left or "").strip().lower()
    right_text = (right or "").strip().lower()
    if not left_text or not right_text:
        return 0.0, 0.0
    left_tokens = text_tokens(left_text)
    right_tokens = text_tokens(right_text)
    overlap = 0.0
    union = left_tokens | right_tokens
    if union:
        overlap = len(left_tokens & right_tokens) / max(1, len(union))
    sequence = SequenceMatcher(None, left_text, right_text).ratio()
    score = (overlap * 0.55) + (sequence * 0.45)
    return round(score, 4), round(overlap, 4)


def episode_name_from_linked_path(linked_path: Path) -> str:
    suffix = "_linked_segments"
    stem = linked_path.stem
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem


def build_original_line_library(cfg: dict) -> dict[str, list[dict]]:
    audio_index = build_audio_segment_index(resolve_project_path("data/raw/audio"))
    scene_index = build_scene_clip_index(resolve_project_path(cfg["paths"]["scene_clips"]))
    library: dict[str, list[dict]] = {}
    for linked_path in find_linked_segment_files(cfg):
        episode_name = episode_name_from_linked_path(linked_path)
        for entry in read_json(linked_path, []):
            speaker_name = (entry.get("speaker_name") or "").strip()
            text = (entry.get("text") or "").strip()
            segment_id = (entry.get("segment_id") or "").strip()
            scene_id = (entry.get("scene_id") or "").strip()
            if not useful_character_name(speaker_name) or not text or not segment_id or not scene_id:
                continue
            audio_candidates = [path for path in audio_index.get(segment_id, []) if path.exists()]
            scene_clip_path = scene_index.get((episode_name, scene_id))
            if not audio_candidates or scene_clip_path is None or not scene_clip_path.exists():
                continue
            start = float(entry.get("start", 0.0) or 0.0)
            end = float(entry.get("end", 0.0) or 0.0)
            if end <= start:
                continue
            library.setdefault(speaker_name, []).append(
                {
                    "episode_name": episode_name,
                    "scene_id": scene_id,
                    "segment_id": segment_id,
                    "text": text,
                    "audio_path": str(audio_candidates[0]),
                    "scene_clip_path": str(scene_clip_path),
                    "start": start,
                    "end": end,
                    "duration_seconds": round(end - start, 3),
                    "speaker_reference_frames": entry.get("speaker_reference_frames", []) or [],
                    "score_tokens": sorted(text_tokens(text)),
                }
            )
    return library


def select_retrieval_segment(character: str, text: str, library: dict[str, list[dict]], clone_cfg: dict) -> dict | None:
    if not bool(clone_cfg.get("enable_original_line_reuse", True)):
        return None
    threshold = float(clone_cfg.get("original_line_similarity_threshold", 0.74))
    min_overlap = float(clone_cfg.get("original_line_min_token_overlap", 0.34))
    best_entry: dict | None = None
    best_score = 0.0
    best_overlap = 0.0
    for entry in library.get(character, []):
        score, overlap = text_similarity_score(text, entry.get("text", ""))
        if score > best_score or (score == best_score and overlap > best_overlap):
            best_score = score
            best_overlap = overlap
            best_entry = entry
    if best_entry is None:
        return None
    if best_score < threshold or best_overlap < min_overlap:
        return None
    selected = dict(best_entry)
    selected["match_score"] = round(best_score, 4)
    selected["token_overlap"] = round(best_overlap, 4)
    return selected


def pick_character_preview_paths(character: str, char_map: dict, max_images: int = 2) -> list[Path]:
    images: list[Path] = []
    for payload in char_map.get("clusters", {}).values():
        if payload.get("ignored"):
            continue
        if payload.get("name") != character:
            continue
        preview_dir = Path(payload.get("preview_dir", ""))
        if not preview_dir.exists():
            continue
        for pattern in ("*_crop.jpg", "*_context.jpg", "*.jpg"):
            for candidate in sorted(preview_dir.glob(pattern)):
                if candidate not in images:
                    images.append(candidate)
            if images:
                break
        if images:
            break
    return images[:max_images]


def pick_character_context_paths(character: str, char_map: dict, max_images: int = 2) -> list[Path]:
    images: list[Path] = []
    for payload in char_map.get("clusters", {}).values():
        if payload.get("ignored"):
            continue
        if payload.get("name") != character:
            continue
        preview_dir = Path(payload.get("preview_dir", ""))
        if not preview_dir.exists():
            continue
        for pattern in ("*_context.jpg", "*_speaker_frame_*.jpg", "*.jpg"):
            for candidate in sorted(preview_dir.glob(pattern)):
                if candidate not in images and candidate.exists():
                    images.append(candidate)
            if images:
                break
        if images:
            break
    return images[:max_images]


def build_voice_reference_library(cfg: dict) -> dict[str, list[Path]]:
    audio_index = build_audio_segment_index(resolve_project_path("data/raw/audio"))
    library: dict[str, list[Path]] = {}
    for linked_path in find_linked_segment_files(cfg):
        for entry in read_json(linked_path, []):
            speaker_name = (entry.get("speaker_name") or "").strip()
            if not useful_character_name(speaker_name):
                continue
            segment_id = (entry.get("segment_id") or "").strip()
            if not segment_id:
                continue
            for audio_path in audio_index.get(segment_id, []):
                if audio_path.exists() and audio_path not in library.setdefault(speaker_name, []):
                    library[speaker_name].append(audio_path)
    return library


def select_voice_reference_paths(
    character: str,
    voice_library: dict[str, list[Path]],
    max_segments: int,
    target_seconds: float,
) -> list[Path]:
    candidates = []
    for path in voice_library.get(character, []):
        if not path.exists():
            continue
        duration = audio_duration_seconds(path)
        if duration <= 0.35:
            continue
        candidates.append((duration, path))
    candidates.sort(key=lambda item: (-item[0], item[1].name))

    selected: list[Path] = []
    total_seconds = 0.0
    for duration, path in candidates:
        selected.append(path)
        total_seconds += duration
        if len(selected) >= max_segments or total_seconds >= target_seconds:
            break
    return selected


def prepare_voice_reference_wav(
    ffmpeg: Path,
    character: str,
    source_paths: list[Path],
    output_wav: Path,
    sample_rate: int,
) -> Path | None:
    if not source_paths:
        return None
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    if output_wav.exists() and output_wav.stat().st_size > 0:
        return output_wav

    if len(source_paths) == 1:
        run_command(
            [
                str(ffmpeg),
                "-hide_banner",
                "-y",
                "-i",
                str(source_paths[0]),
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                str(output_wav),
            ],
            quiet=True,
        )
        return output_wav if output_wav.exists() else None

    filter_inputs = "".join(f"[{index}:a:0]" for index in range(len(source_paths)))
    command = [str(ffmpeg), "-hide_banner", "-y"]
    for path in source_paths:
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-filter_complex",
            f"{filter_inputs}concat=n={len(source_paths)}:v=0:a=1[a]",
            "-map",
            "[a]",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            str(output_wav),
        ]
    )
    run_command(command, quiet=True)
    return output_wav if output_wav.exists() else None


def estimate_voice_profile(reference_wav: Path | None, character: str, available_voice_ids: list[str], base_rate: int) -> dict:
    profile = {
        "character": character,
        "reference_wav": str(reference_wav) if reference_wav else "",
        "voice_id": available_voice_ids[0] if available_voice_ids else "",
        "rate": base_rate,
        "duration_seconds": round(audio_duration_seconds(reference_wav), 3) if reference_wav else 0.0,
        "pitch_hz": 0.0,
        "energy": 0.0,
    }
    if not reference_wav or not reference_wav.exists():
        return profile

    pitch_hz = 0.0
    energy = 0.0
    try:
        import librosa
        import numpy as np

        samples, sample_rate = librosa.load(str(reference_wav), sr=None, mono=True)
        if samples.size:
            energy = float(np.sqrt(np.mean(np.square(samples))))
            try:
                f0 = librosa.yin(samples, fmin=70, fmax=350, sr=sample_rate)
                voiced = f0[(f0 > 70) & (f0 < 350)]
                if voiced.size:
                    pitch_hz = float(np.median(voiced))
            except Exception:
                pitch_hz = 0.0
    except Exception:
        pitch_hz = 0.0
        energy = 0.0

    voice_index = abs(hash(character.lower())) % max(1, len(available_voice_ids))
    rate_offset = 0
    if pitch_hz:
        if pitch_hz >= 210:
            rate_offset += 10
        elif pitch_hz <= 130:
            rate_offset -= 10
    if energy:
        if energy >= 0.09:
            rate_offset += 5
        elif energy <= 0.03:
            rate_offset -= 5

    profile["voice_id"] = available_voice_ids[voice_index] if available_voice_ids else ""
    profile["rate"] = max(140, min(220, base_rate + rate_offset))
    profile["pitch_hz"] = round(pitch_hz, 2)
    profile["energy"] = round(energy, 5)
    return profile


@lru_cache(maxsize=1)
def xtts_available() -> bool:
    try:
        from TTS.api import TTS  # noqa: F401
    except Exception:
        return False
    return True


@lru_cache(maxsize=2)
def load_xtts_model(model_name: str, device: str):
    from TTS.api import TTS

    try:
        model = TTS(model_name=model_name, progress_bar=False)
    except TypeError:
        model = TTS(model_name=model_name)
    if hasattr(model, "to"):
        model = model.to(device)
    return model


def synthesize_speech(
    text: str,
    output_wav: Path,
    speaker_name: str,
    voice_id: str,
    rate: int,
    reference_wav: Path | None,
    clone_cfg: dict,
    cfg: dict,
) -> dict:
    clone_engine = requested_voice_clone_engine(clone_cfg)
    use_voice_clone = (
        bool(clone_cfg.get("enable_voice_cloning", True))
        and reference_wav is not None
        and clone_engine in {"auto", "xtts"}
    )
    fallback_reason = ""
    if bool(clone_cfg.get("enable_voice_cloning", True)) and reference_wav is not None and clone_engine not in {"auto", "xtts"}:
        fallback_reason = f"voice_clone_engine_{clone_engine}"
    if use_voice_clone and xtts_available():
        if not xtts_license_accepted(clone_cfg):
            fallback_reason = "xtts_license_not_accepted"
        else:
            model_name = str(clone_cfg.get("xtts_model_name", "tts_models/multilingual/multi-dataset/xtts_v2"))
            language = str(clone_cfg.get("xtts_language", "de"))
            device = preferred_torch_device(cfg)
            try:
                model = load_xtts_model(model_name, device)
                remove_if_exists(output_wav)
                model.tts_to_file(
                    text=text,
                    speaker_wav=str(reference_wav),
                    language=language,
                    file_path=str(output_wav),
                )
                if output_wav.exists() and output_wav.stat().st_size > 0:
                    return {
                        "engine": "xtts",
                        "speaker_name": speaker_name,
                        "reference_wav": str(reference_wav),
                        "device": device,
                        "voice_cloned": True,
                        "fallback_reason": "",
                    }
                fallback_reason = "xtts_no_output"
            except Exception as exc:
                fallback_reason = str(exc)
                warn(f"Voice-Cloning fuer {speaker_name} fehlgeschlagen, Fallback auf pyttsx3: {exc}")

    remove_if_exists(output_wav)
    synthesize_tts(text, output_wav, voice_id, rate)
    return {
        "engine": "pyttsx3",
        "speaker_name": speaker_name,
        "reference_wav": str(reference_wav) if reference_wav else "",
        "device": preferred_torch_device(cfg),
        "voice_cloned": False,
        "fallback_reason": fallback_reason,
    }


def list_tts_voices() -> list[str]:
    import pyttsx3

    engine = pyttsx3.init()
    voices = engine.getProperty("voices") or []
    engine.stop()

    def sort_key(voice) -> tuple[int, str]:
        voice_id = getattr(voice, "id", "").lower()
        voice_name = getattr(voice, "name", "").lower()
        german_score = 0 if ("de" in voice_id or "german" in voice_name or "hedda" in voice_name) else 1
        return german_score, f"{voice_name}|{voice_id}"

    ordered = sorted(voices, key=sort_key)
    return [getattr(voice, "id", "") for voice in ordered if getattr(voice, "id", "")]


def list_tts_voice_descriptors() -> list[dict]:
    import pyttsx3

    engine = pyttsx3.init()
    voices = engine.getProperty("voices") or []
    engine.stop()
    descriptors: list[dict] = []
    for voice in voices:
        voice_id = str(getattr(voice, "id", "") or "")
        voice_name = str(getattr(voice, "name", "") or "")
        languages = [str(language) for language in (getattr(voice, "languages", []) or [])]
        normalized = " ".join([voice_id.lower(), voice_name.lower(), " ".join(language.lower() for language in languages)])
        is_german = any(token in normalized for token in ("de-de", "german", "hedda", "deu"))
        descriptors.append(
            {
                "id": voice_id,
                "name": voice_name,
                "languages": languages,
                "is_german": is_german,
            }
        )
    descriptors.sort(key=lambda row: (0 if row.get("is_german") else 1, str(row.get("name", "")).lower(), str(row.get("id", "")).lower()))
    return descriptors


def resolve_system_voice_id(preferred_voice_id: str, voice_descriptors: list[dict], require_german: bool = True) -> str:
    if not voice_descriptors:
        return preferred_voice_id
    descriptor_by_id = {str(row.get("id", "")): row for row in voice_descriptors if row.get("id")}
    preferred = descriptor_by_id.get(str(preferred_voice_id or ""))
    if preferred and (not require_german or bool(preferred.get("is_german"))):
        return str(preferred.get("id", ""))
    german_candidates = [row for row in voice_descriptors if row.get("is_german")]
    if german_candidates:
        return str(german_candidates[0].get("id", ""))
    if preferred:
        return str(preferred.get("id", ""))
    return str(voice_descriptors[0].get("id", ""))


def assign_voices(characters: list[str], base_rate: int, voice_profiles: dict[str, dict] | None = None) -> dict[str, dict]:
    voice_descriptors = list_tts_voice_descriptors()
    voice_ids = [str(row.get("id", "")) for row in voice_descriptors if row.get("id")] or [""]
    assignments = {}
    for index, character in enumerate(characters):
        profile = (voice_profiles or {}).get(character, {})
        preferred_voice_id = str(profile.get("voice_id", "") or voice_ids[index % len(voice_ids)])
        assignments[character] = {
            "voice_id": resolve_system_voice_id(preferred_voice_id, voice_descriptors, require_german=True),
            "rate": int(profile.get("rate", base_rate + ((index % 3) - 1) * 10)),
        }
    assignments["erzähler"] = {
        "voice_id": resolve_system_voice_id(voice_ids[0], voice_descriptors, require_german=True),
        "rate": base_rate,
    }
    return assignments


def synthesize_tts(text: str, output_wav: Path, voice_id: str, rate: int) -> None:
    import pyttsx3

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    engine = pyttsx3.init()
    if voice_id:
        engine.setProperty("voice", voice_id)
    engine.setProperty("rate", rate)
    engine.save_to_file(text, str(output_wav))
    engine.runAndWait()
    engine.stop()
    if not output_wav.exists() or output_wav.stat().st_size == 0:
        raise RuntimeError(f"TTS konnte nicht erzeugt werden: {output_wav}")


def audio_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as handle:
        frame_rate = handle.getframerate() or 22050
        frame_count = handle.getnframes() or 0
    return frame_count / frame_rate if frame_rate else 0.0


def safe_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    from PIL import ImageFont

    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def font_bundle() -> dict[str, ImageFont.ImageFont]:
    from PIL import ImageFont

    windows_fonts = Path("C:/Windows/Fonts")
    return {
        "title": safe_font(
            [str(windows_fonts / "georgiab.ttf"), str(windows_fonts / "arialbd.ttf")],
            54,
        ),
        "subtitle": safe_font(
            [str(windows_fonts / "segoeuib.ttf"), str(windows_fonts / "arialbd.ttf")],
            28,
        ),
        "body": safe_font(
            [str(windows_fonts / "segoeui.ttf"), str(windows_fonts / "arial.ttf")],
            24,
        ),
        "quote": safe_font(
            [str(windows_fonts / "georgia.ttf"), str(windows_fonts / "arial.ttf")],
            34,
        ),
        "small": safe_font(
            [str(windows_fonts / "segoeui.ttf"), str(windows_fonts / "arial.ttf")],
            20,
        ),
    }


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if measure_text(draw, candidate, font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def palette_for_name(name: str) -> tuple[int, int, int]:
    palette = [
        (226, 98, 69),
        (76, 148, 255),
        (73, 185, 124),
        (245, 180, 65),
        (176, 102, 255),
        (255, 129, 181),
    ]
    return palette[sum(ord(char) for char in name) % len(palette)]


def draw_chip(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0] + 30
    height = bbox[3] - bbox[1] + 16
    draw.rounded_rectangle((x, y, x + width, y + height), radius=18, fill=fill + (255,))
    draw.text((x + 15, y + 8), text, font=font, fill="white")
    return width


def pick_reference_images(scene: dict, char_map: dict, max_images: int = 2) -> list[Path]:
    images: list[Path] = []
    for character in scene.get("characters", []):
        for candidate in pick_character_context_paths(character, char_map, max_images=max_images):
            if candidate not in images:
                images.append(candidate)
        if not images:
            for candidate in pick_character_preview_paths(character, char_map, max_images=max_images):
                if candidate not in images:
                    images.append(candidate)
        if len(images) >= max_images:
            break
    return images[:max_images]


def create_background(width: int, height: int, accent: tuple[int, int, int]) -> Image.Image:
    from PIL import Image, ImageDraw, ImageFilter

    image = Image.new("RGB", (width, height), "#10141d")
    draw = ImageDraw.Draw(image)
    for offset in range(height):
        blend = offset / max(1, height - 1)
        r = int(16 * (1 - blend) + accent[0] * 0.35 * blend)
        g = int(20 * (1 - blend) + accent[1] * 0.35 * blend)
        b = int(29 * (1 - blend) + accent[2] * 0.35 * blend)
        draw.line((0, offset, width, offset), fill=(r, g, b))
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.ellipse((-120, -80, 420, 360), fill=accent + (80,))
    overlay_draw.ellipse((width - 420, height - 320, width + 80, height + 80), fill=(255, 255, 255, 18))
    overlay = overlay.filter(ImageFilter.GaussianBlur(18))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def paste_reference_images(canvas: Image.Image, image_paths: list[Path], origin_x: int, origin_y: int, width: int, height: int) -> None:
    from PIL import Image, ImageOps

    if not image_paths:
        return
    slot_width = width
    slot_height = height if len(image_paths) == 1 else (height - 16) // 2
    for index, image_path in enumerate(image_paths[:2]):
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.fit(image, (slot_width, slot_height), method=Image.Resampling.LANCZOS)
        y = origin_y if len(image_paths) == 1 else origin_y + index * (slot_height + 16)
        canvas.paste(image, (origin_x, y))


def audio_envelope(audio_path: Path, fps: int) -> list[float]:
    import numpy as np

    with wave.open(str(audio_path), "rb") as handle:
        frame_rate = handle.getframerate() or 22050
        channel_count = handle.getnchannels() or 1
        sample_width = handle.getsampwidth() or 2
        raw_audio = handle.readframes(handle.getnframes())

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sample_width)
    if dtype is None:
        return [0.0]

    samples = np.frombuffer(raw_audio, dtype=dtype).astype("float32")
    if samples.size == 0:
        return [0.0]
    if channel_count > 1:
        samples = samples.reshape(-1, channel_count).mean(axis=1)

    frame_size = max(1, int(frame_rate / max(1, fps)))
    remainder = samples.size % frame_size
    if remainder:
        samples = np.pad(samples, (0, frame_size - remainder))
    windows = samples.reshape(-1, frame_size)
    envelope = np.mean(np.abs(windows), axis=1)
    scale = float(np.percentile(envelope, 90)) if envelope.size else 0.0
    scale = scale if scale > 1e-6 else 1.0
    normalized = np.clip(envelope / scale, 0.0, 1.0)
    return normalized.tolist() or [0.0]


def apply_audio_reactive_face_effect(
    portrait: Image.Image,
    openness: float,
    jaw_pixels: int,
    blink: bool,
) -> Image.Image:
    from PIL import Image, ImageDraw, ImageFilter

    result = portrait.convert("RGBA")
    width, height = result.size

    if openness > 0.02:
        jaw_top = int(height * 0.60)
        jaw_left = int(width * 0.18)
        jaw_right = int(width * 0.82)
        jaw_region = result.crop((jaw_left, jaw_top, jaw_right, height))
        stretched = jaw_region.resize(
            (jaw_region.width, jaw_region.height + int(jaw_pixels * openness)),
            resample=Image.Resampling.BICUBIC,
        )
        jaw_canvas = Image.new("RGBA", result.size, (0, 0, 0, 0))
        paste_y = min(height - stretched.height, jaw_top)
        jaw_canvas.paste(stretched, (jaw_left, paste_y))
        jaw_canvas.putalpha(int(34 * openness))
        lip_shadow = Image.new("RGBA", result.size, (0, 0, 0, 0))
        lip_draw = ImageDraw.Draw(lip_shadow)
        lip_shadow_y = int(height * 0.74)
        lip_draw.rounded_rectangle(
            (int(width * 0.33), lip_shadow_y, int(width * 0.67), lip_shadow_y + max(5, int(7 + openness * 9))),
            radius=6,
            fill=(24, 10, 14, int(28 + 36 * openness)),
        )
        lip_shadow = lip_shadow.filter(ImageFilter.GaussianBlur(2.2))
        result = Image.alpha_composite(result, jaw_canvas)
        result = Image.alpha_composite(result, lip_shadow)

    if blink:
        blink_overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
        blink_draw = ImageDraw.Draw(blink_overlay)
        eye_top = int(height * 0.34)
        eye_height = max(6, int(height * 0.014))
        for left_x, right_x in ((int(width * 0.23), int(width * 0.43)), (int(width * 0.57), int(width * 0.77))):
            blink_draw.rounded_rectangle(
                (left_x, eye_top, right_x, eye_top + eye_height),
                radius=4,
                fill=(18, 18, 20, 185),
            )
        blink_overlay = blink_overlay.filter(ImageFilter.GaussianBlur(1.4))
        result = Image.alpha_composite(result, blink_overlay)

    return result.convert("RGB")


def create_audio_reactive_portrait_video(
    ffmpeg: Path,
    face_image_paths: list[Path],
    audio_path: Path,
    output_mp4: Path,
    fps: int,
    video_codec: str,
    clone_cfg: dict,
) -> None:
    from PIL import Image, ImageOps

    width = int(clone_cfg.get("portrait_width", PORTRAIT_PANEL_W))
    height = int(clone_cfg.get("portrait_height", PORTRAIT_PANEL_H))
    zoom = float(clone_cfg.get("portrait_zoom", 1.08))
    sensitivity = float(clone_cfg.get("mouth_sensitivity", 1.6))
    jaw_pixels = int(clone_cfg.get("jaw_pixels", 10))
    blink_interval = max(18, int(clone_cfg.get("blink_interval_frames", 90)))
    motion_strength = float(clone_cfg.get("portrait_motion_strength", 4.0))
    crossfade_span = max(6, int(clone_cfg.get("portrait_crossfade_span_frames", 18)))
    usable_images = [path for path in face_image_paths if path.exists()]
    if not usable_images:
        raise FileNotFoundError("Keine Portrait-Bilder fuer den Lipsync-Fallback gefunden.")

    frame_dir = output_mp4.parent / f"{output_mp4.stem}_frames"
    shutil.rmtree(frame_dir, ignore_errors=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    base_faces = [Image.open(path).convert("RGB") for path in usable_images[:6]]
    envelope = audio_envelope(audio_path, fps)
    frame_count = max(1, len(envelope))

    for frame_index in range(frame_count):
        openness = min(1.0, envelope[frame_index] * sensitivity)
        zoom_variation = zoom + 0.012 * math.sin(frame_index / 11.0)
        bob_y = int(motion_strength * math.sin(frame_index / 7.5))
        image_index = (frame_index // crossfade_span) % len(base_faces)
        next_index = (image_index + 1) % len(base_faces)
        blend_phase = (frame_index % crossfade_span) / max(1, crossfade_span - 1)
        base_face = Image.blend(base_faces[image_index], base_faces[next_index], blend_phase)
        scaled = ImageOps.fit(
            base_face,
            (max(width, int(width * zoom_variation)), max(height, int(height * zoom_variation))),
            method=Image.Resampling.LANCZOS,
        )
        crop_x = max(0, (scaled.width - width) // 2)
        crop_y = max(0, (scaled.height - height) // 2 - bob_y)
        crop_y = max(0, min(crop_y, max(0, scaled.height - height)))
        portrait = scaled.crop((crop_x, crop_y, crop_x + width, crop_y + height))
        blink = (frame_index % blink_interval) in {0, 1}
        portrait = apply_audio_reactive_face_effect(portrait, openness, jaw_pixels, blink)
        portrait.save(frame_dir / f"frame_{frame_index:05d}.png")

    def command(codec: str) -> list[str]:
        return [
            str(ffmpeg),
            "-hide_banner",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%05d.png"),
            "-vf",
            f"scale={width}:{height},format=yuv420p",
            *ffmpeg_video_encode_args(codec, quality=20),
            str(output_mp4),
        ]

    try:
        run_ffmpeg_with_codec_fallback(command, video_codec, output_mp4)
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def create_line_card(
    episode_id: str,
    scene: dict,
    line_text: str,
    reference_images: list[Path],
    output_path: Path,
    width: int,
    height: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    from PIL import Image, ImageDraw, ImageOps

    speaker, spoken_text = parse_speaker_line(line_text)
    accent = palette_for_name(speaker)

    if reference_images:
        source = Image.open(reference_images[0]).convert("RGB")
        image = ImageOps.fit(source, (width, height), method=Image.Resampling.LANCZOS)
    else:
        image = create_background(width, height, accent)

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle((0, 0, width, 170), fill=(8, 12, 18, 105))
    overlay_draw.rectangle((0, height - 250, width, height), fill=(5, 8, 14, 195))
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    draw.text((42, 28), episode_id.upper(), font=fonts["small"], fill=(220, 228, 236))
    draw.text((42, 58), scene.get("title", "Szene"), font=fonts["subtitle"], fill="white")

    chip_x = 42
    chip_y = 106
    for character in scene.get("characters", [])[:3]:
        chip_width = draw_chip(draw, chip_x, chip_y, character, fonts["small"], palette_for_name(character))
        chip_x += chip_width + 12

    subtitle_box = (34, height - 218, width - 34, height - 34)
    draw.rounded_rectangle(subtitle_box, radius=30, fill=(6, 10, 16, 228), outline=accent + (235,), width=4)
    draw.text((64, height - 194), speaker.upper(), font=fonts["subtitle"], fill=accent)
    quote_lines = wrap_text(draw, spoken_text, fonts["quote"], width - 128)
    y = height - 142
    for line in quote_lines[:3]:
        draw.text((64, y), line, font=fonts["quote"], fill="white")
        y += 42

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def create_title_card(
    episode_id: str,
    shotlist: dict,
    output_path: Path,
    width: int,
    height: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    from PIL import ImageDraw

    accent = (88, 154, 255)
    image = create_background(width, height, accent)
    draw = ImageDraw.Draw(image)
    display_title = coalesce_text(shotlist.get("display_title", "")) or episode_id.upper()
    episode_title = coalesce_text(shotlist.get("episode_title", ""))
    draw.text((96, 120), "AI Serien Draft", font=fonts["subtitle"], fill=(214, 223, 238))
    draw.text((92, 168), display_title, font=fonts["title"], fill="white")
    focus = ", ".join(shotlist.get("focus_characters", [])[:3]) or "Neue Figuren"
    keywords = ", ".join(shotlist.get("keywords", [])[:6])
    body = f"Titel: {episode_title or display_title}\nHauptfiguren: {focus}\nThemen: {keywords}"
    y = 300
    for line in body.splitlines():
        draw.text((98, y), line, font=fonts["body"], fill=(236, 239, 244))
        y += 42
    draw.rounded_rectangle((92, 470, 1180, 620), radius=36, fill=(255, 255, 255, 28))
    draw.text((120, 520), "Storyboard-Render mit TTS, Szenenkarten und Dialog-Entwurf", font=fonts["quote"], fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def create_end_card(output_path: Path, width: int, height: int, fonts: dict[str, ImageFont.ImageFont]) -> None:
    from PIL import ImageDraw

    image = create_background(width, height, (73, 185, 124))
    draw = ImageDraw.Draw(image)
    draw.text((92, 220), "Ende des Draft-Renderings", font=fonts["title"], fill="white")
    draw.text((98, 320), "Naechster Schritt: bessere Figuren-Namen, mehr Trainingsmaterial, optional echter Voice-Clone.", font=fonts["body"], fill=(235, 241, 244))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def remove_if_exists(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        if path.exists():
            path.unlink()


def run_ffmpeg_with_codec_fallback(command_factory, preferred_codec: str, output_path: Path) -> str:
    codecs_to_try = [preferred_codec]
    if preferred_codec != "libx264":
        codecs_to_try.append("libx264")

    last_exc: Exception | None = None
    for index, codec in enumerate(codecs_to_try):
        try:
            remove_if_exists(output_path)
            run_command(command_factory(codec), quiet=True)
            return codec
        except Exception as exc:
            last_exc = exc
            if index < len(codecs_to_try) - 1:
                warn(f"{output_path.name}: Encoding mit {codec} fehlgeschlagen, Fallback auf libx264.")
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError(f"FFmpeg-Encoding fehlgeschlagen: {output_path}")


def create_silent_segment(
    ffmpeg: Path,
    image_path: Path,
    output_mp4: Path,
    width: int,
    height: int,
    fps: int,
    duration: float,
    video_codec: str,
) -> None:
    def command(codec: str) -> list[str]:
        return [
            str(ffmpeg),
            "-hide_banner",
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-i",
            str(image_path),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t",
            f"{duration:.2f}",
            "-vf",
            f"scale={width}:{height},format=yuv420p",
            *ffmpeg_video_encode_args(codec, quality=20),
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(output_mp4),
        ]

    run_ffmpeg_with_codec_fallback(command, video_codec, output_mp4)


def create_line_segment(
    ffmpeg: Path,
    image_path: Path,
    audio_path: Path,
    output_mp4: Path,
    width: int,
    height: int,
    fps: int,
    audio_pad_seconds: float,
    video_codec: str,
) -> None:
    def command(codec: str) -> list[str]:
        return [
            str(ffmpeg),
            "-hide_banner",
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-i",
            str(image_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            (
                f"[0:v]scale={width}:{height},format=yuv420p[v];"
                f"[1:a]aresample=44100,"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"apad=pad_dur={audio_pad_seconds}[a]"
            ),
            "-map",
            "[v]",
            "-map",
            "[a]",
            *ffmpeg_video_encode_args(codec, quality=20),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-shortest",
            str(output_mp4),
        ]

    run_ffmpeg_with_codec_fallback(command, video_codec, output_mp4)


def create_original_scene_segment(
    ffmpeg: Path,
    scene_clip_path: Path,
    start_seconds: float,
    end_seconds: float,
    output_mp4: Path,
    width: int,
    height: int,
    video_codec: str,
) -> None:
    duration = max(0.12, float(end_seconds) - float(start_seconds))

    def command(codec: str) -> list[str]:
        return [
            str(ffmpeg),
            "-hide_banner",
            "-y",
            "-ss",
            f"{max(0.0, float(start_seconds)):.3f}",
            "-i",
            str(scene_clip_path),
            "-t",
            f"{duration:.3f}",
            "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p",
            *ffmpeg_video_encode_args(codec, quality=19),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(output_mp4),
        ]

    run_ffmpeg_with_codec_fallback(command, video_codec, output_mp4)


def create_lipsync_segment(
    ffmpeg: Path,
    background_image_path: Path,
    portrait_video_path: Path,
    audio_path: Path,
    output_mp4: Path,
    width: int,
    height: int,
    fps: int,
    audio_pad_seconds: float,
    video_codec: str,
) -> None:
    def command(codec: str) -> list[str]:
        return [
            str(ffmpeg),
            "-hide_banner",
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-i",
            str(background_image_path),
            "-i",
            str(portrait_video_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            (
                f"[0:v]scale={width}:{height},format=yuv420p[bg];"
                f"[1:v]scale={PORTRAIT_PANEL_W}:{PORTRAIT_PANEL_H}[portrait];"
                f"[bg][portrait]overlay={PORTRAIT_PANEL_X}:{PORTRAIT_PANEL_Y},format=yuv420p[v];"
                f"[2:a]aresample=44100,"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"apad=pad_dur={audio_pad_seconds}[a]"
            ),
            "-map",
            "[v]",
            "-map",
            "[a]",
            *ffmpeg_video_encode_args(codec, quality=20),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-shortest",
            str(output_mp4),
        ]

    run_ffmpeg_with_codec_fallback(command, video_codec, output_mp4)


def create_final_render(
    ffmpeg: Path,
    rendered_segments: list[Path],
    output_mp4: Path,
    video_codec: str,
) -> str:
    if not rendered_segments:
        raise RuntimeError("Keine Render-Segmente zum Zusammenfügen vorhanden.")

    concat_list_path = output_mp4.parent / f"{output_mp4.stem}_concat.txt"
    concat_lines = []
    for segment_path in rendered_segments:
        normalized = segment_path.resolve().as_posix().replace("'", r"'\''")
        concat_lines.append(f"file '{normalized}'")
    write_text(concat_list_path, "\n".join(concat_lines) + "\n")

    def command(codec: str) -> list[str]:
        return [
            str(ffmpeg),
            "-hide_banner",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            *ffmpeg_video_encode_args(codec, quality=20),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(output_mp4),
        ]

    try:
        return run_ffmpeg_with_codec_fallback(command, video_codec, output_mp4)
    finally:
        if concat_list_path.exists():
            concat_list_path.unlink()


def update_shotlist_with_render(
    shotlist_path: Path,
    shotlist: dict,
    draft_mp4: Path,
    final_mp4: Path,
    draft_manifest_path: Path,
    final_manifest_path: Path,
) -> None:
    shotlist["render_draft"] = str(draft_mp4)
    shotlist["render_final"] = str(final_mp4)
    shotlist["render_manifest"] = str(draft_manifest_path)
    shotlist["render_manifest_final"] = str(final_manifest_path)
    write_json(shotlist_path, shotlist)


def main() -> None:
    rerun_in_runtime()
    headline("Episode als Storyboard-Video rendern")
    cfg = load_config()
    ffmpeg = detect_tool(PROJECT_ROOT / "tools" / "ffmpeg" / "bin", "ffmpeg")
    video_codec = preferred_ffmpeg_video_codec(ffmpeg, cfg)
    shotlist_dir = resolve_project_path("generation/shotlists")
    shotlist_path = find_latest_shotlist(shotlist_dir)
    if shotlist_path is None:
        info("Keine Shotlist zum Rendern gefunden.")
        return

    shotlist = read_json(shotlist_path, {})
    episode_id = shotlist.get("episode_id", shotlist_path.stem)
    autosave_target = str(episode_id or "folge")
    mark_step_started(
        "13_render_episode",
        autosave_target,
        {"shotlist": str(shotlist_path), "episode_id": episode_id},
    )
    scenes = shotlist.get("scenes", [])
    if not scenes:
        info("Die Shotlist enthält keine Szenen.")
        return

    render_cfg = cfg.get("render", {})
    width = int(render_cfg.get("width", 1280))
    height = int(render_cfg.get("height", 720))
    fps = int(render_cfg.get("fps", 30))
    include_title_cards = bool(render_cfg.get("include_title_cards", False))
    audio_pad_seconds = float(render_cfg.get("audio_pad_seconds", 0.35))
    title_card_seconds = float(render_cfg.get("title_card_seconds", 2.5))
    closing_card_seconds = float(render_cfg.get("closing_card_seconds", 2.0))
    base_rate = int(render_cfg.get("voice_rate", 175))
    clone_cfg = cfg.get("cloning", {})
    allow_original_reuse = bool(clone_cfg.get("enable_original_line_reuse", False)) and str(
        shotlist.get("generation_mode", "")
    ).strip().lower() != "synthetic_preview"
    info(f"FFmpeg-Videoencoder: {video_codec}")

    char_map = read_json(resolve_project_path(cfg["paths"]["character_map"]), {"clusters": {}, "aliases": {}})
    voice_reference_library = build_voice_reference_library(cfg)
    original_line_library = build_original_line_library(cfg)
    all_characters = shotlist.get("focus_characters", [])[:]
    for scene in scenes:
        for character in scene.get("characters", []):
            if character not in all_characters:
                all_characters.append(character)
    ensure_foundation_training_ready(cfg, characters=all_characters, for_render=True)
    ensure_adapter_training_ready(cfg, characters=all_characters, for_render=True)
    adapter_status = adapter_training_status(cfg, characters=all_characters)
    ensure_fine_tune_training_ready(cfg, characters=all_characters, for_render=True)
    fine_tune_status = fine_tune_training_status(cfg, characters=all_characters)
    ensure_backend_fine_tune_ready(cfg, characters=all_characters, for_render=True)
    backend_status = backend_fine_tune_status(cfg, characters=all_characters)
    fonts = font_bundle()
    info(
        "Clone-Modi: "
        f"voice={'an' if clone_cfg.get('enable_voice_cloning', True) else 'aus'}, "
        f"face={'an' if clone_cfg.get('enable_face_clone', True) else 'aus'}, "
        f"lipsync={'an' if clone_cfg.get('enable_lipsync', True) else 'aus'}"
    )

    draft_root = resolve_project_path("generation/renders/drafts") / episode_id
    final_root = resolve_project_path("generation/renders/final") / episode_id
    cards_dir = draft_root / "cards"
    audio_dir = draft_root / "audio"
    portraits_dir = draft_root / "portraits"
    segments_dir = draft_root / "segments"
    voice_samples_dir = resolve_project_path(cfg["paths"].get("voice_samples", "characters/voice_samples"))
    voice_models_dir = resolve_project_path(cfg["paths"].get("voice_models", "characters/voice_models"))
    for directory in (cards_dir, audio_dir, portraits_dir, segments_dir, voice_samples_dir, voice_models_dir, final_root):
        directory.mkdir(parents=True, exist_ok=True)

    draft_manifest_path = draft_root / f"{episode_id}_render_manifest.json"
    final_manifest_path = final_root / f"{episode_id}_render_manifest.json"
    final_mp4 = final_root / f"{episode_id}_final.mp4"
    draft_mp4 = draft_root / f"{episode_id}_draft.mp4"
    if render_output_ready(final_mp4) and draft_manifest_path.exists() and final_manifest_path.exists():
        mark_step_completed(
            "13_render_episode",
            autosave_target,
            {
                "episode_id": episode_id,
                "final_render": str(final_mp4),
                "draft_render": str(draft_mp4),
                "segment_count": len(read_json(draft_manifest_path, {}).get("segments", []) or []),
            },
        )
        ok(f"Render bereits vorhanden: {final_mp4}")
        return

    prepared_voice_references: dict[str, Path | None] = {}

    def prepared_reference_wav(character: str) -> Path | None:
        if character in prepared_voice_references:
            return prepared_voice_references[character]
        if not useful_character_name(character):
            prepared_voice_references[character] = None
            return None
        selected_paths = select_voice_reference_paths(
            character,
            voice_reference_library,
            max_segments=int(clone_cfg.get("voice_reference_max_segments", 4)),
            target_seconds=float(clone_cfg.get("voice_reference_target_seconds", 16.0)),
        )
        reference_path = prepare_voice_reference_wav(
            ffmpeg,
            character,
            selected_paths,
            voice_samples_dir / f"{safe_filename_slug(character, 'speaker')}_reference.wav",
            sample_rate=int(clone_cfg.get("reference_audio_sample_rate", 24000)),
        )
        prepared_voice_references[character] = reference_path
        return reference_path

    voice_profiles_path = voice_samples_dir / "voice_profiles.json"
    existing_voice_profiles = read_json(voice_profiles_path, {})
    available_voice_ids = list_tts_voices() or [""]
    voice_profiles: dict[str, dict] = {}
    for character in all_characters:
        reference_wav = prepared_reference_wav(character)
        existing_profile = existing_voice_profiles.get(character, {})
        profile = estimate_voice_profile(reference_wav, character, available_voice_ids, base_rate)
        if existing_profile:
            profile["voice_id"] = existing_profile.get("voice_id", profile["voice_id"])
        voice_profiles[character] = profile
    write_json(voice_profiles_path, voice_profiles)
    voice_model_paths: dict[str, str] = {}
    for character, profile in voice_profiles.items():
        model_path = voice_models_dir / f"{safe_filename_slug(character, 'speaker')}_voice_model.json"
        retrieval_samples = sorted(
            original_line_library.get(character, []),
            key=lambda item: (-float(item.get("duration_seconds", 0.0)), str(item.get("segment_id", ""))),
        )[: int(clone_cfg.get("max_voice_model_samples", 48))]
        model_payload = {
            "character": character,
            "engine": requested_voice_clone_engine(clone_cfg),
            "model_type": "retrieval_profile",
            "profile": profile,
            "sample_count": len(retrieval_samples),
            "samples": retrieval_samples,
        }
        write_json(model_path, model_payload)
        voice_model_paths[character] = str(model_path)
    voice_assignments = assign_voices(all_characters, base_rate, voice_profiles)
    backend_character_index = backend_status.get("character_index", {}) if isinstance(backend_status.get("character_index"), dict) else {}
    backend_voice_profiles = {}
    for character in all_characters:
        status_row = backend_character_index.get(coalesce_text(character).lower(), {})
        backends = status_row.get("backends", {}) if isinstance(status_row.get("backends"), dict) else {}
        voice_backend = backends.get("voice", {}) if isinstance(backends.get("voice"), dict) else {}
        backend_voice_profiles[character] = {
            "voice_clone_ready": bool(status_row.get("voice_clone_ready", False) or voice_backend.get("voice_clone_ready", False)),
            "voice_quality_score": float(status_row.get("voice_quality_score", 0.0) or voice_backend.get("voice_quality_score", 0.0) or 0.0),
            "voice_duration_seconds": float(status_row.get("voice_duration_seconds", 0.0) or voice_backend.get("voice_duration_seconds", 0.0) or 0.0),
            "backend": str(voice_backend.get("backend", "")),
            "bundle_path": str(((voice_backend.get("artifacts", {}) if isinstance(voice_backend.get("artifacts"), dict) else {}).get("bundle_path", "")) or ""),
        }

    manifest = {
        "episode_id": episode_id,
        "episode_label": shotlist.get("episode_label", episode_id),
        "episode_title": shotlist.get("episode_title", ""),
        "display_title": shotlist.get("display_title", episode_id),
        "shotlist": str(shotlist_path),
        "adapter_training_summary": str(adapter_status.get("summary_path", "")) if adapter_status.get("summary_exists") else "",
        "fine_tune_training_summary": str(fine_tune_status.get("summary_path", "")) if fine_tune_status.get("summary_exists") else "",
        "backend_fine_tune_summary": str(backend_status.get("summary_path", "")) if backend_status.get("summary_exists") else "",
        "voice_profiles": str(voice_profiles_path),
        "voice_models": voice_model_paths,
        "backend_voice_profiles": backend_voice_profiles,
        "render_modes": {
            "voice_cloning": bool(clone_cfg.get("enable_voice_cloning", True)),
            "face_clone": bool(clone_cfg.get("enable_face_clone", True)),
            "lipsync": bool(clone_cfg.get("enable_lipsync", True)),
        },
        "segments": [],
    }
    rendered_segments: list[Path] = []
    autosave_state = load_step_autosave("13_render_episode", autosave_target)
    completed_segment_files = {
        str(path)
        for path in autosave_state.get("completed_segment_files", []) or []
        if str(path).strip()
    }

    if include_title_cards:
        title_card = cards_dir / "000_title.png"
        title_segment = segments_dir / "000_title.mp4"
        if not render_output_ready(title_segment):
            create_title_card(episode_id, shotlist, title_card, width, height, fonts)
            create_silent_segment(ffmpeg, title_card, title_segment, width, height, fps, title_card_seconds, video_codec)
        rendered_segments.append(title_segment)
        completed_segment_files.add(str(title_segment))
        save_step_autosave(
            "13_render_episode",
            autosave_target,
            {
                "status": "in_progress",
                "episode_id": episode_id,
                "completed_segment_files": sorted(completed_segment_files),
                "segment_count": len(rendered_segments),
            },
        )
        manifest["segments"].append({"type": "title", "file": str(title_segment), "duration_seconds": title_card_seconds})

    segment_index = 1
    for scene in scenes:
        reference_images = pick_reference_images(scene, char_map)
        for line_index, line in enumerate(scene.get("dialogue_lines", []), start=1):
            speaker, spoken_text = parse_speaker_line(line)
            voice = voice_assignments.get(speaker) or voice_assignments["erzähler"]
            backend_voice_profile = backend_voice_profiles.get(speaker, {})
            audio_path = audio_dir / f"{segment_index:03d}_{scene['scene_id']}_{line_index:02d}.wav"
            card_path = cards_dir / f"{segment_index:03d}_{scene['scene_id']}_{line_index:02d}.png"
            portrait_path = portraits_dir / f"{segment_index:03d}_{scene['scene_id']}_{line_index:02d}_portrait.mp4"
            segment_path = segments_dir / f"{segment_index:03d}_{scene['scene_id']}_{line_index:02d}.mp4"

            speaker_images = pick_character_context_paths(speaker, char_map, max_images=1) or pick_character_preview_paths(speaker, char_map, max_images=1)
            line_reference_images = speaker_images or reference_images
            voice_reference_wav = prepared_reference_wav(speaker)
            planned_source = scene_dialogue_source(scene, line_index - 1)
            retrieved_original = None
            if (
                allow_original_reuse
                and
                planned_source.get("type") == "original_line"
                and Path(str(planned_source.get("video_file", ""))).exists()
                and float(planned_source.get("end", 0.0) or 0.0) > float(planned_source.get("start", 0.0) or 0.0)
            ):
                retrieved_original = {
                    "audio_path": str(planned_source.get("audio_file", "")),
                    "scene_clip_path": str(planned_source.get("video_file", "")),
                    "start": float(planned_source.get("start", 0.0) or 0.0),
                    "end": float(planned_source.get("end", 0.0) or 0.0),
                    "segment_id": str(planned_source.get("segment_id", "")),
                    "match_score": 1.0,
                    "token_overlap": 1.0,
                    "planned_source": True,
                }
            elif allow_original_reuse:
                retrieved_original = select_retrieval_segment(speaker, spoken_text or line, original_line_library, clone_cfg)

            visual_mode = "static_card"
            face_reference_path = ""
            voice_meta: dict
            if render_output_ready(segment_path):
                voice_meta = {
                    "engine": "resume_existing_segment",
                    "speaker_name": speaker,
                    "reference_wav": str(voice_reference_wav) if voice_reference_wav else "",
                    "device": preferred_torch_device(cfg),
                    "voice_cloned": False,
                    "fallback_reason": "",
                }
                visual_mode = "resume_existing_segment"
            else:
                create_line_card(
                    coalesce_text(shotlist.get("episode_label", "")) or episode_id,
                    scene,
                    line,
                    line_reference_images or reference_images,
                    card_path,
                    width,
                    height,
                    fonts,
                )

                if retrieved_original is not None:
                    source_audio_path = Path(str(retrieved_original.get("audio_path", "")))
                    source_scene_clip = Path(str(retrieved_original.get("scene_clip_path", "")))
                    if source_audio_path.exists():
                        shutil.copy2(source_audio_path, audio_path)
                    create_original_scene_segment(
                        ffmpeg,
                        source_scene_clip,
                        float(retrieved_original.get("start", 0.0) or 0.0),
                        float(retrieved_original.get("end", 0.0) or 0.0),
                        segment_path,
                        width,
                        height,
                        video_codec,
                    )
                    frames = [Path(path) for path in retrieved_original.get("speaker_reference_frames", []) if Path(path).exists()]
                    face_reference_path = str(frames[0]) if frames else ""
                    visual_mode = "original_scene_reuse"
                    voice_meta = {
                        "engine": "planned_original_line_reuse" if retrieved_original.get("planned_source") else "original_sample_reuse",
                        "speaker_name": speaker,
                        "reference_wav": str(source_audio_path),
                        "device": preferred_torch_device(cfg),
                        "voice_cloned": True,
                        "fallback_reason": "",
                        "retrieval_score": float(retrieved_original.get("match_score", 0.0) or 0.0),
                        "retrieval_overlap": float(retrieved_original.get("token_overlap", 0.0) or 0.0),
                        "reused_scene_clip": str(source_scene_clip),
                        "reused_segment_id": str(retrieved_original.get("segment_id", "")),
                    }
                else:
                    voice_meta = synthesize_speech(
                        spoken_text or line,
                        audio_path,
                        speaker,
                        voice["voice_id"],
                        int(voice["rate"]),
                        voice_reference_wav,
                        clone_cfg,
                        cfg,
                    )

                    portrait_source_images = pick_character_preview_paths(speaker, char_map, max_images=4) or line_reference_images or reference_images
                    if (
                        bool(clone_cfg.get("enable_face_clone", True))
                        and bool(clone_cfg.get("enable_lipsync", True))
                        and portrait_source_images
                    ):
                        try:
                            face_reference_path = str(portrait_source_images[0])
                            create_audio_reactive_portrait_video(
                                ffmpeg,
                                portrait_source_images,
                                audio_path,
                                portrait_path,
                                fps,
                                video_codec,
                                clone_cfg,
                            )
                            create_lipsync_segment(
                                ffmpeg,
                                card_path,
                                portrait_path,
                                audio_path,
                                segment_path,
                                width,
                                height,
                                fps,
                                audio_pad_seconds,
                                video_codec,
                            )
                            visual_mode = "animated_reference_portrait"
                        except Exception as exc:
                            warn(f"Lip-Sync-Fallback fuer {speaker} fehlgeschlagen, nutze statische Karte: {exc}")
                            create_line_segment(ffmpeg, card_path, audio_path, segment_path, width, height, fps, audio_pad_seconds, video_codec)
                            visual_mode = "static_card"
                    else:
                        create_line_segment(ffmpeg, card_path, audio_path, segment_path, width, height, fps, audio_pad_seconds, video_codec)

            rendered_segments.append(segment_path)
            completed_segment_files.add(str(segment_path))
            save_step_autosave(
                "13_render_episode",
                autosave_target,
                {
                    "status": "in_progress",
                    "episode_id": episode_id,
                    "completed_segment_files": sorted(completed_segment_files),
                    "segment_count": len(rendered_segments),
                    "last_segment": str(segment_path),
                },
            )
            manifest["segments"].append(
                {
                    "type": "dialogue",
                    "scene_id": scene["scene_id"],
                    "speaker": speaker,
                    "text": spoken_text,
                    "audio_file": str(audio_path),
                    "audio_seconds": round(audio_duration_seconds(audio_path), 3),
                    "voice_engine": voice_meta.get("engine", "pyttsx3"),
                    "voice_reference_wav": voice_meta.get("reference_wav", ""),
                    "voice_cloned": bool(voice_meta.get("voice_cloned")),
                    "voice_fallback_reason": voice_meta.get("fallback_reason", ""),
                    "backend_voice_clone_ready": bool(backend_voice_profile.get("voice_clone_ready", False)),
                    "backend_voice_quality_score": float(backend_voice_profile.get("voice_quality_score", 0.0) or 0.0),
                    "backend_voice_duration_seconds": float(backend_voice_profile.get("voice_duration_seconds", 0.0) or 0.0),
                    "backend_voice_bundle": backend_voice_profile.get("bundle_path", ""),
                    "retrieval_score": float(voice_meta.get("retrieval_score", 0.0) or 0.0),
                    "retrieval_overlap": float(voice_meta.get("retrieval_overlap", 0.0) or 0.0),
                    "reused_segment_id": voice_meta.get("reused_segment_id", ""),
                    "reused_scene_clip": voice_meta.get("reused_scene_clip", ""),
                    "visual_mode": visual_mode,
                    "face_reference": face_reference_path,
                    "file": str(segment_path),
                }
            )
            segment_index += 1

    if include_title_cards:
        end_card = cards_dir / "999_end.png"
        end_segment = segments_dir / "999_end.mp4"
        if not render_output_ready(end_segment):
            create_end_card(end_card, width, height, fonts)
            create_silent_segment(ffmpeg, end_card, end_segment, width, height, fps, closing_card_seconds, video_codec)
        rendered_segments.append(end_segment)
        completed_segment_files.add(str(end_segment))
        save_step_autosave(
            "13_render_episode",
            autosave_target,
            {
                "status": "in_progress",
                "episode_id": episode_id,
                "completed_segment_files": sorted(completed_segment_files),
                "segment_count": len(rendered_segments),
            },
        )
        manifest["segments"].append({"type": "end", "file": str(end_segment), "duration_seconds": closing_card_seconds})

    final_video_codec = create_final_render(ffmpeg, rendered_segments, final_mp4, video_codec)
    shutil.copy2(final_mp4, draft_mp4)

    manifest["final_video_codec"] = final_video_codec
    manifest["draft_render"] = str(draft_mp4)
    manifest["final_render"] = str(final_mp4)
    write_json(draft_manifest_path, manifest)
    write_json(final_manifest_path, manifest)
    update_shotlist_with_render(shotlist_path, shotlist, draft_mp4, final_mp4, draft_manifest_path, final_manifest_path)
    mark_step_completed(
        "13_render_episode",
        autosave_target,
        {
            "episode_id": episode_id,
            "final_render": str(final_mp4),
            "draft_render": str(draft_mp4),
            "completed_segment_files": sorted(completed_segment_files),
            "segment_count": len(rendered_segments),
        },
    )
    ok(f"Final-Render erstellt: {final_mp4}")
    ok(f"Draft-Render erstellt: {draft_mp4}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
