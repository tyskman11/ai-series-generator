#!/usr/bin/env python3
from __future__ import annotations

import os
import math
import re
import shutil
import wave
from functools import lru_cache
from pathlib import Path

from pipeline_common import (
    PROJECT_ROOT,
    detect_tool,
    error,
    ffmpeg_video_encode_args,
    headline,
    info,
    load_config,
    ok,
    preferred_ffmpeg_video_codec,
    preferred_torch_device,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
    run_command,
    warn,
    write_json,
)

PORTRAIT_PANEL_X = 860
PORTRAIT_PANEL_Y = 196
PORTRAIT_PANEL_W = 330
PORTRAIT_PANEL_H = 230


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


def useful_character_name(name: str) -> bool:
    cleaned = (name or "").strip()
    if not cleaned:
        return False
    if cleaned.lower() in {"unbekannt", "noface", "erzähler"}:
        return False
    return re.match(r"^(speaker|figur|face)_\d+$", cleaned.lower()) is None


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


def assign_voices(characters: list[str], base_rate: int) -> dict[str, dict]:
    voice_ids = list_tts_voices() or [""]
    assignments = {}
    for index, character in enumerate(characters):
        assignments[character] = {
            "voice_id": voice_ids[index % len(voice_ids)],
            "rate": base_rate + ((index % 3) - 1) * 10,
        }
    assignments["erzähler"] = {"voice_id": voice_ids[0], "rate": base_rate}
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
        mouth_overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
        mouth_draw = ImageDraw.Draw(mouth_overlay)
        mouth_left = int(width * 0.31)
        mouth_right = int(width * 0.69)
        mouth_center_y = int(height * 0.73)
        mouth_height = max(8, int((height * 0.07) + (height * 0.08 * openness)))
        mouth_draw.ellipse(
            (
                mouth_left,
                mouth_center_y - mouth_height // 2,
                mouth_right,
                mouth_center_y + mouth_height // 2,
            ),
            fill=(28, 6, 10, int(120 + 90 * openness)),
        )
        mouth_draw.ellipse(
            (
                mouth_left + 18,
                mouth_center_y - max(4, mouth_height // 4),
                mouth_right - 18,
                mouth_center_y + max(4, mouth_height // 4),
            ),
            outline=(255, 178, 178, int(70 * openness)),
            width=2,
        )
        result = Image.alpha_composite(result, mouth_overlay)

        jaw_region = result.crop((int(width * 0.18), int(height * 0.58), int(width * 0.82), height))
        stretched = jaw_region.resize(
            (jaw_region.width, jaw_region.height + int(jaw_pixels * openness)),
            resample=Image.Resampling.BICUBIC,
        )
        jaw_canvas = Image.new("RGBA", result.size, (0, 0, 0, 0))
        jaw_canvas.paste(stretched, (int(width * 0.18), int(height * 0.58)))
        jaw_canvas.putalpha(int(45 * openness))
        result = Image.alpha_composite(result, jaw_canvas)

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
    face_image_path: Path,
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

    frame_dir = output_mp4.parent / f"{output_mp4.stem}_frames"
    shutil.rmtree(frame_dir, ignore_errors=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    base_face = Image.open(face_image_path).convert("RGB")
    envelope = audio_envelope(audio_path, fps)
    frame_count = max(1, len(envelope))

    for frame_index in range(frame_count):
        openness = min(1.0, envelope[frame_index] * sensitivity)
        phase = frame_index / max(1, frame_count - 1)
        zoom_variation = zoom + 0.015 * math.sin(frame_index / 11.0)
        bob_y = int(4 * math.sin(frame_index / 7.5))
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
    from PIL import Image, ImageDraw

    speaker, spoken_text = parse_speaker_line(line_text)
    accent = palette_for_name(speaker)
    image = create_background(width, height, accent)
    draw = ImageDraw.Draw(image)

    draw.text((70, 50), episode_id.upper(), font=fonts["small"], fill=(215, 223, 238))
    draw.text((70, 82), scene.get("title", "Szene"), font=fonts["title"], fill="white")
    subtitle = f"{scene.get('beat', '')} | {scene.get('location', '')} | {scene.get('mood', '')}"
    draw.text((72, 150), subtitle, font=fonts["subtitle"], fill=(205, 214, 230))

    summary_box = (72, 205, 760, 380)
    draw.rounded_rectangle(summary_box, radius=28, fill=(255, 255, 255, 28), outline=(255, 255, 255, 48), width=2)
    summary_lines = wrap_text(draw, scene.get("summary", ""), fonts["body"], 640)
    y = 228
    for line in summary_lines[:5]:
        draw.text((98, y), line, font=fonts["body"], fill=(236, 239, 244))
        y += 34

    chip_x = 72
    chip_y = 398
    for character in scene.get("characters", [])[:3]:
        chip_width = draw_chip(draw, chip_x, chip_y, character, fonts["small"], palette_for_name(character))
        chip_x += chip_width + 14

    quote_box = (72, 470, 1210, 650)
    draw.rounded_rectangle(quote_box, radius=32, fill=(11, 14, 22, 210), outline=accent + (255,), width=4)
    draw.text((102, 500), speaker.upper(), font=fonts["subtitle"], fill=accent)
    quote_lines = wrap_text(draw, spoken_text, fonts["quote"], 1060)
    y = 548
    for line in quote_lines[:3]:
        draw.text((102, y), line, font=fonts["quote"], fill="white")
        y += 44

    right_panel_x = PORTRAIT_PANEL_X
    draw.rounded_rectangle((right_panel_x - 10, 185, 1210, 440), radius=28, fill=(255, 255, 255, 20))
    paste_reference_images(image, reference_images, right_panel_x, PORTRAIT_PANEL_Y, PORTRAIT_PANEL_W, PORTRAIT_PANEL_H)

    prompt_lines = wrap_text(draw, scene.get("prompt", ""), fonts["small"], 320)
    prompt_y = 660
    for line in prompt_lines[:2]:
        draw.text((860, prompt_y), line, font=fonts["small"], fill=(220, 226, 235))
        prompt_y += 26

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
    draw.text((96, 120), "AI Serien Draft", font=fonts["subtitle"], fill=(214, 223, 238))
    draw.text((92, 168), episode_id.upper(), font=fonts["title"], fill="white")
    focus = ", ".join(shotlist.get("focus_characters", [])[:3]) or "Neue Figuren"
    keywords = ", ".join(shotlist.get("keywords", [])[:6])
    body = f"Hauptfiguren: {focus}\nThemen: {keywords}"
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

    def command(codec: str) -> list[str]:
        cmd = [str(ffmpeg), "-hide_banner", "-y"]
        for segment_path in rendered_segments:
            cmd.extend(["-i", str(segment_path)])
        filter_inputs = "".join(f"[{index}:v:0][{index}:a:0]" for index in range(len(rendered_segments)))
        cmd.extend(
            [
                "-filter_complex",
                f"{filter_inputs}concat=n={len(rendered_segments)}:v=1:a=1[v][a]",
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
                str(output_mp4),
            ]
        )
        return cmd

    return run_ffmpeg_with_codec_fallback(command, video_codec, output_mp4)


def update_shotlist_with_render(shotlist_path: Path, shotlist: dict, output_mp4: Path, manifest_path: Path) -> None:
    shotlist["render_draft"] = str(output_mp4)
    shotlist["render_manifest"] = str(manifest_path)
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
    scenes = shotlist.get("scenes", [])
    if not scenes:
        info("Die Shotlist enthält keine Szenen.")
        return

    render_cfg = cfg.get("render", {})
    width = int(render_cfg.get("width", 1280))
    height = int(render_cfg.get("height", 720))
    fps = int(render_cfg.get("fps", 30))
    audio_pad_seconds = float(render_cfg.get("audio_pad_seconds", 0.35))
    title_card_seconds = float(render_cfg.get("title_card_seconds", 2.5))
    closing_card_seconds = float(render_cfg.get("closing_card_seconds", 2.0))
    base_rate = int(render_cfg.get("voice_rate", 175))
    clone_cfg = cfg.get("cloning", {})
    info(f"FFmpeg-Videoencoder: {video_codec}")

    char_map = read_json(resolve_project_path(cfg["paths"]["character_map"]), {"clusters": {}, "aliases": {}})
    voice_reference_library = build_voice_reference_library(cfg)
    all_characters = shotlist.get("focus_characters", [])[:]
    for scene in scenes:
        for character in scene.get("characters", []):
            if character not in all_characters:
                all_characters.append(character)
    voice_assignments = assign_voices(all_characters, base_rate)
    fonts = font_bundle()
    info(
        "Clone-Modi: "
        f"voice={'an' if clone_cfg.get('enable_voice_cloning', True) else 'aus'}, "
        f"face={'an' if clone_cfg.get('enable_face_clone', True) else 'aus'}, "
        f"lipsync={'an' if clone_cfg.get('enable_lipsync', True) else 'aus'}"
    )

    render_root = resolve_project_path("generation/renders/drafts") / episode_id
    cards_dir = render_root / "cards"
    audio_dir = render_root / "audio"
    portraits_dir = render_root / "portraits"
    segments_dir = render_root / "segments"
    voice_samples_dir = resolve_project_path(cfg["paths"].get("voice_samples", "characters/voice_samples"))
    for directory in (cards_dir, audio_dir, portraits_dir, segments_dir, voice_samples_dir):
        directory.mkdir(parents=True, exist_ok=True)

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

    manifest = {
        "episode_id": episode_id,
        "shotlist": str(shotlist_path),
        "render_modes": {
            "voice_cloning": bool(clone_cfg.get("enable_voice_cloning", True)),
            "face_clone": bool(clone_cfg.get("enable_face_clone", True)),
            "lipsync": bool(clone_cfg.get("enable_lipsync", True)),
        },
        "segments": [],
    }
    rendered_segments: list[Path] = []

    title_card = cards_dir / "000_title.png"
    title_segment = segments_dir / "000_title.mp4"
    create_title_card(episode_id, shotlist, title_card, width, height, fonts)
    create_silent_segment(ffmpeg, title_card, title_segment, width, height, fps, title_card_seconds, video_codec)
    rendered_segments.append(title_segment)
    manifest["segments"].append({"type": "title", "file": str(title_segment), "duration_seconds": title_card_seconds})

    segment_index = 1
    for scene in scenes:
        reference_images = pick_reference_images(scene, char_map)
        for line_index, line in enumerate(scene.get("dialogue_lines", []), start=1):
            speaker, spoken_text = parse_speaker_line(line)
            voice = voice_assignments.get(speaker) or voice_assignments["erzähler"]
            audio_path = audio_dir / f"{segment_index:03d}_{scene['scene_id']}_{line_index:02d}.wav"
            card_path = cards_dir / f"{segment_index:03d}_{scene['scene_id']}_{line_index:02d}.png"
            portrait_path = portraits_dir / f"{segment_index:03d}_{scene['scene_id']}_{line_index:02d}_portrait.mp4"
            segment_path = segments_dir / f"{segment_index:03d}_{scene['scene_id']}_{line_index:02d}.mp4"

            speaker_images = pick_character_preview_paths(speaker, char_map, max_images=1)
            line_reference_images = speaker_images or reference_images
            voice_reference_wav = prepared_reference_wav(speaker)

            create_line_card(episode_id, scene, line, line_reference_images or reference_images, card_path, width, height, fonts)
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

            visual_mode = "static_card"
            face_reference_path = ""
            if (
                bool(clone_cfg.get("enable_face_clone", True))
                and bool(clone_cfg.get("enable_lipsync", True))
                and line_reference_images
            ):
                try:
                    face_reference_path = str(line_reference_images[0])
                    create_audio_reactive_portrait_video(
                        ffmpeg,
                        line_reference_images[0],
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
                    visual_mode = "audio_reactive_face_clone"
                except Exception as exc:
                    warn(f"Lip-Sync-Fallback fuer {speaker} fehlgeschlagen, nutze statische Karte: {exc}")
                    create_line_segment(ffmpeg, card_path, audio_path, segment_path, width, height, fps, audio_pad_seconds, video_codec)
                    visual_mode = "static_card"
            else:
                create_line_segment(ffmpeg, card_path, audio_path, segment_path, width, height, fps, audio_pad_seconds, video_codec)

            rendered_segments.append(segment_path)
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
                    "visual_mode": visual_mode,
                    "face_reference": face_reference_path,
                    "file": str(segment_path),
                }
            )
            segment_index += 1

    end_card = cards_dir / "999_end.png"
    end_segment = segments_dir / "999_end.mp4"
    create_end_card(end_card, width, height, fonts)
    create_silent_segment(ffmpeg, end_card, end_segment, width, height, fps, closing_card_seconds, video_codec)
    rendered_segments.append(end_segment)
    manifest["segments"].append({"type": "end", "file": str(end_segment), "duration_seconds": closing_card_seconds})

    output_mp4 = render_root / f"{episode_id}_draft.mp4"
    final_video_codec = create_final_render(ffmpeg, rendered_segments, output_mp4, video_codec)

    manifest_path = render_root / f"{episode_id}_render_manifest.json"
    manifest["final_video_codec"] = final_video_codec
    write_json(manifest_path, manifest)
    update_shotlist_with_render(shotlist_path, shotlist, output_mp4, manifest_path)
    ok(f"Draft-Render erstellt: {output_mp4}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
