#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

from pipeline_common import (
    LiveProgressReporter,
    PROJECT_ROOT,
    detect_tool,
    error,
    headline,
    info,
    latest_matching_file,
    load_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a draft preview and a final voiced storyboard episode.")
    parser.add_argument("--episode-id", help="Target a specific episode ID such as episode_09 or folge_09.")
    parser.add_argument("--force", action="store_true", help="Recreate existing draft and final renders.")
    return parser.parse_args()


def find_latest_shotlist(shotlist_dir: Path) -> Path | None:
    episode_id = os.environ.get("SERIES_RENDER_EPISODE", "").strip()
    if episode_id:
        candidate = shotlist_dir / f"{episode_id}.json"
        if candidate.exists():
            return candidate
    return latest_matching_file(shotlist_dir, "*.json")


def storyboard_assets_root(cfg: dict, episode_id: str) -> Path:
    raw_root = str((cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}).get("storyboard_assets", "generation/storyboard_assets"))
    candidate = Path(raw_root)
    base_root = candidate if candidate.is_absolute() else resolve_project_path(raw_root)
    return base_root / episode_id


def scene_frame_candidates(assets_root: Path, scene_id: str) -> list[tuple[str, Path]]:
    return [
        ("backend_frame", assets_root / scene_id / "frame.png"),
        ("backend_preview", assets_root / scene_id / "preview.jpg"),
        ("storyboard_asset", assets_root / f"{scene_id}.png"),
        ("storyboard_asset", assets_root / f"{scene_id}.jpg"),
    ]


def first_existing_scene_frame(assets_root: Path, scene_id: str) -> tuple[str, Path] | None:
    for source_type, candidate in scene_frame_candidates(assets_root, scene_id):
        if candidate.exists() and candidate.stat().st_size > 0:
            return source_type, candidate
    return None


def generated_storyboard_scene_frame(cfg: dict, episode_id: str, scene_id: str) -> Path | None:
    assets_root = storyboard_assets_root(cfg, episode_id)
    match = first_existing_scene_frame(assets_root, scene_id)
    return match[1] if match else None


def fit_image(path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)


def theme_color(scene: dict, scene_index: int) -> tuple[int, int, int]:
    source = " ".join(
        [
            str(scene.get("title", "")),
            str(scene.get("beat", "")),
            str(scene.get("mood", "")),
            str(scene.get("summary", "")),
        ]
    ).lower()
    if not source:
        source = f"scene-{scene_index}"
    seed = sum(ord(char) for char in source)
    return (56 + (seed % 120), 72 + ((seed // 5) % 112), 96 + ((seed // 9) % 104))


def draw_multiline_block(
    draw: ImageDraw.ImageDraw,
    *,
    lines: list[str],
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    spacing: int,
) -> int:
    current_y = y
    for line in lines:
        draw.text((x, current_y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, current_y), line, font=font)
        current_y = bbox[3] + spacing
    return current_y


def placeholder_scene_frame(scene: dict, width: int, height: int, scene_index: int) -> Image.Image:
    accent = theme_color(scene, scene_index)
    image = Image.new("RGB", (width, height), (18, 24, 32))
    draw = ImageDraw.Draw(image)
    overlay = Image.new("RGBA", (width, height), accent + (0,))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle((0, 0, width, height), fill=accent + (42,))
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()
    draw.rectangle((60, 60, width - 60, height - 60), outline=(230, 232, 236), width=2)
    draw.rectangle((60, height - 230, width - 60, height - 60), fill=(10, 12, 16))
    draw.text((90, 90), f"{scene.get('scene_id', 'scene')} | {scene.get('title', 'Storyboard Scene')}", font=title_font, fill=(245, 245, 245))

    characters = ", ".join(scene.get("characters", []) or []) or "No characters assigned"
    summary = str(scene.get("summary", "")).strip() or "No scene summary available."
    mood = str(scene.get("mood", "")).strip() or "n/a"
    location = str(scene.get("location", "")).strip() or "n/a"
    dialogue_lines = scene.get("dialogue_lines", []) if isinstance(scene.get("dialogue_lines", []), list) else []
    excerpt = dialogue_lines[:3] or ["No dialogue lines available yet."]
    wrapped_summary = textwrap.wrap(summary, width=78)[:3]
    wrapped_dialogue: list[str] = []
    for line in excerpt:
        wrapped_dialogue.extend(textwrap.wrap(str(line), width=72)[:2] or [str(line)])
    content_lines = [
        f"Characters: {characters}",
        f"Location: {location}",
        f"Mood: {mood}",
        "",
        "Summary:",
        *wrapped_summary,
        "",
        "Dialogue Preview:",
        *wrapped_dialogue,
    ]
    draw_multiline_block(
        draw,
        lines=content_lines,
        x=90,
        y=height - 210,
        font=body_font,
        fill=(240, 240, 240),
        spacing=5,
    )
    return image


def build_scene_frame(scene: dict, scene_index: int, assets_root: Path, width: int, height: int) -> tuple[Image.Image, dict]:
    scene_id = str(scene.get("scene_id", "")).strip() or f"scene_{scene_index + 1:04d}"
    source = first_existing_scene_frame(assets_root, scene_id)
    meta = {
        "scene_id": scene_id,
        "asset_source_type": "placeholder",
        "asset_source_path": "",
    }
    if source is None:
        return placeholder_scene_frame(scene, width, height, scene_index), meta
    source_type, source_path = source
    meta["asset_source_type"] = source_type
    meta["asset_source_path"] = str(source_path)
    return fit_image(source_path, width, height), meta


def compose_scene_card(scene: dict, scene_index: int, assets_root: Path, width: int, height: int) -> tuple[Image.Image, dict]:
    base, meta = build_scene_frame(scene, scene_index, assets_root, width, height)
    image = base.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, height - 190, width, height), fill=(8, 10, 14, 196))
    draw.rectangle((0, 0, width, 92), fill=(8, 10, 14, 182))
    accent = theme_color(scene, scene_index)
    draw.rectangle((0, 0, width, 8), fill=accent + (255,))
    merged = Image.alpha_composite(image, overlay).convert("RGB")

    draw = ImageDraw.Draw(merged)
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()
    scene_id = meta["scene_id"]
    title = str(scene.get("title", "")).strip() or "Storyboard Scene"
    characters = ", ".join(scene.get("characters", []) or []) or "No characters"
    summary = str(scene.get("summary", "")).strip()
    dialogue_lines = scene.get("dialogue_lines", []) if isinstance(scene.get("dialogue_lines", []), list) else []
    dialogue_preview = " | ".join(str(line) for line in dialogue_lines[:2]) or "No dialogue preview"

    draw.text((40, 26), f"{scene_id} | {title}", font=title_font, fill=(248, 248, 248))
    footer_lines = [
        f"Characters: {characters}",
        f"Summary: {textwrap.shorten(summary, width=120, placeholder='...') if summary else 'n/a'}",
        f"Dialogue: {textwrap.shorten(dialogue_preview, width=120, placeholder='...')}",
        f"Source: {meta['asset_source_type']}",
    ]
    draw_multiline_block(
        draw,
        lines=footer_lines,
        x=40,
        y=height - 165,
        font=body_font,
        fill=(242, 242, 242),
        spacing=5,
    )
    return merged, meta


def title_card(width: int, height: int, episode_id: str, display_title: str, episode_title: str) -> Image.Image:
    image = Image.new("RGB", (width, height), ImageColor.getrgb("#0f1720"))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, height), fill=(15, 23, 32))
    draw.rectangle((80, 80, width - 80, height - 80), outline=(240, 244, 248), width=2)
    font = ImageFont.load_default()
    lines = [
        "AI Series Preview Render",
        "",
        display_title or episode_id,
        episode_title or "",
        "",
        "Storyboard-driven local preview video",
    ]
    draw_multiline_block(draw, lines=lines, x=120, y=150, font=font, fill=(248, 248, 248), spacing=10)
    return image


def closing_card(width: int, height: int, display_title: str, scene_count: int) -> Image.Image:
    image = Image.new("RGB", (width, height), ImageColor.getrgb("#121212"))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, height), fill=(18, 18, 18))
    font = ImageFont.load_default()
    lines = [
        display_title or "Preview Episode",
        "",
        f"Rendered storyboard scenes: {scene_count}",
        "Draft and final preview written locally",
    ]
    draw_multiline_block(draw, lines=lines, x=130, y=220, font=font, fill=(244, 244, 244), spacing=12)
    return image


def safe_duration_seconds(scene: dict) -> float:
    duration = float(scene.get("estimated_runtime_seconds", 0.0) or 0.0)
    return max(2.4, min(8.0, duration if duration > 0 else 3.8))


def render_output_ready(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def scene_dialogue_source(scene: dict, line_index: int) -> dict:
    sources = scene.get("dialogue_sources", []) if isinstance(scene.get("dialogue_sources", []), list) else []
    if 0 <= line_index < len(sources) and isinstance(sources[line_index], dict):
        return sources[line_index]
    return {}


def text_similarity_score(expected_text: str, candidate_text: str) -> tuple[float, float]:
    left_tokens = {token.lower() for token in str(expected_text).replace(".", " ").replace(",", " ").split() if token.strip()}
    right_tokens = {token.lower() for token in str(candidate_text).replace(".", " ").replace(",", " ").split() if token.strip()}
    overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    ratio = difflib.SequenceMatcher(None, str(expected_text).lower(), str(candidate_text).lower()).ratio()
    return (ratio * 0.65) + (overlap * 0.35), overlap


def select_retrieval_segment(character_name: str, line_text: str, library: dict, cfg: dict) -> dict | None:
    if not bool(cfg.get("enable_original_line_reuse", False)):
        return None
    threshold = float(cfg.get("original_line_similarity_threshold", 0.74) or 0.74)
    min_overlap = float(cfg.get("original_line_min_token_overlap", 0.34) or 0.34)
    best_match: dict | None = None
    best_score = -1.0
    for entry in library.get(character_name, []) if isinstance(library.get(character_name, []), list) else []:
        score, overlap = text_similarity_score(line_text, str(entry.get("text", "")))
        if score < threshold or overlap < min_overlap:
            continue
        if score > best_score:
            best_score = score
            best_match = dict(entry)
            best_match["match_score"] = round(score, 4)
            best_match["token_overlap"] = round(overlap, 4)
    return best_match


def clean_text(value: object) -> str:
    return str(value or "").strip()


def usable_library_speaker_name(name: str) -> bool:
    lowered = clean_text(name).lower()
    if not lowered:
        return False
    if lowered in {"unknown", "speaker_unknown", "no face", "noface", "ignore", "ignored"}:
        return False
    if lowered.startswith("speaker_"):
        return False
    return True


def transcript_episode_id(path: Path) -> str:
    suffix = "_segments"
    stem = path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def linked_segment_index(cfg: dict) -> dict[str, dict[str, dict]]:
    linked_root = resolve_project_path(cfg["paths"]["linked_segments"])
    index: dict[str, dict[str, dict]] = {}
    for linked_path in sorted(linked_root.glob("*_linked_segments.json")):
        episode_id = linked_path.stem.replace("_linked_segments", "")
        rows = read_json(linked_path, [])
        episode_index: dict[str, dict] = {}
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            segment_id = clean_text(row.get("segment_id", ""))
            if segment_id:
                episode_index[segment_id] = row
        if episode_index:
            index[episode_id] = episode_index
    return index


def build_original_line_library(cfg: dict) -> dict[str, list[dict]]:
    transcript_root = resolve_project_path(cfg["paths"]["speaker_transcripts"])
    scene_root = resolve_project_path(cfg["paths"]["scene_clips"])
    linked_rows = linked_segment_index(cfg)
    library: dict[str, list[dict]] = {}
    for transcript_path in sorted(transcript_root.glob("*_segments.json")):
        episode_id = transcript_episode_id(transcript_path)
        episode_linked = linked_rows.get(episode_id, {})
        rows = read_json(transcript_path, [])
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            segment_id = clean_text(row.get("segment_id", ""))
            linked = episode_linked.get(segment_id, {})
            speaker_name = clean_text(linked.get("speaker_name", ""))
            text = clean_text(linked.get("text", "") or row.get("text", ""))
            if not usable_library_speaker_name(speaker_name) or not text:
                continue
            scene_id = clean_text(row.get("scene_id", "") or linked.get("scene_id", ""))
            scene_clip_path = scene_root / episode_id / f"{scene_id}.mp4" if scene_id else Path()
            library.setdefault(speaker_name, []).append(
                {
                    "episode_id": episode_id,
                    "scene_id": scene_id,
                    "segment_id": segment_id,
                    "text": text,
                    "audio_path": clean_text(row.get("audio_file", "")),
                    "scene_clip_path": str(scene_clip_path) if scene_id else "",
                    "start": float(row.get("start", 0.0) or 0.0),
                    "end": float(row.get("end", 0.0) or 0.0),
                }
            )
    return library


def build_voice_lookup(cfg: dict) -> dict[str, dict]:
    voice_map = read_json(resolve_project_path(cfg["paths"]["voice_map"]), {"clusters": {}, "aliases": {}})
    lookup: dict[str, dict] = {}
    for cluster_id, payload in (voice_map.get("clusters", {}) or {}).items():
        if not isinstance(payload, dict):
            continue
        speaker_name = clean_text(payload.get("name", ""))
        if not speaker_name:
            continue
        lookup.setdefault(
            speaker_name,
            {
                "cluster_id": cluster_id,
                "name": speaker_name,
                "linked_face_cluster": clean_text(payload.get("linked_face_cluster", "")),
                "auto_named": bool(payload.get("auto_named", False)),
            },
        )
    return lookup


def parse_dialogue_line(raw_line: str, source: dict) -> tuple[str, str]:
    line = clean_text(raw_line)
    fallback_speaker = clean_text(source.get("speaker", ""))
    if ":" in line:
        speaker, text = line.split(":", 1)
        return clean_text(speaker) or fallback_speaker or "Narrator", clean_text(text)
    return fallback_speaker or "Narrator", line


def estimate_dialogue_duration_seconds(text: str, voice_rate: int, audio_pad_seconds: float) -> float:
    tokens = re.findall(r"\w+", clean_text(text), flags=re.UNICODE)
    word_count = max(1, len(tokens))
    punctuation_pauses = sum(clean_text(text).count(mark) for mark in ".!?;:")
    words_per_minute = max(90, int(voice_rate or 175))
    spoken_seconds = (word_count / words_per_minute) * 60.0
    pause_seconds = min(1.6, punctuation_pauses * 0.12)
    return max(0.9, spoken_seconds + pause_seconds + max(0.12, float(audio_pad_seconds or 0.0) * 0.5))


def build_scene_voice_plan(
    scene: dict,
    scene_duration_seconds: float,
    scene_start_seconds: float,
    original_line_library: dict[str, list[dict]],
    voice_lookup: dict[str, dict],
    cloning_cfg: dict,
    render_cfg: dict,
) -> list[dict]:
    dialogue_lines = scene.get("dialogue_lines", []) if isinstance(scene.get("dialogue_lines", []), list) else []
    if not dialogue_lines:
        return []
    voice_rate = int(render_cfg.get("voice_rate", 175) or 175)
    audio_pad_seconds = float(render_cfg.get("audio_pad_seconds", 0.35) or 0.35)
    prepared: list[dict] = []
    for line_index, raw_line in enumerate(dialogue_lines):
        source = scene_dialogue_source(scene, line_index)
        speaker_name, line_text = parse_dialogue_line(str(raw_line), source)
        if not line_text:
            continue
        retrieval_segment: dict | None = None
        if clean_text(source.get("type", "")) == "original_line" and clean_text(source.get("segment_id", "")):
            retrieval_segment = {
                "segment_id": clean_text(source.get("segment_id", "")),
                "text": clean_text(source.get("text", "") or line_text),
                "audio_path": clean_text(source.get("audio_file", "")),
                "scene_clip_path": clean_text(source.get("video_file", "")),
                "start": float(source.get("start", 0.0) or 0.0),
                "end": float(source.get("end", 0.0) or 0.0),
                "match_score": 1.0,
                "token_overlap": 1.0,
            }
        elif usable_library_speaker_name(speaker_name):
            retrieval_segment = select_retrieval_segment(speaker_name, line_text, original_line_library, cloning_cfg)
        prepared.append(
            {
                "line_index": line_index,
                "speaker_name": speaker_name,
                "text": line_text,
                "source": source,
                "retrieval_segment": retrieval_segment or {},
                "voice_profile": voice_lookup.get(speaker_name, {}),
                "raw_duration_seconds": estimate_dialogue_duration_seconds(line_text, voice_rate, audio_pad_seconds),
            }
        )

    raw_total = sum(float(row.get("raw_duration_seconds", 0.0) or 0.0) for row in prepared)
    scale = 1.0
    if raw_total > max(0.0, scene_duration_seconds):
        scale = max(0.35, scene_duration_seconds / max(raw_total, 0.001))

    cursor = float(scene_start_seconds)
    scene_end = float(scene_start_seconds) + max(0.0, float(scene_duration_seconds))
    plan: list[dict] = []
    for row in prepared:
        duration = round(float(row["raw_duration_seconds"]) * scale, 3)
        start_seconds = round(cursor, 3)
        end_seconds = round(min(scene_end, cursor + duration), 3)
        retrieval_segment = row["retrieval_segment"] if isinstance(row["retrieval_segment"], dict) else {}
        voice_profile = row["voice_profile"] if isinstance(row["voice_profile"], dict) else {}
        plan.append(
            {
                "line_index": int(row["line_index"]),
                "speaker_name": row["speaker_name"],
                "text": row["text"],
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "estimated_duration_seconds": round(max(0.0, end_seconds - start_seconds), 3),
                "audio_strategy": "reuse_original_segment" if retrieval_segment else "synthesize_preview_tts",
                "source_type": clean_text((row["source"] or {}).get("type", "")) or "generated",
                "source_segment_id": clean_text((row["source"] or {}).get("segment_id", "")),
                "voice_profile": {
                    "cluster_id": clean_text(voice_profile.get("cluster_id", "")),
                    "name": clean_text(voice_profile.get("name", "")),
                    "linked_face_cluster": clean_text(voice_profile.get("linked_face_cluster", "")),
                    "auto_named": bool(voice_profile.get("auto_named", False)),
                },
                "retrieval_segment": retrieval_segment,
            }
        )
        cursor += duration
    return plan


def format_srt_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, int(round(float(seconds or 0.0) * 1000.0)))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def render_subtitle_preview_srt(voice_plan_lines: list[dict]) -> str:
    blocks: list[str] = []
    for index, line in enumerate(voice_plan_lines, start=1):
        speaker_name = clean_text(line.get("speaker_name", "")) or "Narrator"
        text = clean_text(line.get("text", ""))
        if not text:
            continue
        start_seconds = float(line.get("start_seconds", 0.0) or 0.0)
        end_seconds = max(start_seconds + 0.4, float(line.get("end_seconds", start_seconds + 0.4) or (start_seconds + 0.4)))
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_timestamp(start_seconds)} --> {format_srt_timestamp(end_seconds)}",
                    f"{speaker_name}: {text}",
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def resolve_system_voice_id(default_voice_id: str, voices: list[dict], require_german: bool = True) -> str:
    if not require_german:
        return default_voice_id
    for voice in voices:
        if str(voice.get("id", "")) == default_voice_id and bool(voice.get("is_german", False)):
            return default_voice_id
    for voice in voices:
        languages = [str(value).lower() for value in voice.get("languages", [])] if isinstance(voice.get("languages", []), list) else []
        if bool(voice.get("is_german", False)) or any(language.startswith("de") for language in languages):
            return str(voice.get("id", default_voice_id))
    return default_voice_id


def describe_system_voices(engine) -> list[dict]:
    payloads: list[dict] = []
    for voice in engine.getProperty("voices") or []:
        languages = []
        for language in getattr(voice, "languages", []) or []:
            text = str(language)
            text = text.replace("b'\\x05", "").replace("b'\\x02", "").replace("'", "")
            languages.append(text)
        lowered_languages = [value.lower() for value in languages]
        voice_name = str(getattr(voice, "name", "") or "")
        payloads.append(
            {
                "id": str(getattr(voice, "id", "") or ""),
                "name": voice_name,
                "languages": languages,
                "is_german": "german" in voice_name.lower() or any(language.startswith("de") for language in lowered_languages),
            }
        )
    return payloads


def build_audio_segment_plan(voice_plan_lines: list[dict], total_duration: float) -> list[dict]:
    duration_limit = max(0.0, float(total_duration or 0.0))
    segments: list[dict] = []
    cursor = 0.0
    for line in sorted(voice_plan_lines, key=lambda item: float(item.get("start_seconds", 0.0) or 0.0)):
        start_seconds = max(0.0, min(duration_limit, float(line.get("start_seconds", 0.0) or 0.0)))
        end_seconds = max(start_seconds, min(duration_limit, float(line.get("end_seconds", start_seconds) or start_seconds)))
        line_duration = max(0.0, end_seconds - start_seconds)
        gap_duration = max(0.0, start_seconds - cursor)
        if gap_duration > 0.01:
            segments.append({"kind": "silence", "duration_seconds": round(gap_duration, 3)})
        if line_duration > 0.01:
            segments.append(
                {
                    "kind": "line",
                    "line_index": int(line.get("line_index", 0) or 0),
                    "duration_seconds": round(line_duration, 3),
                }
            )
        cursor = max(cursor, end_seconds)
    tail_duration = max(0.0, duration_limit - cursor)
    if tail_duration > 0.01 or not segments:
        segments.append({"kind": "silence", "duration_seconds": round(tail_duration if segments else duration_limit, 3)})
    return segments


def synthesize_voice_lines(temp_root: Path, voice_plan_lines: list[dict], render_cfg: dict) -> tuple[dict[int, Path], dict]:
    try:
        import pyttsx3
    except Exception as exc:
        raise RuntimeError("pyttsx3 is not available for voiced final episode rendering.") from exc

    sample_rate = int(render_cfg.get("audio_sample_rate", 22050) or 22050)
    engine = pyttsx3.init()
    voices = describe_system_voices(engine)
    configured_voice_id = clean_text(render_cfg.get("voice_id", ""))
    default_voice_id = configured_voice_id or clean_text(engine.getProperty("voice"))
    voice_id = resolve_system_voice_id(default_voice_id, voices, require_german=bool(render_cfg.get("prefer_german_voice", True)))
    voice_rate = int(render_cfg.get("voice_rate", 175) or 175)
    voice_volume = float(render_cfg.get("voice_volume", 1.0) or 1.0)
    output_map: dict[int, Path] = {}
    try:
        if voice_id:
            engine.setProperty("voice", voice_id)
        engine.setProperty("rate", voice_rate)
        engine.setProperty("volume", max(0.0, min(1.0, voice_volume)))
        for line in voice_plan_lines:
            line_index = int(line.get("line_index", 0) or 0)
            text = clean_text(line.get("text", ""))
            if not text:
                continue
            target = temp_root / f"line_{line_index:04d}_raw.wav"
            engine.save_to_file(text, str(target))
            output_map[line_index] = target
        engine.runAndWait()
    finally:
        try:
            engine.stop()
        except Exception:
            pass

    missing = [index for index, path in output_map.items() if not path.exists() or path.stat().st_size <= 0]
    if missing:
        raise RuntimeError(f"Voice synthesis failed for {len(missing)} dialogue lines.")
    return output_map, {"backend": "pyttsx3", "voice_id": voice_id, "sample_rate": sample_rate, "voices": voices}


def create_silence_audio(ffmpeg: Path, duration_seconds: float, output_path: Path, sample_rate: int) -> None:
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={sample_rate}:cl=mono",
        "-t",
        f"{max(0.01, float(duration_seconds or 0.0)):.3f}",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_path),
    ]
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0 or not render_output_ready(output_path):
        raise RuntimeError(f"Could not create silence audio segment {output_path.name}.")


def normalize_line_audio(ffmpeg: Path, input_path: Path, duration_seconds: float, output_path: Path, sample_rate: int) -> None:
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-af",
        "apad",
        "-t",
        f"{max(0.01, float(duration_seconds or 0.0)):.3f}",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_path),
    ]
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0 or not render_output_ready(output_path):
        raise RuntimeError(f"Could not normalize dialogue audio {input_path.name}.")


def concat_audio_segments(ffmpeg: Path, segment_paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        concat_path = Path(handle.name)
        for segment_path in segment_paths:
            escaped = str(segment_path).replace("'", "''")
            handle.write(f"file '{escaped}'\n")
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-c",
        "copy",
        str(output_path),
    ]
    try:
        result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0 or not render_output_ready(output_path):
            raise RuntimeError(f"Could not assemble final episode audio {output_path.name}.")
    finally:
        try:
            concat_path.unlink()
        except FileNotFoundError:
            pass


def mux_episode_audio(ffmpeg: Path, video_path: Path, audio_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0 or not render_output_ready(output_path):
        raise RuntimeError(f"Could not mux final voiced episode {output_path.name}.")


def render_episode_audio_track(
    ffmpeg: Path,
    voice_plan_lines: list[dict],
    total_duration: float,
    render_cfg: dict,
    temp_root: Path,
    output_path: Path,
) -> dict:
    audio_root = temp_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    sample_rate = int(render_cfg.get("audio_sample_rate", 22050) or 22050)
    synthesized_map, voice_meta = synthesize_voice_lines(audio_root, voice_plan_lines, render_cfg)
    segment_plan = build_audio_segment_plan(voice_plan_lines, total_duration)
    normalized_map: dict[int, Path] = {}
    for line in voice_plan_lines:
        line_index = int(line.get("line_index", 0) or 0)
        if line_index in normalized_map:
            continue
        normalized_path = audio_root / f"line_{line_index:04d}.wav"
        normalize_line_audio(
            ffmpeg,
            synthesized_map[line_index],
            float(line.get("estimated_duration_seconds", 0.0) or 0.0),
            normalized_path,
            sample_rate,
        )
        normalized_map[line_index] = normalized_path
    concat_segments: list[Path] = []
    silence_index = 0
    for segment in segment_plan:
        if segment.get("kind") == "silence":
            duration_seconds = float(segment.get("duration_seconds", 0.0) or 0.0)
            if duration_seconds <= 0.01:
                continue
            silence_path = audio_root / f"silence_{silence_index:04d}.wav"
            create_silence_audio(ffmpeg, duration_seconds, silence_path, sample_rate)
            concat_segments.append(silence_path)
            silence_index += 1
            continue
        line_index = int(segment.get("line_index", 0) or 0)
        concat_segments.append(normalized_map[line_index])
    if not concat_segments:
        silence_path = audio_root / "silence_full.wav"
        create_silence_audio(ffmpeg, total_duration, silence_path, sample_rate)
        concat_segments.append(silence_path)
    concat_audio_segments(ffmpeg, concat_segments, output_path)
    return {
        "audio_track": str(output_path),
        "audio_backend": voice_meta.get("backend", "pyttsx3"),
        "voice_id": voice_meta.get("voice_id", ""),
        "sample_rate": sample_rate,
        "segment_count": len(concat_segments),
    }


def run_ffmpeg_with_codec_fallback(command_factory, video_codec: str, output_path: Path) -> str:
    attempted: list[str] = []
    for codec in [video_codec, "libx264", "mpeg4"]:
        if codec in attempted:
            continue
        attempted.append(codec)
        command = command_factory(codec)
        result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode == 0 and render_output_ready(output_path):
            return codec
    raise RuntimeError(f"FFmpeg render failed for {output_path.name}")


def create_final_render(ffmpeg_path: Path, segment_paths: list[Path], output_path: Path, video_codec: str) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        concat_path = Path(handle.name)
        for segment_path in segment_paths:
            escaped = str(segment_path).replace("'", "''")
            handle.write(f"file '{escaped}'\n")

    def command_factory(codec: str) -> list[str]:
        return [
            str(ffmpeg_path),
            "-hide_banner",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-an",
            "-c:v",
            codec,
            str(output_path),
        ]

    try:
        return run_ffmpeg_with_codec_fallback(command_factory, video_codec, output_path)
    finally:
        try:
            concat_path.unlink()
        except FileNotFoundError:
            pass


def encode_video(ffmpeg: Path, concat_path: Path, output_path: Path, fps: int, width: int, height: int, crf: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-vf",
        f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg render failed for {output_path.name}: {(result.stdout or '').strip()[-600:]}")


def build_concat_file(entries: list[tuple[Path, float]], output_path: Path) -> None:
    lines: list[str] = []
    for frame_path, duration in entries:
        escaped = str(frame_path).replace("'", "''")
        lines.append(f"file '{escaped}'")
        lines.append(f"duration {duration:.3f}")
    if entries:
        escaped = str(entries[-1][0]).replace("'", "''")
        lines.append(f"file '{escaped}'")
    write_text(output_path, "\n".join(lines) + "\n")


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Render Episode")
    cfg = load_config()
    shotlist_dir = resolve_project_path("generation/shotlists")
    shotlist_path = (shotlist_dir / f"{args.episode_id}.json") if args.episode_id else find_latest_shotlist(shotlist_dir)
    if shotlist_path is None or not shotlist_path.exists():
        info("No shotlist found for render.")
        return

    shotlist = read_json(shotlist_path, {})
    episode_id = str(shotlist.get("episode_id", shotlist_path.stem))
    scenes = shotlist.get("scenes", []) if isinstance(shotlist.get("scenes", []), list) else []
    if not scenes:
        info("No scenes found in the shotlist. Run 14_generate_episode_from_trained_model.py first.")
        return

    render_cfg = cfg.get("render", {}) if isinstance(cfg.get("render"), dict) else {}
    cloning_cfg = cfg.get("cloning", {}) if isinstance(cfg.get("cloning"), dict) else {}
    width = int(render_cfg.get("width", 1280))
    height = int(render_cfg.get("height", 720))
    fps = max(12, int(render_cfg.get("fps", 30)))
    title_card_seconds = max(0.0, float(render_cfg.get("title_card_seconds", 2.5) or 0.0))
    closing_card_seconds = max(0.0, float(render_cfg.get("closing_card_seconds", 2.0) or 0.0))
    include_title_cards = bool(render_cfg.get("include_title_cards", False))

    ffmpeg = detect_tool(PROJECT_ROOT / "tools" / "ffmpeg" / "bin", "ffmpeg")
    assets_root = Path(str(shotlist.get("storyboard_assets_root", ""))).resolve() if shotlist.get("storyboard_assets_root") else storyboard_assets_root(cfg, episode_id)
    render_root = resolve_project_path("generation/renders")
    temp_frame_root = render_root / "tmp" / episode_id
    draft_path = render_root / "drafts" / f"{episode_id}.mp4"
    final_path = render_root / "final" / f"{episode_id}.mp4"
    final_video_only_path = temp_frame_root / f"{episode_id}_video_only.mp4"
    manifest_path = render_root / "final" / f"{episode_id}_render_manifest.json"
    voice_plan_path = render_root / "final" / f"{episode_id}_voice_plan.json"
    subtitle_preview_path = render_root / "final" / f"{episode_id}_dialogue_preview.srt"
    dialogue_audio_path = render_root / "final" / f"{episode_id}_dialogue_audio.wav"
    concat_path = temp_frame_root / "frames.txt"

    autosave_target = episode_id
    mark_step_started("18_render_episode", autosave_target, {"episode_id": episode_id, "shotlist": str(shotlist_path)})
    reporter = LiveProgressReporter(
        script_name="17_render_episode.py",
        total=len(scenes) + 3,
        phase_label="Render Episode",
        parent_label=episode_id,
    )
    try:
        if draft_path.exists() and final_path.exists() and not args.force:
            reporter.finish(current_label=episode_id, extra_label="Draft and final render already exist")
            mark_step_completed(
                "18_render_episode",
                autosave_target,
                {"episode_id": episode_id, "draft_render": str(draft_path), "final_render": str(final_path), "manifest": str(manifest_path)},
            )
            ok(f"Render already available: {episode_id}")
            return

        temp_frame_root.mkdir(parents=True, exist_ok=True)
        entries: list[tuple[Path, float]] = []
        manifest_scenes: list[dict] = []
        voice_plan_scenes: list[dict] = []
        voice_plan_lines: list[dict] = []
        original_line_library = build_original_line_library(cfg)
        voice_lookup = build_voice_lookup(cfg)
        timeline_cursor = 0.0
        render_mode = "voiced_storyboard_episode"
        audio_track_meta: dict[str, object] = {}
        audio_render_error = ""

        if include_title_cards and title_card_seconds > 0.0:
            reporter.update(0, current_label="Title Card", extra_label="Running now: render opening title card", force=True)
            title_path = temp_frame_root / "0000_title.png"
            title_image = title_card(
                width,
                height,
                episode_id,
                str(shotlist.get("display_title", episode_id)),
                str(shotlist.get("episode_title", "")),
            )
            title_image.save(title_path, quality=95)
            entries.append((title_path, title_card_seconds))
            timeline_cursor += title_card_seconds

        for index, scene in enumerate(scenes, start=1):
            scene_id = str(scene.get("scene_id", "")).strip() or f"scene_{index:04d}"
            reporter.update(index - 1, current_label=scene_id, extra_label="Running now: compose storyboard render frame", force=True)
            scene_image, scene_meta = compose_scene_card(scene, index - 1, assets_root, width, height)
            frame_path = temp_frame_root / f"{index:04d}_{scene_id}.png"
            scene_image.save(frame_path, quality=95)
            duration = safe_duration_seconds(scene)
            entries.append((frame_path, duration))
            scene_voice_plan = build_scene_voice_plan(
                scene,
                duration,
                timeline_cursor,
                original_line_library,
                voice_lookup,
                cloning_cfg,
                render_cfg,
            )
            voice_plan_scenes.append(
                {
                    "scene_id": scene_id,
                    "duration_seconds": duration,
                    "line_count": len(scene_voice_plan),
                    "lines": scene_voice_plan,
                }
            )
            voice_plan_lines.extend(scene_voice_plan)
            manifest_scenes.append(
                {
                    "scene_id": scene_id,
                    "title": scene.get("title", ""),
                    "duration_seconds": duration,
                    "frame_path": str(frame_path),
                    "asset_source_type": scene_meta.get("asset_source_type", ""),
                    "asset_source_path": scene_meta.get("asset_source_path", ""),
                    "characters": scene.get("characters", []) if isinstance(scene.get("characters", []), list) else [],
                    "summary": scene.get("summary", ""),
                    "generation_plan": scene.get("generation_plan", {}) if isinstance(scene.get("generation_plan", {}), dict) else {},
                    "voice_line_count": len(scene_voice_plan),
                }
            )
            reporter.update(index, current_label=scene_id, extra_label=f"Frame source: {scene_meta.get('asset_source_type', 'placeholder')}")
            timeline_cursor += duration

        if closing_card_seconds > 0.0:
            closing_index = len(entries) + 1
            reporter.update(len(scenes), current_label="Closing Card", extra_label="Running now: render closing card", force=True)
            closing_path = temp_frame_root / f"{closing_index:04d}_closing.png"
            closing_image = closing_card(width, height, str(shotlist.get("display_title", episode_id)), len(scenes))
            closing_image.save(closing_path, quality=95)
            entries.append((closing_path, closing_card_seconds))
            timeline_cursor += closing_card_seconds

        reporter.update(len(scenes) + 1, current_label="Concat List", extra_label="Running now: assemble FFmpeg concat list")
        build_concat_file(entries, concat_path)

        reporter.update(len(scenes) + 2, current_label="Draft Render", extra_label="Running now: encode draft and final video", force=True)
        encode_video(ffmpeg, concat_path, draft_path, fps, min(width, 960), min(height, 540), crf=28)
        encode_video(ffmpeg, concat_path, final_video_only_path, fps, width, height, crf=20)

        try:
            audio_track_meta = render_episode_audio_track(
                ffmpeg,
                voice_plan_lines,
                timeline_cursor,
                render_cfg,
                temp_frame_root,
                dialogue_audio_path,
            )
            mux_episode_audio(ffmpeg, final_video_only_path, dialogue_audio_path, final_path)
        except Exception as audio_exc:
            render_mode = "silent_storyboard_preview_fallback"
            audio_render_error = str(audio_exc)
            shutil.copyfile(final_video_only_path, final_path)
            info(f"Audio render fallback active for {episode_id}: {audio_render_error}")

        voice_plan_payload = {
            "episode_id": episode_id,
            "render_mode": render_mode,
            "voice_plan_mode": "timed_dialogue_preview",
            "subtitle_preview": str(subtitle_preview_path),
            "audio_track": str(dialogue_audio_path) if audio_track_meta else "",
            "audio_track_available": bool(audio_track_meta),
            "audio_render_error": audio_render_error,
            "scene_count": len(voice_plan_scenes),
            "line_count": len(voice_plan_lines),
            "scenes": voice_plan_scenes,
        }
        write_json(voice_plan_path, voice_plan_payload)
        write_text(subtitle_preview_path, render_subtitle_preview_srt(voice_plan_lines))

        manifest = {
            "episode_id": episode_id,
            "display_title": shotlist.get("display_title", episode_id),
            "episode_title": shotlist.get("episode_title", ""),
            "shotlist_path": str(shotlist_path),
            "storyboard_assets_root": str(assets_root),
            "render_mode": render_mode,
            "voice_plan_mode": "timed_dialogue_preview",
            "draft_render": str(draft_path),
            "final_render": str(final_path),
            "final_video_only_render": str(final_video_only_path),
            "voice_plan": str(voice_plan_path),
            "subtitle_preview": str(subtitle_preview_path),
            "dialogue_audio": str(dialogue_audio_path) if audio_track_meta else "",
            "audio_track_available": bool(audio_track_meta),
            "audio_render_error": audio_render_error,
            "audio_track_meta": audio_track_meta,
            "ffmpeg": str(ffmpeg),
            "fps": fps,
            "width": width,
            "height": height,
            "include_title_cards": include_title_cards,
            "scene_count": len(scenes),
            "scenes": manifest_scenes,
        }
        write_json(manifest_path, manifest)
        shotlist["render_manifest"] = str(manifest_path)
        shotlist["draft_render"] = str(draft_path)
        shotlist["final_render"] = str(final_path)
        shotlist["dialogue_audio"] = str(dialogue_audio_path) if audio_track_meta else ""
        shotlist["voice_plan"] = str(voice_plan_path)
        shotlist["subtitle_preview"] = str(subtitle_preview_path)
        write_json(shotlist_path, shotlist)

        reporter.finish(current_label=episode_id, extra_label=f"Rendered {len(scenes)} scenes to draft preview and final episode")
        mark_step_completed(
            "18_render_episode",
            autosave_target,
            {
                "episode_id": episode_id,
                "draft_render": str(draft_path),
                "final_render": str(final_path),
                "dialogue_audio": str(dialogue_audio_path) if audio_track_meta else "",
                "render_mode": render_mode,
                "manifest": str(manifest_path),
            },
        )
        ok(f"Episode rendered: {episode_id}")
    except Exception as exc:
        mark_step_failed("18_render_episode", str(exc), autosave_target, {"episode_id": episode_id})
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
