#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import os
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
    parser = argparse.ArgumentParser(description="Render a draft and final storyboard preview video.")
    parser.add_argument("--episode-id", help="Target a specific episode ID such as folge_09.")
    parser.add_argument("--force", action="store_true", help="Recreate existing draft and final renders.")
    return parser.parse_args()


def find_latest_shotlist(shotlist_dir: Path) -> Path | None:
    episode_id = os.environ.get("SERIES_RENDER_EPISODE", "").strip()
    if episode_id:
        candidate = shotlist_dir / f"{episode_id}.json"
        if candidate.exists():
            return candidate
    files = sorted(shotlist_dir.glob("folge_*.json"))
    return files[-1] if files else None


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
    width = int(render_cfg.get("width", 1280))
    height = int(render_cfg.get("height", 720))
    fps = max(12, int(render_cfg.get("fps", 30)))
    title_card_seconds = max(0.0, float(render_cfg.get("title_card_seconds", 2.5) or 0.0))
    closing_card_seconds = max(0.0, float(render_cfg.get("closing_card_seconds", 2.0) or 0.0))
    include_title_cards = bool(render_cfg.get("include_title_cards", False))

    ffmpeg = detect_tool(PROJECT_ROOT / "tools" / "ffmpeg" / "bin", "ffmpeg")
    assets_root = Path(str(shotlist.get("storyboard_assets_root", ""))).resolve() if shotlist.get("storyboard_assets_root") else storyboard_assets_root(cfg, episode_id)
    render_root = resolve_project_path("generation/renders")
    draft_path = render_root / "drafts" / f"{episode_id}.mp4"
    final_path = render_root / "final" / f"{episode_id}.mp4"
    manifest_path = render_root / "final" / f"{episode_id}_render_manifest.json"
    temp_frame_root = render_root / "tmp" / episode_id
    concat_path = temp_frame_root / "frames.txt"

    autosave_target = episode_id
    mark_step_started("18_render_episode", autosave_target, {"episode_id": episode_id, "shotlist": str(shotlist_path)})
    reporter = LiveProgressReporter(
        script_name="18_render_episode.py",
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

        for index, scene in enumerate(scenes, start=1):
            scene_id = str(scene.get("scene_id", "")).strip() or f"scene_{index:04d}"
            reporter.update(index - 1, current_label=scene_id, extra_label="Running now: compose storyboard render frame", force=True)
            scene_image, scene_meta = compose_scene_card(scene, index - 1, assets_root, width, height)
            frame_path = temp_frame_root / f"{index:04d}_{scene_id}.png"
            scene_image.save(frame_path, quality=95)
            duration = safe_duration_seconds(scene)
            entries.append((frame_path, duration))
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
                }
            )
            reporter.update(index, current_label=scene_id, extra_label=f"Frame source: {scene_meta.get('asset_source_type', 'placeholder')}")

        if closing_card_seconds > 0.0:
            closing_index = len(entries) + 1
            reporter.update(len(scenes), current_label="Closing Card", extra_label="Running now: render closing card", force=True)
            closing_path = temp_frame_root / f"{closing_index:04d}_closing.png"
            closing_image = closing_card(width, height, str(shotlist.get("display_title", episode_id)), len(scenes))
            closing_image.save(closing_path, quality=95)
            entries.append((closing_path, closing_card_seconds))

        reporter.update(len(scenes) + 1, current_label="Concat List", extra_label="Running now: assemble FFmpeg concat list")
        build_concat_file(entries, concat_path)

        reporter.update(len(scenes) + 2, current_label="Draft Render", extra_label="Running now: encode draft and final video", force=True)
        encode_video(ffmpeg, concat_path, draft_path, fps, min(width, 960), min(height, 540), crf=28)
        encode_video(ffmpeg, concat_path, final_path, fps, width, height, crf=20)

        manifest = {
            "episode_id": episode_id,
            "display_title": shotlist.get("display_title", episode_id),
            "episode_title": shotlist.get("episode_title", ""),
            "shotlist_path": str(shotlist_path),
            "storyboard_assets_root": str(assets_root),
            "render_mode": "silent_storyboard_preview",
            "draft_render": str(draft_path),
            "final_render": str(final_path),
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
        write_json(shotlist_path, shotlist)

        reporter.finish(current_label=episode_id, extra_label=f"Rendered {len(scenes)} storyboard scenes to draft and final video")
        mark_step_completed(
            "18_render_episode",
            autosave_target,
            {"episode_id": episode_id, "draft_render": str(draft_path), "final_render": str(final_path), "manifest": str(manifest_path)},
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
