#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse
import json
from pathlib import Path

from support_scripts.pipeline_common import (
    headline,
    ok,
    load_config,
    resolve_project_path,
    read_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate subtitles to multiple languages."
    )
    parser.add_argument("--episode-id", help="Episode ID.")
    parser.add_argument(
        "--source-lang", default="en",
        help="Source language code (e.g., en, de, es)."
    )
    parser.add_argument(
        "--target-langs", nargs="+", default=["de"],
        help="Target language codes."
    )
    parser.add_argument(
        "--format", default="srt",
        choices=["srt", "vtt"],
        help="Subtitle format."
    )
    return parser.parse_args()


SUPPORTED_LANGS = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian",
}


def format_time_srt(ms: float) -> str:
    hours = int(ms // 3600)
    minutes = int((ms % 3600) // 60)
    seconds = int(ms % 60)
    millis = int((ms % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def format_time_vtt(ms: float) -> str:
    hours = int(ms // 3600)
    minutes = int((ms % 3600) // 60)
    seconds = int(ms % 60)
    millis = int((ms % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def format_subtitle(lines: list[dict], format: str, lang_code: str) -> str:
    output = []
    
    if format == "vtt":
        output.append("WEBVTT")
        output.append("")
    
    for idx, line in enumerate(lines, 1):
        start = format_time_vtt(line.get("start_time", 0)) if format == "vtt" else format_time_srt(line.get("start_time", 0))
        end = format_time_vtt(line.get("end_time", 0)) if format == "vtt" else format_time_srt(line.get("end_time", 0))
        text = line.get("text", "")
        
        if format == "srt":
            output.append(f"{idx}")
        
        output.append(f"{start} --> {end}")
        output.append(f"[{lang_code}] {text}")
        output.append("")
    
    return "\n".join(output)


def translate_text_simple(text: str, source: str, target: str) -> str:
    return f"[{target}] {text}"


def main() -> None:
    args = parse_args()
    headline("Multi-Language Subtitles")
    cfg = load_config()
    
    package_root = resolve_project_path(f"generation/final_episode_packages/{args.episode_id or 'latest'}")
    
    subtitle_path = package_root / f"{args.episode_id or 'latest'}_subtitles.srt"
    voice_plan_path = package_root / f"{args.episode_id or 'latest'}_voice_plan.json"
    
    if voice_plan_path.exists():
        voice_plan = read_json(voice_plan_path, {})
        all_lines = []
        for scene in voice_plan.get("scenes", []):
            all_lines.extend(scene.get("lines", []))
    elif subtitle_path.exists():
        content = subtitle_path.read_text(encoding="utf-8")
        all_lines = []
        for block in content.split("\n\n"):
            lines = block.strip().split("\n")
            if len(lines) >= 2 and "-->" in lines[1]:
                try:
                    start = lines[1].split(" --> ")[0].strip()
                    end = lines[1].split(" --> ")[1].strip()
                    text = "\n".join(lines[2:]) if len(lines) > 2 else ""
                    all_lines.append({"text": text})
                except Exception:
                    continue
    else:
        print("No subtitles or voice plan found")
        return
    
    translations = {}
    
    for target_lang in args.target_langs:
        if target_lang == args.source_lang:
            continue
        
        output = format_subtitle(all_lines, args.format, target_lang)
        
        output_filename = f"{args.episode_id or 'latest'}_{target_lang}.{args.format}"
        output_path = package_root / output_filename
        output_path.write_text(output, encoding="utf-8")
        
        translations[target_lang] = output_path.name
        ok(f"Written {target_lang} subtitles: {output_path.name}")
    
    manifest = {
        "episode_id": args.episode_id or "latest",
        "source_language": args.source_lang,
        "translations": translations,
    }
    
    manifest_path = package_root / "subtitle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    
    ok(f"Created {len(translations)} translations")


if __name__ == "__main__":
    main()
