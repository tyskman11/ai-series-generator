#!/usr/bin/env python3
from __future__ import annotations

from pipeline_common import (
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


def main() -> None:
    rerun_in_runtime()
    headline("Serienbibel aufbauen")
    cfg = load_config()
    mark_step_started("12_build_series_bible", "global")
    model_path = resolve_project_path(cfg["paths"]["series_model"])
    model = read_json(model_path, {})
    if not model:
        info("Kein trainiertes Serienmodell gefunden.")
        return

    top_characters = model.get("characters", [])[:8]
    top_keywords = model.get("keywords", [])[:12]
    scene_library = model.get("scene_library", [])

    bible_json = {
        "trained_at": model.get("trained_at"),
        "scene_count": model.get("scene_count", 0),
        "dataset_files": model.get("dataset_files", []),
        "main_characters": top_characters,
        "recurring_keywords": top_keywords,
        "reference_scenes": scene_library[:20],
    }

    markdown_lines = [
        "# Automatische Serienbibel",
        "",
        f"- Trainiert am: {model.get('trained_at', 'unbekannt')}",
        f"- Ausgewertete Szenen: {model.get('scene_count', 0)}",
        "",
        "## Hauptfiguren",
        "",
    ]
    for character in top_characters:
        markdown_lines.append(
            f"- {character['name']}: {character.get('scene_count', 0)} Szenen, {character.get('line_count', 0)} Sprechanteile"
        )
    markdown_lines.extend(["", "## Wiederkehrende Themen", ""])
    markdown_lines.extend([f"- {keyword}" for keyword in top_keywords])
    markdown_lines.extend(["", "## Referenzszenen", ""])
    for scene in scene_library[:12]:
        markdown_lines.append(
            f"- {scene['episode_id']} / {scene['scene_id']}: {', '.join(scene.get('characters', []))} | {scene.get('transcript', '')[:180]}"
        )
    markdown_lines.append("")

    try:
        bible_json_path = resolve_project_path(cfg["paths"]["series_bible_json"])
        bible_markdown_path = resolve_project_path(cfg["paths"]["series_bible_markdown"])
        write_json(bible_json_path, bible_json)
        write_text(bible_markdown_path, "\n".join(markdown_lines))
        mark_step_completed(
            "12_build_series_bible",
            "global",
            {"series_bible_json": str(bible_json_path), "series_bible_markdown": str(bible_markdown_path)},
        )
        ok("Serienbibel wurde aktualisiert.")
    except Exception as exc:
        mark_step_failed("12_build_series_bible", str(exc), "global")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
