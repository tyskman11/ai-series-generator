#!/usr/bin/env python3
from __future__ import annotations

import argparse

from pipeline_common import (
    add_shared_worker_arguments,
    distributed_item_lease,
    distributed_step_runtime_root,
    error,
    headline,
    info,
    list_generated_episode_artifacts,
    LiveProgressReporter,
    load_config,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    read_json,
    rerun_in_runtime,
    resolve_project_path,
    shared_worker_id_for_args,
    shared_workers_enabled_for_args,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the current series bible snapshot")
    add_shared_worker_arguments(parser)
    return parser.parse_args()


def build_series_bible_payload(model: dict, generated_episodes: list[dict]) -> tuple[dict, str]:
    top_characters = model.get("characters", [])[:8]
    top_keywords = model.get("keywords", [])[:12]
    scene_library = model.get("scene_library", [])
    recent_generated_episodes = list(generated_episodes[:10])

    bible_json = {
        "trained_at": model.get("trained_at"),
        "scene_count": model.get("scene_count", 0),
        "dataset_files": model.get("dataset_files", []),
        "main_characters": top_characters,
        "recurring_keywords": top_keywords,
        "reference_scenes": scene_library[:20],
        "recent_generated_episodes": recent_generated_episodes,
    }

    markdown_lines = [
        "# Automatic Series Bible",
        "",
        f"- Trained at: {model.get('trained_at', 'unknown')}",
        f"- Evaluated scenes: {model.get('scene_count', 0)}",
        "",
        "## Main Characters",
        "",
    ]
    for character in top_characters:
        markdown_lines.append(
            f"- {character['name']}: {character.get('scene_count', 0)} scenes, {character.get('line_count', 0)} dialogue segments"
        )
    markdown_lines.extend(["", "## Recurring Themes", ""])
    markdown_lines.extend([f"- {keyword}" for keyword in top_keywords])
    markdown_lines.extend(["", "## Reference Scenes", ""])
    for scene in scene_library[:12]:
        markdown_lines.append(
            f"- {scene['episode_id']} / {scene['scene_id']}: {', '.join(scene.get('characters', []))} | {scene.get('transcript', '')[:180]}"
        )
    markdown_lines.extend(["", "## Recent Generated Episodes", ""])
    if recent_generated_episodes:
        for episode in recent_generated_episodes:
            markdown_lines.extend(
                [
                    f"### {episode.get('display_title') or episode.get('episode_id') or 'Generated Episode'}",
                    f"- Episode ID: {episode.get('episode_id') or '-'}",
                    f"- Episode title: {episode.get('episode_title') or '-'}",
                    f"- Render mode: {episode.get('render_mode') or '-'}",
                    f"- Production readiness: {episode.get('production_readiness') or '-'}",
                    f"- Delivery bundle: {episode.get('delivery_bundle_root') or '-'}",
                    f"- Delivery manifest: {episode.get('delivery_manifest') or '-'}",
                    f"- Delivery watch episode: {episode.get('delivery_episode') or '-'}",
                    f"- Stable latest delivery episode: {episode.get('latest_delivery_episode') or '-'}",
                    f"- Final render: {episode.get('final_render') or '-'}",
                    f"- Full generated episode: {episode.get('full_generated_episode') or '-'}",
                    f"- Production package: {episode.get('production_package') or '-'}",
                    f"- Render manifest: {episode.get('render_manifest') or '-'}",
                    f"- Scene count: {episode.get('scene_count', 0)}",
                    f"- Generated scene videos: {episode.get('generated_scene_video_count', 0)}",
                    f"- Scene video coverage: {int(round(float(episode.get('scene_video_completion_ratio', 0.0) or 0.0) * 100.0))}%",
                    f"- Scene dialogue tracks: {episode.get('scene_dialogue_audio_count', 0)}",
                    f"- Scene dialogue coverage: {int(round(float(episode.get('scene_dialogue_completion_ratio', 0.0) or 0.0) * 100.0))}%",
                    f"- Scene master clips: {episode.get('scene_master_clip_count', 0)}",
                    f"- Scene master coverage: {int(round(float(episode.get('scene_master_completion_ratio', 0.0) or 0.0) * 100.0))}%",
                    f"- Backend runner status: {episode.get('backend_runner_status') or '-'}",
                    f"- Backend runner coverage: {int(round(float(episode.get('backend_runner_coverage_ratio', 0.0) or 0.0) * 100.0))}%",
                    f"- Backend runners ready: {episode.get('backend_runner_ready_count', 0)}/{episode.get('backend_runner_expected_count', 0)}",
                    f"- Master backend runner: {episode.get('master_backend_runner_status') or '-'}",
                    f"- Remaining backend tasks: {', '.join(episode.get('remaining_backend_tasks', [])) or '-'}",
                    "",
                ]
            )
    else:
        markdown_lines.append("- No generated episodes were found yet.")
        markdown_lines.append("")
    return bible_json, "\n".join(markdown_lines)


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Build Series Bible")
    cfg = load_config()
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)
    mark_step_started("18_build_series_bible", "global")
    lease_root = distributed_step_runtime_root("18_build_series_bible", "global")
    if shared_workers:
        info(f"Shared NAS workers: enabled ({worker_id})")
    reporter = LiveProgressReporter(
        script_name="18_build_series_bible.py",
        total=3,
        phase_label="Build Series Bible",
        parent_label="global",
    )
    with distributed_item_lease(
        root=lease_root,
        lease_name="global",
        cfg=cfg,
        worker_id=worker_id,
        enabled=shared_workers,
        meta={"step": "18_build_series_bible", "scope": "global", "worker_id": worker_id},
    ) as acquired:
        if not acquired:
            info("The series bible is already being rebuilt by another worker.")
            return
        model_path = resolve_project_path(cfg["paths"]["series_model"])
        reporter.update(0, current_label="Load Series Model", extra_label="Running now: load the trained model for the series bible", force=True)
        model = read_json(model_path, {})
        if not model:
            reporter.finish(current_label="Series Model", extra_label="Stopped: no trained model found")
            info("No trained series model was found.")
            return

        reporter.update(1, current_label="Generate Bible Content", extra_label="Running now: collect main characters, themes and reference scenes")
        generated_episodes = list_generated_episode_artifacts(cfg, limit=10)
        bible_json, bible_markdown = build_series_bible_payload(model, generated_episodes)

        try:
            bible_json_path = resolve_project_path(cfg["paths"]["series_bible_json"])
            bible_markdown_path = resolve_project_path(cfg["paths"]["series_bible_markdown"])
            reporter.update(2, current_label="Write Files", extra_label="Running now: save JSON and Markdown series bible")
            write_json(bible_json_path, bible_json)
            write_text(bible_markdown_path, bible_markdown)
            reporter.finish(current_label="Series Bible", extra_label=f"Written: {bible_json_path.name} and {bible_markdown_path.name}")
            mark_step_completed(
                "18_build_series_bible",
                "global",
                {
                    "series_bible_json": str(bible_json_path),
                    "series_bible_markdown": str(bible_markdown_path),
                    "generated_episode_count": len(generated_episodes),
                    "latest_generated_episode": generated_episodes[0] if generated_episodes else {},
                },
            )
            ok("Series Bible was updated.")
        except Exception as exc:
            mark_step_failed("18_build_series_bible", str(exc), "global")
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise

