#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "ai_series_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from support_scripts.pipeline_common import (
    SCRIPT_DIR,
    WORKSPACE_ROOT,
    generated_episode_artifacts,
    headline,
    info,
    latest_generated_episode_artifacts,
    load_config,
    ok,
    read_json,
    rerun_in_runtime,
    resolve_stored_project_path,
    runtime_python,
    stored_path_if_present,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn quality-gate regeneration hints into a retry manifest and optionally rerun weak scenes."
    )
    parser.add_argument("--episode-id", help="Target episode ID. Uses the newest generated episode by default.")
    parser.add_argument("--min-quality", type=float, help="Forward a minimum quality override to 18_quality_gate.py.")
    parser.add_argument("--max-weak-scenes", type=int, help="Forward a weak-scene limit override to 18_quality_gate.py.")
    parser.add_argument(
        "--max-regeneration-batch",
        type=int,
        help="Forward a regeneration-batch override to 18_quality_gate.py.",
    )
    parser.add_argument(
        "--refresh-quality-gate",
        action="store_true",
        help="Re-run 18_quality_gate.py before writing the regeneration manifest.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Run the current whole-episode retry chain (15 -> 16 -> 17 and optionally 19).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Write or run the retry plan even if the current queue is empty.",
    )
    parser.add_argument(
        "--update-bible",
        action="store_true",
        help="Also rebuild 20_build_series_bible.py after a successful retry run.",
    )
    parser.add_argument(
        "--max-regeneration-retries",
        type=int,
        help="Override how many rerender attempts each weak scene may receive before it stops being queued.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Forward --strict to 18_quality_gate.py when refreshing or applying the retry plan.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    return str(value or "").strip()


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def resolve_episode_artifacts(cfg: dict[str, Any], episode_id: str | None) -> dict[str, Any]:
    requested = clean_text(episode_id)
    if not requested or requested.lower() == "latest":
        return latest_generated_episode_artifacts(cfg)
    return generated_episode_artifacts(cfg, requested)


def effective_retry_limit(cfg: dict[str, Any], override: int | None = None) -> int:
    release_cfg = cfg.get("release_mode", {}) if isinstance(cfg.get("release_mode"), dict) else {}
    if override is not None:
        return max(0, int(override))
    return max(0, int(release_cfg.get("max_regeneration_retries", 3) or 3))


def quality_gate_report_path(artifacts: dict[str, Any]) -> Path:
    explicit = stored_path_if_present(artifacts.get("quality_gate_report", ""))
    if explicit and explicit.exists() and explicit.is_file():
        return explicit
    delivery_root = stored_path_if_present(artifacts.get("delivery_bundle_root", ""))
    episode_id = clean_text(artifacts.get("episode_id", "")) or "episode"
    if delivery_root and delivery_root.exists() and delivery_root.is_dir():
        return delivery_root / f"{episode_id}_quality_gate.json"
    package_path = stored_path_if_present(artifacts.get("production_package", ""))
    if package_path and package_path.exists() and package_path.is_file():
        return package_path.parent / f"{episode_id}_quality_gate.json"
    return Path(f"{episode_id}_quality_gate.json")


def run_script(script_name: str, args: list[str], *, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    command = [str(runtime_python()), str(WORKSPACE_ROOT / script_name), *args]
    result = subprocess.run(command, cwd=str(SCRIPT_DIR), text=True)
    if not allow_failure and result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}.")
    return result


def quality_gate_override_requested(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            getattr(args, "min_quality", None),
            getattr(args, "max_weak_scenes", None),
            getattr(args, "max_regeneration_batch", None),
            getattr(args, "max_regeneration_retries", None),
        )
    ) or bool(getattr(args, "strict", False))


def ensure_quality_gate_report(
    cfg: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    min_quality: float | None = None,
    max_weak_scenes: int | None = None,
    max_regeneration_batch: int | None = None,
    strict: bool = False,
    max_regeneration_retries: int | None = None,
    refresh: bool = False,
) -> tuple[Path, dict[str, Any]]:
    episode_id = clean_text(artifacts.get("episode_id", "")) or clean_text(artifacts.get("display_title", "")) or "episode"
    report_path = quality_gate_report_path(artifacts)
    if refresh or not report_path.exists():
        gate_args = ["--episode-id", episode_id, "--no-auto-retry"]
        gate_args.extend(
            quality_gate_override_args(
                min_quality=min_quality,
                max_weak_scenes=max_weak_scenes,
                max_regeneration_batch=max_regeneration_batch,
                max_regeneration_retries=max_regeneration_retries,
                strict=strict,
            )
        )
        result = run_script("18_quality_gate.py", gate_args, allow_failure=True)
        if result.returncode not in (0, 1):
            raise RuntimeError(f"18_quality_gate.py exited unexpectedly with code {result.returncode}.")
    if not report_path.exists():
        raise RuntimeError(f"Quality gate report is missing: {report_path}")
    report = read_json(report_path, {})
    if not isinstance(report, dict):
        raise RuntimeError(f"Quality gate report is not valid JSON: {report_path}")
    return report_path, report


def queue_manifest_path(production_package_path: Path, episode_id: str) -> Path:
    return production_package_path.parent / f"{episode_id}_regeneration_queue.json"


def quality_gate_override_args(
    *,
    min_quality: float | None = None,
    max_weak_scenes: int | None = None,
    max_regeneration_batch: int | None = None,
    max_regeneration_retries: int | None = None,
    strict: bool = False,
) -> list[str]:
    gate_args: list[str] = []
    if min_quality is not None:
        gate_args.extend(["--min-quality", str(float(min_quality))])
    if max_weak_scenes is not None:
        gate_args.extend(["--max-weak-scenes", str(int(max_weak_scenes))])
    if max_regeneration_batch is not None:
        gate_args.extend(["--max-regeneration-batch", str(int(max_regeneration_batch))])
    if max_regeneration_retries is not None:
        gate_args.extend(["--max-regeneration-retries", str(int(max_regeneration_retries))])
    if strict:
        gate_args.append("--strict")
    return gate_args


def regeneration_scope_from_entry(entry: dict[str, Any]) -> str:
    explicit = clean_text(entry.get("regeneration_scope", ""))
    if explicit:
        return explicit
    hints = entry.get("regeneration_hints", {}) if isinstance(entry.get("regeneration_hints"), dict) else {}
    if hints.get("rerun_voice_clone") or hints.get("rerun_lipsync") or hints.get("collect_missing_references"):
        return "voice_lipsync"
    if hints.get("reduce_template_lines") or hints.get("increase_speaker_specific_phrases") or hints.get("use_relationship_conflict"):
        return "story_dialogue"
    return "scene_rerender"


def regeneration_sections_for_scope(scope: str) -> list[str]:
    if scope == "voice_lipsync":
        return ["voice_clone", "lip_sync"]
    if scope == "story_dialogue":
        return ["story", "dialogue", "voice_metadata", "storyboard", "render"]
    if scope == "references":
        return ["voice_references", "character_references"]
    return ["storyboard", "render", "quality_gate"]


def regeneration_scope_counts(queue: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in queue:
        if not isinstance(entry, dict):
            continue
        scope = regeneration_scope_from_entry(entry)
        counts[scope] = counts.get(scope, 0) + 1
    return counts


def rerun_strategy_for_queue(queue: list[dict[str, Any]], scene_ids: list[str]) -> str:
    if not scene_ids:
        return "full_episode_pipeline"
    scopes = [regeneration_scope_from_entry(entry) for entry in queue if isinstance(entry, dict)]
    if scopes and all(scope == "voice_lipsync" for scope in scopes):
        return "voice_lipsync_only"
    if any(scope == "story_dialogue" for scope in scopes):
        return "story_dialogue_refresh"
    return "scene_selective"


def build_rerun_plan(
    episode_id: str,
    *,
    strict: bool,
    update_bible: bool,
    min_quality: float | None = None,
    max_weak_scenes: int | None = None,
    max_regeneration_batch: int | None = None,
    max_regeneration_retries: int | None = None,
    scene_ids: list[str] | None = None,
    regeneration_queue: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    gate_args = ["--episode-id", episode_id, "--no-auto-retry"]
    gate_args.extend(
        quality_gate_override_args(
            min_quality=min_quality,
            max_weak_scenes=max_weak_scenes,
            max_regeneration_batch=max_regeneration_batch,
            max_regeneration_retries=max_regeneration_retries,
            strict=strict,
        )
    )
    queue = regeneration_queue if isinstance(regeneration_queue, list) else []
    strategy = rerun_strategy_for_queue(queue, scene_ids or [])
    plan: list[dict[str, Any]] = []
    if strategy == "story_dialogue_refresh":
        plan.extend(
            [
                {
                    "script": "14_generate_episode.py",
                    "args": ["--episode-id", episode_id],
                    "note": "Story/dialogue realism failed; refresh the behavior-guided shotlist before visual/audio rebuild.",
                },
                {
                    "script": "15_generate_storyboard_assets.py",
                    "args": ["--episode-id", episode_id, "--force"],
                    "note": "Rebuild storyboard asset requests from the refreshed shotlist.",
                },
            ]
        )
    storyboard_args = ["--episode-id", episode_id, "--force"]
    if scene_ids:
        storyboard_args.extend(["--scene-ids", *scene_ids])
        storyboard_note = f"Scene-selective storyboard backend rerun for {len(scene_ids)} scene(s)."
    else:
        storyboard_note = "Full episode storyboard backend rerun."
    if strategy != "voice_lipsync_only":
        plan.append(
            {
                "script": "16_run_storyboard_backend.py",
                "args": storyboard_args,
                "note": storyboard_note,
            }
        )
    plan.extend(
        [
            {
                "script": "17_render_episode.py",
                "args": ["--episode-id", episode_id, "--force"],
                "note": (
                    "Voice/lip-sync retry only; preserve story where possible."
                    if strategy == "voice_lipsync_only"
                    else "Render retries rebuild the full episode package and master."
                ),
            },
            {
                "script": "18_quality_gate.py",
                "args": gate_args,
                "note": "Recompute release status and the weak-scene queue after rerender.",
            },
        ]
    )
    if update_bible:
        plan.append(
            {
                "script": "20_build_series_bible.py",
                "args": [],
                "note": "Refresh the series bible after the retry run.",
            }
        )
    return plan


def build_regeneration_reason(queue_entry: dict[str, Any]) -> str:
    weaknesses = queue_entry.get("weaknesses", []) if isinstance(queue_entry.get("weaknesses"), list) else []
    failed_reasons = queue_entry.get("failed_reasons", []) if isinstance(queue_entry.get("failed_reasons"), list) else []
    hints = queue_entry.get("regeneration_hints", {}) if isinstance(queue_entry.get("regeneration_hints"), dict) else {}
    weakness_text = ", ".join(clean_text(item) for item in weaknesses if clean_text(item))
    failed_text = ", ".join(clean_text(item) for item in failed_reasons if clean_text(item))
    hint_text = ", ".join(key for key, enabled in hints.items() if enabled)
    quality_percent = int(queue_entry.get("quality_percent", 0) or 0)
    parts = [part for part in (weakness_text, failed_text, f"hints: {hint_text}" if hint_text else "") if part]
    if parts:
        return f"Weak scene ({quality_percent}%): {'; '.join(parts)}"
    if weakness_text:
        return f"Weak scene ({quality_percent}%): {weakness_text}"
    return f"Weak scene ({quality_percent}%) fell below the regeneration watch threshold."


def apply_regeneration_request_to_scene_package(
    scene_package: dict[str, Any],
    queue_entry: dict[str, Any],
    manifest_path: Path,
    requested_at: str,
    max_regeneration_retries: int,
    *,
    increment_retry: bool,
) -> dict[str, Any]:
    updated = dict(scene_package) if isinstance(scene_package, dict) else {}
    quality = updated.get("quality_assessment", {}) if isinstance(updated.get("quality_assessment"), dict) else {}
    next_retries = int(quality.get("regeneration_retries", 0) or 0) + (1 if increment_retry else 0)
    quality = dict(quality)
    quality["scene_id"] = clean_text(updated.get("scene_id", "")) or clean_text(queue_entry.get("scene_id", ""))
    quality["regeneration_retries"] = next_retries
    quality["regeneration_retry_limit"] = int(max_regeneration_retries)
    quality["max_regeneration_retries"] = int(max_regeneration_retries)
    quality["last_regeneration_requested_at"] = requested_at
    quality["last_regeneration_reason"] = build_regeneration_reason(queue_entry)
    quality["last_regeneration_request_source"] = "19_regenerate_weak_scenes.py"
    quality["last_regeneration_queue_manifest"] = str(manifest_path)
    regeneration_scope = regeneration_scope_from_entry(queue_entry)
    regeneration_hints = queue_entry.get("regeneration_hints", {}) if isinstance(queue_entry.get("regeneration_hints"), dict) else {}
    failed_reasons = queue_entry.get("failed_reasons", []) if isinstance(queue_entry.get("failed_reasons"), list) else []
    quality["last_regeneration_apply_mode"] = regeneration_scope
    quality["last_regeneration_hints"] = regeneration_hints
    quality["last_regeneration_failed_reasons"] = failed_reasons
    quality["last_regeneration_queue_entry"] = dict(queue_entry)
    quality["queued_for_regeneration"] = True
    updated["quality_assessment"] = quality
    updated["regeneration_scope"] = regeneration_scope
    updated["regeneration_hints"] = regeneration_hints
    updated["regeneration_requested_sections"] = regeneration_sections_for_scope(regeneration_scope)
    return updated


def mark_scene_regeneration_applied(
    scene_package: dict[str, Any],
    *,
    applied_at: str,
) -> dict[str, Any]:
    updated = dict(scene_package) if isinstance(scene_package, dict) else {}
    quality = updated.get("quality_assessment", {}) if isinstance(updated.get("quality_assessment"), dict) else {}
    quality = dict(quality)
    quality["queued_for_regeneration"] = False
    quality["last_regeneration_applied_at"] = applied_at
    updated["quality_assessment"] = quality
    return updated


def persist_package_scene_updates(
    package_path: Path,
    queue: list[dict[str, Any]],
    manifest_path: Path,
    requested_at: str,
    max_regeneration_retries: int,
    *,
    increment_retry: bool = False,
    applied_at: str | None = None,
) -> dict[str, Any]:
    package_payload = read_json(package_path, {})
    if not isinstance(package_payload, dict):
        raise RuntimeError(f"Production package is not valid JSON: {package_path}")

    queue_by_scene = {
        clean_text(entry.get("scene_id", "")): entry
        for entry in queue
        if isinstance(entry, dict) and clean_text(entry.get("scene_id", ""))
    }
    scene_ids = [scene_id for scene_id in queue_by_scene]
    scenes = package_payload.get("scenes", []) if isinstance(package_payload.get("scenes"), list) else []
    updated_scenes: list[dict[str, Any]] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            updated_scenes.append(scene)
            continue
        scene_id = clean_text(scene.get("scene_id", ""))
        updated_scene = dict(scene)
        if scene_id in queue_by_scene:
            updated_scene = apply_regeneration_request_to_scene_package(
                updated_scene,
                queue_by_scene[scene_id],
                manifest_path,
                requested_at,
                max_regeneration_retries,
                increment_retry=increment_retry,
            )
            if applied_at:
                updated_scene = mark_scene_regeneration_applied(updated_scene, applied_at=applied_at)
        updated_scenes.append(updated_scene)
    package_payload["scenes"] = updated_scenes

    scene_package_paths = package_payload.get("scene_package_paths", []) if isinstance(package_payload.get("scene_package_paths"), list) else []
    for index, raw_path in enumerate(scene_package_paths):
        scene_package_path = resolve_stored_project_path(raw_path)
        if not scene_package_path.exists():
            continue
        scene_payload = read_json(scene_package_path, {})
        if not isinstance(scene_payload, dict):
            continue
        scene_id = clean_text(scene_payload.get("scene_id", ""))
        if not scene_id and index < len(updated_scenes) and isinstance(updated_scenes[index], dict):
            scene_id = clean_text(updated_scenes[index].get("scene_id", ""))
        if scene_id not in queue_by_scene:
            continue
        scene_payload = apply_regeneration_request_to_scene_package(
            scene_payload,
            queue_by_scene[scene_id],
            manifest_path,
            requested_at,
            max_regeneration_retries,
            increment_retry=increment_retry,
        )
        if applied_at:
            scene_payload = mark_scene_regeneration_applied(scene_payload, applied_at=applied_at)
        write_json(scene_package_path, scene_payload)

    package_payload["regeneration_queue_manifest"] = str(manifest_path)
    package_payload["regeneration_requested_scene_ids"] = scene_ids
    package_payload["regeneration_last_requested_at"] = requested_at
    package_payload["regeneration_queue_count"] = len(queue)
    package_payload["regeneration_apply_requested"] = bool(increment_retry)
    if increment_retry:
        package_payload["regeneration_apply_requested_at"] = requested_at
    if applied_at:
        package_payload["regeneration_last_applied_at"] = applied_at
    write_json(package_path, package_payload)
    return package_payload


def persist_artifact_metadata(
    artifacts: dict[str, Any],
    manifest_path: Path,
    queue: list[dict[str, Any]],
    requested_at: str,
    *,
    apply_requested: bool,
    applied_at: str | None = None,
) -> None:
    updates: dict[str, Any] = {
        "regeneration_queue_manifest": str(manifest_path),
        "regeneration_requested_scene_ids": [
            clean_text(entry.get("scene_id", ""))
            for entry in queue
            if isinstance(entry, dict) and clean_text(entry.get("scene_id", ""))
        ],
        "regeneration_last_requested_at": requested_at,
        "regeneration_queue_count": len(queue),
        "regeneration_apply_requested": bool(apply_requested and not applied_at),
    }
    if apply_requested:
        updates["regeneration_apply_requested_at"] = requested_at
    if applied_at:
        updates["regeneration_last_applied_at"] = applied_at
    for raw_path in (artifacts.get("shotlist", ""), artifacts.get("render_manifest", "")):
        target_path = resolve_stored_project_path(raw_path)
        if not target_path.exists():
            continue
        payload = read_json(target_path, {})
        if not isinstance(payload, dict):
            continue
        payload.update(updates)
        write_json(target_path, payload)


def build_queue_manifest(
    *,
    artifacts: dict[str, Any],
    report_path: Path,
    report: dict[str, Any],
    manifest_path: Path,
    requested_at: str,
    max_regeneration_retries: int,
    min_quality: float | None,
    max_weak_scenes: int | None,
    max_regeneration_batch: int | None,
    apply_requested: bool,
    update_bible: bool,
    strict: bool,
) -> dict[str, Any]:
    episode_id = clean_text(artifacts.get("episode_id", "")) or "episode"
    queue = report.get("regeneration_queue", []) if isinstance(report.get("regeneration_queue"), list) else []
    scene_ids = [
        clean_text(entry.get("scene_id", ""))
        for entry in queue
        if isinstance(entry, dict) and clean_text(entry.get("scene_id", ""))
    ]
    scope_counts = regeneration_scope_counts(queue)
    rerun_scope = rerun_strategy_for_queue(queue, scene_ids)
    rerun_reason = {
        "voice_lipsync_only": f"Voice/lip-sync retry for {len(scene_ids)} flagged scene(s).",
        "story_dialogue_refresh": f"Story/dialogue refresh for {len(scene_ids)} flagged scene(s).",
        "scene_selective": f"Scene-selective backend rerun for {len(scene_ids)} flagged scene(s).",
        "full_episode_pipeline": "No flagged scenes; full episode pipeline rerun.",
    }.get(rerun_scope, "Full episode pipeline rerun.")
    return {
        "episode_id": episode_id,
        "display_title": clean_text(artifacts.get("display_title", "")),
        "requested_at": requested_at,
        "apply_requested": bool(apply_requested),
        "update_bible_requested": bool(update_bible),
        "strict_quality_gate": bool(strict),
        "quality_gate_report": str(report_path),
        "release_gate": dict(report.get("release_gate", {}) or {}),
        "warnings": list(report.get("warnings", []) or []),
        "regeneration_queue_count": len(queue),
        "regeneration_queue_scene_ids": scene_ids,
        "regeneration_scope_counts": scope_counts,
        "quality_gate_overrides": {
            "min_quality": min_quality,
            "max_weak_scenes": max_weak_scenes,
            "max_regeneration_batch": max_regeneration_batch,
            "max_regeneration_retries": max_regeneration_retries,
            "strict": bool(strict),
        },
        "max_regeneration_retries": int(max_regeneration_retries),
        "rerun_scope": rerun_scope,
        "rerun_reason": rerun_reason,
        "rerun_plan": build_rerun_plan(
            episode_id,
            strict=strict,
            update_bible=update_bible,
            min_quality=min_quality,
            max_weak_scenes=max_weak_scenes,
            max_regeneration_batch=max_regeneration_batch,
            max_regeneration_retries=max_regeneration_retries,
            scene_ids=scene_ids if scene_ids else None,
            regeneration_queue=queue,
        ),
        "regeneration_queue": queue,
        "manifest_path": str(manifest_path),
    }


def update_manifest_after_apply(manifest_path: Path, *, applied_at: str, artifacts: dict[str, Any]) -> None:
    manifest = read_json(manifest_path, {})
    if not isinstance(manifest, dict):
        return
    refreshed_report_path = quality_gate_report_path(artifacts)
    refreshed_report = read_json(refreshed_report_path, {}) if refreshed_report_path.exists() else {}
    manifest["apply_requested"] = False
    manifest["apply_completed"] = True
    manifest["last_apply_completed_at"] = applied_at
    manifest["post_apply_quality_gate_report"] = str(refreshed_report_path) if refreshed_report_path.exists() else ""
    manifest["post_apply_release_gate"] = (
        dict(refreshed_report.get("release_gate", {}) or {})
        if isinstance(refreshed_report, dict)
        else {}
    )
    manifest["post_apply_regeneration_queue_count"] = len(
        refreshed_report.get("regeneration_queue", []) if isinstance(refreshed_report.get("regeneration_queue"), list) else []
    ) if isinstance(refreshed_report, dict) else 0
    write_json(manifest_path, manifest)


def print_plan(manifest: dict[str, Any]) -> None:
    print(f"Episode: {clean_text(manifest.get('episode_id', '')) or '-'}")
    print(f"Requested at: {clean_text(manifest.get('requested_at', '')) or '-'}")
    print(f"Queue size: {int(manifest.get('regeneration_queue_count', 0) or 0)}")
    print(f"Manifest: {clean_text(manifest.get('manifest_path', '')) or '-'}")
    for step in manifest.get("rerun_plan", []) if isinstance(manifest.get("rerun_plan"), list) else []:
        if not isinstance(step, dict):
            continue
        script = clean_text(step.get("script", ""))
        args = " ".join(clean_text(arg) for arg in step.get("args", []) if clean_text(arg))
        note = clean_text(step.get("note", ""))
        print(f"PLAN: python {script} {args}".strip())
        if note:
            print(f"NOTE: {note}")


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Regenerate Weak Scenes")
    cfg = load_config()
    artifacts = resolve_episode_artifacts(cfg, args.episode_id)
    if not artifacts:
        raise RuntimeError("No generated episode artifacts were found.")

    episode_id = clean_text(artifacts.get("episode_id", "")) or clean_text(args.episode_id) or "episode"
    production_package_path = stored_path_if_present(artifacts.get("production_package", ""))
    if not production_package_path or not production_package_path.exists() or not production_package_path.is_file():
        raise RuntimeError(f"Production package is missing: {production_package_path}")

    max_regeneration_retries = effective_retry_limit(cfg, args.max_regeneration_retries)
    refresh_quality_gate = bool(args.refresh_quality_gate or quality_gate_override_requested(args))
    if refresh_quality_gate and not args.refresh_quality_gate:
        info("Quality gate overrides detected. Refreshing 18_quality_gate.py before building the regeneration manifest.")
    report_path, report = ensure_quality_gate_report(
        cfg,
        artifacts,
        min_quality=args.min_quality,
        max_weak_scenes=args.max_weak_scenes,
        max_regeneration_batch=args.max_regeneration_batch,
        strict=bool(args.strict),
        max_regeneration_retries=max_regeneration_retries,
        refresh=refresh_quality_gate,
    )
    requested_at = utc_timestamp()
    manifest_path = queue_manifest_path(production_package_path, episode_id)
    manifest = build_queue_manifest(
        artifacts=artifacts,
        report_path=report_path,
        report=report,
        manifest_path=manifest_path,
        requested_at=requested_at,
        max_regeneration_retries=max_regeneration_retries,
        min_quality=args.min_quality,
        max_weak_scenes=args.max_weak_scenes,
        max_regeneration_batch=args.max_regeneration_batch,
        apply_requested=bool(args.apply),
        update_bible=bool(args.update_bible),
        strict=bool(args.strict),
    )
    queue = manifest.get("regeneration_queue", []) if isinstance(manifest.get("regeneration_queue"), list) else []
    write_json(manifest_path, manifest)
    persist_artifact_metadata(
        artifacts,
        manifest_path,
        queue,
        requested_at,
        apply_requested=bool(args.apply),
    )

    if queue:
        info(f"Queued {len(queue)} weak scenes for {episode_id}.")
    else:
        info(f"No weak scenes are currently queued for {episode_id}.")
    print_plan(manifest)

    if not queue and not args.force:
        ok("Regeneration manifest written. No retry run was needed.")
        return

    persist_package_scene_updates(
        production_package_path,
        queue,
        manifest_path,
        requested_at,
        max_regeneration_retries,
        increment_retry=bool(args.apply),
    )

    if not args.apply:
        ok("Regeneration manifest written. Re-run with --apply to execute the retry plan.")
        return

    rerun_plan = manifest.get("rerun_plan", []) if isinstance(manifest.get("rerun_plan"), list) else []
    for step in rerun_plan:
        if not isinstance(step, dict):
            continue
        script_name = clean_text(step.get("script", ""))
        extra_args = [clean_text(arg) for arg in step.get("args", []) if clean_text(arg)]
        info(f"Running {script_name} {' '.join(extra_args)}".strip())
        allow_failure = script_name == "18_quality_gate.py"
        result = run_script(script_name, extra_args, allow_failure=allow_failure)
        if script_name == "18_quality_gate.py" and result.returncode not in (0, 1):
            raise RuntimeError(f"{script_name} failed with exit code {result.returncode}.")

    applied_at = utc_timestamp()
    refreshed_artifacts = resolve_episode_artifacts(cfg, episode_id)
    refreshed_package_path = stored_path_if_present(refreshed_artifacts.get("production_package", ""))
    if refreshed_package_path and refreshed_package_path.exists() and refreshed_package_path.is_file():
        persist_package_scene_updates(
            refreshed_package_path,
            queue,
            manifest_path,
            requested_at,
            max_regeneration_retries,
            increment_retry=False,
            applied_at=applied_at,
        )
    persist_artifact_metadata(
        refreshed_artifacts,
        manifest_path,
        queue,
        requested_at,
        apply_requested=True,
        applied_at=applied_at,
    )
    update_manifest_after_apply(manifest_path, applied_at=applied_at, artifacts=refreshed_artifacts)
    ok("Weak-scene regeneration rerun completed.")


if __name__ == "__main__":
    main()

