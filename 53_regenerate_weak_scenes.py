#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline_common import (
    SCRIPT_DIR,
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
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn quality-gate regeneration hints into a retry manifest and optionally rerun weak scenes."
    )
    parser.add_argument("--episode-id", help="Target episode ID. Uses the newest generated episode by default.")
    parser.add_argument(
        "--refresh-quality-gate",
        action="store_true",
        help="Re-run 52_quality_gate.py before writing the regeneration manifest.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Run the current whole-episode retry chain (54 -> 15 -> 52 and optionally 16).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Write or run the retry plan even if the current queue is empty.",
    )
    parser.add_argument(
        "--update-bible",
        action="store_true",
        help="Also rebuild 16_build_series_bible.py after a successful retry run.",
    )
    parser.add_argument(
        "--max-regeneration-retries",
        type=int,
        help="Override how many rerender attempts each weak scene may receive before it stops being queued.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Forward --strict to 52_quality_gate.py when refreshing or applying the retry plan.",
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
    explicit = resolve_stored_project_path(artifacts.get("quality_gate_report", ""))
    if explicit.exists():
        return explicit
    delivery_root = resolve_stored_project_path(artifacts.get("delivery_bundle_root", ""))
    episode_id = clean_text(artifacts.get("episode_id", "")) or "episode"
    if delivery_root.exists():
        return delivery_root / f"{episode_id}_quality_gate.json"
    package_path = resolve_stored_project_path(artifacts.get("production_package", ""))
    if package_path.exists():
        return package_path.parent / f"{episode_id}_quality_gate.json"
    return Path(f"{episode_id}_quality_gate.json")


def run_script(script_name: str, args: list[str], *, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    command = [str(runtime_python()), str(SCRIPT_DIR / script_name), *args]
    result = subprocess.run(command, cwd=str(SCRIPT_DIR), text=True)
    if not allow_failure and result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}.")
    return result


def ensure_quality_gate_report(
    cfg: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    strict: bool = False,
    max_regeneration_retries: int | None = None,
    refresh: bool = False,
) -> tuple[Path, dict[str, Any]]:
    episode_id = clean_text(artifacts.get("episode_id", "")) or clean_text(artifacts.get("display_title", "")) or "episode"
    report_path = quality_gate_report_path(artifacts)
    if refresh or not report_path.exists():
        gate_args = ["--episode-id", episode_id]
        if max_regeneration_retries is not None:
            gate_args.extend(["--max-regeneration-retries", str(int(max_regeneration_retries))])
        if strict:
            gate_args.append("--strict")
        result = run_script("52_quality_gate.py", gate_args, allow_failure=True)
        if result.returncode not in (0, 1):
            raise RuntimeError(f"52_quality_gate.py exited unexpectedly with code {result.returncode}.")
    if not report_path.exists():
        raise RuntimeError(f"Quality gate report is missing: {report_path}")
    report = read_json(report_path, {})
    if not isinstance(report, dict):
        raise RuntimeError(f"Quality gate report is not valid JSON: {report_path}")
    return report_path, report


def queue_manifest_path(production_package_path: Path, episode_id: str) -> Path:
    return production_package_path.parent / f"{episode_id}_regeneration_queue.json"


def build_rerun_plan(episode_id: str, *, strict: bool, update_bible: bool, max_regeneration_retries: int | None) -> list[dict[str, Any]]:
    gate_args = ["--episode-id", episode_id]
    if max_regeneration_retries is not None:
        gate_args.extend(["--max-regeneration-retries", str(int(max_regeneration_retries))])
    if strict:
        gate_args.append("--strict")
    plan: list[dict[str, Any]] = [
        {
            "script": "54_run_storyboard_backend.py",
            "args": ["--episode-id", episode_id, "--force"],
            "note": "Current storyboard backend reruns operate on the whole episode.",
        },
        {
            "script": "15_render_episode.py",
            "args": ["--episode-id", episode_id, "--force"],
            "note": "Current render retries rebuild the full episode package and master.",
        },
        {
            "script": "52_quality_gate.py",
            "args": gate_args,
            "note": "Recompute release status and the weak-scene queue after rerender.",
        },
    ]
    if update_bible:
        plan.append(
            {
                "script": "16_build_series_bible.py",
                "args": [],
                "note": "Refresh the series bible after the retry run.",
            }
        )
    return plan


def build_regeneration_reason(queue_entry: dict[str, Any]) -> str:
    weaknesses = queue_entry.get("weaknesses", []) if isinstance(queue_entry.get("weaknesses"), list) else []
    weakness_text = ", ".join(clean_text(item) for item in weaknesses if clean_text(item))
    quality_percent = int(queue_entry.get("quality_percent", 0) or 0)
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
    quality["last_regeneration_request_source"] = "53_regenerate_weak_scenes.py"
    quality["last_regeneration_queue_manifest"] = str(manifest_path)
    quality["last_regeneration_apply_mode"] = "full_episode_rerender"
    quality["last_regeneration_queue_entry"] = dict(queue_entry)
    quality["queued_for_regeneration"] = True
    updated["quality_assessment"] = quality
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
    apply_requested: bool,
    update_bible: bool,
    strict: bool,
) -> dict[str, Any]:
    episode_id = clean_text(artifacts.get("episode_id", "")) or "episode"
    queue = report.get("regeneration_queue", []) if isinstance(report.get("regeneration_queue"), list) else []
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
        "regeneration_queue_scene_ids": [
            clean_text(entry.get("scene_id", ""))
            for entry in queue
            if isinstance(entry, dict) and clean_text(entry.get("scene_id", ""))
        ],
        "max_regeneration_retries": int(max_regeneration_retries),
        "rerun_scope": "full_episode_pipeline",
        "rerun_reason": (
            "Scene-selective backend retries are not exposed yet, so this helper reruns the current "
            "episode-level storyboard backend and render steps."
        ),
        "rerun_plan": build_rerun_plan(
            episode_id,
            strict=strict,
            update_bible=update_bible,
            max_regeneration_retries=max_regeneration_retries,
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
    production_package_path = resolve_stored_project_path(artifacts.get("production_package", ""))
    if not production_package_path.exists():
        raise RuntimeError(f"Production package is missing: {production_package_path}")

    max_regeneration_retries = effective_retry_limit(cfg, args.max_regeneration_retries)
    report_path, report = ensure_quality_gate_report(
        cfg,
        artifacts,
        strict=bool(args.strict),
        max_regeneration_retries=max_regeneration_retries,
        refresh=bool(args.refresh_quality_gate),
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
        run_script(script_name, extra_args, allow_failure=False)

    applied_at = utc_timestamp()
    refreshed_artifacts = resolve_episode_artifacts(cfg, episode_id)
    refreshed_package_path = resolve_stored_project_path(refreshed_artifacts.get("production_package", ""))
    if refreshed_package_path.exists():
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
