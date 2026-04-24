#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_common import (
    generated_episode_artifacts,
    headline,
    info,
    latest_generated_episode_artifacts,
    load_config,
    ok,
    queue_scenes_for_regeneration,
    read_json,
    release_quality_gate,
    resolve_stored_project_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a release-style quality gate for a generated episode.")
    parser.add_argument("--episode-id", help="Target episode ID. Uses the newest generated episode by default.")
    parser.add_argument("--min-quality", type=float, help="Override the minimum episode quality threshold.")
    parser.add_argument("--max-weak-scenes", type=int, help="Override the maximum release-threshold weak scenes.")
    parser.add_argument(
        "--max-regeneration-batch",
        type=int,
        help="Override how many scenes should be suggested for the regeneration queue.",
    )
    parser.add_argument(
        "--max-regeneration-retries",
        type=int,
        help="Override how many rerender attempts each weak scene may receive before it stops being queued.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if warnings remain even when the quality gate passes.")
    parser.add_argument("--print-json", action="store_true", help="Also print the full gate report as JSON.")
    return parser.parse_args()


def resolve_episode_artifacts(cfg: dict[str, Any], episode_id: str | None) -> dict[str, Any]:
    requested = str(episode_id or "").strip()
    if not requested or requested.lower() == "latest":
        return latest_generated_episode_artifacts(cfg)
    return generated_episode_artifacts(cfg, requested)


def gate_config(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = dict(cfg)
    release_cfg = dict(cfg.get("release_mode", {}) or {}) if isinstance(cfg.get("release_mode"), dict) else {}
    if args.min_quality is not None:
        release_cfg["min_episode_quality"] = float(args.min_quality)
    if args.max_weak_scenes is not None:
        release_cfg["max_weak_scenes"] = int(args.max_weak_scenes)
    if args.max_regeneration_batch is not None:
        release_cfg["max_regeneration_batch"] = int(args.max_regeneration_batch)
    if args.max_regeneration_retries is not None:
        release_cfg["max_regeneration_retries"] = int(args.max_regeneration_retries)
    merged["release_mode"] = release_cfg
    return merged


def load_scene_quality_rows(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    package_path = resolve_stored_project_path(artifacts.get("production_package", ""))
    if not package_path.exists():
        return []
    package_payload = read_json(package_path, {})
    if not isinstance(package_payload, dict):
        return []
    rows: list[dict[str, Any]] = []
    for scene in package_payload.get("scenes", []) or []:
        if not isinstance(scene, dict):
            continue
        quality = scene.get("quality_assessment", {}) if isinstance(scene.get("quality_assessment"), dict) else {}
        if quality:
            rows.append(quality)
    return rows


def artifact_path_exists(path_value: object) -> bool:
    candidate = resolve_stored_project_path(path_value)
    return candidate.exists() and candidate.is_file()


def build_warnings(artifacts: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if str(artifacts.get("production_readiness", "") or "").strip().lower() != "ready":
        warnings.append(f"Production readiness is {artifacts.get('production_readiness') or 'unknown'}.")
    if int(artifacts.get("backend_runner_failed_count", 0) or 0) > 0:
        warnings.append(
            f"External backend runners still failed for {int(artifacts.get('backend_runner_failed_count', 0) or 0)} tasks."
        )
    if int(artifacts.get("backend_runner_pending_count", 0) or 0) > 0:
        warnings.append(
            f"External backend runners still have {int(artifacts.get('backend_runner_pending_count', 0) or 0)} pending tasks."
        )
    if not artifact_path_exists(artifacts.get("production_package", "")):
        warnings.append("Production package path is missing.")
    if not artifact_path_exists(artifacts.get("final_render", "")):
        warnings.append("Final render path is missing.")
    if not artifact_path_exists(artifacts.get("full_generated_episode", "")):
        warnings.append("Full generated-episode master path is missing.")
    if not artifact_path_exists(artifacts.get("delivery_episode", "")):
        warnings.append("Delivery watchable episode path is missing.")
    return warnings


def quality_gate_report_path(artifacts: dict[str, Any]) -> Path:
    delivery_root = resolve_stored_project_path(artifacts.get("delivery_bundle_root", ""))
    if delivery_root.exists():
        return delivery_root / f"{artifacts.get('episode_id', 'episode')}_quality_gate.json"
    package_path = resolve_stored_project_path(artifacts.get("production_package", ""))
    if package_path.exists():
        return package_path.parent / f"{artifacts.get('episode_id', 'episode')}_quality_gate.json"
    return Path(f"{artifacts.get('episode_id', 'episode')}_quality_gate.json")


def persist_quality_gate_result(artifacts: dict[str, Any], report_path: Path, report: dict[str, Any]) -> None:
    shotlist_path = resolve_stored_project_path(artifacts.get("shotlist", ""))
    render_manifest_path = resolve_stored_project_path(artifacts.get("render_manifest", ""))
    release_gate = report.get("release_gate", {}) if isinstance(report.get("release_gate"), dict) else {}
    updates = {
        "quality_gate_report": str(report_path),
        "release_gate": release_gate,
        "release_gate_passed": bool(release_gate.get("passed", False)),
        "quality_gate_warnings": list(report.get("warnings", []) or []),
        "regeneration_queue_count": len(report.get("regeneration_queue", []) or []),
    }
    for target_path in (shotlist_path, render_manifest_path):
        if not target_path.exists():
            continue
        payload = read_json(target_path, {})
        if not isinstance(payload, dict):
            continue
        payload.update(updates)
        write_json(target_path, payload)


def main() -> None:
    args = parse_args()
    headline("Quality Gate Check")
    cfg = load_config()
    effective_cfg = gate_config(cfg, args)
    artifacts = resolve_episode_artifacts(effective_cfg, args.episode_id)

    if not artifacts:
        info("No generated episode artifacts were found.")
        raise SystemExit(1)

    scene_quality_rows = load_scene_quality_rows(artifacts)
    release_result = release_quality_gate(artifacts, effective_cfg)
    release_cfg = effective_cfg.get("release_mode", {}) if isinstance(effective_cfg.get("release_mode"), dict) else {}
    regeneration_queue = queue_scenes_for_regeneration(
        scene_quality_rows,
        watch_threshold=float(release_cfg.get("watch_threshold", 0.52) or 0.52),
        release_threshold=float(release_cfg.get("min_episode_quality", 0.68) or 0.68),
        max_regeneration_batch=int(release_cfg.get("max_regeneration_batch", 8) or 8),
        max_regeneration_retries=int(release_cfg.get("max_regeneration_retries", 3) or 3),
    )
    warnings = build_warnings(artifacts)
    strict_fail = bool(args.strict and warnings)

    report = {
        "episode_id": artifacts.get("episode_id", ""),
        "display_title": artifacts.get("display_title", ""),
        "render_mode": artifacts.get("render_mode", ""),
        "production_readiness": artifacts.get("production_readiness", ""),
        "quality_label": artifacts.get("quality_label", ""),
        "quality_percent": int(artifacts.get("quality_percent", 0) or 0),
        "minimum_scene_quality_label": artifacts.get("minimum_scene_quality_label", ""),
        "minimum_scene_quality_percent": int(artifacts.get("minimum_scene_quality_percent", 0) or 0),
        "scene_ids_below_watch_threshold": list(artifacts.get("scene_ids_below_watch_threshold", []) or []),
        "scene_ids_below_release_threshold": list(artifacts.get("scene_ids_below_release_threshold", []) or []),
        "remaining_backend_tasks": list(artifacts.get("remaining_backend_tasks", []) or []),
        "release_gate": release_result,
        "warnings": warnings,
        "strict_fail": strict_fail,
        "max_regeneration_retries": int(release_cfg.get("max_regeneration_retries", 3) or 3),
        "regeneration_queue": regeneration_queue,
    }

    report_path = quality_gate_report_path(artifacts)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    persist_quality_gate_result(artifacts, report_path, report)

    print(f"Episode: {artifacts.get('episode_id') or '-'}")
    print(f"Display title: {artifacts.get('display_title') or '-'}")
    print(f"Readiness: {artifacts.get('production_readiness') or '-'}")
    print(
        f"Quality: {int(artifacts.get('quality_percent', 0) or 0)}% "
        f"({artifacts.get('quality_label') or 'unknown'})"
    )
    print(
        f"Minimum scene quality: {int(artifacts.get('minimum_scene_quality_percent', 0) or 0)}% "
        f"({artifacts.get('minimum_scene_quality_label') or 'unknown'})"
    )
    print(
        f"Release gate: {'PASS' if release_result.get('passed') else 'FAIL'} | "
        f"min={int(float(release_result.get('min_quality_required', 0.68) or 0.68) * 100)}% | "
        f"weak scenes={int(release_result.get('weak_scene_count', 0) or 0)}/"
        f"{int(release_result.get('max_weak_scenes_allowed', 0) or 0)}"
    )
    print(f"Regeneration queue size: {len(regeneration_queue)}")
    print(f"Report: {report_path}")

    if warnings:
        for warning in warnings:
            print(f"WARNING: {warning}")
    else:
        ok("No quality-gate warnings remain.")

    if args.print_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    if release_result.get("passed") and not strict_fail:
        ok("QUALITY GATE PASSED")
        return

    info("QUALITY GATE FAILED")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
