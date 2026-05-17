#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "ai_series_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from support_scripts.pipeline_common import (
    SCRIPT_DIR,
    WORKSPACE_ROOT,
    add_shared_worker_arguments,
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
    runtime_python,
    stored_path_if_present,
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
    parser.add_argument(
        "--auto-retry",
        action="store_true",
        help="Automatically trigger 19_regenerate_weak_scenes.py --apply when the quality gate fails and weak scenes are queued.",
    )
    parser.add_argument(
        "--no-auto-retry",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    add_shared_worker_arguments(parser)
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


def release_mode_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("release_mode", {}) or {}) if isinstance(cfg.get("release_mode"), dict) else {}


def auto_retry_enabled(cfg: dict[str, Any], args: argparse.Namespace) -> bool:
    if bool(getattr(args, "no_auto_retry", False)):
        return False
    release_cfg = release_mode_config(cfg)
    return bool(args.auto_retry or release_cfg.get("auto_retry_failed_gate", False))


def strict_warnings_enabled(cfg: dict[str, Any], args: argparse.Namespace) -> bool:
    release_cfg = release_mode_config(cfg)
    return bool(args.strict or release_cfg.get("strict_warnings", False))


def retry_until_pass_enabled(cfg: dict[str, Any]) -> bool:
    release_cfg = release_mode_config(cfg)
    return bool(release_cfg.get("retry_until_pass", True))


def max_auto_retry_cycles(cfg: dict[str, Any]) -> int:
    release_cfg = release_mode_config(cfg)
    return max(0, int(release_cfg.get("max_auto_retry_cycles", 12) or 0))


def force_full_rerender_when_queue_empty(cfg: dict[str, Any]) -> bool:
    release_cfg = release_mode_config(cfg)
    return bool(release_cfg.get("auto_retry_force_full_rerender_when_queue_empty", True))


def report_release_gate_passed(report: dict[str, Any]) -> bool:
    release_gate = report.get("release_gate", {}) if isinstance(report.get("release_gate"), dict) else {}
    return bool(release_gate.get("passed", False)) and not bool(report.get("strict_fail", False))


def report_regeneration_queue(report: dict[str, Any]) -> list[dict[str, Any]]:
    queue = report.get("regeneration_queue", []) if isinstance(report.get("regeneration_queue"), list) else []
    return [entry for entry in queue if isinstance(entry, dict)]


def build_auto_retry_command(
    cfg: dict[str, Any],
    episode_id: str,
    args: argparse.Namespace | None = None,
    *,
    strict: bool = False,
    force: bool = False,
) -> list[str]:
    release_cfg = release_mode_config(cfg)
    command = [
        str(runtime_python()),
        str(WORKSPACE_ROOT / "19_regenerate_weak_scenes.py"),
        "--episode-id",
        episode_id,
        "--apply",
        "--max-regeneration-retries",
        str(int(release_cfg.get("max_regeneration_retries", 3) or 3)),
    ]
    if args is not None and args.min_quality is not None:
        command.extend(["--min-quality", str(float(args.min_quality))])
    if args is not None and args.max_weak_scenes is not None:
        command.extend(["--max-weak-scenes", str(int(args.max_weak_scenes))])
    if args is not None and args.max_regeneration_batch is not None:
        command.extend(["--max-regeneration-batch", str(int(args.max_regeneration_batch))])
    if strict:
        command.append("--strict")
    if force:
        command.append("--force")
    if bool(release_cfg.get("auto_retry_update_bible", False)):
        command.append("--update-bible")
    return command


def reload_quality_gate_report(cfg: dict[str, Any], episode_id: str) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    refreshed_cfg = load_config()
    refreshed_artifacts = resolve_episode_artifacts(refreshed_cfg, episode_id)
    if not refreshed_artifacts:
        refreshed_artifacts = resolve_episode_artifacts(cfg, episode_id)
    report_path = quality_gate_report_path(refreshed_artifacts)
    if not report_path.exists():
        raise RuntimeError(f"Quality gate report is missing after auto-retry: {report_path}")
    report = read_json(report_path, {})
    if not isinstance(report, dict):
        raise RuntimeError(f"Quality gate report is not valid JSON after auto-retry: {report_path}")
    return refreshed_cfg, refreshed_artifacts, report_path, report


def load_scene_quality_rows(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    package_path = stored_path_if_present(artifacts.get("production_package", ""))
    if not package_path or not package_path.exists() or not package_path.is_file():
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
    candidate = stored_path_if_present(path_value)
    if not candidate:
        return False
    return candidate.exists() and candidate.is_file()


def load_production_package(artifacts: dict[str, Any]) -> dict[str, Any]:
    package_path = stored_path_if_present(artifacts.get("production_package", ""))
    if not package_path or not package_path.exists() or not package_path.is_file():
        return {}
    payload = read_json(package_path, {})
    return payload if isinstance(payload, dict) else {}


def safe_ratio(numerator: int, denominator: int) -> float:
    return round(float(numerator) / max(1.0, float(denominator)), 4)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def scene_content_quality_checks(package_payload: dict[str, Any]) -> dict[str, Any]:
    scenes = package_payload.get("scenes", []) if isinstance(package_payload.get("scenes", []), list) else []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    counters = {
        "missing_behavior_scene_count": 0,
        "missing_conflict_or_purpose_scene_count": 0,
        "missing_relationship_context_scene_count": 0,
        "missing_voice_metadata_line_count": 0,
        "missing_voice_clone_output_scene_count": 0,
        "missing_lipsync_output_scene_count": 0,
        "missing_reference_data_line_count": 0,
        "template_heavy_scene_count": 0,
    }
    total_lines = 0
    total_template_lines = 0
    if not package_payload:
        return {
            "scene_count": 0,
            "scene_rows": [],
            "warnings": ["Production package content could not be inspected."],
            **counters,
            "generic_template_line_ratio": 0.0,
            "placeholder_metrics": {
                "character_style_similarity": "pending_external_metric",
                "lip_sync_confidence": "pending_backend_metric",
            },
        }

    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene.get("scene_id", "") or "scene").strip()
        behavior_constraints = scene.get("behavior_constraints", []) if isinstance(scene.get("behavior_constraints", []), list) else []
        dialogue_style_constraints = (
            scene.get("dialogue_style_constraints", [])
            if isinstance(scene.get("dialogue_style_constraints", []), list)
            else []
        )
        relationship_context = scene.get("relationship_context", []) if isinstance(scene.get("relationship_context", []), list) else []
        has_behavior = bool(behavior_constraints and dialogue_style_constraints)
        has_conflict_or_purpose = bool(str(scene.get("scene_purpose", "") or "").strip() and str(scene.get("conflict", "") or "").strip())
        if not has_behavior:
            counters["missing_behavior_scene_count"] += 1
        if not has_conflict_or_purpose:
            counters["missing_conflict_or_purpose_scene_count"] += 1
        if not relationship_context:
            counters["missing_relationship_context_scene_count"] += 1

        dialogue_sources = scene.get("dialogue_sources", []) if isinstance(scene.get("dialogue_sources", []), list) else []
        scene_template_lines = sum(
            1
            for item in dialogue_sources
            if isinstance(item, dict) and str(item.get("type", "") or "").strip() == "generated_template"
        )
        total_template_lines += scene_template_lines
        total_lines += len(dialogue_sources)
        template_ratio = safe_ratio(scene_template_lines, len(dialogue_sources))
        if dialogue_sources and template_ratio > 0.5:
            counters["template_heavy_scene_count"] += 1

        voice_clone = scene.get("voice_clone", {}) if isinstance(scene.get("voice_clone", {}), dict) else {}
        voice_lines = voice_clone.get("lines", []) if isinstance(voice_clone.get("lines", []), list) else []
        for line in voice_lines:
            if not isinstance(line, dict):
                counters["missing_voice_metadata_line_count"] += 1
                continue
            missing_metadata = (
                not str(line.get("emotion", "") or "").strip()
                or not str(line.get("pace", "") or "").strip()
                or safe_float(line.get("target_duration_seconds", 0.0), 0.0) <= 0.0
                or not isinstance(line.get("voice_reference_priority", []), list)
            )
            if missing_metadata:
                counters["missing_voice_metadata_line_count"] += 1
            if not line.get("reference_audio_candidates"):
                counters["missing_reference_data_line_count"] += 1
        voice_outputs = voice_clone.get("target_outputs", {}) if isinstance(voice_clone.get("target_outputs", {}), dict) else {}
        if bool(voice_clone.get("required", False)) and not artifact_path_exists(voice_outputs.get("scene_dialogue_audio", "")):
            counters["missing_voice_clone_output_scene_count"] += 1

        lip_sync = scene.get("lip_sync", {}) if isinstance(scene.get("lip_sync", {}), dict) else {}
        lipsync_outputs = lip_sync.get("target_outputs", {}) if isinstance(lip_sync.get("target_outputs", {}), dict) else {}
        if bool(lip_sync.get("required", False)) and not artifact_path_exists(lipsync_outputs.get("lipsync_video", "")):
            counters["missing_lipsync_output_scene_count"] += 1
        rows.append(
            {
                "scene_id": scene_id,
                "has_behavior_constraints": has_behavior,
                "has_conflict_or_purpose": has_conflict_or_purpose,
                "has_relationship_context": bool(relationship_context),
                "generic_template_line_ratio": template_ratio,
                "voice_line_count": len(voice_lines),
                "lipsync_backend": str(lip_sync.get("selected_backend", "") or "").strip(),
            }
        )

    if counters["missing_behavior_scene_count"]:
        warnings.append(f"{counters['missing_behavior_scene_count']} scene(s) are missing behavior or dialogue-style constraints.")
    if counters["missing_conflict_or_purpose_scene_count"]:
        warnings.append(f"{counters['missing_conflict_or_purpose_scene_count']} scene(s) are missing scene purpose/conflict metadata.")
    if counters["missing_relationship_context_scene_count"]:
        warnings.append(f"{counters['missing_relationship_context_scene_count']} scene(s) are missing relationship context.")
    if counters["missing_voice_metadata_line_count"]:
        warnings.append(f"{counters['missing_voice_metadata_line_count']} voice line(s) are missing emotion/pace/energy/duration metadata.")
    if counters["missing_reference_data_line_count"]:
        warnings.append(f"{counters['missing_reference_data_line_count']} voice line(s) are missing character reference audio candidates.")
    if counters["missing_voice_clone_output_scene_count"]:
        warnings.append(f"{counters['missing_voice_clone_output_scene_count']} scene(s) are missing voice-clone output audio.")
    if counters["missing_lipsync_output_scene_count"]:
        warnings.append(f"{counters['missing_lipsync_output_scene_count']} scene(s) are missing lip-sync output video.")
    if counters["template_heavy_scene_count"]:
        warnings.append(f"{counters['template_heavy_scene_count']} scene(s) are still too template-heavy.")
    return {
        "scene_count": len(rows),
        "scene_rows": rows,
        "warnings": warnings,
        **counters,
        "generic_template_line_ratio": safe_ratio(total_template_lines, total_lines),
        "placeholder_metrics": {
            "character_style_similarity": "pending_external_metric",
            "lip_sync_confidence": "pending_backend_metric",
        },
    }


def build_warnings(artifacts: dict[str, Any], content_checks: dict[str, Any] | None = None) -> list[str]:
    warnings: list[str] = []
    readiness = str(artifacts.get("production_readiness", "") or "").strip().lower()
    if readiness not in {"ready", "fully_generated_episode_ready"}:
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
    if isinstance(content_checks, dict):
        warnings.extend(str(item) for item in (content_checks.get("warnings", []) or []) if str(item).strip())
    return warnings


def quality_gate_report_path(artifacts: dict[str, Any]) -> Path:
    delivery_root = stored_path_if_present(artifacts.get("delivery_bundle_root", ""))
    if delivery_root and delivery_root.exists() and delivery_root.is_dir():
        return delivery_root / f"{artifacts.get('episode_id', 'episode')}_quality_gate.json"
    package_path = stored_path_if_present(artifacts.get("production_package", ""))
    if package_path and package_path.exists() and package_path.is_file():
        return package_path.parent / f"{artifacts.get('episode_id', 'episode')}_quality_gate.json"
    return Path(f"{artifacts.get('episode_id', 'episode')}_quality_gate.json")


def persist_quality_gate_result(artifacts: dict[str, Any], report_path: Path, report: dict[str, Any]) -> None:
    shotlist_path = stored_path_if_present(artifacts.get("shotlist", ""))
    render_manifest_path = stored_path_if_present(artifacts.get("render_manifest", ""))
    release_gate = report.get("release_gate", {}) if isinstance(report.get("release_gate"), dict) else {}
    updates = {
        "quality_gate_report": str(report_path),
        "release_gate": release_gate,
        "release_gate_passed": bool(release_gate.get("passed", False)),
        "quality_gate_warnings": list(report.get("warnings", []) or []),
        "regeneration_queue_count": len(report.get("regeneration_queue", []) or []),
    }
    for target_path in (shotlist_path, render_manifest_path):
        if not target_path or not target_path.exists() or not target_path.is_file():
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
    release_cfg = release_mode_config(effective_cfg)
    regeneration_queue = queue_scenes_for_regeneration(
        scene_quality_rows,
        watch_threshold=float(release_cfg.get("watch_threshold", 0.52) or 0.52),
        release_threshold=float(release_cfg.get("min_episode_quality", 0.68) or 0.68),
        max_regeneration_batch=int(release_cfg.get("max_regeneration_batch", 8) or 8),
        max_regeneration_retries=int(release_cfg.get("max_regeneration_retries", 3) or 3),
    )
    production_package_payload = load_production_package(artifacts)
    content_checks = scene_content_quality_checks(production_package_payload)
    warnings = build_warnings(artifacts, content_checks)
    strict_fail = bool(strict_warnings_enabled(effective_cfg, args) and warnings)

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
        "content_quality_checks": content_checks,
        "strict_fail": strict_fail,
        "max_regeneration_retries": int(release_cfg.get("max_regeneration_retries", 3) or 3),
        "regeneration_queue": regeneration_queue,
    }

    report_path = quality_gate_report_path(artifacts)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    persist_quality_gate_result(artifacts, report_path, report)

    episode_id_text = str(artifacts.get("episode_id") or "-")
    display_title_text = str(artifacts.get("display_title") or "").strip()
    episode_label = (
        f"{display_title_text} ({episode_id_text})"
        if display_title_text and display_title_text != episode_id_text
        else episode_id_text
    )
    print(f"Episode: {episode_label}")
    print(f"Display title: {display_title_text or '-'}")
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

    if auto_retry_enabled(effective_cfg, args):
        episode_id = str(
            artifacts.get("episode_id", "")
            or artifacts.get("display_title", "")
            or args.episode_id
            or ""
        ).strip()
        if not episode_id:
            raise RuntimeError("Auto-retry could not resolve an episode ID for the failing quality gate.")
        retry_forever = retry_until_pass_enabled(effective_cfg)
        cycle_limit = max_auto_retry_cycles(effective_cfg)
        current_queue = regeneration_queue
        cycle = 0
        while retry_forever or cycle == 0:
            if cycle_limit and cycle >= cycle_limit:
                info(f"Auto-retry stopped after {cycle_limit} cycle(s) without a passing release gate.")
                break
            force_retry = not bool(current_queue)
            if force_retry and not force_full_rerender_when_queue_empty(effective_cfg):
                info("Auto-retry has no weak-scene queue left and full rerender fallback is disabled.")
                break
            cycle += 1
            if current_queue:
                info(
                    f"Auto-retry cycle {cycle}: regenerating/rerendering "
                    f"{len(current_queue)} weak scene(s)."
                )
            else:
                info(f"Auto-retry cycle {cycle}: forcing a full rerender because the gate still fails.")
            retry_cmd = build_auto_retry_command(
                effective_cfg,
                episode_id,
                args,
                strict=strict_warnings_enabled(effective_cfg, args),
                force=force_retry,
            )
            result = subprocess.run(retry_cmd, cwd=str(WORKSPACE_ROOT), text=True)
            if result.returncode != 0:
                raise SystemExit(result.returncode)
            _refreshed_cfg, refreshed_artifacts, refreshed_report_path, refreshed_report = reload_quality_gate_report(
                effective_cfg,
                episode_id,
            )
            print(f"Auto-retry report: {refreshed_report_path}")
            print(f"Auto-retry release gate: {'PASS' if report_release_gate_passed(refreshed_report) else 'FAIL'}")
            if args.print_json:
                print(json.dumps(refreshed_report, indent=2, ensure_ascii=False))
            if report_release_gate_passed(refreshed_report):
                ok("QUALITY GATE PASSED AFTER AUTO-RETRY")
                return
            current_queue = report_regeneration_queue(refreshed_report)
            if not retry_forever:
                break
            if refreshed_artifacts:
                info("Auto-retry finished a cycle, but the refreshed episode still does not pass the release gate.")

    raise SystemExit(1)


if __name__ == "__main__":
    main()

