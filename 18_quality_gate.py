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
from datetime import datetime, timezone
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
    resolve_project_path,
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


def regeneration_queue_has_blocked_scope(queue: list[dict[str, Any]]) -> bool:
    for entry in queue:
        if not isinstance(entry, dict):
            continue
        scope = str(entry.get("regeneration_scope", "") or "").strip()
        if scope.startswith("blocked_"):
            return True
    return False


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


def merge_realism_into_scene_quality_rows(
    scene_quality_rows: list[dict[str, Any]],
    content_checks: dict[str, Any],
) -> list[dict[str, Any]]:
    realism_index = {
        str(row.get("scene_id", "") or "").strip(): row
        for row in (content_checks.get("realism_rows", []) if isinstance(content_checks.get("realism_rows", []), list) else [])
        if isinstance(row, dict) and str(row.get("scene_id", "") or "").strip()
    }
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in scene_quality_rows:
        if not isinstance(row, dict):
            continue
        scene_id = str(row.get("scene_id", "") or "").strip()
        realism = realism_index.get(scene_id, {})
        combined = dict(row)
        if realism:
            original_score = safe_float(combined.get("quality_score", 0.0), 0.0)
            realism_score = safe_float(realism.get("realism_score", original_score), original_score)
            combined["realism_score"] = realism_score
            combined["realism_percent"] = int(round(realism_score * 100.0))
            combined["quality_score"] = min(original_score, realism_score)
            combined["quality_percent"] = int(round(combined["quality_score"] * 100.0))
            weaknesses = list(combined.get("weaknesses", []) or []) if isinstance(combined.get("weaknesses", []), list) else []
            weaknesses.extend(str(reason) for reason in realism.get("failed_reasons", []) if str(reason).strip())
            combined["weaknesses"] = list(dict.fromkeys(weaknesses))
            combined["regeneration_hints"] = realism.get("regeneration_hints", {}) if isinstance(realism.get("regeneration_hints", {}), dict) else {}
            combined["regeneration_scope"] = str(realism.get("regeneration_scope", "") or "").strip()
            component_scores = dict(combined.get("component_scores", {}) or {}) if isinstance(combined.get("component_scores", {}), dict) else {}
            component_scores["realism"] = realism_score
            combined["component_scores"] = component_scores
        merged.append(combined)
        if scene_id:
            seen.add(scene_id)
    for scene_id, realism in realism_index.items():
        if scene_id in seen:
            continue
        realism_score = safe_float(realism.get("realism_score", 0.0), 0.0)
        merged.append(
            {
                "scene_id": scene_id,
                "quality_score": realism_score,
                "quality_percent": int(round(realism_score * 100.0)),
                "realism_score": realism_score,
                "realism_percent": int(round(realism_score * 100.0)),
                "component_scores": {"realism": realism_score},
                "weaknesses": list(realism.get("failed_reasons", []) or []),
                "regeneration_hints": realism.get("regeneration_hints", {}) if isinstance(realism.get("regeneration_hints", {}), dict) else {},
                "regeneration_scope": str(realism.get("regeneration_scope", "") or "").strip(),
            }
        )
    return merged


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


def clamp_score(value: object, default: float = 0.0) -> float:
    return max(0.0, min(1.0, safe_float(value, default)))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def finished_episode_mode_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    value = cfg.get("finished_episode_mode", {})
    return dict(value) if isinstance(value, dict) else {}


def finished_episode_mode_enabled(cfg: dict[str, Any] | None) -> bool:
    return bool(finished_episode_mode_config(cfg).get("enabled", False))


def technical_metric(metric: str, status: str, score: float, reason: str, inputs: dict[str, Any] | None = None, tool: str = "") -> dict[str, Any]:
    return {
        "metric": metric,
        "status": status,
        "score": round(clamp_score(score), 4),
        "reason": reason,
        "inputs": inputs if isinstance(inputs, dict) else {},
        "tool": tool,
        "created_at": utc_now_iso(),
    }


def real_video_source_type(value: object) -> bool:
    return str(value or "").strip() in {"generated_scene_video", "generated_lipsync_video", "final_scene_master"}


def backend_manifest_payload(path_value: object) -> dict[str, Any]:
    path = stored_path_if_present(path_value)
    if not path or not path.exists() or not path.is_file():
        return {}
    payload = read_json(path, {})
    return payload if isinstance(payload, dict) else {}


def scene_backend_manifest_rows(scene: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifests = scene.get("backend_manifests", {}) if isinstance(scene.get("backend_manifests", {}), dict) else {}
    for runner_name, path_value in manifests.items():
        payload = backend_manifest_payload(path_value)
        rows.append(
            {
                "runner_name": str(runner_name),
                "manifest_path": str(path_value or ""),
                "exists": bool(payload),
                "payload": payload,
            }
        )
    for row in scene.get("backend_manifest_rows", []) if isinstance(scene.get("backend_manifest_rows", []), list) else []:
        if not isinstance(row, dict):
            continue
        path_value = row.get("manifest_path", "")
        if any(existing.get("manifest_path") == str(path_value or "") for existing in rows):
            continue
        payload = backend_manifest_payload(path_value)
        rows.append(
            {
                "runner_name": str(row.get("runner_name", "") or ""),
                "manifest_path": str(path_value or ""),
                "exists": bool(payload),
                "payload": payload,
            }
        )
    return rows


def scene_backend_integrity(scene: dict[str, Any]) -> dict[str, Any]:
    rows = scene_backend_manifest_rows(scene)
    required_sections = [
        ("finished_episode_video_runner", scene.get("video_generation", {})),
        ("finished_episode_voice_runner", scene.get("voice_clone", {})),
        ("finished_episode_lipsync_runner", scene.get("lip_sync", {})),
    ]
    missing: list[str] = []
    fallback: list[str] = []
    placeholder: list[str] = []
    stale: list[str] = []
    row_by_runner = {str(row.get("runner_name", "")): row for row in rows}
    for runner_name, section in required_sections:
        if not isinstance(section, dict) or not bool(section.get("required", False)):
            continue
        row = row_by_runner.get(runner_name)
        if not row or not bool(row.get("exists", False)):
            missing.append(runner_name)
            continue
        payload = row.get("payload", {}) if isinstance(row.get("payload", {}), dict) else {}
        if bool(payload.get("fallback_used", False)):
            fallback.append(runner_name)
        if bool(payload.get("placeholder_used", False)):
            placeholder.append(runner_name)
        if bool(payload.get("stale_output", False)):
            stale.append(runner_name)
    problem_count = len(missing) + len(fallback) + len(placeholder) + len(stale)
    expected_count = sum(1 for _, section in required_sections if isinstance(section, dict) and bool(section.get("required", False)))
    score = 1.0 if expected_count == 0 else max(0.0, 1.0 - problem_count / max(1, expected_count))
    return {
        "manifest_count": len(rows),
        "expected_count": expected_count,
        "missing": missing,
        "fallback": fallback,
        "placeholder": placeholder,
        "stale": stale,
        "score": round(score, 4),
    }


GENERIC_DIALOGUE_MARKERS = {
    "clear plan",
    "chaos",
    "step by step",
    "we need a plan",
    "wir brauchen einen plan",
    "schritt für schritt",
    "durcheinander",
    "funktioniert am ende doch",
}


def scene_voice_lines(scene: dict[str, Any]) -> list[dict[str, Any]]:
    voice_clone = scene.get("voice_clone", {}) if isinstance(scene.get("voice_clone", {}), dict) else {}
    return [line for line in (voice_clone.get("lines", []) if isinstance(voice_clone.get("lines", []), list) else []) if isinstance(line, dict)]


def scene_dialogue_texts(scene: dict[str, Any]) -> list[str]:
    return [str(line.get("text", "") or "").strip() for line in scene_voice_lines(scene) if str(line.get("text", "") or "").strip()]


def generic_dialogue_ratio(texts: list[str], dialogue_sources: list[dict[str, Any]]) -> float:
    generic_count = 0
    for text in texts:
        lowered = text.lower()
        repeated_tokens = len(lowered.split()) != len(set(lowered.split())) and len(lowered.split()) <= 10
        if repeated_tokens or any(marker in lowered for marker in GENERIC_DIALOGUE_MARKERS):
            generic_count += 1
    template_count = 0.0
    for item in dialogue_sources:
        if not isinstance(item, dict):
            continue
        source_type = str(item.get("type", "") or "").strip()
        if source_type == "generated_template":
            template_count += 1.0
        elif source_type == "behavior_guided":
            template_count += 0.35
    return safe_ratio(generic_count + template_count, max(len(texts), len(dialogue_sources)))


def speaker_change_ratio(lines: list[dict[str, Any]]) -> float:
    speakers = [str(line.get("speaker_name", "") or line.get("speaker", "") or "").strip() for line in lines]
    speakers = [speaker for speaker in speakers if speaker]
    if len(speakers) <= 1:
        return 0.0
    changes = sum(1 for left, right in zip(speakers, speakers[1:]) if left != right)
    return round(changes / max(1, len(speakers) - 1), 4)


def scene_realism_row(
    scene: dict[str, Any],
    *,
    has_behavior: bool,
    has_conflict_or_purpose: bool,
    has_relationship_context: bool,
    has_scene_function: bool,
    has_shot_plan: bool,
    has_set_context: bool,
    has_audio_mix: bool,
    has_real_motion_video: bool,
    backend_integrity: dict[str, Any],
    template_ratio: float,
    missing_voice_metadata: int,
    missing_reference_data: int,
    missing_voice_output: bool,
    missing_lipsync_output: bool,
) -> dict[str, Any]:
    scene_id = str(scene.get("scene_id", "") or "scene").strip()
    voice_lines = scene_voice_lines(scene)
    texts = scene_dialogue_texts(scene)
    writer_room_plan = scene.get("writer_room_plan", {}) if isinstance(scene.get("writer_room_plan", {}), dict) else {}
    callback_targets = scene.get("callback_targets", []) if isinstance(scene.get("callback_targets", []), list) else []
    lip_sync = scene.get("lip_sync", {}) if isinstance(scene.get("lip_sync", {}), dict) else {}
    backend_candidates = lip_sync.get("backend_candidates", []) if isinstance(lip_sync.get("backend_candidates", []), list) else []
    selected_backend = str(lip_sync.get("selected_backend", "") or "").strip()
    backend_available = any(
        isinstance(candidate, dict)
        and str(candidate.get("name", "") or "").strip() == selected_backend
        and bool(candidate.get("available", False))
        for candidate in backend_candidates
    ) if selected_backend else False
    if selected_backend == "wav2lip" and not backend_candidates:
        backend_available = True
    line_count = max(1, len(texts))
    behavior_score = 1.0 if has_behavior and bool(writer_room_plan) else 0.45 if has_behavior else 0.12
    dialogue_style_score = max(0.0, min(1.0, 0.92 - template_ratio * 0.7))
    if speaker_change_ratio(voice_lines) < 0.35 and len(voice_lines) > 2:
        dialogue_style_score -= 0.2
    relationship_score = 1.0 if has_relationship_context else 0.28
    voice_metadata_score = max(0.0, 1.0 - (missing_voice_metadata / line_count))
    reference_coverage_score = max(0.0, 1.0 - (missing_reference_data / line_count))
    template_penalty = min(0.55, template_ratio * 0.55)
    lipsync_backend_score = 1.0 if backend_available and not missing_lipsync_output else 0.55 if selected_backend else 0.18
    scene_structure_score = 1.0 if has_conflict_or_purpose and callback_targets and has_scene_function else 0.62 if has_conflict_or_purpose else 0.22
    video_motion_score = 1.0 if has_real_motion_video else 0.12
    shot_editing_score = 1.0 if has_shot_plan else 0.18
    set_consistency_score = 1.0 if has_set_context else 0.28
    audio_mix_score = 1.0 if has_audio_mix else 0.2
    backend_integrity_score = clamp_score(backend_integrity.get("score", 0.0), 0.0)
    realism_score = max(
        0.0,
        min(
            1.0,
            (
                behavior_score * 0.12
                + dialogue_style_score * 0.13
                + relationship_score * 0.09
                + voice_metadata_score * 0.09
                + reference_coverage_score * 0.07
                + lipsync_backend_score * 0.08
                + scene_structure_score * 0.11
                + video_motion_score * 0.12
                + shot_editing_score * 0.07
                + set_consistency_score * 0.06
                + audio_mix_score * 0.06
                + backend_integrity_score * 0.1
            )
            - template_penalty,
        ),
    )
    failed_reasons: list[str] = []
    hints = {
        "reduce_template_lines": False,
        "increase_speaker_specific_phrases": False,
        "use_relationship_conflict": False,
        "add_scene_payoff": False,
        "repair_voice_metadata": False,
        "rerun_voice_clone": False,
        "rerun_lipsync": False,
        "collect_missing_references": False,
        "generate_real_motion_video": False,
        "repair_shot_plan": False,
        "repair_set_context": False,
        "rerun_audio_mix": False,
        "repair_backend_manifests": False,
    }
    if template_ratio > 0.45:
        failed_reasons.append("dialogue too template-heavy")
        hints["reduce_template_lines"] = True
        hints["increase_speaker_specific_phrases"] = True
    if not has_behavior or not writer_room_plan:
        failed_reasons.append("writer-room behavior plan missing")
        hints["increase_speaker_specific_phrases"] = True
    if not has_relationship_context:
        failed_reasons.append("relationship dynamic missing")
        hints["use_relationship_conflict"] = True
    if not callback_targets:
        failed_reasons.append("no callback or payoff detected")
        hints["add_scene_payoff"] = True
    if missing_voice_metadata:
        failed_reasons.append("voice metadata incomplete")
        hints["repair_voice_metadata"] = True
    if missing_reference_data:
        failed_reasons.append("voice reference coverage incomplete")
        hints["collect_missing_references"] = True
    if missing_voice_output:
        failed_reasons.append("voice-clone output missing")
        hints["rerun_voice_clone"] = True
    if missing_lipsync_output or not selected_backend:
        failed_reasons.append("lip-sync backend/output missing")
        hints["rerun_lipsync"] = True
    if not has_scene_function:
        failed_reasons.append("scene has no clear episode-arc function")
        hints["add_scene_payoff"] = True
    if not has_shot_plan:
        failed_reasons.append("shot plan missing")
        hints["repair_shot_plan"] = True
    if not has_set_context:
        failed_reasons.append("set continuity context missing")
        hints["repair_set_context"] = True
    if not has_real_motion_video:
        failed_reasons.append("real motion video missing")
        hints["generate_real_motion_video"] = True
    if not has_audio_mix:
        failed_reasons.append("audio mix missing")
        hints["rerun_audio_mix"] = True
    if backend_integrity.get("missing"):
        failed_reasons.append("backend manifest missing")
        hints["repair_backend_manifests"] = True
    if backend_integrity.get("fallback") or backend_integrity.get("placeholder"):
        failed_reasons.append("backend manifest reports fallback or placeholder output")
        hints["repair_backend_manifests"] = True
    if backend_integrity.get("stale"):
        failed_reasons.append("backend manifest reports stale output")
        hints["repair_backend_manifests"] = True
    if hints["collect_missing_references"]:
        scope = "blocked_missing_references"
    elif backend_integrity.get("missing"):
        scope = "blocked_missing_backend"
    elif hints["rerun_lipsync"] and not hints["rerun_voice_clone"]:
        scope = "lipsync_only"
    elif hints["rerun_voice_clone"] and not hints["rerun_lipsync"]:
        scope = "voice_only"
    elif hints["rerun_audio_mix"]:
        scope = "audio_mix_only"
    elif hints["repair_shot_plan"]:
        scope = "shot_plan"
    elif hints["generate_real_motion_video"]:
        scope = "visual_rerender"
    elif hints["rerun_voice_clone"] or hints["rerun_lipsync"]:
        scope = "voice_lipsync"
    elif hints["reduce_template_lines"] or hints["increase_speaker_specific_phrases"] or hints["use_relationship_conflict"]:
        scope = "story_dialogue"
    else:
        scope = "scene_rerender"
    return {
        "scene_id": scene_id,
        "realism_score": round(realism_score, 4),
        "realism_percent": int(round(realism_score * 100.0)),
        "component_scores": {
            "behavior_score": round(behavior_score, 4),
            "dialogue_style_score": round(max(0.0, dialogue_style_score), 4),
            "relationship_score": round(relationship_score, 4),
            "voice_metadata_score": round(voice_metadata_score, 4),
            "reference_coverage_score": round(reference_coverage_score, 4),
            "template_penalty": round(template_penalty, 4),
            "lipsync_backend_score": round(lipsync_backend_score, 4),
            "scene_structure_score": round(scene_structure_score, 4),
            "video_motion_score": round(video_motion_score, 4),
            "shot_editing_score": round(shot_editing_score, 4),
            "set_consistency_score": round(set_consistency_score, 4),
            "audio_mix_score": round(audio_mix_score, 4),
            "backend_integrity_score": round(backend_integrity_score, 4),
        },
        "backend_integrity": backend_integrity,
        "failed_reasons": failed_reasons,
        "regeneration_hints": hints,
        "regeneration_scope": scope,
    }


def scene_content_quality_checks(package_payload: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    scenes = package_payload.get("scenes", []) if isinstance(package_payload.get("scenes", []), list) else []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    finished_cfg = finished_episode_mode_config(cfg)
    counters = {
        "missing_behavior_scene_count": 0,
        "missing_conflict_or_purpose_scene_count": 0,
        "missing_relationship_context_scene_count": 0,
        "missing_scene_function_count": 0,
        "missing_shot_plan_scene_count": 0,
        "missing_set_context_scene_count": 0,
        "missing_audio_mix_scene_count": 0,
        "missing_real_motion_video_scene_count": 0,
        "missing_backend_manifest_scene_count": 0,
        "fallback_backend_manifest_scene_count": 0,
        "placeholder_backend_manifest_scene_count": 0,
        "stale_backend_manifest_scene_count": 0,
        "missing_voice_metadata_line_count": 0,
        "missing_voice_clone_output_scene_count": 0,
        "missing_lipsync_output_scene_count": 0,
        "missing_reference_data_line_count": 0,
        "template_heavy_scene_count": 0,
    }
    total_lines = 0
    total_template_lines = 0
    realism_rows: list[dict[str, Any]] = []
    if not package_payload:
        return {
            "scene_count": 0,
            "scene_rows": [],
            "realism_rows": [],
            "average_realism_score": 0.0,
            "warnings": ["Production package content could not be inspected."],
            **counters,
            "generic_template_line_ratio": 0.0,
            "technical_metrics": [
                technical_metric("identity_consistency_score", "unavailable", 0.0, "production package missing"),
                technical_metric("lipsync_confidence_score", "unavailable", 0.0, "production package missing"),
            ],
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
        has_scene_function = bool(str(scene.get("scene_function", "") or "").strip() or str((scene.get("writer_room_plan", {}) if isinstance(scene.get("writer_room_plan", {}), dict) else {}).get("scene_function", "") or "").strip())
        has_shot_plan = bool(scene.get("shot_plan", []) if isinstance(scene.get("shot_plan", []), list) else [])
        has_set_context = bool(str(scene.get("location_id", "") or "").strip() and isinstance(scene.get("set_context", {}), dict) and scene.get("set_context", {}))
        audio_mix = scene.get("audio_mix", {}) if isinstance(scene.get("audio_mix", {}), dict) else {}
        audio_mix_stems = audio_mix.get("stems", {}) if isinstance(audio_mix.get("stems", {}), dict) else {}
        final_mix_path = audio_mix_stems.get("final_mix", "") or audio_mix.get("final_mix", "")
        has_audio_mix = not bool(audio_mix.get("required", True)) or artifact_path_exists(final_mix_path)
        current_outputs = scene.get("current_generated_outputs", {}) if isinstance(scene.get("current_generated_outputs", {}), dict) else {}
        has_real_motion_video = real_video_source_type(current_outputs.get("video_source_type", ""))
        backend_integrity = scene_backend_integrity(scene)
        if not has_behavior:
            counters["missing_behavior_scene_count"] += 1
        if not has_conflict_or_purpose:
            counters["missing_conflict_or_purpose_scene_count"] += 1
        if not relationship_context:
            counters["missing_relationship_context_scene_count"] += 1
        if not has_scene_function:
            counters["missing_scene_function_count"] += 1
        if not has_shot_plan:
            counters["missing_shot_plan_scene_count"] += 1
        if not has_set_context:
            counters["missing_set_context_scene_count"] += 1
        if not has_audio_mix:
            counters["missing_audio_mix_scene_count"] += 1
        if not has_real_motion_video:
            counters["missing_real_motion_video_scene_count"] += 1
        if backend_integrity.get("missing"):
            counters["missing_backend_manifest_scene_count"] += 1
        if backend_integrity.get("fallback"):
            counters["fallback_backend_manifest_scene_count"] += 1
        if backend_integrity.get("placeholder"):
            counters["placeholder_backend_manifest_scene_count"] += 1
        if backend_integrity.get("stale"):
            counters["stale_backend_manifest_scene_count"] += 1

        dialogue_sources = scene.get("dialogue_sources", []) if isinstance(scene.get("dialogue_sources", []), list) else []
        dialogue_texts = scene_dialogue_texts(scene)
        scene_template_lines = sum(
            1
            for item in dialogue_sources
            if isinstance(item, dict) and str(item.get("type", "") or "").strip() == "generated_template"
        )
        total_template_lines += scene_template_lines
        total_lines += len(dialogue_sources)
        template_ratio = max(safe_ratio(scene_template_lines, len(dialogue_sources)), generic_dialogue_ratio(dialogue_texts, dialogue_sources))
        if dialogue_sources and template_ratio > 0.5:
            counters["template_heavy_scene_count"] += 1

        voice_clone = scene.get("voice_clone", {}) if isinstance(scene.get("voice_clone", {}), dict) else {}
        voice_lines = voice_clone.get("lines", []) if isinstance(voice_clone.get("lines", []), list) else []
        scene_missing_voice_metadata = 0
        scene_missing_reference_data = 0
        for line in voice_lines:
            if not isinstance(line, dict):
                counters["missing_voice_metadata_line_count"] += 1
                scene_missing_voice_metadata += 1
                continue
            missing_metadata = (
                not str(line.get("emotion", "") or "").strip()
                or not str(line.get("pace", "") or "").strip()
                or safe_float(line.get("target_duration_seconds", 0.0), 0.0) <= 0.0
                or not str(line.get("delivery_notes", "") or "").strip()
                or not isinstance(line.get("voice_reference_priority", []), list)
            )
            if missing_metadata:
                counters["missing_voice_metadata_line_count"] += 1
                scene_missing_voice_metadata += 1
            if not line.get("reference_audio_candidates"):
                counters["missing_reference_data_line_count"] += 1
                scene_missing_reference_data += 1
        voice_outputs = voice_clone.get("target_outputs", {}) if isinstance(voice_clone.get("target_outputs", {}), dict) else {}
        missing_voice_output = bool(voice_clone.get("required", False)) and not artifact_path_exists(voice_outputs.get("scene_dialogue_audio", ""))
        if missing_voice_output:
            counters["missing_voice_clone_output_scene_count"] += 1

        lip_sync = scene.get("lip_sync", {}) if isinstance(scene.get("lip_sync", {}), dict) else {}
        lipsync_outputs = lip_sync.get("target_outputs", {}) if isinstance(lip_sync.get("target_outputs", {}), dict) else {}
        missing_lipsync_output = bool(lip_sync.get("required", False)) and not artifact_path_exists(lipsync_outputs.get("lipsync_video", ""))
        if missing_lipsync_output:
            counters["missing_lipsync_output_scene_count"] += 1
        realism_row = scene_realism_row(
            scene,
            has_behavior=has_behavior,
            has_conflict_or_purpose=has_conflict_or_purpose,
            has_relationship_context=bool(relationship_context),
            has_scene_function=has_scene_function,
            has_shot_plan=has_shot_plan,
            has_set_context=has_set_context,
            has_audio_mix=has_audio_mix,
            has_real_motion_video=has_real_motion_video,
            backend_integrity=backend_integrity,
            template_ratio=template_ratio,
            missing_voice_metadata=scene_missing_voice_metadata,
            missing_reference_data=scene_missing_reference_data,
            missing_voice_output=missing_voice_output,
            missing_lipsync_output=missing_lipsync_output,
        )
        realism_rows.append(realism_row)
        rows.append(
            {
                "scene_id": scene_id,
                "has_behavior_constraints": has_behavior,
                "has_conflict_or_purpose": has_conflict_or_purpose,
                "has_relationship_context": bool(relationship_context),
                "has_scene_function": has_scene_function,
                "has_shot_plan": has_shot_plan,
                "has_set_context": has_set_context,
                "has_audio_mix": has_audio_mix,
                "has_real_motion_video": has_real_motion_video,
                "backend_integrity": backend_integrity,
                "generic_template_line_ratio": template_ratio,
                "voice_line_count": len(voice_lines),
                "lipsync_backend": str(lip_sync.get("selected_backend", "") or "").strip(),
                "realism_score": realism_row["realism_score"],
                "failed_reasons": realism_row["failed_reasons"],
                "regeneration_hints": realism_row["regeneration_hints"],
            }
        )

    if counters["missing_behavior_scene_count"]:
        warnings.append(f"{counters['missing_behavior_scene_count']} scene(s) are missing behavior or dialogue-style constraints.")
    if counters["missing_conflict_or_purpose_scene_count"]:
        warnings.append(f"{counters['missing_conflict_or_purpose_scene_count']} scene(s) are missing scene purpose/conflict metadata.")
    if counters["missing_relationship_context_scene_count"]:
        warnings.append(f"{counters['missing_relationship_context_scene_count']} scene(s) are missing relationship context.")
    if counters["missing_scene_function_count"]:
        warnings.append(f"{counters['missing_scene_function_count']} scene(s) have no clear episode-arc function.")
    if counters["missing_shot_plan_scene_count"]:
        warnings.append(f"{counters['missing_shot_plan_scene_count']} scene(s) are missing shot plans.")
    if counters["missing_set_context_scene_count"]:
        warnings.append(f"{counters['missing_set_context_scene_count']} scene(s) are missing set continuity context.")
    if counters["missing_audio_mix_scene_count"]:
        warnings.append(f"{counters['missing_audio_mix_scene_count']} scene(s) are missing final audio mix output.")
    if counters["missing_real_motion_video_scene_count"]:
        warnings.append(f"{counters['missing_real_motion_video_scene_count']} scene(s) do not have real generated motion/lip-sync video.")
    if counters["missing_backend_manifest_scene_count"]:
        warnings.append(f"{counters['missing_backend_manifest_scene_count']} scene(s) are missing required backend manifests.")
    if counters["fallback_backend_manifest_scene_count"] or counters["placeholder_backend_manifest_scene_count"]:
        warnings.append("One or more backend manifests report fallback or placeholder outputs.")
    if counters["stale_backend_manifest_scene_count"]:
        warnings.append(f"{counters['stale_backend_manifest_scene_count']} scene(s) have stale backend manifests/outputs.")
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
    weak_realism = [row for row in realism_rows if safe_float(row.get("realism_score", 1.0), 1.0) < 0.72]
    if weak_realism:
        warnings.append(f"{len(weak_realism)} scene(s) have weak realism scores below 72%.")
    average_realism = round(
        sum(safe_float(row.get("realism_score", 0.0), 0.0) for row in realism_rows) / max(1, len(realism_rows)),
        4,
    )
    technical_metrics = [
        technical_metric(
            "video_duration_score",
            "unavailable",
            0.0,
            "external duration probe not configured",
            {"scene_count": len(rows)},
        ),
        technical_metric(
            "audio_duration_score",
            "unavailable",
            0.0,
            "external duration probe not configured",
            {"scene_count": len(rows)},
        ),
        technical_metric(
            "voice_reference_quality_score",
            "measured",
            1.0 - safe_ratio(counters["missing_reference_data_line_count"], max(1, total_lines)),
            "reference candidate coverage heuristic",
        ),
        technical_metric(
            "voice_output_quality_score",
            "measured",
            1.0 - safe_ratio(counters["missing_voice_clone_output_scene_count"], max(1, len(rows))),
            "voice output presence heuristic",
        ),
        technical_metric("lipsync_confidence_score", "unavailable", 0.0, "lip-sync backend did not return a confidence metric"),
        technical_metric("asr_transcript_match_score", "unavailable", 0.0, "post-render ASR verification not configured"),
        technical_metric(
            "identity_consistency_score",
            "measured",
            1.0 - safe_ratio(counters["missing_set_context_scene_count"], max(1, len(rows))),
            "continuity metadata coverage heuristic",
        ),
        technical_metric(
            "shot_continuity_score",
            "measured",
            1.0 - safe_ratio(counters["missing_shot_plan_scene_count"], max(1, len(rows))),
            "shot plan coverage heuristic",
        ),
        technical_metric(
            "set_consistency_score",
            "measured",
            1.0 - safe_ratio(counters["missing_set_context_scene_count"], max(1, len(rows))),
            "set bible coverage heuristic",
        ),
        technical_metric(
            "audio_mix_score",
            "measured",
            1.0 - safe_ratio(counters["missing_audio_mix_scene_count"], max(1, len(rows))),
            "audio mix output coverage heuristic",
        ),
        technical_metric(
            "edit_completion_score",
            "measured",
            1.0 if package_payload.get("edit_decision_list") else 0.0,
            "edit decision list presence",
        ),
    ]
    return {
        "scene_count": len(rows),
        "scene_rows": rows,
        "realism_rows": realism_rows,
        "average_realism_score": average_realism,
        "warnings": warnings,
        **counters,
        "generic_template_line_ratio": safe_ratio(total_template_lines, total_lines),
        "finished_episode_mode_enabled": bool(finished_cfg.get("enabled", False)),
        "technical_metrics": technical_metrics,
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


def build_finished_episode_gate(
    artifacts: dict[str, Any],
    content_checks: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    mode = finished_episode_mode_config(cfg)
    if not bool(mode.get("enabled", False)):
        return {
            "passed": True,
            "readiness": "disabled",
            "blockers": [],
            "warnings": ["finished_episode_mode is disabled"],
            "required_actions": [],
        }
    blockers: list[str] = []
    warnings: list[str] = []
    required_actions: list[str] = []

    average_realism = safe_float(content_checks.get("average_realism_score", 0.0), 0.0)
    min_episode_realism = safe_float(mode.get("min_episode_realism_score", 0.82), 0.82)
    min_scene_realism = safe_float(mode.get("min_scene_realism_score", 0.72), 0.72)
    weak_scenes = [
        row
        for row in content_checks.get("realism_rows", [])
        if isinstance(row, dict) and safe_float(row.get("realism_score", 0.0), 0.0) < min_scene_realism
    ]
    if average_realism < min_episode_realism:
        blockers.append(f"episode realism score {average_realism:.2f} is below {min_episode_realism:.2f}")
        required_actions.append("Regenerate weak scenes until the average realism score reaches the finished-episode threshold.")
    if weak_scenes:
        blockers.append(f"{len(weak_scenes)} scene(s) are below the finished-episode scene realism threshold")
        required_actions.append("Use the regeneration queue scopes to repair the listed weak scenes.")

    if bool(mode.get("require_real_motion_video", True)) and int(content_checks.get("missing_real_motion_video_scene_count", 0) or 0):
        blockers.append("one or more scenes have no real generated motion/lip-sync video")
        required_actions.append("Run the configured video/lip-sync backends; still-frame motion is only preview quality.")
    if bool(mode.get("require_voice_clone_audio", True)) and int(content_checks.get("missing_voice_clone_output_scene_count", 0) or 0):
        blockers.append("one or more scenes are missing cloned dialogue audio")
        required_actions.append("Rerun voice cloning and collect missing voice references where needed.")
    if bool(mode.get("require_lipsync_video", True)) and int(content_checks.get("missing_lipsync_output_scene_count", 0) or 0):
        blockers.append("one or more scenes are missing lip-sync output")
        required_actions.append("Rerun the preferred lip-sync backend for the affected scenes.")
    if bool(mode.get("require_audio_mix", True)) and int(content_checks.get("missing_audio_mix_scene_count", 0) or 0):
        blockers.append("one or more scenes are missing a final audio mix")
        required_actions.append("Create dialogue, ambience, music/SFX stems and a normalized final mix.")
    if bool(mode.get("require_scene_continuity", True)):
        if int(content_checks.get("missing_shot_plan_scene_count", 0) or 0):
            blockers.append("one or more scenes are missing shot plans")
        if int(content_checks.get("missing_set_context_scene_count", 0) or 0):
            blockers.append("one or more scenes are missing set continuity context")
        if int(content_checks.get("missing_scene_function_count", 0) or 0):
            blockers.append("one or more scenes have no episode-arc function")
        if any("shot plans" in blocker or "set continuity" in blocker or "episode-arc" in blocker for blocker in blockers):
            required_actions.append("Regenerate story metadata so every scene has a function, set, shot plan, and continuity lock.")
    if bool(mode.get("require_backend_manifests", True)) and int(content_checks.get("missing_backend_manifest_scene_count", 0) or 0):
        blockers.append("required backend manifests are missing")
        required_actions.append("Rerun backend tasks so every real output has a machine-readable manifest.")
    if bool(mode.get("block_placeholder_outputs", True)):
        if int(content_checks.get("fallback_backend_manifest_scene_count", 0) or 0):
            blockers.append("backend manifests report fallback outputs")
        if int(content_checks.get("placeholder_backend_manifest_scene_count", 0) or 0):
            blockers.append("backend manifests report placeholder outputs")
        if any("fallback" in blocker or "placeholder" in blocker for blocker in blockers):
            required_actions.append("Replace fallback/placeholder outputs with real backend-generated media.")
    if bool(mode.get("block_stale_outputs", True)) and int(content_checks.get("stale_backend_manifest_scene_count", 0) or 0):
        blockers.append("backend manifests report stale outputs")
        required_actions.append("Force rerender stale backend outputs.")

    if not artifact_path_exists(artifacts.get("full_generated_episode", "")):
        blockers.append("full generated episode master is missing")
        required_actions.append("Run the finished episode master backend.")
    if not artifact_path_exists(artifacts.get("delivery_episode", "")):
        warnings.append("delivery watch file is missing; export will be review-only until delivery is written")

    blockers = list(dict.fromkeys(blockers))
    required_actions = list(dict.fromkeys(required_actions))
    return {
        "passed": not blockers,
        "readiness": "release_ready" if not blockers else "blocked",
        "blockers": blockers,
        "warnings": warnings,
        "required_actions": required_actions,
        "min_episode_realism_score": min_episode_realism,
        "min_scene_realism_score": min_scene_realism,
        "average_realism_score": average_realism,
        "weak_scene_count": len(weak_scenes),
    }


def quality_reports_root(cfg: dict[str, Any]) -> Path:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    raw_path = str(paths.get("quality_reports", "generation/quality_reports") or "generation/quality_reports").strip()
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else resolve_project_path(raw_path)


def build_realism_report_payload(
    artifacts: dict[str, Any],
    content_checks: dict[str, Any],
    finished_gate: dict[str, Any],
) -> dict[str, Any]:
    component_totals: dict[str, list[float]] = {}
    for row in content_checks.get("realism_rows", []) if isinstance(content_checks.get("realism_rows", []), list) else []:
        if not isinstance(row, dict):
            continue
        scores = row.get("component_scores", {}) if isinstance(row.get("component_scores", {}), dict) else {}
        for key, value in scores.items():
            component_totals.setdefault(key, []).append(clamp_score(value))

    def average_component(*keys: str) -> float:
        values: list[float] = []
        for key in keys:
            values.extend(component_totals.get(key, []))
        return round(sum(values) / max(1, len(values)), 4)

    weakest = sorted(
        [row for row in content_checks.get("realism_rows", []) if isinstance(row, dict)],
        key=lambda row: safe_float(row.get("realism_score", 0.0), 0.0),
    )[:8]
    return {
        "episode_id": artifacts.get("episode_id", ""),
        "display_title": artifacts.get("display_title", ""),
        "finished_episode_readiness": finished_gate.get("readiness", "unknown"),
        "realism_score": content_checks.get("average_realism_score", 0.0),
        "scores": {
            "story_score": average_component("scene_structure_score", "behavior_score"),
            "dialogue_score": average_component("dialogue_style_score"),
            "character_consistency_score": average_component("reference_coverage_score", "set_consistency_score"),
            "voice_score": average_component("voice_metadata_score"),
            "lip_sync_score": average_component("lipsync_backend_score"),
            "video_motion_score": average_component("video_motion_score"),
            "shot_editing_score": average_component("shot_editing_score"),
            "audio_mix_score": average_component("audio_mix_score"),
            "set_continuity_score": average_component("set_consistency_score"),
            "backend_integrity_score": average_component("backend_integrity_score"),
        },
        "main_blockers": list(finished_gate.get("blockers", []) or []),
        "required_actions": list(finished_gate.get("required_actions", []) or []),
        "weakest_scenes": weakest,
        "technical_metrics": content_checks.get("technical_metrics", []) if isinstance(content_checks.get("technical_metrics", []), list) else [],
        "created_at": utc_now_iso(),
    }


def realism_report_markdown(payload: dict[str, Any]) -> str:
    scores = payload.get("scores", {}) if isinstance(payload.get("scores", {}), dict) else {}
    lines = [
        f"# Realism Report - {payload.get('episode_id') or 'episode'}",
        "",
        "## Overall",
        f"- Finished episode readiness: {payload.get('finished_episode_readiness', 'unknown')}",
        f"- Realism score: {int(round(safe_float(payload.get('realism_score', 0.0), 0.0) * 100))}%",
        "- Main blockers:",
    ]
    blockers = payload.get("main_blockers", []) if isinstance(payload.get("main_blockers", []), list) else []
    lines.extend([f"  - {blocker}" for blocker in blockers] or ["  - none"])
    lines.extend(["", "## Scores"])
    for key, value in scores.items():
        lines.append(f"- {key}: {int(round(safe_float(value, 0.0) * 100))}%")
    lines.extend(["", "## Required Fixes"])
    actions = payload.get("required_actions", []) if isinstance(payload.get("required_actions", []), list) else []
    lines.extend([f"{index}. {action}" for index, action in enumerate(actions, start=1)] or ["1. No blocking actions remain."])
    lines.extend(["", "## Weakest Scenes"])
    weakest = payload.get("weakest_scenes", []) if isinstance(payload.get("weakest_scenes", []), list) else []
    for scene in weakest:
        if isinstance(scene, dict):
            reasons = ", ".join(str(item) for item in scene.get("failed_reasons", []) if str(item).strip())
            lines.append(f"- {scene.get('scene_id', 'scene')}: {int(round(safe_float(scene.get('realism_score', 0.0), 0.0) * 100))}% - {reasons or 'no details'}")
    if not weakest:
        lines.append("- none")
    return "\n".join(lines).rstrip() + "\n"


def write_realism_reports(
    cfg: dict[str, Any],
    artifacts: dict[str, Any],
    content_checks: dict[str, Any],
    finished_gate: dict[str, Any],
) -> dict[str, str]:
    episode_id = str(artifacts.get("episode_id", "") or "episode").strip() or "episode"
    root = quality_reports_root(cfg) if isinstance(cfg.get("paths", {}), dict) and cfg.get("paths", {}) else quality_gate_report_path(artifacts).parent
    root.mkdir(parents=True, exist_ok=True)
    payload = build_realism_report_payload(artifacts, content_checks, finished_gate)
    json_path = root / f"{episode_id}_realism_report.json"
    md_path = root / f"{episode_id}_realism_report.md"
    write_json(json_path, payload)
    md_path.write_text(realism_report_markdown(payload), encoding="utf-8")
    return {"realism_report_json": str(json_path), "realism_report_markdown": str(md_path)}


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
        "finished_episode_gate": report.get("finished_episode_gate", {}) if isinstance(report.get("finished_episode_gate", {}), dict) else {},
        "realism_report_json": report.get("realism_report_json", ""),
        "realism_report_markdown": report.get("realism_report_markdown", ""),
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
    production_package_payload = load_production_package(artifacts)
    content_checks = scene_content_quality_checks(production_package_payload, effective_cfg)
    scene_quality_rows = merge_realism_into_scene_quality_rows(scene_quality_rows, content_checks)
    release_result = release_quality_gate(artifacts, effective_cfg)
    finished_gate = build_finished_episode_gate(artifacts, content_checks, effective_cfg)
    if finished_episode_mode_enabled(effective_cfg) and not bool(finished_gate.get("passed", False)):
        release_result = dict(release_result)
        release_result["passed"] = False
        release_result["release_ready"] = False
        release_result["finished_episode_gate_passed"] = False
    else:
        release_result = dict(release_result)
        release_result["finished_episode_gate_passed"] = bool(finished_gate.get("passed", False))
    realism_report_paths = write_realism_reports(effective_cfg, artifacts, content_checks, finished_gate)
    release_cfg = release_mode_config(effective_cfg)
    regeneration_queue = queue_scenes_for_regeneration(
        scene_quality_rows,
        watch_threshold=float(release_cfg.get("watch_threshold", 0.52) or 0.52),
        release_threshold=float(release_cfg.get("min_episode_quality", 0.68) or 0.68),
        max_regeneration_batch=int(release_cfg.get("max_regeneration_batch", 8) or 8),
        max_regeneration_retries=int(release_cfg.get("max_regeneration_retries", 3) or 3),
    )
    warnings = build_warnings(artifacts, content_checks)
    warnings.extend(f"Finished episode blocker: {blocker}" for blocker in finished_gate.get("blockers", []) if str(blocker).strip())
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
        "finished_episode_gate": finished_gate,
        "warnings": warnings,
        "content_quality_checks": content_checks,
        "strict_fail": strict_fail,
        "max_regeneration_retries": int(release_cfg.get("max_regeneration_retries", 3) or 3),
        "regeneration_queue": regeneration_queue,
        **realism_report_paths,
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
    print(
        f"Finished episode gate: {'PASS' if finished_gate.get('passed') else 'FAIL'} | "
        f"readiness={finished_gate.get('readiness', 'unknown')} | "
        f"realism={int(round(safe_float(finished_gate.get('average_realism_score', 0.0), 0.0) * 100))}%"
    )
    print(f"Regeneration queue size: {len(regeneration_queue)}")
    print(f"Report: {report_path}")
    print(f"Realism report: {realism_report_paths.get('realism_report_markdown', '-')}")

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
            if regeneration_queue_has_blocked_scope(current_queue):
                info("Auto-retry stopped because the regeneration queue contains blocked causes that require references or backend setup.")
                break
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
            if regeneration_queue_has_blocked_scope(current_queue):
                info("Auto-retry stopped after refresh because remaining causes are blocked, not retryable render tasks.")
                break
            if not retry_forever:
                break
            if refreshed_artifacts:
                info("Auto-retry finished a cycle, but the refreshed episode still does not pass the release gate.")

    raise SystemExit(1)


if __name__ == "__main__":
    main()

