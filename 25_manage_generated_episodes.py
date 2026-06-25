#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR / "ai_series_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from support_scripts.pipeline_common import (  # noqa: E402
    generated_episode_artifacts,
    headline,
    info,
    list_generated_episode_artifacts,
    load_config,
    ok,
    read_json,
    resolve_project_path,
    resolve_stored_project_path,
    warn,
    write_json,
)


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_stdio()


EPISODE_REPORT_SUFFIXES = (
    "_quality_gate.json",
    "_realism_report.json",
    "_regeneration_queue.json",
)

OPEN_TARGETS = {
    "best-video",
    "folder",
    "delivery",
    "package",
    "quality",
    "realism",
    "shotlist",
    "render-manifest",
}

IGNORED_EPISODE_NAMES = {"latest", "archive", "generated_episodes"}
LIVE_STALE_SECONDS = 20 * 60
WATCH_DEFAULT_SECONDS = 5.0


@dataclass
class EpisodeRecord:
    episode_id: str
    display_title: str
    readiness: str = ""
    quality_percent: float = 0.0
    minimum_scene_quality_percent: float = 0.0
    release_gate_passed: bool = False
    finished_gate_passed: bool = False
    realism_score: float = 0.0
    scene_count: int = 0
    generated_scene_video_count: int = 0
    scene_dialogue_audio_count: int = 0
    scene_master_clip_count: int = 0
    backend_runner_status: str = ""
    backend_runner_failed_count: int = 0
    regeneration_queue_count: int = 0
    updated_at: float = 0.0
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    paths: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "display_title": self.display_title,
            "readiness": self.readiness,
            "quality_percent": self.quality_percent,
            "minimum_scene_quality_percent": self.minimum_scene_quality_percent,
            "release_gate_passed": self.release_gate_passed,
            "finished_gate_passed": self.finished_gate_passed,
            "realism_score": self.realism_score,
            "scene_count": self.scene_count,
            "generated_scene_video_count": self.generated_scene_video_count,
            "scene_dialogue_audio_count": self.scene_dialogue_audio_count,
            "scene_master_clip_count": self.scene_master_clip_count,
            "backend_runner_status": self.backend_runner_status,
            "backend_runner_failed_count": self.backend_runner_failed_count,
            "regeneration_queue_count": self.regeneration_queue_count,
            "updated_at": self.updated_at,
            "updated_at_text": format_timestamp(self.updated_at),
            "warnings": list(self.warnings),
            "blockers": list(self.blockers),
            "paths": dict(self.paths),
        }


@dataclass
class LiveGenerationStatus:
    status: str
    active: bool
    stale: bool
    current_step: str = ""
    current_episode: str = ""
    current_file: str = ""
    updated_at: float = 0.0
    eta_text: str = "calculating"
    completed_steps: int = 0
    total_steps: int = 0
    active_workers: list[dict[str, Any]] = field(default_factory=list)
    stale_steps: list[dict[str, Any]] = field(default_factory=list)
    latest_episode: dict[str, Any] = field(default_factory=dict)
    current_status_path: str = ""
    current_status_markdown_path: str = ""
    messages: list[str] = field(default_factory=list)

    @property
    def progress_percent(self) -> int:
        if self.total_steps <= 0:
            return 0
        return int((100 * max(0, min(self.completed_steps, self.total_steps))) / self.total_steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "active": self.active,
            "stale": self.stale,
            "current_step": self.current_step,
            "current_episode": self.current_episode,
            "current_file": self.current_file,
            "updated_at": self.updated_at,
            "updated_at_text": format_timestamp(self.updated_at),
            "updated_age": format_age(time.time() - self.updated_at) if self.updated_at else "",
            "eta_text": self.eta_text,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "progress_percent": self.progress_percent,
            "active_workers": list(self.active_workers),
            "stale_steps": list(self.stale_steps),
            "latest_episode": dict(self.latest_episode),
            "current_status_path": self.current_status_path,
            "current_status_markdown_path": self.current_status_markdown_path,
            "messages": list(self.messages),
        }


def coalesce_text(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return default


def bool_from_gate(value: object) -> bool:
    if isinstance(value, dict):
        return bool(value.get("passed", False))
    return bool(value)


def format_timestamp(timestamp: float) -> str:
    if not timestamp:
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def parse_timestamp(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return 0.0
    if text.replace(".", "", 1).isdigit():
        return as_float(text)
    normalized = text.replace("Z", "+00:00")
    try:
        return time.mktime(time.strptime(text, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        pass
    try:
        from datetime import datetime

        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is not None:
            return parsed.timestamp()
        return time.mktime(parsed.timetuple())
    except ValueError:
        return 0.0


def format_age(seconds: float) -> str:
    seconds = max(0, int(seconds or 0))
    if seconds < 60:
        return f"{seconds}s"
    minutes, rem_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {rem_seconds:02d}s"
    hours, rem_minutes = divmod(minutes, 60)
    return f"{hours}h {rem_minutes:02d}m"


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "calculating"
    seconds_int = max(0, int(float(seconds or 0)))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, rem_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{rem_seconds:02d}"


def existing_path(path_value: object) -> Path | None:
    if not path_value:
        return None
    path = resolve_stored_project_path(str(path_value))
    return path if path.exists() else None


def latest_mtime(paths: list[Path]) -> float:
    timestamps: list[float] = []
    for path in paths:
        try:
            if path.exists():
                timestamps.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(timestamps) if timestamps else 0.0


def strip_known_episode_suffix(filename_stem: str) -> str:
    for suffix in ("_render_manifest", "_delivery_manifest", "_quality_gate", "_realism_report", "_regeneration_queue"):
        if filename_stem.endswith(suffix):
            return filename_stem[: -len(suffix)]
    return filename_stem


def discover_episode_ids(cfg: dict[str, Any]) -> list[str]:
    episode_ids: set[str] = set()

    for row in list_generated_episode_artifacts(cfg):
        episode_id = coalesce_text(row.get("episode_id") if isinstance(row, dict) else "")
        if episode_id and episode_id.lower() not in IGNORED_EPISODE_NAMES:
            episode_ids.add(episode_id)

    discovery_roots = [
        resolve_project_path("generation/final_episode_packages"),
        resolve_project_path("generation/renders/deliveries"),
        resolve_project_path("generation/storyboard_assets"),
        resolve_project_path("generation/final_episode_packages"),
    ]
    for root in discovery_roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                if child.name.lower() not in IGNORED_EPISODE_NAMES:
                    episode_ids.add(child.name)

    file_patterns = [
        (resolve_project_path("generation/shotlists"), "*.json"),
        (resolve_project_path("generation/renders/final"), "*_render_manifest.json"),
        (resolve_project_path("generation/quality_reports"), "*.json"),
        (resolve_project_path("generation/renders/deliveries"), "*/*_quality_gate.json"),
    ]
    for root, pattern in file_patterns:
        if not root.exists():
            continue
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            stem = strip_known_episode_suffix(path.stem)
            if stem and stem.lower() not in IGNORED_EPISODE_NAMES:
                episode_ids.add(stem)

    return sorted(episode_ids, key=episode_sort_key)


def episode_sort_key(episode_id: str) -> tuple[int, str]:
    digits = "".join(char for char in episode_id if char.isdigit())
    return (as_int(digits, 999999), episode_id.lower())


def load_quality_gate_for_episode(episode_id: str, artifacts: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    candidates = [
        artifacts.get("quality_gate_report") if isinstance(artifacts, dict) else "",
        resolve_project_path(f"generation/renders/deliveries/{episode_id}/{episode_id}_quality_gate.json"),
        resolve_project_path(f"generation/quality_reports/{episode_id}_quality_gate.json"),
    ]
    for candidate in candidates:
        path = existing_path(candidate) if not isinstance(candidate, Path) else candidate
        if path and path.exists():
            return read_json(path, {}), path
    return {}, None


def load_realism_report_path(episode_id: str) -> Path | None:
    candidates = [
        resolve_project_path(f"generation/quality_reports/{episode_id}_realism_report.md"),
        resolve_project_path(f"generation/quality_reports/{episode_id}_realism_report.json"),
        resolve_project_path(f"generation/renders/deliveries/{episode_id}/{episode_id}_realism_report.md"),
        resolve_project_path(f"generation/renders/deliveries/{episode_id}/{episode_id}_realism_report.json"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def collect_record_paths(episode_id: str, artifacts: dict[str, Any], quality_report_path: Path | None) -> dict[str, str]:
    keys = [
        "story_prompt",
        "shotlist",
        "render_manifest",
        "production_package",
        "production_package_root",
        "delivery_bundle_root",
        "delivery_manifest",
        "delivery_episode",
        "latest_delivery_root",
        "latest_delivery_manifest",
        "latest_delivery_episode",
        "draft_render",
        "final_render",
        "dialogue_audio",
        "voice_plan",
        "subtitle_preview",
        "full_generated_episode",
        "regeneration_queue_manifest",
    ]
    paths: dict[str, str] = {}
    for key in keys:
        value = artifacts.get(key) if isinstance(artifacts, dict) else ""
        if value:
            paths[key] = str(resolve_stored_project_path(str(value)))

    if quality_report_path:
        paths["quality_gate_report"] = str(quality_report_path)
    realism_path = load_realism_report_path(episode_id)
    if realism_path:
        paths["realism_report"] = str(realism_path)

    fallback_paths = {
        "shotlist": resolve_project_path(f"generation/shotlists/{episode_id}.json"),
        "production_package_root": resolve_project_path(f"generation/final_episode_packages/{episode_id}"),
        "delivery_bundle_root": resolve_project_path(f"generation/renders/deliveries/{episode_id}"),
        "final_render": resolve_project_path(f"generation/renders/final/{episode_id}.mp4"),
        "draft_render": resolve_project_path(f"generation/renders/drafts/{episode_id}.mp4"),
        "render_manifest": resolve_project_path(f"generation/renders/final/{episode_id}_render_manifest.json"),
    }
    for key, path in fallback_paths.items():
        if key not in paths and path.exists():
            paths[key] = str(path)
    return paths


def build_episode_record(cfg: dict[str, Any], episode_id: str) -> EpisodeRecord:
    artifacts = generated_episode_artifacts(cfg, episode_id)
    quality_gate, quality_report_path = load_quality_gate_for_episode(episode_id, artifacts)
    finished_gate = quality_gate.get("finished_episode_gate", {}) if isinstance(quality_gate, dict) else {}
    release_gate = (
        quality_gate.get("release_gate", {})
        if isinstance(quality_gate, dict) and isinstance(quality_gate.get("release_gate"), dict)
        else artifacts.get("release_gate", {})
    )
    paths = collect_record_paths(episode_id, artifacts, quality_report_path)
    existing_paths = [path for path in (existing_path(value) for value in paths.values()) if path is not None]

    blockers = []
    warnings = []
    if isinstance(finished_gate, dict):
        blockers.extend(str(item) for item in finished_gate.get("blockers", []) or [])
        warnings.extend(str(item) for item in finished_gate.get("warnings", []) or [])
    warnings.extend(str(item) for item in artifacts.get("quality_gate_warnings", []) or [])

    quality_percent = as_float(
        quality_gate.get("quality_percent") if isinstance(quality_gate, dict) else 0.0,
        as_float(artifacts.get("quality_percent")),
    )
    if quality_percent <= 0:
        quality_percent = as_float(artifacts.get("quality_percent"))
    minimum_quality_percent = as_float(
        quality_gate.get("minimum_scene_quality_percent") if isinstance(quality_gate, dict) else 0.0,
        as_float(artifacts.get("minimum_scene_quality_percent")),
    )
    if minimum_quality_percent <= 0:
        minimum_quality_percent = as_float(artifacts.get("minimum_scene_quality_percent"))

    realism_score = as_float(
        finished_gate.get("realism_score") if isinstance(finished_gate, dict) else 0.0,
        as_float(quality_gate.get("episode_realism_score") if isinstance(quality_gate, dict) else 0.0),
    )
    if realism_score <= 0 and isinstance(quality_gate, dict):
        realism_score = as_float(quality_gate.get("realism_score"))

    return EpisodeRecord(
        episode_id=episode_id,
        display_title=coalesce_text(artifacts.get("display_title"), episode_id),
        readiness=coalesce_text(
            quality_gate.get("readiness") if isinstance(quality_gate, dict) else "",
            artifacts.get("production_readiness"),
        ),
        quality_percent=quality_percent,
        minimum_scene_quality_percent=minimum_quality_percent,
        release_gate_passed=bool_from_gate(release_gate) or bool(artifacts.get("release_gate_passed", False)),
        finished_gate_passed=bool_from_gate(finished_gate),
        realism_score=realism_score,
        scene_count=as_int(artifacts.get("scene_count")),
        generated_scene_video_count=as_int(artifacts.get("generated_scene_video_count")),
        scene_dialogue_audio_count=as_int(artifacts.get("scene_dialogue_audio_count")),
        scene_master_clip_count=as_int(artifacts.get("scene_master_clip_count")),
        backend_runner_status=coalesce_text(artifacts.get("backend_runner_status")),
        backend_runner_failed_count=as_int(artifacts.get("backend_runner_failed_count")),
        regeneration_queue_count=as_int(
            quality_gate.get("regeneration_queue_size") if isinstance(quality_gate, dict) else 0,
            as_int(artifacts.get("regeneration_queue_count")),
        ),
        updated_at=latest_mtime(existing_paths),
        warnings=dedupe_texts(warnings),
        blockers=dedupe_texts(blockers),
        paths=paths,
        artifacts=artifacts,
    )


def dedupe_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def list_episode_records(cfg: dict[str, Any]) -> list[EpisodeRecord]:
    return [build_episode_record(cfg, episode_id) for episode_id in discover_episode_ids(cfg)]


def current_pipeline_status_path() -> Path:
    return resolve_project_path("runtime/autosaves/24_process_next_episode/current_status.json")


def current_pipeline_status_markdown_path() -> Path:
    return resolve_project_path("runtime/autosaves/24_process_next_episode/current_status.md")


def step_autosave_root_path() -> Path:
    return resolve_project_path("runtime/autosaves/steps")


def distributed_runtime_root_path() -> Path:
    return resolve_project_path("runtime/distributed")


def load_current_pipeline_status() -> dict[str, Any]:
    path = current_pipeline_status_path()
    return read_json(path, {}) if path.exists() else {}


def extract_eta_text(status: dict[str, Any]) -> str:
    for key in ("overall_eta", "overall_eta_text", "current_eta", "eta", "eta_text"):
        text = coalesce_text(status.get(key))
        if text:
            return text
    for key in ("overall_eta_seconds", "current_eta_seconds", "eta_seconds", "remaining_seconds"):
        if key in status:
            return format_duration(as_float(status.get(key)))
    return "calculating"


def summarize_global_progress(status: dict[str, Any]) -> tuple[int, int]:
    progress = status.get("global_progress", []) if isinstance(status.get("global_progress"), list) else []
    if progress:
        total = len(progress)
        completed = sum(1 for row in progress if isinstance(row, dict) and str(row.get("status", "")).lower() == "completed")
        running = any(isinstance(row, dict) and str(row.get("status", "")).lower() == "running" for row in progress)
        return min(total, completed + (1 if running else 0)), total
    planned = status.get("global_planned_steps", []) if isinstance(status.get("global_planned_steps"), list) else []
    completed_labels = status.get("global_completed_step_labels", []) if isinstance(status.get("global_completed_step_labels"), list) else []
    return len(completed_labels), len(planned)


def load_step_autosaves() -> list[dict[str, Any]]:
    root = step_autosave_root_path()
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in root.rglob("*.json"):
        payload = read_json(path, {})
        if not isinstance(payload, dict):
            continue
        updated_at = parse_timestamp(payload.get("updated_at")) or path.stat().st_mtime
        rows.append(
            {
                "path": str(path),
                "step": coalesce_text(payload.get("step"), path.parent.name),
                "target": coalesce_text(payload.get("target"), path.stem),
                "status": coalesce_text(payload.get("status"), "unknown"),
                "episode_id": coalesce_text(payload.get("episode_id")),
                "updated_at": updated_at,
                "updated_at_text": format_timestamp(updated_at),
                "age": format_age(time.time() - updated_at),
                "error": coalesce_text(payload.get("error")),
            }
        )
    rows.sort(key=lambda item: float(item.get("updated_at", 0.0) or 0.0), reverse=True)
    return rows


def load_worker_leases() -> list[dict[str, Any]]:
    root = distributed_runtime_root_path()
    rows: list[dict[str, Any]] = []
    now = time.time()
    if not root.exists():
        return rows
    for path in root.rglob("*.json"):
        payload = read_json(path, {})
        if not isinstance(payload, dict) or "owner_id" not in payload:
            continue
        meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
        heartbeat_at = parse_timestamp(payload.get("heartbeat_at"))
        expires_at = parse_timestamp(payload.get("expires_at"))
        active = bool(expires_at and expires_at > now)
        rows.append(
            {
                "path": str(path),
                "owner_id": coalesce_text(payload.get("owner_id")),
                "worker_id": coalesce_text(meta.get("worker_id"), payload.get("owner_id")),
                "hostname": coalesce_text(meta.get("hostname")),
                "pid": coalesce_text(meta.get("pid")),
                "step": coalesce_text(meta.get("step"), path.parts[-3] if len(path.parts) >= 3 else ""),
                "scope": coalesce_text(meta.get("scope"), meta.get("episode_id"), path.stem),
                "episode_id": coalesce_text(meta.get("episode_id")),
                "has_gpu": bool(meta.get("has_gpu", False)),
                "heartbeat_at": heartbeat_at,
                "heartbeat_age": format_age(now - heartbeat_at) if heartbeat_at else "",
                "expires_at": expires_at,
                "expires_in": format_duration(expires_at - now) if expires_at else "",
                "active": active,
            }
        )
    rows.sort(key=lambda item: (not bool(item.get("active")), -(float(item.get("heartbeat_at", 0.0) or 0.0))))
    return rows


def build_live_generation_status(cfg: dict[str, Any]) -> LiveGenerationStatus:
    del cfg
    status_payload = load_current_pipeline_status()
    now = time.time()
    status_updated = parse_timestamp(status_payload.get("updated_at")) if isinstance(status_payload, dict) else 0.0
    completed, total = summarize_global_progress(status_payload if isinstance(status_payload, dict) else {})
    workers = load_worker_leases()
    active_workers = [row for row in workers if bool(row.get("active"))]
    step_rows = load_step_autosaves()
    stale_steps = [
        row
        for row in step_rows
        if str(row.get("status", "")).lower() == "in_progress"
        and float(row.get("updated_at", 0.0) or 0.0) > 0
        and now - float(row.get("updated_at", 0.0) or 0.0) > LIVE_STALE_SECONDS
    ]
    status_age = (now - status_updated) if status_updated else 0.0
    status_running = str(status_payload.get("status", "")).lower() == "running" if isinstance(status_payload, dict) else False
    active = bool(active_workers) or bool(status_running and status_updated and status_age <= LIVE_STALE_SECONDS)
    stale = bool(status_running and status_updated and status_age > LIVE_STALE_SECONDS and not active_workers)
    latest_episode = status_payload.get("latest_generated_episode", {}) if isinstance(status_payload.get("latest_generated_episode"), dict) else {}

    messages: list[str] = []
    if active_workers:
        messages.append(f"{len(active_workers)} active distributed worker lease(s).")
    if stale:
        messages.append(f"Current 24_process_next_episode autosave is stale ({format_age(status_age)} old).")
    if stale_steps:
        messages.append(f"{len(stale_steps)} in-progress step autosave(s) look stale.")
    if not active and not stale:
        messages.append("No active generation worker was detected.")

    return LiveGenerationStatus(
        status="active" if active else "stale" if stale else "idle",
        active=active,
        stale=stale,
        current_step=coalesce_text(status_payload.get("current_step") if isinstance(status_payload, dict) else ""),
        current_episode=coalesce_text(
            latest_episode.get("episode_id") if isinstance(latest_episode, dict) else "",
            status_payload.get("current_episode_name") if isinstance(status_payload, dict) else "",
        ),
        current_file=coalesce_text(status_payload.get("current_episode_file") if isinstance(status_payload, dict) else ""),
        updated_at=status_updated,
        eta_text=extract_eta_text(status_payload if isinstance(status_payload, dict) else {}),
        completed_steps=completed,
        total_steps=total,
        active_workers=active_workers,
        stale_steps=stale_steps[:8],
        latest_episode=latest_episode if isinstance(latest_episode, dict) else {},
        current_status_path=str(current_pipeline_status_path()) if current_pipeline_status_path().exists() else "",
        current_status_markdown_path=str(current_pipeline_status_markdown_path()) if current_pipeline_status_markdown_path().exists() else "",
        messages=messages,
    )


def dashboard_bar(current: float, total: float, width: int = 26) -> str:
    total_value = max(1.0, float(total or 1.0))
    current_value = max(0.0, min(float(current or 0.0), total_value))
    filled = int(round((current_value / total_value) * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def format_live_status(status: LiveGenerationStatus) -> str:
    lines = [
        "Live Generation",
        "-" * 72,
        f"Status       : {status.status.upper()}",
        f"Current step : {status.current_step or '-'}",
        f"Episode      : {status.current_episode or '-'}",
        f"Source file  : {status.current_file or '-'}",
        f"Updated      : {format_timestamp(status.updated_at) or '-'} ({format_age(time.time() - status.updated_at) if status.updated_at else 'no status file'})",
        f"ETA          : {status.eta_text or 'calculating'}",
    ]
    if status.total_steps:
        lines.append(
            f"Pipeline     : {dashboard_bar(status.completed_steps, status.total_steps)} "
            f"{status.completed_steps}/{status.total_steps} ({status.progress_percent}%)"
        )
    latest = status.latest_episode
    if latest:
        lines.extend(
            [
                "",
                "Latest generated episode:",
                f"- id/readiness : {coalesce_text(latest.get('episode_id')) or '-'} / {coalesce_text(latest.get('production_readiness')) or '-'}",
                f"- scenes       : {as_int(latest.get('scene_count'))} | videos {as_int(latest.get('generated_scene_video_count'))} | voices {as_int(latest.get('scene_dialogue_audio_count'))} | masters {as_int(latest.get('scene_master_clip_count'))}",
                f"- backend      : {coalesce_text(latest.get('backend_runner_status')) or '-'} | failed {as_int(latest.get('backend_runner_failed_count'))}",
            ]
        )
    if status.active_workers:
        lines.extend(["", "Active workers:"])
        for worker in status.active_workers[:12]:
            gpu = "gpu" if worker.get("has_gpu") else "cpu"
            lines.append(
                f"- {worker.get('worker_id') or worker.get('owner_id')} | {worker.get('step') or '-'} "
                f"| {worker.get('scope') or '-'} | {gpu} | heartbeat {worker.get('heartbeat_age') or '-'} | expires {worker.get('expires_in') or '-'}"
            )
    if status.stale_steps:
        lines.extend(["", "Stale in-progress step autosaves:"])
        for row in status.stale_steps:
            lines.append(f"- {row.get('step')} / {row.get('target')} | updated {row.get('age')} ago")
    if status.messages:
        lines.extend(["", "Notes:"])
        lines.extend(f"- {message}" for message in status.messages)
    if status.current_status_path:
        lines.extend(["", f"Status JSON  : {status.current_status_path}"])
    if status.current_status_markdown_path:
        lines.append(f"Status MD    : {status.current_status_markdown_path}")
    return "\n".join(lines)


def format_percent(value: float) -> str:
    if value <= 1.0 and value > 0:
        value *= 100.0
    return f"{value:.0f}%"


def format_episode_table(records: list[EpisodeRecord]) -> str:
    headers = ["Episode", "Title", "Ready", "Q", "Rel", "Fin", "Scenes", "Updated"]
    rows: list[list[str]] = []
    for row in records:
        rows.append(
            [
                row.episode_id,
                row.display_title[:36],
                row.readiness or "-",
                format_percent(row.quality_percent),
                "OK" if row.release_gate_passed else "FAIL",
                "OK" if row.finished_gate_passed else "FAIL",
                str(row.scene_count or "-"),
                format_timestamp(row.updated_at) or "-",
            ]
        )
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    lines = ["  ".join(header.ljust(width) for header, width in zip(headers, widths))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend("  ".join(value.ljust(width) for value, width in zip(row, widths)) for row in rows)
    return "\n".join(lines)


def best_video_path(record: EpisodeRecord) -> Path | None:
    for key in ("latest_delivery_episode", "delivery_episode", "full_generated_episode", "final_render", "draft_render"):
        path = existing_path(record.paths.get(key, ""))
        if path and path.is_file():
            return path
    return None


def target_path_for_record(record: EpisodeRecord, target: str) -> Path | None:
    if target == "best-video":
        return best_video_path(record)
    if target == "folder":
        for key in ("latest_delivery_root", "delivery_bundle_root", "production_package_root"):
            path = existing_path(record.paths.get(key, ""))
            if path:
                return path
        video = best_video_path(record)
        return video.parent if video else None
    target_key = {
        "delivery": "latest_delivery_root",
        "package": "production_package_root",
        "quality": "quality_gate_report",
        "realism": "realism_report",
        "shotlist": "shotlist",
        "render-manifest": "render_manifest",
    }.get(target, "")
    if target_key:
        return existing_path(record.paths.get(target_key, ""))
    return None


def open_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ensure_inside_generation(path: Path, generation_root: Path) -> Path:
    resolved = path.resolve()
    root = generation_root.resolve()
    try:
        return resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to archive a path outside generation/: {path}") from exc


def archive_candidates_for_episode(record: EpisodeRecord) -> list[Path]:
    episode_id = record.episode_id
    candidates: list[Path] = []
    for key in (
        "production_package_root",
        "delivery_bundle_root",
        "latest_delivery_root",
        "story_prompt",
        "shotlist",
        "render_manifest",
        "draft_render",
        "final_render",
        "dialogue_audio",
        "voice_plan",
        "subtitle_preview",
        "full_generated_episode",
        "quality_gate_report",
        "realism_report",
        "regeneration_queue_manifest",
    ):
        path = existing_path(record.paths.get(key, ""))
        if path:
            candidates.append(path)

    glob_patterns = [
        f"renders/final/{episode_id}*",
        f"renders/drafts/{episode_id}*",
        f"renders/tmp/{episode_id}",
        f"storyboard_assets/{episode_id}",
        f"storyboard_requests/{episode_id}*",
        f"scene_packages/{episode_id}",
        f"final_episode_packages/{episode_id}",
        f"quality_reports/{episode_id}*",
    ]
    generation_root = resolve_project_path("generation")
    for pattern in glob_patterns:
        candidates.extend(generation_root.glob(pattern))

    unique: dict[str, Path] = {}
    for path in candidates:
        if not path.exists():
            continue
        try:
            rel = ensure_inside_generation(path, generation_root)
        except RuntimeError:
            continue
        if rel.parts and rel.parts[0] == "archive":
            continue
        unique[str(path.resolve()).lower()] = path
    return prune_nested_archive_candidates(sorted(unique.values(), key=lambda item: (len(item.parts), str(item).lower())))


def prune_nested_archive_candidates(candidates: list[Path]) -> list[Path]:
    kept: list[Path] = []
    resolved_kept: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if any(is_path_relative_to(resolved, parent) for parent in resolved_kept):
            continue
        kept.append(path)
        resolved_kept.append(resolved)
    return kept


def is_path_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return path != parent
    except ValueError:
        return False


def archive_episode(record: EpisodeRecord, *, dry_run: bool = False) -> dict[str, Any]:
    generation_root = resolve_project_path("generation")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_root = generation_root / "archive" / "generated_episodes" / f"{timestamp}_{record.episode_id}"
    candidates = archive_candidates_for_episode(record)
    moved: list[dict[str, str]] = []

    for source in candidates:
        relative = ensure_inside_generation(source, generation_root)
        target = archive_root / relative
        moved.append({"source": str(source), "target": str(target)})
        if dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError(f"Archive target already exists: {target}")
        shutil.move(str(source), str(target))

    manifest = {
        "episode_id": record.episode_id,
        "display_title": record.display_title,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": dry_run,
        "archive_root": str(archive_root),
        "moved": moved,
    }
    if not dry_run:
        write_json(archive_root / "archive_manifest.json", manifest)
    return manifest


def delete_episode_outputs(record: EpisodeRecord, *, dry_run: bool = False) -> dict[str, Any]:
    candidates = archive_candidates_for_episode(record)
    deleted: list[str] = []
    for source in candidates:
        ensure_inside_generation(source, resolve_project_path("generation"))
        deleted.append(str(source))
        if dry_run:
            continue
        if source.is_dir():
            shutil.rmtree(source)
        else:
            source.unlink()
    return {
        "episode_id": record.episode_id,
        "display_title": record.display_title,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": dry_run,
        "deleted": deleted,
    }


def print_episode_details(record: EpisodeRecord) -> None:
    print(json.dumps(record.to_dict(), indent=2, ensure_ascii=False))


def clear_console() -> None:
    command = "cls" if os.name == "nt" else "clear"
    try:
        os.system(command)
    except Exception:
        print("\n" * 4)


def print_live_status(cfg: dict[str, Any], *, json_output: bool = False) -> None:
    status = build_live_generation_status(cfg)
    if json_output:
        print(json.dumps(status.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_live_status(status))


def watch_live_status(cfg: dict[str, Any], interval_seconds: float, *, json_output: bool = False) -> None:
    while True:
        clear_console()
        headline("Generated Episode Live Monitor")
        print_live_status(cfg, json_output=json_output)
        print()
        info(f"Live update every {interval_seconds:.1f}s. Press Ctrl+C to stop.")
        time.sleep(max(1.0, interval_seconds))


def select_record(records: list[EpisodeRecord], episode_id: str) -> EpisodeRecord:
    for record in records:
        if record.episode_id == episode_id:
            return record
    raise RuntimeError(f"Generated episode was not found: {episode_id}")


def run_gui(cfg: dict[str, Any]) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox, simpledialog, ttk
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"Tk GUI is not available: {exc}") from exc

    root = tk.Tk()
    root.title("Generated Episode Manager")
    root.geometry("1160x720")

    records: list[EpisodeRecord] = []
    checked_episode_ids: set[str] = set()
    selected_episode = tk.StringVar(value="")
    status_text = tk.StringVar(value="")

    live_frame = ttk.LabelFrame(root, text="Live generation")
    live_detail = tk.Text(live_frame, height=9, wrap="word")
    live_detail.configure(state="disabled")

    columns = ("checked", "episode", "title", "ready", "quality", "release", "finished", "scenes", "updated")
    tree = ttk.Treeview(root, columns=columns, show="headings", height=16)
    labels = {
        "checked": "Select",
        "episode": "Episode",
        "title": "Title",
        "ready": "Readiness",
        "quality": "Quality",
        "release": "Release",
        "finished": "Finished",
        "scenes": "Scenes",
        "updated": "Updated",
    }
    widths = {
        "checked": 62,
        "episode": 110,
        "title": 290,
        "ready": 150,
        "quality": 70,
        "release": 70,
        "finished": 80,
        "scenes": 70,
        "updated": 150,
    }
    for column in columns:
        tree.heading(column, text=labels[column])
        tree.column(column, width=widths[column], anchor="w")

    detail = tk.Text(root, height=18, wrap="word")
    detail.configure(state="disabled")

    def set_live_detail(text: str) -> None:
        live_detail.configure(state="normal")
        live_detail.delete("1.0", "end")
        live_detail.insert("1.0", text)
        live_detail.configure(state="disabled")

    def refresh_live() -> None:
        set_live_detail(format_live_status(build_live_generation_status(cfg)))

    def current_record() -> EpisodeRecord | None:
        episode_id = selected_episode.get()
        for item in records:
            if item.episode_id == episode_id:
                return item
        return None

    def checked_records() -> list[EpisodeRecord]:
        return [row for row in records if row.episode_id in checked_episode_ids]

    def set_detail(text: str) -> None:
        detail.configure(state="normal")
        detail.delete("1.0", "end")
        detail.insert("1.0", text)
        detail.configure(state="disabled")

    def sync_episode_records() -> None:
        nonlocal records
        previous_selection = selected_episode.get()
        records = list_episode_records(cfg)
        valid_ids = {row.episode_id for row in records}
        checked_episode_ids.intersection_update(valid_ids)
        for item in tree.get_children():
            tree.delete(item)
        for row in records:
            tree.insert(
                "",
                "end",
                iid=row.episode_id,
                values=(
                    "[x]" if row.episode_id in checked_episode_ids else "[ ]",
                    row.episode_id,
                    row.display_title,
                    row.readiness or "-",
                    format_percent(row.quality_percent),
                    "OK" if row.release_gate_passed else "FAIL",
                    "OK" if row.finished_gate_passed else "FAIL",
                    row.scene_count or "-",
                    format_timestamp(row.updated_at) or "-",
                ),
            )
        status_text.set(
            f"Loaded {len(records)} generated episode(s), {len(checked_episode_ids)} checked. "
            f"Last refresh: {format_timestamp(time.time())}"
        )
        if records:
            selected = previous_selection if any(row.episode_id == previous_selection for row in records) else records[0].episode_id
            tree.selection_set(selected)
            tree.focus(selected)
            selected_episode.set(selected)
            update_detail()
        else:
            selected_episode.set("")
            set_detail("No generated episodes found.")

    def refresh_all() -> None:
        refresh_live()
        sync_episode_records()

    def redraw_checked_column() -> None:
        for row in records:
            if tree.exists(row.episode_id):
                values = list(tree.item(row.episode_id, "values"))
                if values:
                    values[0] = "[x]" if row.episode_id in checked_episode_ids else "[ ]"
                    tree.item(row.episode_id, values=values)
        status_text.set(
            f"Loaded {len(records)} generated episode(s), {len(checked_episode_ids)} checked. "
            f"Last refresh: {format_timestamp(time.time())}"
        )

    def toggle_checked(episode_id: str) -> None:
        if episode_id in checked_episode_ids:
            checked_episode_ids.remove(episode_id)
        else:
            checked_episode_ids.add(episode_id)
        redraw_checked_column()

    def select_all() -> None:
        checked_episode_ids.clear()
        checked_episode_ids.update(row.episode_id for row in records)
        redraw_checked_column()

    def clear_checks() -> None:
        checked_episode_ids.clear()
        redraw_checked_column()

    def on_tree_click(event: object) -> None:
        region = tree.identify("region", getattr(event, "x", 0), getattr(event, "y", 0))
        if region != "cell":
            return
        column = tree.identify_column(getattr(event, "x", 0))
        item_id = tree.identify_row(getattr(event, "y", 0))
        if column == "#1" and item_id:
            toggle_checked(str(item_id))

    def update_detail(_event: object | None = None) -> None:
        selection = tree.selection()
        if selection:
            selected_episode.set(str(selection[0]))
        row = current_record()
        if not row:
            set_detail("No episode selected.")
            return
        parts = [
            f"{row.episode_id} - {row.display_title}",
            "",
            f"Readiness: {row.readiness or '-'}",
            f"Quality: {format_percent(row.quality_percent)} | minimum scene: {format_percent(row.minimum_scene_quality_percent)}",
            f"Release gate: {'OK' if row.release_gate_passed else 'FAIL'}",
            f"Finished episode gate: {'OK' if row.finished_gate_passed else 'FAIL'} | realism: {format_percent(row.realism_score)}",
            f"Scenes: {row.scene_count} | videos: {row.generated_scene_video_count} | voices: {row.scene_dialogue_audio_count} | masters: {row.scene_master_clip_count}",
            f"Backend: {row.backend_runner_status or '-'} | failed tasks: {row.backend_runner_failed_count}",
            f"Regeneration queue: {row.regeneration_queue_count}",
            "",
            "Blockers:",
            *(f"- {item}" for item in (row.blockers or ["none"])),
            "",
            "Warnings:",
            *(f"- {item}" for item in (row.warnings or ["none"])),
            "",
            "Paths:",
            *(f"- {key}: {value}" for key, value in sorted(row.paths.items())),
        ]
        set_detail("\n".join(parts))

    def open_target(target: str) -> None:
        row = current_record()
        if not row:
            messagebox.showwarning("No episode", "Select an episode first.")
            return
        path = target_path_for_record(row, target)
        if not path:
            messagebox.showwarning("Missing target", f"No existing {target} target was found for {row.episode_id}.")
            return
        open_path(path)

    def archive_checked() -> None:
        selected_rows = checked_records()
        if not selected_rows:
            messagebox.showwarning("No checked episodes", "Check one or more generated episodes first.")
            return
        candidates_by_episode = {row.episode_id: archive_candidates_for_episode(row) for row in selected_rows}
        total_candidates = sum(len(items) for items in candidates_by_episode.values())
        if not total_candidates:
            messagebox.showinfo("Nothing to archive", "No active generated files were found for the checked episodes.")
            return
        if not messagebox.askyesno(
            "Archive checked episodes",
            f"Archive {total_candidates} active generated path(s) for {len(selected_rows)} checked episode(s)?\n\n"
            "This moves generation outputs to generation/archive/generated_episodes and can be inspected later.",
        ):
            return
        manifests = [archive_episode(row) for row in selected_rows if candidates_by_episode.get(row.episode_id)]
        messagebox.showinfo("Archived", f"Archived {len(manifests)} generated episode(s).")
        checked_episode_ids.clear()
        sync_episode_records()

    def delete_checked() -> None:
        selected_rows = checked_records()
        if not selected_rows:
            messagebox.showwarning("No checked episodes", "Check one or more generated episodes first.")
            return
        candidates_by_episode = {row.episode_id: archive_candidates_for_episode(row) for row in selected_rows}
        total_candidates = sum(len(items) for items in candidates_by_episode.values())
        if not total_candidates:
            messagebox.showinfo("Nothing to delete", "No active generated files were found for the checked episodes.")
            return
        if not messagebox.askyesno(
            "Delete checked episodes",
            f"Delete {total_candidates} active generated path(s) for {len(selected_rows)} checked episode(s)?\n\n"
            "This permanently removes generated outputs only. Source episodes, training data, and reviewed character data are not touched.",
        ):
            return
        typed = simpledialog.askstring(
            "Confirm delete",
            "Type DELETE to permanently delete generated outputs for all checked episodes.",
            parent=root,
        )
        if typed != "DELETE":
            messagebox.showinfo("Delete cancelled", "Confirmation did not match. Nothing was deleted.")
            return
        results = [delete_episode_outputs(row) for row in selected_rows if candidates_by_episode.get(row.episode_id)]
        deleted_count = sum(len(result["deleted"]) for result in results)
        messagebox.showinfo("Deleted", f"Deleted {deleted_count} generated path(s) for {len(results)} episode(s).")
        checked_episode_ids.clear()
        sync_episode_records()

    def open_live_status_file() -> None:
        status = build_live_generation_status(cfg)
        target = existing_path(status.current_status_path) or existing_path(status.current_status_markdown_path)
        if not target:
            messagebox.showwarning("Missing status", "No live status file was found yet.")
            return
        open_path(target)

    tree.bind("<<TreeviewSelect>>", update_detail)
    tree.bind("<Button-1>", on_tree_click, add="+")
    live_frame.grid(row=0, column=0, columnspan=8, sticky="nsew", padx=10, pady=(10, 4))
    live_detail.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
    ttk.Button(live_frame, text="Open Status", command=open_live_status_file).grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
    live_frame.columnconfigure(0, weight=1)
    live_frame.columnconfigure(1, weight=0)
    live_frame.rowconfigure(0, weight=1)

    tree.grid(row=1, column=0, columnspan=8, sticky="nsew", padx=10, pady=4)
    detail.grid(row=2, column=0, columnspan=8, sticky="nsew", padx=10, pady=4)

    ttk.Button(root, text="Refresh", command=refresh_all).grid(row=3, column=0, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Select All", command=select_all).grid(row=3, column=1, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Clear Checks", command=clear_checks).grid(row=3, column=2, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Open Video", command=lambda: open_target("best-video")).grid(row=3, column=3, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Open Folder", command=lambda: open_target("folder")).grid(row=3, column=4, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Quality Report", command=lambda: open_target("quality")).grid(row=3, column=5, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Realism Report", command=lambda: open_target("realism")).grid(row=3, column=6, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Production Package", command=lambda: open_target("package")).grid(row=3, column=7, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Archive Checked", command=archive_checked).grid(row=4, column=0, columnspan=3, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Delete Checked", command=delete_checked).grid(row=4, column=3, columnspan=3, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Close", command=root.destroy).grid(row=4, column=6, columnspan=2, sticky="ew", padx=6, pady=6)
    ttk.Label(root, textvariable=status_text).grid(row=5, column=0, columnspan=8, sticky="w", padx=10, pady=(0, 8))

    root.columnconfigure(0, weight=1)
    for index in range(1, 8):
        root.columnconfigure(index, weight=1)
    root.rowconfigure(0, weight=1)
    root.rowconfigure(1, weight=2)
    root.rowconfigure(2, weight=3)

    refresh_all()
    root.mainloop()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage generated episode outputs.")
    parser.add_argument("--gui", action="store_true", help="Open the Tk episode manager GUI.")
    parser.add_argument("--list", action="store_true", help="List generated episodes in the terminal.")
    parser.add_argument("--live", action="store_true", help="Show current live generation/render status once.")
    parser.add_argument("--watch", action="store_true", help="Continuously update the live generation/render status.")
    parser.add_argument("--update-seconds", dest="update_seconds", type=float, default=WATCH_DEFAULT_SECONDS, help="Live update interval for --watch.")
    parser.add_argument("--refresh-seconds", dest="update_seconds", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--print-json", action="store_true", help="Print generated episode records as JSON.")
    parser.add_argument("--episode-id", help="Target one generated episode.")
    parser.add_argument("--open", dest="open_target", choices=sorted(OPEN_TARGETS), help="Open a generated artifact.")
    parser.add_argument("--archive", action="store_true", help="Move active generated outputs for --episode-id into generation/archive.")
    parser.add_argument("--delete", action="store_true", help="Permanently delete active generated outputs for --episode-id from generation/.")
    parser.add_argument("--dry-run", action="store_true", help="Show what archive/delete would change without changing anything.")
    parser.add_argument("--confirm", help="Required for --archive or --delete. Must equal --episode-id.")
    return parser.parse_args(argv)


def should_open_gui_by_default(argv: list[str] | None = None) -> bool:
    raw_args = sys.argv[1:] if argv is None else list(argv)
    return len(raw_args) == 0


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config()

    if should_open_gui_by_default(argv):
        args.gui = True

    if args.gui:
        run_gui(cfg)
        return

    if args.watch:
        watch_live_status(cfg, float(args.update_seconds or WATCH_DEFAULT_SECONDS), json_output=bool(args.print_json))
        return

    if args.live:
        headline("Generated Episode Live Monitor")
        print_live_status(cfg, json_output=bool(args.print_json))
        if not (args.list or args.episode_id or args.open_target or args.archive or args.delete):
            return

    headline("Manage Generated Episodes")
    records = list_episode_records(cfg)
    if args.episode_id:
        records = [select_record(records, args.episode_id)]

    if args.print_json:
        print(json.dumps([row.to_dict() for row in records], indent=2, ensure_ascii=False))
    else:
        if not records:
            info("No generated episodes were found.")
        else:
            print(format_episode_table(records))

    if args.open_target:
        if not args.episode_id:
            raise RuntimeError("--open requires --episode-id.")
        record = select_record(records, args.episode_id)
        path = target_path_for_record(record, args.open_target)
        if not path:
            raise RuntimeError(f"No existing {args.open_target} target found for {record.episode_id}.")
        info(f"Opening {path}")
        open_path(path)

    if args.archive and args.delete:
        raise RuntimeError("Use either --archive or --delete, not both.")

    if args.archive:
        if not args.episode_id:
            raise RuntimeError("--archive requires --episode-id.")
        if not args.dry_run and args.confirm != args.episode_id:
            raise RuntimeError("--archive requires --confirm with the same value as --episode-id.")
        record = select_record(records, args.episode_id)
        manifest = archive_episode(record, dry_run=bool(args.dry_run))
        if args.dry_run:
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
            info(f"Dry run found {len(manifest['moved'])} path(s) for archiving.")
        else:
            ok(f"Archived {len(manifest['moved'])} path(s) to {manifest['archive_root']}")

    if args.delete:
        if not args.episode_id:
            raise RuntimeError("--delete requires --episode-id.")
        if not args.dry_run and args.confirm != args.episode_id:
            raise RuntimeError("--delete requires --confirm with the same value as --episode-id.")
        record = select_record(records, args.episode_id)
        result = delete_episode_outputs(record, dry_run=bool(args.dry_run))
        if args.dry_run:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            info(f"Dry run found {len(result['deleted'])} generated path(s) for deletion.")
        else:
            ok(f"Deleted {len(result['deleted'])} generated path(s) for {record.episode_id}.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        warn("Interrupted.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
