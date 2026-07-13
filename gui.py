#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR / "ai_series_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from support_scripts.pipeline_common import (  # noqa: E402
    CONFIG_PATH,
    CONFIG_TEMPLATE_PATH,
    DEFAULT_CONFIG,
    deep_merge,
    generated_episode_artifacts,
    headline,
    info,
    list_generated_episode_artifacts,
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


def gui_log(message: str) -> None:
    line = f"[25 GUI {time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    try:
        log_path = PROJECT_DIR / "logs" / "gui_manager.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(line + "\n")
    except Exception:
        pass


def gui_log_error(message: str, exc: BaseException | None = None) -> None:
    gui_log(f"ERROR: {message}")
    print(f"[25 GUI {time.strftime('%H:%M:%S')}] ERROR: {message}", file=sys.stderr, flush=True)
    if exc is not None:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        sys.stderr.flush()
        try:
            log_path = PROJECT_DIR / "logs" / "gui_manager.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8", errors="replace") as handle:
                traceback.print_exception(type(exc), exc, exc.__traceback__, file=handle)
        except Exception:
            pass


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

SEASON_ASSET_KINDS = ("intro", "outro")
ASSET_OPEN_TARGETS = {
    "video",
    "folder",
    "manifest",
    "generated-folder",
    "generated-video",
    "generated-package",
}

IGNORED_EPISODE_NAMES = {"latest", "archive", "generated_episodes"}
LIVE_STALE_SECONDS = 20 * 60
WATCH_DEFAULT_SECONDS = 5.0
LIVE_SCAN_TIME_LIMIT_SECONDS = 2.0
LIVE_SCAN_MAX_JSON_FILES = 1200
LIVE_SCAN_MAX_DIRECTORIES = 800


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
class SeasonAssetRecord:
    asset_id: str
    season_id: str
    asset_kind: str
    display_title: str
    status: str = "missing"
    duration_seconds: float = 0.0
    generated_video_count: int = 0
    generated_image_count: int = 0
    fallback_used: bool = False
    placeholder_used: bool = False
    updated_at: float = 0.0
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    paths: dict[str, str] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "season_id": self.season_id,
            "asset_kind": self.asset_kind,
            "display_title": self.display_title,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "generated_video_count": self.generated_video_count,
            "generated_image_count": self.generated_image_count,
            "fallback_used": self.fallback_used,
            "placeholder_used": self.placeholder_used,
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
    season_assets: list[dict[str, Any]] = field(default_factory=list)
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
            "season_assets": list(self.season_assets),
            "current_status_path": self.current_status_path,
            "current_status_markdown_path": self.current_status_markdown_path,
            "messages": list(self.messages),
        }


@dataclass
class JsonScanResult:
    paths: list[Path] = field(default_factory=list)
    truncated: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class ProjectStorageRecord:
    record_id: str
    category: str
    display_name: str
    relative_path: str
    path: str
    kind: str
    size_bytes: int = 0
    item_count: int = 0
    updated_at: float = 0.0
    editable_json: bool = False
    archive_allowed: bool = False
    delete_allowed: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "category": self.category,
            "display_name": self.display_name,
            "relative_path": self.relative_path,
            "path": self.path,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "item_count": self.item_count,
            "updated_at": self.updated_at,
            "updated_at_text": format_timestamp(self.updated_at),
            "editable_json": self.editable_json,
            "archive_allowed": self.archive_allowed,
            "delete_allowed": self.delete_allowed,
            "note": self.note,
        }


PROJECT_STORAGE_SCOPES: tuple[tuple[str, str, str, str, int, bool], ...] = (
    ("Imports", "Imported source data", "data", "Imported episodes, extracted scenes, audio and transcripts.", 5, True),
    ("Characters", "Character databases", "characters", "Face, speaker, voice and relationship mappings.", 4, True),
    ("Series bible", "Series bible", "series_bible", "Series knowledge, behavior reports and writer-room data.", 3, True),
    ("Training", "Training data and models", "training", "Prepared datasets and local training results.", 3, True),
    ("Generation", "Generation workspaces", "generation", "Current generated packages and reports. Episode tabs offer focused management.", 3, True),
    ("Exports", "Exports", "exports", "Export packages prepared for review or delivery.", 3, True),
    ("Logs", "Logs", "logs", "Pipeline, backend and GUI diagnostics.", 2, True),
    ("Runtime", "Runtime and local models", "runtime", "Project-local environments, caches and downloaded models.", 2, True),
    ("Archives", "Managed archives", "archives", "Archives created by the project and the GUI.", 3, True),
    ("Config", "Project configuration", "configs", "Active project configuration and public template.", 2, False),
)
PROJECT_STORAGE_MAX_RECORDS = 900
PROJECT_STORAGE_ARCHIVE_DIR = "archives/gui_project_data"
PROJECT_JSON_BACKUP_DIR = "runtime/gui_backups/json"
PROJECT_STORAGE_IGNORED_NAMES = {"__pycache__", ".git", ".cache", ".pytest_cache"}
PROJECT_MEDIA_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".wav", ".mp3", ".flac", ".m4a"}


def project_root_path() -> Path:
    return resolve_project_path("").resolve(strict=False)


def project_relative_path(path: Path) -> Path:
    root = project_root_path()
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to manage a path outside ai_series_project: {path}") from exc


def project_storage_scope_for_path(path: Path) -> tuple[str, str, str, str, int, bool] | None:
    relative = project_relative_path(path)
    if not relative.parts:
        return None
    for scope in PROJECT_STORAGE_SCOPES:
        if relative.parts[0] == scope[2]:
            return scope
    return None


def project_storage_path_stats(path: Path, *, max_entries: int = 1500) -> tuple[int, int, float, bool]:
    if not path.exists():
        return 0, 0, 0.0, False
    try:
        if path.is_file():
            stat = path.stat()
            return int(stat.st_size), 1, float(stat.st_mtime), False
    except OSError:
        return 0, 0, 0.0, False

    total_size = 0
    item_count = 0
    latest = 0.0
    truncated = False
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        for child in children:
            if child.name in PROJECT_STORAGE_IGNORED_NAMES:
                continue
            item_count += 1
            if item_count > max_entries:
                truncated = True
                return total_size, item_count - 1, latest, truncated
            try:
                stat = child.stat()
                latest = max(latest, float(stat.st_mtime))
                if child.is_file():
                    total_size += int(stat.st_size)
                elif child.is_dir() and not child.is_symlink():
                    stack.append(child)
            except OSError:
                continue
    return total_size, item_count, latest, truncated


def iter_project_storage_paths(root: Path, *, max_depth: int, max_records: int) -> list[Path]:
    paths: list[Path] = []
    if not root.exists():
        return paths
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack and len(paths) < max_records:
        current, depth = stack.pop()
        try:
            children = sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        except OSError:
            continue
        for child in children:
            if child.name in PROJECT_STORAGE_IGNORED_NAMES:
                continue
            paths.append(child)
            if len(paths) >= max_records:
                break
            try:
                if child.is_dir() and not child.is_symlink() and depth < max_depth:
                    stack.append((child, depth + 1))
            except OSError:
                continue
    return paths


def project_storage_record_for_path(
    path: Path,
    scope: tuple[str, str, str, str, int, bool],
    *,
    root_summary: tuple[int, int, float, bool] | None = None,
) -> ProjectStorageRecord:
    category, label, _relative_root, note, _max_depth, mutable = scope
    relative = project_relative_path(path)
    is_root = relative.as_posix() == scope[2]
    try:
        is_file = path.is_file()
        is_directory = path.is_dir()
        stat = path.stat()
        updated_at = float(stat.st_mtime)
        size_bytes = int(stat.st_size) if is_file else 0
    except OSError:
        is_file = False
        is_directory = False
        updated_at = 0.0
        size_bytes = 0
    item_count = 0
    if is_root and root_summary is not None:
        size_bytes, item_count, summary_updated, truncated = root_summary
        updated_at = max(updated_at, summary_updated)
        if truncated:
            note = f"{note} Size/count are bounded for responsiveness."
    elif is_directory:
        try:
            item_count = sum(1 for _ in path.iterdir())
        except OSError:
            item_count = 0
    suffix = path.suffix.lower()
    is_archive = bool(relative.parts and relative.parts[0] == "archives")
    editable_json = bool(is_file and suffix == ".json" and mutable)
    if relative.as_posix() in {"configs/project.template.json"}:
        editable_json = False
    if editable_json:
        kind = "JSON database"
    elif is_file and suffix in PROJECT_MEDIA_SUFFIXES and relative.parts and relative.parts[0] == "data":
        kind = "Imported media"
    elif is_directory:
        kind = "Folder"
    elif is_file:
        kind = "File"
    else:
        kind = "Missing"
    return ProjectStorageRecord(
        record_id=relative.as_posix(),
        category=category,
        display_name=label if is_root else path.name,
        relative_path=relative.as_posix(),
        path=str(path),
        kind=kind,
        size_bytes=size_bytes,
        item_count=item_count,
        updated_at=updated_at,
        editable_json=editable_json,
        archive_allowed=bool(mutable and not is_archive),
        delete_allowed=bool(mutable),
        note=note,
    )


def list_project_storage_records(*, max_records: int = PROJECT_STORAGE_MAX_RECORDS) -> list[ProjectStorageRecord]:
    records: list[ProjectStorageRecord] = []
    scoped_roots: list[tuple[tuple[str, str, str, str, int, bool], Path]] = []
    for scope in PROJECT_STORAGE_SCOPES:
        root = resolve_project_path(scope[2])
        if not root.exists():
            continue
        scoped_roots.append((scope, root))

    remaining = max(len(scoped_roots), max_records)
    for scope, root in scoped_roots:
        if remaining <= 0:
            break
        summary = project_storage_path_stats(root)
        records.append(project_storage_record_for_path(root, scope, root_summary=summary))
        remaining -= 1

    for index, (scope, root) in enumerate(scoped_roots):
        if remaining <= 0:
            break
        scopes_left = max(1, len(scoped_roots) - index)
        per_scope_limit = max(1, remaining // scopes_left)
        before_count = len(records)
        for path in iter_project_storage_paths(root, max_depth=scope[4], max_records=per_scope_limit):
            records.append(project_storage_record_for_path(path, scope))
            remaining -= 1
            if remaining <= 0:
                break
        if len(records) == before_count:
            continue
    return sorted(records, key=lambda row: (row.category.lower(), row.relative_path.lower()))


def select_project_storage_record(records: list[ProjectStorageRecord], record_id: str) -> ProjectStorageRecord:
    for record in records:
        if record.record_id == record_id:
            return record
    raise RuntimeError(f"Managed project item was not found: {record_id}")


def project_storage_mutation_paths(records: list[ProjectStorageRecord], *, operation: str) -> list[Path]:
    allowed = "archive_allowed" if operation == "archive" else "delete_allowed"
    candidates: dict[str, Path] = {}
    for record in records:
        if not bool(getattr(record, allowed, False)):
            continue
        path = Path(record.path)
        project_relative_path(path)
        if not path.exists():
            continue
        candidates[str(path.resolve(strict=False)).lower()] = path
    return prune_nested_archive_candidates(sorted(candidates.values(), key=lambda item: (len(item.parts), str(item).lower())))


def archive_project_storage_records(records: list[ProjectStorageRecord], *, dry_run: bool = False) -> dict[str, Any]:
    candidates = project_storage_mutation_paths(records, operation="archive")
    root = project_root_path()
    archive_root = root / PROJECT_STORAGE_ARCHIVE_DIR / time.strftime("%Y%m%d_%H%M%S")
    moved: list[dict[str, str]] = []
    for source in candidates:
        relative = project_relative_path(source)
        target = archive_root / relative
        moved.append({"source": str(source), "target": str(target)})
        if dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    manifest = {
        "action": "archive_project_storage",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": dry_run,
        "archive_root": str(archive_root),
        "moved": moved,
    }
    if not dry_run and moved:
        write_json(archive_root / "archive_manifest.json", manifest)
    return manifest


def delete_project_storage_records(records: list[ProjectStorageRecord], *, dry_run: bool = False) -> dict[str, Any]:
    candidates = project_storage_mutation_paths(records, operation="delete")
    deleted: list[str] = []
    for source in candidates:
        project_relative_path(source)
        deleted.append(str(source))
        if dry_run:
            continue
        if source.is_dir():
            shutil.rmtree(source)
        else:
            source.unlink()
    return {
        "action": "delete_project_storage",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": dry_run,
        "deleted": deleted,
    }


def read_json_database(record: ProjectStorageRecord) -> Any:
    if not record.editable_json:
        raise RuntimeError(f"This project item is not an editable JSON database: {record.relative_path}")
    path = Path(record.path)
    project_relative_path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def json_database_summary(record: ProjectStorageRecord) -> dict[str, Any]:
    payload = read_json_database(record)
    if isinstance(payload, dict):
        summary = f"JSON object with {len(payload)} top-level key(s)."
    elif isinstance(payload, list):
        summary = f"JSON list with {len(payload)} item(s)."
    else:
        summary = f"JSON scalar of type {type(payload).__name__}."
    return {"path": record.path, "summary": summary, "payload_type": type(payload).__name__}


def save_json_database(record: ProjectStorageRecord, text: str) -> dict[str, str]:
    if not record.editable_json:
        raise RuntimeError(f"This project item is not an editable JSON database: {record.relative_path}")
    payload = json.loads(text)
    path = Path(record.path)
    relative = project_relative_path(path)
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"JSON database no longer exists: {path}")
    backup = project_root_path() / PROJECT_JSON_BACKUP_DIR / time.strftime("%Y%m%d_%H%M%S") / relative
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)
    temporary = path.with_name(f"{path.name}.gui-edit.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return {"path": str(path), "backup": str(backup), "backup_path": str(backup)}


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


def format_file_size(size_bytes: int | float | None) -> str:
    size = max(0, float(size_bytes or 0))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return "0 B"


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


def configured_season_ids(cfg: dict[str, Any]) -> list[str]:
    season_ids: set[str] = set()
    generation_cfg = cfg.get("generation", {}) if isinstance(cfg.get("generation", {}), dict) else {}
    asset_configs = [
        cfg.get(f"season_{kind}", {})
        for kind in SEASON_ASSET_KINDS
        if isinstance(cfg.get(f"season_{kind}", {}), dict)
    ]
    for value in [generation_cfg.get("default_season_id"), *(asset.get("default_season_id") for asset in asset_configs)]:
        season_id = coalesce_text(value)
        if season_id:
            season_ids.add(season_id)
    for asset_cfg in asset_configs:
        profiles = asset_cfg.get("profiles", {}) if isinstance(asset_cfg.get("profiles", {}), dict) else {}
        season_ids.update(coalesce_text(key) for key in profiles if coalesce_text(key))
    root = season_assets_root_path()
    if root.exists():
        season_ids.update(child.name for child in root.iterdir() if child.is_dir() and not child.name.startswith("."))
    return sorted(season_ids or {"season_01"})


def season_assets_root_path() -> Path:
    return resolve_project_path("generation/season_assets")


def season_asset_root_path(season_id: str, asset_kind: str) -> Path:
    return season_assets_root_path() / season_id / asset_kind


def season_asset_id(season_id: str, asset_kind: str) -> str:
    return f"{season_id}_{asset_kind}"


def count_media_files(root: Path, suffixes: tuple[str, ...]) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def first_existing_media(root: Path, suffixes: tuple[str, ...]) -> Path | None:
    if not root.exists():
        return None
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            return path
    return None


def season_asset_manifest_path(asset_root: Path, asset_kind: str) -> Path:
    return asset_root / f"{asset_kind}_manifest.json"


def season_asset_generated_package_path(asset_root: Path, asset_kind: str) -> Path | None:
    scenes_root = asset_root / "generated" / "scenes"
    preferred = scenes_root / f"season_{asset_kind}_production.json"
    if preferred.exists():
        return preferred
    if scenes_root.exists():
        for path in sorted(scenes_root.glob("*_production.json")):
            if path.is_file():
                return path
    return None


def configured_asset_source_path(cfg: dict[str, Any], season_id: str, asset_kind: str) -> Path | None:
    asset_cfg = cfg.get(f"season_{asset_kind}", {}) if isinstance(cfg.get(f"season_{asset_kind}", {}), dict) else {}
    profiles = asset_cfg.get("profiles", {}) if isinstance(asset_cfg.get("profiles", {}), dict) else {}
    profile = profiles.get(season_id, {}) if isinstance(profiles.get(season_id, {}), dict) else {}
    source = coalesce_text(profile.get("source_video"))
    if not source:
        return None
    return resolve_stored_project_path(source)


def build_season_asset_record(cfg: dict[str, Any], season_id: str, asset_kind: str) -> SeasonAssetRecord:
    asset_root = season_asset_root_path(season_id, asset_kind)
    manifest_path = season_asset_manifest_path(asset_root, asset_kind)
    manifest = read_json_safely(manifest_path, {}) if manifest_path.exists() else {}
    generated_root = asset_root / "generated"
    generated_video_root = generated_root / "videos"
    generated_image_root = generated_root / "images"
    generated_manifest_root = generated_root / "manifests"
    generated_package = season_asset_generated_package_path(asset_root, asset_kind)

    manifest_video = coalesce_text(manifest.get("canonical_video") if isinstance(manifest, dict) else "")
    canonical_video = existing_path(manifest_video) if manifest_video else existing_path(asset_root / f"{asset_kind}.mp4")
    generated_video = first_existing_media(generated_video_root, (".mp4", ".mov", ".mkv", ".webm"))
    generated_video_count = count_media_files(generated_video_root, (".mp4", ".mov", ".mkv", ".webm"))
    generated_image_count = count_media_files(generated_image_root, (".png", ".jpg", ".jpeg", ".webp"))
    fallback_used = bool(manifest.get("fallback_used", False)) if isinstance(manifest, dict) else False
    placeholder_used = bool(manifest.get("placeholder_used", False)) if isinstance(manifest, dict) else False

    paths: dict[str, str] = {"folder": str(asset_root)}
    if canonical_video:
        paths["canonical_video"] = str(canonical_video)
    if manifest_path.exists():
        paths["manifest"] = str(manifest_path)
    if generated_root.exists():
        paths["generated_folder"] = str(generated_root)
    if generated_video:
        paths["generated_video"] = str(generated_video)
    if generated_package:
        paths["generated_package"] = str(generated_package)
    if generated_image_root.exists():
        paths["generated_images"] = str(generated_image_root)
    if generated_manifest_root.exists():
        paths["generated_manifests"] = str(generated_manifest_root)
    source = configured_asset_source_path(cfg, season_id, asset_kind)
    if source:
        paths["configured_source_video"] = str(source)

    warnings: list[str] = []
    blockers: list[str] = []
    if fallback_used:
        blockers.append("Asset manifest reports fallback_used=true.")
    if placeholder_used:
        blockers.append("Asset manifest reports placeholder_used=true.")
    if asset_root.exists() and not canonical_video:
        warnings.append(f"No canonical {asset_kind}.mp4 was found yet.")
    if canonical_video and not manifest_path.exists():
        warnings.append("Canonical media exists without a lock manifest.")

    if fallback_used or placeholder_used:
        status = "blocked_fallback"
    elif canonical_video and manifest_path.exists():
        source_origin = coalesce_text(manifest.get("source_origin") if isinstance(manifest, dict) else "")
        status = source_origin or "ready"
    elif canonical_video:
        status = "ready_unmanaged"
    elif generated_video:
        status = "generated_pending_lock"
    elif generated_root.exists() or asset_root.exists():
        status = "incomplete"
    else:
        status = "missing"

    existing_paths = [path for path in (existing_path(value) for value in paths.values()) if path is not None]
    return SeasonAssetRecord(
        asset_id=season_asset_id(season_id, asset_kind),
        season_id=season_id,
        asset_kind=asset_kind,
        display_title=f"{season_id} {asset_kind}",
        status=status,
        duration_seconds=as_float(manifest.get("duration_seconds") if isinstance(manifest, dict) else 0.0),
        generated_video_count=generated_video_count,
        generated_image_count=generated_image_count,
        fallback_used=fallback_used,
        placeholder_used=placeholder_used,
        updated_at=latest_mtime(existing_paths + ([asset_root] if asset_root.exists() else [])),
        warnings=dedupe_texts(warnings),
        blockers=dedupe_texts(blockers),
        paths=paths,
        manifest=manifest if isinstance(manifest, dict) else {},
    )


def list_season_asset_records(cfg: dict[str, Any]) -> list[SeasonAssetRecord]:
    records: list[SeasonAssetRecord] = []
    known_roots: set[tuple[str, str]] = set()
    root = season_assets_root_path()
    if root.exists():
        for season_dir in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if not season_dir.is_dir() or season_dir.name.startswith("."):
                continue
            for kind_dir in sorted(season_dir.iterdir(), key=lambda item: item.name.lower()):
                if kind_dir.is_dir() and kind_dir.name in SEASON_ASSET_KINDS:
                    known_roots.add((season_dir.name, kind_dir.name))

    for season_id in configured_season_ids(cfg):
        for asset_kind in SEASON_ASSET_KINDS:
            known_roots.add((season_id, asset_kind))

    for season_id, asset_kind in sorted(known_roots, key=lambda item: (item[0].lower(), item[1])):
        records.append(build_season_asset_record(cfg, season_id, asset_kind))
    return records


def current_pipeline_status_path() -> Path:
    return resolve_project_path("runtime/autosaves/24_process_next_episode/current_status.json")


def current_pipeline_status_markdown_path() -> Path:
    return resolve_project_path("runtime/autosaves/24_process_next_episode/current_status.md")


def step_autosave_root_path() -> Path:
    return resolve_project_path("runtime/autosaves/steps")


def distributed_runtime_root_path() -> Path:
    return resolve_project_path("runtime/distributed")


def read_json_safely(path: Path, default: Any) -> Any:
    try:
        return read_json(path, default)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return default


def load_manager_config() -> dict[str, Any]:
    template = read_json_safely(CONFIG_TEMPLATE_PATH, {})
    if not isinstance(template, dict):
        template = {}
    existing = read_json_safely(CONFIG_PATH, {})
    if not isinstance(existing, dict):
        existing = {}
    # 25 is a viewer/cleanup tool. It must not block on mkdir calls across a slow NAS.
    return deep_merge(deep_merge(DEFAULT_CONFIG, template), existing)


def scan_json_files_bounded(
    root: Path,
    *,
    max_files: int = LIVE_SCAN_MAX_JSON_FILES,
    max_directories: int = LIVE_SCAN_MAX_DIRECTORIES,
    time_limit_seconds: float = LIVE_SCAN_TIME_LIMIT_SECONDS,
) -> JsonScanResult:
    result = JsonScanResult()
    try:
        if not root.exists():
            return result
    except OSError as exc:
        result.errors.append(f"{root}: {exc}")
        return result

    deadline = time.monotonic() + max(0.1, float(time_limit_seconds or 0.1))
    stack = [root]
    directories_seen = 0

    while stack:
        if time.monotonic() >= deadline or len(result.paths) >= max_files or directories_seen >= max_directories:
            result.truncated = True
            break
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    if (
                        time.monotonic() >= deadline
                        or len(result.paths) >= max_files
                        or directories_seen >= max_directories
                    ):
                        result.truncated = True
                        break
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            directories_seen += 1
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".json"):
                            result.paths.append(Path(entry.path))
                    except OSError as exc:
                        result.errors.append(f"{entry.path}: {exc}")
        except OSError as exc:
            result.errors.append(f"{current}: {exc}")
    return result


def load_current_pipeline_status() -> dict[str, Any]:
    path = current_pipeline_status_path()
    return read_json_safely(path, {}) if path.exists() else {}


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


def load_step_autosaves_with_diagnostics() -> tuple[list[dict[str, Any]], list[str]]:
    root = step_autosave_root_path()
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    scan = scan_json_files_bounded(root)
    if scan.truncated:
        notes.append(
            f"Step autosave scan was limited after {len(scan.paths)} JSON file(s); the manager stays usable."
        )
    if scan.errors:
        notes.append(f"Step autosave scan skipped {len(scan.errors)} unreadable path(s).")
    for path in scan.paths:
        payload = read_json_safely(path, {})
        if not isinstance(payload, dict):
            continue
        try:
            fallback_mtime = path.stat().st_mtime
        except OSError:
            fallback_mtime = 0.0
        updated_at = parse_timestamp(payload.get("updated_at")) or fallback_mtime
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
    return rows, notes


def load_step_autosaves() -> list[dict[str, Any]]:
    rows, _notes = load_step_autosaves_with_diagnostics()
    return rows


def load_worker_leases_with_diagnostics() -> tuple[list[dict[str, Any]], list[str]]:
    root = distributed_runtime_root_path()
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    now = time.time()
    scan = scan_json_files_bounded(root)
    if scan.truncated:
        notes.append(
            f"Distributed worker lease scan was limited after {len(scan.paths)} JSON file(s); use Refresh after stale runtime cleanup if a worker is missing."
        )
    if scan.errors:
        notes.append(f"Distributed worker lease scan skipped {len(scan.errors)} unreadable path(s).")
    for path in scan.paths:
        payload = read_json_safely(path, {})
        if not isinstance(payload, dict) or "owner_id" not in payload:
            continue
        meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
        heartbeat_at = parse_timestamp(payload.get("heartbeat_at"))
        expires_at = parse_timestamp(payload.get("expires_at"))
        heartbeat_age = now - heartbeat_at if heartbeat_at else 0.0
        lease_not_expired = bool(expires_at and expires_at > now)
        heartbeat_fresh = bool(heartbeat_at and heartbeat_age <= LIVE_STALE_SECONDS)
        active = bool(lease_not_expired and heartbeat_fresh)
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
                "heartbeat_age": format_age(heartbeat_age) if heartbeat_at else "",
                "expires_at": expires_at,
                "expires_in": format_duration(expires_at - now) if expires_at else "",
                "lease_not_expired": lease_not_expired,
                "heartbeat_fresh": heartbeat_fresh,
                "active": active,
            }
        )
    rows.sort(key=lambda item: (not bool(item.get("active")), -(float(item.get("heartbeat_at", 0.0) or 0.0))))
    return rows, notes


def load_worker_leases() -> list[dict[str, Any]]:
    rows, _notes = load_worker_leases_with_diagnostics()
    return rows


def season_asset_live_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in list_season_asset_records(cfg):
        rows.append(
            {
                "asset_id": record.asset_id,
                "season_id": record.season_id,
                "asset_kind": record.asset_kind,
                "status": record.status,
                "duration_seconds": record.duration_seconds,
                "updated_at": record.updated_at,
                "canonical_video": record.paths.get("canonical_video", ""),
                "manifest": record.paths.get("manifest", ""),
                "fallback_used": record.fallback_used,
                "placeholder_used": record.placeholder_used,
            }
        )
    return rows


def format_season_asset_live_summary(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "-"
    parts: list[str] = []
    for row in rows[:6]:
        label = f"{coalesce_text(row.get('season_id'))} {coalesce_text(row.get('asset_kind'))}".strip()
        status = coalesce_text(row.get("status")) or "missing"
        parts.append(f"{label}: {status}")
    if len(rows) > 6:
        parts.append(f"+{len(rows) - 6} more")
    return " | ".join(parts)


def build_live_generation_status(cfg: dict[str, Any], *, include_step_autosaves: bool = False) -> LiveGenerationStatus:
    status_payload = load_current_pipeline_status()
    now = time.time()
    status_updated = parse_timestamp(status_payload.get("updated_at")) if isinstance(status_payload, dict) else 0.0
    completed, total = summarize_global_progress(status_payload if isinstance(status_payload, dict) else {})
    workers, worker_notes = load_worker_leases_with_diagnostics()
    active_workers = [row for row in workers if bool(row.get("active"))]
    stale_worker_leases = [
        row
        for row in workers
        if bool(row.get("lease_not_expired")) and not bool(row.get("active")) and not bool(row.get("heartbeat_fresh"))
    ]
    step_rows: list[dict[str, Any]] = []
    step_notes: list[str] = []
    if include_step_autosaves:
        step_rows, step_notes = load_step_autosaves_with_diagnostics()
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
    messages.extend(worker_notes)
    messages.extend(step_notes)
    if active_workers:
        messages.append(f"{len(active_workers)} active distributed worker lease(s).")
    if status_running and status_updated and status_age > LIVE_STALE_SECONDS and active_workers:
        messages.append(
            f"Current 24_process_next_episode status file is old ({format_age(status_age)}), "
            "but fresh worker heartbeats still exist; progress reporting may be stale."
        )
    if stale_worker_leases:
        messages.append(
            f"{len(stale_worker_leases)} worker lease(s) still exist but have stale heartbeats; they are not counted as active."
        )
    if stale:
        messages.append(f"Current 24_process_next_episode autosave is stale ({format_age(status_age)} old).")
    if not include_step_autosaves:
        messages.append("Step autosave scan is skipped for fast manager startup.")
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
        season_assets=season_asset_live_rows(cfg),
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
        f"Season assets: {format_season_asset_live_summary(status.season_assets)}",
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


def format_asset_table(records: list[SeasonAssetRecord]) -> str:
    headers = ["Asset", "Season", "Kind", "Status", "Dur", "Videos", "Images", "Updated"]
    rows: list[list[str]] = []
    for row in records:
        rows.append(
            [
                row.asset_id,
                row.season_id,
                row.asset_kind,
                row.status or "-",
                format_duration(row.duration_seconds) if row.duration_seconds else "-",
                str(row.generated_video_count or "-"),
                str(row.generated_image_count or "-"),
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


def target_path_for_asset(record: SeasonAssetRecord, target: str) -> Path | None:
    target_key = {
        "video": "canonical_video",
        "folder": "folder",
        "manifest": "manifest",
        "generated-folder": "generated_folder",
        "generated-video": "generated_video",
        "generated-package": "generated_package",
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


def archive_candidates_for_season_asset(record: SeasonAssetRecord) -> list[Path]:
    generation_root = resolve_project_path("generation")
    candidates: list[Path] = []
    folder = existing_path(record.paths.get("folder", ""))
    if folder:
        candidates.append(folder)
    for key in (
        "canonical_video",
        "manifest",
        "generated_folder",
        "generated_video",
        "generated_package",
        "generated_images",
        "generated_manifests",
    ):
        path = existing_path(record.paths.get(key, ""))
        if path:
            candidates.append(path)

    unique: dict[str, Path] = {}
    expected_root = season_asset_root_path(record.season_id, record.asset_kind)
    for path in candidates:
        if not path.exists():
            continue
        try:
            ensure_inside_generation(path, generation_root)
            path.resolve().relative_to(expected_root.resolve())
        except (RuntimeError, ValueError):
            continue
        unique[str(path.resolve()).lower()] = path
    return prune_nested_archive_candidates(sorted(unique.values(), key=lambda item: (len(item.parts), str(item).lower())))


def archive_season_asset(record: SeasonAssetRecord, *, dry_run: bool = False) -> dict[str, Any]:
    generation_root = resolve_project_path("generation")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_root = generation_root / "archive" / "season_assets" / f"{timestamp}_{record.asset_id}"
    candidates = archive_candidates_for_season_asset(record)
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
        "asset_id": record.asset_id,
        "season_id": record.season_id,
        "asset_kind": record.asset_kind,
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


def delete_season_asset_outputs(record: SeasonAssetRecord, *, dry_run: bool = False) -> dict[str, Any]:
    candidates = archive_candidates_for_season_asset(record)
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
        "asset_id": record.asset_id,
        "season_id": record.season_id,
        "asset_kind": record.asset_kind,
        "display_title": record.display_title,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": dry_run,
        "deleted": deleted,
    }


def print_episode_details(record: EpisodeRecord) -> None:
    print(json.dumps(record.to_dict(), indent=2, ensure_ascii=False))


def print_asset_details(record: SeasonAssetRecord) -> None:
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


def select_asset_record(records: list[SeasonAssetRecord], asset_id: str) -> SeasonAssetRecord:
    for record in records:
        if record.asset_id == asset_id:
            return record
    raise RuntimeError(f"Season asset was not found: {asset_id}")


def selected_or_checked_asset_records(
    records: list[SeasonAssetRecord],
    checked_asset_ids: set[str],
    selected_asset_id: str,
    *,
    fallback_to_single_actionable: bool = False,
) -> list[SeasonAssetRecord]:
    checked = [row for row in records if row.asset_id in checked_asset_ids]
    if checked:
        return checked
    selected = coalesce_text(selected_asset_id)
    if not selected:
        if fallback_to_single_actionable:
            actionable = [
                row
                for row in records
                if archive_candidates_for_season_asset(row)
            ]
            if len(actionable) == 1:
                return actionable
        return []
    selected_rows = [row for row in records if row.asset_id == selected]
    if selected_rows:
        return selected_rows
    if fallback_to_single_actionable:
        actionable = [
            row
            for row in records
            if archive_candidates_for_season_asset(row)
        ]
        if len(actionable) == 1:
            return actionable
    return []


def run_gui(cfg: dict[str, Any]) -> None:
    gui_log("Starting Generated Episode Manager GUI.")
    try:
        import tkinter as tk
        from tkinter import messagebox, simpledialog, ttk
    except Exception as exc:  # pragma: no cover - environment dependent
        gui_log_error("Tk GUI is not available.", exc)
        raise RuntimeError(f"Tk GUI is not available: {exc}") from exc

    root = tk.Tk()
    root.title("Generated Episode Manager")
    root.geometry("1160x720")
    root.minsize(900, 560)
    gui_log("Tk window created.")
    try:
        root.lift()
        root.attributes("-topmost", True)
        root.after(800, lambda: root.attributes("-topmost", False))
        root.focus_force()
    except Exception:
        pass

    records: list[EpisodeRecord] = []
    asset_records: list[SeasonAssetRecord] = []
    project_records: list[ProjectStorageRecord] = []
    checked_episode_ids: set[str] = set()
    checked_asset_ids: set[str] = set()
    checked_project_ids: set[str] = set()
    selected_episode = tk.StringVar(value="")
    selected_asset = tk.StringVar(value="")
    selected_project = tk.StringVar(value="")
    gui_log_file = PROJECT_DIR / "logs" / "gui_manager.log"
    status_text = tk.StringVar(value=f"Starting 25 manager GUI ... Log: {gui_log_file}")
    ui_events: queue.Queue[tuple[str, Any]] = queue.Queue()
    refresh_state = {"running": False}

    main_pane = ttk.PanedWindow(root, orient=tk.VERTICAL)
    live_frame = ttk.LabelFrame(main_pane, text="Live generation")
    notebook = ttk.Notebook(main_pane)
    intro_tab = ttk.Frame(notebook)
    outro_tab = ttk.Frame(notebook)
    episodes_tab = ttk.Frame(notebook)
    project_tab = ttk.Frame(notebook)
    intro_frame = ttk.LabelFrame(intro_tab, text="Intro assets")
    outro_frame = ttk.LabelFrame(outro_tab, text="Outro assets")
    episode_pane = ttk.PanedWindow(episodes_tab, orient=tk.HORIZONTAL)
    list_frame = ttk.LabelFrame(episode_pane, text="Generated episodes")
    detail_frame = ttk.LabelFrame(episode_pane, text="Episode details")
    project_pane = ttk.PanedWindow(project_tab, orient=tk.HORIZONTAL)
    project_list_frame = ttk.LabelFrame(project_pane, text="Project data, imports and databases")
    project_detail_frame = ttk.LabelFrame(project_pane, text="Selected project item")
    project_actions = ttk.Frame(project_tab)

    live_detail = tk.Text(live_frame, height=9, wrap="word")
    live_detail.configure(state="disabled")
    live_scroll = ttk.Scrollbar(live_frame, orient="vertical", command=live_detail.yview)
    live_detail.configure(yscrollcommand=live_scroll.set)
    activity_detail = tk.Text(live_frame, height=5, wrap="word")
    activity_detail.configure(state="disabled")
    activity_scroll = ttk.Scrollbar(live_frame, orient="vertical", command=activity_detail.yview)
    activity_detail.configure(yscrollcommand=activity_scroll.set)

    columns = ("checked", "episode", "title", "ready", "quality", "release", "finished", "scenes", "updated")
    tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=16)
    tree_y_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
    tree_x_scroll = ttk.Scrollbar(list_frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=tree_y_scroll.set, xscrollcommand=tree_x_scroll.set)
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
        tree.column(column, width=widths[column], minwidth=45, anchor="w", stretch=True)
    tree.column("checked", minwidth=58, stretch=False)

    asset_columns = ("checked", "asset", "kind", "season", "status", "duration", "videos", "images", "updated")
    intro_tree = ttk.Treeview(intro_frame, columns=asset_columns, show="headings", height=12)
    intro_y_scroll = ttk.Scrollbar(intro_frame, orient="vertical", command=intro_tree.yview)
    intro_x_scroll = ttk.Scrollbar(intro_frame, orient="horizontal", command=intro_tree.xview)
    intro_tree.configure(yscrollcommand=intro_y_scroll.set, xscrollcommand=intro_x_scroll.set)
    outro_tree = ttk.Treeview(outro_frame, columns=asset_columns, show="headings", height=12)
    outro_y_scroll = ttk.Scrollbar(outro_frame, orient="vertical", command=outro_tree.yview)
    outro_x_scroll = ttk.Scrollbar(outro_frame, orient="horizontal", command=outro_tree.xview)
    outro_tree.configure(yscrollcommand=outro_y_scroll.set, xscrollcommand=outro_x_scroll.set)
    asset_labels = {
        "checked": "Select",
        "asset": "Asset",
        "kind": "Kind",
        "season": "Season",
        "status": "Status",
        "duration": "Duration",
        "videos": "Videos",
        "images": "Images",
        "updated": "Updated",
    }
    asset_widths = {
        "checked": 62,
        "asset": 150,
        "kind": 70,
        "season": 100,
        "status": 150,
        "duration": 90,
        "videos": 70,
        "images": 70,
        "updated": 150,
    }
    for asset_tree in (intro_tree, outro_tree):
        for column in asset_columns:
            asset_tree.heading(column, text=asset_labels[column])
            asset_tree.column(column, width=asset_widths[column], minwidth=45, anchor="w", stretch=True)
        asset_tree.column("checked", minwidth=58, stretch=False)

    project_columns = ("checked", "category", "name", "type", "size", "items", "updated", "access")
    project_tree = ttk.Treeview(project_list_frame, columns=project_columns, show="headings", height=16)
    project_y_scroll = ttk.Scrollbar(project_list_frame, orient="vertical", command=project_tree.yview)
    project_x_scroll = ttk.Scrollbar(project_list_frame, orient="horizontal", command=project_tree.xview)
    project_tree.configure(yscrollcommand=project_y_scroll.set, xscrollcommand=project_x_scroll.set)
    project_labels = {
        "checked": "Select",
        "category": "Category",
        "name": "Name",
        "type": "Type",
        "size": "Size",
        "items": "Items",
        "updated": "Updated",
        "access": "Actions",
    }
    project_widths = {
        "checked": 62,
        "category": 115,
        "name": 220,
        "type": 120,
        "size": 100,
        "items": 75,
        "updated": 150,
        "access": 135,
    }
    for column in project_columns:
        project_tree.heading(column, text=project_labels[column])
        project_tree.column(column, width=project_widths[column], minwidth=45, anchor="w", stretch=True)
    project_tree.column("checked", minwidth=58, stretch=False)

    detail = tk.Text(detail_frame, height=18, wrap="none")
    detail.configure(state="disabled")
    detail_y_scroll = ttk.Scrollbar(detail_frame, orient="vertical", command=detail.yview)
    detail_x_scroll = ttk.Scrollbar(detail_frame, orient="horizontal", command=detail.xview)
    detail.configure(yscrollcommand=detail_y_scroll.set, xscrollcommand=detail_x_scroll.set)
    project_detail = tk.Text(project_detail_frame, height=18, wrap="none")
    project_detail.configure(state="disabled")
    project_detail_y_scroll = ttk.Scrollbar(project_detail_frame, orient="vertical", command=project_detail.yview)
    project_detail_x_scroll = ttk.Scrollbar(project_detail_frame, orient="horizontal", command=project_detail.xview)
    project_detail.configure(yscrollcommand=project_detail_y_scroll.set, xscrollcommand=project_detail_x_scroll.set)

    def set_live_detail(text: str) -> None:
        live_detail.configure(state="normal")
        live_detail.delete("1.0", "end")
        live_detail.insert("1.0", text)
        live_detail.configure(state="disabled")

    def append_activity(message: str, *, log: bool = True) -> None:
        if log:
            gui_log(message)
        activity_detail.configure(state="normal")
        activity_detail.insert("end", f"[{time.strftime('%H:%M:%S')}] {message}\n")
        activity_detail.see("end")
        activity_detail.configure(state="disabled")

    def append_activity_error(message: str, exc: BaseException | str | None = None, *, log: bool = True) -> None:
        if log:
            if isinstance(exc, BaseException):
                gui_log_error(message, exc)
            else:
                gui_log(f"ERROR: {message}")
        activity_detail.configure(state="normal")
        activity_detail.insert("end", f"[{time.strftime('%H:%M:%S')}] ERROR: {message}\n")
        if exc is not None:
            activity_detail.insert("end", f"{exc}\n")
        activity_detail.see("end")
        activity_detail.configure(state="disabled")

    def report_callback_exception(exc_type: type[BaseException], exc: BaseException, tb: object) -> None:
        append_activity_error("Unhandled Tk callback exception.", exc)
        status_text.set(f"GUI callback failed: {exc}")

    root.report_callback_exception = report_callback_exception

    def queue_activity(message: str) -> None:
        gui_log(message)
        ui_events.put(("activity", message))

    def queue_error(message: str, exc: BaseException) -> None:
        gui_log_error(message, exc)
        ui_events.put(("error", {"message": message, "detail": "".join(traceback.format_exception_only(type(exc), exc)).strip()}))

    def refresh_live() -> None:
        started = time.monotonic()
        append_activity("Loading live generation status ...")
        set_live_detail(format_live_status(build_live_generation_status(cfg)))
        append_activity(f"Live generation status loaded in {time.monotonic() - started:.2f}s.")

    def current_record() -> EpisodeRecord | None:
        episode_id = selected_episode.get()
        for item in records:
            if item.episode_id == episode_id:
                return item
        return None

    def checked_records() -> list[EpisodeRecord]:
        return [row for row in records if row.episode_id in checked_episode_ids]

    def active_asset_tree_order() -> list[object]:
        try:
            active_tab_index = notebook.index(notebook.select())
        except Exception:
            active_tab_index = 0
        if active_tab_index == 1:
            return [outro_tree, intro_tree]
        return [intro_tree, outro_tree]

    def asset_tree_for_id(asset_id: str) -> object | None:
        if intro_tree.exists(asset_id):
            return intro_tree
        if outro_tree.exists(asset_id):
            return outro_tree
        return None

    def select_asset_row(asset_id: str, *, switch_tab: bool = True) -> bool:
        asset_id = coalesce_text(asset_id)
        if not asset_id:
            return False
        target_tree = asset_tree_for_id(asset_id)
        if target_tree is None:
            return False
        try:
            target_tree.selection_set(asset_id)
            target_tree.focus(asset_id)
            target_tree.see(asset_id)
        except Exception:
            return False
        if switch_tab:
            notebook.select(intro_tab if target_tree is intro_tree else outro_tab)
        selected_asset.set(asset_id)
        return True

    def selected_asset_id_from_gui() -> str:
        candidates: list[str] = []
        for asset_tree in active_asset_tree_order():
            selection = asset_tree.selection()
            candidates.extend(str(item) for item in selection if str(item))
            focus = str(asset_tree.focus() or "")
            if focus:
                candidates.append(focus)
        stored = selected_asset.get()
        if stored:
            candidates.append(stored)
        valid_ids = {row.asset_id for row in asset_records}
        for candidate in candidates:
            if candidate in valid_ids and asset_tree_for_id(candidate) is not None:
                selected_asset.set(candidate)
                return candidate
        return ""

    def current_asset() -> SeasonAssetRecord | None:
        asset_id = selected_asset_id_from_gui()
        for item in asset_records:
            if item.asset_id == asset_id:
                return item
        return None

    def checked_assets() -> list[SeasonAssetRecord]:
        return [row for row in asset_records if row.asset_id in checked_asset_ids]

    def asset_action_records() -> list[SeasonAssetRecord]:
        return selected_or_checked_asset_records(
            asset_records,
            checked_asset_ids,
            selected_asset_id_from_gui(),
            fallback_to_single_actionable=True,
        )

    def selected_project_id_from_gui() -> str:
        candidates = [str(item) for item in project_tree.selection() if str(item)]
        focus = str(project_tree.focus() or "")
        if focus:
            candidates.append(focus)
        stored = selected_project.get()
        if stored:
            candidates.append(stored)
        valid_ids = {row.record_id for row in project_records}
        for candidate in candidates:
            if candidate in valid_ids and project_tree.exists(candidate):
                selected_project.set(candidate)
                return candidate
        return ""

    def current_project_record() -> ProjectStorageRecord | None:
        record_id = selected_project_id_from_gui()
        for item in project_records:
            if item.record_id == record_id:
                return item
        return None

    def checked_project_records() -> list[ProjectStorageRecord]:
        return [row for row in project_records if row.record_id in checked_project_ids]

    def project_action_records() -> list[ProjectStorageRecord]:
        checked = checked_project_records()
        if checked:
            return checked
        current = current_project_record()
        return [current] if current is not None else []

    def set_detail(text: str) -> None:
        detail.configure(state="normal")
        detail.delete("1.0", "end")
        detail.insert("1.0", text)
        detail.configure(state="disabled")

    def set_project_detail(text: str) -> None:
        project_detail.configure(state="normal")
        project_detail.delete("1.0", "end")
        project_detail.insert("1.0", text)
        project_detail.configure(state="disabled")

    def set_loaded_status() -> None:
        status_text.set(
            f"Loaded {len(records)} generated episode(s), {len(checked_episode_ids)} checked; "
            f"{len(asset_records)} intro/outro asset(s), {len(checked_asset_ids)} checked; "
            f"{len(project_records)} project item(s), {len(checked_project_ids)} checked. "
            f"Last refresh: {format_timestamp(time.time())}"
        )

    def apply_episode_records(new_records: list[EpisodeRecord]) -> None:
        nonlocal records
        previous_selection = selected_episode.get()
        records = list(new_records)
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
        set_loaded_status()
        if records:
            selected = previous_selection if any(row.episode_id == previous_selection for row in records) else records[0].episode_id
            tree.selection_set(selected)
            tree.focus(selected)
            selected_episode.set(selected)
            update_detail()
        else:
            selected_episode.set("")
            set_detail("No generated episodes found.")

    def sync_episode_records() -> None:
        started = time.monotonic()
        append_activity("Scanning generated episode outputs ...")
        new_records = list_episode_records(cfg)
        append_activity(f"Found {len(new_records)} generated episode record(s) in {time.monotonic() - started:.2f}s.")
        apply_episode_records(new_records)

    def apply_asset_records(new_asset_records: list[SeasonAssetRecord]) -> None:
        nonlocal asset_records
        previous_selection = selected_asset.get()
        asset_records = list(new_asset_records)
        valid_ids = {row.asset_id for row in asset_records}
        checked_asset_ids.intersection_update(valid_ids)
        for asset_tree in (intro_tree, outro_tree):
            for item in asset_tree.get_children():
                asset_tree.delete(item)
        for row in asset_records:
            target_tree = intro_tree if row.asset_kind == "intro" else outro_tree
            target_tree.insert(
                "",
                "end",
                iid=row.asset_id,
                values=(
                    "[x]" if row.asset_id in checked_asset_ids else "[ ]",
                    row.asset_id,
                    row.asset_kind,
                    row.season_id,
                    row.status or "-",
                    format_duration(row.duration_seconds) if row.duration_seconds else "-",
                    row.generated_video_count or "-",
                    row.generated_image_count or "-",
                    format_timestamp(row.updated_at) or "-",
                ),
            )
        set_loaded_status()
        if asset_records:
            selected = previous_selection if any(row.asset_id == previous_selection for row in asset_records) else asset_records[0].asset_id
            select_asset_row(selected)
            update_asset_detail()
        else:
            selected_asset.set("")

    def sync_asset_records() -> None:
        started = time.monotonic()
        append_activity("Scanning season intro/outro assets ...")
        new_asset_records = list_season_asset_records(cfg)
        append_activity(f"Found {len(new_asset_records)} intro/outro asset record(s) in {time.monotonic() - started:.2f}s.")
        apply_asset_records(new_asset_records)

    def apply_project_records(new_project_records: list[ProjectStorageRecord]) -> None:
        nonlocal project_records
        previous_selection = selected_project.get()
        project_records = list(new_project_records)
        valid_ids = {row.record_id for row in project_records}
        checked_project_ids.intersection_update(valid_ids)
        for item in project_tree.get_children():
            project_tree.delete(item)
        for row in project_records:
            actions = []
            if row.editable_json:
                actions.append("edit JSON")
            if row.archive_allowed:
                actions.append("archive")
            if row.delete_allowed:
                actions.append("delete")
            project_tree.insert(
                "",
                "end",
                iid=row.record_id,
                values=(
                    "[x]" if row.record_id in checked_project_ids else "[ ]",
                    row.category,
                    row.display_name,
                    row.kind,
                    format_file_size(row.size_bytes),
                    row.item_count or "-",
                    format_timestamp(row.updated_at) or "-",
                    ", ".join(actions) or "view only",
                ),
            )
        set_loaded_status()
        if project_records:
            selected = previous_selection if any(row.record_id == previous_selection for row in project_records) else project_records[0].record_id
            project_tree.selection_set(selected)
            project_tree.focus(selected)
            project_tree.see(selected)
            selected_project.set(selected)
            update_project_detail()
        else:
            selected_project.set("")
            set_project_detail("No managed project data was found yet.")

    def sync_project_records() -> None:
        started = time.monotonic()
        append_activity("Scanning project imports, databases and workspaces ...")
        new_project_records = list_project_storage_records()
        append_activity(f"Found {len(new_project_records)} managed project item(s) in {time.monotonic() - started:.2f}s.")
        apply_project_records(new_project_records)

    def refresh_worker() -> None:
        started = time.monotonic()
        try:
            queue_activity("Refresh started.")
            ui_events.put(("status", "Loading live generation status ..."))

            live_started = time.monotonic()
            queue_activity("Loading live generation status ...")
            live_text = format_live_status(build_live_generation_status(cfg))
            queue_activity(f"Live generation status loaded in {time.monotonic() - live_started:.2f}s.")

            episode_started = time.monotonic()
            ui_events.put(("status", "Loading generated episodes ..."))
            queue_activity("Scanning generated episode outputs ...")
            new_records = list_episode_records(cfg)
            queue_activity(f"Found {len(new_records)} generated episode record(s) in {time.monotonic() - episode_started:.2f}s.")

            asset_started = time.monotonic()
            ui_events.put(("status", "Loading season intros/outros ..."))
            queue_activity("Scanning season intro/outro assets ...")
            new_asset_records = list_season_asset_records(cfg)
            queue_activity(f"Found {len(new_asset_records)} intro/outro asset record(s) in {time.monotonic() - asset_started:.2f}s.")

            project_started = time.monotonic()
            ui_events.put(("status", "Scanning project imports and databases ..."))
            queue_activity("Scanning project imports, databases and workspaces ...")
            new_project_records = list_project_storage_records()
            queue_activity(f"Found {len(new_project_records)} managed project item(s) in {time.monotonic() - project_started:.2f}s.")

            ui_events.put(
                (
                    "refresh_result",
                    {
                        "live_text": live_text,
                        "records": new_records,
                        "asset_records": new_asset_records,
                        "project_records": new_project_records,
                        "elapsed": time.monotonic() - started,
                    },
                )
            )
        except Exception as exc:
            queue_error("GUI refresh failed.", exc)
            ui_events.put(("refresh_error", {"message": str(exc)}))

    def refresh_all() -> None:
        if refresh_state["running"]:
            append_activity("Refresh already running; please wait.")
            return
        refresh_state["running"] = True
        status_text.set("Refresh queued ...")
        append_activity("Refresh queued in background.")
        threading.Thread(target=refresh_worker, name="step25-refresh", daemon=True).start()

    def process_ui_events() -> None:
        try:
            processed = 0
            max_events_per_tick = 12
            while processed < max_events_per_tick:
                try:
                    kind, payload = ui_events.get_nowait()
                except queue.Empty:
                    break
                processed += 1
                if kind == "activity":
                    append_activity(str(payload), log=False)
                elif kind == "error":
                    if isinstance(payload, dict):
                        append_activity_error(str(payload.get("message", "Error")), str(payload.get("detail", "")), log=False)
                elif kind == "status":
                    status_text.set(str(payload))
                elif kind == "refresh_result" and isinstance(payload, dict):
                    try:
                        set_live_detail(str(payload.get("live_text", "")))
                        apply_episode_records(list(payload.get("records", [])))
                        apply_asset_records(list(payload.get("asset_records", [])))
                        apply_project_records(list(payload.get("project_records", [])))
                        refresh_state["running"] = False
                        elapsed = as_float(payload.get("elapsed"))
                        append_activity(f"Refresh finished in {elapsed:.2f}s.")
                        set_loaded_status()
                    except Exception as exc:
                        refresh_state["running"] = False
                        append_activity_error("GUI refresh result could not be applied.", exc)
                        status_text.set(f"Refresh failed while applying results: {exc}")
                elif kind == "refresh_error":
                    refresh_state["running"] = False
                    message = str(payload.get("message", payload)) if isinstance(payload, dict) else str(payload)
                    status_text.set(f"Refresh failed: {message}")
                    set_detail(f"Refresh failed:\n\n{message}\n\nSee terminal/log output for the full traceback.")
                    messagebox.showerror("Refresh failed", f"{message}\n\nSee terminal/log output for details.")
        except Exception as exc:
            gui_log_error("UI event polling failed.", exc)
            refresh_state["running"] = False
            status_text.set(f"UI event polling failed: {exc}")
        finally:
            delay_ms = 10 if not ui_events.empty() else 150
            root.after(delay_ms, process_ui_events)

    def heartbeat() -> None:
        if refresh_state["running"]:
            status_text.set(f"Refresh running ... {format_timestamp(time.time())}")
        root.after(1000, heartbeat)

    def redraw_checked_column() -> None:
        for row in records:
            if tree.exists(row.episode_id):
                values = list(tree.item(row.episode_id, "values"))
                if values:
                    values[0] = "[x]" if row.episode_id in checked_episode_ids else "[ ]"
                    tree.item(row.episode_id, values=values)
        set_loaded_status()

    def redraw_asset_checked_column() -> None:
        for row in asset_records:
            target_tree = intro_tree if intro_tree.exists(row.asset_id) else outro_tree if outro_tree.exists(row.asset_id) else None
            if target_tree is not None:
                values = list(target_tree.item(row.asset_id, "values"))
                if values:
                    values[0] = "[x]" if row.asset_id in checked_asset_ids else "[ ]"
                    target_tree.item(row.asset_id, values=values)
        set_loaded_status()

    def redraw_project_checked_column() -> None:
        for row in project_records:
            if project_tree.exists(row.record_id):
                values = list(project_tree.item(row.record_id, "values"))
                if values:
                    values[0] = "[x]" if row.record_id in checked_project_ids else "[ ]"
                    project_tree.item(row.record_id, values=values)
        set_loaded_status()

    def toggle_checked(episode_id: str) -> None:
        if episode_id in checked_episode_ids:
            checked_episode_ids.remove(episode_id)
        else:
            checked_episode_ids.add(episode_id)
        redraw_checked_column()

    def select_all() -> None:
        append_activity("Select All pressed: checking episodes, intro/outro assets and manageable project data.")
        checked_episode_ids.clear()
        checked_episode_ids.update(row.episode_id for row in records)
        checked_asset_ids.clear()
        checked_asset_ids.update(row.asset_id for row in asset_records)
        checked_project_ids.clear()
        checked_project_ids.update(
            row.record_id for row in project_records if row.archive_allowed or row.delete_allowed
        )
        redraw_checked_column()
        redraw_asset_checked_column()
        redraw_project_checked_column()

    def clear_checks() -> None:
        append_activity("Clear Checks pressed: clearing episodes, intro/outro assets and project data.")
        checked_episode_ids.clear()
        checked_asset_ids.clear()
        checked_project_ids.clear()
        redraw_checked_column()
        redraw_asset_checked_column()
        redraw_project_checked_column()

    def toggle_asset_checked(asset_id: str) -> None:
        select_asset_row(asset_id)
        if asset_id in checked_asset_ids:
            checked_asset_ids.remove(asset_id)
        else:
            checked_asset_ids.add(asset_id)
        redraw_asset_checked_column()
        update_asset_detail()

    def toggle_project_checked(record_id: str) -> None:
        if record_id in checked_project_ids:
            checked_project_ids.remove(record_id)
        else:
            checked_project_ids.add(record_id)
        redraw_project_checked_column()
        update_project_detail()

    def on_tree_click(event: object) -> None:
        region = tree.identify("region", getattr(event, "x", 0), getattr(event, "y", 0))
        if region != "cell":
            return
        column = tree.identify_column(getattr(event, "x", 0))
        item_id = tree.identify_row(getattr(event, "y", 0))
        if column == "#1" and item_id:
            toggle_checked(str(item_id))

    def on_asset_tree_click(asset_tree: object, event: object) -> None:
        region = asset_tree.identify("region", getattr(event, "x", 0), getattr(event, "y", 0))
        if region != "cell":
            return
        column = asset_tree.identify_column(getattr(event, "x", 0))
        item_id = asset_tree.identify_row(getattr(event, "y", 0))
        if item_id:
            select_asset_row(str(item_id))
        if column == "#1" and item_id:
            toggle_asset_checked(str(item_id))
        elif item_id:
            update_asset_detail()

    def on_asset_tree_release(asset_tree: object, event: object) -> None:
        item_id = asset_tree.identify_row(getattr(event, "y", 0))
        if not item_id:
            return
        select_asset_row(str(item_id))
        update_asset_detail()

    def on_project_tree_click(event: object) -> None:
        region = project_tree.identify("region", getattr(event, "x", 0), getattr(event, "y", 0))
        if region != "cell":
            return
        column = project_tree.identify_column(getattr(event, "x", 0))
        item_id = str(project_tree.identify_row(getattr(event, "y", 0)) or "")
        if item_id:
            project_tree.selection_set(item_id)
            project_tree.focus(item_id)
            selected_project.set(item_id)
        if column == "#1" and item_id:
            toggle_project_checked(item_id)
        elif item_id:
            update_project_detail()

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

    def update_asset_detail(_event: object | None = None) -> None:
        selected_asset_id_from_gui()
        row = current_asset()
        if not row:
            set_detail("No intro/outro asset selected.")
            return
        parts = [
            f"{row.display_title} ({row.asset_id})",
            "",
            f"Season: {row.season_id}",
            f"Kind: {row.asset_kind}",
            f"Status: {row.status or '-'}",
            f"Duration: {format_duration(row.duration_seconds) if row.duration_seconds else '-'}",
            f"Generated videos: {row.generated_video_count} | images: {row.generated_image_count}",
            f"Fallback used: {row.fallback_used} | placeholder used: {row.placeholder_used}",
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

    def update_project_detail(_event: object | None = None) -> None:
        record_id = selected_project_id_from_gui()
        if record_id:
            selected_project.set(record_id)
        row = current_project_record()
        if not row:
            set_project_detail("No project data item selected.")
            return
        parts = [
            f"{row.category}: {row.display_name}",
            "",
            f"Type: {row.kind}",
            f"Path: {row.relative_path}",
            f"Size: {format_file_size(row.size_bytes)}",
            f"Items: {row.item_count}",
            f"Updated: {format_timestamp(row.updated_at) or '-'}",
            f"JSON editing: {'available' if row.editable_json else 'not available'}",
            f"Archive: {'available' if row.archive_allowed else 'not available'}",
            f"Delete: {'available' if row.delete_allowed else 'not available'}",
            "",
            f"Purpose: {row.note or '-'}",
        ]
        if row.editable_json:
            try:
                summary = json_database_summary(row)
                parts.extend(["", f"Database summary: {summary.get('summary', '-')}"])
            except Exception as exc:
                parts.extend(["", f"Database validation warning: {exc}"])
        parts.extend(
            [
                "",
                "Safety: destructive actions are limited to ai_series_project and are blocked while a pipeline run is active.",
                "JSON edits are backed up under runtime/gui_backups/json before saving.",
            ]
        )
        set_project_detail("\n".join(parts))

    def on_notebook_tab_changed(_event: object | None = None) -> None:
        try:
            active_tab_index = notebook.index(notebook.select())
        except Exception:
            active_tab_index = 0
        if active_tab_index in (0, 1):
            selected_asset_id_from_gui()
            update_asset_detail()
        elif active_tab_index == 2:
            update_detail()
        else:
            update_project_detail()

    def active_tab_is_asset_tab() -> bool:
        try:
            return notebook.index(notebook.select()) in (0, 1)
        except Exception:
            return bool(checked_asset_ids)

    def active_tab_is_project_tab() -> bool:
        try:
            return notebook.index(notebook.select()) == 3
        except Exception:
            return bool(checked_project_ids)

    def generic_checked_action_should_use_assets() -> bool:
        if active_tab_is_project_tab():
            return False
        if active_tab_is_asset_tab():
            return True
        return bool(checked_asset_ids) and not bool(checked_episode_ids)

    def open_target(target: str) -> None:
        append_activity(f"Open episode target requested: {target}.")
        row = current_record()
        if not row:
            append_activity("Open episode target ignored: no episode selected.")
            messagebox.showwarning("No episode", "Select an episode first.")
            return
        path = target_path_for_record(row, target)
        if not path:
            append_activity(f"Open episode target missing: {target} for {row.episode_id}.")
            messagebox.showwarning("Missing target", f"No existing {target} target was found for {row.episode_id}.")
            return
        append_activity(f"Opening episode target for {row.episode_id}: {path}")
        open_path(path)

    def open_asset_target(target: str) -> None:
        append_activity(f"Open intro/outro target requested: {target}.")
        rows = asset_action_records()
        row = rows[0] if rows else current_asset()
        if not row:
            append_activity("Open intro/outro target ignored: no asset selected.")
            messagebox.showwarning("No intro/outro asset", "Select an intro/outro asset first.")
            return
        path = target_path_for_asset(row, target)
        if not path:
            append_activity(f"Open intro/outro target missing: {target} for {row.asset_id}.")
            messagebox.showwarning("Missing target", f"No existing {target} target was found for {row.asset_id}.")
            return
        append_activity(f"Opening intro/outro target for {row.asset_id}: {path}")
        open_path(path)

    def project_storage_is_busy() -> bool:
        status = build_live_generation_status(cfg)
        return bool(status.active)

    def require_project_storage_idle(action: str) -> bool:
        if not project_storage_is_busy():
            return True
        message = (
            f"{action} is blocked while the pipeline is active. Wait for the current step to finish, "
            "then refresh this manager and try again."
        )
        append_activity(message)
        messagebox.showwarning("Pipeline is active", message)
        return False

    def open_project_target(target: str) -> None:
        row = current_project_record()
        if not row:
            messagebox.showwarning("No project item", "Select a project item first.")
            return
        path = Path(row.path)
        project_relative_path(path)
        if target == "folder" and path.is_file():
            path = path.parent
        if not path.exists():
            messagebox.showwarning("Missing project item", f"The selected project item no longer exists:\n{row.relative_path}")
            return
        append_activity(f"Opening project {target}: {row.relative_path}")
        open_path(path)

    def validate_project_json() -> None:
        row = current_project_record()
        if not row or not row.editable_json:
            messagebox.showwarning("No JSON database", "Select an editable JSON database first.")
            return
        try:
            summary = json_database_summary(row)
        except Exception as exc:
            append_activity_error(f"JSON validation failed for {row.relative_path}.", exc)
            messagebox.showerror("Invalid JSON", f"{row.relative_path}\n\n{exc}")
            return
        append_activity(f"JSON database validated: {row.relative_path} ({summary.get('summary', '-')})")
        messagebox.showinfo("JSON valid", f"{row.relative_path}\n\n{summary.get('summary', 'Valid JSON.')}")
        update_project_detail()

    def edit_project_json() -> None:
        row = current_project_record()
        if not row or not row.editable_json:
            messagebox.showwarning("No JSON database", "Select an editable JSON database first.")
            return
        if not require_project_storage_idle("Editing project databases"):
            return
        try:
            current_payload = read_json_database(row)
            initial_text = json.dumps(current_payload, ensure_ascii=False, indent=2) + "\n"
        except Exception as exc:
            append_activity_error(f"Could not open JSON database {row.relative_path}.", exc)
            messagebox.showerror("Open JSON failed", f"{row.relative_path}\n\n{exc}")
            return

        editor = tk.Toplevel(root)
        editor.title(f"Edit database - {row.relative_path}")
        editor.geometry("900x640")
        editor.minsize(620, 420)
        ttk.Label(
            editor,
            text=(
                f"{row.relative_path}\n"
                "Saving validates JSON and writes a project-local backup first. Configuration templates are intentionally read-only."
            ),
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 6))
        editor_text = tk.Text(editor, wrap="none", undo=True)
        editor_y_scroll = ttk.Scrollbar(editor, orient="vertical", command=editor_text.yview)
        editor_x_scroll = ttk.Scrollbar(editor, orient="horizontal", command=editor_text.xview)
        editor_text.configure(yscrollcommand=editor_y_scroll.set, xscrollcommand=editor_x_scroll.set)
        editor_text.insert("1.0", initial_text)
        editor_text.grid(row=1, column=0, sticky="nsew", padx=(10, 0), pady=(0, 8))
        editor_y_scroll.grid(row=1, column=1, sticky="ns", pady=(0, 8))
        editor_x_scroll.grid(row=2, column=0, sticky="ew", padx=(10, 0), pady=(0, 8))

        def validate_editor() -> None:
            try:
                json.loads(editor_text.get("1.0", "end-1c"))
            except json.JSONDecodeError as exc:
                messagebox.showerror("Invalid JSON", f"Line {exc.lineno}, column {exc.colno}: {exc.msg}", parent=editor)
                return
            messagebox.showinfo("JSON valid", "The editor content is valid JSON.", parent=editor)

        def save_editor() -> None:
            if not require_project_storage_idle("Saving project databases"):
                return
            try:
                result = save_json_database(row, editor_text.get("1.0", "end-1c"))
            except Exception as exc:
                append_activity_error(f"Could not save JSON database {row.relative_path}.", exc)
                messagebox.showerror("Save failed", str(exc), parent=editor)
                return
            append_activity(f"Saved JSON database {row.relative_path}; backup: {result.get('backup_path', '-')}")
            messagebox.showinfo(
                "JSON saved",
                f"Saved {row.relative_path}.\n\nBackup:\n{result.get('backup_path', '-')}",
                parent=editor,
            )
            editor.destroy()
            refresh_all()

        ttk.Button(editor, text="Validate JSON", command=safe_command("Validate JSON editor", validate_editor)).grid(row=3, column=0, sticky="w", padx=10, pady=(0, 10))
        ttk.Button(editor, text="Save with Backup", command=safe_command("Save JSON editor", save_editor)).grid(row=3, column=1, sticky="e", padx=6, pady=(0, 10))
        ttk.Button(editor, text="Cancel", command=editor.destroy).grid(row=3, column=2, sticky="e", padx=(0, 10), pady=(0, 10))
        editor.columnconfigure(0, weight=1)
        editor.rowconfigure(1, weight=1)

    def archive_selected_project_data() -> None:
        selected_rows = project_action_records()
        candidates = project_storage_mutation_paths(selected_rows, operation="archive")
        if not candidates:
            messagebox.showwarning("Nothing to archive", "Select or check one or more archivable project items first.")
            return
        if not require_project_storage_idle("Archiving project data"):
            return
        if not messagebox.askyesno(
            "Archive selected project data",
            f"Archive {len(candidates)} selected project path(s)?\n\n"
            "Files will be moved inside ai_series_project/archives/gui_project_data and can be inspected later.",
        ):
            return
        result = archive_project_storage_records(selected_rows)
        append_activity(f"Archived {len(result.get('moved', []))} project path(s) to {result.get('archive_root', '-')}")
        messagebox.showinfo("Project data archived", f"Archived {len(result.get('moved', []))} path(s).")
        checked_project_ids.clear()
        refresh_all()

    def delete_selected_project_data() -> None:
        selected_rows = project_action_records()
        candidates = project_storage_mutation_paths(selected_rows, operation="delete")
        if not candidates:
            messagebox.showwarning("Nothing to delete", "Select or check one or more deletable project items first.")
            return
        if not require_project_storage_idle("Deleting project data"):
            return
        typed = simpledialog.askstring(
            "Confirm project data deletion",
            f"This permanently deletes {len(candidates)} project path(s).\n\nType DELETE to confirm.",
            parent=root,
        )
        if typed != "DELETE":
            messagebox.showinfo("Delete cancelled", "Confirmation did not match. Nothing was deleted.")
            return
        result = delete_project_storage_records(selected_rows)
        append_activity(f"Deleted {len(result.get('deleted', []))} project path(s).")
        messagebox.showinfo("Project data deleted", f"Deleted {len(result.get('deleted', []))} path(s).")
        checked_project_ids.clear()
        refresh_all()

    def archive_checked() -> None:
        if active_tab_is_project_tab():
            append_activity("Archive Checked routed to managed project data because the Project tab is active.")
            archive_selected_project_data()
            return
        if generic_checked_action_should_use_assets():
            append_activity("Archive Checked routed to intro/outro assets because the active tab or checked rows are intro/outro.")
            archive_checked_assets()
            return
        append_activity("Archive checked episodes requested.")
        selected_rows = checked_records()
        if not selected_rows:
            append_activity("Archive checked episodes ignored: no episodes checked.")
            messagebox.showwarning("No checked episodes", "Check one or more generated episodes first.")
            return
        candidates_by_episode = {row.episode_id: archive_candidates_for_episode(row) for row in selected_rows}
        total_candidates = sum(len(items) for items in candidates_by_episode.values())
        append_activity(f"Archive checked episodes candidates: {len(selected_rows)} episode(s), {total_candidates} path(s).")
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
        append_activity(f"Archived {len(manifests)} generated episode(s).")
        messagebox.showinfo("Archived", f"Archived {len(manifests)} generated episode(s).")
        checked_episode_ids.clear()
        sync_episode_records()

    def delete_checked() -> None:
        if active_tab_is_project_tab():
            append_activity("Delete Checked routed to managed project data because the Project tab is active.")
            delete_selected_project_data()
            return
        if generic_checked_action_should_use_assets():
            append_activity("Delete Checked routed to intro/outro assets because the active tab or checked rows are intro/outro.")
            delete_checked_assets()
            return
        append_activity("Delete checked episodes requested.")
        selected_rows = checked_records()
        if not selected_rows:
            append_activity("Delete checked episodes ignored: no episodes checked.")
            messagebox.showwarning("No checked episodes", "Check one or more generated episodes first.")
            return
        candidates_by_episode = {row.episode_id: archive_candidates_for_episode(row) for row in selected_rows}
        total_candidates = sum(len(items) for items in candidates_by_episode.values())
        append_activity(f"Delete checked episodes candidates: {len(selected_rows)} episode(s), {total_candidates} path(s).")
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
        append_activity(f"Deleted {deleted_count} generated episode path(s).")
        messagebox.showinfo("Deleted", f"Deleted {deleted_count} generated path(s) for {len(results)} episode(s).")
        checked_episode_ids.clear()
        sync_episode_records()

    def archive_checked_assets() -> None:
        append_activity("Archive selected intro/outro assets requested.")
        selected_rows = asset_action_records()
        if not selected_rows:
            append_activity("Archive selected intro/outro assets ignored: no asset selected or checked.")
            messagebox.showwarning("No intro/outro assets", "Select or check one or more intro/outro assets first.")
            return
        candidates_by_asset = {row.asset_id: archive_candidates_for_season_asset(row) for row in selected_rows}
        total_candidates = sum(len(items) for items in candidates_by_asset.values())
        append_activity(f"Archive intro/outro candidates: {len(selected_rows)} asset(s), {total_candidates} path(s).")
        if not total_candidates:
            messagebox.showinfo("Nothing to archive", "No generated intro/outro files were found for the selected assets.")
            return
        if not messagebox.askyesno(
            "Archive checked intro/outro assets",
            f"Archive {total_candidates} generated path(s) for {len(selected_rows)} selected intro/outro asset(s)?\n\n"
            "This moves season assets to generation/archive/season_assets and can be inspected later.",
        ):
            return
        manifests = [archive_season_asset(row) for row in selected_rows if candidates_by_asset.get(row.asset_id)]
        append_activity(f"Archived {len(manifests)} intro/outro asset(s).")
        messagebox.showinfo("Archived", f"Archived {len(manifests)} intro/outro asset(s).")
        checked_asset_ids.clear()
        sync_asset_records()

    def delete_checked_assets() -> None:
        append_activity("Delete selected intro/outro assets requested.")
        selected_rows = asset_action_records()
        if not selected_rows:
            append_activity("Delete selected intro/outro assets ignored: no asset selected or checked.")
            messagebox.showwarning("No intro/outro assets", "Select or check one or more intro/outro assets first.")
            return
        candidates_by_asset = {row.asset_id: archive_candidates_for_season_asset(row) for row in selected_rows}
        total_candidates = sum(len(items) for items in candidates_by_asset.values())
        append_activity(f"Delete intro/outro candidates: {len(selected_rows)} asset(s), {total_candidates} path(s).")
        if not total_candidates:
            messagebox.showinfo("Nothing to delete", "No generated intro/outro files were found for the selected assets.")
            return
        if not messagebox.askyesno(
            "Delete checked intro/outro assets",
            f"Delete {total_candidates} generated path(s) for {len(selected_rows)} selected intro/outro asset(s)?\n\n"
            "This permanently removes generated season assets only. Configured source intro videos are not touched.",
        ):
            return
        typed = simpledialog.askstring(
            "Confirm delete",
            "Type DELETE to permanently delete generated outputs for all checked intro/outro assets.",
            parent=root,
        )
        if typed != "DELETE":
            messagebox.showinfo("Delete cancelled", "Confirmation did not match. Nothing was deleted.")
            return
        results = [delete_season_asset_outputs(row) for row in selected_rows if candidates_by_asset.get(row.asset_id)]
        deleted_count = sum(len(result["deleted"]) for result in results)
        append_activity(f"Deleted {deleted_count} intro/outro asset path(s).")
        messagebox.showinfo("Deleted", f"Deleted {deleted_count} generated path(s) for {len(results)} intro/outro asset(s).")
        checked_asset_ids.clear()
        sync_asset_records()

    def open_live_status_file() -> None:
        append_activity("Open live status file requested.")
        status = build_live_generation_status(cfg)
        target = existing_path(status.current_status_path) or existing_path(status.current_status_markdown_path)
        if not target:
            append_activity("Open live status file ignored: no status file found.")
            messagebox.showwarning("Missing status", "No live status file was found yet.")
            return
        append_activity(f"Opening live status file: {target}")
        open_path(target)

    def safe_command(label: str, callback: object) -> Any:
        def wrapped(*args: Any) -> Any:
            try:
                return callback(*args)  # type: ignore[misc]
            except Exception as exc:
                append_activity_error(f"{label} failed.", exc)
                status_text.set(f"{label} failed: {exc}")
                try:
                    messagebox.showerror(f"{label} failed", f"{exc}\n\nSee GUI activity log and terminal/log file.")
                except Exception:
                    pass
                return None

        return wrapped

    tree.bind("<<TreeviewSelect>>", safe_command("Episode selection", update_detail))
    tree.bind("<Button-1>", safe_command("Episode table click", on_tree_click), add="+")
    for asset_tree in (intro_tree, outro_tree):
        asset_tree.bind("<<TreeviewSelect>>", safe_command("Intro/outro selection", update_asset_detail))
        asset_tree.bind(
            "<ButtonRelease-1>",
            safe_command(
                "Intro/outro table click",
                lambda event, current_asset_tree=asset_tree: on_asset_tree_click(current_asset_tree, event),
            ),
            add="+",
        )
    project_tree.bind("<<TreeviewSelect>>", safe_command("Project selection", update_project_detail))
    project_tree.bind("<Button-1>", safe_command("Project table click", on_project_tree_click), add="+")
    notebook.bind("<<NotebookTabChanged>>", safe_command("Tab change", on_notebook_tab_changed))
    notebook.add(intro_tab, text="Intro")
    notebook.add(outro_tab, text="Outro")
    notebook.add(episodes_tab, text="Folgen")
    notebook.add(project_tab, text="Projekt")
    main_pane.grid(row=0, column=0, columnspan=8, sticky="nsew", padx=10, pady=(10, 4))
    main_pane.add(live_frame)
    main_pane.add(notebook)
    episode_pane.add(detail_frame)
    episode_pane.add(list_frame)

    live_detail.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
    live_scroll.grid(row=0, column=1, sticky="ns", pady=6)
    ttk.Button(live_frame, text="Open Status", command=safe_command("Open Status", open_live_status_file)).grid(row=0, column=2, rowspan=2, sticky="nsew", padx=6, pady=6)
    activity_detail.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))
    activity_scroll.grid(row=1, column=1, sticky="ns", pady=(0, 6))
    live_frame.columnconfigure(0, weight=1)
    live_frame.columnconfigure(1, weight=0)
    live_frame.columnconfigure(2, weight=0)
    live_frame.rowconfigure(0, weight=1)
    live_frame.rowconfigure(1, weight=1)

    tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=(6, 0))
    tree_y_scroll.grid(row=0, column=1, sticky="ns", pady=(6, 0))
    tree_x_scroll.grid(row=1, column=0, sticky="ew", padx=(6, 0), pady=(0, 6))
    list_frame.columnconfigure(0, weight=1)
    list_frame.rowconfigure(0, weight=1)

    intro_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
    intro_tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=(6, 0))
    intro_y_scroll.grid(row=0, column=1, sticky="ns", pady=(6, 0))
    intro_x_scroll.grid(row=1, column=0, sticky="ew", padx=(6, 0), pady=(0, 6))
    intro_tab.columnconfigure(0, weight=1)
    intro_tab.rowconfigure(0, weight=1)
    intro_frame.columnconfigure(0, weight=1)
    intro_frame.rowconfigure(0, weight=1)

    outro_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
    outro_tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=(6, 0))
    outro_y_scroll.grid(row=0, column=1, sticky="ns", pady=(6, 0))
    outro_x_scroll.grid(row=1, column=0, sticky="ew", padx=(6, 0), pady=(0, 6))
    outro_tab.columnconfigure(0, weight=1)
    outro_tab.rowconfigure(0, weight=1)
    outro_frame.columnconfigure(0, weight=1)
    outro_frame.rowconfigure(0, weight=1)

    episode_pane.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
    episodes_tab.columnconfigure(0, weight=1)
    episodes_tab.rowconfigure(0, weight=1)

    detail.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=(6, 0))
    detail_y_scroll.grid(row=0, column=1, sticky="ns", pady=(6, 0))
    detail_x_scroll.grid(row=1, column=0, sticky="ew", padx=(6, 0), pady=(0, 6))
    detail_frame.columnconfigure(0, weight=1)
    detail_frame.rowconfigure(0, weight=1)

    project_pane.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
    project_pane.add(project_list_frame)
    project_pane.add(project_detail_frame)
    project_tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=(6, 0))
    project_y_scroll.grid(row=0, column=1, sticky="ns", pady=(6, 0))
    project_x_scroll.grid(row=1, column=0, sticky="ew", padx=(6, 0), pady=(0, 6))
    project_list_frame.columnconfigure(0, weight=1)
    project_list_frame.rowconfigure(0, weight=1)
    project_detail.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=(6, 0))
    project_detail_y_scroll.grid(row=0, column=1, sticky="ns", pady=(6, 0))
    project_detail_x_scroll.grid(row=1, column=0, sticky="ew", padx=(6, 0), pady=(0, 6))
    project_detail_frame.columnconfigure(0, weight=1)
    project_detail_frame.rowconfigure(0, weight=1)
    ttk.Button(project_actions, text="Refresh Project Data", command=safe_command("Refresh", refresh_all)).grid(row=0, column=0, sticky="ew", padx=3, pady=3)
    ttk.Button(
        project_actions,
        text="Select Manageable",
        command=safe_command(
            "Select Project Data",
            lambda: (
                checked_project_ids.clear(),
                checked_project_ids.update(row.record_id for row in project_records if row.archive_allowed or row.delete_allowed),
                redraw_project_checked_column(),
            ),
        ),
    ).grid(row=0, column=1, sticky="ew", padx=3, pady=3)
    ttk.Button(project_actions, text="Clear Project Checks", command=safe_command("Clear Project Checks", lambda: (checked_project_ids.clear(), redraw_project_checked_column()))).grid(row=0, column=2, sticky="ew", padx=3, pady=3)
    ttk.Button(project_actions, text="Open Item", command=safe_command("Open Project Item", lambda: open_project_target("item"))).grid(row=0, column=3, sticky="ew", padx=3, pady=3)
    ttk.Button(project_actions, text="Open Folder", command=safe_command("Open Project Folder", lambda: open_project_target("folder"))).grid(row=0, column=4, sticky="ew", padx=3, pady=3)
    ttk.Button(project_actions, text="Validate JSON", command=safe_command("Validate Project JSON", validate_project_json)).grid(row=1, column=0, sticky="ew", padx=3, pady=3)
    ttk.Button(project_actions, text="Edit JSON", command=safe_command("Edit Project JSON", edit_project_json)).grid(row=1, column=1, sticky="ew", padx=3, pady=3)
    ttk.Button(project_actions, text="Archive Selected", command=safe_command("Archive Project Data", archive_selected_project_data)).grid(row=1, column=2, columnspan=2, sticky="ew", padx=3, pady=3)
    ttk.Button(project_actions, text="Delete Selected", command=safe_command("Delete Project Data", delete_selected_project_data)).grid(row=1, column=4, sticky="ew", padx=3, pady=3)
    for index in range(5):
        project_actions.columnconfigure(index, weight=1)
    project_actions.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
    project_tab.columnconfigure(0, weight=1)
    project_tab.rowconfigure(0, weight=1)

    ttk.Button(root, text="Refresh", command=safe_command("Refresh", refresh_all)).grid(row=1, column=0, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Select All", command=safe_command("Select All", select_all)).grid(row=1, column=1, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Clear Checks", command=safe_command("Clear Checks", clear_checks)).grid(row=1, column=2, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Open Video", command=safe_command("Open Video", lambda: open_target("best-video"))).grid(row=1, column=3, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Open Folder", command=safe_command("Open Folder", lambda: open_target("folder"))).grid(row=1, column=4, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Quality Report", command=safe_command("Quality Report", lambda: open_target("quality"))).grid(row=1, column=5, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Realism Report", command=safe_command("Realism Report", lambda: open_target("realism"))).grid(row=1, column=6, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Production Package", command=safe_command("Production Package", lambda: open_target("package"))).grid(row=1, column=7, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Archive Checked Episodes / Active Tab", command=safe_command("Archive Checked", archive_checked)).grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Delete Checked Episodes / Active Tab", command=safe_command("Delete Checked", delete_checked)).grid(row=2, column=3, columnspan=3, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Close", command=root.destroy).grid(row=2, column=6, columnspan=2, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Open Intro/Outro Video", command=safe_command("Open Intro/Outro Video", lambda: open_asset_target("video"))).grid(row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Open Intro/Outro Folder", command=safe_command("Open Intro/Outro Folder", lambda: open_asset_target("folder"))).grid(row=3, column=2, columnspan=2, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Open Intro/Outro Manifest", command=safe_command("Open Intro/Outro Manifest", lambda: open_asset_target("manifest"))).grid(row=3, column=4, columnspan=2, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Archive Checked Intro/Outro", command=safe_command("Archive Selected Assets", archive_checked_assets)).grid(row=4, column=0, columnspan=4, sticky="ew", padx=6, pady=6)
    ttk.Button(root, text="Delete Checked Intro/Outro", command=safe_command("Delete Selected Assets", delete_checked_assets)).grid(row=4, column=4, columnspan=4, sticky="ew", padx=6, pady=6)
    ttk.Label(root, textvariable=status_text).grid(row=5, column=0, columnspan=8, sticky="w", padx=10, pady=(0, 8))

    root.columnconfigure(0, weight=1)
    for index in range(1, 8):
        root.columnconfigure(index, weight=1)
    root.rowconfigure(0, weight=1)

    set_live_detail(
        "Starting GUI...\n\n"
        "The first refresh will begin after the window is visible.\n\n"
        f"GUI log: {gui_log_file}"
    )
    set_detail(
        "Starting GUI...\n\n"
        "Loading status, generated episodes, and season intro/outro assets shortly.\n\n"
        f"GUI log: {gui_log_file}"
    )
    set_project_detail(
        "Starting project data management...\n\n"
        "Loading imported media, JSON databases, training, generation, exports, logs and project-local runtime data shortly.\n\n"
        "JSON edits create a backup before saving. Archive and delete are blocked while the pipeline is active."
    )
    gui_log("GUI layout ready. Scheduling initial refresh.")
    root.after(50, process_ui_events)
    root.after(250, heartbeat)
    root.after(100, refresh_all)
    gui_log("Entering Tk main loop.")
    root.mainloop()
    gui_log("Tk main loop exited.")


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
    parser.add_argument("--assets", action="store_true", help="List managed season intro/outro assets instead of generated episodes.")
    parser.add_argument("--asset-id", help="Target one season intro/outro asset, for example season_01_intro.")
    parser.add_argument("--open-asset", dest="open_asset_target", choices=sorted(ASSET_OPEN_TARGETS), help="Open a season intro/outro artifact.")
    parser.add_argument("--archive-asset", action="store_true", help="Move active generated outputs for --asset-id into generation/archive/season_assets.")
    parser.add_argument("--delete-asset", action="store_true", help="Permanently delete active generated outputs for --asset-id from generation/.")
    parser.add_argument("--dry-run", action="store_true", help="Show what archive/delete would change without changing anything.")
    parser.add_argument("--confirm", help="Required for --archive or --delete. Must equal --episode-id.")
    return parser.parse_args(argv)


def should_open_gui_by_default(argv: list[str] | None = None) -> bool:
    raw_args = sys.argv[1:] if argv is None else list(argv)
    return len(raw_args) == 0


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if should_open_gui_by_default(argv):
        args.gui = True

    if args.gui:
        gui_log("Loading manager configuration ...")
    cfg = load_manager_config()
    if args.gui:
        gui_log("Manager configuration loaded.")

    if args.gui:
        run_gui(cfg)
        return

    if args.watch:
        watch_live_status(cfg, float(args.update_seconds or WATCH_DEFAULT_SECONDS), json_output=bool(args.print_json))
        return

    if args.live:
        headline("Generated Episode Live Monitor")
        print_live_status(cfg, json_output=bool(args.print_json))
        if not (
            args.list
            or args.episode_id
            or args.open_target
            or args.archive
            or args.delete
            or args.assets
            or args.asset_id
            or args.open_asset_target
            or args.archive_asset
            or args.delete_asset
        ):
            return

    asset_requested = bool(args.assets or args.asset_id or args.open_asset_target or args.archive_asset or args.delete_asset)
    episode_requested = bool((args.list and not args.assets) or args.episode_id or args.open_target or args.archive or args.delete)

    if asset_requested and not episode_requested:
        headline("Manage Season Intros/Outros")
        asset_records = list_season_asset_records(cfg)
        if args.asset_id:
            asset_records = [select_asset_record(asset_records, args.asset_id)]

        if args.print_json:
            print(json.dumps([row.to_dict() for row in asset_records], indent=2, ensure_ascii=False))
        else:
            if not asset_records:
                info("No managed intro/outro assets were found.")
            else:
                print(format_asset_table(asset_records))

        if args.open_asset_target:
            if not args.asset_id:
                raise RuntimeError("--open-asset requires --asset-id.")
            asset = select_asset_record(asset_records, args.asset_id)
            path = target_path_for_asset(asset, args.open_asset_target)
            if not path:
                raise RuntimeError(f"No existing {args.open_asset_target} target found for {asset.asset_id}.")
            info(f"Opening {path}")
            open_path(path)

        if args.archive_asset and args.delete_asset:
            raise RuntimeError("Use either --archive-asset or --delete-asset, not both.")

        if args.archive_asset:
            if not args.asset_id:
                raise RuntimeError("--archive-asset requires --asset-id.")
            if not args.dry_run and args.confirm != args.asset_id:
                raise RuntimeError("--archive-asset requires --confirm with the same value as --asset-id.")
            asset = select_asset_record(asset_records, args.asset_id)
            manifest = archive_season_asset(asset, dry_run=bool(args.dry_run))
            if args.dry_run:
                print(json.dumps(manifest, indent=2, ensure_ascii=False))
                info(f"Dry run found {len(manifest['moved'])} path(s) for archiving.")
            else:
                ok(f"Archived {len(manifest['moved'])} path(s) to {manifest['archive_root']}")

        if args.delete_asset:
            if not args.asset_id:
                raise RuntimeError("--delete-asset requires --asset-id.")
            if not args.dry_run and args.confirm != args.asset_id:
                raise RuntimeError("--delete-asset requires --confirm with the same value as --asset-id.")
            asset = select_asset_record(asset_records, args.asset_id)
            result = delete_season_asset_outputs(asset, dry_run=bool(args.dry_run))
            if args.dry_run:
                print(json.dumps(result, indent=2, ensure_ascii=False))
                info(f"Dry run found {len(result['deleted'])} generated path(s) for deletion.")
            else:
                ok(f"Deleted {len(result['deleted'])} generated path(s) for {asset.asset_id}.")
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
