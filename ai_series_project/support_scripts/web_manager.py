#!/usr/bin/env python3
"""Read-only public dashboard and authenticated NAS project administration."""

from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.util
import json
import mimetypes
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
WEB_ROOT = PROJECT_DIR / "web" / "manager"
AUTH_PATH = PROJECT_DIR / "runtime" / "web" / "admin_credentials.json"
WEB_LOG_PATH = PROJECT_DIR / "logs" / "web_manager.log"
PIPELINE_LOG_ROOT = PROJECT_DIR / "logs" / "web_manager" / "pipeline_runs"
REVIEW_SCRIPT_PATH = SCRIPT_ROOT / "05_review_unknowns.py"
SESSION_COOKIE = "ai_series_web_session"
MAX_REQUEST_BYTES = 1024 * 1024
AUTH_WINDOW_SECONDS = 60.0
AUTH_MAX_FAILURES = 6
PASSWORD_ITERATIONS = 600_000
SESSION_TTL_SECONDS = 12 * 60 * 60
BROWSER_WORKER_TTL_SECONDS = 45.0
BROWSER_TASK_LEASE_SECONDS = 120.0
BROWSER_FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

RESOURCE_PROFILES: dict[str, dict[str, Any]] = {
    "eco": {"cpu_percent": 35, "gpu_memory_percent": 50, "priority": "low"},
    "balanced": {"cpu_percent": 65, "gpu_memory_percent": 75, "priority": "normal"},
    "performance": {"cpu_percent": 100, "gpu_memory_percent": 95, "priority": "normal"},
}


def _bounded_integer(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def normalize_resource_settings(value: Any) -> dict[str, Any]:
    """Return a portable host-side resource budget for a web-started pipeline."""
    requested = value if isinstance(value, dict) else {}
    profile = str(requested.get("profile", "balanced") or "balanced").strip().lower()
    if profile not in {*RESOURCE_PROFILES, "custom"}:
        profile = "balanced"
    defaults = RESOURCE_PROFILES.get(profile, RESOURCE_PROFILES["balanced"])
    cpu_percent = _bounded_integer(requested.get("cpu_percent"), int(defaults["cpu_percent"]), 10, 100)
    gpu_memory_percent = _bounded_integer(
        requested.get("gpu_memory_percent"), int(defaults["gpu_memory_percent"]), 0, 100
    )
    priority = str(requested.get("priority", defaults["priority"]) or defaults["priority"]).strip().lower()
    if priority not in {"low", "normal"}:
        priority = str(defaults["priority"])
    logical_cpu_count = max(1, int(os.cpu_count() or 1))
    cpu_threads = max(1, min(logical_cpu_count, round(logical_cpu_count * cpu_percent / 100.0)))
    return {
        "profile": profile,
        "cpu_percent": cpu_percent,
        "cpu_threads": cpu_threads,
        "logical_cpu_count": logical_cpu_count,
        "gpu_memory_percent": gpu_memory_percent,
        "priority": priority,
    }


def web_log(message: str) -> None:
    line = f"[WEB {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    try:
        WEB_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with WEB_LOG_PATH.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def is_loopback_host(host: str) -> bool:
    return str(host or "").strip().lower() in {"127.0.0.1", "localhost", "::1"}


def _password_hash(password: str, salt: bytes, iterations: int = PASSWORD_ITERATIONS) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))


def configure_admin_credentials(username: str, password: str) -> Path:
    final_username = str(username or "").strip()
    if not final_username:
        raise RuntimeError("Administrator username must not be empty.")
    if len(password) < 12:
        raise RuntimeError("Administrator password must contain at least 12 characters.")
    salt = secrets.token_bytes(24)
    digest = _password_hash(password, salt)
    payload = {
        "schema_version": 1,
        "username": final_username,
        "password_algorithm": "pbkdf2_hmac_sha256",
        "password_iterations": PASSWORD_ITERATIONS,
        "password_salt": base64.b64encode(salt).decode("ascii"),
        "password_hash": base64.b64encode(digest).decode("ascii"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    AUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = AUTH_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(AUTH_PATH)
    try:
        AUTH_PATH.chmod(0o600)
    except OSError:
        pass
    return AUTH_PATH


def load_admin_credentials() -> dict[str, Any]:
    if not AUTH_PATH.exists():
        raise RuntimeError(
            "Web administrator credentials are not configured. Run "
            "26_web_manager.py --configure-admin on the project host first."
        )
    try:
        payload = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read web administrator credentials: {AUTH_PATH}") from exc
    required = {"username", "password_salt", "password_hash", "password_iterations"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise RuntimeError(f"Web administrator credentials are incomplete: {AUTH_PATH}")
    return payload


def verify_admin_credentials(credentials: dict[str, Any], username: str, password: str) -> bool:
    try:
        salt = base64.b64decode(str(credentials["password_salt"]), validate=True)
        expected = base64.b64decode(str(credentials["password_hash"]), validate=True)
        actual = _password_hash(password, salt, int(credentials["password_iterations"]))
    except Exception:
        return False
    user_matches = hmac.compare_digest(str(username or "").strip(), str(credentials.get("username", "")))
    password_matches = hmac.compare_digest(actual, expected)
    return bool(user_matches and password_matches)


def parse_range_header(value: str, file_size: int) -> tuple[int, int] | None:
    text = str(value or "").strip()
    if not text:
        return None
    if not text.startswith("bytes=") or "," in text:
        raise ValueError("Only one byte range is supported.")
    start_text, separator, end_text = text[6:].partition("-")
    if not separator:
        raise ValueError("Invalid byte range.")
    if not start_text:
        suffix = int(end_text)
        if suffix <= 0:
            raise ValueError("Invalid byte range suffix.")
        start = max(0, file_size - suffix)
        return start, max(start, file_size - 1)
    start = int(start_text)
    end = int(end_text) if end_text else file_size - 1
    if start < 0 or start >= file_size or end < start:
        raise ValueError("Requested byte range is outside the file.")
    return start, min(end, file_size - 1)


def local_access_urls(host: str, port: int) -> list[str]:
    urls = [f"http://127.0.0.1:{port}/"]
    if is_loopback_host(host):
        return urls
    try:
        addresses = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        addresses = []
    for row in addresses:
        address = row[4][0]
        if address.startswith("127."):
            continue
        url = f"http://{address}:{port}/"
        if url not in urls:
            urls.append(url)
    return urls


def _path_inside_project(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(PROJECT_DIR.resolve(strict=False))
    except ValueError as exc:
        raise RuntimeError(f"Refusing to read a file outside ai_series_project: {path}") from exc
    return resolved


def _load_review_module() -> Any:
    spec = importlib.util.spec_from_file_location("ai_series_review_web", REVIEW_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load review step: {REVIEW_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SessionStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sessions: dict[str, float] = {}

    def create(self) -> str:
        token = secrets.token_urlsafe(36)
        with self.lock:
            self.sessions[token] = time.time() + SESSION_TTL_SECONDS
        return token

    def valid(self, token: str) -> bool:
        now = time.time()
        with self.lock:
            expired = [key for key, expiry in self.sessions.items() if expiry <= now]
            for key in expired:
                self.sessions.pop(key, None)
            return bool(token and self.sessions.get(token, 0.0) > now)

    def revoke(self, token: str) -> None:
        with self.lock:
            self.sessions.pop(token, None)


class WebManagerService:
    def __init__(self, manager: Any, cfg: dict[str, Any]) -> None:
        self.manager = manager
        self.cfg = cfg
        self.lock = threading.RLock()
        self.pipeline_process: subprocess.Popen[Any] | None = None
        self.pipeline_log_path: Path | None = None
        self.pipeline_resources: dict[str, Any] | None = None
        self.review_module: Any | None = None
        self.browser_workers: dict[str, dict[str, Any]] = {}
        self.browser_tasks: dict[str, dict[str, Any]] = {}

    def status(self) -> dict[str, Any]:
        with self.lock:
            payload = self.manager.build_live_generation_status(self.cfg).to_dict()
            payload["local_web_process"] = self.local_process_status()
            payload["execution_host"] = socket.gethostname()
            payload["logical_cpu_count"] = max(1, int(os.cpu_count() or 1))
            payload["browser_workers"] = self.browser_worker_status()
            return payload

    def public_overview(self) -> dict[str, Any]:
        with self.lock:
            status = self.manager.build_live_generation_status(self.cfg).to_dict()
            episodes = self.manager.list_episode_records(self.cfg)
            assets = self.manager.list_season_asset_records(self.cfg)
        return self._public_overview_from_records(status, episodes, assets)

    @staticmethod
    def _public_overview_from_records(status: dict[str, Any], episodes: list[Any], assets: list[Any]) -> dict[str, Any]:
        qualities = [float(row.quality_percent or 0.0) for row in episodes if float(row.quality_percent or 0.0) > 0]
        public_status = {
            "status": status.get("status", "UNKNOWN"),
            "active": bool(status.get("active")),
            "stale": bool(status.get("stale")),
            "current_step": str(status.get("current_step") or ""),
            "current_episode": str(status.get("current_episode") or ""),
            "updated_at_text": str(status.get("updated_at_text") or ""),
            "updated_age": str(status.get("updated_age") or ""),
            "eta_text": str(status.get("eta_text") or "calculating"),
            "completed_steps": int(status.get("completed_steps", 0) or 0),
            "total_steps": int(status.get("total_steps", 0) or 0),
            "progress_percent": int(status.get("progress_percent", 0) or 0),
            "active_worker_count": len(status.get("active_workers", []) or []),
        }
        return {
            "status": public_status,
            "statistics": {
                "generated_episode_count": len(episodes),
                "finished_episode_count": sum(1 for row in episodes if row.finished_gate_passed),
                "release_ready_count": sum(1 for row in episodes if row.release_gate_passed),
                "average_quality_percent": round(sum(qualities) / len(qualities), 1) if qualities else 0.0,
                "season_asset_count": len(assets),
                "ready_season_asset_count": sum(1 for row in assets if row.status not in {"", "missing"}),
                "open_regeneration_items": sum(int(row.regeneration_queue_count or 0) for row in episodes),
            },
            "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def overview(self) -> dict[str, Any]:
        with self.lock:
            episode_records = self.manager.list_episode_records(self.cfg)
            asset_records = self.manager.list_season_asset_records(self.cfg)
            status = self.manager.build_live_generation_status(self.cfg).to_dict()
            status["local_web_process"] = self.local_process_status()
            status["execution_host"] = socket.gethostname()
            status["logical_cpu_count"] = max(1, int(os.cpu_count() or 1))
            status["browser_workers"] = self.browser_worker_status()
        public = self._public_overview_from_records(status, episode_records, asset_records)
        return {
            **public,
            "status": status,
            "episodes": [record.to_dict() for record in episode_records],
            "assets": [record.to_dict() for record in asset_records],
        }

    def storage(self, max_records: int = 240) -> list[dict[str, Any]]:
        limit = max(20, min(int(max_records), self.manager.PROJECT_STORAGE_MAX_RECORDS))
        with self.lock:
            return [record.to_dict() for record in self.manager.list_project_storage_records(max_records=limit)]

    def _storage_records(self, record_ids: list[str]) -> list[Any]:
        records = self.manager.list_project_storage_records(max_records=self.manager.PROJECT_STORAGE_MAX_RECORDS)
        return [self.manager.select_project_storage_record(records, record_id) for record_id in record_ids]

    def read_json_database(self, record_id: str) -> dict[str, Any]:
        with self.lock:
            record = self._storage_records([record_id])[0]
            payload = self.manager.read_json_database(record)
            return {"record": record.to_dict(), "text": json.dumps(payload, ensure_ascii=False, indent=2) + "\n"}

    def save_json_database(self, record_id: str, text: str) -> dict[str, Any]:
        self.require_idle_pipeline()
        with self.lock:
            record = self._storage_records([record_id])[0]
            return self.manager.save_json_database(record, text)

    def require_idle_pipeline(self) -> None:
        status = self.manager.build_live_generation_status(self.cfg).to_dict()
        if status.get("active"):
            raise RuntimeError(f"Project mutations are blocked while {status.get('current_step') or 'the pipeline'} is active.")

    def mutate_episodes(self, ids: list[str], action: str, dry_run: bool = False) -> dict[str, Any]:
        self.require_idle_pipeline()
        results: list[dict[str, Any]] = []
        with self.lock:
            records = self.manager.list_episode_records(self.cfg)
            for episode_id in ids:
                record = self.manager.select_record(records, episode_id)
                if action == "archive":
                    results.append(self.manager.archive_episode(record, dry_run=dry_run))
                elif action == "delete":
                    results.append(self.manager.delete_episode_outputs(record, dry_run=dry_run))
                else:
                    raise RuntimeError(f"Unsupported episode action: {action}")
        return {"action": action, "results": results, "dry_run": dry_run}

    def mutate_assets(self, ids: list[str], action: str, dry_run: bool = False) -> dict[str, Any]:
        self.require_idle_pipeline()
        results: list[dict[str, Any]] = []
        with self.lock:
            records = self.manager.list_season_asset_records(self.cfg)
            for asset_id in ids:
                record = self.manager.select_asset_record(records, asset_id)
                if action == "archive":
                    results.append(self.manager.archive_season_asset(record, dry_run=dry_run))
                elif action == "delete":
                    results.append(self.manager.delete_season_asset_outputs(record, dry_run=dry_run))
                else:
                    raise RuntimeError(f"Unsupported asset action: {action}")
        return {"action": action, "results": results, "dry_run": dry_run}

    def mutate_storage(self, ids: list[str], action: str, dry_run: bool = False) -> dict[str, Any]:
        self.require_idle_pipeline()
        with self.lock:
            records = self._storage_records(ids)
            if action == "archive":
                return self.manager.archive_project_storage_records(records, dry_run=dry_run)
            if action == "delete":
                return self.manager.delete_project_storage_records(records, dry_run=dry_run)
            raise RuntimeError(f"Unsupported storage action: {action}")

    def media_path(self, kind: str, record_id: str) -> Path:
        with self.lock:
            if kind == "episode":
                record = self.manager.select_record(self.manager.list_episode_records(self.cfg), record_id)
                path = self.manager.best_video_path(record)
            elif kind == "asset":
                record = self.manager.select_asset_record(self.manager.list_season_asset_records(self.cfg), record_id)
                path = self.manager.target_path_for_asset(record, "video")
            else:
                raise RuntimeError(f"Unsupported media kind: {kind}")
        if not path or not path.exists() or not path.is_file():
            raise RuntimeError(f"No playable media exists for {record_id}.")
        return _path_inside_project(path)

    def local_process_status(self) -> dict[str, Any]:
        process = self.pipeline_process
        if process is None:
            return {
                "active": False,
                "pid": 0,
                "log_available": bool(self.pipeline_log_path),
                "resources": self.pipeline_resources or {},
            }
        return {
            "active": process.poll() is None,
            "pid": int(process.pid),
            "exit_code": process.poll(),
            "log_available": bool(self.pipeline_log_path),
            "resources": self.pipeline_resources or {},
        }

    def start_pipeline(self, resources: Any = None) -> dict[str, Any]:
        self.require_idle_pipeline()
        with self.lock:
            if self.pipeline_process is not None and self.pipeline_process.poll() is None:
                raise RuntimeError("A pipeline process started by this web server is already active.")
            script = SCRIPT_ROOT / "24_process_next_episode.py"
            if not script.exists():
                raise RuntimeError(f"Pipeline entry point is missing: {script}")
            resource_runner = PROJECT_DIR / "support_scripts" / "resource_limited_pipeline.py"
            if not resource_runner.exists():
                raise RuntimeError(f"Host resource controller is missing: {resource_runner}")
            self.pipeline_resources = normalize_resource_settings(resources)
            PIPELINE_LOG_ROOT.mkdir(parents=True, exist_ok=True)
            self.pipeline_log_path = PIPELINE_LOG_ROOT / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
            environment = os.environ.copy()
            environment.update(
                {
                    "SERIES_DISABLE_NETWORK": "1",
                    "SERIES_DISABLE_DOWNLOADS": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "DIFFUSERS_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                    "SERIES_WEB_CPU_PERCENT": str(self.pipeline_resources["cpu_percent"]),
                    "SERIES_CPU_THREADS": str(self.pipeline_resources["cpu_threads"]),
                    "OMP_NUM_THREADS": str(self.pipeline_resources["cpu_threads"]),
                    "MKL_NUM_THREADS": str(self.pipeline_resources["cpu_threads"]),
                    "OPENBLAS_NUM_THREADS": str(self.pipeline_resources["cpu_threads"]),
                    "NUMEXPR_NUM_THREADS": str(self.pipeline_resources["cpu_threads"]),
                    "SERIES_GPU_MEMORY_PERCENT": str(self.pipeline_resources["gpu_memory_percent"]),
                    "SERIES_WEB_PROCESS_PRIORITY": str(self.pipeline_resources["priority"]),
                }
            )
            if int(self.pipeline_resources["gpu_memory_percent"]) <= 0:
                environment["CUDA_VISIBLE_DEVICES"] = ""
            creationflags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) if os.name == "nt" else 0
            with self.pipeline_log_path.open("ab") as output:
                self.pipeline_process = subprocess.Popen(
                    [sys.executable, str(resource_runner), str(script)],
                    cwd=str(SCRIPT_ROOT),
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=environment,
                    creationflags=creationflags,
                    start_new_session=os.name != "nt",
                )
            web_log(
                f"Started offline full pipeline as local PID {self.pipeline_process.pid} on {socket.gethostname()} "
                f"with resources {self.pipeline_resources}"
            )
            return {"started": True, "pid": self.pipeline_process.pid, "resources": self.pipeline_resources}

    def stop_pipeline(self) -> dict[str, Any]:
        with self.lock:
            process = self.pipeline_process
            if process is None or process.poll() is not None:
                raise RuntimeError("No pipeline process started by this web server is active.")
            pid = int(process.pid)
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, timeout=30)
            else:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait(timeout=15)
            web_log(f"Stopped web-started local pipeline PID {pid}")
            return {"stopped": True, "pid": pid, "exit_code": process.poll()}

    def latest_pipeline_log(self, max_bytes: int = 96_000) -> dict[str, Any]:
        path = self.pipeline_log_path
        if path is None or not path.exists():
            candidates = sorted(PIPELINE_LOG_ROOT.glob("pipeline_*.log"), key=lambda item: item.stat().st_mtime, reverse=True)
            path = candidates[0] if candidates else None
        if path is None or not path.exists():
            return {"path": "", "text": "No web-started pipeline log is available."}
        with path.open("rb") as handle:
            size = path.stat().st_size
            handle.seek(max(0, size - max_bytes))
            text = handle.read(max_bytes).decode("utf-8", errors="replace")
        return {"path": str(path), "text": text}

    def _browser_metrics_path(self) -> Path:
        return PROJECT_DIR / "generation" / "quality_reports" / "browser_workers" / "browser_worker_metrics.json"

    def _reclaim_browser_workers(self) -> None:
        now = time.time()
        active_ids = {
            worker_id
            for worker_id, row in self.browser_workers.items()
            if now - float(row.get("last_seen", 0.0) or 0.0) <= BROWSER_WORKER_TTL_SECONDS
        }
        for task in self.browser_tasks.values():
            if task.get("status") != "leased":
                continue
            leased_at = float(task.get("leased_at", 0.0) or 0.0)
            if str(task.get("worker_id", "")) not in active_ids or now - leased_at > BROWSER_TASK_LEASE_SECONDS:
                task.update({"status": "queued", "worker_id": "", "leased_at": 0.0})

    def browser_worker_status(self) -> dict[str, Any]:
        with self.lock:
            self._reclaim_browser_workers()
            now = time.time()
            active = [
                row
                for row in self.browser_workers.values()
                if now - float(row.get("last_seen", 0.0) or 0.0) <= BROWSER_WORKER_TTL_SECONDS
            ]
            counts = {
                status: sum(1 for task in self.browser_tasks.values() if task.get("status") == status)
                for status in ("queued", "leased", "completed", "failed")
            }
            return {
                "active_count": len(active),
                "workers": [
                    {
                        "worker_id": str(row.get("worker_id", ""))[:12],
                        "profile": str(row.get("profile", "balanced")),
                        "cpu_intensity": int(row.get("cpu_intensity", 60) or 60),
                        "hardware_concurrency": int(row.get("hardware_concurrency", 0) or 0),
                        "webgpu_available": bool(row.get("webgpu_available", False)),
                        "completed_tasks": int(row.get("completed_tasks", 0) or 0),
                        "failed_tasks": int(row.get("failed_tasks", 0) or 0),
                        "last_seen_age_seconds": round(max(0.0, now - float(row.get("last_seen", now) or now)), 1),
                    }
                    for row in active
                ],
                "tasks": counts,
                "worker_lifetime": "while_page_open",
                "generation_capable": False,
                "supported_task_types": ["frame_quality_metrics"],
            }

    def register_browser_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile = str(payload.get("profile", "balanced") or "balanced").strip().lower()
        if profile not in {"eco", "balanced", "performance", "custom"}:
            profile = "balanced"
        cpu_intensity = _bounded_integer(payload.get("cpu_intensity"), 60, 10, 100)
        worker_id = secrets.token_urlsafe(18)
        with self.lock:
            self.browser_workers[worker_id] = {
                "worker_id": worker_id,
                "profile": profile,
                "cpu_intensity": cpu_intensity,
                "hardware_concurrency": _bounded_integer(payload.get("hardware_concurrency"), 0, 0, 512),
                "webgpu_available": bool(payload.get("webgpu_available", False)),
                "completed_tasks": 0,
                "failed_tasks": 0,
                "registered_at": time.time(),
                "last_seen": time.time(),
            }
            status = self.browser_worker_status()
        web_log(
            f"Registered temporary browser worker {worker_id[:12]} with profile={profile}, "
            f"cpu_intensity={cpu_intensity}%"
        )
        return {"worker_id": worker_id, "status": status}

    def heartbeat_browser_worker(self, worker_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            row = self.browser_workers.get(str(worker_id or ""))
            if not row:
                raise RuntimeError("Browser worker registration expired; register again.")
            row["last_seen"] = time.time()
            if "cpu_intensity" in payload:
                row["cpu_intensity"] = _bounded_integer(payload.get("cpu_intensity"), int(row["cpu_intensity"]), 10, 100)
            return self.browser_worker_status()

    def unregister_browser_worker(self, worker_id: str) -> dict[str, Any]:
        with self.lock:
            worker_id = str(worker_id or "")
            self.browser_workers.pop(worker_id, None)
            for task in self.browser_tasks.values():
                if task.get("status") == "leased" and task.get("worker_id") == worker_id:
                    task.update({"status": "queued", "worker_id": "", "leased_at": 0.0})
            return self.browser_worker_status()

    def _load_browser_metrics(self) -> dict[str, Any]:
        path = self._browser_metrics_path()
        if not path.exists():
            return {"schema_version": 1, "metrics": []}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"schema_version": 1, "metrics": []}
        return payload if isinstance(payload, dict) else {"schema_version": 1, "metrics": []}

    def queue_browser_frame_checks(self, limit: int = 48) -> dict[str, Any]:
        final_limit = max(1, min(int(limit), 120))
        generation_root = PROJECT_DIR / "generation"
        roots = [
            generation_root / "storyboard_assets",
            generation_root / "final_episode_packages",
            generation_root / "season_assets",
            generation_root / "renders",
        ]
        candidates: list[tuple[int, int, Path]] = []
        inspected = 0
        for root in roots:
            if not root.exists():
                continue
            for candidate in root.rglob("*"):
                inspected += 1
                if inspected > 6000:
                    break
                if candidate.is_file() and candidate.suffix.lower() in BROWSER_FRAME_EXTENSIONS:
                    try:
                        stat = candidate.stat()
                    except OSError:
                        continue
                    candidates.append((int(stat.st_mtime_ns), int(stat.st_size), candidate))
            if inspected > 6000:
                break
        candidates.sort(key=lambda row: row[0], reverse=True)
        metrics_payload = self._load_browser_metrics()
        measured_signatures = {
            str(row.get("source_signature", ""))
            for row in metrics_payload.get("metrics", [])
            if isinstance(row, dict) and str(row.get("source_signature", ""))
        }
        existing_signatures = {
            str(row.get("source_signature", ""))
            for row in self.browser_tasks.values()
            if row.get("status") in {"queued", "leased", "completed"}
        }
        queued = 0
        with self.lock:
            for mtime_ns, size, path in candidates:
                relative_path = path.resolve(strict=False).relative_to(PROJECT_DIR.resolve(strict=False)).as_posix()
                signature = hashlib.sha256(f"{relative_path}|{mtime_ns}|{size}".encode("utf-8")).hexdigest()
                if signature in measured_signatures or signature in existing_signatures:
                    continue
                task_id = secrets.token_urlsafe(16)
                self.browser_tasks[task_id] = {
                    "task_id": task_id,
                    "task_type": "frame_quality_metrics",
                    "path": str(path),
                    "relative_path": relative_path,
                    "source_signature": signature,
                    "source_mtime_ns": mtime_ns,
                    "source_size": size,
                    "status": "queued",
                    "worker_id": "",
                    "created_at": time.time(),
                    "leased_at": 0.0,
                }
                queued += 1
                if queued >= final_limit:
                    break
        return {"queued": queued, "inspected": inspected, "status": self.browser_worker_status()}

    def claim_browser_task(self, worker_id: str) -> dict[str, Any]:
        with self.lock:
            worker = self.browser_workers.get(str(worker_id or ""))
            if not worker:
                raise RuntimeError("Browser worker registration expired; register again.")
            worker["last_seen"] = time.time()
            self._reclaim_browser_workers()
            task = next((row for row in self.browser_tasks.values() if row.get("status") == "queued"), None)
            if task is None:
                return {"task": None, "retry_after_seconds": 3}
            task.update({"status": "leased", "worker_id": worker_id, "leased_at": time.time()})
            return {
                "task": {
                    "task_id": task["task_id"],
                    "task_type": task["task_type"],
                    "input_url": f"/api/browser-worker/input?worker_id={urllib.parse.quote(worker_id)}&task_id={urllib.parse.quote(task['task_id'])}",
                    "source_size": task["source_size"],
                }
            }

    def browser_task_input_path(self, worker_id: str, task_id: str) -> Path:
        with self.lock:
            task = self.browser_tasks.get(str(task_id or ""))
            if not task or task.get("status") != "leased" or task.get("worker_id") != worker_id:
                raise RuntimeError("Browser task is not leased to this worker.")
            return _path_inside_project(Path(str(task.get("path", ""))))

    @staticmethod
    def _browser_frame_score(result: dict[str, Any]) -> float:
        mean_luma = max(0.0, min(1.0, float(result.get("mean_luma", 0.0) or 0.0)))
        luma_stddev = max(0.0, min(1.0, float(result.get("luma_stddev", 0.0) or 0.0)))
        edge_score = max(0.0, min(1.0, float(result.get("edge_score", 0.0) or 0.0)))
        clipped = max(0.0, min(1.0, float(result.get("dark_ratio", 0.0) or 0.0) + float(result.get("bright_ratio", 0.0) or 0.0)))
        exposure_score = max(0.0, 1.0 - abs(mean_luma - 0.5) / 0.5)
        detail_score = min(1.0, 0.45 * (luma_stddev / 0.18) + 0.55 * (edge_score / 0.16))
        return round(max(0.0, min(1.0, 0.4 * exposure_score + 0.25 * (1.0 - clipped) + 0.35 * detail_score)), 4)

    def complete_browser_task(self, worker_id: str, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            worker = self.browser_workers.get(str(worker_id or ""))
            task = self.browser_tasks.get(str(task_id or ""))
            if not worker or not task or task.get("status") != "leased" or task.get("worker_id") != worker_id:
                raise RuntimeError("Browser task lease is missing or expired.")
            worker["last_seen"] = time.time()
            if str(payload.get("status", "success")) != "success":
                task["status"] = "failed"
                task["error"] = str(payload.get("error", "browser calculation failed"))[:500]
                worker["failed_tasks"] = int(worker.get("failed_tasks", 0) or 0) + 1
                return {"saved": False, "status": self.browser_worker_status()}
            result = payload.get("result", {}) if isinstance(payload.get("result", {}), dict) else {}
            width = _bounded_integer(result.get("width"), 0, 1, 32768)
            height = _bounded_integer(result.get("height"), 0, 1, 32768)
            normalized_result = {
                "width": width,
                "height": height,
                "sample_width": _bounded_integer(result.get("sample_width"), width, 1, 4096),
                "sample_height": _bounded_integer(result.get("sample_height"), height, 1, 4096),
                "mean_luma": round(max(0.0, min(1.0, float(result.get("mean_luma", 0.0) or 0.0))), 6),
                "luma_stddev": round(max(0.0, min(1.0, float(result.get("luma_stddev", 0.0) or 0.0))), 6),
                "edge_score": round(max(0.0, min(1.0, float(result.get("edge_score", 0.0) or 0.0))), 6),
                "dark_ratio": round(max(0.0, min(1.0, float(result.get("dark_ratio", 0.0) or 0.0))), 6),
                "bright_ratio": round(max(0.0, min(1.0, float(result.get("bright_ratio", 0.0) or 0.0))), 6),
            }
            score = self._browser_frame_score(normalized_result)
            metric_row = {
                "metric": "browser_frame_quality_score",
                "status": "measured",
                "score": score,
                "reason": "browser Canvas frame exposure/detail measurement",
                "inputs": normalized_result,
                "tool": "browser_worker_canvas_v1",
                "source_path": task["relative_path"],
                "source_signature": task["source_signature"],
                "source_mtime_ns": task["source_mtime_ns"],
                "source_size": task["source_size"],
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            report = self._load_browser_metrics()
            rows = [row for row in report.get("metrics", []) if isinstance(row, dict)]
            rows = [row for row in rows if row.get("source_signature") != task["source_signature"]]
            rows.append(metric_row)
            report = {"schema_version": 1, "updated_at": metric_row["created_at"], "metrics": rows[-3000:]}
            report_path = self._browser_metrics_path()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = report_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(report_path)
            task.update({"status": "completed", "completed_at": time.time(), "score": score})
            worker["completed_tasks"] = int(worker.get("completed_tasks", 0) or 0) + 1
            return {"saved": True, "metric": metric_row, "status": self.browser_worker_status()}

    def _review(self) -> Any:
        if self.review_module is None:
            self.review_module = _load_review_module()
        return self.review_module

    def _review_maps(self) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        review = self._review()
        char_map = review.read_json(review.resolve_project_path(self.cfg["paths"]["character_map"]), {"clusters": {}, "aliases": {}})
        voice_map = review.read_json(review.resolve_project_path(self.cfg["paths"]["voice_map"]), {"clusters": {}, "aliases": {}})
        return review, char_map, voice_map

    @staticmethod
    def _existing_review_preview(review: Any, cluster_id: str, payload: dict[str, Any], sample: int = 0) -> Path:
        preview_dir = review.preview_dir_path(payload)
        if not preview_dir or not preview_dir.is_dir():
            raise RuntimeError(f"No NAS preview folder exists for {cluster_id}.")
        montages = sorted(preview_dir.glob("*_montage.jpg"))
        contexts = sorted(preview_dir.glob("*_context.jpg"))
        regular = [path for path in sorted(preview_dir.glob("*.jpg")) if "_crop" not in path.name]
        candidates = montages or contexts or regular
        if not candidates:
            raise RuntimeError(f"No existing NAS preview image exists for {cluster_id}.")
        index = max(0, min(int(sample), len(candidates) - 1))
        return _path_inside_project(candidates[index])

    def review_preview_path(self, cluster_id: str, sample: int = 0) -> Path:
        review, char_map, _voice_map = self._review_maps()
        payload = char_map.get("clusters", {}).get(cluster_id)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unknown face cluster: {cluster_id}")
        return self._existing_review_preview(review, cluster_id, payload, sample)

    def review_overview(self, include_named: bool = False, limit: int = 100) -> dict[str, Any]:
        review, char_map, voice_map = self._review_maps()
        candidates = review.face_review_candidates(char_map, include_named, max(1, min(int(limit), 300)), set())
        rows: list[dict[str, Any]] = []
        for cluster_id, payload in candidates:
            preview_available = bool(str(payload.get("preview_dir", "") or "").strip())
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "name": str(payload.get("name") or cluster_id),
                    "samples": int(payload.get("samples", 0) or 0),
                    "scenes": int(payload.get("scene_count", payload.get("scenes", 0)) or 0),
                    "detections": int(payload.get("detection_count", payload.get("detections", 0)) or 0),
                    "priority": bool(payload.get("priority", False)),
                    "role_hint": str(review.suggested_face_role(payload)),
                    "review_hint": str(review.suggested_face_action_hint(payload)),
                    "preview_available": preview_available,
                }
            )
        queue = review.read_json(review.resolve_project_path(self.cfg["paths"]["review_queue"]), {"items": []})
        queue_items = queue.get("items", []) if isinstance(queue, dict) else []
        voice_rows = [
            {
                "speaker_id": speaker_id,
                "name": str(payload.get("name") or speaker_id),
                "samples": int(payload.get("samples", payload.get("segment_count", 0)) or 0),
            }
            for speaker_id, payload in sorted(voice_map.get("clusters", {}).items())
        ]
        return {
            "faces": rows,
            "known_names": review.created_face_names(char_map),
            "voice_clusters": voice_rows,
            "queue_summary": review.review_queue_summary(queue_items),
            "offline": True,
        }

    def assign_review_face(self, cluster_id: str, name: str, priority: bool = False) -> dict[str, Any]:
        self.require_idle_pipeline()
        final_name = str(name or "").strip()
        if not final_name:
            raise RuntimeError("Character name must not be empty.")
        with self.lock:
            review, char_map, voice_map = self._review_maps()
            if cluster_id not in char_map.get("clusters", {}):
                raise RuntimeError(f"Unknown face cluster: {cluster_id}")
            payload = review.assign_character_name(char_map, cluster_id, final_name, priority=bool(priority))
            changed_files, review_count = review.persist_updates(self.cfg, char_map, voice_map)
        return {
            "cluster_id": cluster_id,
            "name": str(payload.get("name") or final_name),
            "linked_files_updated": changed_files,
            "open_review_cases": review_count,
        }

    def rename_review_name(self, old_name: str, new_name: str, priority: bool = False) -> dict[str, Any]:
        self.require_idle_pipeline()
        if not str(old_name or "").strip() or not str(new_name or "").strip():
            raise RuntimeError("Both old and new character names are required.")
        with self.lock:
            review, char_map, voice_map = self._review_maps()
            summary = review.rename_name_everywhere(char_map, voice_map, old_name, new_name, priority=bool(priority))
            changed_files, review_count = review.persist_updates(self.cfg, char_map, voice_map)
        return {**summary, "linked_files_updated": changed_files, "open_review_cases": review_count}


class AuthLimiter:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.failures: dict[str, list[float]] = {}

    def blocked(self, address: str) -> bool:
        now = time.time()
        with self.lock:
            recent = [stamp for stamp in self.failures.get(address, []) if now - stamp < AUTH_WINDOW_SECONDS]
            self.failures[address] = recent
            return len(recent) >= AUTH_MAX_FAILURES

    def record_failure(self, address: str) -> None:
        with self.lock:
            self.failures.setdefault(address, []).append(time.time())

    def clear(self, address: str) -> None:
        with self.lock:
            self.failures.pop(address, None)


def make_handler(service: WebManagerService, credentials: dict[str, Any]) -> type[BaseHTTPRequestHandler]:
    limiter = AuthLimiter()
    sessions = SessionStore()
    secure_cookie = str(os.environ.get("SERIES_WEB_SECURE_COOKIE", "")).strip().lower() in {"1", "true", "yes", "on"}

    class WebManagerHandler(BaseHTTPRequestHandler):
        server_version = "AISeriesWeb/2.0"

        def log_message(self, format_string: str, *args: object) -> None:
            web_log(f"{self.client_address[0]} {format_string % args}")

        def _security_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; media-src 'self'; style-src 'self'; "
                "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; form-action 'self'",
            )

        def _send_json(self, payload: Any, status: int = HTTPStatus.OK, *, cookie: str = "") -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(int(status))
            self._security_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            if cookie:
                self.send_header("Set-Cookie", cookie)
            self.end_headers()
            self.wfile.write(body)

        def _send_error_json(self, status: int, message: str) -> None:
            self._send_json({"ok": False, "error": message}, status)

        def _read_json(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError as exc:
                raise RuntimeError("Invalid Content-Length header.") from exc
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise RuntimeError("Request body is empty or too large.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("JSON request body must be an object.")
            return payload

        def _session_token(self) -> str:
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            morsel = cookie.get(SESSION_COOKIE)
            return morsel.value if morsel else ""

        def _authenticated(self) -> bool:
            return sessions.valid(self._session_token())

        def _require_auth(self, *, mutation: bool = False) -> bool:
            if not self._authenticated():
                self._send_error_json(HTTPStatus.UNAUTHORIZED, "Administrator login required.")
                return False
            if mutation and self.headers.get("X-Series-Web", "") != "1":
                self._send_error_json(HTTPStatus.FORBIDDEN, "Missing same-origin mutation header.")
                return False
            return True

        def _serve_static(self, request_path: str) -> None:
            mapping = {
                "/": WEB_ROOT / "index.html",
                "/index.html": WEB_ROOT / "index.html",
                "/app.js": WEB_ROOT / "app.js",
                "/styles.css": WEB_ROOT / "styles.css",
                "/browser-compute-worker.js": WEB_ROOT / "browser-compute-worker.js",
                "/service-worker.js": WEB_ROOT / "service-worker.js",
                "/manifest.webmanifest": WEB_ROOT / "manifest.webmanifest",
            }
            path = mapping.get(request_path)
            if path is None or not path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or (
                "application/manifest+json" if path.suffix == ".webmanifest" else "application/octet-stream"
            )
            self.send_response(HTTPStatus.OK)
            self._security_headers()
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Cache-Control", "no-cache" if path.name in {"index.html", "service-worker.js"} else "public, max-age=3600")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_file(self, path: Path) -> None:
            size = path.stat().st_size
            try:
                byte_range = parse_range_header(self.headers.get("Range", ""), size)
            except (TypeError, ValueError):
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start, end = byte_range if byte_range is not None else (0, max(0, size - 1))
            length = max(0, end - start + 1)
            self.send_response(HTTPStatus.PARTIAL_CONTENT if byte_range is not None else HTTPStatus.OK)
            self._security_headers()
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Disposition", f'inline; filename="{path.name.replace(chr(34), "")}"')
            self.send_header("Cache-Control", "no-store")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if byte_range is not None:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path in {
                "/",
                "/index.html",
                "/app.js",
                "/styles.css",
                "/browser-compute-worker.js",
                "/service-worker.js",
                "/manifest.webmanifest",
            }:
                self._serve_static(parsed.path)
                return
            if parsed.path == "/api/health":
                self._send_json({"ok": True, "authenticated": self._authenticated(), "admin_configured": True})
                return
            if parsed.path == "/api/public/overview":
                try:
                    self._send_json({"ok": True, **service.public_overview()})
                except Exception as exc:
                    self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            if not self._require_auth():
                return
            query = urllib.parse.parse_qs(parsed.query)
            try:
                if parsed.path == "/api/status":
                    self._send_json({"ok": True, "status": service.status()})
                elif parsed.path == "/api/overview":
                    self._send_json({"ok": True, **service.overview()})
                elif parsed.path == "/api/storage":
                    self._send_json({"ok": True, "records": service.storage(int(query.get("limit", ["240"])[0]))})
                elif parsed.path == "/api/storage/json":
                    self._send_json({"ok": True, **service.read_json_database(query.get("id", [""])[0])})
                elif parsed.path == "/api/media":
                    self._serve_file(service.media_path(query.get("kind", [""])[0], query.get("id", [""])[0]))
                elif parsed.path == "/api/pipeline/log":
                    self._send_json({"ok": True, **service.latest_pipeline_log()})
                elif parsed.path == "/api/browser-worker/status":
                    self._send_json({"ok": True, "status": service.browser_worker_status()})
                elif parsed.path == "/api/browser-worker/input":
                    self._serve_file(
                        service.browser_task_input_path(
                            query.get("worker_id", [""])[0],
                            query.get("task_id", [""])[0],
                        )
                    )
                elif parsed.path == "/api/review":
                    include_named = query.get("include_named", ["0"])[0] == "1"
                    self._send_json({"ok": True, **service.review_overview(include_named=include_named)})
                elif parsed.path == "/api/review/preview":
                    cluster_id = query.get("id", [""])[0]
                    sample = int(query.get("sample", ["0"])[0])
                    self._serve_file(service.review_preview_path(cluster_id, sample))
                else:
                    self._send_error_json(HTTPStatus.NOT_FOUND, "API endpoint not found.")
            except Exception as exc:
                web_log(f"GET {parsed.path} failed: {exc}")
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/auth/login":
                address = self.client_address[0]
                if limiter.blocked(address):
                    self._send_error_json(HTTPStatus.TOO_MANY_REQUESTS, "Too many failed login attempts. Try again shortly.")
                    return
                try:
                    payload = self._read_json()
                    valid = verify_admin_credentials(credentials, str(payload.get("username", "")), str(payload.get("password", "")))
                except Exception:
                    valid = False
                if not valid:
                    limiter.record_failure(address)
                    self._send_error_json(HTTPStatus.UNAUTHORIZED, "Invalid username or password.")
                    return
                limiter.clear(address)
                token = sessions.create()
                cookie = f"{SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={SESSION_TTL_SECONDS}"
                if secure_cookie:
                    cookie += "; Secure"
                self._send_json({"ok": True, "username": credentials.get("username", "")}, cookie=cookie)
                return
            if not self._require_auth(mutation=True):
                return
            try:
                payload = self._read_json()
                if parsed.path == "/api/auth/logout":
                    sessions.revoke(self._session_token())
                    cookie = f"{SESSION_COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict"
                    if secure_cookie:
                        cookie += "; Secure"
                    self._send_json({"ok": True}, cookie=cookie)
                    return
                if parsed.path == "/api/pipeline/start":
                    self._send_json({"ok": True, **service.start_pipeline(payload.get("resources", {}))})
                    return
                if parsed.path == "/api/pipeline/stop":
                    self._send_json({"ok": True, **service.stop_pipeline()})
                    return
                if parsed.path == "/api/browser-worker/register":
                    self._send_json({"ok": True, **service.register_browser_worker(payload)})
                    return
                if parsed.path == "/api/browser-worker/heartbeat":
                    worker_id = str(payload.get("worker_id", ""))
                    self._send_json({"ok": True, "status": service.heartbeat_browser_worker(worker_id, payload)})
                    return
                if parsed.path == "/api/browser-worker/unregister":
                    self._send_json(
                        {"ok": True, "status": service.unregister_browser_worker(str(payload.get("worker_id", "")))}
                    )
                    return
                if parsed.path == "/api/browser-worker/queue":
                    self._send_json({"ok": True, **service.queue_browser_frame_checks(int(payload.get("limit", 48) or 48))})
                    return
                if parsed.path == "/api/browser-worker/claim":
                    self._send_json({"ok": True, **service.claim_browser_task(str(payload.get("worker_id", "")))})
                    return
                if parsed.path == "/api/browser-worker/result":
                    result = service.complete_browser_task(
                        str(payload.get("worker_id", "")),
                        str(payload.get("task_id", "")),
                        payload,
                    )
                    self._send_json({"ok": True, **result})
                    return
                if parsed.path == "/api/storage/json":
                    result = service.save_json_database(str(payload.get("id", "")), str(payload.get("text", "")))
                    self._send_json({"ok": True, "result": result})
                    return
                if parsed.path == "/api/review/assign":
                    result = service.assign_review_face(
                        str(payload.get("cluster_id", "")),
                        str(payload.get("name", "")),
                        bool(payload.get("priority", False)),
                    )
                    self._send_json({"ok": True, "result": result})
                    return
                if parsed.path == "/api/review/rename":
                    result = service.rename_review_name(
                        str(payload.get("old_name", "")),
                        str(payload.get("new_name", "")),
                        bool(payload.get("priority", False)),
                    )
                    self._send_json({"ok": True, "result": result})
                    return
                action = str(payload.get("action", "")).strip().lower()
                ids = [str(item).strip() for item in payload.get("ids", []) if str(item).strip()]
                if action not in {"archive", "delete"} or not ids:
                    raise RuntimeError("Select records and choose archive or delete.")
                expected_confirmation = "DELETE" if action == "delete" else "ARCHIVE"
                if str(payload.get("confirmation", "")).strip().upper() != expected_confirmation:
                    raise RuntimeError(f"Confirmation must equal {expected_confirmation}.")
                dry_run = bool(payload.get("dry_run", False))
                if parsed.path == "/api/episodes/mutate":
                    result = service.mutate_episodes(ids, action, dry_run)
                elif parsed.path == "/api/assets/mutate":
                    result = service.mutate_assets(ids, action, dry_run)
                elif parsed.path == "/api/storage/mutate":
                    result = service.mutate_storage(ids, action, dry_run)
                else:
                    self._send_error_json(HTTPStatus.NOT_FOUND, "API endpoint not found.")
                    return
                self._send_json({"ok": True, "result": result})
            except Exception as exc:
                web_log(f"POST {parsed.path} failed: {exc}")
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    return WebManagerHandler


def run_web_manager(
    manager: Any,
    cfg: dict[str, Any],
    *,
    host: str = "0.0.0.0",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    if not WEB_ROOT.joinpath("index.html").exists():
        raise RuntimeError(f"Web manager assets are missing: {WEB_ROOT}")
    credentials = load_admin_credentials()
    service = WebManagerService(manager, cfg)
    server = ThreadingHTTPServer((host, int(port)), make_handler(service, credentials))
    server.daemon_threads = True
    urls = local_access_urls(host, int(server.server_address[1]))
    web_log(f"Read-only statistics and administrator manager listening on {host}:{server.server_address[1]}")
    print("\nAI Series Web Statistics", flush=True)
    for url in urls:
        print(f"  URL: {url}", flush=True)
    print(f"  Administrator: {credentials.get('username', '')}", flush=True)
    print(f"  Credentials: salted hash stored only at {AUTH_PATH}", flush=True)
    print("  Network mode: offline; no model or public-reference downloads are started by this server", flush=True)
    if not is_loopback_host(host):
        print("  Security: use a trusted VPN or HTTPS reverse proxy; never expose this HTTP port directly.", flush=True)
    print("  Stop server: Ctrl+C\n", flush=True)
    if open_browser:
        try:
            webbrowser.open(urls[0])
        except Exception as exc:
            web_log(f"Could not open the local browser automatically: {exc}")
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        web_log("Web manager stopped.")
