#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_backend_context() -> dict[str, Any]:
    return load_json(str(os.environ.get("SERIES_BACKEND_CONTEXT_JSON", "") or ""))


def ensure_parent(path: str) -> None:
    candidate = Path(path)
    if candidate.suffix:
        candidate.parent.mkdir(parents=True, exist_ok=True)
    else:
        candidate.mkdir(parents=True, exist_ok=True)


def write_context_file(prefix: str, payload: dict[str, Any]) -> Path:
    temp_root = Path(tempfile.gettempdir()) / "ai_series_quality_backends"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".json",
        prefix=f"{prefix}_",
        delete=False,
        dir=temp_root,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        return Path(handle.name)


def command_shell_flag() -> bool:
    return os.name == "nt"


def run_delegated_backend(
    *,
    env_var_name: str,
    context_payload: dict[str, Any],
    context_prefix: str,
    cwd: str = "",
) -> int:
    command_text = str(os.environ.get(env_var_name, "") or "").strip()
    if not command_text:
        raise RuntimeError(
            f"Missing backend command. Set {env_var_name} before running the quality-first episode pipeline."
        )

    context_path = write_context_file(context_prefix, context_payload)
    env = os.environ.copy()
    env["SERIES_BACKEND_CONTEXT_JSON"] = str(context_path)
    for key, value in context_payload.items():
        env_key = f"SERIES_{key.upper()}"
        env[env_key] = str(value)

    working_directory = cwd.strip() or str(Path.cwd())
    if command_shell_flag():
        completed = subprocess.run(
            command_text,
            shell=True,
            cwd=working_directory,
            env=env,
            check=False,
        )
    else:
        completed = subprocess.run(
            shlex.split(command_text),
            shell=False,
            cwd=working_directory,
            env=env,
            check=False,
        )
    return int(completed.returncode)


def choose_scene_package_root(scene_package_path: str) -> str:
    candidate = Path(scene_package_path)
    if candidate.exists():
        return str(candidate.parent)
    return str(Path.cwd())


def existing_path(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text)
    return candidate if candidate.exists() else None


def first_existing_path(*values: object) -> Path | None:
    for value in values:
        candidate = existing_path(value)
        if candidate is not None:
            return candidate
    return None


def copy_if_needed(source: Path, target: Path) -> None:
    ensure_parent(str(target))
    if source.resolve() == target.resolve():
        return
    target.write_bytes(source.read_bytes())


def find_project_local_ffmpeg() -> str:
    script_path = Path(__file__).resolve()
    for parent in [script_path.parent, *script_path.parents]:
        candidate_dir = parent / "ai_series_project" / "tools" / "ffmpeg" / "bin"
        if candidate_dir.exists():
            for name in ("ffmpeg.exe", "ffmpeg"):
                candidate = candidate_dir / name
                if candidate.exists() and candidate.is_file():
                    return str(candidate)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    raise RuntimeError("No FFmpeg binary found. Run 00_prepare_runtime.py so the project-local FFmpeg build is available.")


def print_runtime_error(exc: Exception) -> int:
    print(f"[ERROR] {exc}", file=sys.stderr)
    return 1
