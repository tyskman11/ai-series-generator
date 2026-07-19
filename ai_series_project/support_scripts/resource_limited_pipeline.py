#!/usr/bin/env python3
"""Apply portable host resource limits before running the numbered pipeline."""

from __future__ import annotations

import ctypes
import os
import runpy
import socket
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
ALLOWED_SCRIPT = (SCRIPT_ROOT / "24_process_next_episode.py").resolve(strict=False)


def _integer_environment(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(float(os.environ.get(name, str(default)) or default))
    except ValueError:
        parsed = default
    return max(minimum, min(maximum, parsed))


def apply_cpu_affinity(thread_count: int) -> int:
    logical_count = max(1, int(os.cpu_count() or 1))
    selected_count = max(1, min(logical_count, int(thread_count)))
    selected = set(range(selected_count))
    if os.name == "nt":
        mask = sum(1 << index for index in selected)
        kernel32 = ctypes.windll.kernel32
        if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), ctypes.c_size_t(mask)):
            raise OSError("Windows rejected the requested CPU affinity mask.")
    elif hasattr(os, "sched_setaffinity"):
        available = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(logical_count))
        os.sched_setaffinity(0, set(available[:selected_count]))
    return selected_count


def apply_process_priority(priority: str) -> str:
    normalized = str(priority or "normal").strip().lower()
    if normalized != "low":
        return "normal"
    if os.name == "nt":
        below_normal_priority_class = 0x00004000
        kernel32 = ctypes.windll.kernel32
        if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), below_normal_priority_class):
            raise OSError("Windows rejected the requested process priority.")
    elif hasattr(os, "nice"):
        os.nice(10)
    return "low"


def resolve_pipeline_script(argument: str) -> Path:
    candidate = Path(argument).resolve(strict=False)
    if candidate != ALLOWED_SCRIPT or not candidate.is_file():
        raise RuntimeError(f"Resource controller only permits the numbered pipeline entry point: {ALLOWED_SCRIPT}")
    return candidate


def main() -> int:
    if len(sys.argv) != 2:
        raise RuntimeError("Expected exactly one numbered pipeline script path.")
    script = resolve_pipeline_script(sys.argv[1])
    requested_threads = _integer_environment("SERIES_CPU_THREADS", max(1, int(os.cpu_count() or 1)), 1, max(1, int(os.cpu_count() or 1)))
    selected_threads = apply_cpu_affinity(requested_threads)
    priority = apply_process_priority(os.environ.get("SERIES_WEB_PROCESS_PRIORITY", "normal"))
    gpu_budget = _integer_environment("SERIES_GPU_MEMORY_PERCENT", 100, 0, 100)
    print(
        "[WEB RESOURCE] "
        f"host={socket.gethostname()} cpu_threads={selected_threads}/{max(1, int(os.cpu_count() or 1))} "
        f"priority={priority} gpu_memory_budget={gpu_budget}%",
        flush=True,
    )
    sys.argv = [str(script)]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
