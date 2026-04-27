#!/usr/bin/env python3
"""Backend Preset Benchmarking Tool.

Compares different backend runner presets against test scenes and produces
a ranked recommendation report.

Usage:
    python backend_preset_benchmark.py [--preset-file PRESETS.JSON] [--output REPORT.JSON]
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from pipeline_common import (
    compare_backend_runners,
    info,
    ok,
    warn,
    write_json,
)

# ---------------------------------------------------------------------------
# Default presets
# ---------------------------------------------------------------------------

DEFAULT_PRESETS: list[dict[str, Any]] = [
    {
        "name": "quality_first",
        "description": "Maximum quality, slower generation",
        "enabled": True,
        "success_outputs": ["{frame_path}"],
        "timeout_seconds": 600,
        "quality_weight": 0.9,
        "speed_weight": 0.3,
    },
    {
        "name": "speed_first",
        "description": "Fast generation, lower quality",
        "enabled": True,
        "success_outputs": ["{frame_path}"],
        "timeout_seconds": 120,
        "quality_weight": 0.4,
        "speed_weight": 0.9,
    },
    {
        "name": "balanced",
        "description": "Balanced quality and speed",
        "enabled": True,
        "success_outputs": ["{frame_path}"],
        "timeout_seconds": 300,
        "quality_weight": 0.7,
        "speed_weight": 0.7,
    },
    {
        "name": "disabled_runner",
        "description": "Disabled runner for comparison baseline",
        "enabled": False,
        "success_outputs": [],
        "timeout_seconds": 0,
        "quality_weight": 0.0,
        "speed_weight": 0.0,
    },
]


# ---------------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------------

def build_test_scene() -> dict[str, Any]:
    """Return a minimal test scene payload for benchmarking."""
    return {
        "scene_id": "bench_scene_01",
        "frame_path": str(Path("/tmp/bench_frame.png")),
        "description": "Benchmark test scene",
    }


def run_benchmark(
    presets: list[dict[str, Any]],
    test_scene: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the benchmark against all presets and return a ranked report."""
    if test_scene is None:
        test_scene = build_test_scene()

    start_time = time.time()
    raw_results = compare_backend_runners(presets, test_scene)
    elapsed = time.time() - start_time

    # Compute composite score: weighted average of quality and speed
    for result in raw_results:
        qw = float(result.get("quality_weight", result.get("estimated_quality", 0.0)))
        sw = float(result.get("speed_weight", result.get("estimated_speed", 0.0)))
        composite = round((qw + sw) / 2.0, 4)
        result["composite_score"] = composite
        # Normalize key: compare_backend_runners uses "runner_name", we alias to "name"
        if "name" not in result and "runner_name" in result:
            result["name"] = result["runner_name"]

    # Sort by composite score descending
    raw_results.sort(key=lambda x: x.get("composite_score", 0.0), reverse=True)

    # Assign ranks
    for rank, result in enumerate(raw_results, start=1):
        result["rank"] = rank

    report = {
        "benchmark_id": f"bench_{uuid.uuid4().hex[:8]}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 3),
        "preset_count": len(presets),
        "test_scene_id": test_scene.get("scene_id", "unknown"),
        "results": raw_results,
        "recommended_preset": raw_results[0].get("name") if raw_results else None,
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark backend runner presets and produce a ranked report."
    )
    parser.add_argument(
        "--preset-file",
        help="Path to a JSON file containing preset definitions.",
    )
    parser.add_argument(
        "--output",
        help="Path to write the benchmark report JSON.",
        default="ai_series_project/logs/backend_benchmark_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load presets
    if args.preset_file and Path(args.preset_file).exists():
        presets = json.loads(Path(args.preset_file).read_text(encoding="utf-8"))
        if not isinstance(presets, list):
            presets = [presets]
        info(f"Loaded {len(presets)} presets from {args.preset_file}")
    else:
        presets = deepcopy(DEFAULT_PRESETS)
        info(f"Using {len(presets)} default presets")

    # Run benchmark
    report = run_benchmark(presets)

    # Write report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    ok(f"Benchmark report written to {output_path}")

    # Print summary
    print()
    print("=" * 60)
    print("Backend Preset Benchmark Results")
    print("=" * 60)
    for r in report["results"]:
        status = "RECOMMENDED" if r.get("rank") == 1 else ""
        print(
            f"  #{r['rank']}  {r['name']:<20s}  "
            f"quality={r.get('estimated_quality', 0):.2f}  "
            f"speed={r.get('estimated_speed', 0):.2f}  "
            f"composite={r.get('composite_score', 0):.4f}  "
            f"enabled={r.get('enabled', False)}  {status}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
