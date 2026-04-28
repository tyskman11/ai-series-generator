import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend_preset_benchmark import (
    DEFAULT_PRESETS,
    build_test_scene,
    run_benchmark,
)


class BackendPresetBenchmarkTests(unittest.TestCase):
    """Tests for backend_preset_benchmark module."""

    def test_default_presets_is_list(self):
        self.assertIsInstance(DEFAULT_PRESETS, list)
        self.assertGreater(len(DEFAULT_PRESETS), 0)

    def test_default_presets_have_required_keys(self):
        required_keys = {"name", "enabled", "success_outputs"}
        for preset in DEFAULT_PRESETS:
            self.assertTrue(
                required_keys.issubset(preset.keys()),
                f"Preset {preset.get('name')} missing keys: {required_keys - preset.keys()}",
            )

    def test_build_test_scene_returns_dict(self):
        scene = build_test_scene()
        self.assertIsInstance(scene, dict)
        self.assertIn("scene_id", scene)
        self.assertIn("frame_path", scene)

    def test_run_benchmark_returns_report(self):
        report = run_benchmark(DEFAULT_PRESETS)
        self.assertIsInstance(report, dict)
        self.assertIn("benchmark_id", report)
        self.assertIn("results", report)
        self.assertIn("recommended_preset", report)
        self.assertIn("preset_count", report)
        self.assertIn("elapsed_seconds", report)
        self.assertIn("timestamp", report)

    def test_run_benchmark_results_sorted_by_composite_score(self):
        report = run_benchmark(DEFAULT_PRESETS)
        scores = [r["composite_score"] for r in report["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_run_benchmark_ranks_assigned(self):
        report = run_benchmark(DEFAULT_PRESETS)
        ranks = [r["rank"] for r in report["results"]]
        self.assertEqual(ranks, list(range(1, len(ranks) + 1)))

    def test_run_benchmark_recommended_is_top_ranked(self):
        report = run_benchmark(DEFAULT_PRESETS)
        self.assertEqual(
            report["recommended_preset"],
            report["results"][0]["name"],
        )

    def test_run_benchmark_disabled_runner_has_zero_scores(self):
        report = run_benchmark(DEFAULT_PRESETS)
        disabled = [r for r in report["results"] if not r.get("enabled")]
        for r in disabled:
            self.assertEqual(r["estimated_quality"], 0.0)
            self.assertEqual(r["estimated_speed"], 0.0)

    def test_run_benchmark_enabled_runner_has_positive_scores(self):
        report = run_benchmark(DEFAULT_PRESETS)
        enabled = [r for r in report["results"] if r.get("enabled")]
        for r in enabled:
            self.assertGreater(r["estimated_quality"], 0.0)
            self.assertGreater(r["estimated_speed"], 0.0)

    def test_run_benchmark_with_custom_presets(self):
        custom_presets = [
            {
                "name": "custom_fast",
                "enabled": True,
                "success_outputs": ["{frame_path}"],
                "quality_weight": 0.5,
                "speed_weight": 0.9,
            },
            {
                "name": "custom_slow",
                "enabled": True,
                "success_outputs": ["{frame_path}"],
                "quality_weight": 0.9,
                "speed_weight": 0.3,
            },
        ]
        report = run_benchmark(custom_presets)
        self.assertEqual(report["preset_count"], 2)
        self.assertEqual(len(report["results"]), 2)
        # custom_fast has composite 0.7 vs custom_slow 0.6, so custom_fast ranks higher
        self.assertEqual(report["recommended_preset"], "custom_fast")
        self.assertEqual(report["results"][0]["composite_score"], 0.7)
        self.assertEqual(report["results"][1]["composite_score"], 0.6)

    def test_run_benchmark_with_empty_presets(self):
        report = run_benchmark([])
        self.assertEqual(report["preset_count"], 0)
        self.assertEqual(report["results"], [])
        self.assertIsNone(report["recommended_preset"])

    def test_run_benchmark_writes_report_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            presets = deepcopy(DEFAULT_PRESETS)
            report = run_benchmark(presets)
            # Simulate what main() does
            from pipeline_common import write_json
            write_json(output_path, report)
            self.assertTrue(output_path.exists())
            loaded = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["benchmark_id"], report["benchmark_id"])


if __name__ == "__main__":
    unittest.main()
