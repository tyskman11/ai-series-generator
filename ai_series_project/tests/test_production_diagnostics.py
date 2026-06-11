from __future__ import annotations

import sys
import socket
import unittest
from pathlib import Path
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from support_scripts import production_diagnostics


class ProductionDiagnosticsTests(unittest.TestCase):
    def test_reference_dashboard_reports_missing_voice_evidence(self) -> None:
        character_map = {"clusters": {"face_001": {"name": "Babe"}}}
        voice_map = {"clusters": {}}

        with mock.patch.object(production_diagnostics, "voice_model_rows", return_value=[]), mock.patch.object(
            production_diagnostics,
            "reference_audio_for_character",
            return_value=[],
        ):
            dashboard = production_diagnostics.build_reference_quality_dashboard(
                {},
                character_map=character_map,
                voice_map=voice_map,
            )

        self.assertEqual(dashboard["character_count"], 1)
        self.assertEqual(dashboard["characters"][0]["name"], "Babe")
        self.assertEqual(dashboard["characters"][0]["status"], "needs_review")
        self.assertIn("no named speaker cluster", dashboard["characters"][0]["issues"])
        self.assertIn("no usable voice-reference file", dashboard["characters"][0]["issues"])

    def test_worker_capability_snapshot_exposes_routing_profiles_and_ready_runners(self) -> None:
        snapshot = production_diagnostics.build_worker_capability_snapshot(
            {},
            {
                "runtime": {
                    "gpu": {"available": True, "memory_total_mb": 16384, "devices": [{"name": "GPU"}]},
                    "package_status": {"torch": True, "quality_generation": True, "voice_cloning": True},
                },
                "runners": [
                    {"name": "shot_video", "enabled": True, "prerequisite_gaps": []},
                    {"name": "missing_lipsync", "enabled": True, "prerequisite_gaps": ["checkpoint missing"]},
                ],
            },
        )

        self.assertTrue(snapshot["has_gpu"])
        self.assertEqual(snapshot["hostname"], socket.gethostname())
        self.assertEqual(snapshot["gpu_memory_mb"], 16384)
        self.assertIn("shot_video", snapshot["ready_backend_runners"])
        self.assertNotIn("missing_lipsync", snapshot["ready_backend_runners"])
        self.assertEqual(snapshot["routing_profiles"]["shot_video"]["min_memory_mb"], 12288)


if __name__ == "__main__":
    unittest.main()
