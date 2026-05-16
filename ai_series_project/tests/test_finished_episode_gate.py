from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = PROJECT_DIR


def load_module(filename: str, module_name: str):
    target = ROOT / filename if filename.startswith("support_scripts/") else SCRIPT_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP22 = load_module("23_generate_finished_episodes.py", "step22_finished_gate_test")


class FinishedEpisodeGateTests(unittest.TestCase):
    def test_ensure_finished_episode_outputs_rejects_fallback_quality_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            render_manifest = root / "render_manifest.json"
            production_package = root / "production_package.json"
            final_render = root / "final_render.mp4"
            full_generated_episode = root / "full_generated_episode.mp4"
            delivery_manifest = root / "delivery_manifest.json"
            delivery_episode = root / "delivery_episode.mp4"
            for path in (final_render, full_generated_episode, delivery_episode):
                path.write_bytes(b"ok")
            delivery_manifest.write_text("{}", encoding="utf-8")
            render_manifest.write_text(
                '{"audio_track_meta": {"audio_backend": "pyttsx3"}}',
                encoding="utf-8",
            )
            production_package.write_text(
                (
                    '{"scenes": ['
                    '{"scene_id": "scene_0001", '
                    '"current_preview_assets": {"asset_source_type": "placeholder"}, '
                    '"current_generated_outputs": {"local_composed_scene_video": true}}'
                    "]}"
                ),
                encoding="utf-8",
            )
            episode_outputs = {
                "final_render": str(final_render),
                "full_generated_episode": str(full_generated_episode),
                "production_package": str(production_package),
                "render_manifest": str(render_manifest),
                "delivery_manifest": str(delivery_manifest),
                "delivery_episode": str(delivery_episode),
                "release_gate_passed": True,
                "production_readiness": "fully_generated_episode_ready",
                "scenes_below_release_threshold": 0,
                "remaining_backend_tasks": [],
                "backend_runner_expected_count": 0,
                "backend_runner_ready_count": 0,
                "backend_runner_failed_count": 0,
                "backend_runner_pending_count": 0,
            }

            with self.assertRaisesRegex(RuntimeError, "fallback dialogue audio backend"):
                STEP22.ensure_finished_episode_outputs({}, "folge_99", episode_outputs)


if __name__ == "__main__":
    unittest.main()



