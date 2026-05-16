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


STEP51 = load_module("21_export_package.py", "step50_export_package")


class ExportPackageTests(unittest.TestCase):
    def test_build_common_export_payload_includes_render_and_release_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            final_render = root / "episode_final.mp4"
            quality_gate = root / "episode_quality_gate.json"
            queue_manifest = root / "episode_regeneration_queue.json"
            delivery_manifest = root / "delivery_manifest.json"
            delivery_episode = root / "delivery_episode.mp4"
            latest_delivery_manifest = root / "latest_delivery_manifest.json"
            latest_delivery_episode = root / "latest_delivery_episode.mp4"
            production_package_path = root / "episode_production_package.json"
            for path in (
                final_render,
                quality_gate,
                queue_manifest,
                delivery_manifest,
                delivery_episode,
                latest_delivery_manifest,
                latest_delivery_episode,
                production_package_path,
            ):
                path.write_text("x", encoding="utf-8")

            cfg = {"render": {"width": 1920, "height": 1080, "fps": 24}}
            artifacts = {
                "episode_id": "episode_001",
                "display_title": "Episode 001",
                "render_mode": "finished",
                "final_render": str(final_render),
                "production_package": str(production_package_path),
                "quality_gate_report": str(quality_gate),
                "release_gate": {"passed": True, "weak_scene_count": 0},
                "release_gate_passed": True,
                "quality_gate_warnings": ["none"],
                "regeneration_queue_manifest": str(queue_manifest),
                "regeneration_requested_scene_ids": ["scene_001"],
                "delivery_manifest": str(delivery_manifest),
                "delivery_episode": str(delivery_episode),
                "latest_delivery_manifest": str(latest_delivery_manifest),
                "latest_delivery_episode": str(latest_delivery_episode),
            }
            production_package = {"scenes": []}

            payload = STEP51.build_common_export_payload(cfg, artifacts, production_package, "json", root / "export")

            self.assertEqual(payload["render_profile"], {"width": 1920, "height": 1080, "fps": 24.0})
            self.assertEqual(payload["quality_gate_report"], str(quality_gate))
            self.assertTrue(payload["release_gate_passed"])
            self.assertEqual(payload["release_gate"]["weak_scene_count"], 0)
            self.assertEqual(payload["quality_gate_warnings"], ["none"])
            self.assertEqual(payload["regeneration_queue_manifest"], str(queue_manifest))
            self.assertEqual(payload["regeneration_requested_scene_ids"], ["scene_001"])
            self.assertEqual(payload["delivery_manifest"], str(delivery_manifest))
            self.assertEqual(payload["delivery_episode"], str(delivery_episode))
            self.assertEqual(payload["latest_delivery_manifest"], str(latest_delivery_manifest))
            self.assertEqual(payload["latest_delivery_episode"], str(latest_delivery_episode))

    def test_copy_referenced_media_copies_delivery_and_gate_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            media_root = root / "media"
            paths = {}
            for name in (
                "final_render.mp4",
                "delivery_manifest.json",
                "delivery_episode.mp4",
                "latest_delivery_manifest.json",
                "latest_delivery_episode.mp4",
                "quality_gate_report.json",
                "regeneration_queue_manifest.json",
            ):
                path = root / name
                path.write_text(name, encoding="utf-8")
                paths[name] = path

            payload = {
                "final_render": str(paths["final_render.mp4"]),
                "delivery_manifest": str(paths["delivery_manifest.json"]),
                "delivery_episode": str(paths["delivery_episode.mp4"]),
                "latest_delivery_manifest": str(paths["latest_delivery_manifest.json"]),
                "latest_delivery_episode": str(paths["latest_delivery_episode.mp4"]),
                "quality_gate_report": str(paths["quality_gate_report.json"]),
                "regeneration_queue_manifest": str(paths["regeneration_queue_manifest.json"]),
                "scenes": [],
            }

            copied = STEP51.copy_referenced_media(payload, media_root)

            self.assertIn("delivery_manifest", copied)
            self.assertIn("delivery_episode", copied)
            self.assertIn("latest_delivery_manifest", copied)
            self.assertIn("latest_delivery_episode", copied)
            self.assertIn("quality_gate_report", copied)
            self.assertIn("regeneration_queue_manifest", copied)
            self.assertTrue(Path(copied["quality_gate_report"]).exists())
            self.assertTrue(Path(copied["delivery_episode"]).exists())

    def test_export_for_davinci_uses_render_profile_from_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = {
                "episode_id": "episode_001",
                "display_title": "Episode 001",
                "production_readiness": "ready",
                "quality_percent": 88,
                "release_gate_passed": True,
                "render_profile": {"width": 3840, "height": 2160, "fps": 23.976},
                "scenes": [],
            }

            result = STEP51.export_for_davinci(payload, root)
            timeline = STEP51.read_json(Path(result["timeline_path"]), {})

            self.assertEqual(timeline["resolution"]["width"], 3840)
            self.assertEqual(timeline["resolution"]["height"], 2160)
            self.assertEqual(timeline["resolution"]["fps"], 23.976)
            self.assertTrue(timeline["release_gate_passed"])

    def test_export_for_premiere_escapes_special_characters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = {
                "episode_id": "episode_001",
                "display_title": "Episode: \"Test\" & <Special>",
                "scenes": [
                    {
                        "scene_id": "scene_001",
                        "title": "Scene with 'quotes' & <tags>",
                        "duration_seconds": 10.5,
                        "video_path": "",
                        "frame_path": "",
                    }
                ],
            }

            result = STEP51.export_for_premiere(payload, root)
            project_path = Path(result["project_path"])
            self.assertTrue(project_path.exists())
            content = project_path.read_text(encoding="utf-8")
            # Ensure special characters are escaped
            self.assertNotIn("<Special>", content)
            self.assertIn("&lt;Special&gt;", content)
            self.assertNotIn("& <tags>", content)
            self.assertIn("&lt;tags&gt;", content)
            self.assertIn("&amp;", content)
            self.assertEqual(result["clip_count"], 1)

    def test_export_for_json_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = {
                "episode_id": "episode_001",
                "display_title": "Episode 001",
                "scenes": [],
            }

            result = STEP51.export_for_json(payload, root)
            manifest_path = Path(result["manifest_path"])
            self.assertTrue(manifest_path.exists())
            data = STEP51.read_json(manifest_path, {})
            self.assertEqual(data["episode_id"], "episode_001")
            self.assertEqual(data["display_title"], "Episode 001")


if __name__ == "__main__":
    unittest.main()



