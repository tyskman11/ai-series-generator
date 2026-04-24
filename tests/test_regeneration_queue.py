from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from pipeline_common import queue_scenes_for_regeneration, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP15 = load_module("15_render_episode.py", "step15_regeneration")
STEP53 = load_module("53_regenerate_weak_scenes.py", "step53_regeneration")


class RegenerationQueueTests(unittest.TestCase):
    def test_queue_scenes_for_regeneration_honors_retry_limit(self) -> None:
        queue = queue_scenes_for_regeneration(
            [
                {"scene_id": "scene_a", "quality_score": 0.21, "regeneration_retries": 0},
                {"scene_id": "scene_b", "quality_score": 0.31, "regeneration_retries": 1},
                {"scene_id": "scene_c", "quality_score": 0.18, "regeneration_retries": 2},
                {"scene_id": "scene_d", "quality_score": 0.79, "regeneration_retries": 0},
            ],
            watch_threshold=0.52,
            release_threshold=0.68,
            max_regeneration_batch=8,
            max_regeneration_retries=2,
        )

        self.assertEqual([row["scene_id"] for row in queue], ["scene_a", "scene_b"])
        self.assertTrue(all(row["can_retry"] for row in queue))
        self.assertEqual(queue[0]["retry_limit"], 2)

    def test_merge_scene_regeneration_metadata_preserves_retry_history(self) -> None:
        merged = STEP15.merge_scene_regeneration_metadata(
            {"scene_id": "scene_001", "quality_score": 0.44, "quality_label": "weak"},
            {
                "regeneration_retries": 2,
                "regeneration_retry_limit": 3,
                "last_regeneration_requested_at": "2026-04-24T10:00:00Z",
                "last_regeneration_reason": "Weak scene (44%): lip sync drift",
                "queued_for_regeneration": True,
                "last_regeneration_queue_entry": {"scene_id": "scene_001", "quality_percent": 44},
            },
        )

        self.assertEqual(merged["regeneration_retries"], 2)
        self.assertEqual(merged["regeneration_retry_limit"], 3)
        self.assertEqual(merged["max_regeneration_retries"], 3)
        self.assertEqual(merged["last_regeneration_requested_at"], "2026-04-24T10:00:00Z")
        self.assertTrue(merged["queued_for_regeneration"])
        self.assertEqual(merged["last_regeneration_queue_entry"]["scene_id"], "scene_001")

    def test_regeneration_metadata_clears_apply_request_after_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            master_dir = root / "master"
            scenes_dir = root / "scenes"
            master_dir.mkdir(parents=True, exist_ok=True)
            scenes_dir.mkdir(parents=True, exist_ok=True)

            scene_path = scenes_dir / "scene_001_production.json"
            package_path = master_dir / "episode_001_production_package.json"
            manifest_path = master_dir / "episode_001_regeneration_queue.json"
            shotlist_path = root / "episode_001_shotlist.json"
            render_manifest_path = root / "episode_001_render_manifest.json"

            scene_payload = {
                "scene_id": "scene_001",
                "quality_assessment": {
                    "scene_id": "scene_001",
                    "quality_score": 0.41,
                    "regeneration_retries": 1,
                },
            }
            write_json(scene_path, scene_payload)
            write_json(
                package_path,
                {
                    "episode_id": "episode_001",
                    "scenes": [dict(scene_payload)],
                    "scene_package_paths": [str(scene_path)],
                },
            )
            write_json(shotlist_path, {"episode_id": "episode_001"})
            write_json(render_manifest_path, {"episode_id": "episode_001"})

            queue = [
                {
                    "scene_id": "scene_001",
                    "quality_percent": 41,
                    "weaknesses": ["lip sync drift"],
                }
            ]
            requested_at = "2026-04-24T10:00:00Z"
            applied_at = "2026-04-24T10:05:00Z"
            artifacts = {
                "shotlist": str(shotlist_path),
                "render_manifest": str(render_manifest_path),
            }

            STEP53.persist_package_scene_updates(
                package_path,
                queue,
                manifest_path,
                requested_at,
                3,
                increment_retry=True,
            )
            STEP53.persist_artifact_metadata(
                artifacts,
                manifest_path,
                queue,
                requested_at,
                apply_requested=True,
            )

            queued_package = read_json(package_path, {})
            self.assertTrue(queued_package["regeneration_apply_requested"])
            self.assertEqual(queued_package["scenes"][0]["quality_assessment"]["regeneration_retries"], 2)
            self.assertTrue(queued_package["scenes"][0]["quality_assessment"]["queued_for_regeneration"])

            STEP53.persist_package_scene_updates(
                package_path,
                queue,
                manifest_path,
                requested_at,
                3,
                increment_retry=False,
                applied_at=applied_at,
            )
            STEP53.persist_artifact_metadata(
                artifacts,
                manifest_path,
                queue,
                requested_at,
                apply_requested=True,
                applied_at=applied_at,
            )

            applied_package = read_json(package_path, {})
            applied_scene = applied_package["scenes"][0]["quality_assessment"]
            self.assertFalse(applied_package["regeneration_apply_requested"])
            self.assertEqual(applied_package["regeneration_last_applied_at"], applied_at)
            self.assertFalse(applied_scene["queued_for_regeneration"])
            self.assertEqual(applied_scene["last_regeneration_applied_at"], applied_at)

            updated_shotlist = read_json(shotlist_path, {})
            updated_render_manifest = read_json(render_manifest_path, {})
            self.assertFalse(updated_shotlist["regeneration_apply_requested"])
            self.assertFalse(updated_render_manifest["regeneration_apply_requested"])
            self.assertEqual(updated_shotlist["regeneration_last_applied_at"], applied_at)
            self.assertEqual(updated_render_manifest["regeneration_last_applied_at"], applied_at)


if __name__ == "__main__":
    unittest.main()
