from __future__ import annotations

import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline_common import queue_scenes_for_regeneration, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP15 = load_module("15_render_episode.py", "step15_regeneration")
STEP52 = load_module("52_quality_gate.py", "step52_regeneration")
STEP53 = load_module("53_regenerate_weak_scenes.py", "step53_regeneration")


class RegenerationQueueTests(unittest.TestCase):
    def test_build_auto_retry_command_honors_retry_config(self) -> None:
        command = STEP52.build_auto_retry_command(
            {
                "release_mode": {
                    "max_regeneration_retries": 5,
                    "auto_retry_update_bible": True,
                }
            },
            "episode_001",
            argparse.Namespace(
                min_quality=0.81,
                max_weak_scenes=1,
                max_regeneration_batch=4,
            ),
            strict=True,
        )

        self.assertEqual(command[2:6], ["--episode-id", "episode_001", "--apply", "--max-regeneration-retries"])
        self.assertEqual(command[6], "5")
        self.assertIn("--min-quality", command)
        self.assertIn("0.81", command)
        self.assertIn("--max-weak-scenes", command)
        self.assertIn("1", command)
        self.assertIn("--max-regeneration-batch", command)
        self.assertIn("4", command)
        self.assertIn("--strict", command)
        self.assertIn("--update-bible", command)
        self.assertTrue(str(command[1]).endswith("53_regenerate_weak_scenes.py"))

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

    def test_quality_gate_main_propagates_auto_retry_failure(self) -> None:
        args = argparse.Namespace(
            episode_id="episode_001",
            min_quality=None,
            max_weak_scenes=None,
            max_regeneration_batch=None,
            max_regeneration_retries=None,
            strict=False,
            print_json=False,
            auto_retry=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "episode_001_quality_gate.json"
            with mock.patch.object(STEP52, "parse_args", return_value=args), mock.patch.object(
                STEP52, "load_config", return_value={"release_mode": {"max_regeneration_retries": 3}}
            ), mock.patch.object(
                STEP52,
                "resolve_episode_artifacts",
                return_value={"episode_id": "episode_001", "display_title": "Episode 001"},
            ), mock.patch.object(
                STEP52, "load_scene_quality_rows", return_value=[{"scene_id": "scene_001", "quality_score": 0.31}]
            ), mock.patch.object(
                STEP52, "release_quality_gate", return_value={"passed": False, "min_quality_required": 0.68, "weak_scene_count": 1, "max_weak_scenes_allowed": 2}
            ), mock.patch.object(
                STEP52, "queue_scenes_for_regeneration", return_value=[{"scene_id": "scene_001", "quality_percent": 31}]
            ), mock.patch.object(
                STEP52, "build_warnings", return_value=[]
            ), mock.patch.object(
                STEP52, "quality_gate_report_path", return_value=report_path
            ), mock.patch.object(
                STEP52, "persist_quality_gate_result"
            ), mock.patch.object(
                STEP52, "headline"
            ), mock.patch.object(
                STEP52, "info"
            ), mock.patch.object(
                STEP52, "ok"
            ), mock.patch.object(
                STEP52.subprocess, "run", return_value=mock.Mock(returncode=7)
            ):
                with self.assertRaises(SystemExit) as exc:
                    STEP52.main()
        self.assertEqual(exc.exception.code, 7)

    def test_quality_gate_main_accepts_successful_auto_retry(self) -> None:
        args = argparse.Namespace(
            episode_id="episode_001",
            min_quality=None,
            max_weak_scenes=None,
            max_regeneration_batch=None,
            max_regeneration_retries=None,
            strict=False,
            print_json=False,
            auto_retry=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "episode_001_quality_gate.json"
            with mock.patch.object(STEP52, "parse_args", return_value=args), mock.patch.object(
                STEP52, "load_config", return_value={"release_mode": {"max_regeneration_retries": 3}}
            ), mock.patch.object(
                STEP52,
                "resolve_episode_artifacts",
                return_value={"episode_id": "episode_001", "display_title": "Episode 001"},
            ), mock.patch.object(
                STEP52, "load_scene_quality_rows", return_value=[{"scene_id": "scene_001", "quality_score": 0.31}]
            ), mock.patch.object(
                STEP52, "release_quality_gate", return_value={"passed": False, "min_quality_required": 0.68, "weak_scene_count": 1, "max_weak_scenes_allowed": 2}
            ), mock.patch.object(
                STEP52, "queue_scenes_for_regeneration", return_value=[{"scene_id": "scene_001", "quality_percent": 31}]
            ), mock.patch.object(
                STEP52, "build_warnings", return_value=[]
            ), mock.patch.object(
                STEP52, "quality_gate_report_path", return_value=report_path
            ), mock.patch.object(
                STEP52, "persist_quality_gate_result"
            ), mock.patch.object(
                STEP52,
                "reload_quality_gate_report",
                return_value=(
                    {"release_mode": {"max_regeneration_retries": 3}},
                    {"episode_id": "episode_001", "display_title": "Episode 001"},
                    report_path,
                    {"release_gate": {"passed": True}, "strict_fail": False},
                ),
            ), mock.patch.object(
                STEP52, "headline"
            ), mock.patch.object(
                STEP52, "info"
            ), mock.patch.object(
                STEP52, "ok"
            ), mock.patch.object(
                STEP52.subprocess, "run", return_value=mock.Mock(returncode=0)
            ):
                STEP52.main()


if __name__ == "__main__":
    unittest.main()
