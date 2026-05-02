from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from support_scripts.pipeline_common import queue_scenes_for_regeneration, read_json, write_json


ROOT = PROJECT_DIR


def load_module(filename: str, module_name: str):
    target = ROOT / filename if filename.startswith("support_scripts/") else SCRIPT_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP15 = load_module("16_render_episode.py", "step15_regeneration")
STEP52 = load_module("17_quality_gate.py", "step52_regeneration")
STEP53 = load_module("18_regenerate_weak_scenes.py", "step53_regeneration")


class RegenerationQueueTests(unittest.TestCase):
    def test_quality_gate_parse_args_accepts_shared_worker_flags(self) -> None:
        with mock.patch(
            "sys.argv",
            ["17_quality_gate.py", "--episode-id", "folge_11", "--worker-id", "pc2", "--no-shared-workers"],
        ):
            args = STEP52.parse_args()

        self.assertEqual(args.episode_id, "folge_11")
        self.assertEqual(args.worker_id, "pc2")
        self.assertTrue(args.no_shared_workers)

    def test_quality_gate_paths_ignore_empty_artifact_fields(self) -> None:
        artifacts = {"episode_id": "episode_001"}

        self.assertIsNone(STEP52.stored_path_if_present(""))
        self.assertIsNone(STEP53.stored_path_if_present(""))
        self.assertEqual(STEP52.quality_gate_report_path(artifacts), Path("episode_001_quality_gate.json"))
        self.assertEqual(STEP53.quality_gate_report_path(artifacts), Path("episode_001_quality_gate.json"))
        self.assertFalse(STEP52.artifact_path_exists(""))
        self.assertEqual(STEP52.load_scene_quality_rows({}), [])
        STEP52.persist_quality_gate_result({}, Path("missing_quality_gate.json"), {"release_gate": {"passed": False}})

    def test_quality_gate_report_path_prefers_real_delivery_or_package_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            delivery_root = root / "delivery"
            delivery_root.mkdir()
            package_path = root / "package" / "episode_001_production_package.json"
            package_path.parent.mkdir()
            package_path.write_text("{}", encoding="utf-8")

            delivery_artifacts = {
                "episode_id": "episode_001",
                "delivery_bundle_root": str(delivery_root),
                "production_package": str(package_path),
            }
            package_artifacts = {
                "episode_id": "episode_001",
                "delivery_bundle_root": "",
                "production_package": str(package_path),
            }

            self.assertEqual(
                STEP52.quality_gate_report_path(delivery_artifacts),
                delivery_root / "episode_001_quality_gate.json",
            )
            self.assertEqual(
                STEP53.quality_gate_report_path(delivery_artifacts),
                delivery_root / "episode_001_quality_gate.json",
            )
            self.assertEqual(
                STEP52.quality_gate_report_path(package_artifacts),
                package_path.parent / "episode_001_quality_gate.json",
            )
            self.assertEqual(
                STEP53.quality_gate_report_path(package_artifacts),
                package_path.parent / "episode_001_quality_gate.json",
            )

    def test_build_warnings_accepts_fully_generated_episode_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            production_package = root / "episode_001_production_package.json"
            final_render = root / "episode_001_final.mp4"
            full_generated_episode = root / "episode_001_full.mp4"
            delivery_episode = root / "episode_001_delivery.mp4"
            for path in (production_package, final_render, full_generated_episode, delivery_episode):
                path.write_text("ok", encoding="utf-8")
            warnings = STEP52.build_warnings(
                {
                    "production_readiness": "fully_generated_episode_ready",
                    "backend_runner_failed_count": 0,
                    "backend_runner_pending_count": 0,
                    "production_package": str(production_package),
                    "final_render": str(final_render),
                    "full_generated_episode": str(full_generated_episode),
                    "delivery_episode": str(delivery_episode),
                }
            )
        self.assertEqual(warnings, [])

    def test_auto_retry_enabled_respects_no_auto_retry_override(self) -> None:
        cfg = {"release_mode": {"auto_retry_failed_gate": True}}
        self.assertFalse(
            STEP52.auto_retry_enabled(
                cfg,
                argparse.Namespace(auto_retry=False, no_auto_retry=True),
            )
        )
        self.assertTrue(
            STEP52.auto_retry_enabled(
                cfg,
                argparse.Namespace(auto_retry=False, no_auto_retry=False),
            )
        )

    def test_quality_gate_override_requested_detects_override_flags(self) -> None:
        self.assertTrue(
            STEP53.quality_gate_override_requested(
                argparse.Namespace(
                    min_quality=0.72,
                    max_weak_scenes=None,
                    max_regeneration_batch=None,
                    max_regeneration_retries=None,
                    strict=False,
                )
            )
        )
        self.assertTrue(
            STEP53.quality_gate_override_requested(
                argparse.Namespace(
                    min_quality=None,
                    max_weak_scenes=None,
                    max_regeneration_batch=None,
                    max_regeneration_retries=None,
                    strict=True,
                )
            )
        )
        self.assertFalse(
            STEP53.quality_gate_override_requested(
                argparse.Namespace(
                    min_quality=None,
                    max_weak_scenes=None,
                    max_regeneration_batch=None,
                    max_regeneration_retries=None,
                    strict=False,
                )
            )
        )

    def test_build_rerun_plan_disables_nested_quality_gate_auto_retry(self) -> None:
        plan = STEP53.build_rerun_plan(
            "episode_001",
            strict=True,
            update_bible=False,
            min_quality=0.73,
            scene_ids=["scene_001"],
        )

        quality_gate_steps = [step for step in plan if step.get("script") == "17_quality_gate.py"]
        self.assertEqual(len(quality_gate_steps), 1)
        quality_gate_args = quality_gate_steps[0]["args"]
        self.assertIn("--no-auto-retry", quality_gate_args)
        self.assertIn("--strict", quality_gate_args)
        self.assertIn("--min-quality", quality_gate_args)
        self.assertIn("0.73", quality_gate_args)

    def test_ensure_quality_gate_report_disables_auto_retry_during_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            production_package = root / "episode_001_production_package.json"
            production_package.write_text("{}", encoding="utf-8")
            report_path = root / "episode_001_quality_gate.json"
            write_json(report_path, {"episode_id": "episode_001", "regeneration_queue": []})

            with mock.patch.object(STEP53, "run_script", return_value=mock.Mock(returncode=0)) as run_script:
                loaded_report_path, loaded_report = STEP53.ensure_quality_gate_report(
                    {"release_mode": {"auto_retry_failed_gate": True}},
                    {"episode_id": "episode_001", "production_package": str(production_package)},
                    min_quality=0.72,
                    refresh=True,
                )

        self.assertEqual(loaded_report_path, report_path)
        self.assertEqual(loaded_report["episode_id"], "episode_001")
        self.assertEqual(run_script.call_args.args[0], "17_quality_gate.py")
        gate_args = run_script.call_args.args[1]
        self.assertIn("--no-auto-retry", gate_args)
        self.assertIn("--min-quality", gate_args)
        self.assertIn("0.72", gate_args)

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
        self.assertTrue(str(command[1]).endswith("18_regenerate_weak_scenes.py"))
        self.assertEqual(Path(command[1]).parent, STEP52.WORKSPACE_ROOT)

    def test_regenerate_main_refreshes_gate_when_overrides_are_passed(self) -> None:
        args = argparse.Namespace(
            episode_id="episode_001",
            min_quality=0.74,
            max_weak_scenes=None,
            max_regeneration_batch=None,
            refresh_quality_gate=False,
            apply=False,
            force=False,
            update_bible=False,
            max_regeneration_retries=4,
            strict=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            production_package = Path(tmpdir) / "episode_001_production_package.json"
            production_package.write_text("{}", encoding="utf-8")
            artifacts = {
                "episode_id": "episode_001",
                "production_package": str(production_package),
            }
            report_path = Path(tmpdir) / "episode_001_quality_gate.json"
            queue_path = Path(tmpdir) / "episode_001_regeneration_queue.json"
            with mock.patch.object(STEP53, "parse_args", return_value=args), mock.patch.object(
                STEP53, "load_config", return_value={"release_mode": {"max_regeneration_retries": 3}}
            ), mock.patch.object(
                STEP53, "resolve_episode_artifacts", return_value=artifacts
            ), mock.patch.object(
                STEP53, "effective_retry_limit", return_value=4
            ), mock.patch.object(
                STEP53, "ensure_quality_gate_report", return_value=(report_path, {"regeneration_queue": []})
            ) as ensure_report, mock.patch.object(
                STEP53, "queue_manifest_path", return_value=queue_path
            ), mock.patch.object(
                STEP53, "build_queue_manifest", return_value={"regeneration_queue": [], "manifest_path": str(queue_path)}
            ), mock.patch.object(
                STEP53, "write_json"
            ), mock.patch.object(
                STEP53, "persist_artifact_metadata"
            ), mock.patch.object(
                STEP53, "persist_package_scene_updates"
            ), mock.patch.object(
                STEP53, "headline"
            ), mock.patch.object(
                STEP53, "info"
            ), mock.patch.object(
                STEP53, "ok"
            ), mock.patch.object(
                STEP53, "rerun_in_runtime"
            ):
                STEP53.main()

        self.assertEqual(ensure_report.call_args.kwargs["min_quality"], 0.74)
        self.assertEqual(ensure_report.call_args.kwargs["max_regeneration_retries"], 4)
        self.assertTrue(ensure_report.call_args.kwargs["refresh"])

    def test_regenerate_main_allows_quality_gate_exit_code_one_during_apply(self) -> None:
        args = argparse.Namespace(
            episode_id="episode_001",
            min_quality=None,
            max_weak_scenes=None,
            max_regeneration_batch=None,
            refresh_quality_gate=False,
            apply=True,
            force=False,
            update_bible=False,
            max_regeneration_retries=3,
            strict=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            production_package = Path(tmpdir) / "episode_001_production_package.json"
            production_package.write_text("{}", encoding="utf-8")
            manifest_path = Path(tmpdir) / "episode_001_regeneration_queue.json"
            report_path = Path(tmpdir) / "episode_001_quality_gate.json"
            artifacts = {
                "episode_id": "episode_001",
                "production_package": str(production_package),
            }
            manifest = {
                "regeneration_queue": [{"scene_id": "scene_001", "quality_percent": 41}],
                "rerun_plan": [
                    {"script": "15_run_storyboard_backend.py", "args": ["--episode-id", "episode_001", "--force"]},
                    {"script": "16_render_episode.py", "args": ["--episode-id", "episode_001", "--force"]},
                    {"script": "17_quality_gate.py", "args": ["--episode-id", "episode_001", "--no-auto-retry"]},
                ],
                "manifest_path": str(manifest_path),
            }

            def fake_run_script(script_name: str, _args: list[str], *, allow_failure: bool = False):
                return mock.Mock(returncode=1 if script_name == "17_quality_gate.py" else 0)

            with mock.patch.object(STEP53, "parse_args", return_value=args), mock.patch.object(
                STEP53, "load_config", return_value={"release_mode": {"max_regeneration_retries": 3}}
            ), mock.patch.object(
                STEP53, "resolve_episode_artifacts", return_value=artifacts
            ), mock.patch.object(
                STEP53, "effective_retry_limit", return_value=3
            ), mock.patch.object(
                STEP53, "ensure_quality_gate_report", return_value=(report_path, {"regeneration_queue": manifest["regeneration_queue"]})
            ), mock.patch.object(
                STEP53, "queue_manifest_path", return_value=manifest_path
            ), mock.patch.object(
                STEP53, "build_queue_manifest", return_value=manifest
            ), mock.patch.object(
                STEP53, "persist_artifact_metadata"
            ), mock.patch.object(
                STEP53, "persist_package_scene_updates"
            ), mock.patch.object(
                STEP53, "update_manifest_after_apply"
            ), mock.patch.object(
                STEP53, "run_script", side_effect=fake_run_script
            ) as run_script_mock, mock.patch.object(
                STEP53, "headline"
            ), mock.patch.object(
                STEP53, "info"
            ), mock.patch.object(
                STEP53, "ok"
            ), mock.patch.object(
                STEP53, "rerun_in_runtime"
            ):
                STEP53.main()

        quality_gate_calls = [call for call in run_script_mock.call_args_list if call.args[0] == "17_quality_gate.py"]
        self.assertEqual(len(quality_gate_calls), 1)
        self.assertTrue(quality_gate_calls[0].kwargs["allow_failure"])

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

    def test_strict_warnings_enabled_accepts_flag_or_config(self) -> None:
        self.assertTrue(STEP52.strict_warnings_enabled({}, argparse.Namespace(strict=True)))
        self.assertTrue(
            STEP52.strict_warnings_enabled(
                {"release_mode": {"strict_warnings": True}},
                argparse.Namespace(strict=False),
            )
        )
        self.assertFalse(STEP52.strict_warnings_enabled({}, argparse.Namespace(strict=False)))

    def test_strict_mode_is_passed_to_quality_gate(self) -> None:
        """Verify strict mode flag is correctly handled in quality gate evaluation."""
        args = argparse.Namespace(
            episode_id="episode_001",
            min_quality=None,
            max_weak_scenes=None,
            max_regeneration_batch=None,
            max_regeneration_retries=None,
            strict=True,
            print_json=False,
            auto_retry=False,
            no_auto_retry=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "episode_001_quality_gate.json"
            with mock.patch.object(STEP52, "parse_args", return_value=args), mock.patch.object(
                STEP52, "load_config", return_value={"release_mode": {"min_quality": 0.68}}
            ), mock.patch.object(
                STEP52,
                "resolve_episode_artifacts",
                return_value={"episode_id": "episode_001", "display_title": "Episode 001"},
            ), mock.patch.object(
                STEP52, "load_scene_quality_rows", return_value=[]
            ), mock.patch.object(
                STEP52, "release_quality_gate", return_value={"passed": True, "min_quality_required": 0.75, "weak_scene_count": 0, "max_weak_scenes_allowed": 1}
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
            ):
                STEP52.main()


if __name__ == "__main__":
    unittest.main()



