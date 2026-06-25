from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent


def load_module(filename: str, module_name: str):
    target = SCRIPT_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP25 = load_module("25_manage_generated_episodes.py", "step25_manage_generated_episodes")


def temp_project_patches(root: Path, **extra):
    def resolve_project_path(relative_path: str) -> Path:
        return root / relative_path

    def resolve_stored_project_path(path_value: str | Path | None) -> Path:
        text = str(path_value or "")
        candidate = Path(text)
        if candidate.is_absolute():
            return candidate
        return root / text

    patches = {
        "resolve_project_path": resolve_project_path,
        "resolve_stored_project_path": resolve_stored_project_path,
        "list_generated_episode_artifacts": lambda _cfg: [],
        "generated_episode_artifacts": lambda _cfg, episode_id: {"episode_id": episode_id, "display_title": episode_id},
    }
    patches.update(extra)
    return mock.patch.multiple(STEP25, **patches)


class GeneratedEpisodeManagerTests(unittest.TestCase):
    def test_discover_episode_ids_merges_sources_and_skips_latest_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "generation/final_episode_packages/folge_01").mkdir(parents=True)
            (root / "generation/renders/deliveries/latest").mkdir(parents=True)
            (root / "generation/quality_reports").mkdir(parents=True)
            (root / "generation/quality_reports/folge_02_quality_gate.json").write_text("{}", encoding="utf-8")

            with temp_project_patches(
                root,
                list_generated_episode_artifacts=lambda _cfg: [
                    {"episode_id": "latest"},
                    {"episode_id": "folge_03"},
                ],
            ):
                episode_ids = STEP25.discover_episode_ids({})

            self.assertEqual(episode_ids, ["folge_01", "folge_02", "folge_03"])

    def test_build_episode_record_merges_quality_gate_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            quality = root / "generation/renders/deliveries/folge_01/folge_01_quality_gate.json"
            quality.parent.mkdir(parents=True)
            quality.write_text(
                """
                {
                  "readiness": "finished",
                  "quality_percent": 91,
                  "minimum_scene_quality_percent": 84,
                  "release_gate": {"passed": true},
                  "finished_episode_gate": {
                    "passed": false,
                    "realism_score": 0.71,
                    "blockers": ["missing lipsync"],
                    "warnings": ["needs remix"]
                  },
                  "regeneration_queue_size": 2
                }
                """,
                encoding="utf-8",
            )
            video = root / "generation/renders/final/folge_01.mp4"
            video.parent.mkdir(parents=True)
            video.write_text("video", encoding="utf-8")

            def artifacts(_cfg, episode_id: str) -> dict:
                return {
                    "episode_id": episode_id,
                    "display_title": "Folge 01: Test",
                    "production_readiness": "storyboard_only",
                    "scene_count": 4,
                    "generated_scene_video_count": 3,
                    "scene_dialogue_audio_count": 2,
                    "scene_master_clip_count": 1,
                    "quality_gate_report": str(quality),
                    "final_render": str(video),
                }

            with temp_project_patches(root, generated_episode_artifacts=artifacts):
                record = STEP25.build_episode_record({}, "folge_01")

            self.assertEqual(record.display_title, "Folge 01: Test")
            self.assertEqual(record.readiness, "finished")
            self.assertEqual(record.quality_percent, 91)
            self.assertEqual(record.minimum_scene_quality_percent, 84)
            self.assertTrue(record.release_gate_passed)
            self.assertFalse(record.finished_gate_passed)
            self.assertEqual(record.realism_score, 0.71)
            self.assertIn("missing lipsync", record.blockers)
            self.assertIn("needs remix", record.warnings)
            self.assertEqual(record.regeneration_queue_count, 2)
            self.assertEqual(record.paths["final_render"], str(video))

    def test_archive_candidates_prune_nested_paths_and_stay_in_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_root = root / "generation/final_episode_packages/folge_01"
            package_file = package_root / "master/folge_01_production_package.json"
            final_render = root / "generation/renders/final/folge_01.mp4"
            package_file.parent.mkdir(parents=True)
            final_render.parent.mkdir(parents=True)
            package_file.write_text("{}", encoding="utf-8")
            final_render.write_text("video", encoding="utf-8")

            record = STEP25.EpisodeRecord(
                episode_id="folge_01",
                display_title="Folge 01",
                paths={
                    "production_package_root": str(package_root),
                    "production_package": str(package_file),
                    "final_render": str(final_render),
                },
            )

            with temp_project_patches(root):
                candidates = STEP25.archive_candidates_for_episode(record)
                manifest = STEP25.archive_episode(record, dry_run=True)

            self.assertIn(package_root, candidates)
            self.assertIn(final_render, candidates)
            self.assertNotIn(package_file, candidates)
            self.assertEqual(len(manifest["moved"]), 2)
            self.assertTrue(all("generation" in row["source"] for row in manifest["moved"]))

    def test_delete_episode_outputs_removes_only_generated_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_root = root / "generation/final_episode_packages/folge_01"
            final_render = root / "generation/renders/final/folge_01.mp4"
            source_episode = root / "data/raw/episodes/source.mp4"
            package_root.mkdir(parents=True)
            final_render.parent.mkdir(parents=True)
            source_episode.parent.mkdir(parents=True)
            (package_root / "package.json").write_text("{}", encoding="utf-8")
            final_render.write_text("video", encoding="utf-8")
            source_episode.write_text("source", encoding="utf-8")

            record = STEP25.EpisodeRecord(
                episode_id="folge_01",
                display_title="Folge 01",
                paths={
                    "production_package_root": str(package_root),
                    "final_render": str(final_render),
                },
            )

            with temp_project_patches(root):
                result = STEP25.delete_episode_outputs(record)

            self.assertEqual(len(result["deleted"]), 2)
            self.assertFalse(package_root.exists())
            self.assertFalse(final_render.exists())
            self.assertTrue(source_episode.exists())

    def test_live_status_reads_current_status_and_active_worker_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            status_path = root / "runtime/autosaves/24_process_next_episode/current_status.json"
            lease_path = root / "runtime/distributed/17_render_episode/episodes/folge_01.json"
            status_path.parent.mkdir(parents=True)
            lease_path.parent.mkdir(parents=True)
            now = time.time()
            status_path.write_text(
                """
                {
                  "status": "running",
                  "updated_at": %s,
                  "current_step": "17_render_episode.py",
                  "current_episode_file": "source.mp4",
                  "global_progress": [
                    {"script_name": "16_run_storyboard_backend.py", "status": "completed"},
                    {"script_name": "17_render_episode.py", "status": "running"},
                    {"script_name": "18_quality_gate.py", "status": "pending"}
                  ],
                  "latest_generated_episode": {
                    "episode_id": "folge_01",
                    "production_readiness": "storyboard_only",
                    "scene_count": 3,
                    "generated_scene_video_count": 1
                  }
                }
                """
                % now,
                encoding="utf-8",
            )
            lease_path.write_text(
                """
                {
                  "owner_id": "worker-1",
                  "heartbeat_at": %s,
                  "expires_at": %s,
                  "meta": {
                    "worker_id": "worker-1",
                    "hostname": "host-a",
                    "step": "17_render_episode",
                    "episode_id": "folge_01",
                    "has_gpu": true
                  }
                }
                """
                % (now, now + 300),
                encoding="utf-8",
            )

            with temp_project_patches(root):
                status = STEP25.build_live_generation_status({})

            self.assertTrue(status.active)
            self.assertFalse(status.stale)
            self.assertEqual(status.current_step, "17_render_episode.py")
            self.assertEqual(status.current_episode, "folge_01")
            self.assertEqual(status.completed_steps, 2)
            self.assertEqual(status.total_steps, 3)
            self.assertEqual(len(status.active_workers), 1)
            self.assertIn("worker-1", STEP25.format_live_status(status))

    def test_live_status_marks_stale_running_autosave_without_active_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            status_path = root / "runtime/autosaves/24_process_next_episode/current_status.json"
            step_path = root / "runtime/autosaves/steps/17_render_episode/folge_01.json"
            status_path.parent.mkdir(parents=True)
            step_path.parent.mkdir(parents=True)
            old = time.time() - STEP25.LIVE_STALE_SECONDS - 120
            status_path.write_text(
                '{"status": "running", "updated_at": %s, "current_step": "17_render_episode.py"}' % old,
                encoding="utf-8",
            )
            step_path.write_text(
                '{"status": "in_progress", "updated_at": %s, "step": "17_render_episode", "target": "folge_01"}' % old,
                encoding="utf-8",
            )

            with temp_project_patches(root):
                status = STEP25.build_live_generation_status({})

            self.assertFalse(status.active)
            self.assertTrue(status.stale)
            self.assertEqual(status.status, "stale")
            self.assertEqual(len(status.stale_steps), 1)

    def test_no_arguments_default_to_gui_for_double_click(self) -> None:
        self.assertTrue(STEP25.should_open_gui_by_default([]))
        self.assertFalse(STEP25.should_open_gui_by_default(["--list"]))


if __name__ == "__main__":
    unittest.main()
