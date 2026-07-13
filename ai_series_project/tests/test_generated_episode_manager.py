from __future__ import annotations

import importlib.util
import json
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


STEP25 = load_module("gui.py", "generated_episode_manager_gui")


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

    def test_list_season_asset_records_finds_intro_and_outro(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            intro_root = root / "generation/season_assets/season_01/intro"
            outro_root = root / "generation/season_assets/season_01/outro"
            intro_video = intro_root / "intro.mp4"
            outro_video = outro_root / "outro.mp4"
            intro_root.mkdir(parents=True)
            outro_root.mkdir(parents=True)
            intro_video.write_text("intro", encoding="utf-8")
            outro_video.write_text("outro", encoding="utf-8")
            (intro_root / "intro_manifest.json").write_text(
                '{"canonical_video": "%s", "duration_seconds": 12.5, "source_origin": "backend_generated"}'
                % intro_video.as_posix(),
                encoding="utf-8",
            )
            (outro_root / "outro_manifest.json").write_text(
                '{"canonical_video": "%s", "duration_seconds": 5.0, "source_origin": "approved_source"}'
                % outro_video.as_posix(),
                encoding="utf-8",
            )

            with temp_project_patches(root):
                records = STEP25.list_season_asset_records({"generation": {"default_season_id": "season_01"}})

            by_id = {record.asset_id: record for record in records}
            self.assertIn("season_01_intro", by_id)
            self.assertIn("season_01_outro", by_id)
            self.assertEqual(by_id["season_01_intro"].status, "backend_generated")
            self.assertEqual(by_id["season_01_outro"].status, "approved_source")
            self.assertEqual(STEP25.target_path_for_asset(by_id["season_01_intro"], "video"), intro_video)

    def test_delete_season_asset_outputs_does_not_touch_configured_source_intro(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            asset_root = root / "generation/season_assets/season_01/intro"
            source_intro = root / "assets/season_intros/season_01/intro.mp4"
            asset_root.mkdir(parents=True)
            source_intro.parent.mkdir(parents=True)
            (asset_root / "intro.mp4").write_text("generated intro", encoding="utf-8")
            (asset_root / "intro_manifest.json").write_text("{}", encoding="utf-8")
            source_intro.write_text("approved source", encoding="utf-8")

            cfg = {
                "generation": {"default_season_id": "season_01"},
                "season_intro": {
                    "profiles": {
                        "season_01": {"source_video": "assets/season_intros/season_01/intro.mp4"}
                    }
                },
            }
            with temp_project_patches(root):
                record = STEP25.build_season_asset_record(cfg, "season_01", "intro")
                result = STEP25.delete_season_asset_outputs(record)

            self.assertFalse(asset_root.exists())
            self.assertTrue(source_intro.exists())
            self.assertEqual(len(result["deleted"]), 1)

    def test_archive_season_asset_prunes_nested_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            asset_root = root / "generation/season_assets/season_01/outro"
            asset_root.mkdir(parents=True)
            (asset_root / "outro.mp4").write_text("outro", encoding="utf-8")
            (asset_root / "outro_manifest.json").write_text("{}", encoding="utf-8")

            with temp_project_patches(root):
                record = STEP25.build_season_asset_record({}, "season_01", "outro")
                candidates = STEP25.archive_candidates_for_season_asset(record)
                manifest = STEP25.archive_season_asset(record, dry_run=True)

            self.assertEqual(candidates, [asset_root])
            self.assertEqual(len(manifest["moved"]), 1)

    def test_selected_or_checked_asset_records_uses_selection_when_nothing_checked(self) -> None:
        intro = STEP25.SeasonAssetRecord(asset_id="season_01_intro", season_id="season_01", asset_kind="intro", display_title="Intro")
        outro = STEP25.SeasonAssetRecord(asset_id="season_01_outro", season_id="season_01", asset_kind="outro", display_title="Outro")

        selected = STEP25.selected_or_checked_asset_records([intro, outro], set(), "season_01_outro")

        self.assertEqual(selected, [outro])

    def test_selected_or_checked_asset_records_prefers_checked_batch(self) -> None:
        intro = STEP25.SeasonAssetRecord(asset_id="season_01_intro", season_id="season_01", asset_kind="intro", display_title="Intro")
        outro = STEP25.SeasonAssetRecord(asset_id="season_01_outro", season_id="season_01", asset_kind="outro", display_title="Outro")

        selected = STEP25.selected_or_checked_asset_records(
            [intro, outro],
            {"season_01_intro"},
            "season_01_outro",
        )

        self.assertEqual(selected, [intro])

    def test_selected_or_checked_asset_records_can_fallback_to_single_actionable_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            intro_root = root / "generation/season_assets/season_01/intro"
            intro_root.mkdir(parents=True)
            (intro_root / "intro.mp4").write_text("intro", encoding="utf-8")
            intro = STEP25.SeasonAssetRecord(
                asset_id="season_01_intro",
                season_id="season_01",
                asset_kind="intro",
                display_title="Intro",
                paths={"folder": str(intro_root)},
            )
            outro = STEP25.SeasonAssetRecord(
                asset_id="season_01_outro",
                season_id="season_01",
                asset_kind="outro",
                display_title="Outro",
            )

            with temp_project_patches(root):
                selected = STEP25.selected_or_checked_asset_records(
                    [intro, outro],
                    set(),
                    "",
                    fallback_to_single_actionable=True,
                )

            self.assertEqual(selected, [intro])

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
            formatted = STEP25.format_live_status(status)
            self.assertIn("worker-1", formatted)
            self.assertIn("Season assets", formatted)

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
                status = STEP25.build_live_generation_status({}, include_step_autosaves=True)

            self.assertFalse(status.active)
            self.assertTrue(status.stale)
            self.assertEqual(status.status, "stale")
            self.assertEqual(len(status.stale_steps), 1)

    def test_live_status_does_not_count_old_heartbeat_lease_as_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            status_path = root / "runtime/autosaves/24_process_next_episode/current_status.json"
            lease_path = root / "runtime/distributed/16_run_storyboard_backend/episodes/folge_01.json"
            status_path.parent.mkdir(parents=True)
            lease_path.parent.mkdir(parents=True)
            old = time.time() - STEP25.LIVE_STALE_SECONDS - 60
            status_path.write_text(
                '{"status": "running", "updated_at": %s, "current_step": "16_run_storyboard_backend.py"}' % old,
                encoding="utf-8",
            )
            lease_path.write_text(
                """
                {
                  "owner_id": "worker-old",
                  "heartbeat_at": %s,
                  "expires_at": %s,
                  "meta": {
                    "worker_id": "worker-old",
                    "step": "16_run_storyboard_backend",
                    "episode_id": "folge_01"
                  }
                }
                """
                % (old, time.time() + 3600),
                encoding="utf-8",
            )

            with temp_project_patches(root):
                status = STEP25.build_live_generation_status({})

            self.assertFalse(status.active)
            self.assertTrue(status.stale)
            self.assertEqual(status.status, "stale")
            self.assertIn("stale heartbeats", STEP25.format_live_status(status))

    def test_live_status_is_idle_without_runtime_status_or_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            with temp_project_patches(root):
                status = STEP25.build_live_generation_status({})

            self.assertFalse(status.active)
            self.assertFalse(status.stale)
            self.assertEqual(status.status, "idle")
            self.assertIn("No active generation worker", STEP25.format_live_status(status))

    def test_project_storage_lists_imports_databases_and_read_only_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "data/raw/episodes/source.mp4"
            face_map = root / "characters/face_map.json"
            template = root / "configs/project.template.json"
            source.parent.mkdir(parents=True)
            face_map.parent.mkdir(parents=True)
            template.parent.mkdir(parents=True)
            source.write_bytes(b"source")
            face_map.write_text('{"face_001": "Babe"}', encoding="utf-8")
            template.write_text('{"name": "template"}', encoding="utf-8")

            with temp_project_patches(root):
                records = STEP25.list_project_storage_records()

            by_path = {record.relative_path: record for record in records}
            self.assertEqual(by_path["data/raw/episodes/source.mp4"].kind, "Imported media")
            self.assertTrue(by_path["characters/face_map.json"].editable_json)
            self.assertFalse(by_path["configs/project.template.json"].editable_json)
            self.assertFalse(by_path["configs/project.template.json"].delete_allowed)

    def test_project_storage_roots_are_immediately_available_before_detail_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root = root / "data/raw"
            characters_root = root / "characters"
            data_root.mkdir(parents=True)
            characters_root.mkdir(parents=True)
            for index in range(80):
                (data_root / f"source_{index:03d}.mp4").write_bytes(b"source")

            with temp_project_patches(root):
                roots = STEP25.list_project_storage_root_records()
                records = STEP25.list_project_storage_records(max_records=12)

            root_paths = {record.relative_path for record in roots}
            all_paths = {record.relative_path for record in records}
            self.assertEqual(root_paths, {"characters", "data"})
            self.assertTrue(root_paths.issubset(all_paths))
            self.assertLessEqual(len(records), 12)

    def test_project_json_save_creates_project_local_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            face_map = root / "characters/face_map.json"
            face_map.parent.mkdir(parents=True)
            face_map.write_text('{"face_001": "Babe"}', encoding="utf-8")

            with temp_project_patches(root):
                scope = STEP25.project_storage_scope_for_path(face_map)
                self.assertIsNotNone(scope)
                record = STEP25.project_storage_record_for_path(face_map, scope)
                result = STEP25.save_json_database(record, json.dumps({"face_001": "Kenzie"}))

            self.assertEqual(json.loads(face_map.read_text(encoding="utf-8")), {"face_001": "Kenzie"})
            backup = Path(str(result["backup_path"]))
            self.assertTrue(backup.exists())
            self.assertEqual(json.loads(backup.read_text(encoding="utf-8")), {"face_001": "Babe"})

    def test_project_storage_archive_and_delete_stay_inside_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_folder = root / "data/raw/episodes"
            source_file = source_folder / "source.mp4"
            source_folder.mkdir(parents=True)
            source_file.write_bytes(b"source")

            with temp_project_patches(root):
                scope = STEP25.project_storage_scope_for_path(source_folder)
                self.assertIsNotNone(scope)
                record = STEP25.project_storage_record_for_path(source_folder, scope)
                archived = STEP25.archive_project_storage_records([record])

                archived_target = Path(str(archived["moved"][0]["target"]))
                archived_scope = STEP25.project_storage_scope_for_path(archived_target)
                self.assertIsNotNone(archived_scope)
                archived_record = STEP25.project_storage_record_for_path(archived_target, archived_scope)
                self.assertTrue(archived_target.exists())
                deleted = STEP25.delete_project_storage_records([archived_record])

            self.assertFalse(source_folder.exists())
            self.assertEqual(deleted["deleted"], [str(archived_target)])
            self.assertFalse(archived_target.exists())

    def test_default_gui_start_uses_lightweight_manager_config(self) -> None:
        calls: list[dict] = []

        def fake_run_gui(cfg: dict) -> None:
            calls.append(cfg)

        with mock.patch.object(STEP25, "load_manager_config", return_value={"paths": {}}), mock.patch.object(
            STEP25, "run_gui", fake_run_gui
        ):
            STEP25.main([])

        self.assertEqual(calls, [{"paths": {}}])

    def test_no_arguments_default_to_gui_for_double_click(self) -> None:
        self.assertTrue(STEP25.should_open_gui_by_default([]))
        self.assertFalse(STEP25.should_open_gui_by_default(["--list"]))


if __name__ == "__main__":
    unittest.main()
