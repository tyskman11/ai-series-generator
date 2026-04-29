from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP09 = load_module("09_prepare_foundation_training.py", "step09_foundation_prepare")


class FoundationPrepareTests(unittest.TestCase):
    def test_resolved_download_plan_rows_reports_local_ready_state_without_download_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "downloads" / "image" / "sdxl"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "weights.bin").write_bytes(b"demo")
            STEP09.write_download_metadata(
                {"target_dir": str(target_dir)},
                {"model_id": "stabilityai/sdxl", "kind": "image", "revision": "abc123revision"},
            )

            rows = STEP09.resolved_download_plan_rows(
                [{"kind": "image", "model_id": "stabilityai/sdxl", "target_dir": str(target_dir)}],
                [],
            )

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["ready"])
        self.assertEqual(rows[0]["revision"], "abc123revision")
        self.assertFalse(rows[0]["downloaded"])

    def test_load_existing_manifests_merges_all_manifest_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_root = Path(tmpdir) / "training" / "foundation" / "manifests"
            manifest_root.mkdir(parents=True, exist_ok=True)
            STEP09.write_json(manifest_root / "babe_manifest.json", {"name": "Babe", "slug": "babe"})
            STEP09.write_json(manifest_root / "kenzie_manifest.json", {"name": "Kenzie", "slug": "kenzie"})
            cfg = {"paths": {"foundation_manifests": str(manifest_root)}}

            manifests = STEP09.load_existing_manifests(cfg)

        self.assertEqual([row["name"] for row in manifests], ["Babe", "Kenzie"])

    def test_main_marks_step_completed_when_no_candidates_exist(self) -> None:
        cfg = {
            "foundation_training": {},
            "paths": {
                "character_map": "characters/maps/character_map.json",
                "series_model": "generation/model/series_model.json",
                "datasets_video_training": "data/datasets/video_training",
                "foundation_manifests": "training/foundation/manifests",
                "foundation_plans": "training/foundation/plans",
                "foundation_downloads": "training/foundation/downloads",
            },
        }
        args = mock.Mock(
            episode=None,
            force=False,
            limit_characters=0,
            download_models=False,
            skip_downloads=True,
            shared_worker=False,
            worker_id="",
        )

        with mock.patch.object(STEP09, "rerun_in_runtime"), mock.patch.object(
            STEP09,
            "parse_args",
            return_value=args,
        ), mock.patch.object(
            STEP09,
            "load_config",
            return_value=cfg,
        ), mock.patch.object(
            STEP09,
            "detect_tool",
            return_value=Path("ffmpeg"),
        ), mock.patch.object(
            STEP09,
            "read_json",
            side_effect=[{"clusters": {}, "aliases": {}}, {}],
        ), mock.patch.object(
            STEP09,
            "read_dataset_rows",
            return_value=[],
        ), mock.patch.object(
            STEP09,
            "character_training_candidates",
            return_value=[],
        ), mock.patch.object(
            STEP09,
            "mark_step_started",
        ), mock.patch.object(
            STEP09,
            "mark_step_completed",
        ) as completed_mock:
            STEP09.main()

        completed_mock.assert_called_once()
        payload = completed_mock.call_args.args[2]
        self.assertEqual(payload["manifest_count"], 0)
        self.assertEqual(payload["candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
