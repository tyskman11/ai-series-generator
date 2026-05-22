from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = PROJECT_DIR


def load_module(filename: str, module_name: str):
    target = ROOT / filename if filename.startswith("support_scripts/") else SCRIPT_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PIPELINE = load_module("support_scripts/pipeline_common.py", "pipeline_common_step23_test")
STEP23 = load_module("24_process_next_episode.py", "step23_process_test")


class ProcessNextEpisodeTests(unittest.TestCase):
    def test_step23_imports_shared_project_root(self) -> None:
        self.assertTrue(hasattr(STEP23, "PROJECT_ROOT"))
        self.assertEqual(STEP23.PROJECT_ROOT, PIPELINE.PROJECT_ROOT)

    def test_source_episode_limit_defaults_to_all_and_supports_single(self) -> None:
        with mock.patch("sys.argv", ["24_process_next_episode.py"]):
            args = STEP23.parse_args()
        self.assertIsNone(STEP23.source_episode_limit(args))

        with mock.patch("sys.argv", ["24_process_next_episode.py", "--single"]):
            args = STEP23.parse_args()
        self.assertEqual(STEP23.source_episode_limit(args), 1)

        with mock.patch("sys.argv", ["24_process_next_episode.py", "--max-source-episodes", "3"]):
            args = STEP23.parse_args()
        self.assertEqual(STEP23.source_episode_limit(args), 3)

    def test_discover_episode_backlog_includes_raw_scene_and_inbox_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = {
                "paths": {
                    "inbox_episodes": "data/inbox/episodes",
                    "episodes": "data/raw/episodes",
                    "metadata": "data/processed/metadata",
                    "scene_clips": "data/processed/scene_clips",
                    "scene_index": "data/processed/scene_index",
                    "speaker_transcripts": "data/processed/speaker_transcripts",
                    "speaker_segments": "data/processed/speaker_segments",
                    "linked_segments": "data/processed/linked_segments",
                    "faces": "data/processed/faces",
                }
            }
            for rel_path in cfg["paths"].values():
                (root / rel_path).mkdir(parents=True, exist_ok=True)

            (root / cfg["paths"]["episodes"] / "raw_only.mp4").write_bytes(b"raw")
            scene_backlog = root / cfg["paths"]["scene_clips"] / "scene_only"
            scene_backlog.mkdir(parents=True)
            (scene_backlog / "scene_0001.mp4").write_bytes(b"scene")
            (root / cfg["paths"]["inbox_episodes"] / "inbox_only.mp4").write_bytes(b"inbox")

            completed_scene = root / cfg["paths"]["scene_clips"] / "completed"
            completed_scene.mkdir(parents=True)
            (completed_scene / "scene_0001.mp4").write_bytes(b"scene")
            (root / cfg["paths"]["scene_index"] / "completed_split_success.json").write_text(
                json.dumps({"clip_count": 1}),
                encoding="utf-8",
            )
            step03_version = STEP23.script_process_version("03_diarize_and_transcribe.py")
            step04_version = STEP23.script_process_version("04_link_faces_and_speakers.py")
            (root / cfg["paths"]["speaker_transcripts"] / "completed_segments.json").write_text(
                json.dumps([{"process_version": step03_version}]),
                encoding="utf-8",
            )
            completed_speaker_dir = root / cfg["paths"]["speaker_segments"] / "completed"
            completed_speaker_dir.mkdir(parents=True)
            (completed_speaker_dir / "_speaker_clusters.json").write_text(
                json.dumps({"process_version": step03_version}),
                encoding="utf-8",
            )
            (root / cfg["paths"]["linked_segments"] / "completed_linked_segments.json").write_text(
                json.dumps([{"segment_id": "completed_scene_0001_0001"}]),
                encoding="utf-8",
            )
            completed_faces_dir = root / cfg["paths"]["faces"] / "completed"
            completed_faces_dir.mkdir(parents=True)
            (completed_faces_dir / "completed_face_summary.json").write_text(
                json.dumps([{"scene_id": "scene_0001"}]),
                encoding="utf-8",
            )
            (completed_faces_dir / "_face_linking_success.json").write_text(
                json.dumps({"process_version": step04_version, "linked_row_count": 1, "face_scene_count": 1}),
                encoding="utf-8",
            )

            with mock.patch.object(STEP23, "resolve_project_path", side_effect=lambda rel: root / rel):
                tasks = STEP23.discover_episode_backlog(cfg, STEP23.default_state(), root / cfg["paths"]["inbox_episodes"])

        self.assertEqual([task["episode_name"] for task in tasks], ["raw_only", "scene_only", "inbox_only"])
        self.assertEqual([task["source_kind"] for task in tasks], ["raw", "processed", "inbox"])

    def test_split_step_without_marker_is_rebuilt_when_raw_file_remains(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = {
                "paths": {
                    "episodes": "data/raw/episodes",
                    "scene_clips": "data/processed/scene_clips",
                    "scene_index": "data/processed/scene_index",
                }
            }
            for rel_path in cfg["paths"].values():
                (root / rel_path).mkdir(parents=True, exist_ok=True)
            (root / cfg["paths"]["episodes"] / "aborted.mp4").write_bytes(b"raw")
            scene_dir = root / cfg["paths"]["scene_clips"] / "aborted"
            scene_dir.mkdir(parents=True)
            (scene_dir / "scene_0001.mp4").write_bytes(b"partial")

            with mock.patch.object(STEP23, "resolve_project_path", side_effect=lambda rel: root / rel):
                self.assertFalse(STEP23.split_step_completed_from_artifacts(cfg, "aborted", "aborted.mp4"))


if __name__ == "__main__":
    unittest.main()



