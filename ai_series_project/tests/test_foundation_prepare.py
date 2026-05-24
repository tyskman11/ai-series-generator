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
import wave
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


STEP09 = load_module("09_prepare_foundation_training.py", "step09_foundation_prepare")


def write_tiny_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 1600)


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

    def test_voice_reference_candidates_ignore_directory_reference_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            voice_samples = project_root / "voice_samples"
            voice_models = project_root / "voice_models"
            voice_samples.mkdir(parents=True, exist_ok=True)
            voice_models.mkdir(parents=True, exist_ok=True)
            valid_sample = voice_samples / "triple_g_ref.wav"
            valid_sample.write_bytes(b"demo")
            STEP09.write_json(
                voice_models / "triple_g_voice_model.json",
                {"reference_audio": "."},
            )
            cfg = {
                "paths": {
                    "voice_samples": str(voice_samples),
                    "voice_models": str(voice_models),
                }
            }

            with mock.patch.object(
                STEP09,
                "resolve_project_path",
                side_effect=lambda relative_path: project_root / relative_path,
            ):
                candidates = STEP09.voice_reference_candidates(cfg, "Triple G")

        self.assertEqual(candidates, [valid_sample])

    def test_prepare_character_dataset_ignores_directory_audio_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            voice_samples = project_root / "voice_samples"
            voice_models = project_root / "voice_models"
            foundation_frames = project_root / "foundation_frames"
            foundation_video = project_root / "foundation_video"
            foundation_voice = project_root / "foundation_voice"
            for path in (voice_samples, voice_models, foundation_frames, foundation_video, foundation_voice):
                path.mkdir(parents=True, exist_ok=True)

            valid_reference = voice_samples / "triple_g_ref.wav"
            valid_reference.write_bytes(b"wav")
            STEP09.write_json(
                voice_models / "triple_g_voice_model.json",
                {"reference_audio": "."},
            )

            cfg = {
                "foundation_training": {
                    "max_frame_samples_per_character": 0,
                    "max_video_clips_per_character": 0,
                    "max_voice_segments_per_character": 4,
                },
                "paths": {
                    "voice_samples": str(voice_samples),
                    "voice_models": str(voice_models),
                    "foundation_frames": str(foundation_frames),
                    "foundation_video": str(foundation_video),
                    "foundation_voice": str(foundation_voice),
                },
            }
            character = {"name": "Triple G", "slug": "triple_g"}
            rows = [
                {
                    "episode_id": "e01",
                    "scene_id": "s01",
                    "transcript_segments": [
                        {
                            "speaker_name": "Triple G",
                            "audio_file": ".",
                            "start": 0.0,
                            "end": 1.0,
                            "text": "ignored directory",
                        }
                    ],
                }
            ]

            with mock.patch.object(
                STEP09,
                "resolve_project_path",
                side_effect=lambda relative_path: project_root / relative_path,
            ):
                manifest = STEP09.prepare_character_dataset(Path("ffmpeg"), character, rows, cfg, force=False)

        self.assertEqual(len(manifest["voice_samples"]), 1)
        self.assertEqual(manifest["voice_samples"][0]["source_type"], "curated_reference")
        self.assertTrue(manifest["voice_samples"][0]["path"].endswith("triple_g_001.wav"))

    def test_prepare_character_dataset_requires_visible_character_for_visual_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            video_file = project_root / "episode.mp4"
            video_file.write_bytes(b"not-real-video-but-mocked")
            foundation_frames = project_root / "foundation_frames"
            foundation_video = project_root / "foundation_video"
            foundation_voice = project_root / "foundation_voice"
            voice_samples = project_root / "voice_samples"
            voice_models = project_root / "voice_models"
            for path in (foundation_frames, foundation_video, foundation_voice, voice_samples, voice_models):
                path.mkdir(parents=True, exist_ok=True)
            cfg = {
                "foundation_training": {
                    "max_frame_samples_per_character": 4,
                    "max_video_clips_per_character": 4,
                    "max_voice_segments_per_character": 0,
                },
                "paths": {
                    "voice_samples": str(voice_samples),
                    "voice_models": str(voice_models),
                    "foundation_frames": str(foundation_frames),
                    "foundation_video": str(foundation_video),
                    "foundation_voice": str(foundation_voice),
                },
            }
            character = {"name": "Babe Carano", "slug": "babe_carano"}
            rows = [
                {
                    "episode_id": "e01",
                    "scene_id": "wrong_face",
                    "video_file": str(video_file),
                    "duration_seconds": 4.0,
                    "transcript_segments": [
                        {
                            "speaker_name": "Babe Carano",
                            "visible_character_names": ["Kenzie Bell"],
                            "start": 0.0,
                            "end": 2.0,
                            "text": "Babe ist nicht im Bild.",
                        }
                    ],
                },
                {
                    "episode_id": "e01",
                    "scene_id": "right_face",
                    "video_file": str(video_file),
                    "duration_seconds": 4.0,
                    "transcript_segments": [
                        {
                            "speaker_name": "Babe Carano",
                            "visible_character_names": ["Babe Carano"],
                            "start": 1.0,
                            "end": 3.0,
                            "text": "Babe ist sichtbar.",
                        }
                    ],
                },
            ]

            with mock.patch.object(STEP09, "export_frame", return_value=True), mock.patch.object(
                STEP09,
                "export_clip",
                return_value=True,
            ):
                manifest = STEP09.prepare_character_dataset(Path("ffmpeg"), character, rows, cfg, force=False)

        self.assertEqual([sample["scene_id"] for sample in manifest["frame_samples"]], ["right_face"])
        self.assertEqual([sample["scene_id"] for sample in manifest["video_samples"]], ["right_face"])
        self.assertEqual(manifest["sample_diagnostics"]["visual_rows_without_character_evidence"], 1)

    def test_prepare_character_dataset_counts_all_voice_sample_languages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            voice_a = project_root / "a.wav"
            voice_b = project_root / "b.wav"
            write_tiny_wav(voice_a)
            write_tiny_wav(voice_b)
            foundation_frames = project_root / "foundation_frames"
            foundation_video = project_root / "foundation_video"
            foundation_voice = project_root / "foundation_voice"
            voice_samples = project_root / "voice_samples"
            voice_models = project_root / "voice_models"
            for path in (foundation_frames, foundation_video, foundation_voice, voice_samples, voice_models):
                path.mkdir(parents=True, exist_ok=True)
            cfg = {
                "transcription": {"language": "de"},
                "foundation_training": {
                    "max_frame_samples_per_character": 0,
                    "max_video_clips_per_character": 0,
                    "max_voice_segments_per_character": 4,
                },
                "paths": {
                    "voice_samples": str(voice_samples),
                    "voice_models": str(voice_models),
                    "foundation_frames": str(foundation_frames),
                    "foundation_video": str(foundation_video),
                    "foundation_voice": str(foundation_voice),
                },
            }
            character = {"name": "Babe Carano", "slug": "babe_carano"}
            rows = [
                {
                    "episode_id": "e01",
                    "scene_id": "s01",
                    "transcript_segments": [
                        {
                            "speaker_name": "Babe Carano",
                            "speaker_face_cluster": "face_babe",
                            "audio_file": str(voice_a),
                            "start": 0.0,
                            "end": 1.0,
                            "language": "de",
                            "text": "Erste saubere Referenz.",
                        },
                        {
                            "speaker_name": "Babe Carano",
                            "speaker_face_cluster": "face_babe",
                            "audio_file": str(voice_b),
                            "start": 1.2,
                            "end": 2.5,
                            "language": "de",
                            "text": "Zweite saubere Referenz.",
                        },
                    ],
                }
            ]

            manifest = STEP09.prepare_character_dataset(
                Path("ffmpeg"),
                character,
                rows,
                cfg,
                force=False,
                known_clusters={"face_babe": "Babe Carano"},
            )

        self.assertEqual(manifest["voice_language_counts"], {"de": 2})


if __name__ == "__main__":
    unittest.main()



