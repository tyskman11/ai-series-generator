from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def load_module(filename: str, module_name: str):
    target = SCRIPT_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP03 = load_module("03_diarize_and_transcribe.py", "step03_shared_finalize")


class SharedFinalizeTests(unittest.TestCase):
    def test_forced_transcription_language_locks_segment_language(self) -> None:
        class FakeModel:
            def transcribe(self, *_args, **_kwargs):
                return {
                    "language": "en",
                    "text": "Drop that what?",
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "text": "Drop that what?",
                            "language": "en",
                        }
                    ],
                }

        cfg = {
            "transcription": {
                "task": "transcribe",
                "language": "de",
                "merge_gap_seconds": 0.35,
                "min_segment_seconds": 0.6,
            }
        }
        with mock.patch.object(STEP03, "wav_duration_seconds", return_value=1.0):
            payload = STEP03.transcribe_scene(FakeModel(), Path("scene_0387.wav"), cfg, use_fp16=False)

        self.assertEqual(payload["detected_language"], "de")
        self.assertEqual(payload["segments"][0]["language"], "de")

    def test_completed_artifacts_repair_stale_autosave(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            episode_dir = root / "scene_clips" / "Episode01"
            episode_dir.mkdir(parents=True)
            (episode_dir / "scene_0001.mp4").write_bytes(b"fake scene")
            transcript_dir = root / "speaker_transcripts"
            cache_dir = root / "speaker_segments" / "Episode01"
            transcript_dir.mkdir(parents=True)
            cache_dir.mkdir(parents=True)

            rows = [
                {
                    "process_version": STEP03.PROCESS_VERSION,
                    "scene_id": "scene_0001",
                    "segment_id": "scene_0001_seg_001",
                    "text": "Hallo zusammen.",
                    "language": "de",
                }
            ]
            (transcript_dir / "Episode01_segments.json").write_text(json.dumps(rows), encoding="utf-8")
            (cache_dir / "scene_0001.json").write_text(json.dumps(rows), encoding="utf-8")
            (cache_dir / "_speaker_clusters.json").write_text(
                json.dumps(
                    {
                        "clusters": [],
                        "process_version": STEP03.PROCESS_VERSION,
                        "scene_count": 1,
                        "segment_count": 1,
                        "detected_language": "",
                    }
                ),
                encoding="utf-8",
            )
            cfg = {
                "transcription": {"language": "auto"},
                "paths": {
                    "speaker_segments": "speaker_segments",
                    "speaker_transcripts": "speaker_transcripts",
                },
            }

            with mock.patch.object(
                STEP03,
                "resolve_project_path",
                side_effect=lambda rel: root / rel,
            ), mock.patch.object(
                STEP03,
                "completed_step_state",
                return_value={},
            ), mock.patch.object(
                STEP03,
                "load_step_autosave",
                return_value={"status": "in_progress", "worker_id": "worker-old"},
            ), mock.patch.object(
                STEP03,
                "mark_step_completed",
            ) as mark_mock:
                self.assertTrue(STEP03.episode_transcription_completed(episode_dir, cfg))

            mark_mock.assert_called_once()
            step_name, target_name, payload = mark_mock.call_args.args
            self.assertEqual(step_name, "03_diarize_and_transcribe")
            self.assertEqual(target_name, "Episode01")
            self.assertTrue(payload["self_healed_autosave"])
            self.assertEqual(payload["repaired_from_status"], "in_progress")
            self.assertEqual(payload["previous_worker_id"], "worker-old")
            self.assertEqual(payload["scene_count"], 1)
            self.assertEqual(payload["segment_count"], 1)

    def test_render_direct_lease_uses_recoverable_worker_metadata(self) -> None:
        render_source = (SCRIPT_ROOT / "17_render_episode.py").read_text(encoding="utf-8")

        self.assertIn("distributed_worker_metadata", render_source)
        self.assertIn('"step": "17_render_episode"', render_source)

    def test_shared_worker_retries_finalize_after_initial_lease_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            episode_dir = root / "scene_clips" / "Episode01"
            episode_dir.mkdir(parents=True)
            (episode_dir / "scene_0001.mp4").write_bytes(b"fake scene")
            completion_marker = root / "done.flag"
            combined_file = root / "speaker_transcripts" / "Episode01_segments.json"
            cluster_file = root / "speaker_segments" / "Episode01" / "_speaker_clusters.json"

            cfg = {
                "transcription": {"model_name": "tiny", "task": "transcribe"},
                "paths": {
                    "speaker_segments": "speaker_segments",
                    "speaker_transcripts": "speaker_transcripts",
                },
                "distributed": {
                    "enabled": True,
                    "lease_ttl_seconds": 60,
                    "heartbeat_interval_seconds": 10,
                    "poll_interval_seconds": 2,
                },
            }

            acquire_attempts: list[str] = []

            def fake_acquire(*_args, **_kwargs):
                acquire_attempts.append("attempt")
                if len(acquire_attempts) == 1:
                    return None
                return {"owner_id": "worker-b"}

            def fake_completed(_episode_dir: Path, _cfg: dict) -> bool:
                return completion_marker.exists()

            def fake_finalize(*_args, **_kwargs):
                completion_marker.write_text("done", encoding="utf-8")
                combined_file.parent.mkdir(parents=True, exist_ok=True)
                combined_file.write_text("[]", encoding="utf-8")
                cluster_file.parent.mkdir(parents=True, exist_ok=True)
                cluster_file.write_text("{}", encoding="utf-8")
                return combined_file, cluster_file, 0

            class FakeHeartbeat:
                def __init__(self, **_kwargs):
                    pass

                def start(self) -> None:
                    pass

                def stop(self) -> None:
                    pass

            with mock.patch.object(STEP03, "episode_transcription_completed", side_effect=fake_completed), mock.patch.object(
                STEP03,
                "resolve_project_path",
                side_effect=lambda rel: root / rel,
            ), mock.patch.object(
                STEP03,
                "distributed_step_runtime_root",
                return_value=root / "leases",
            ), mock.patch.object(
                STEP03,
                "load_step_autosave",
                return_value={},
            ), mock.patch.object(
                STEP03,
                "active_scene_lease_ids",
                return_value=[],
            ), mock.patch.object(
                STEP03,
                "mark_step_started",
            ), mock.patch.object(
                STEP03,
                "mark_step_completed",
            ), mock.patch.object(
                STEP03,
                "completed_step_state",
                return_value=True,
            ), mock.patch.object(
                STEP03,
                "detect_episode_language",
                return_value="de",
            ), mock.patch.object(
                STEP03,
                "config_with_transcription_language",
                side_effect=lambda local_cfg, _language: local_cfg,
            ), mock.patch.object(
                STEP03,
                "completed_scene_cache_ids",
                return_value=["scene_0001"],
            ), mock.patch.object(
                STEP03,
                "acquire_distributed_lease",
                side_effect=fake_acquire,
            ), mock.patch.object(
                STEP03,
                "DistributedLeaseHeartbeat",
                FakeHeartbeat,
            ), mock.patch.object(
                STEP03,
                "release_distributed_lease",
            ) as release_mock, mock.patch.object(
                STEP03,
                "finalize_episode_transcription_outputs",
                side_effect=fake_finalize,
            ), mock.patch.object(
                STEP03.time,
                "sleep",
            ):
                contributed = STEP03.process_episode_dir(
                    episode_dir,
                    cfg,
                    ffmpeg=Path("ffmpeg"),
                    model=object(),
                    use_fp16=False,
                    speaker_backend="speechbrain",
                    speechbrain_model=None,
                    device="cpu",
                    worker_id="worker-b",
                    shared_workers=True,
                )

        self.assertTrue(contributed)
        self.assertEqual(len(acquire_attempts), 2)
        release_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
