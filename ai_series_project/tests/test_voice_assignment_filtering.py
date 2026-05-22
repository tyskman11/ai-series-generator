from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from support_scripts.pipeline_common import voice_segment_reference_eligible


def load_module(filename: str, module_name: str):
    target = SCRIPT_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP03 = load_module("03_diarize_and_transcribe.py", "step03_voice_filtering")
STEP04 = load_module("04_link_faces_and_speakers.py", "step04_voice_filtering")
STEP05 = load_module("05_review_unknowns.py", "step05_voice_filtering")
STEP09 = load_module("09_prepare_foundation_training.py", "step09_voice_filtering")
STEP17 = load_module("17_render_episode.py", "step17_voice_filtering")


def write_tiny_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 1600)


class VoiceAssignmentFilteringTests(unittest.TestCase):
    def test_common_voice_reference_filter_rejects_music_markers(self) -> None:
        self.assertFalse(
            voice_segment_reference_eligible(
                {
                    "text": "[Musik]",
                    "start": 0.0,
                    "end": 2.0,
                    "speech_confidence": 0.95,
                    "audio_content_type": "music",
                },
                {"transcription": {}},
                default_when_unscored=False,
            )
        )

    def test_speaker_clustering_keeps_music_noise_unknown(self) -> None:
        cfg = {
            "transcription": {
                "speaker_cluster_high_quality_min_seconds": 1.0,
                "speaker_cluster_min_segments": 2,
                "speaker_cluster_min_speech_confidence": 0.52,
            }
        }
        rows = [
            {
                "scene_id": "scene_0001",
                "segment_id": "seg_001",
                "start": 0.0,
                "end": 1.4,
                "text": "Hallo Babe",
                "voice_embedding": [1.0, 0.0],
                "speech_confidence": 0.86,
                "audio_content_type": "speech",
                "speaker_cluster_eligible": True,
            },
            {
                "scene_id": "scene_0002",
                "segment_id": "seg_002",
                "start": 0.0,
                "end": 1.5,
                "text": "Das ist meine Stimme",
                "voice_embedding": [0.98, 0.02],
                "speech_confidence": 0.84,
                "audio_content_type": "speech",
                "speaker_cluster_eligible": True,
            },
            {
                "scene_id": "scene_0003",
                "segment_id": "seg_music",
                "start": 0.0,
                "end": 2.0,
                "text": "[music]",
                "voice_embedding": [1.0, 0.0],
                "speech_confidence": 0.0,
                "audio_content_type": "music",
                "speaker_cluster_eligible": False,
                "voice_reference_eligible": False,
            },
        ]

        assigned_rows, clusters = STEP03.assign_speaker_clusters(rows, 0.9, cfg)

        by_id = {row["segment_id"]: row["speaker_cluster"] for row in assigned_rows}
        self.assertEqual(by_id["seg_001"], "speaker_001")
        self.assertEqual(by_id["seg_002"], "speaker_001")
        self.assertEqual(by_id["seg_music"], "speaker_unknown")
        self.assertEqual(len(clusters), 1)

    def test_face_linking_does_not_register_non_speech_speaker_clusters(self) -> None:
        voice_map = {"clusters": {}}
        added = STEP04.ensure_voice_clusters_for_transcripts(
            voice_map,
            [
                {"speaker_cluster": "speaker_001", "text": "Hallo", "speech_confidence": 0.8, "audio_content_type": "speech"},
                {
                    "speaker_cluster": "speaker_002",
                    "text": "[Applaus]",
                    "speech_confidence": 0.0,
                    "audio_content_type": "applause",
                    "speaker_cluster_eligible": False,
                },
            ],
        )

        self.assertEqual(added, 1)
        self.assertIn("speaker_001", voice_map["clusters"])
        self.assertNotIn("speaker_002", voice_map["clusters"])

    def test_review_auto_link_ignores_music_rows_even_with_single_visible_face(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            linked_root = root / "linked"
            linked_root.mkdir()
            (linked_root / "episode_linked_segments.json").write_text(
                json.dumps(
                    [
                        {
                            "speaker_cluster": "speaker_007",
                            "speaker_name": "speaker_007",
                            "visible_face_clusters": ["face_babe"],
                            "speaker_face_cluster": None,
                            "text": "[music]",
                            "speech_confidence": 0.0,
                            "audio_content_type": "music",
                            "speaker_cluster_eligible": False,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            cfg = {"paths": {"linked_segments": str(linked_root)}}
            char_map = {"clusters": {"face_babe": {"name": "Babe", "auto_named": False, "ignored": False}}}
            voice_map = {"clusters": {"speaker_007": {"name": "speaker_007", "auto_named": True}}, "aliases": {}}

            summary = STEP05.auto_link_speakers_from_single_visible_faces(cfg, char_map, voice_map)

        self.assertEqual(summary["matched"], 0)
        self.assertNotIn("linked_face_cluster", voice_map["clusters"]["speaker_007"])

    def test_foundation_voice_candidates_filter_non_speech_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clean = root / "clean.wav"
            music = root / "music.wav"
            write_tiny_wav(clean)
            write_tiny_wav(music)
            rows = [
                {
                    "episode_id": "e01",
                    "scene_id": "s01",
                    "transcript_segments": [
                        {
                            "speaker_name": "Babe",
                            "audio_file": str(music),
                            "start": 0.0,
                            "end": 2.0,
                            "text": "[Musik]",
                            "speech_confidence": 0.0,
                            "audio_content_type": "music",
                            "voice_reference_eligible": False,
                        },
                        {
                            "speaker_name": "Babe",
                            "audio_file": str(clean),
                            "start": 2.1,
                            "end": 4.1,
                            "text": "Ich bin wirklich Babe.",
                            "speech_confidence": 0.9,
                            "audio_content_type": "speech",
                            "voice_reference_eligible": True,
                        },
                    ],
                }
            ]

            candidates = STEP09.original_voice_candidates(rows, "Babe", {"transcription": {}})

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["audio_path"], clean)

    def test_render_reference_segments_skip_ineligible_voice_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clean = root / "clean.wav"
            noise = root / "noise.wav"
            clean.write_bytes(b"clean")
            noise.write_bytes(b"noise")
            library = {
                "Babe": [
                    {
                        "audio_path": str(noise),
                        "text": "[noise]",
                        "start": 0.0,
                        "end": 2.0,
                        "speech_confidence": 0.0,
                        "audio_content_type": "noise",
                        "voice_reference_eligible": False,
                    },
                    {
                        "audio_path": str(clean),
                        "text": "Das ist eine saubere Referenz.",
                        "start": 2.0,
                        "end": 4.0,
                        "speech_confidence": 0.88,
                        "audio_content_type": "speech",
                        "voice_reference_eligible": True,
                    },
                ]
            }

            segments = STEP17.collect_speaker_reference_segments(library, "Babe", 4)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["audio_path"], str(clean))


if __name__ == "__main__":
    unittest.main()
