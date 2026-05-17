from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from support_scripts import pipeline_common


def load_module(filename: str, module_name: str):
    target = SCRIPT_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP08B = load_module("08b_analyze_behavior_model.py", "step08b_behavior_pipeline_test")
STEP08 = load_module("08_train_series_model.py", "step08_behavior_pipeline_test")
STEP14 = load_module("14_generate_episode.py", "step14_behavior_pipeline_test")
STEP17 = load_module("17_render_episode.py", "step17_behavior_pipeline_test")
STEP18 = load_module("18_quality_gate.py", "step18_behavior_pipeline_test")


def minimal_behavior_model() -> dict:
    return {
        "schema_version": 1,
        "dominant_language": "de",
        "speaking_style": {
            "Babe": {
                "average_words_per_line": 6.0,
                "energy_level": 0.66,
                "energy_label": "medium",
                "typical_phrases": ["das ist wichtig"],
                "recurring_reactions": ["moment mal"],
            },
            "Kenzie": {
                "average_words_per_line": 5.0,
                "energy_level": 0.58,
                "energy_label": "medium",
                "typical_phrases": ["ich habe eine idee"],
                "recurring_reactions": ["warte kurz"],
            },
        },
        "relationship_behavior": {
            "Babe||Kenzie": {
                "characters": ["Babe", "Kenzie"],
                "typical_dynamic": "Babe escalates; Kenzie reframes the problem.",
                "conversation_leader": "Babe",
            }
        },
        "dialogue_patterns": {
            "setup_reaction_punchline": {
                "default_pattern": "setup -> reaction -> complication -> punchline/callback"
            }
        },
        "defaults": {
            "line_pace": "natural",
            "line_energy": 0.52,
            "line_emotion": "focused",
            "scene_conflict": "small misunderstanding escalates and resolves within the scene",
        },
    }


class BehaviorModelPipelineTests(unittest.TestCase):
    def test_behavior_model_is_created_from_minimal_rows(self) -> None:
        rows = [
            {
                "episode_id": "ep_01",
                "scene_id": "scene_0001",
                "duration_seconds": 24.0,
                "language": "de",
                "transcript": "Babe: Moment mal, das ist wichtig!\nKenzie: Ich habe eine Idee?",
            }
        ]

        model, summary = STEP08B.build_behavior_model(
            {},
            rows=rows,
            relationship_payload={"relationships": []},
            series_model={},
        )

        self.assertEqual(model["source_counts"]["line_records"], 2)
        self.assertIn("Babe", model["speaking_style"])
        self.assertIn("scene_behavior", model)
        self.assertIn("Behavior Model Summary", summary)

    def test_behavior_model_missing_data_uses_defaults(self) -> None:
        model, summary = STEP08B.build_behavior_model(
            {},
            rows=[],
            relationship_payload={"relationships": []},
            series_model={},
        )

        self.assertTrue(model["diagnostics"])
        self.assertTrue(model["scene_behavior"]["defaulted"])
        self.assertIn("No usable speaker lines", "\n".join(model["diagnostics"]))
        self.assertIn("Diagnostics", summary)

    def test_generation_scene_contains_behavior_and_voice_metadata(self) -> None:
        cfg = copy.deepcopy(pipeline_common.DEFAULT_CONFIG)
        cfg["generation"]["min_dialogue_lines_per_scene"] = 2
        cfg["generation"]["max_dialogue_lines_per_scene"] = 3
        cfg["generation"]["target_episode_minutes_fallback"] = 5.0
        model = {
            "dataset_files": ["sample_dataset.json"],
            "main_characters": ["Babe", "Kenzie"],
            "keywords": ["controller", "arcade"],
            "speaker_line_library": {
                "Babe": [{"text": "Moment mal, das ist wichtig!", "language": "de"}],
                "Kenzie": [{"text": "Ich habe eine Idee.", "language": "de"}],
            },
            "language_counts": {"de": 4},
            "dominant_language": "de",
            "speakers": {"Babe": {}, "Kenzie": {}},
            "average_segment_duration_seconds": 2.4,
            "source_episode_durations": {"ep_01": 300.0},
            "character_reference_library": {},
            "scene_library": [],
            "behavior_model": minimal_behavior_model(),
        }

        with mock.patch.object(STEP08, "load_character_continuity_memory", return_value={"characters": {}}), mock.patch.object(
            STEP08,
            "derive_prompt_constraints_from_bible",
            return_value={"positive": ["warm sitcom light"], "negative": [], "guidance": {}},
        ):
            package, _markdown = STEP08.generate_episode_package(model, cfg, episode_index=1)

        scene = package["scenes"][0]
        self.assertTrue(scene["behavior_constraints"])
        self.assertTrue(scene["dialogue_style_constraints"])
        self.assertTrue(scene["dialogue_voice_metadata"])
        self.assertIn("scene_purpose", scene["generation_plan"])
        self.assertIn("behavior constraints", scene["generation_plan"]["positive_prompt"])

    def test_step14_storyboard_request_preserves_behavior_metadata(self) -> None:
        self.assertTrue(hasattr(STEP14.STEP08, "load_behavior_model"))
        request = STEP14.storyboard_episode_request(
            "folge_001",
            {
                "episode_id": "folge_001",
                "display_title": "Folge 1: Test",
                "scenes": [
                    {
                        "scene_id": "scene_0001",
                        "scene_purpose": "start conflict",
                        "conflict": "missing controller",
                        "behavior_constraints": ["Babe leads"],
                        "dialogue_style_constraints": ["short reactions"],
                        "dialogue_voice_metadata": [{"speaker": "Babe", "emotion": "focused"}],
                    }
                ],
            },
        )

        scene = request["scene_requests"][0]
        self.assertEqual(scene["conflict"], "missing controller")
        self.assertEqual(scene["dialogue_voice_metadata"][0]["emotion"], "focused")

    def test_render_voice_plan_and_quality_gate_read_metadata(self) -> None:
        voice_plan = STEP17.build_scene_voice_plan(
            {
                "scene_id": "scene_0001",
                "language": "de",
                "dialogue_lines": ["Babe: Moment mal, das ist wichtig!"],
                "dialogue_voice_metadata": [
                    {
                        "speaker": "Babe",
                        "emotion": "excited",
                        "pace": "quick",
                        "energy": 0.75,
                        "target_duration_seconds": 2.2,
                        "voice_reference_priority": ["trained_character_voice_model"],
                    }
                ],
            },
            4.0,
            0.0,
            {},
            {},
            {"voice_reference_max_segments": 2},
            {"voice_rate": 175, "audio_pad_seconds": 0.2},
            {"paths": {}},
        )

        self.assertEqual(voice_plan[0]["emotion"], "excited")
        self.assertEqual(voice_plan[0]["pace"], "quick")
        self.assertGreaterEqual(voice_plan[0]["target_duration_seconds"], 2.2)

        checks = STEP18.scene_content_quality_checks(
            {
                "scenes": [
                    {
                        "scene_id": "scene_0001",
                        "voice_clone": {"required": True, "lines": [{"speaker_name": "Babe", "text": "Hallo"}]},
                        "lip_sync": {"required": True, "target_outputs": {}},
                    }
                ]
            }
        )
        self.assertGreater(checks["missing_behavior_scene_count"], 0)
        self.assertGreater(checks["missing_voice_metadata_line_count"], 0)
        self.assertGreater(checks["missing_lipsync_output_scene_count"], 0)


if __name__ == "__main__":
    unittest.main()
