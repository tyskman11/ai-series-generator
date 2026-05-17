from __future__ import annotations

import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


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


STEP08 = load_module("08_train_series_model.py", "step08_finished_episode_mode_test")
STEP18 = load_module("18_quality_gate.py", "step18_finished_episode_mode_test")
STEP19 = load_module("19_regenerate_weak_scenes.py", "step19_finished_episode_mode_test")
STEP21 = load_module("21_export_package.py", "step21_finished_episode_mode_test")


def minimal_series_model() -> dict:
    return {
        "dataset_files": ["sample_dataset.json"],
        "main_characters": ["Babe", "Kenzie"],
        "keywords": ["demo bug", "shortcut"],
        "speaker_line_library": {
            "Babe": [{"text": "Moment, ich habe das im Griff.", "language": "de"}],
            "Kenzie": [{"text": "Das war nicht getestet.", "language": "de"}],
        },
        "language_counts": {"de": 8},
        "dominant_language": "de",
        "speakers": {"Babe": {}, "Kenzie": {}},
        "average_segment_duration_seconds": 2.4,
        "source_episode_durations": {"ep_01": 300.0},
        "character_reference_library": {},
        "scene_library": [],
        "behavior_model": {
            "schema_version": 2,
            "speaking_style": {
                "Babe": {"average_words_per_line": 6.0, "energy_level": 0.7, "typical_dialogue_function": {"drives_plan": 2}},
                "Kenzie": {"average_words_per_line": 5.0, "energy_level": 0.55, "typical_dialogue_function": {"contradicts": 2}},
            },
            "relationship_behavior": {
                "Babe||Kenzie": {
                    "characters": ["Babe", "Kenzie"],
                    "typical_dynamic": "fast loyal disagreement",
                    "conversation_starter": "Babe",
                    "conflict_resolver": "Kenzie",
                    "dialogue_tempo": "fast",
                }
            },
            "scene_behavior": {"callback_candidates": ["shortcut"], "typical_beat_sequence": ["cold_open", "setup", "resolution"]},
            "dialogue_patterns": {"setup_reaction_punchline": {"default_pattern": "setup -> reaction -> punchline"}},
            "defaults": {},
        },
    }


class FinishedEpisodeModePipelineTests(unittest.TestCase):
    def test_generation_creates_blueprint_shot_plan_set_bible_and_edl(self) -> None:
        cfg = copy.deepcopy(pipeline_common.DEFAULT_CONFIG)
        cfg["generation"]["target_episode_minutes_fallback"] = 5.0
        cfg["generation"]["target_scene_duration_seconds"] = 40.0

        package, _markdown = STEP08.generate_episode_package(minimal_series_model(), cfg, episode_index=1)

        self.assertIn("episode_blueprint", package)
        self.assertIn("set_bible", package)
        self.assertIn("edit_decision_list", package)
        self.assertIn("audio_mix_plan", package)
        first_scene = package["scenes"][0]
        self.assertTrue(first_scene["shot_plan"])
        self.assertTrue(first_scene["character_continuity_lock"])
        self.assertTrue(first_scene["dialogue_line_metadata"])
        self.assertIn("scene_function", first_scene["writer_room_plan"])

    def test_finished_episode_mode_blocks_placeholder_and_missing_manifests(self) -> None:
        package = {
            "scenes": [
                {
                    "scene_id": "scene_0001",
                    "current_generated_outputs": {"video_source_type": "local_motion_fallback"},
                    "voice_clone": {"required": True, "lines": [{"speaker_name": "Babe", "text": "Test"}], "target_outputs": {}},
                    "lip_sync": {"required": True, "target_outputs": {}},
                    "audio_mix": {"required": True, "stems": {}},
                    "dialogue_sources": [{"type": "generated_template"}],
                }
            ]
        }
        cfg = {"finished_episode_mode": {"enabled": True}}

        checks = STEP18.scene_content_quality_checks(package, cfg)
        gate = STEP18.build_finished_episode_gate({"episode_id": "folge_001"}, checks, cfg)

        self.assertFalse(gate["passed"])
        self.assertGreater(checks["missing_real_motion_video_scene_count"], 0)
        self.assertGreater(checks["missing_backend_manifest_scene_count"], 0)
        self.assertTrue(any("real generated motion" in blocker for blocker in gate["blockers"]))

    def test_backend_manifest_with_fallback_blocks_finished_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "video_manifest.json"
            pipeline_common.write_json(
                manifest,
                {
                    "status": "success",
                    "backend": "finished_episode_video_runner",
                    "fallback_used": True,
                    "placeholder_used": False,
                    "stale_output": False,
                },
            )
            package = {
                "scenes": [
                    {
                        "scene_id": "scene_0001",
                        "scene_function": "setup",
                        "scene_purpose": "setup",
                        "conflict": "test conflict",
                        "behavior_constraints": ["Babe drives"],
                        "dialogue_style_constraints": ["short reactions"],
                        "relationship_context": [{"characters": ["Babe", "Kenzie"]}],
                        "shot_plan": [{"shot_id": "scene_0001_shot_001"}],
                        "location_id": "main_workroom",
                        "set_context": {"set_id": "main_workroom"},
                        "current_generated_outputs": {"video_source_type": "generated_scene_video"},
                        "video_generation": {"required": True},
                        "voice_clone": {"required": False, "lines": []},
                        "lip_sync": {"required": False},
                        "audio_mix": {"required": False},
                        "backend_manifests": {"finished_episode_video_runner": str(manifest)},
                    }
                ]
            }
            checks = STEP18.scene_content_quality_checks(package, {"finished_episode_mode": {"enabled": True}})

        self.assertEqual(checks["fallback_backend_manifest_scene_count"], 1)
        self.assertIn("backend manifest reports fallback or placeholder output", checks["realism_rows"][0]["failed_reasons"])

    def test_realism_report_payload_contains_finished_episode_scores(self) -> None:
        content_checks = {
            "average_realism_score": 0.64,
            "realism_rows": [
                {
                    "scene_id": "scene_0001",
                    "realism_score": 0.64,
                    "component_scores": {
                        "scene_structure_score": 0.7,
                        "dialogue_style_score": 0.6,
                        "video_motion_score": 0.2,
                        "backend_integrity_score": 0.4,
                    },
                    "failed_reasons": ["real motion video missing"],
                }
            ],
            "technical_metrics": [],
        }
        payload = STEP18.build_realism_report_payload(
            {"episode_id": "folge_001", "display_title": "Folge 1"},
            content_checks,
            {"readiness": "blocked", "blockers": ["real motion video missing"], "required_actions": ["rerun video"]},
        )

        self.assertIn("video_motion_score", payload["scores"])
        self.assertEqual(payload["finished_episode_readiness"], "blocked")
        self.assertEqual(payload["weakest_scenes"][0]["scene_id"], "scene_0001")

    def test_regeneration_scopes_are_precise_and_blocked_scopes_do_not_loop(self) -> None:
        self.assertEqual(STEP19.regeneration_scope_from_entry({"regeneration_hints": {"rerun_voice_clone": True}}), "voice_only")
        self.assertEqual(STEP19.regeneration_scope_from_entry({"regeneration_hints": {"rerun_lipsync": True}}), "lipsync_only")
        self.assertEqual(STEP19.regeneration_scope_from_entry({"regeneration_hints": {"rerun_audio_mix": True}}), "audio_mix_only")
        self.assertEqual(STEP19.regeneration_scope_from_entry({"regeneration_hints": {"repair_backend_manifests": True}}), "blocked_missing_backend")

        plan = STEP19.build_rerun_plan(
            "folge_001",
            strict=False,
            update_bible=False,
            scene_ids=["scene_0001"],
            regeneration_queue=[{"scene_id": "scene_0001", "regeneration_scope": "blocked_missing_backend"}],
        )

        self.assertEqual(len(plan), 1)
        self.assertTrue(plan[0]["blocked"])
        self.assertEqual(plan[0]["script"], "18_quality_gate.py")

    def test_export_distinguishes_review_and_finished_episode_packages(self) -> None:
        artifacts = {"episode_id": "folge_001", "release_gate_passed": False, "quality_gate_report": "gate.json"}
        quality_gate = {"finished_episode_gate": {"passed": False, "blockers": ["missing video"]}}
        self.assertEqual(STEP21.classify_export_type("auto", artifacts, quality_gate), "review_export")

        payload = STEP21.build_common_export_payload(
            {},
            artifacts,
            {"scenes": [], "edit_decision_list": []},
            "json",
            Path("export"),
            quality_gate=quality_gate,
            export_type="finished_episode_export",
        )

        with self.assertRaisesRegex(RuntimeError, "finished_episode_export is blocked"):
            STEP21.enforce_export_type("finished_episode_export", payload)


if __name__ == "__main__":
    unittest.main()
