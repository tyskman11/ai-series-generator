from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

from support_scripts import pipeline_common


ROOT = Path(__file__).resolve().parents[1]


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP08 = load_module("08_train_series_model.py", "step08_generation_quality")
STEP15 = load_module("15_render_episode.py", "step15_generation_quality")


class GenerationQualityTests(unittest.TestCase):
    def test_build_scene_generation_plan_includes_style_and_continuity_guidance(self) -> None:
        with mock.patch.object(
            STEP08,
            "load_character_continuity_memory",
            return_value={
                "characters": {
                    "Babe": {
                        "last_episode_id": "episode_003",
                        "continuity": {"outfit": "blue jacket", "accessories": "arcade headset"},
                    }
                }
            },
        ), mock.patch.object(
            STEP08,
            "derive_prompt_constraints_from_bible",
            return_value={
                "positive": ["warm neon palette"],
                "negative": ["washed out lighting"],
                "guidance": {"camera": "eye-level sitcom coverage", "angle": "clean medium angle"},
            },
        ):
            plan = STEP08.build_scene_generation_plan(
                "scene_0001",
                0,
                "Plan",
                "arcade",
                ["Babe", "Kenzie"],
                "Summary",
                {"character_reference_library": {}, "scene_library": []},
                set(),
                "scene_0000",
            )

        self.assertEqual(plan["camera_plan"]["lens"], plan["camera_plan"]["lens_hint"])
        self.assertEqual(plan["camera_plan"]["movement"], plan["camera_plan"]["camera_move"])
        self.assertEqual(plan["control_hints"]["pose_emphasis"], plan["camera_plan"]["pose_hint"])
        self.assertEqual(plan["style_constraints"]["positive"][0], "warm neon palette")
        self.assertEqual(plan["character_continuity"][0]["character"], "Babe")
        self.assertIn("blue jacket", plan["positive_prompt"])
        self.assertIn("washed out lighting", plan["negative_prompt"])
        self.assertTrue(plan["quality_targets"]["style_guidance_available"])

    def test_build_video_generation_prompt_reads_dict_camera_and_control_hints(self) -> None:
        prompt = STEP15.build_video_generation_prompt(
            {
                "summary": "Characters recover the missing controller.",
                "characters": ["Babe", "Kenzie"],
                "location": "Arcade floor",
                "mood": "tense but playful",
            },
            {
                "positive_prompt": "storyboard frame",
                "camera_plan": {"camera": "medium two-shot", "focus": "balanced eye-level framing"},
                "control_hints": {"pose_emphasis": "conversation starter beat", "style_camera": "eye-level sitcom coverage"},
                "continuity": {"previous_scene_id": "scene_0003"},
                "style_constraints": {
                    "positive": ["warm neon palette"],
                    "negative": ["washed out lighting"],
                    "guidance": {"lighting": "motivated practicals"},
                },
                "character_continuity": [{"character": "Babe", "outfit": "blue jacket"}],
                "quality_targets": {
                    "desired_visual_traits": ["clean sitcom contrast"],
                    "preserve_character_identity": True,
                    "preserve_series_style": True,
                },
            },
        )

        self.assertIn("characters: Babe, Kenzie", prompt)
        self.assertIn("location: Arcade floor", prompt)
        self.assertIn("mood: tense but playful", prompt)
        self.assertIn("camera: medium two-shot", prompt)
        self.assertIn("conversation starter beat", prompt)
        self.assertIn("continuity anchor from scene_0003", prompt)
        self.assertIn("warm neon palette", prompt)
        self.assertIn("avoid: washed out lighting", prompt)
        self.assertIn("lighting: motivated practicals", prompt)
        self.assertIn("keep Babe: blue jacket", prompt)
        self.assertIn("quality targets: clean sitcom contrast", prompt)
        self.assertIn("preserve character identity", prompt)

    def test_scene_quality_assessment_rewards_style_and_continuity_guidance(self) -> None:
        baseline = pipeline_common.scene_quality_assessment(
            scene_id="scene_0001",
            current_outputs={"has_scene_master_clip": True},
            reference_slot_count=1,
            continuity_active=False,
        )
        improved = pipeline_common.scene_quality_assessment(
            scene_id="scene_0001",
            current_outputs={"has_scene_master_clip": True, "has_visual_beat_reference_images": True},
            reference_slot_count=2,
            continuity_active=True,
            continuity_character_count=2,
            style_guidance_available=True,
            quality_targets_available=True,
        )

        self.assertGreater(
            improved["component_scores"]["continuity"],
            baseline["component_scores"]["continuity"],
        )
        self.assertIn("series_style_guidance_present", improved["strengths"])

    def test_scene_quality_assessment_penalizes_placeholder_and_pyttsx3_fallbacks(self) -> None:
        fallback = pipeline_common.scene_quality_assessment(
            scene_id="scene_0002",
            current_outputs={
                "asset_source_type": "placeholder",
                "video_source_type": "local_motion_fallback",
                "has_generated_scene_video": True,
                "has_scene_dialogue_audio": True,
                "has_scene_master_clip": True,
                "audio_backend": "pyttsx3",
                "local_composed_scene_video": True,
            },
            voice_required=True,
            lipsync_required=True,
        )
        generated = pipeline_common.scene_quality_assessment(
            scene_id="scene_0002",
            current_outputs={
                "asset_source_type": "generated_episode_frame",
                "video_source_type": "generated_lipsync_video",
                "has_generated_scene_video": True,
                "has_scene_dialogue_audio": True,
                "has_scene_master_clip": True,
                "audio_backend": "reused_original_segments",
            },
            voice_required=True,
            lipsync_required=True,
        )

        self.assertLess(fallback["component_scores"]["visual"], generated["component_scores"]["visual"])
        self.assertLess(fallback["component_scores"]["audio"], generated["component_scores"]["audio"])
        self.assertLess(fallback["quality_score"], generated["quality_score"])


if __name__ == "__main__":
    unittest.main()



