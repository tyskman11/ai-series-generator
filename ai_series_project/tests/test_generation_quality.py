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
from pathlib import Path
from unittest import mock

from support_scripts import pipeline_common


ROOT = PROJECT_DIR


def load_module(filename: str, module_name: str):
    target = ROOT / filename if filename.startswith("support_scripts/") else SCRIPT_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP08 = load_module("08_train_series_model.py", "step08_generation_quality")
STEP15 = load_module("17_render_episode.py", "step15_generation_quality")


class GenerationQualityTests(unittest.TestCase):
    def test_backend_progress_tracks_generated_shot_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_frame = root / "shot_001.png"
            second_frame = root / "shot_002.png"
            first_frame.write_bytes(b"generated-frame")
            scene_package = {
                "shot_packages": [
                    {
                        "shot_id": "scene_0001_shot_001",
                        "target_outputs": {"primary_frame": str(first_frame)},
                    },
                    {
                        "shot_id": "scene_0001_shot_002",
                        "target_outputs": {"primary_frame": str(second_frame)},
                    },
                ]
            }

            targets = STEP15.backend_progress_targets(
                scene_package,
                "finished_episode_image_runner",
            )
            snapshot = STEP15.backend_progress_snapshot(targets)

        self.assertEqual(snapshot["completed"], 1)
        self.assertEqual(snapshot["total"], 2)
        self.assertEqual(snapshot["current_label"], "scene_0001_shot_002")

    def test_backend_live_progress_includes_stats_and_eta_before_first_output(self) -> None:
        reporter = mock.Mock()
        monitor = STEP15.BackendLiveProgressMonitor(
            reporter=reporter,
            runner_name="finished_episode_image_runner",
            scene_id="scene_0001",
            targets=[
                {"label": "scene_0001_shot_001", "path": Path("missing-shot-001.png")},
                {"label": "scene_0001_shot_002", "path": Path("missing-shot-002.png")},
            ],
            overall_base=31.0,
            overall_span=0.85,
            completed_units_before=0,
            total_units=2,
        )

        monitor._publish()

        update = reporter.update.call_args
        self.assertIn("Image generation: 0/2 ready, 2 remaining", update.kwargs["extra_label"])
        self.assertIn("elapsed", update.kwargs["extra_label"])
        self.assertEqual(update.kwargs["scope_eta_seconds"], 600.0)
        self.assertEqual(update.kwargs["overall_eta_seconds"], 600.0)

    def test_live_progress_reporter_accepts_explicit_eta_overrides(self) -> None:
        reporter = pipeline_common.LiveProgressReporter(
            script_name="17_render_episode.py",
            total=32,
            phase_label="Render Episode",
            parent_label="folge_02",
        )

        rendered = "\n".join(
            reporter._render_lines(
                31,
                current_label="scene_0001 / shot_001",
                scope_current=0,
                scope_total=8,
                scope_label="Backend",
                scope_eta_seconds=600,
                overall_eta_seconds=1200,
            )
        )

        self.assertIn("Current ETA", rendered)
        self.assertIn("Overall ETA", rendered)
        self.assertNotIn("calculating", rendered)

    def test_safe_duration_seconds_expands_dialogue_heavy_scene(self) -> None:
        duration = STEP15.safe_duration_seconds(
            {
                "estimated_runtime_seconds": 8.0,
                "dialogue_lines": [
                    "Babe: And of course we only notice that right in the middle of the stress.",
                    "Triple G: Then someone must have skipped a crucial step somewhere.",
                    "Babe: Now double is becoming bigger even though it looked simple a moment ago.",
                ],
            }
        )

        self.assertGreater(duration, 8.0)

    def test_safe_duration_seconds_preserves_long_planned_scene_runtime(self) -> None:
        duration = STEP15.safe_duration_seconds(
            {
                "estimated_runtime_seconds": 54.0,
                "dialogue_lines": [
                    "Babe: We need to keep this beat long enough for an actual story turn.",
                ],
            }
        )

        self.assertEqual(duration, 54.0)

    def test_render_subtitle_preview_srt_keeps_lines_visible_longer(self) -> None:
        srt = STEP15.render_subtitle_preview_srt(
            [
                {
                    "speaker_name": "Babe",
                    "text": "And of course we only notice that right in the middle of the stress.",
                    "start_seconds": 1.0,
                    "end_seconds": 1.4,
                }
            ]
        )

        self.assertIn("00:00:01,000 -->", srt)
        self.assertNotIn("00:00:01,000 --> 00:00:01,400", srt)

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

    def test_build_scene_generation_plan_applies_show_profile_and_toolkit_guidance(self) -> None:
        plan = STEP08.build_scene_generation_plan(
            "scene_0002",
            1,
            "Escalation",
            "demo",
            ["Babe", "Kenzie"],
            "A demo escalates.",
            {"character_reference_library": {}, "scene_library": []},
            set(),
            "scene_0001",
            continuity_memory={},
            style_constraints={},
            show_profile={
                "profile_id": "sitcom_fast",
                "style_rules": ["bright source-series palette"],
                "camera_rules": ["favor reaction closeups"],
                "continuity_rules": ["keep main-office props stable"],
            },
            toolkit_guidance={"prompt_fragments": ["preserve callback pacing"]},
        )

        self.assertIn("bright source-series palette", plan["positive_prompt"])
        self.assertIn("keep main-office props stable", plan["positive_prompt"])
        self.assertIn("preserve callback pacing", plan["positive_prompt"])
        self.assertIn("favor reaction closeups", plan["control_hints"]["show_profile_camera"])
        self.assertEqual(plan["quality_targets"]["show_profile_id"], "sitcom_fast")
        self.assertTrue(plan["quality_targets"]["toolkit_guidance_available"])

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
        self.assertNotIn("generic cartoon", prompt)

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

    def test_scene_quality_assessment_reaches_100_for_complete_generated_scene(self) -> None:
        quality = pipeline_common.scene_quality_assessment(
            scene_id="scene_0003",
            current_outputs={
                "asset_source_type": "generated_episode_frame",
                "video_source_type": "generated_lipsync_video",
                "has_generated_scene_video": True,
                "has_generated_primary_frame": True,
                "has_scene_dialogue_audio": True,
                "has_scene_master_clip": True,
                "has_visual_beat_reference_images": True,
                "audio_backend": "xtts_voice_clone",
            },
            voice_required=True,
            lipsync_required=True,
            reference_slot_count=5,
            continuity_active=True,
            continuity_character_count=4,
            style_guidance_available=True,
            quality_targets_available=True,
        )

        self.assertEqual(quality["quality_percent"], 100)
        self.assertEqual(quality["component_scores"]["audio"], 1.0)
        self.assertEqual(quality["component_scores"]["lip_sync"], 1.0)
        self.assertEqual(quality["quality_label"], "series_quality_candidate")

    def test_completion_summary_requires_scene_dialogue_audio_for_fully_generated_episode(self) -> None:
        summary = pipeline_common.generated_episode_completion_summary(
            scene_count=2,
            generated_scene_video_count=2,
            scene_dialogue_audio_count=1,
            scene_master_clip_count=2,
            render_mode="full_generated_episode",
            final_render="final.mp4",
            full_generated_episode="master.mp4",
        )

        self.assertNotEqual(summary["production_readiness"], "fully_generated_episode_ready")
        self.assertIn("materialize missing scene dialogue audio", summary["remaining_backend_tasks"])


if __name__ == "__main__":
    unittest.main()



