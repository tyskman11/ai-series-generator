from __future__ import annotations

import unittest
from unittest.mock import patch

import pipeline_common


class QualityFirstModeTests(unittest.TestCase):
    def test_quality_first_requirements_report_flags_missing_runners_and_fallbacks(self) -> None:
        cfg = {
            "release_mode": {"enabled": False, "min_episode_quality": 0.68, "max_weak_scenes": 2},
            "cloning": {
                "allow_system_tts_fallback": True,
                "enable_original_line_reuse": False,
                "enable_lipsync": False,
                "voice_clone_engine": "pyttsx3",
            },
            "external_backends": {},
        }

        report = pipeline_common.quality_first_requirements_report(cfg)

        self.assertFalse(report["ready"])
        self.assertIn("release_mode.enabled must be true", report["missing"])
        self.assertIn(
            "external_backends.finished_episode_video_runner must be enabled with a non-empty command_template",
            report["missing"],
        )
        self.assertTrue(report["warnings"])

    def test_quality_first_requirements_report_accepts_fully_configured_quality_stack(self) -> None:
        runner = {"enabled": True, "command_template": ["python", "runner.py", "{scene_video}"]}
        cfg = {
            "release_mode": {"enabled": True, "min_episode_quality": 0.9, "max_weak_scenes": 0},
            "cloning": {
                "allow_system_tts_fallback": False,
                "enable_original_line_reuse": True,
                "enable_lipsync": True,
                "voice_clone_engine": "xtts",
            },
            "external_backends": {
                "storyboard_scene_runner": dict(runner),
                "finished_episode_image_runner": dict(runner),
                "finished_episode_video_runner": dict(runner),
                "finished_episode_voice_runner": dict(runner),
                "finished_episode_lipsync_runner": dict(runner),
                "finished_episode_master_runner": dict(runner),
            },
        }

        report = pipeline_common.quality_first_requirements_report(cfg)

        self.assertTrue(report["ready"])
        self.assertEqual(report["missing"], [])

    def test_ensure_quality_first_ready_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "original-episode quality"):
            pipeline_common.ensure_quality_first_ready({}, context_label="57_generate_finished_episodes.py")

    def test_quality_first_report_flags_missing_runner_environment_variables(self) -> None:
        runner = {
            "enabled": True,
            "command_template": ["python", "runner.py", "{scene_video}"],
            "required_environment_variables": ["SERIES_IMAGE_BACKEND_COMMAND"],
        }
        cfg = {
            "release_mode": {"enabled": True, "min_episode_quality": 0.9, "max_weak_scenes": 0},
            "cloning": {
                "allow_system_tts_fallback": False,
                "enable_original_line_reuse": True,
                "enable_lipsync": True,
                "voice_clone_engine": "xtts",
            },
            "external_backends": {
                "storyboard_scene_runner": {"enabled": True, "command_template": ["python", "runner.py"]},
                "finished_episode_image_runner": dict(runner),
                "finished_episode_video_runner": {"enabled": True, "command_template": ["python", "runner.py"]},
                "finished_episode_voice_runner": {"enabled": True, "command_template": ["python", "runner.py"]},
                "finished_episode_lipsync_runner": {"enabled": True, "command_template": ["python", "runner.py"]},
                "finished_episode_master_runner": {"enabled": True, "command_template": ["python", "runner.py"]},
            },
        }

        with patch.dict("os.environ", {}, clear=False):
            report = pipeline_common.quality_first_requirements_report(cfg)

        self.assertFalse(report["ready"])
        self.assertIn(
            "external_backends.finished_episode_image_runner requires environment variable 'SERIES_IMAGE_BACKEND_COMMAND'",
            report["missing"],
        )


if __name__ == "__main__":
    unittest.main()
