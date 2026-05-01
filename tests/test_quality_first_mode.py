from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import pipeline_common

MODULE_PATH = Path(__file__).resolve().parents[1] / "58_configure_quality_backends.py"
SPEC = importlib.util.spec_from_file_location("step58_quality_backends", MODULE_PATH)
STEP58 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(STEP58)


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

    def test_render_external_backend_template_includes_runtime_python(self) -> None:
        rendered = pipeline_common.render_external_backend_template("{python}")
        self.assertEqual(rendered, str(pipeline_common.runtime_python()))

    def test_resolve_external_backend_command_part_uses_project_root_for_runner_scripts(self) -> None:
        resolved = pipeline_common.resolve_external_backend_command_part("tools/quality_backends/image_runner.py")
        self.assertTrue(resolved.endswith("tools\\quality_backends\\image_runner.py") or resolved.endswith("tools/quality_backends/image_runner.py"))
        self.assertTrue(Path(resolved).is_absolute())

    def test_configured_quality_backends_use_runtime_python_and_project_local_defaults(self) -> None:
        backends = STEP58.configured_backends()
        image_runner = backends["finished_episode_image_runner"]
        master_runner = backends["finished_episode_master_runner"]

        self.assertEqual(image_runner["command_template"][0], "{python}")
        self.assertEqual(image_runner["required_commands"], [])
        self.assertIn("SERIES_IMAGE_BACKEND_COMMAND", image_runner["environment"])
        self.assertIn("project_local_image_backend.py", image_runner["environment"]["SERIES_IMAGE_BACKEND_COMMAND"])
        self.assertEqual(master_runner["command_template"][0], "{python}")
        self.assertEqual(master_runner["required_commands"], [])

    def test_python_command_prerequisite_uses_active_runtime(self) -> None:
        cfg = {
            "external_backends": {
                "finished_episode_image_runner": {
                    "enabled": True,
                    "command_template": ["{python}", "runner.py"],
                    "required_commands": ["python"],
                }
            }
        }

        with patch("pipeline_common.shutil.which", return_value=None):
            missing = pipeline_common.external_backend_runner_prerequisite_gaps(cfg, "finished_episode_image_runner")

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
