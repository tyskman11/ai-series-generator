from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from support_scripts import pipeline_common
from tools.quality_backends import backend_common

MODULE_PATH = PROJECT_DIR / "support_scripts/configure_quality_backends.py"
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
                "enable_voice_cloning": True,
                "force_voice_cloning": True,
                "require_trained_voice_models": True,
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
            pipeline_common.ensure_quality_first_ready({}, context_label="23_generate_finished_episodes.py")

    def test_quality_first_report_flags_missing_runner_environment_variables(self) -> None:
        runner = {
            "enabled": True,
            "command_template": ["python", "runner.py", "{scene_video}"],
            "required_environment_variables": ["SERIES_IMAGE_BACKEND_COMMAND"],
        }
        cfg = {
            "release_mode": {"enabled": True, "min_episode_quality": 0.9, "max_weak_scenes": 0},
            "cloning": {
                "enable_voice_cloning": True,
                "force_voice_cloning": True,
                "require_trained_voice_models": True,
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
        self.assertTrue(
            resolved.endswith("tools\\quality_backends\\image_runner.py")
            or resolved.endswith("tools/quality_backends/image_runner.py")
        )
        self.assertTrue(Path(resolved).is_absolute())

    def test_delegated_backend_command_parts_use_project_root_for_runner_scripts(self) -> None:
        resolved = backend_common.resolve_delegated_command_parts(
            ["python", "tools/quality_backends/project_local_image_backend.py", "--scene-package", "scene.json"]
        )
        self.assertEqual(Path(resolved[0]).name.lower(), Path(sys.executable).name.lower())
        self.assertTrue(
            resolved[1].endswith("tools\\quality_backends\\project_local_image_backend.py")
            or resolved[1].endswith("tools/quality_backends/project_local_image_backend.py")
        )
        self.assertTrue(Path(resolved[1]).is_absolute())

    def test_find_project_local_ffmpeg_prefers_platform_binary_from_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "ai_series_project"
            runtime_bin = project_root / "runtime" / "host_runtime" / "ffmpeg" / "bin"
            tools_bin = project_root / "tools" / "ffmpeg" / "bin"
            runtime_bin.mkdir(parents=True, exist_ok=True)
            tools_bin.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                runtime_target = runtime_bin / "ffmpeg.exe"
                wrong_target = tools_bin / "ffmpeg"
            else:
                runtime_target = runtime_bin / "ffmpeg"
                wrong_target = tools_bin / "ffmpeg.exe"
            runtime_target.write_bytes(b"runtime")
            wrong_target.write_bytes(b"wrong")
            with patch.object(backend_common, "PROJECT_DIR", project_root), patch("shutil.which", return_value=""):
                resolved = backend_common.find_project_local_ffmpeg()
            self.assertEqual(resolved, str(runtime_target))

    def test_configured_quality_backends_require_real_visual_backend_commands(self) -> None:
        backends = STEP58.configured_backends()
        storyboard_runner = backends["storyboard_scene_runner"]
        image_runner = backends["finished_episode_image_runner"]
        video_runner = backends["finished_episode_video_runner"]
        lipsync_runner = backends["finished_episode_lipsync_runner"]
        voice_runner = backends["finished_episode_voice_runner"]
        master_runner = backends["finished_episode_master_runner"]

        self.assertEqual(storyboard_runner["command_template"][0], "{python}")
        self.assertIn("storyboard_runner.py", storyboard_runner["command_template"][1])
        self.assertIn("local_diffusion_image_backend.py", storyboard_runner["environment"]["SERIES_STORYBOARD_BACKEND_COMMAND"])
        self.assertEqual(storyboard_runner["required_python_modules"], ["diffusers"])
        self.assertEqual(image_runner["command_template"][0], "{python}")
        self.assertEqual(image_runner["required_commands"], [])
        self.assertIn("SERIES_IMAGE_BACKEND_COMMAND", image_runner["environment"])
        self.assertIn("local_diffusion_image_backend.py", image_runner["environment"]["SERIES_IMAGE_BACKEND_COMMAND"])
        self.assertEqual(image_runner["required_python_modules"], ["diffusers"])
        self.assertIn("local_ltx_video_backend.py", video_runner["environment"]["SERIES_VIDEO_BACKEND_COMMAND"])
        self.assertEqual(video_runner["required_python_modules"], ["diffusers"])
        self.assertIn("local_wav2lip_backend.py", lipsync_runner["environment"]["SERIES_LIPSYNC_BACKEND_COMMAND"])
        self.assertIn("local_xtts_voice_backend.py", voice_runner["environment"]["SERIES_VOICE_BACKEND_COMMAND"])
        self.assertEqual(voice_runner["required_python_modules"], ["TTS"])
        self.assertEqual(master_runner["command_template"][0], "{python}")
        self.assertEqual(master_runner["required_commands"], [])

    def test_quality_first_report_rejects_fallback_visual_backends(self) -> None:
        runner = {"enabled": True, "command_template": ["python", "runner.py"]}
        cfg = {
            "release_mode": {
                "enabled": True,
                "min_episode_quality": 0.9,
                "max_weak_scenes": 0,
                "allow_project_local_fallback_backends": False,
            },
            "cloning": {
                "enable_voice_cloning": True,
                "force_voice_cloning": True,
                "require_trained_voice_models": True,
                "allow_system_tts_fallback": False,
                "enable_original_line_reuse": True,
                "enable_lipsync": True,
                "voice_clone_engine": "xtts",
            },
            "external_backends": {
                "storyboard_scene_runner": dict(runner),
                "finished_episode_image_runner": {
                    "enabled": True,
                    "command_template": ["python", "tools/quality_backends/image_runner.py"],
                    "environment": {"SERIES_IMAGE_BACKEND_COMMAND": '"{python}" "tools/quality_backends/project_local_image_backend.py"'},
                },
                "finished_episode_video_runner": dict(runner),
                "finished_episode_voice_runner": dict(runner),
                "finished_episode_lipsync_runner": dict(runner),
                "finished_episode_master_runner": dict(runner),
            },
        }

        report = pipeline_common.quality_first_requirements_report(cfg)

        self.assertFalse(report["ready"])
        self.assertIn(
            "external_backends.finished_episode_image_runner uses fallback backend 'project_local_image_backend.py' instead of a real generation backend",
            report["missing"],
        )

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

        with patch("support_scripts.pipeline_common.shutil.which", return_value=None):
            missing = pipeline_common.external_backend_runner_prerequisite_gaps(cfg, "finished_episode_image_runner")

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
