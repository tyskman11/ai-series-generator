from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
QUALITY_BACKEND_DIR = PROJECT_DIR / "tools" / "quality_backends"
if str(QUALITY_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(QUALITY_BACKEND_DIR))

from support_scripts import pipeline_common
from tools.quality_backends import backend_common, local_diffusion_image_backend

MODULE_PATH = PROJECT_DIR / "support_scripts/configure_quality_backends.py"
SPEC = importlib.util.spec_from_file_location("step58_quality_backends", MODULE_PATH)
STEP58 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(STEP58)


class QualityFirstModeTests(unittest.TestCase):
    def test_quality_backend_configuration_repairs_public_identity_adapter_target(self) -> None:
        config = {
            "foundation_training": {"identity_adapter_model": "old/private-adapter"},
            "quality_backend_assets": {
                "targets": [{"name": "image_identity_adapter", "kind": "huggingface", "public_no_login": False}]
            },
        }

        STEP58.ensure_quality_asset_targets(config)

        target = next(
            item
            for item in config["quality_backend_assets"]["targets"]
            if item.get("name") == "image_identity_adapter"
        )
        self.assertEqual(config["foundation_training"]["identity_adapter_model"], "h94/IP-Adapter")
        self.assertEqual(target["repo_id"], "h94/IP-Adapter")
        self.assertTrue(target["public_no_login"])

    def test_quality_backend_configuration_repairs_public_voice_base_target(self) -> None:
        config = {
            "foundation_training": {"voice_base_model": "old/private-voice"},
            "quality_backend_assets": {
                "targets": [{"name": "voice_base_model", "kind": "huggingface", "public_no_login": False}]
            },
        }

        STEP58.ensure_quality_asset_targets(config)

        target = next(
            item
            for item in config["quality_backend_assets"]["targets"]
            if item.get("name") == "voice_base_model"
        )
        self.assertEqual(config["foundation_training"]["voice_base_model"], "openbmb/VoxCPM2")
        self.assertEqual(target["repo_id"], "openbmb/VoxCPM2")
        self.assertTrue(target["public_no_login"])

    def test_active_local_generation_profile_selects_anime_without_machine_paths(self) -> None:
        profile = pipeline_common.active_local_generation_profile(
            {
                "generation": {"model_profile": "anime"},
                "local_generation": {
                    "profiles": {
                        "anime": {
                            "image_model_id": "cagliostrolab/animagine-xl-4.0",
                            "video_model_id": "Wan-AI/Wan2.1-T2V-1.3B",
                            "video_model_family": "wan",
                        }
                    }
                },
            }
        )

        self.assertEqual(profile["profile_id"], "anime")
        self.assertEqual(profile["image_model_id"], "cagliostrolab/animagine-xl-4.0")
        self.assertEqual(profile["video_model_family"], "wan")

    def test_quality_first_rejects_authenticated_or_gated_model_target(self) -> None:
        report = pipeline_common.quality_first_requirements_report(
            {
                "quality_backend_assets": {
                    "targets": [
                        {
                            "name": "restricted",
                            "kind": "huggingface",
                            "repo_id": "black-forest-labs/FLUX.2-dev",
                            "public_no_login": False,
                        }
                    ]
                }
            }
        )

        self.assertIn("quality_backend_assets.targets.restricted must set public_no_login=true", report["missing"])
        self.assertIn("quality_backend_assets.targets.restricted references a gated/login model", report["missing"])

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
            "generation": {"model_profile": "anime", "allow_fallbacks": False},
            "local_generation": {
                "enabled": True,
                "local_models_only": True,
                "allow_runtime_model_downloads": False,
                "require_public_non_gated_models": True,
                "scriptwriter": {"enabled": True, "engine": "transformers", "local_files_only": True},
                "profiles": {
                    "anime": {
                        "image_model_id": "cagliostrolab/animagine-xl-4.0",
                        "image_model_dir": "tools/quality_models/image/cagliostrolab__animagine-xl-4.0",
                        "identity_model_id": "cagliostrolab/animagine-xl-4.0",
                        "identity_model_dir": "tools/quality_models/image/cagliostrolab__animagine-xl-4.0",
                        "video_model_id": "Wan-AI/Wan2.1-T2V-1.3B",
                        "video_model_dir": "tools/quality_models/video/Wan-AI__Wan2.1-T2V-1.3B",
                        "video_model_family": "wan",
                    }
                },
            },
            "cloning": {
                "enable_voice_cloning": True,
                "force_voice_cloning": True,
                "require_trained_voice_models": True,
                "allow_system_tts_fallback": False,
                "enable_original_line_reuse": True,
                "enable_lipsync": True,
                "voice_clone_engine": "voxcpm2",
                "voice_model_id": "openbmb/VoxCPM2",
                "voice_model_dir": "tools/quality_models/voice/openbmb__VoxCPM2",
                "voice_model_local_files_only": True,
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
                "voice_clone_engine": "voxcpm2",
                "voice_model_local_files_only": True,
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

    def test_delegated_backend_runs_directly_from_unc_working_directory(self) -> None:
        command = '"python" "tools/quality_backends/local_diffusion_image_backend.py"'
        unc_cwd = r"\\server\share\series\scene_0001"
        with patch.dict(os.environ, {"SERIES_STORYBOARD_BACKEND_COMMAND": command}, clear=False), patch.object(
            backend_common,
            "write_context_file",
            return_value=Path(tempfile.gettempdir()) / "storyboard_context.json",
        ), patch.object(
            backend_common.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run_mock:
            returncode = backend_common.run_delegated_backend(
                env_var_name="SERIES_STORYBOARD_BACKEND_COMMAND",
                context_payload={"scene_id": "scene_0001"},
                context_prefix="storyboard",
                cwd=unc_cwd,
            )

        self.assertEqual(returncode, 0)
        command_parts = run_mock.call_args.args[0]
        self.assertIsInstance(command_parts, list)
        self.assertEqual(command_parts[0], sys.executable)
        self.assertTrue(command_parts[1].endswith("local_diffusion_image_backend.py"))
        self.assertEqual(run_mock.call_args.kwargs["cwd"], unc_cwd)
        self.assertFalse(run_mock.call_args.kwargs["shell"])

    def test_short_foreign_language_markers_are_rejected_for_german_generation(self) -> None:
        self.assertGreater(pipeline_common.language_text_marker_score("gracias", "es"), 0)
        self.assertGreater(pipeline_common.language_text_marker_score("znowu", "pl"), 0)
        self.assertGreater(pipeline_common.language_text_marker_score("dzięki za oglądanie", "pl"), 0)

    def test_local_sdxl_prompt_prioritizes_visible_characters_and_camera(self) -> None:
        prompt, _ = local_diffusion_image_backend.prompt_from_context(
            {
                "positive_prompt": (
                    "long story prompt, behavior constraints, dialogue style, callbacks, "
                    "relationship details, unrelated prose"
                )
            },
            {
                "characters": ["Kenzie Bell", "Hudson Gimble"],
                "title": "Komplikation: Plan",
                "camera_plan": {
                    "shot_type": "medium two-shot",
                    "composition": "balanced conversational framing",
                    "pose_hint": "animated disagreement",
                },
            },
        )

        self.assertIn("canonical cast identities available", prompt)
        self.assertIn("Kenzie Bell, Hudson Gimble", prompt)
        self.assertIn("exact same facial identity", prompt)
        self.assertIn("medium two-shot", prompt)
        self.assertNotIn("dialogue style", prompt)
        self.assertLess(len(prompt.split()), 77)

    def test_local_image_backend_uses_qwen_default_and_sdxl_identity_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            qwen_dir = root / "Qwen__Qwen-Image"
            sdxl_dir = root / "stabilityai__stable-diffusion-xl-base-1.0"
            for directory in (qwen_dir, sdxl_dir):
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "model_index.json").write_text("{}", encoding="utf-8")
                (directory / "weights.safetensors").write_bytes(b"weights")

            with patch.dict(os.environ, {}, clear=True):
                with patch.object(local_diffusion_image_backend, "DEFAULT_IMAGE_MODEL_DIR", qwen_dir):
                    with patch.object(local_diffusion_image_backend, "SDXL_IDENTITY_MODEL_DIR", sdxl_dir):
                        with patch.object(
                            local_diffusion_image_backend,
                            "FALLBACK_IMAGE_MODEL_DIRS",
                            [qwen_dir, sdxl_dir],
                        ):
                            self.assertEqual(
                                local_diffusion_image_backend.resolve_model_dir(require_identity_adapter=False),
                                qwen_dir.resolve(strict=False),
                            )
                            self.assertEqual(
                                local_diffusion_image_backend.resolve_model_dir(require_identity_adapter=True),
                                sdxl_dir.resolve(strict=False),
                            )
                            self.assertEqual(
                                local_diffusion_image_backend.image_model_family(
                                    "Qwen/Qwen-Image",
                                    qwen_dir,
                                ),
                                "qwen",
                            )
                            self.assertEqual(
                                local_diffusion_image_backend.image_model_family(
                                    "stabilityai/stable-diffusion-xl-base-1.0",
                                    sdxl_dir,
                                ),
                                "sdxl",
                            )

    def test_local_image_backend_filters_montage_and_context_identity_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            preview_dir = root / "characters" / "previews" / "face_001"
            safe_crop = preview_dir / "scene_0001_f10_1_crop.jpg"
            montage = preview_dir / "face_001_montage.jpg"
            context = preview_dir / "scene_0001_f10_1_context.jpg"
            for path in (safe_crop, montage, context):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"image")

            scene_package = {
                "reference_slots": [
                    {
                        "type": "character",
                        "name": "Babe Carano",
                        "portrait_images": [str(montage), str(safe_crop)],
                        "context_images": [str(context)],
                    }
                ]
            }

            references = local_diffusion_image_backend.reference_images_by_character(
                scene_package,
                ["Babe Carano"],
            )

            self.assertEqual(references["Babe Carano"], [safe_crop])
            self.assertTrue(local_diffusion_image_backend.identity_reference_is_safe(safe_crop))
            self.assertFalse(local_diffusion_image_backend.identity_reference_is_safe(montage))
            self.assertFalse(local_diffusion_image_backend.identity_reference_is_safe(context))

    def test_local_image_backend_rejects_resume_manifest_with_montage_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "shot_image_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "inputs": {
                            "identity_reference_images": [
                                str(root / "face_001" / "face_001_montage.jpg"),
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            self.assertFalse(local_diffusion_image_backend.manifest_identity_references_are_safe(manifest))

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
        self.assertEqual(storyboard_runner["environment"]["SERIES_IMAGE_ALLOW_CPU"], "1")
        self.assertFalse(storyboard_runner["requires_gpu"])
        self.assertTrue(storyboard_runner["prefer_gpu"])
        self.assertTrue(storyboard_runner["allow_cpu_execution"])
        self.assertEqual(storyboard_runner["timeout_seconds"], 0)
        self.assertEqual(storyboard_runner["required_python_modules"], ["diffusers"])
        self.assertEqual(image_runner["command_template"][0], "{python}")
        self.assertEqual(image_runner["required_commands"], [])
        self.assertIn("SERIES_IMAGE_BACKEND_COMMAND", image_runner["environment"])
        self.assertIn("local_diffusion_image_backend.py", image_runner["environment"]["SERIES_IMAGE_BACKEND_COMMAND"])
        self.assertEqual(image_runner["environment"]["SERIES_IMAGE_MODEL_ID"], "Qwen/Qwen-Image")
        self.assertEqual(
            image_runner["environment"]["SERIES_IMAGE_MODEL_DIR"],
            "tools/quality_models/image/Qwen__Qwen-Image",
        )
        self.assertEqual(
            image_runner["environment"]["SERIES_IMAGE_IDENTITY_MODEL_ID"],
            "stabilityai/stable-diffusion-xl-base-1.0",
        )
        self.assertEqual(
            image_runner["environment"]["SERIES_IMAGE_IDENTITY_MODEL_DIR"],
            "tools/quality_models/image/stabilityai__stable-diffusion-xl-base-1.0",
        )
        self.assertEqual(image_runner["environment"]["SERIES_IMAGE_ALLOW_CPU"], "1")
        self.assertEqual(image_runner["environment"]["SERIES_IMAGE_WIDTH"], "1216")
        self.assertEqual(image_runner["environment"]["SERIES_IMAGE_HEIGHT"], "704")
        self.assertEqual(image_runner["environment"]["SERIES_IMAGE_INFERENCE_STEPS"], "50")
        self.assertEqual(image_runner["environment"]["SERIES_IMAGE_GUIDANCE_SCALE"], "4.0")
        self.assertEqual(image_runner["environment"]["SERIES_IMAGE_QUALITY_PRESET"], "qwen_image_source_series")
        self.assertEqual(image_runner["environment"]["SERIES_IMAGE_RESUME_SHOTS"], "1")
        self.assertEqual(image_runner["timeout_seconds"], 0)
        self.assertTrue(image_runner["allow_cpu_execution"])
        self.assertEqual(image_runner["required_python_modules"], ["diffusers"])
        self.assertIn("local_wan_video_backend.py", video_runner["environment"]["SERIES_VIDEO_BACKEND_COMMAND"])
        self.assertEqual(video_runner["environment"]["SERIES_VIDEO_RESUME_SHOTS"], "1")
        self.assertEqual(video_runner["environment"]["SERIES_VIDEO_LATEST_MODEL_ID"], "Wan-AI/Wan2.1-T2V-1.3B")
        self.assertEqual(
            video_runner["environment"]["SERIES_VIDEO_LATEST_MODEL_DIR"],
            "tools/quality_models/video/Wan-AI__Wan2.1-T2V-1.3B",
        )
        self.assertEqual(video_runner["environment"]["SERIES_VIDEO_MODEL_ID"], "Wan-AI/Wan2.1-T2V-1.3B")
        self.assertEqual(
            video_runner["environment"]["SERIES_VIDEO_MODEL_DIR"],
            "tools/quality_models/video/Wan-AI__Wan2.1-T2V-1.3B",
        )
        self.assertEqual(video_runner["environment"]["SERIES_VIDEO_MODEL_FAMILY"], "wan")
        self.assertEqual(video_runner["environment"]["SERIES_VIDEO_COMPATIBILITY_MODE"], "local_wan_diffusers")
        self.assertEqual(video_runner["environment"]["SERIES_VIDEO_WIDTH"], "1216")
        self.assertEqual(video_runner["environment"]["SERIES_VIDEO_HEIGHT"], "704")
        self.assertEqual(video_runner["environment"]["SERIES_VIDEO_QUALITY_PRESET"], "source_series_high")
        self.assertEqual(video_runner["timeout_seconds"], 0)
        self.assertEqual(video_runner["required_python_modules"], ["diffusers"])
        self.assertIn("local_wav2lip_backend.py", lipsync_runner["environment"]["SERIES_LIPSYNC_BACKEND_COMMAND"])
        self.assertEqual(lipsync_runner["timeout_seconds"], 0)
        self.assertIn("local_voxcpm_voice_backend.py", voice_runner["environment"]["SERIES_VOICE_BACKEND_COMMAND"])
        self.assertEqual(voice_runner["environment"]["SERIES_VOICE_BACKEND_TIMEOUT_SECONDS"], "0")
        self.assertEqual(voice_runner["environment"]["SERIES_VOICE_MODEL_ID"], "openbmb/VoxCPM2")
        self.assertEqual(voice_runner["environment"]["SERIES_VOICE_MIN_REFERENCE_SECONDS"], "6.0")
        self.assertEqual(voice_runner["timeout_seconds"], 0)
        self.assertEqual(voice_runner["required_python_modules"], ["voxcpm", "soundfile"])
        self.assertEqual(master_runner["command_template"][0], "{python}")
        self.assertEqual(master_runner["timeout_seconds"], 0)
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
                "voice_clone_engine": "voxcpm2",
                "voice_model_id": "openbmb/VoxCPM2",
                "voice_model_dir": "tools/quality_models/voice/openbmb__VoxCPM2",
                "voice_model_local_files_only": True,
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
