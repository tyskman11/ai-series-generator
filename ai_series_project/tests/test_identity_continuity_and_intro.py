from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
BACKEND_DIR = PROJECT_DIR / "tools" / "quality_backends"
for candidate in (PROJECT_DIR, BACKEND_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from support_scripts import pipeline_common, prepare_quality_backends


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP08 = load_module(SCRIPT_ROOT / "08_train_series_model.py", "step08_identity_continuity_test")
STEP17 = load_module(SCRIPT_ROOT / "17_render_episode.py", "step17_identity_continuity_test")
STEP18 = load_module(SCRIPT_ROOT / "18_quality_gate.py", "step18_identity_continuity_test")
IMAGE_BACKEND = load_module(
    BACKEND_DIR / "local_diffusion_image_backend.py",
    "local_diffusion_image_backend_test",
)
VIDEO_BACKEND = load_module(
    BACKEND_DIR / "local_ltx_video_backend.py",
    "local_ltx_video_backend_test",
)


class IdentityContinuityAndIntroTests(unittest.TestCase):
    def test_image_backend_preserves_identity_adapter_attention_processors(self) -> None:
        class FakeVae:
            def __init__(self) -> None:
                self.slicing_enabled = False
                self.tiling_enabled = False

            def enable_slicing(self) -> None:
                self.slicing_enabled = True

            def enable_tiling(self) -> None:
                self.tiling_enabled = True

        class FakePipeline:
            def __init__(self) -> None:
                self.vae = FakeVae()
                self.attention_slicing_enabled = False

            def enable_attention_slicing(self) -> None:
                self.attention_slicing_enabled = True

        pipeline = FakePipeline()

        IMAGE_BACKEND.enable_pipeline_memory_optimizations(
            pipeline,
            identity_adapter_loaded=True,
        )

        self.assertFalse(pipeline.attention_slicing_enabled)
        self.assertTrue(pipeline.vae.slicing_enabled)
        self.assertTrue(pipeline.vae.tiling_enabled)

    def test_image_backend_uses_attention_slicing_without_identity_adapter(self) -> None:
        pipeline = mock.Mock()
        pipeline.vae = None

        IMAGE_BACKEND.enable_pipeline_memory_optimizations(
            pipeline,
            identity_adapter_loaded=False,
        )

        pipeline.enable_attention_slicing.assert_called_once_with()
        pipeline.enable_vae_slicing.assert_called_once_with()
        pipeline.enable_vae_tiling.assert_called_once_with()

    def test_strict_render_preflight_reports_missing_identity_references(self) -> None:
        cfg = {
            "release_mode": {"force_finished_episode_generation": True},
            "cloning": {
                "force_voice_cloning": True,
                "voice_clone_engine": "voxcpm2",
            },
        }
        shotlist = {
            "scenes": [
                {
                    "characters": ["Kenzie Bell"],
                    "character_continuity_lock": {
                        "Kenzie Bell": {"reference_images": []},
                    },
                    "generation_plan": {
                        "reference_slots": [
                            {
                                "type": "character",
                                "name": "Kenzie Bell",
                                "portrait_images": [],
                                "context_images": [],
                            }
                        ]
                    },
                }
            ]
        }

        gaps = STEP17.strict_render_preflight_gaps(cfg, shotlist)

        self.assertTrue(any("Kenzie Bell" in gap for gap in gaps))

    def test_strict_render_preflight_accepts_existing_identity_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reference = Path(tmpdir) / "kenzie_crop.jpg"
            reference.write_bytes(b"reference")
            cfg = {
                "release_mode": {"force_finished_episode_generation": True},
                "cloning": {
                    "force_voice_cloning": True,
                    "voice_clone_engine": "voxcpm2",
                },
            }
            shotlist = {
                "scenes": [
                    {
                        "characters": ["Kenzie Bell"],
                        "character_continuity_lock": {
                            "Kenzie Bell": {"reference_images": [str(reference)]},
                        },
                    }
                ]
            }

            gaps = STEP17.strict_render_preflight_gaps(cfg, shotlist)

        self.assertEqual(gaps, [])

    def test_primary_identity_preview_is_preferred(self) -> None:
        directories = STEP08.trusted_identity_preview_dirs(
            "face_secondary",
            {
                "preview_dir": "characters/previews/face_secondary",
                "identity_primary_cluster": "face_primary",
                "identity_cluster_ids": ["face_secondary", "face_other"],
            },
        )

        self.assertEqual(directories[0], "characters/previews/face_primary")

    def test_reference_board_input_covers_each_visible_character_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            babe_one = root / "babe_01.jpg"
            babe_two = root / "babe_02.jpg"
            kenzie_one = root / "kenzie_01.jpg"
            kenzie_two = root / "kenzie_02.jpg"
            for path in (babe_one, babe_two, kenzie_one, kenzie_two):
                path.write_bytes(b"reference")
            scene = {
                "character_continuity_lock": {
                    "Babe": {"reference_images": [str(babe_one), str(babe_two)]},
                    "Kenzie": {"reference_images": [str(kenzie_one), str(kenzie_two)]},
                }
            }

            references = IMAGE_BACKEND.reference_image_paths(scene, ["Babe", "Kenzie"])

        self.assertEqual([path.name for path in references], [
            "babe_01.jpg",
            "kenzie_01.jpg",
            "babe_02.jpg",
            "kenzie_02.jpg",
        ])

    def test_identity_gate_requires_every_visible_character(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            babe_reference = root / "babe.jpg"
            kenzie_reference = root / "kenzie.jpg"
            babe_reference.write_bytes(b"babe")
            kenzie_reference.write_bytes(b"kenzie")
            manifest = root / "shot_manifest.json"
            pipeline_common.write_json(
                manifest,
                {
                    "identity_conditioning": {
                        "adapter_loaded": True,
                        "reference_count": 2,
                        "characters_conditioned": ["Babe"],
                    }
                },
            )
            scene = {
                "characters": ["Babe", "Kenzie"],
                "character_continuity_lock": {
                    "Babe": {"reference_images": [str(babe_reference)]},
                    "Kenzie": {"reference_images": [str(kenzie_reference)]},
                },
                "shot_packages": [
                    {
                        "characters_visible": ["Babe", "Kenzie"],
                        "target_outputs": {"manifest": str(manifest)},
                    }
                ],
            }

            status = STEP18.scene_identity_status(scene)

        self.assertTrue(status["reference_ready"])
        self.assertFalse(status["identity_conditioned"])

    def test_identity_gate_reads_image_manifest_and_rejects_unverified_multi_character_board(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            babe_reference = root / "babe_crop.jpg"
            kenzie_reference = root / "kenzie_crop.jpg"
            babe_reference.write_bytes(b"babe")
            kenzie_reference.write_bytes(b"kenzie")
            manifest = root / "shot_image_manifest.json"
            pipeline_common.write_json(
                manifest,
                {
                    "identity_conditioning": {
                        "adapter_loaded": True,
                        "reference_count": 2,
                        "characters_conditioned": ["Babe", "Kenzie"],
                        "reference_safety": True,
                    },
                    "identity_contract": {
                        "expected_visible_characters": ["Babe", "Kenzie"],
                        "expected_visible_character_count": 2,
                        "maximum_allowed_visible_people": 2,
                        "verification_status": "unverified_multi_character_identity_board",
                        "identity_risk": "high",
                        "regional_identity_control": False,
                    },
                },
            )
            scene = {
                "characters": ["Babe", "Kenzie"],
                "character_continuity_lock": {
                    "Babe": {"reference_images": [str(babe_reference)]},
                    "Kenzie": {"reference_images": [str(kenzie_reference)]},
                },
                "shot_packages": [
                    {
                        "shot_id": "scene_0001_shot_001",
                        "characters_visible": ["Babe", "Kenzie"],
                        "target_outputs": {"image_manifest": str(manifest)},
                    }
                ],
            }

            status = STEP18.scene_identity_status(scene)

        self.assertTrue(status["reference_ready"])
        self.assertEqual(status["unverified_multi_character_shot_count"], 1)
        self.assertFalse(status["identity_conditioned"])

    def test_local_identity_backend_blocks_unverified_multi_character_generation(self) -> None:
        contract = IMAGE_BACKEND.shot_identity_contract(
            ["Babe", "Kenzie"],
            [],
            ["Babe", "Kenzie"],
            [],
        )

        self.assertFalse(contract["generation_allowed"])
        self.assertEqual(contract["identity_risk"], "high")

    def test_quality_gate_requires_multiple_portrait_references_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reference = Path(tmpdir) / "babe_crop.jpg"
            reference.write_bytes(b"babe")
            scene = {
                "characters": ["Babe"],
                "character_continuity_lock": {"Babe": {"reference_images": [str(reference)]}},
            }

            status = STEP18.scene_identity_status(scene, min_reference_images_per_character=3)

        self.assertFalse(status["reference_ready"])
        self.assertEqual(status["reference_counts"]["Babe"], 1)
        self.assertEqual(status["insufficient_reference_characters"], ["Babe"])

    def test_image_backend_shot_prompt_forbids_extra_people_for_visible_cast(self) -> None:
        prompt = IMAGE_BACKEND.shot_prompt(
            "live-action sitcom frame",
            {
                "shot_type": "medium close-up",
                "characters_visible": ["Babe"],
                "purpose": "cover a single speaker beat",
            },
        )

        self.assertIn("exactly 1 person total", prompt)
        self.assertIn("only Babe", prompt)
        self.assertIn("no extras", prompt)

    def test_generated_shot_plan_uses_face_safe_establishing_and_single_speaker_shots(self) -> None:
        shot_plan = STEP08.build_scene_shot_plan(
            scene_id="scene_0001",
            scene_function="setup",
            scene_characters=["Babe", "Kenzie"],
            dialogue=["Babe: Test", "Kenzie: Nein"],
            dialogue_metadata=[{"speaker": "Babe"}, {"speaker": "Kenzie"}],
            duration_seconds=14.0,
            location_id="office",
        )

        self.assertEqual(shot_plan[0]["characters_visible"], [])
        dialogue_shots = [row for row in shot_plan if row["dialogue_line_indices"]]
        self.assertEqual(dialogue_shots[0]["characters_visible"], ["Babe"])
        self.assertEqual(dialogue_shots[1]["characters_visible"], ["Kenzie"])

    def test_identity_gate_rejects_montage_reference_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            babe_reference = root / "scene_0001_f10_1_crop.jpg"
            babe_reference.write_bytes(b"babe")
            manifest = root / "shot_manifest.json"
            pipeline_common.write_json(
                manifest,
                {
                    "identity_conditioning": {
                        "adapter_loaded": True,
                        "reference_count": 1,
                        "characters_conditioned": ["Babe"],
                        "reference_safety": False,
                        "unsafe_reference_images": [str(root / "face_001_montage.jpg")],
                    }
                },
            )
            scene = {
                "scene_id": "scene_0001",
                "characters": ["Babe"],
                "character_continuity_lock": {
                    "Babe": {"reference_images": [str(babe_reference)]},
                },
                "shot_packages": [
                    {
                        "characters_visible": ["Babe"],
                        "target_outputs": {"manifest": str(manifest)},
                    }
                ],
            }

            status = STEP18.scene_identity_status(scene)
            checks = STEP18.scene_content_quality_checks({"scenes": [scene]}, {"finished_episode_mode": {"enabled": True}})

        self.assertTrue(status["reference_ready"])
        self.assertFalse(status["identity_references_safe"])
        self.assertFalse(status["identity_conditioned"])
        self.assertEqual(checks["unsafe_identity_reference_scene_count"], 1)
        self.assertIn("identity references include montage/contact-sheet images", checks["realism_rows"][0]["failed_reasons"])

    def test_scene_without_visible_people_does_not_create_identity_blocker(self) -> None:
        status = STEP18.scene_identity_status({"characters": [], "shot_packages": []})

        self.assertTrue(status["reference_ready"])
        self.assertTrue(status["identity_conditioned"])

    def test_old_asset_list_is_extended_with_identity_adapter(self) -> None:
        cfg = copy.deepcopy(pipeline_common.DEFAULT_CONFIG)
        cfg["quality_backend_assets"] = {
            "targets": [
                {
                    "name": "legacy_image_model",
                    "kind": "huggingface",
                    "repo_id": "example/model",
                    "target_dir": "tools/quality_models/image/example__model",
                }
            ]
        }

        targets = prepare_quality_backends.quality_backend_asset_targets(cfg)

        self.assertTrue(any(row.get("name") == "legacy_image_model" for row in targets))
        self.assertTrue(any(row.get("name") == "image_identity_adapter" for row in targets))

    def test_locked_season_intro_hash_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            intro = Path(tmpdir) / "intro.mp4"
            intro.write_bytes(b"fixed-season-intro")
            expected_hash = hashlib.sha256(intro.read_bytes()).hexdigest()
            package = {
                "season_intro": {
                    "season_id": "season_01",
                    "canonical_video": str(intro),
                    "canonical_sha256": expected_hash,
                    "status": "locked_existing",
                }
            }
            cfg = {
                "season_intro": {
                    "enabled": True,
                    "default_season_id": "season_01",
                    "require_in_finished_episode_mode": True,
                }
            }

            status = STEP18.season_intro_status(package, cfg)

        self.assertTrue(status["required"])
        self.assertTrue(status["ready"])
        self.assertEqual(status["actual_hash"], expected_hash)

    def test_finished_episode_defaults_keep_season_intro_separate(self) -> None:
        self.assertTrue(pipeline_common.DEFAULT_CONFIG["season_intro"]["enabled"])
        self.assertFalse(
            pipeline_common.DEFAULT_CONFIG["season_intro"]["require_in_finished_episode_mode"]
        )
        self.assertFalse(pipeline_common.DEFAULT_CONFIG["season_intro"]["auto_generate_if_missing"])

    def test_generated_intro_package_uses_real_multi_shot_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            package = STEP17.build_generated_season_intro_package(
                {"season_intro": {"generated_duration_seconds": 12.0}},
                {
                    "active_character_groups": [{"label": "Test Series"}],
                    "focus_characters": ["Babe"],
                    "scenes": [],
                },
                "season_01",
                {},
                Path(tmpdir),
            )

        scene_package = package["scene_package"]
        self.assertEqual(scene_package["video_generation"]["mode"], "generated_season_intro_video")
        self.assertEqual(len(scene_package["shot_packages"]), 3)
        self.assertEqual(scene_package["shot_packages"][0]["characters_visible"], [])
        self.assertIn("image_manifest", scene_package["shot_packages"][0]["target_outputs"])
        self.assertIn("video_manifest", scene_package["shot_packages"][0]["target_outputs"])
        prompt = IMAGE_BACKEND.shot_prompt("opening sequence", scene_package["shot_packages"][0])
        self.assertIn("no people visible", prompt)

    def test_people_free_intro_prompt_does_not_request_faces(self) -> None:
        prompt = IMAGE_BACKEND.compact_visual_prompt(
            {},
            {
                "characters": [],
                "image_generation": {"mode": "generated_season_intro_keyframes"},
            },
            "polished opening sequence",
        )

        self.assertIn("no people visible", prompt)
        self.assertNotIn("visible faces", prompt)

    def test_generated_intro_gate_requires_runner_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            intro = root / "intro.mp4"
            image_manifest = root / "image.json"
            video_manifest = root / "video.json"
            intro.write_bytes(b"generated-intro")
            image_manifest.write_text("{}", encoding="utf-8")
            video_manifest.write_text("{}", encoding="utf-8")
            expected_hash = hashlib.sha256(intro.read_bytes()).hexdigest()
            package = {
                "season_intro": {
                    "season_id": "season_01",
                    "source_origin": "backend_generated",
                    "canonical_video": str(intro),
                    "canonical_sha256": expected_hash,
                    "backend_runner_statuses": {
                        "finished_episode_image_runner": "completed",
                        "finished_episode_video_runner": "completed",
                    },
                    "backend_manifests": {
                        "finished_episode_image_runner": str(image_manifest),
                        "finished_episode_video_runner": str(video_manifest),
                    },
                    "fallback_used": False,
                }
            }
            cfg = {
                "season_intro": {
                    "enabled": True,
                    "default_season_id": "season_01",
                    "require_in_finished_episode_mode": True,
                }
            }

            ready = STEP18.season_intro_status(package, cfg)
            package["season_intro"]["fallback_used"] = True
            blocked = STEP18.season_intro_status(package, cfg)

        self.assertTrue(ready["ready"])
        self.assertTrue(ready["backend_integrity_ready"])
        self.assertFalse(blocked["ready"])
        self.assertIn("fallback backend used", blocked["missing_backend_evidence"])

    def test_generated_intro_runners_publish_live_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package = STEP17.build_generated_season_intro_package(
                {"season_intro": {"generated_duration_seconds": 12.0}},
                {"active_character_groups": [{"label": "Test Series"}], "scenes": []},
                "season_01",
                {},
                root,
            )

            runner_configs = {}

            def fake_runner(runner_cfg, runner_name, *, context, **_kwargs):
                runner_configs[runner_name] = runner_cfg
                output = (
                    Path(context["primary_frame"])
                    if runner_name == "finished_episode_image_runner"
                    else Path(context["scene_video"])
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"generated")
                return {
                    "runner_name": runner_name,
                    "status": "completed",
                    "command": ["real_backend"],
                    "command_text": "real_backend",
                    "produced_outputs": [str(output)],
                }

            reporter = mock.Mock()
            with mock.patch.object(
                STEP17,
                "build_generated_season_intro_package",
                return_value=package,
            ), mock.patch.object(
                STEP17,
                "run_external_backend_runner",
                side_effect=fake_runner,
            ), mock.patch.object(
                STEP17,
                "write_backend_task_manifest",
                side_effect=lambda **kwargs: str(
                    root / f"{kwargs['runner_name']}.json"
                ),
            ):
                STEP17.generate_season_intro_video(
                    {},
                    {},
                    "season_01",
                    {},
                    root,
                    reporter=reporter,
                )

        self.assertGreaterEqual(reporter.update.call_count, 4)
        labels = [
            str(call.kwargs.get("current_label", ""))
            for call in reporter.update.call_args_list
        ]
        self.assertTrue(any("season_01 intro" in label for label in labels))
        self.assertTrue(
            all(call.kwargs.get("scope_eta_seconds") is not None for call in reporter.update.call_args_list)
        )
        image_runner = runner_configs["finished_episode_image_runner"]["external_backends"]["finished_episode_image_runner"]
        video_runner = runner_configs["finished_episode_video_runner"]["external_backends"]["finished_episode_video_runner"]
        self.assertEqual(image_runner["timeout_seconds"], 0)
        self.assertEqual(video_runner["timeout_seconds"], 0)
        self.assertEqual(image_runner["environment"]["SERIES_IMAGE_MODEL_ID"], "Qwen/Qwen-Image")
        self.assertEqual(
            image_runner["environment"]["SERIES_IMAGE_IDENTITY_MODEL_DIR"],
            "tools/quality_models/image/stabilityai__stable-diffusion-xl-base-1.0",
        )
        self.assertEqual(image_runner["environment"]["SERIES_IMAGE_RESUME_SHOTS"], "1")
        self.assertEqual(video_runner["environment"]["SERIES_VIDEO_RESUME_SHOTS"], "1")
        self.assertEqual(video_runner["environment"]["SERIES_VIDEO_LATEST_MODEL_ID"], "Wan-AI/Wan2.1-T2V-1.3B")
        self.assertEqual(video_runner["environment"]["SERIES_VIDEO_MODEL_ID"], "Wan-AI/Wan2.1-T2V-1.3B")
        self.assertEqual(video_runner["environment"]["SERIES_VIDEO_MODEL_FAMILY"], "wan")
        self.assertEqual(video_runner["environment"]["SERIES_VIDEO_WIDTH"], "1216")
        self.assertEqual(video_runner["environment"]["SERIES_VIDEO_HEIGHT"], "704")

    def test_ltx_video_prompt_uses_identity_set_and_writer_room_context(self) -> None:
        scene_package = {
            "summary": "Babe tries to hide a shortcut before the demo.",
            "characters": ["Babe Carano", "Kenzie Bell"],
            "behavior_constraints": ["Babe deflects blame quickly"],
            "dialogue_style_constraints": ["Kenzie uses analytical corrections"],
            "conflict": "the shortcut broke the demo",
            "set_context": {
                "name": "Game Shakers office",
                "visual_description": "bright studio set with game posters",
                "lighting": "bright sitcom studio lighting",
                "camera_axis": "front-facing multi-camera setup",
                "key_props": ["tablet", "monitors"],
            },
            "writer_room_plan": {
                "scene_function": "escalation",
                "conflict_source": "Babe hides an improvisation",
                "who_drives_scene": "Babe",
                "who_resists_scene": "Kenzie",
                "who_gets_punchline": "Hudson",
            },
            "character_continuity_lock": {
                "Babe Carano": {"outfit_lock": True, "hair_lock": True, "voice_lock": True},
                "Kenzie Bell": {"outfit_lock": True, "hair_lock": True, "voice_lock": True},
            },
            "dialogue_line_metadata": [
                {
                    "line_index": 0,
                    "speaker": "Babe Carano",
                    "dialogue_function": "deflects blame",
                    "physical_action": "hides tablet",
                    "facial_expression": "forced confidence",
                }
            ],
            "shot_packages": [
                {
                    "shot_id": "scene_0001_shot_001",
                    "characters_visible": ["Babe Carano", "Kenzie Bell"],
                    "dialogue_line_indices": [0],
                }
            ],
            "video_generation": {"prompt": "generate a grounded sitcom shot"},
        }

        prompt = VIDEO_BACKEND.prompt_from_package(
            {"shot_id": "scene_0001_shot_001", "shot_type": "medium two-shot"},
            scene_package,
        )
        negative = VIDEO_BACKEND.negative_prompt_from_package(scene_package)

        self.assertIn("visible characters: Babe Carano, Kenzie Bell", prompt)
        self.assertIn("same canonical face from references", prompt)
        self.assertIn("Game Shakers office", prompt)
        self.assertIn("who gets punchline: Hudson", prompt)
        self.assertIn("shot acting beats", prompt)
        self.assertIn("no gender swaps", prompt)
        self.assertIn("wrong gender", negative)

    def test_local_video_model_dir_uses_only_the_configured_wan_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "wan"
            model_dir.mkdir(parents=True)
            (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"weights")
            with mock.patch.object(VIDEO_BACKEND, "DEFAULT_VIDEO_MODEL_DIR", model_dir), mock.patch.object(
                VIDEO_BACKEND, "FALLBACK_VIDEO_MODEL_DIRS", [model_dir]
            ), mock.patch.dict(
                "os.environ",
                {"SERIES_VIDEO_MODEL_DIR": "", "SERIES_VIDEO_MODEL_ID": ""},
                clear=False,
            ):
                resolved = VIDEO_BACKEND.resolve_model_dir()

        self.assertEqual(resolved, model_dir.resolve(strict=False))

    def test_ltx_intro_resume_keeps_completed_shot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            completed_clip = root / "shot_001.mp4"
            completed_manifest = root / "shot_001.json"
            pending_clip = root / "shot_002.mp4"
            pending_manifest = root / "shot_002.json"
            completed_clip.write_bytes(b"video")
            completed_manifest.write_text("{}", encoding="utf-8")
            scene_package = {
                "shot_packages": [
                    {
                        "shot_id": "shot_001",
                        "target_outputs": {
                            "primary_frame": str(root / "shot_001.png"),
                            "video_clip": str(completed_clip),
                            "video_manifest": str(completed_manifest),
                        },
                    },
                    {
                        "shot_id": "shot_002",
                        "target_outputs": {
                            "primary_frame": str(root / "shot_002.png"),
                            "video_clip": str(pending_clip),
                            "video_manifest": str(pending_manifest),
                        },
                    },
                ]
            }

            def fake_generate(_context, _package, output):
                output.write_bytes(b"new-video")

            with mock.patch.dict("os.environ", {"SERIES_VIDEO_RESUME_SHOTS": "1"}), mock.patch.object(
                VIDEO_BACKEND,
                "generate_ltx_video",
                side_effect=fake_generate,
            ) as generate_mock, mock.patch.object(
                VIDEO_BACKEND,
                "write_shot_manifest",
            ), mock.patch.object(
                VIDEO_BACKEND,
                "concat_video_clips",
            ) as concat_mock:
                result = VIDEO_BACKEND.generate_ltx_shots({}, scene_package, root / "scene.mp4")

        self.assertTrue(result)
        self.assertEqual(generate_mock.call_count, 1)
        self.assertEqual(generate_mock.call_args.args[2], pending_clip)
        self.assertEqual(concat_mock.call_args.args[0], [completed_clip, pending_clip])

    def test_sdxl_intro_resume_keeps_completed_shot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            completed_frame = root / "shot_001.png"
            completed_manifest = root / "shot_001.json"
            pending_frame = root / "shot_002.png"
            completed_frame.write_bytes(b"image")
            completed_manifest.write_text("{}", encoding="utf-8")
            scene_package = {
                "characters": [],
                "shot_packages": [
                    {
                        "shot_id": "shot_001",
                        "characters_visible": [],
                        "target_outputs": {
                            "primary_frame": str(completed_frame),
                            "image_manifest": str(completed_manifest),
                        },
                    },
                    {
                        "shot_id": "shot_002",
                        "characters_visible": [],
                        "target_outputs": {
                            "primary_frame": str(pending_frame),
                            "image_manifest": str(root / "shot_002.json"),
                        },
                    },
                ],
            }

            def fake_generate(_prompt, _negative, output, _seed, **_kwargs):
                output.write_bytes(b"new-image")
                return {}

            with mock.patch.dict("os.environ", {"SERIES_IMAGE_RESUME_SHOTS": "1"}), mock.patch.object(
                IMAGE_BACKEND,
                "generate_image",
                side_effect=fake_generate,
            ) as generate_mock, mock.patch.object(
                IMAGE_BACKEND,
                "shot_manifest",
            ):
                paths = IMAGE_BACKEND.generate_shot_images({}, scene_package, "prompt", "negative")

        self.assertEqual(paths, [completed_frame, pending_frame])
        self.assertEqual(generate_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
