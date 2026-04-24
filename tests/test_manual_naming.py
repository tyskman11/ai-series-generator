from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from PIL import Image

from pipeline_common import (
    acquire_distributed_lease,
    adapter_training_status,
    backend_fine_tune_status,
    canonical_person_name,
    detect_tool,
    ensure_foundation_training_ready,
    fine_tune_training_status,
    foundation_training_status,
    generated_episode_completion_summary,
    generated_episode_artifacts,
    list_generated_episode_artifacts,
    has_manual_person_name,
    has_primary_person_name,
    is_background_person_name,
    latest_matching_file,
    open_face_review_item_count,
    open_review_item_count,
    pip_install_command,
    platform_tool_filenames,
    release_distributed_lease,
    resolve_stored_project_path,
    runtime_python,
    external_tool_arg,
    external_tool_command,
)


ROOT = Path(__file__).resolve().parents[1]


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP05 = load_module("05_link_faces_and_speakers.py", "step05")
STEP00 = load_module("00_prepare_runtime.py", "step00")
STEP03 = load_module("03_split_scenes.py", "step03")
STEP02 = load_module("02_import_episode.py", "step02")
STEP04 = load_module("04_diarize_and_transcribe.py", "step04")
STEP06 = load_module("07_build_dataset.py", "step06")
STEP07 = load_module("08_train_series_model.py", "step07")
STEP08 = load_module("06_review_unknowns.py", "step08")
STEP10 = load_module("17_render_episode.py", "step10")
STEP10TRAIN = load_module("10_train_foundation_models.py", "step10train")
STEP13 = load_module("09_prepare_foundation_training.py", "step13")
STEP15 = load_module("14_generate_episode_from_trained_model.py", "step15")
STEP15ASSETS = load_module("15_generate_storyboard_assets.py", "step15assets")
STEP16BACKEND = load_module("16_run_storyboard_backend.py", "step16backend")
STEP16 = load_module("20_refresh_after_manual_review.py", "step16")
STEP17 = load_module("11_train_adapter_models.py", "step17")
STEP18 = load_module("12_train_fine_tune_models.py", "step18")
STEP19 = load_module("13_run_backend_finetunes.py", "step19")
STEP19PREVIEW = load_module("19_generate_finished_episodes.py", "step19preview")
STEPBIBLE = load_module("18_build_series_bible.py", "stepbible")
STEP99 = load_module("99_process_next_episode.py", "step99")
STEPRENDER = load_module("17_render_episode.py", "steprender")


class ManualNamingTests(unittest.TestCase):
    def test_pip_install_command_adds_break_system_packages_when_supported(self) -> None:
        with mock.patch("pipeline_common.pip_supports_break_system_packages", return_value=True):
            command = pip_install_command("C:/Python/python.exe", "--upgrade", "wheel")
        self.assertEqual(
            command,
            ["C:\\Python\\python.exe", "-m", "pip", "install", "--break-system-packages", "--upgrade", "wheel"],
        )

    def test_pip_install_command_skips_break_system_packages_when_unsupported(self) -> None:
        with mock.patch("pipeline_common.pip_supports_break_system_packages", return_value=False):
            command = pip_install_command("C:/Python/python.exe", "--upgrade", "wheel")
        self.assertEqual(
            command,
            ["C:\\Python\\python.exe", "-m", "pip", "install", "--upgrade", "wheel"],
        )

    def test_distributed_lease_allows_takeover_after_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lease_root = Path(tmpdir)
            with mock.patch("pipeline_common.time.time", return_value=100.0):
                first = acquire_distributed_lease(lease_root, "scene_001", "worker-a", 30.0, meta={"scene_id": "scene_001"})
            self.assertIsNotNone(first)
            with mock.patch("pipeline_common.time.time", return_value=105.0):
                second = acquire_distributed_lease(lease_root, "scene_001", "worker-b", 30.0)
            self.assertIsNone(second)
            with mock.patch("pipeline_common.time.time", return_value=150.0):
                takeover = acquire_distributed_lease(lease_root, "scene_001", "worker-b", 30.0)
            self.assertIsNotNone(takeover)
            self.assertEqual(takeover["owner_id"], "worker-b")
            self.assertTrue(release_distributed_lease(lease_root, "scene_001", "worker-b"))

    def test_step04_scene_cache_completed_accepts_empty_finished_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "scene_001.json"
            cache_file.write_text("[]", encoding="utf-8")
            self.assertTrue(STEP04.scene_cache_completed(cache_file))
            cache_file.write_text(json.dumps([{"process_version": STEP04.PROCESS_VERSION - 1}]), encoding="utf-8")
            self.assertFalse(STEP04.scene_cache_completed(cache_file))

    def test_step04_parse_args_supports_shared_worker_flags(self) -> None:
        with mock.patch(
            "sys.argv",
            ["04_diarize_and_transcribe.py", "--episode", "Demo.S01E01", "--no-shared-workers", "--worker-id", "pc2"],
        ):
            args = STEP04.parse_args()

        self.assertEqual(args.episode, "Demo.S01E01")
        self.assertTrue(args.no_shared_workers)
        self.assertEqual(args.worker_id, "pc2")

    def test_prepare_runtime_installs_torch_before_torch_dependent_groups(self) -> None:
        install_order: list[str] = []

        def fake_install_group(_py, name, _modules, _packages, required=True, pip_extra_args=None):
            install_order.append(name)
            return True

        with mock.patch.object(STEP00, "ensure_project_structure", return_value={"cloning": {}}), mock.patch.object(
            STEP00, "ensure_venv", return_value=Path("C:/Python/python.exe")
        ), mock.patch.object(STEP00, "runtime_environment_tag", return_value="test"), mock.patch.object(
            STEP00, "runtime_venv_dir", return_value=Path("C:/runtime")
        ), mock.patch.object(
            STEP00, "pip_install_command", return_value=["python", "-m", "pip", "install"]
        ), mock.patch.object(
            STEP00, "run", return_value=mock.Mock(returncode=0, stdout="")
        ), mock.patch.object(
            STEP00, "install_group", side_effect=fake_install_group
        ), mock.patch.object(
            STEP00, "install_ffmpeg_binaries", return_value={"ffmpeg": "ffmpeg"}
        ), mock.patch.object(
            STEP00, "install_torch_stack", side_effect=lambda *_args, **_kwargs: (install_order.append("torch") or True, {"cuda_available": False})
        ), mock.patch.object(
            STEP00, "write_json"
        ), mock.patch.object(
            STEP00, "mark_step_started"
        ), mock.patch.object(
            STEP00, "mark_step_completed"
        ), mock.patch.object(
            STEP00, "nvidia_gpu_available", return_value=False
        ), mock.patch.object(
            STEP00, "headline"
        ), mock.patch.object(
            STEP00, "info"
        ), mock.patch.object(
            STEP00, "ok"
        ):
            STEP00.main()

        self.assertLess(install_order.index("core_ai"), install_order.index("torch"))
        self.assertLess(install_order.index("torch"), install_order.index("speech_to_text"))
        self.assertLess(install_order.index("torch"), install_order.index("face_recognition"))
        self.assertLess(install_order.index("torch"), install_order.index("speaker_embeddings"))

    def test_platform_tool_filenames_are_os_specific(self) -> None:
        self.assertEqual(platform_tool_filenames("ffmpeg", "windows"), ["ffmpeg.exe", "ffmpeg"])
        self.assertEqual(platform_tool_filenames("ffmpeg", "linux"), ["ffmpeg"])

    def test_detect_tool_ignores_windows_exe_on_linux(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_dir = Path(tmpdir)
            (bin_dir / "ffmpeg.exe").write_text("not-a-linux-binary", encoding="utf-8")
            with mock.patch("pipeline_common.current_os", return_value="linux"), mock.patch(
                "pipeline_common.tool_on_path", return_value=None
            ):
                with self.assertRaises(FileNotFoundError):
                    detect_tool(bin_dir, "ffmpeg")

    def test_detect_tool_prefers_linux_binary_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_dir = Path(tmpdir)
            linux_binary = bin_dir / "ffmpeg"
            linux_binary.write_text("linux-binary", encoding="utf-8")
            (bin_dir / "ffmpeg.exe").write_text("windows-binary", encoding="utf-8")
            with mock.patch("pipeline_common.current_os", return_value="linux"):
                self.assertEqual(detect_tool(bin_dir, "ffmpeg"), linux_binary)

    def test_ffmpeg_asset_spec_uses_matching_archive_per_os(self) -> None:
        with mock.patch.object(STEP00, "current_os", return_value="linux"), mock.patch.object(
            STEP00, "current_architecture", return_value="x86_64"
        ):
            linux_spec = STEP00.ffmpeg_asset_spec()
        with mock.patch.object(STEP00, "current_os", return_value="windows"), mock.patch.object(
            STEP00, "current_architecture", return_value="x86_64"
        ):
            windows_spec = STEP00.ffmpeg_asset_spec()

        self.assertTrue(linux_spec["url"].endswith("ffmpeg-master-latest-linux64-gpl.tar.xz"))
        self.assertTrue(windows_spec["url"].endswith("ffmpeg-master-latest-win64-gpl.zip"))

    def test_prepare_runtime_pip_install_command_keeps_break_system_packages_install(self) -> None:
        with mock.patch.object(
            STEP00,
            "pip_install_command",
            return_value=["/usr/bin/python3", "-m", "pip", "install", "--break-system-packages", "--upgrade", "numpy"],
        ):
            command = STEP00.runtime_pip_install_command(Path("/tmp/runtime/bin/python3"), "--upgrade", "numpy")

        self.assertEqual(
            command,
            ["/usr/bin/python3", "-m", "pip", "install", "--break-system-packages", "--upgrade", "numpy"],
        )

    def test_runtime_python_uses_active_python_on_linux(self) -> None:
        with mock.patch("pipeline_common.current_os", return_value="linux"), mock.patch("pipeline_common.sys.executable", "/usr/bin/python3"):
            self.assertEqual(runtime_python(), Path("/usr/bin/python3").resolve())

    def test_ensure_venv_uses_active_python_on_linux(self) -> None:
        with mock.patch.object(STEP00.sys, "platform", "linux"), mock.patch.object(STEP00.sys, "executable", "/usr/bin/python3"), mock.patch.object(
            STEP00,
            "headline",
        ), mock.patch.object(
            STEP00,
            "info",
        ):
            self.assertEqual(STEP00.ensure_venv(), Path("/usr/bin/python3").resolve())

    def test_latest_matching_file_uses_mtime_not_filename_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            older = root / "folge_09.json"
            newer = root / "episode_special.json"
            older.write_text("{}", encoding="utf-8")
            newer.write_text("{}", encoding="utf-8")
            os.utime(older, (1000, 1000))
            os.utime(newer, (2000, 2000))

            self.assertEqual(latest_matching_file(root, "*.json"), newer)

    def test_preview_latest_episode_id_uses_newest_story_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            older = root / "folge_09.md"
            newer = root / "episode_special.md"
            older.write_text("# older", encoding="utf-8")
            newer.write_text("# newer", encoding="utf-8")
            os.utime(older, (1000, 1000))
            os.utime(newer, (2000, 2000))

            with mock.patch.object(STEP19PREVIEW, "story_dir", return_value=root):
                self.assertEqual(STEP19PREVIEW.latest_episode_id(), "episode_special")

    def test_manual_name_detection(self) -> None:
        self.assertTrue(has_manual_person_name("Babe Carano"))
        self.assertTrue(has_manual_person_name("Mr. Sammich"))
        self.assertTrue(has_manual_person_name("statist"))
        self.assertTrue(is_background_person_name("statist"))
        self.assertFalse(has_primary_person_name("statist"))
        self.assertEqual(canonical_person_name("Statisten"), "statist")
        self.assertFalse(has_manual_person_name("face_001"))
        self.assertFalse(has_manual_person_name("figur_022"))
        self.assertFalse(has_manual_person_name("speaker_003"))
        self.assertFalse(has_manual_person_name("stimme_004"))
        self.assertFalse(has_manual_person_name("noface"))

    def test_voice_link_keeps_speaker_id_until_face_is_manually_named(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {"name": "face_001", "ignored": False, "auto_named": True},
            }
        }
        voice_map = {"clusters": {}}
        speaker_votes = {"speaker_001": {"face_001": 3.0}}

        STEP05.resolve_voice_names(voice_map, speaker_votes, char_map)

        payload = voice_map["clusters"]["speaker_001"]
        self.assertEqual(payload["linked_face_cluster"], "face_001")
        self.assertEqual(payload["name"], "speaker_001")
        self.assertTrue(payload["auto_named"])

    def test_voice_link_uses_manual_face_name_when_available(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {"name": "Babe Carano", "ignored": False, "auto_named": False},
            }
        }
        voice_map = {"clusters": {}}
        speaker_votes = {"speaker_001": {"face_001": 3.0}}

        STEP05.resolve_voice_names(voice_map, speaker_votes, char_map)

        payload = voice_map["clusters"]["speaker_001"]
        self.assertEqual(payload["linked_face_cluster"], "face_001")
        self.assertEqual(payload["name"], "Babe Carano")
        self.assertTrue(payload["auto_named"])

    def test_unknown_speaker_rescues_from_single_visible_named_face(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {"name": "Babe Carano", "ignored": False, "auto_named": False},
                "face_002": {"name": "statist", "ignored": False, "background_role": True},
                "face_003": {"name": "face_003", "ignored": False, "auto_named": True},
            }
        }

        rescued = STEP05.rescue_unknown_speaker_from_single_visible_face(char_map, ["face_001", "face_002"])
        unresolved = STEP05.rescue_unknown_speaker_from_single_visible_face(char_map, ["face_002", "face_003"])

        self.assertEqual(rescued, ("face_001", "Babe Carano"))
        self.assertIsNone(unresolved)

    def test_unknown_speaker_rescue_requires_one_named_primary_face(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {"name": "Babe Carano", "ignored": False, "auto_named": False},
                "face_002": {"name": "Kenzie Bell", "ignored": False, "auto_named": False},
            }
        }

        rescued = STEP05.rescue_unknown_speaker_from_single_visible_face(char_map, ["face_001", "face_002"])

        self.assertIsNone(rescued)

    def test_step05_live_reporter_uses_scene_payload_face_clusters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            episode_dir = root / "episode_001"
            episode_dir.mkdir(parents=True, exist_ok=True)
            (episode_dir / "scene_001.mp4").write_bytes(b"placeholder")

            transcripts_dir = root / "speaker_transcripts"
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            linked_dir = root / "linked_segments"
            linked_dir.mkdir(parents=True, exist_ok=True)
            faces_root = root / "faces"
            previews_root = root / "previews"
            maps_dir = root / "maps"
            review_dir = root / "review"
            maps_dir.mkdir(parents=True, exist_ok=True)
            review_dir.mkdir(parents=True, exist_ok=True)

            (maps_dir / "character_map.json").write_text('{"clusters": {}}', encoding="utf-8")
            (maps_dir / "voice_map.json").write_text('{"clusters": {}}', encoding="utf-8")
            (review_dir / "review_queue.json").write_text('{"items": []}', encoding="utf-8")
            (transcripts_dir / "episode_001_segments.json").write_text(
                json.dumps(
                    [
                        {
                            "scene_id": "scene_001",
                            "segment_id": "segment_001",
                            "start": 0.0,
                            "end": 1.0,
                            "speaker_cluster": "speaker_unknown",
                            "text": "hello",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            cfg = {
                "paths": {
                    "speaker_transcripts": str(transcripts_dir),
                    "linked_segments": str(linked_dir),
                    "faces": str(faces_root),
                    "character_map": str(maps_dir / "character_map.json"),
                    "voice_map": str(maps_dir / "voice_map.json"),
                    "review_queue": str(review_dir / "review_queue.json"),
                }
            }
            char_map = {"clusters": {}}
            voice_map = {"clusters": {}}
            live_reporter = mock.Mock()

            with mock.patch.object(STEP05, "episode_face_linking_completed", return_value=False), mock.patch.object(
                STEP05, "reset_step_outputs"
            ), mock.patch.object(
                STEP05,
                "process_scene_faces",
                return_value={"scene_id": "scene_001", "face_clusters": ["face_001", "face_002"], "detections": []},
            ), mock.patch.object(
                STEP05, "save_step_autosave"
            ), mock.patch.object(
                STEP05, "mark_step_started"
            ), mock.patch.object(
                STEP05, "mark_step_completed"
            ), mock.patch.object(
                STEP05, "mark_step_failed"
            ), mock.patch.object(
                STEP05, "prune_face_clusters", return_value=["face_001", "face_002"]
            ), mock.patch.object(
                STEP05, "visible_faces_for_segment", return_value=[]
            ), mock.patch.object(
                STEP05, "resolve_voice_names"
            ), mock.patch.object(
                STEP05, "LiveProgressReporter"
            ), mock.patch.object(
                STEP05, "info"
            ), mock.patch.object(
                STEP05, "ok"
            ):
                result = STEP05.process_episode_dir(
                    episode_dir,
                    cfg,
                    fresh_run=False,
                    faces_root=faces_root,
                    linked_root=linked_dir,
                    previews_root=previews_root,
                    char_map=char_map,
                    voice_map=voice_map,
                    engine=None,
                    interactive=False,
                    auto_open=False,
                    sample_every=1,
                    max_faces_per_frame=4,
                    max_visible_faces_per_segment=3,
                    segment_padding_seconds=0.5,
                    threshold=0.8,
                    live_reporter=live_reporter,
                    episode_index=1,
                    episode_total=1,
                )

        self.assertTrue(result)
        self.assertEqual(live_reporter.update.call_args_list[0].kwargs["extra_label"], "Face clusters so far: 2")

    def test_episode_generation_uses_generic_focus_without_manual_names(self) -> None:
        model = {
            "characters": [],
            "speakers": {"speaker_001": 12, "speaker_002": 8},
            "keywords": ["app", "chaos"],
            "dataset_files": [],
            "scene_count": 5,
            "speaker_samples": {},
            "character_reference_library": {},
            "scene_library": [],
        }
        cfg = {
            "generation": {
                "seed": 42,
                "default_scene_count": 2,
                "min_dialogue_lines_per_scene": 4,
                "max_dialogue_lines_per_scene": 6,
            }
        }

        package, markdown = STEP07.generate_episode_package(model, cfg)

        self.assertEqual(package["focus_characters"][:2], ["Hauptfigur A", "Hauptfigur B"])
        self.assertNotIn("speaker_001", markdown)
        self.assertNotIn("figur_001", markdown)

    def test_episode_generation_builds_multi_reference_storyboard_plan(self) -> None:
        model = {
            "characters": [
                {"name": "Babe", "scene_count": 20, "line_count": 30, "priority": True, "face_cluster_count": 3},
                {"name": "Kenzie", "scene_count": 18, "line_count": 25, "priority": True, "face_cluster_count": 2},
            ],
            "speakers": {"Babe": 20, "Kenzie": 19},
            "keywords": ["videospiele", "chaos", "plan"],
            "dataset_files": ["demo_dataset.json"],
            "scene_count": 8,
            "speaker_samples": {"Babe": ["Wir schaffen das."], "Kenzie": ["Dann machen wir es richtig."]},
            "speaker_line_library": {"Babe": [], "Kenzie": []},
            "character_reference_library": {
                "Babe": {
                    "context_images": ["C:/refs/babe_context.jpg"],
                    "portrait_images": ["C:/refs/babe_portrait.jpg"],
                    "priority": True,
                },
                "Kenzie": {
                    "context_images": ["C:/refs/kenzie_context.jpg"],
                    "portrait_images": ["C:/refs/kenzie_portrait.jpg"],
                    "priority": True,
                },
            },
            "scene_library": [
                {
                    "episode_id": "Game.Shakers.S01E01",
                    "scene_id": "scene_0003",
                    "characters": ["Babe", "Kenzie"],
                    "keywords": ["videospiele"],
                    "video_file": "C:/scenes/scene_0003.mp4",
                    "duration_seconds": 12.0,
                }
            ],
            "average_segment_duration_seconds": 2.7,
            "source_episode_durations": {"Game.Shakers.S01E01": 1200},
        }
        cfg = {
            "generation": {
                "seed": 42,
                "default_scene_count": 2,
                "min_dialogue_lines_per_scene": 4,
                "max_dialogue_lines_per_scene": 6,
                "match_source_episode_runtime": True,
                "target_scene_duration_seconds": 42.0,
                "estimated_dialogue_line_seconds": 2.7,
            }
        }

        package, markdown = STEP07.generate_episode_package(model, cfg, episode_index=2)

        self.assertEqual(package["storyboard_plan_mode"], "multi_reference_storyboard")
        self.assertTrue(package["scenes"])
        first_scene = package["scenes"][0]
        first_plan = first_scene["generation_plan"]
        self.assertEqual(first_plan["model_mode"], "multi_reference_storyboard")
        self.assertTrue(first_plan["reference_slots"])
        self.assertEqual(first_plan["reference_slots"][0]["slot"], "subject_1")
        self.assertIn("positive_prompt", first_plan)
        self.assertIn("negative_prompt", first_plan)
        if len(package["scenes"]) > 1:
            second_plan = package["scenes"][1]["generation_plan"]
            self.assertEqual(second_plan["continuity"]["previous_scene_id"], package["scenes"][0]["scene_id"])
        self.assertIn("Shot Plan:", markdown)

    def test_storyboard_backend_requests_are_written_from_shotlist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = {"paths": {"storyboard_requests": str(root / "generation" / "storyboard_requests")}}
            shotlist_payload = {
                "display_title": "Folge 99: Test",
                "episode_title": "Test",
                "focus_characters": ["Babe", "Kenzie"],
                "keywords": ["videospiele"],
                "storyboard_plan_mode": "multi_reference_storyboard",
                "scenes": [
                    {
                        "scene_id": "scene_0001",
                        "title": "Cold Open",
                        "characters": ["Babe", "Kenzie"],
                        "summary": "Test summary",
                        "location": "Set 1",
                        "mood": "energetisch",
                        "generation_plan": {
                            "batch_prompt_line": "Cold Open | Babe, Kenzie",
                            "positive_prompt": "prompt",
                            "negative_prompt": "negative",
                        },
                    }
                ],
            }
            result = STEP15.write_storyboard_backend_requests(cfg, "folge_99", shotlist_payload)
            self.assertTrue(Path(result["episode_request_path"]).exists())
            self.assertTrue(Path(result["prompt_preview_path"]).exists())
            self.assertEqual(len(result["scene_request_paths"]), 1)
            payload = json.loads(Path(result["episode_request_path"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["episode_id"], "folge_99")
            self.assertEqual(payload["scene_requests"][0]["scene_id"], "scene_0001")

    def test_build_scene_asset_writes_backend_input_with_ready_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            assets_root = root / "generation" / "storyboard_assets" / "folge_99"
            refs_root = root / "refs"
            refs_root.mkdir(parents=True, exist_ok=True)
            reference_image = refs_root / "babe_ref.jpg"
            Image.new("RGB", (640, 360), color=(120, 80, 150)).save(reference_image)

            backend_dir = root / "backend_runs" / "babe" / "image"
            backend_dir.mkdir(parents=True, exist_ok=True)
            job_path = backend_dir / "training_job.json"
            bundle_path = backend_dir / "model_bundle.json"
            weights_path = backend_dir / "image_weights.bin"
            job_path.write_text("{}", encoding="utf-8")
            bundle_path.write_text("{}", encoding="utf-8")
            weights_path.write_bytes(b"demo")

            backend_index = {
                "babe": {
                    "character": "Babe",
                    "training_ready": True,
                    "backends": {
                        "image": {
                            "backend": "lora-image",
                            "ready": True,
                            "artifacts": {
                                "job_path": str(job_path),
                                "bundle_path": str(bundle_path),
                                "weights_path": str(weights_path),
                            },
                        }
                    },
                }
            }
            scene_request = {
                "scene_id": "scene_0001",
                "title": "Cold Open",
                "characters": ["Babe"],
                "generation_plan": {
                    "positive_prompt": "prompt",
                    "negative_prompt": "negative",
                    "batch_prompt_line": "scene line",
                    "reference_slots": [
                        {"type": "character", "portrait_images": [str(reference_image)], "context_images": []}
                    ],
                },
            }

            result = STEP15ASSETS.build_scene_asset(
                Path("ffmpeg"),
                assets_root,
                "folge_99",
                scene_request,
                backend_index,
                1280,
                720,
                False,
            )

            self.assertEqual(result["source_type"], "reference_seed")
            self.assertIn("Babe", result["backend_ready_image_characters"])
            backend_input = Path(result["backend_input_path"])
            self.assertTrue(backend_input.exists())
            payload = json.loads(backend_input.read_text(encoding="utf-8"))
            self.assertEqual(payload["scene_id"], "scene_0001")
            self.assertEqual(payload["backend_candidates"]["image"][0]["character"], "Babe")
            self.assertEqual(Path(payload["backend_candidates"]["image"][0]["bundle_path"]), bundle_path)

    def test_generated_storyboard_scene_frame_prefers_existing_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            asset = root / "generation" / "storyboard_assets" / "folge_07" / "scene_0003.png"
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.write_bytes(b"demo")
            cfg = {"paths": {"storyboard_assets": str(root / "generation" / "storyboard_assets")}}
            result = STEPRENDER.generated_storyboard_scene_frame(cfg, "folge_07", "scene_0003")
            self.assertEqual(result, asset)

    def test_resolve_stored_project_path_rebases_old_ai_series_project_path(self) -> None:
        rebased_target = ROOT / "ai_series_project" / "tmp" / "rebased_demo.json"
        rebased_target.parent.mkdir(parents=True, exist_ok=True)
        rebased_target.write_text("{}", encoding="utf-8")
        old_path = Path("C:/Old/Workspace/KI Serien Training/ai_series_project/tmp/rebased_demo.json")
        self.assertEqual(resolve_stored_project_path(old_path), rebased_target)

    def test_external_tool_arg_strips_windows_extended_unc_prefix(self) -> None:
        with mock.patch("pipeline_common.current_os", return_value="windows"):
            normalized = external_tool_arg(r"\\?\UNC\DXP4800PLUS-41A\share\file.wav")
            command = external_tool_command([r"\\?\UNC\DXP4800PLUS-41A\share\ffmpeg.exe", "-i", r"\\?\C:\tmp\in.wav"])

        self.assertEqual(normalized, r"\\DXP4800PLUS-41A\share\file.wav")
        self.assertEqual(command[0], r"\\DXP4800PLUS-41A\share\ffmpeg.exe")
        self.assertEqual(command[2], r"C:\tmp\in.wav")

    def test_normalize_placeholder_maps_rewrites_old_auto_names(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {"name": "figur_001", "ignored": False, "auto_named": True, "aliases": ["figur_001"]},
                "face_002": {"name": "Babe Carano", "ignored": False, "auto_named": True, "aliases": []},
            },
            "aliases": {"figur_001": "face_001"},
        }
        voice_map = {
            "clusters": {
                "speaker_001": {"name": "figur_001", "linked_face_cluster": "face_001", "auto_named": True},
                "speaker_002": {"name": "Babe Carano", "linked_face_cluster": "face_002", "auto_named": True},
            },
            "aliases": {},
        }

        changed_faces, changed_voices = STEP08.normalize_placeholder_maps(char_map, voice_map)

        self.assertGreaterEqual(changed_faces, 1)
        self.assertGreaterEqual(changed_voices, 1)
        self.assertEqual(char_map["clusters"]["face_001"]["name"], "face_001")
        self.assertEqual(char_map["clusters"]["face_001"]["aliases"], [])
        self.assertEqual(char_map["clusters"]["face_002"]["name"], "Babe Carano")
        self.assertFalse(char_map["clusters"]["face_002"]["auto_named"])
        self.assertEqual(char_map["aliases"], {"babe carano": "face_002"})
        self.assertEqual(voice_map["clusters"]["speaker_001"]["name"], "speaker_001")
        self.assertEqual(voice_map["clusters"]["speaker_002"]["name"], "Babe Carano")

    def test_auto_match_known_faces_merges_unknown_cluster_before_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            linked_root = temp_root / "linked_segments"
            linked_root.mkdir(parents=True, exist_ok=True)
            review_queue = temp_root / "review_queue.json"
            character_map_path = temp_root / "character_map.json"
            voice_map_path = temp_root / "voice_map.json"
            linked_file = linked_root / "demo_linked_segments.json"

            with linked_file.open("w", encoding="utf-8") as handle:
                json.dump(
                    [
                        {
                            "scene_id": "scene_0001",
                            "speaker_cluster": "speaker_001",
                            "speaker_name": "speaker_001",
                            "visible_face_clusters": ["face_099"],
                            "visible_character_names": ["face_099"],
                            "speaker_face_cluster": "face_099",
                        }
                    ],
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )

            cfg = {
                "paths": {
                    "linked_segments": str(linked_root),
                    "review_queue": str(review_queue),
                    "character_map": str(character_map_path),
                    "voice_map": str(voice_map_path),
                },
                "character_detection": {
                    "embedding_threshold": 0.80,
                    "review_known_face_threshold": 0.88,
                    "review_known_face_margin": 0.03,
                },
            }
            char_map = {
                "clusters": {
                    "face_001": {
                        "name": "Babe Carano",
                        "ignored": False,
                        "auto_named": False,
                        "priority": True,
                        "embedding": [1.0, 0.0, 0.0],
                        "samples": 3,
                        "preview_dir": str(temp_root / "face_001"),
                    },
                    "face_099": {
                        "name": "face_099",
                        "ignored": False,
                        "auto_named": True,
                        "embedding": [0.99, 0.01, 0.0],
                        "samples": 1,
                        "preview_dir": str(temp_root / "face_099"),
                    },
                },
                "aliases": {"babe carano": "face_001"},
            }
            voice_map = {
                "clusters": {
                    "speaker_001": {
                        "name": "speaker_001",
                        "linked_face_cluster": "face_099",
                        "auto_named": True,
                    }
                },
                "aliases": {},
            }

            summary = STEP08.auto_match_known_faces(cfg, char_map, voice_map)
            changed_linked_files, review_count = STEP08.persist_updates(cfg, char_map, voice_map)

            self.assertEqual(summary["matched"], 1)
            self.assertEqual(summary["linked_files"], 1)
            self.assertNotIn("face_099", char_map["clusters"])
            self.assertEqual(voice_map["clusters"]["speaker_001"]["linked_face_cluster"], "face_001")
            self.assertEqual(voice_map["clusters"]["speaker_001"]["name"], "Babe Carano")
            self.assertGreaterEqual(changed_linked_files, 1)
            self.assertEqual(review_count, 0)

            with linked_file.open("r", encoding="utf-8") as handle:
                rows = json.load(handle)
            self.assertEqual(rows[0]["visible_face_clusters"], ["face_001"])
            self.assertEqual(rows[0]["speaker_face_cluster"], "face_001")
            self.assertEqual(rows[0]["visible_character_names"], ["Babe Carano"])
            self.assertEqual(rows[0]["speaker_name"], "Babe Carano")

    def test_auto_learn_remaining_reviews_iterates_until_no_face_matches_remain(self) -> None:
        cfg = {
            "paths": {"linked_segments": str(ROOT / "ai_series_project" / "data" / "processed" / "linked_segments")},
            "character_detection": {
                "review_known_face_threshold": 0.74,
                "review_known_face_margin": 0.05,
                "review_known_face_consensus_threshold": 0.60,
                "review_known_face_min_consensus": 1,
                "review_known_face_strong_match_threshold": 0.80,
            },
        }
        char_map = {
            "clusters": {
                "face_001": {
                    "name": "Babe",
                    "ignored": False,
                    "auto_named": False,
                    "scene_count": 12,
                    "detection_count": 20,
                    "samples": 6,
                    "embedding": [1.0, 0.0],
                },
                "face_002": {
                    "name": "face_002",
                    "ignored": False,
                    "auto_named": True,
                    "scene_count": 2,
                    "detection_count": 4,
                    "samples": 1,
                    "embedding": [0.97, 0.03],
                },
                "face_003": {
                    "name": "face_003",
                    "ignored": False,
                    "auto_named": True,
                    "scene_count": 1,
                    "detection_count": 2,
                    "samples": 1,
                    "embedding": [0.965, 0.035],
                },
            },
            "aliases": {"babe": "face_001"},
        }
        voice_map = {"clusters": {}, "aliases": {}}

        summary = STEP08.auto_learn_remaining_reviews(cfg, char_map, voice_map)

        self.assertEqual(summary["matched_faces"], 2)
        self.assertNotIn("face_002", char_map["clusters"])
        self.assertNotIn("face_003", char_map["clusters"])

    def test_auto_link_speakers_from_single_visible_faces_uses_open_review_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            linked_root = temp_root / "linked_segments"
            linked_root.mkdir(parents=True, exist_ok=True)
            linked_file = linked_root / "demo_linked_segments.json"
            rows = [
                {
                    "scene_id": "scene_0001",
                    "speaker_cluster": "speaker_007",
                    "speaker_name": "speaker_007",
                    "visible_face_clusters": ["face_001"],
                    "visible_character_names": ["Babe"],
                    "speaker_face_cluster": None,
                },
                {
                    "scene_id": "scene_0002",
                    "speaker_cluster": "speaker_007",
                    "speaker_name": "speaker_007",
                    "visible_face_clusters": ["face_001"],
                    "visible_character_names": ["Babe"],
                    "speaker_face_cluster": None,
                },
                {
                    "scene_id": "scene_0003",
                    "speaker_cluster": "speaker_007",
                    "speaker_name": "speaker_007",
                    "visible_face_clusters": ["face_001"],
                    "visible_character_names": ["Babe"],
                    "speaker_face_cluster": None,
                },
            ]
            linked_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            cfg = {"paths": {"linked_segments": str(linked_root)}}
            char_map = {
                "clusters": {
                    "face_001": {"name": "Babe", "ignored": False, "auto_named": False},
                }
            }
            voice_map = {"clusters": {"speaker_007": {"name": "speaker_007", "auto_named": True}}, "aliases": {}}

            summary = STEP08.auto_link_speakers_from_single_visible_faces(cfg, char_map, voice_map)

            self.assertEqual(summary["matched"], 1)
            self.assertEqual(voice_map["clusters"]["speaker_007"]["linked_face_cluster"], "face_001")
            self.assertEqual(voice_map["clusters"]["speaker_007"]["name"], "Babe")

    def test_plan_known_face_matches_skips_background_and_ignored_faces(self) -> None:
        cfg = {
            "character_detection": {
                "embedding_threshold": 0.80,
                "review_known_face_threshold": 0.74,
                "review_known_face_margin": 0.08,
            }
        }
        char_map = {
            "clusters": {
                "face_001": {"name": "statist", "ignored": False, "auto_named": False, "background_role": True, "embedding": [1.0, 0.0]},
                "face_002": {"name": "noface", "ignored": True, "auto_named": False, "embedding": [1.0, 0.0]},
                "face_003": {"name": "face_003", "ignored": False, "auto_named": True, "embedding": [0.99, 0.01]},
            },
            "aliases": {},
        }

        matches = STEP08.plan_known_face_matches(cfg, char_map)

        self.assertEqual(matches, {})

    def test_plan_known_face_matches_uses_identity_level_reference(self) -> None:
        cfg = {
            "character_detection": {
                "embedding_threshold": 0.80,
                "review_known_face_threshold": 0.74,
                "review_known_face_margin": 0.08,
            }
        }
        char_map = {
            "clusters": {
                "face_001": {
                    "name": "Babe",
                    "ignored": False,
                    "auto_named": False,
                    "priority": True,
                    "scene_count": 12,
                    "embedding": [1.0, 0.0],
                },
                "face_002": {
                    "name": "Babe",
                    "ignored": False,
                    "auto_named": False,
                    "scene_count": 3,
                    "embedding": [0.92, 0.08],
                },
                "face_900": {
                    "name": "face_900",
                    "ignored": False,
                    "auto_named": True,
                    "scene_count": 2,
                    "embedding": [0.95, 0.05],
                },
                "face_300": {
                    "name": "Kenzie",
                    "ignored": False,
                    "auto_named": False,
                    "scene_count": 10,
                    "embedding": [0.0, 1.0],
                },
            },
            "aliases": {"babe": "face_001", "kenzie": "face_300"},
        }

        matches = STEP08.plan_known_face_matches(cfg, char_map)

        self.assertIn("face_900", matches)
        self.assertEqual(matches["face_900"]["target_cluster"], "face_001")
        self.assertEqual(matches["face_900"]["target_identity"], "Babe")

    def test_known_face_reference_identities_keep_multiple_quality_references(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {
                    "name": "Babe",
                    "ignored": False,
                    "auto_named": False,
                    "priority": True,
                    "scene_count": 10,
                    "detection_count": 30,
                    "samples": 6,
                    "embedding": [1.0, 0.0],
                },
                "face_002": {
                    "name": "Babe",
                    "ignored": False,
                    "auto_named": False,
                    "scene_count": 6,
                    "detection_count": 20,
                    "samples": 3,
                    "embedding": [0.96, 0.04],
                },
                "face_003": {
                    "name": "Babe",
                    "ignored": False,
                    "auto_named": False,
                    "scene_count": 1,
                    "detection_count": 3,
                    "samples": 1,
                    "embedding": [0.9, 0.1],
                },
            },
            "aliases": {"babe": "face_001"},
        }

        identities = STEP08.known_face_reference_identities(
            char_map,
            {"character_detection": {"review_known_face_reference_count": 2, "review_known_face_min_reference_quality": 5.0}},
        )

        self.assertIn("Babe", identities)
        self.assertEqual(identities["Babe"]["primary_cluster"], "face_001")
        self.assertEqual([row["cluster_id"] for row in identities["Babe"]["references"]], ["face_001", "face_002"])
        self.assertGreater(identities["Babe"]["identity_strength"], 5.0)

    def test_plan_known_face_matches_requires_consensus_or_strong_single(self) -> None:
        cfg = {
            "character_detection": {
                "review_known_face_threshold": 0.72,
                "review_known_face_margin": 0.05,
                "review_known_face_consensus_threshold": 0.95,
                "review_known_face_min_consensus": 2,
                "review_known_face_strong_match_threshold": 1.01,
            }
        }
        char_map = {
            "clusters": {
                "face_001": {
                    "name": "Babe",
                    "ignored": False,
                    "auto_named": False,
                    "scene_count": 10,
                    "detection_count": 20,
                    "samples": 5,
                    "embedding": [1.0, 0.0],
                },
                "face_002": {
                    "name": "Babe",
                    "ignored": False,
                    "auto_named": False,
                    "scene_count": 8,
                    "detection_count": 16,
                    "samples": 4,
                    "embedding": [0.55, 0.45],
                },
                "face_003": {
                    "name": "Kenzie",
                    "ignored": False,
                    "auto_named": False,
                    "scene_count": 10,
                    "detection_count": 20,
                    "samples": 5,
                    "embedding": [0.0, 1.0],
                },
                "face_999": {
                    "name": "face_999",
                    "ignored": False,
                    "auto_named": True,
                    "scene_count": 1,
                    "detection_count": 3,
                    "samples": 1,
                    "embedding": [0.99, 0.01],
                },
            },
            "aliases": {"babe": "face_001", "kenzie": "face_003"},
        }

        matches = STEP08.plan_known_face_matches(cfg, char_map)

        self.assertNotIn("face_999", matches)

    def test_plan_known_face_matches_relaxes_consensus_for_strong_identity(self) -> None:
        cfg = {
            "character_detection": {
                "review_known_face_threshold": 0.72,
                "review_known_face_margin": 0.05,
                "review_known_face_consensus_threshold": 0.95,
                "review_known_face_min_consensus": 2,
                "review_known_face_strong_match_threshold": 1.01,
                "review_known_face_identity_relaxed_consensus_strength": 5.0,
            }
        }
        char_map = {
            "clusters": {
                "face_001": {
                    "name": "Babe",
                    "ignored": False,
                    "auto_named": False,
                    "priority": True,
                    "scene_count": 12,
                    "detection_count": 26,
                    "samples": 6,
                    "embedding": [1.0, 0.0],
                },
                "face_002": {
                    "name": "Babe",
                    "ignored": False,
                    "auto_named": False,
                    "scene_count": 10,
                    "detection_count": 20,
                    "samples": 5,
                    "embedding": [0.99, 0.01],
                },
                "face_003": {
                    "name": "Babe",
                    "ignored": False,
                    "auto_named": False,
                    "scene_count": 8,
                    "detection_count": 18,
                    "samples": 4,
                    "embedding": [0.98, 0.02],
                },
                "face_100": {
                    "name": "Kenzie",
                    "ignored": False,
                    "auto_named": False,
                    "scene_count": 8,
                    "detection_count": 18,
                    "samples": 4,
                    "embedding": [0.0, 1.0],
                },
                "face_999": {
                    "name": "face_999",
                    "ignored": False,
                    "auto_named": True,
                    "scene_count": 1,
                    "detection_count": 3,
                    "samples": 1,
                    "embedding": [0.985, 0.015],
                },
            },
            "aliases": {"babe": "face_001", "kenzie": "face_100"},
        }

        matches = STEP08.plan_known_face_matches(cfg, char_map)

        self.assertIn("face_999", matches)
        self.assertEqual(matches["face_999"]["target_identity"], "Babe")
        self.assertEqual(matches["face_999"]["effective_min_consensus"], 1)

    def test_plan_known_face_matches_keeps_weak_identity_strict(self) -> None:
        cfg = {
            "character_detection": {
                "review_known_face_threshold": 0.72,
                "review_known_face_margin": 0.05,
                "review_known_face_consensus_threshold": 0.95,
                "review_known_face_min_consensus": 2,
                "review_known_face_strong_match_threshold": 1.01,
                "review_known_face_identity_weak_strength": 2.0,
                "review_known_face_identity_weak_threshold_penalty": 0.03,
                "review_known_face_identity_weak_margin_penalty": 0.02,
            }
        }
        char_map = {
            "clusters": {
                "face_001": {
                    "name": "Babe",
                    "ignored": False,
                    "auto_named": False,
                    "scene_count": 1,
                    "detection_count": 3,
                    "samples": 1,
                    "embedding": [1.0, 0.0],
                },
                "face_100": {
                    "name": "Kenzie",
                    "ignored": False,
                    "auto_named": False,
                    "scene_count": 10,
                    "detection_count": 20,
                    "samples": 5,
                    "embedding": [0.0, 1.0],
                },
                "face_999": {
                    "name": "face_999",
                    "ignored": False,
                    "auto_named": True,
                    "scene_count": 1,
                    "detection_count": 3,
                    "samples": 1,
                    "embedding": [0.985, 0.015],
                },
            },
            "aliases": {"babe": "face_001", "kenzie": "face_100"},
        }

        matches = STEP08.plan_known_face_matches(cfg, char_map)

        self.assertNotIn("face_999", matches)

    def test_assign_statist_marks_background_role(self) -> None:
        char_map = {"clusters": {}, "aliases": {}}

        payload = STEP08.assign_character_name(char_map, "face_099", "Statisten")

        self.assertEqual(payload["name"], "statist")
        self.assertTrue(payload["background_role"])
        self.assertEqual(payload["aliases"], [])

    def test_assign_priority_marks_main_character(self) -> None:
        char_map = {"clusters": {}, "aliases": {}}

        payload = STEP08.assign_character_name(char_map, "face_022", "Babe Carano", priority=True)

        self.assertEqual(payload["name"], "Babe Carano")
        self.assertTrue(payload["priority"])
        self.assertFalse(payload["background_role"])

    def test_set_priority_updates_existing_named_character(self) -> None:
        char_map = {
            "clusters": {
                "face_022": {"name": "Babe Carano", "ignored": False, "auto_named": False, "priority": False},
            },
            "aliases": {"babe carano": "face_022"},
        }

        cluster_id, payload = STEP08.set_character_priority(char_map, "Babe Carano", True)

        self.assertEqual(cluster_id, "face_022")
        self.assertTrue(payload["priority"])

    def test_assign_same_name_keeps_identity_group_and_inherits_priority(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {"name": "Babe Carano", "ignored": False, "auto_named": False, "priority": True, "aliases": ["babe carano"]},
                "face_002": {"name": "face_002", "ignored": False, "auto_named": True, "priority": False, "aliases": []},
            },
            "aliases": {"babe carano": "face_001"},
        }

        payload = STEP08.assign_character_name(char_map, "face_002", "Babe Carano")
        STEP08.rebuild_character_map_identities(char_map)

        self.assertEqual(payload["name"], "Babe Carano")
        self.assertTrue(payload["priority"])
        self.assertEqual(STEP08.identity_cluster_count(char_map, "Babe Carano"), 2)
        self.assertEqual(char_map["aliases"]["babe carano"], "face_001")
        self.assertEqual(char_map["identities"]["Babe Carano"]["cluster_count"], 2)
        self.assertEqual(char_map["clusters"]["face_002"]["identity_primary_cluster"], "face_001")

    def test_assign_same_name_cannot_accidentally_clear_existing_priority(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {"name": "Kenzie Bell", "ignored": False, "auto_named": False, "priority": True, "aliases": ["kenzie bell"]},
                "face_002": {"name": "face_002", "ignored": False, "auto_named": True, "priority": False, "aliases": []},
            },
            "aliases": {"kenzie bell": "face_001"},
            "identities": {"Kenzie Bell": {"name": "Kenzie Bell", "primary_cluster": "face_001", "cluster_ids": ["face_001"], "cluster_count": 1, "priority": True, "background_role": False}},
        }

        payload = STEP08.assign_character_name(char_map, "face_002", "Kenzie Bell", priority=False)

        self.assertTrue(payload["priority"])

    def test_prompt_priority_returns_true_for_existing_priority_identity(self) -> None:
        char_map = {
            "clusters": {},
            "aliases": {},
            "identities": {
                "Babe Carano": {
                    "name": "Babe Carano",
                    "primary_cluster": "face_001",
                    "cluster_ids": ["face_001"],
                    "cluster_count": 1,
                    "priority": True,
                    "background_role": False,
                }
            },
        }

        self.assertTrue(STEP08.prompt_priority_for_name(char_map, "Babe Carano"))

    def test_known_identity_button_options_prioritize_main_characters(self) -> None:
        char_map = {
            "clusters": {},
            "aliases": {},
            "identities": {
                "Kenzie": {"name": "Kenzie", "primary_cluster": "face_022", "cluster_ids": ["face_022", "face_133"], "cluster_count": 2, "priority": True, "background_role": False},
                "Babe": {"name": "Babe", "primary_cluster": "face_021", "cluster_ids": ["face_021", "face_132", "face_174"], "cluster_count": 3, "priority": True, "background_role": False},
                "Hudson": {"name": "Hudson", "primary_cluster": "face_020", "cluster_ids": ["face_020"], "cluster_count": 1, "priority": False, "background_role": False},
                "statist": {"name": "statist", "primary_cluster": "face_900", "cluster_ids": ["face_900"], "cluster_count": 1, "priority": False, "background_role": True},
            },
        }

        options = STEP08.known_identity_button_options(char_map, limit=10)

        self.assertEqual(options[:3], [("Babe", True, 3), ("Kenzie", True, 2), ("Hudson", False, 1)])

    def test_face_review_candidates_default_limit_zero_returns_all_open_faces(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {"name": "face_001", "ignored": False, "auto_named": True},
                "face_002": {"name": "face_002", "ignored": False, "auto_named": True},
                "face_003": {"name": "Babe", "ignored": False, "auto_named": False},
                "face_004": {"name": "noface", "ignored": True, "auto_named": False},
            }
        }

        candidates = STEP08.face_review_candidates(char_map, include_named=False, limit=0)

        self.assertEqual([cluster_id for cluster_id, _payload in candidates], ["face_001", "face_002"])

    def test_face_review_candidates_respect_session_skips(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {"name": "face_001", "ignored": False, "auto_named": True},
                "face_002": {"name": "face_002", "ignored": False, "auto_named": True},
            }
        }

        candidates = STEP08.face_review_candidates(char_map, include_named=False, limit=0, skipped_clusters={"face_001"})

        self.assertEqual([cluster_id for cluster_id, _payload in candidates], ["face_002"])

    def test_session_case_budget_remaining_decreases_with_handled_faces(self) -> None:
        self.assertEqual(STEP08.session_case_budget_remaining(20, 0), 20)
        self.assertEqual(STEP08.session_case_budget_remaining(20, 1), 19)
        self.assertEqual(STEP08.session_case_budget_remaining(20, 20), 0)
        self.assertEqual(STEP08.session_case_budget_remaining(0, 999), 0)

    def test_session_face_review_candidates_do_not_refill_to_original_limit(self) -> None:
        char_map = {
            "clusters": {
                **{
                    f"face_{index:03d}": {"name": f"face_{index:03d}", "ignored": False, "auto_named": True}
                    for index in range(1, 31)
                }
            }
        }

        candidates = STEP08.session_face_review_candidates(
            char_map,
            include_named=False,
            session_limit=20,
            handled_count=7,
            skipped_clusters=set(),
        )

        self.assertEqual(len(candidates), 13)
        self.assertEqual(candidates[0][0], "face_001")
        self.assertEqual(candidates[-1][0], "face_013")

    def test_session_face_review_candidates_stop_when_budget_is_consumed(self) -> None:
        char_map = {
            "clusters": {
                **{
                    f"face_{index:03d}": {"name": f"face_{index:03d}", "ignored": False, "auto_named": True}
                    for index in range(1, 31)
                }
            }
        }

        limited_candidates = STEP08.session_face_review_candidates(
            char_map,
            include_named=False,
            session_limit=20,
            handled_count=20,
            skipped_clusters=set(),
        )
        all_candidates = STEP08.session_face_review_candidates(
            char_map,
            include_named=False,
            session_limit=0,
            handled_count=20,
            skipped_clusters=set(),
        )

        self.assertEqual(limited_candidates, [])
        self.assertEqual(len(all_candidates), 30)

    def test_created_face_names_only_lists_real_named_figures(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {"name": "Babe", "ignored": False, "auto_named": False},
                "face_002": {"name": "Kenzie", "ignored": False, "auto_named": False},
                "face_003": {"name": "statist", "ignored": False, "auto_named": False, "background_role": True},
                "face_004": {"name": "noface", "ignored": True, "auto_named": False},
                "face_005": {"name": "face_005", "ignored": False, "auto_named": True},
                "face_006": {"name": "Babe", "ignored": False, "auto_named": False},
            }
        }

        names = STEP08.created_face_names(char_map)

        self.assertEqual(names, ["Babe", "Kenzie"])

    def test_suggested_face_role_distinguishes_main_and_background_candidates(self) -> None:
        main_payload = {"scene_count": 12, "samples": 8, "detection_count": 75, "priority": False}
        background_payload = {"scene_count": 1, "samples": 1, "detection_count": 4, "priority": False}
        support_payload = {"scene_count": 4, "samples": 3, "detection_count": 18, "priority": False}

        self.assertEqual(STEP08.suggested_face_role(main_payload), "hauptfigur-kandidat")
        self.assertEqual(STEP08.suggested_face_role(background_payload), "statist-kandidat")
        self.assertEqual(STEP08.suggested_face_role(support_payload), "nebenfigur-kandidat")

    def test_mark_auto_statist_candidates_marks_only_safe_unknowns(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {"name": "face_001", "ignored": False, "auto_named": True, "scene_count": 1, "detection_count": 4, "samples": 1},
                "face_002": {"name": "face_002", "ignored": False, "auto_named": True, "scene_count": 4, "detection_count": 18, "samples": 3},
                "face_003": {"name": "Babe", "ignored": False, "auto_named": False, "scene_count": 1, "detection_count": 4, "samples": 1},
                "face_004": {"name": "face_004", "ignored": False, "auto_named": True, "priority": True, "scene_count": 1, "detection_count": 4, "samples": 1},
            },
            "aliases": {},
        }
        thresholds = {"max_scenes": 2, "max_detections": 12, "max_samples": 3}

        marked = STEP08.mark_auto_statist_candidates(char_map, thresholds)

        self.assertEqual([item["cluster_id"] for item in marked], ["face_001"])
        self.assertEqual(char_map["clusters"]["face_001"]["name"], "statist")
        self.assertTrue(char_map["clusters"]["face_001"]["background_role"])
        self.assertEqual(char_map["clusters"]["face_002"]["name"], "face_002")
        self.assertEqual(char_map["clusters"]["face_003"]["name"], "Babe")
        self.assertTrue(char_map["clusters"]["face_004"]["priority"])

    def test_auto_statist_thresholds_can_be_overridden_from_cli(self) -> None:
        args = argparse.Namespace(statist_max_scenes=1, statist_max_detections=5, statist_max_samples=2)
        cfg = {"character_detection": {"auto_statist_max_scenes": 2, "auto_statist_max_detections": 12, "auto_statist_max_samples": 3}}

        thresholds = STEP08.auto_statist_thresholds(cfg, args)

        self.assertEqual(thresholds, {"max_scenes": 1, "max_detections": 5, "max_samples": 2})

    def test_parse_args_defaults_to_twenty_and_supports_all(self) -> None:
        with mock.patch("sys.argv", ["06_review_unknowns.py"]):
            args = STEP08.parse_args()
        self.assertEqual(args.limit, 20)
        self.assertFalse(args.all)
        self.assertFalse(args.auto_mark_statists)
        self.assertFalse(args.no_auto_mark_statists)

        with mock.patch("sys.argv", ["06_review_unknowns.py", "--all"]):
            args_all = STEP08.parse_args()
        self.assertTrue(args_all.all)

        with mock.patch("sys.argv", ["06_review_unknowns.py", "--auto-mark-statists", "--statist-max-scenes", "1"]):
            args_auto = STEP08.parse_args()
        self.assertTrue(args_auto.auto_mark_statists)
        self.assertEqual(args_auto.statist_max_scenes, 1)

    def test_set_priority_rejects_background_role(self) -> None:
        char_map = {
            "clusters": {
                "face_099": {"name": "statist", "ignored": False, "auto_named": False, "background_role": True},
            },
            "aliases": {},
        }

        with self.assertRaises(ValueError):
            STEP08.set_character_priority(char_map, "face_099", True)

    def test_episode_generation_keeps_statist_as_side_character_only(self) -> None:
        model = {
            "characters": [
                {"name": "Babe Carano", "scene_count": 10, "line_count": 20},
                {"name": "statist", "scene_count": 40, "line_count": 0},
            ],
            "speakers": {"Babe Carano": 12},
            "keywords": ["app", "chaos", "plan"],
            "dataset_files": ["demo.json"],
            "scene_count": 10,
            "speaker_samples": {"Babe Carano": ["Wir schaffen das heute ohne Chaos."]},
        }
        cfg = {
            "generation": {
                "seed": 42,
                "default_scene_count": 4,
                "min_dialogue_lines_per_scene": 4,
                "max_dialogue_lines_per_scene": 4,
            }
        }

        package, markdown = STEP07.generate_episode_package(model, cfg)

        self.assertEqual(package["focus_characters"][0], "Babe Carano")
        self.assertNotIn("statist", package["focus_characters"])
        self.assertTrue(any("statist" in scene["characters"] for scene in package["scenes"]))
        self.assertIn("statist", markdown)

    def test_episode_generation_varies_with_episode_index(self) -> None:
        model = {
            "characters": [
                {"name": "Babe Carano", "scene_count": 10, "line_count": 20},
                {"name": "Kenzie Bell", "scene_count": 9, "line_count": 18},
            ],
            "speakers": {"Babe Carano": 12, "Kenzie Bell": 11},
            "keywords": ["app", "chaos", "plan", "musik"],
            "dataset_files": ["demo.json"],
            "scene_count": 10,
            "speaker_samples": {
                "Babe Carano": ["Wir schaffen das heute ohne Chaos."],
                "Kenzie Bell": ["Ich pruefe erst den Plan und dann die App."],
            },
        }
        cfg = {
            "generation": {
                "seed": 42,
                "default_scene_count": 4,
                "min_dialogue_lines_per_scene": 4,
                "max_dialogue_lines_per_scene": 4,
            }
        }

        package_a, markdown_a = STEP07.generate_episode_package(model, cfg, episode_index=1)
        package_b, markdown_b = STEP07.generate_episode_package(model, cfg, episode_index=2)

        self.assertNotEqual(package_a["keywords"], package_b["keywords"])
        self.assertNotEqual(markdown_a, markdown_b)

    def test_episode_generation_prefers_prioritized_characters(self) -> None:
        model = {
            "characters": [
                {"name": "Nebenfigur", "scene_count": 50, "line_count": 10, "priority": False},
                {"name": "Babe Carano", "scene_count": 5, "line_count": 5, "priority": True},
                {"name": "Kenzie Bell", "scene_count": 4, "line_count": 4, "priority": True},
            ],
            "speakers": {},
            "keywords": ["app", "chaos"],
            "dataset_files": ["demo.json"],
            "scene_count": 5,
            "speaker_samples": {},
        }
        cfg = {"generation": {"seed": 42, "default_scene_count": 2}}

        package, _markdown = STEP07.generate_episode_package(model, cfg, episode_index=2)

        self.assertEqual(package["focus_characters"][:2], ["Babe Carano", "Kenzie Bell"])

    def test_build_series_model_counts_multiple_face_clusters_per_character(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {"name": "Babe Carano", "ignored": False, "priority": True, "scene_count": 4, "detection_count": 20},
                "face_002": {"name": "Babe Carano", "ignored": False, "priority": False, "scene_count": 3, "detection_count": 11},
                "face_003": {"name": "Kenzie Bell", "ignored": False, "priority": True, "scene_count": 5, "detection_count": 25},
            },
            "aliases": {"babe carano": "face_001", "kenzie bell": "face_003"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "demo_dataset.json"
            dataset_path.write_text("[]", encoding="utf-8")
            cfg = {"paths": {"linked_segments": str(Path(tmpdir) / "linked"), "scene_clips": str(Path(tmpdir) / "scene_clips")}}
            model = STEP07.build_series_model([dataset_path], cfg, char_map)

        babe = next(row for row in model["characters"] if row["name"] == "Babe Carano")
        self.assertEqual(babe["face_cluster_count"], 2)

    def test_episode_generation_assigns_real_title_metadata(self) -> None:
        model = {
            "characters": [
                {"name": "Babe Carano", "scene_count": 10, "line_count": 20, "priority": True},
                {"name": "Kenzie Bell", "scene_count": 9, "line_count": 18, "priority": True},
            ],
            "speakers": {"Babe Carano": 12, "Kenzie Bell": 11},
            "keywords": ["baumhaus", "spiel", "chaos"],
            "dataset_files": ["demo.json"],
            "scene_count": 10,
            "speaker_samples": {},
        }
        cfg = {"generation": {"seed": 42, "default_scene_count": 4}}

        package, markdown = STEP07.generate_episode_package(model, cfg, episode_index=5)

        self.assertEqual(package["episode_label"], "Folge 05")
        self.assertIn("episode_title", package)
        self.assertTrue(package["episode_title"])
        self.assertIn("Folge 05:", package["display_title"])
        self.assertEqual(package["generation_mode"], "synthetic_preview")
        self.assertIn("# Folge 05:", markdown)

    def test_episode_title_filters_weak_keywords_and_uses_clean_fallback(self) -> None:
        title = STEP07.build_episode_title(
            ["Babe Carano", "Kenzie Bell"],
            ["weiß", "wieder", "soll", "meinem"],
            random.Random(42),
            12,
        )

        self.assertNotIn("Weiß", title)
        self.assertNotIn("Wieder", title)
        self.assertNotIn("Soll", title)
        self.assertNotIn("Meinem", title)
        self.assertTrue(title)

    def test_episode_title_prefers_original_dialogue_anchor(self) -> None:
        model = {
            "speaker_line_library": {
                "Kenzie Bell": [
                    {
                        "text": "Der Rollerteller! Der ist wie ein ganz normaler Teller, aber es sind Räder unten angebracht.",
                    }
                ]
            }
        }

        title = STEP07.build_episode_title(
            ["Babe Carano", "Kenzie Bell"],
            ["teller", "spaghetti"],
            random.Random(42),
            12,
            model=model,
        )

        self.assertIn("Rollerteller", title)
        self.assertNotEqual(title, "Die Teller")

    def test_episode_generation_uses_original_dialogue_anchor_for_title(self) -> None:
        model = {
            "characters": [
                {"name": "Babe Carano", "scene_count": 10, "line_count": 20, "priority": True},
                {"name": "Kenzie Bell", "scene_count": 9, "line_count": 18, "priority": True},
            ],
            "speakers": {"Babe Carano": 12, "Kenzie Bell": 11},
            "keywords": ["teller", "geld", "chaos"],
            "dataset_files": ["demo.json"],
            "scene_count": 10,
            "speaker_samples": {},
            "speaker_line_library": {
                "Kenzie Bell": [
                    {
                        "text": "Der Rollerteller! Der ist wie ein ganz normaler Teller, aber es sind Räder unten angebracht.",
                    }
                ]
            },
        }
        cfg = {"generation": {"seed": 42, "default_scene_count": 4}}

        package, _markdown = STEP07.generate_episode_package(model, cfg, episode_index=13)

        self.assertIn("Rollerteller", package["episode_title"])

    def test_episode_runtime_matches_source_episode_average(self) -> None:
        model = {
            "source_episode_durations": {
                "ep1": 21 * 60,
                "ep2": 23 * 60,
            }
        }
        cfg = {"generation": {"match_source_episode_runtime": True, "target_episode_minutes_fallback": 22.0}}

        target_seconds = STEP07.select_target_runtime_seconds(model, cfg)

        self.assertEqual(target_seconds, 22 * 60)

    def test_generate_step_uses_existing_trained_model(self) -> None:
        model = {
            "characters": [
                {"name": "Babe", "scene_count": 20, "line_count": 10, "priority": True},
                {"name": "Kenzie", "scene_count": 18, "line_count": 9, "priority": True},
            ],
            "speakers": {},
            "keywords": ["spiel", "chaos"],
            "dataset_files": ["demo.json"],
            "scene_count": 20,
            "speaker_samples": {},
            "source_episode_durations": {"ep1": 22 * 60},
            "average_segment_duration_seconds": 2.7,
        }
        cfg = {"generation": {"match_source_episode_runtime": True, "target_episode_minutes_fallback": 22.0}}

        package, markdown = STEP07.generate_episode_package(model, cfg, episode_index=9)

        self.assertEqual(package["episode_label"], "Folge 09")
        self.assertEqual(package["generation_mode"], "synthetic_preview")
        self.assertIn("Folge 09:", markdown)

    def test_create_final_render_uses_concat_list_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            segment_a = temp_root / "segment_a.mp4"
            segment_b = temp_root / "segment_b.mp4"
            output_mp4 = temp_root / "final.mp4"
            segment_a.write_text("a", encoding="utf-8")
            segment_b.write_text("b", encoding="utf-8")
            captured: dict[str, object] = {}

            original_runner = STEP10.run_ffmpeg_with_codec_fallback

            def fake_runner(command_factory, video_codec, output_path):
                captured["codec"] = video_codec
                command = command_factory(video_codec)
                captured["command"] = command
                concat_path = Path(command[command.index("-i") + 1])
                captured["concat_path"] = concat_path
                captured["concat_text"] = concat_path.read_text(encoding="utf-8")
                output_path.write_text("rendered", encoding="utf-8")
                return video_codec

            STEP10.run_ffmpeg_with_codec_fallback = fake_runner
            try:
                codec = STEP10.create_final_render(Path("ffmpeg"), [segment_a, segment_b], output_mp4, "libx264")
            finally:
                STEP10.run_ffmpeg_with_codec_fallback = original_runner

            self.assertEqual(codec, "libx264")
            self.assertIn("-f", captured["command"])
            self.assertIn("concat", captured["command"])
            self.assertIn("segment_a.mp4", str(captured["concat_text"]))
            self.assertIn("segment_b.mp4", str(captured["concat_text"]))
            self.assertFalse(Path(captured["concat_path"]).exists())

    def test_resolve_system_voice_id_prefers_german_voice(self) -> None:
        voices = [
            {"id": "en-zira", "name": "Zira", "languages": ["en-US"], "is_german": False},
            {"id": "de-hedda", "name": "Hedda", "languages": ["de-DE"], "is_german": True},
        ]

        selected = STEP10.resolve_system_voice_id("en-zira", voices, require_german=True)

        self.assertEqual(selected, "de-hedda")

    def test_build_dialogue_reuses_speaker_samples_when_available(self) -> None:
        model = {
            "speaker_samples": {
                "Babe": ["Sag bitte, dass die Lösung nicht noch mehr Improvisation braucht."],
                "Kenzie": ["Ich glaube, ich sehe endlich, warum song die ganze Zeit blockiert war."],
            }
        }
        cfg = {"generation": {"min_dialogue_lines_per_scene": 4, "max_dialogue_lines_per_scene": 4}}

        dialogue, dialogue_sources = STEP07.build_dialogue(
            "Wendepunkt",
            ["Babe", "Kenzie"],
            model,
            random.Random(42),
            cfg,
            "song",
        )

        joined = "\n".join(dialogue)
        self.assertIn("Ich glaube, ich sehe endlich, warum song die ganze Zeit blockiert war.", joined)
        self.assertIn("Sag bitte, dass die Lösung nicht noch mehr Improvisation braucht.", joined)
        self.assertEqual(len(dialogue_sources), 4)

    def test_build_dialogue_prefers_original_line_sources_when_enabled(self) -> None:
        model = {
            "speaker_samples": {},
            "speaker_line_library": {
                "Babe Carano": [
                    {
                        "episode_id": "ep1",
                        "scene_id": "scene_0001",
                        "segment_id": "scene_0001_seg_001",
                        "text": "Wir schaffen das heute ohne Chaos.",
                        "start": 0.0,
                        "end": 1.2,
                        "audio_file": "demo.wav",
                        "video_file": "scene_0001.mp4",
                        "keywords": ["chaos", "plan"],
                    }
                ],
                "Kenzie Bell": [
                    {
                        "episode_id": "ep1",
                        "scene_id": "scene_0002",
                        "segment_id": "scene_0002_seg_002",
                        "text": "Ich pruefe erst den Plan und dann die App.",
                        "start": 0.0,
                        "end": 1.4,
                        "audio_file": "demo2.wav",
                        "video_file": "scene_0002.mp4",
                        "keywords": ["plan", "app"],
                    }
                ],
            },
        }
        cfg = {
            "generation": {
                "min_dialogue_lines_per_scene": 4,
                "max_dialogue_lines_per_scene": 4,
                "prefer_original_dialogue_remix": True,
            }
        }

        dialogue, dialogue_sources = STEP07.build_dialogue(
            "Plan",
            ["Babe Carano", "Kenzie Bell"],
            model,
            random.Random(42),
            cfg,
            "plan",
        )

        self.assertTrue(any(source.get("type") == "original_line" for source in dialogue_sources))
        self.assertIn("Wir schaffen das heute ohne Chaos.", "\n".join(dialogue))

    def test_scene_dialogue_source_returns_matching_entry(self) -> None:
        scene = {
            "dialogue_sources": [
                {"type": "original_line", "segment_id": "scene_0001_seg_001"},
                {},
            ]
        }

        self.assertEqual(STEP10.scene_dialogue_source(scene, 0).get("segment_id"), "scene_0001_seg_001")
        self.assertEqual(STEP10.scene_dialogue_source(scene, 1), {})
        self.assertEqual(STEP10.scene_dialogue_source(scene, 9), {})

    def test_resolve_segment_character_name_prefers_visible_named_character(self) -> None:
        row = {"characters_visible": ["Babe"]}
        segment = {"speaker_name": "speaker_001", "visible_character_names": ["Babe"]}

        resolved = STEP07.resolve_segment_character_name(segment, row)

        self.assertEqual(resolved, "Babe")

    def test_text_similarity_score_prefers_close_match(self) -> None:
        strong_score, strong_overlap = STEP10.text_similarity_score(
            "Gut, nächste Frage.",
            "Gut, nächste Frage.",
        )
        weak_score, weak_overlap = STEP10.text_similarity_score(
            "Gut, nächste Frage.",
            "Wir gehen jetzt nach Hause.",
        )

        self.assertGreater(strong_score, weak_score)
        self.assertGreater(strong_overlap, weak_overlap)

    def test_select_retrieval_segment_uses_best_matching_original_line(self) -> None:
        library = {
            "Babe Carano": [
                {
                    "segment_id": "scene_0010_seg_001",
                    "text": "Gut, nächste Frage.",
                    "audio_path": "demo_a.wav",
                    "scene_clip_path": "scene_0010.mp4",
                    "start": 0.0,
                    "end": 1.0,
                },
                {
                    "segment_id": "scene_0042_seg_003",
                    "text": "Wir sollten sofort loslaufen.",
                    "audio_path": "demo_b.wav",
                    "scene_clip_path": "scene_0042.mp4",
                    "start": 1.0,
                    "end": 2.0,
                },
            ]
        }
        cfg = {
            "enable_original_line_reuse": True,
            "original_line_similarity_threshold": 0.74,
            "original_line_min_token_overlap": 0.34,
        }

        selected = STEP10.select_retrieval_segment("Babe Carano", "Gut, nächste Frage.", library, cfg)

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["segment_id"], "scene_0010_seg_001")
        self.assertGreaterEqual(selected["match_score"], 0.74)

    def test_select_retrieval_segment_rejects_weak_match(self) -> None:
        library = {
            "Babe Carano": [
                {
                    "segment_id": "scene_0042_seg_003",
                    "text": "Wir sollten sofort loslaufen.",
                    "audio_path": "demo_b.wav",
                    "scene_clip_path": "scene_0042.mp4",
                    "start": 1.0,
                    "end": 2.0,
                },
            ]
        }
        cfg = {
            "enable_original_line_reuse": True,
            "original_line_similarity_threshold": 0.74,
            "original_line_min_token_overlap": 0.34,
        }

        selected = STEP10.select_retrieval_segment("Babe Carano", "Gut, nächste Frage.", library, cfg)

        self.assertIsNone(selected)

    def test_build_original_line_library_uses_linked_speaker_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            linked_root = root / "data" / "processed" / "linked_segments"
            transcript_root = root / "data" / "processed" / "speaker_transcripts"
            linked_root.mkdir(parents=True, exist_ok=True)
            transcript_root.mkdir(parents=True, exist_ok=True)
            (linked_root / "demo_episode_linked_segments.json").write_text(
                json.dumps(
                    [
                        {
                            "segment_id": "scene_0001_seg_001",
                            "scene_id": "scene_0001",
                            "speaker_name": "Babe Carano",
                            "text": "Gut, nächste Frage.",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (transcript_root / "demo_episode_segments.json").write_text(
                json.dumps(
                    [
                        {
                            "segment_id": "scene_0001_seg_001",
                            "scene_id": "scene_0001",
                            "audio_file": str(root / "audio.wav"),
                            "start": 0.0,
                            "end": 1.0,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            cfg = {
                "paths": {
                    "linked_segments": "data/processed/linked_segments",
                    "speaker_transcripts": "data/processed/speaker_transcripts",
                    "scene_clips": "data/processed/scene_clips",
                }
            }

            with mock.patch("pipeline_common.PROJECT_ROOT", root):
                library = STEP10.build_original_line_library(cfg)

            self.assertIn("Babe Carano", library)
            entry = library["Babe Carano"][0]
            self.assertEqual(entry["segment_id"], "scene_0001_seg_001")
            self.assertEqual(entry["audio_path"], str(root / "audio.wav"))
            self.assertTrue(entry["scene_clip_path"].endswith("demo_episode\\scene_0001.mp4") or entry["scene_clip_path"].endswith("demo_episode/scene_0001.mp4"))

    def test_build_scene_voice_plan_scales_to_scene_duration_and_reuses_sources(self) -> None:
        scene = {
            "scene_id": "scene_0004",
            "dialogue_lines": [
                "Babe Carano: Gut, nächste Frage.",
                "Kenzie Bell: Wir gehen weiter.",
            ],
            "dialogue_sources": [
                {
                    "type": "original_line",
                    "speaker": "Babe Carano",
                    "segment_id": "scene_0010_seg_001",
                    "text": "Gut, nächste Frage.",
                    "audio_file": "demo_a.wav",
                    "video_file": "scene_0010.mp4",
                    "start": 0.0,
                    "end": 1.0,
                },
                {},
            ],
        }
        library = {
            "Kenzie Bell": [
                {
                    "segment_id": "scene_0042_seg_003",
                    "text": "Wir gehen weiter.",
                    "audio_path": "demo_b.wav",
                    "scene_clip_path": "scene_0042.mp4",
                    "start": 1.0,
                    "end": 2.0,
                }
            ]
        }
        voice_lookup = {
            "Babe Carano": {
                "cluster_id": "speaker_001",
                "name": "Babe Carano",
                "linked_face_cluster": "face_001",
                "auto_named": False,
            }
        }
        cloning_cfg = {
            "enable_original_line_reuse": True,
            "original_line_similarity_threshold": 0.74,
            "original_line_min_token_overlap": 0.34,
        }
        render_cfg = {"voice_rate": 175, "audio_pad_seconds": 0.35}

        plan = STEP10.build_scene_voice_plan(scene, 2.2, 5.0, library, voice_lookup, cloning_cfg, render_cfg)

        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0]["start_seconds"], 5.0)
        self.assertEqual(plan[0]["audio_strategy"], "reuse_original_segment")
        self.assertEqual(plan[0]["retrieval_segment"]["segment_id"], "scene_0010_seg_001")
        self.assertEqual(plan[0]["voice_profile"]["cluster_id"], "speaker_001")
        self.assertEqual(plan[1]["audio_strategy"], "reuse_original_segment")
        self.assertLessEqual(plan[-1]["end_seconds"], 7.2)

    def test_render_subtitle_preview_srt_formats_dialogue_entries(self) -> None:
        srt = STEP10.render_subtitle_preview_srt(
            [
                {"speaker_name": "Babe", "text": "Hallo zusammen.", "start_seconds": 0.0, "end_seconds": 1.2},
                {"speaker_name": "Kenzie", "text": "Dann legen wir los.", "start_seconds": 1.2, "end_seconds": 2.7},
            ]
        )

        self.assertIn("00:00:00,000 --> 00:00:01,200", srt)
        self.assertIn("Babe: Hallo zusammen.", srt)
        self.assertIn("Kenzie: Dann legen wir los.", srt)

    def test_dialogue_audio_filter_adds_loudness_and_soft_fades(self) -> None:
        filter_graph = STEP10.dialogue_audio_filter(1.2)

        self.assertIn("loudnorm=I=-18:TP=-2:LRA=11", filter_graph)
        self.assertIn("afade=t=in:st=0", filter_graph)
        self.assertIn("afade=t=out", filter_graph)
        self.assertIn("atrim=0:1.200", filter_graph)

    def test_build_audio_segment_plan_inserts_gaps_and_trailing_silence(self) -> None:
        plan = STEP10.build_audio_segment_plan(
            [
                {"line_index": 0, "start_seconds": 1.0, "end_seconds": 2.0},
                {"line_index": 1, "start_seconds": 2.5, "end_seconds": 3.0},
            ],
            4.0,
        )

        self.assertEqual(
            plan,
            [
                {"kind": "silence", "duration_seconds": 1.0},
                {"kind": "line", "line_index": 0, "duration_seconds": 1.0},
                {"kind": "silence", "duration_seconds": 0.5},
                {"kind": "line", "line_index": 1, "duration_seconds": 0.5},
                {"kind": "silence", "duration_seconds": 1.0},
            ],
        )

    def test_build_audio_segment_plan_returns_full_silence_when_no_lines_exist(self) -> None:
        plan = STEP10.build_audio_segment_plan([], 3.5)

        self.assertEqual(plan, [{"kind": "silence", "duration_seconds": 3.5}])

    def test_materialize_original_segment_audio_falls_back_to_scene_clip_extract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_clip_path = root / "scene_clip.mp4"
            output_path = root / "line_0000.wav"
            scene_clip_path.write_bytes(b"video")

            with mock.patch.object(STEPRENDER, "extract_clip_audio") as extract_clip:
                backend = STEPRENDER.materialize_original_segment_audio(
                    Path("ffmpeg"),
                    {"scene_clip_path": str(scene_clip_path), "start": 1.25},
                    1.8,
                    output_path,
                    22050,
                )

        self.assertEqual(backend, "scene_clip_extract")
        extract_clip.assert_called_once()

    def test_first_existing_production_scene_video_prefers_lipsync_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_root = root / "generation" / "final_episode_packages" / "episode_011"
            lipsync_path = package_root / "lipsync" / "scene_0001" / "scene_0001_lipsync.mp4"
            video_path = package_root / "videos" / "scene_0001" / "scene_0001.mp4"
            lipsync_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            lipsync_path.write_bytes(b"lip")
            video_path.write_bytes(b"vid")

            match = STEPRENDER.first_existing_production_scene_video(package_root, "scene_0001")

        self.assertIsNotNone(match)
        self.assertEqual(match[0], "generated_lipsync_video")
        self.assertTrue(str(match[1]).endswith("scene_0001_lipsync.mp4"))

    def test_first_existing_scene_video_source_uses_storyboard_backend_clip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_root = root / "generation" / "final_episode_packages" / "episode_011"
            assets_root = root / "generation" / "storyboard_assets" / "episode_011"
            clip_path = assets_root / "scene_0001" / "clip.mp4"
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            clip_path.write_bytes(b"clip")

            match = STEPRENDER.first_existing_scene_video_source(package_root, assets_root, "scene_0001")

        self.assertIsNotNone(match)
        self.assertEqual(match[0], "storyboard_backend_scene_video")
        self.assertTrue(str(match[1]).endswith("clip.mp4"))

    def test_materialize_scene_backend_frame_writes_scene_pack_without_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            assets_root = root / "assets" / "episode_011"
            assets_root.mkdir(parents=True, exist_ok=True)
            source_path = assets_root / "scene_0001.png"
            Image.new("RGB", (320, 180), (64, 96, 128)).save(source_path)
            payload = {
                "scene_id": "scene_0001",
                "positive_prompt": "series-style shot",
                "camera_plan": {"lens": "medium shot", "movement": "push in"},
            }

            manifest = STEP16BACKEND.materialize_scene_backend_frame(
                assets_root,
                payload,
                320,
                180,
                24,
                23,
                None,
                False,
            )

            self.assertEqual(manifest["backend_mode"], "materialized_local_backend_scene_pack")
            self.assertTrue(Path(manifest["output_path"]).exists())
            self.assertTrue(Path(manifest["poster_path"]).exists())
            self.assertGreaterEqual(len(manifest["alternate_frames"]), 2)
            self.assertEqual(manifest["clip_status"], "not_requested")

    def test_choose_render_mode_prefers_generated_episode_labels(self) -> None:
        self.assertEqual(STEPRENDER.choose_render_mode(4, 4, True), "fully_generated_scene_video_episode")
        self.assertEqual(STEPRENDER.choose_render_mode(4, 2, True), "hybrid_generated_scene_video_episode")
        self.assertEqual(STEPRENDER.choose_render_mode(4, 0, True), "voiced_storyboard_episode")
        self.assertEqual(STEPRENDER.choose_render_mode(4, 2, False), "silent_hybrid_generated_scene_video_fallback")

    def test_materialize_scene_master_clips_muxes_scene_audio_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            package_root = root / "generation" / "final_episode_packages" / "episode_011"
            final_clip_path = root / "tmp" / "scene_0001.mp4"
            scene_audio_path = package_root / "audio" / "scene_0001" / "scene_0001_dialogue.wav"
            final_clip_path.parent.mkdir(parents=True, exist_ok=True)
            scene_audio_path.parent.mkdir(parents=True, exist_ok=True)
            final_clip_path.write_bytes(b"video")
            scene_audio_path.write_bytes(b"audio")

            def _mux(_ffmpeg, _video_path, _audio_path, output_path):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"muxed")

            with mock.patch.object(STEPRENDER, "mux_episode_audio", side_effect=_mux) as mux_audio:
                outputs = STEPRENDER.materialize_scene_master_clips(
                    Path("ffmpeg"),
                    [{"scene_id": "scene_0001", "final_clip_path": str(final_clip_path)}],
                    {"scene_0001": str(scene_audio_path)},
                    package_root,
                )
                self.assertIn("scene_0001", outputs)
                self.assertTrue(Path(outputs["scene_0001"]).exists())
                mux_audio.assert_called_once()

    def test_render_output_ready_rejects_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(STEPRENDER.render_output_ready(Path(tmpdir)))

    def test_encode_full_generated_episode_master_muxes_dialogue_audio_when_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            concat_path = root / "clips.txt"
            output_path = root / "master" / "episode_011_full_generated_episode.mp4"
            dialogue_audio_path = root / "audio" / "episode_011_dialogue.wav"
            concat_path.write_text("file 'clip.mp4'\n", encoding="utf-8")
            dialogue_audio_path.parent.mkdir(parents=True, exist_ok=True)
            dialogue_audio_path.write_bytes(b"audio")

            def _encode(_ffmpeg, _concat_path, video_only_path, _crf):
                video_only_path.parent.mkdir(parents=True, exist_ok=True)
                video_only_path.write_bytes(b"video")

            def _mux(_ffmpeg, _video_path, _audio_path, target_path):
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(b"muxed")

            with mock.patch.object(STEPRENDER, "encode_clip_sequence", side_effect=_encode) as encode_sequence, mock.patch.object(
                STEPRENDER, "mux_episode_audio", side_effect=_mux
            ) as mux_audio:
                meta = STEPRENDER.encode_full_generated_episode_master(
                    Path("ffmpeg"),
                    concat_path,
                    output_path,
                    dialogue_audio_path,
                    20,
                )

            self.assertTrue(meta["audio_muxed"])
            self.assertTrue(output_path.exists())
            self.assertTrue(str(meta["video_only_path"]).endswith("_video_only.mp4"))
            encode_sequence.assert_called_once()
            mux_audio.assert_called_once()

    def test_render_episode_audio_track_mixes_original_segments_and_tts_materializations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            temp_root = root / "tmp"
            output_path = root / "renders" / "episode_011_dialogue_audio.wav"
            package_root = root / "generation" / "final_episode_packages" / "episode_011"
            original_audio_path = root / "audio" / "original_babe.wav"
            original_audio_path.parent.mkdir(parents=True, exist_ok=True)
            original_audio_path.write_bytes(b"audio")
            render_cfg = {"audio_sample_rate": 16000}
            voice_plan_lines = [
                {
                    "line_index": 0,
                    "speaker_name": "Babe Carano",
                    "text": "Los geht's.",
                    "estimated_duration_seconds": 1.1,
                    "start_seconds": 0.0,
                    "end_seconds": 1.1,
                    "retrieval_segment": {"audio_path": str(original_audio_path)},
                },
                {
                    "line_index": 1,
                    "speaker_name": "Kenzie Bell",
                    "text": "Ich bin bereit.",
                    "estimated_duration_seconds": 1.0,
                    "start_seconds": 1.2,
                    "end_seconds": 2.2,
                    "retrieval_segment": {},
                },
            ]
            voice_plan_scenes = [
                {
                    "scene_id": "scene_0001",
                    "scene_start_seconds": 0.0,
                    "scene_end_seconds": 2.4,
                    "duration_seconds": 2.4,
                    "lines": voice_plan_lines,
                }
            ]

            def _write_bytes(path: Path, payload: bytes = b"data") -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)

            def _materialize_original_segment_audio(_ffmpeg, retrieval_segment, _duration_seconds, target_path, _sample_rate):
                if retrieval_segment.get("audio_path"):
                    _write_bytes(target_path, b"orig")
                    return "original_segment_audio"
                return ""

            def _synthesize_voice_lines(_audio_root, requested_lines, _render_cfg):
                synthesized_map = {}
                for line in requested_lines:
                    raw_path = temp_root / "audio" / f"line_{int(line.get('line_index', 0) or 0):04d}_raw.wav"
                    _write_bytes(raw_path, b"tts")
                    synthesized_map[int(line.get("line_index", 0) or 0)] = raw_path
                return synthesized_map, {"backend": "pyttsx3", "voice_id": "voice_de"}

            def _normalize_line_audio(_ffmpeg, input_path, _duration_seconds, output_path_arg, _sample_rate):
                _write_bytes(output_path_arg, input_path.read_bytes() if input_path.exists() else b"norm")

            def _create_silence_audio(_ffmpeg, _duration_seconds, output_path_arg, _sample_rate):
                _write_bytes(output_path_arg, b"silence")

            def _concat_audio_segments(_ffmpeg, _segment_paths, output_path_arg):
                _write_bytes(output_path_arg, b"concat")

            def _materialize_scene_dialogue_tracks(_ffmpeg, _voice_plan_scenes, _line_output_map, _sample_rate, package_root_arg):
                scene_audio_path = package_root_arg / "audio" / "scene_0001" / "scene_0001_dialogue.wav"
                _write_bytes(scene_audio_path, b"scene")
                return {"scene_0001": str(scene_audio_path)}

            with (
                mock.patch.object(STEPRENDER, "materialize_original_segment_audio", side_effect=_materialize_original_segment_audio),
                mock.patch.object(STEPRENDER, "synthesize_voice_lines", side_effect=_synthesize_voice_lines),
                mock.patch.object(STEPRENDER, "normalize_line_audio", side_effect=_normalize_line_audio),
                mock.patch.object(STEPRENDER, "create_silence_audio", side_effect=_create_silence_audio),
                mock.patch.object(STEPRENDER, "concat_audio_segments", side_effect=_concat_audio_segments),
                mock.patch.object(STEPRENDER, "materialize_scene_dialogue_tracks", side_effect=_materialize_scene_dialogue_tracks),
            ):
                meta = STEPRENDER.render_episode_audio_track(
                    Path("ffmpeg"),
                    voice_plan_lines,
                    2.4,
                    render_cfg,
                    temp_root,
                    output_path,
                    package_root,
                    voice_plan_scenes,
                )
                self.assertEqual(meta["audio_backend"], "mixed_original_segment_and_pyttsx3")
                self.assertEqual(meta["reused_original_lines"], 1)
                self.assertEqual(meta["synthesized_lines"], 1)
                self.assertEqual(meta["scene_dialogue_outputs"]["scene_0001"].split("\\")[-1].split("/")[-1], "scene_0001_dialogue.wav")
                self.assertTrue((package_root / "audio" / "babe_carano" / "line_0000.wav").exists())
                self.assertTrue((package_root / "audio" / "kenzie_bell" / "line_0001.wav").exists())
                self.assertTrue(output_path.exists())

    def test_build_episode_production_package_payload_targets_full_generated_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = {
                "paths": {
                    "final_episode_packages": str(root / "generation" / "final_episode_packages"),
                    "voice_samples": str(root / "characters" / "voice_samples"),
                    "voice_models": str(root / "characters" / "voice_models"),
                }
            }
            package_root = root / "generation" / "final_episode_packages" / "episode_009"
            shotlist = {
                "episode_id": "episode_009",
                "display_title": "Episode 09",
                "episode_title": "Full Generation Test",
                "storyboard_request_dir": str(root / "generation" / "storyboard_requests" / "episode_009"),
                "scenes": [
                    {
                        "scene_id": "scene_0001",
                        "title": "Cold Open",
                        "summary": "Babe and Kenzie test a new recording rig.",
                        "location": "Studio",
                        "mood": "focused",
                        "characters": ["Babe Carano", "Kenzie Bell"],
                        "generation_plan": {
                            "positive_prompt": "cinematic studio lighting, game studio, two teenage founders, dynamic blocking",
                            "negative_prompt": "blurry, duplicate face, wrong anatomy",
                            "batch_prompt_line": "Cold Open | Babe, Kenzie",
                            "reference_slots": [{"slot": "subject_1", "character": "Babe Carano", "images": ["C:/refs/babe.png"]}],
                            "camera_plan": [{"camera": "medium shot", "focus": "Babe and Kenzie"}],
                            "control_hints": [{"hint": "motion", "value": "subtle camera push"}],
                            "continuity": {"previous_scene_id": "scene_0000"},
                        },
                    }
                ],
            }
            manifest = {
                "shotlist_path": str(root / "generation" / "shotlists" / "episode_009.json"),
                "render_manifest_path": str(root / "generation" / "renders" / "final" / "episode_009_render_manifest.json"),
                "voice_plan": str(root / "generation" / "renders" / "final" / "episode_009_voice_plan.json"),
                "draft_render": str(root / "generation" / "renders" / "drafts" / "episode_009.mp4"),
                "final_render": str(root / "generation" / "renders" / "final" / "episode_009.mp4"),
                "dialogue_audio": str(root / "generation" / "renders" / "final" / "episode_009_dialogue_audio.wav"),
                "subtitle_preview": str(root / "generation" / "renders" / "final" / "episode_009_dialogue_preview.srt"),
                "scenes": [
                    {
                        "scene_id": "scene_0001",
                        "duration_seconds": 4.2,
                        "frame_path": str(root / "tmp" / "scene_0001.png"),
                        "asset_source_type": "backend_frame",
                        "asset_source_path": str(root / "generation" / "storyboard_assets" / "episode_009" / "scene_0001" / "frame.png"),
                        "video_source_type": "generated_lipsync_video",
                        "video_source_path": str(root / "generation" / "final_episode_packages" / "episode_009" / "lipsync" / "scene_0001" / "scene_0001_lipsync.mp4"),
                        "final_clip_path": str(root / "generation" / "renders" / "tmp" / "episode_009" / "clips" / "scene_0001.mp4"),
                        "scene_dialogue_audio": str(root / "generation" / "final_episode_packages" / "episode_009" / "audio" / "scene_0001" / "scene_0001_dialogue.wav"),
                        "scene_master_clip": str(root / "generation" / "final_episode_packages" / "episode_009" / "master" / "scenes" / "scene_0001_master.mp4"),
                    }
                ],
            }
            voice_plan_payload = {
                "scenes": [
                    {
                        "scene_id": "scene_0001",
                        "duration_seconds": 4.2,
                        "lines": [
                            {
                                "line_index": 0,
                                "speaker_name": "Babe Carano",
                                "text": "Okay, start the capture.",
                                "start_seconds": 0.0,
                                "end_seconds": 1.8,
                                "estimated_duration_seconds": 1.8,
                                "voice_profile": {
                                    "cluster_id": "speaker_001",
                                    "linked_face_cluster": "face_001",
                                    "auto_named": False,
                                    "voice_model_path": "C:/voices/babe_voice_model.json",
                                    "dominant_language": "en",
                                    "language_counts": {"en": 3},
                                    "reference_audio": "C:/voices/babe_reference.wav",
                                },
                                "retrieval_segment": {
                                    "audio_path": "C:/audio/original_babe.wav",
                                    "scene_clip_path": "C:/video/original_scene.mp4",
                                    "segment_id": "seg_001",
                                    "match_score": 0.93,
                                    "language": "en",
                                },
                            }
                        ],
                    }
                ]
            }

            payload = STEPRENDER.build_episode_production_package_payload(
                cfg, "episode_009", shotlist, manifest, voice_plan_payload, package_root
            )

        self.assertEqual(payload["production_goal"], "fully_generated_episode_with_new_storyboard_new_frames_original_voices_and_lip_sync")
        self.assertTrue(payload["backend_requirements"]["image_generation"])
        self.assertTrue(payload["backend_requirements"]["video_generation"])
        self.assertTrue(payload["backend_requirements"]["voice_clone"])
        self.assertTrue(payload["backend_requirements"]["lip_sync"])
        self.assertTrue(payload["target_master_outputs"]["final_master_episode"].endswith("episode_009_full_generated_episode.mp4"))
        self.assertTrue(payload["target_master_outputs"]["scene_master_root"].replace("\\", "/").endswith("master/scenes"))
        self.assertEqual(payload["completion_status"]["generated_scene_video_count"], 1)
        self.assertEqual(payload["completion_status"]["scene_dialogue_audio_count"], 1)
        self.assertEqual(payload["completion_status"]["scene_master_clip_count"], 1)
        self.assertTrue(payload["completion_status"]["all_scene_videos_ready"])
        self.assertTrue(payload["completion_status"]["all_scene_master_clips_ready"])
        self.assertEqual(payload["scenes"][0]["image_generation"]["target_outputs"]["primary_frame"].split("\\")[-1].split("/")[-1], "frame_0001.png")
        self.assertEqual(payload["scenes"][0]["voice_clone"]["lines"][0]["speaker_name"], "Babe Carano")
        self.assertEqual(payload["scenes"][0]["voice_clone"]["lines"][0]["language"], "en")
        self.assertEqual(payload["scenes"][0]["voice_clone"]["lines"][0]["voice_profile"]["voice_model_path"], "C:/voices/babe_voice_model.json")
        self.assertEqual(payload["scenes"][0]["lip_sync"]["target_outputs"]["lipsync_video"].split("\\")[-1].split("/")[-1], "scene_0001_lipsync.mp4")
        self.assertEqual(payload["scenes"][0]["mastering"]["target_outputs"]["scene_master_clip"].split("\\")[-1].split("/")[-1], "scene_0001_master.mp4")
        self.assertTrue(payload["scenes"][0]["current_generated_outputs"]["has_generated_scene_video"])
        self.assertTrue(payload["scenes"][0]["current_generated_outputs"]["has_scene_dialogue_audio"])
        self.assertTrue(payload["scenes"][0]["current_generated_outputs"]["has_scene_master_clip"])

    def test_write_episode_production_package_writes_master_and_scene_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = {
                "paths": {
                    "final_episode_packages": str(root / "generation" / "final_episode_packages"),
                    "voice_samples": str(root / "characters" / "voice_samples"),
                    "voice_models": str(root / "characters" / "voice_models"),
                }
            }
            shotlist = {
                "episode_id": "episode_010",
                "display_title": "Episode 10",
                "episode_title": "Package Writer Test",
                "scenes": [
                    {
                        "scene_id": "scene_0002",
                        "title": "Demo Scene",
                        "summary": "A clean scene summary.",
                        "characters": ["Kenzie Bell"],
                        "generation_plan": {
                            "positive_prompt": "bright sitcom living room",
                            "negative_prompt": "artifact",
                            "batch_prompt_line": "Demo Scene",
                        },
                    }
                ],
            }
            manifest = {
                "shotlist_path": str(root / "episode_010.json"),
                "render_manifest_path": str(root / "episode_010_render_manifest.json"),
                "voice_plan": str(root / "episode_010_voice_plan.json"),
                "draft_render": str(root / "draft.mp4"),
                "final_render": str(root / "final.mp4"),
                "dialogue_audio": "",
                "subtitle_preview": "",
                "scenes": [{"scene_id": "scene_0002", "duration_seconds": 3.0, "frame_path": str(root / "frame.png")}],
            }
            voice_plan_payload = {"scenes": [{"scene_id": "scene_0002", "duration_seconds": 3.0, "lines": []}]}

            written = STEPRENDER.write_episode_production_package(cfg, "episode_010", shotlist, manifest, voice_plan_payload)
            package_path = Path(written["package_path"])
            scene_path = Path(written["scene_package_paths"][0])
            prompt_preview_path = Path(written["prompt_preview_path"])

            self.assertTrue(package_path.exists())
            self.assertTrue(scene_path.exists())
            self.assertTrue(prompt_preview_path.exists())
            payload = json.loads(package_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["episode_id"], "episode_010")
            self.assertEqual(len(payload["scene_package_paths"]), 1)
            self.assertIn("image_prompt =", prompt_preview_path.read_text(encoding="utf-8"))

    def test_generated_episode_artifacts_collects_render_and_package_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            story_prompt_path = root / "generation" / "story_prompts" / "episode_011.md"
            shotlist_path = root / "generation" / "shotlists" / "episode_011.json"
            render_manifest_path = root / "generation" / "renders" / "final" / "episode_011_render_manifest.json"
            package_path = root / "generation" / "final_episode_packages" / "episode_011" / "master" / "episode_011_production_package.json"
            full_generated_episode = root / "generation" / "final_episode_packages" / "episode_011" / "master" / "episode_011_full_generated_episode.mp4"
            story_prompt_path.parent.mkdir(parents=True, exist_ok=True)
            shotlist_path.parent.mkdir(parents=True, exist_ok=True)
            render_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            package_path.parent.mkdir(parents=True, exist_ok=True)
            story_prompt_path.write_text("# Episode 011", encoding="utf-8")
            shotlist_path.write_text(
                json.dumps(
                    {
                        "display_title": "Episode 011",
                        "episode_title": "A New Test Episode",
                        "render_manifest": str(render_manifest_path),
                        "production_package": str(package_path),
                        "full_generated_episode": str(full_generated_episode),
                    }
                ),
                encoding="utf-8",
            )
            render_manifest_path.write_text(
                json.dumps(
                    {
                        "render_mode": "fully_generated_episode",
                        "draft_render": str(root / "generation" / "renders" / "drafts" / "episode_011.mp4"),
                        "final_render": str(root / "generation" / "renders" / "final" / "episode_011.mp4"),
                        "dialogue_audio": str(root / "generation" / "renders" / "final" / "episode_011_dialogue_audio.wav"),
                        "voice_plan": str(root / "generation" / "renders" / "final" / "episode_011_voice_plan.json"),
                        "subtitle_preview": str(root / "generation" / "renders" / "final" / "episode_011_dialogue_preview.srt"),
                        "generated_scene_video_count": 5,
                        "scene_master_clip_count": 5,
                    }
                ),
                encoding="utf-8",
            )
            package_path.write_text(
                json.dumps(
                    {
                        "package_root": str(package_path.parent.parent),
                        "prompt_preview_path": str(package_path.parent / "episode_011_production_prompt_preview.txt"),
                        "completion_status": {
                            "generated_scene_video_count": 5,
                            "scene_dialogue_audio_count": 5,
                            "scene_master_clip_count": 5,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch("pipeline_common.PROJECT_ROOT", root):
                artifacts = generated_episode_artifacts({}, "episode_011")

            self.assertEqual(artifacts["episode_id"], "episode_011")
            self.assertEqual(artifacts["display_title"], "Episode 011")
            self.assertEqual(artifacts["story_prompt"], str(story_prompt_path))
            self.assertEqual(artifacts["render_manifest"], str(render_manifest_path))
            self.assertEqual(artifacts["production_package"], str(package_path))
            self.assertEqual(artifacts["full_generated_episode"], str(full_generated_episode))
            self.assertEqual(artifacts["generated_scene_video_count"], 5)
            self.assertEqual(artifacts["scene_dialogue_audio_count"], 5)
            self.assertEqual(artifacts["scene_master_clip_count"], 5)
            self.assertEqual(artifacts["scene_count"], 5)
            self.assertEqual(artifacts["production_readiness"], "fully_generated_episode_ready")
            self.assertEqual(artifacts["remaining_backend_tasks"], [])
            self.assertAlmostEqual(artifacts["scene_video_completion_ratio"], 1.0)
            self.assertAlmostEqual(artifacts["scene_dialogue_completion_ratio"], 1.0)
            self.assertAlmostEqual(artifacts["scene_master_completion_ratio"], 1.0)

    def test_list_generated_episode_artifacts_returns_newest_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shotlist_dir = root / "generation" / "shotlists"
            shotlist_dir.mkdir(parents=True, exist_ok=True)
            older = shotlist_dir / "episode_010.json"
            newer = shotlist_dir / "episode_011.json"
            older.write_text(json.dumps({"display_title": "Older"}), encoding="utf-8")
            newer.write_text(json.dumps({"display_title": "Newer"}), encoding="utf-8")
            os.utime(older, (older.stat().st_atime, older.stat().st_mtime - 30))

            with mock.patch("pipeline_common.PROJECT_ROOT", root):
                rows = list_generated_episode_artifacts({}, limit=2)

            self.assertEqual([row["episode_id"] for row in rows], ["episode_011", "episode_010"])

    def test_build_series_bible_payload_includes_recent_generated_episodes(self) -> None:
        model = {
            "trained_at": "2026-04-18T12:00:00Z",
            "scene_count": 42,
            "characters": [{"name": "Babe", "scene_count": 9, "line_count": 17}],
            "keywords": ["family", "farm"],
            "scene_library": [{"episode_id": "S01E01", "scene_id": "scene_0010", "characters": ["Babe"], "transcript": "Hello there"}],
        }
        generated = [
            {
                "episode_id": "episode_200",
                "display_title": "Episode 200",
                "episode_title": "The Big Return",
                "render_mode": "fully_generated_episode",
                "final_render": "C:\\demo\\episode_200.mp4",
                "full_generated_episode": "C:\\demo\\episode_200_full_generated_episode.mp4",
                "production_package": "C:\\demo\\episode_200_production_package.json",
                "render_manifest": "C:\\demo\\episode_200_render_manifest.json",
                "generated_scene_video_count": 12,
                "scene_dialogue_audio_count": 12,
                "scene_master_clip_count": 12,
            }
        ]

        payload, markdown = STEPBIBLE.build_series_bible_payload(model, generated)

        self.assertEqual(payload["recent_generated_episodes"][0]["episode_id"], "episode_200")
        self.assertIn("## Recent Generated Episodes", markdown)
        self.assertIn("The Big Return", markdown)
        self.assertIn("episode_200_full_generated_episode.mp4", markdown)

    def test_foundation_training_candidates_skip_background_roles(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {"name": "Babe Carano", "ignored": False, "priority": True, "scene_count": 8},
                "face_002": {"name": "statist", "ignored": False, "priority": False, "scene_count": 12},
                "face_003": {"name": "face_003", "ignored": False, "priority": False, "scene_count": 12},
            }
        }
        series_model = {
            "characters": [
                {"name": "Babe Carano", "scene_count": 8, "line_count": 15, "priority": True},
                {"name": "statist", "scene_count": 30, "line_count": 0, "priority": False},
            ]
        }
        cfg = {"foundation_training": {"min_character_scene_count": 3, "min_character_line_count": 3}}

        candidates = STEP13.character_training_candidates(char_map, series_model, cfg)

        self.assertEqual([row["name"] for row in candidates], ["Babe Carano"])
        self.assertTrue(candidates[0]["priority"])

    def test_foundation_training_download_targets_only_include_configured_models(self) -> None:
        cfg = {
            "foundation_training": {
                "image_base_model": "demo/image-model",
                "video_base_model": "",
                "voice_base_model": "demo/voice-model",
                "use_local_character_voice_models": True,
            },
            "paths": {
                "foundation_downloads": "training/foundation/downloads",
            },
        }

        targets = STEP13.build_download_targets(cfg)

        self.assertEqual([row["kind"] for row in targets], ["image"])
        self.assertIn("demo/image-model", targets[0]["model_id"])

    def test_foundation_training_download_targets_include_voice_model_when_local_voice_models_disabled(self) -> None:
        cfg = {
            "foundation_training": {
                "image_base_model": "demo/image-model",
                "video_base_model": "",
                "voice_base_model": "demo/voice-model",
                "use_local_character_voice_models": False,
            },
            "paths": {
                "foundation_downloads": "training/foundation/downloads",
            },
        }

        targets = STEP13.build_download_targets(cfg)

        self.assertEqual([row["kind"] for row in targets], ["image", "voice"])

    def test_foundation_download_action_downloads_missing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = {"model_id": "demo/image-model", "target_dir": str(Path(tmpdir) / "image")}

            action = STEP13.resolve_download_action(target, "rev-123")

            self.assertEqual(action, "download")

    def test_foundation_download_action_skips_when_revision_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "image"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "weights.bin").write_text("demo", encoding="utf-8")
            target = {"model_id": "demo/image-model", "target_dir": str(target_dir), "kind": "image"}
            STEP13.write_download_metadata(target, {"model_id": "demo/image-model", "revision": "rev-123"})

            action = STEP13.resolve_download_action(target, "rev-123")

            self.assertEqual(action, "skip")

    def test_foundation_download_action_updates_when_revision_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "image"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "weights.bin").write_text("demo", encoding="utf-8")
            target = {"model_id": "demo/image-model", "target_dir": str(target_dir), "kind": "image"}
            STEP13.write_download_metadata(target, {"model_id": "demo/image-model", "revision": "rev-old"})

            action = STEP13.resolve_download_action(target, "rev-new")

            self.assertEqual(action, "update")

    def test_infer_local_revision_from_hf_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "image"
            metadata_dir = target_dir / ".cache" / "huggingface" / "download"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "weights.bin").write_text("demo", encoding="utf-8")
            (metadata_dir / "model_index.json.metadata").write_text(
                "rev-123456789abc\netag-demo\n1776398904.2590818\n",
                encoding="utf-8",
            )
            target = {"model_id": "demo/image-model", "target_dir": str(target_dir), "kind": "image"}

            revision = STEP13.infer_local_revision_from_cache(target)

            self.assertEqual(revision, "rev-123456789abc")

    def test_foundation_download_action_skips_when_revision_matches_via_hf_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "image"
            metadata_dir = target_dir / ".cache" / "huggingface" / "download"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "weights.bin").write_text("demo", encoding="utf-8")
            (metadata_dir / "model_index.json.metadata").write_text(
                "rev-123456789abc\netag-demo\n1776398904.2590818\n",
                encoding="utf-8",
            )
            target = {"model_id": "demo/image-model", "target_dir": str(target_dir), "kind": "image"}

            action = STEP13.resolve_download_action(target, "rev-123456789abc")

            self.assertEqual(action, "skip")

    def test_foundation_download_action_repairs_incomplete_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "image"
            metadata_dir = target_dir / ".cache" / "huggingface" / "download"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "weights.bin").write_text("demo", encoding="utf-8")
            (metadata_dir / "model_index.json.metadata").write_text(
                "rev-123456789abc\netag-demo\n1776398904.2590818\n",
                encoding="utf-8",
            )
            (metadata_dir / "broken.bin.incomplete").write_text("partial", encoding="utf-8")
            target = {"model_id": "demo/image-model", "target_dir": str(target_dir), "kind": "image"}

            action = STEP13.resolve_download_action(target, "rev-123456789abc")

            self.assertEqual(action, "cleanup")

    def test_cleanup_incomplete_download_files_removes_cache_remnants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "image"
            metadata_dir = target_dir / ".cache" / "huggingface" / "download"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            broken = metadata_dir / "broken.bin.incomplete"
            broken.write_text("partial", encoding="utf-8")
            target = {"model_id": "demo/image-model", "target_dir": str(target_dir), "kind": "image"}

            removed = STEP13.cleanup_incomplete_download_files(target)

            self.assertEqual(removed, 1)
            self.assertFalse(broken.exists())

    def test_cleanup_processed_inbox_episode_deletes_only_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            inbox_file = Path(tmpdir) / "demo.mp4"
            inbox_file.write_text("demo", encoding="utf-8")

            removed_existing = STEP99.cleanup_processed_inbox_episode(inbox_file)
            removed_missing = STEP99.cleanup_processed_inbox_episode(inbox_file)

            self.assertTrue(removed_existing)
            self.assertFalse(inbox_file.exists())
            self.assertFalse(removed_missing)

    def test_cleanup_split_working_file_deletes_only_working_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            inbox_file = temp_root / "inbox_demo.mp4"
            working_file = temp_root / "working_demo.mp4"
            inbox_file.write_text("inbox", encoding="utf-8")
            working_file.write_text("working", encoding="utf-8")

            removed_existing = STEP03.cleanup_split_working_file(working_file)
            removed_missing = STEP03.cleanup_split_working_file(working_file)

            self.assertTrue(removed_existing)
            self.assertTrue(inbox_file.exists())
            self.assertFalse(working_file.exists())
            self.assertFalse(removed_missing)

    def test_delete_inbox_after_verified_copy_requires_complete_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            inbox_file = temp_root / "inbox_demo.mp4"
            working_file = temp_root / "working_demo.mp4"
            inbox_file.write_bytes(b"abcdef")
            working_file.write_bytes(b"abcdef")

            removed = STEP02.delete_inbox_after_verified_copy(inbox_file, working_file)

            self.assertTrue(removed)
            self.assertFalse(inbox_file.exists())
            self.assertTrue(working_file.exists())

    def test_delete_inbox_after_verified_copy_keeps_inbox_when_sizes_differ(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            inbox_file = temp_root / "inbox_demo.mp4"
            working_file = temp_root / "working_demo.mp4"
            inbox_file.write_bytes(b"abcdef")
            working_file.write_bytes(b"abc")

            with self.assertRaises(RuntimeError):
                STEP02.delete_inbox_after_verified_copy(inbox_file, working_file)

            self.assertTrue(inbox_file.exists())

    def test_import_episode_parse_args_supports_all(self) -> None:
        with mock.patch("sys.argv", ["02_import_episode.py"]):
            args = STEP02.parse_args()
        self.assertFalse(args.all)
        with mock.patch("sys.argv", ["02_import_episode.py", "--all"]):
            args = STEP02.parse_args()
        self.assertTrue(args.all)

    def test_dataset_parse_args_supports_force(self) -> None:
        with mock.patch("sys.argv", ["07_build_dataset.py", "--force"]):
            args = STEP06.parse_args()
        self.assertTrue(args.force)

    def test_foundation_train_parse_args_supports_force(self) -> None:
        with mock.patch("sys.argv", ["10_train_foundation_models.py", "--force"]):
            args = STEP10TRAIN.parse_args()
        self.assertTrue(args.force)

    def test_build_training_pack_tracks_voice_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            voice_a = root / "voice_a.wav"
            voice_b = root / "voice_b.wav"

            def make_wav(path: Path, seconds: float) -> None:
                import struct
                import wave

                sample_rate = 24000
                frame_count = int(sample_rate * seconds)
                with wave.open(str(path), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(sample_rate)
                    handle.writeframes(b"".join(struct.pack("<h", 0) for _ in range(frame_count)))

            make_wav(voice_a, 4.5)
            make_wav(voice_b, 4.0)

            manifest = {
                "name": "Babe",
                "slug": "babe",
                "voice_samples": [
                    {
                        "path": str(voice_a),
                        "language": "en",
                        "source_type": "original_segment",
                        "episode_id": "episode_001",
                        "scene_id": "scene_0001",
                        "segment_id": "seg_001",
                        "duration_seconds": 4.5,
                        "text": "Let's ship it.",
                    },
                    {
                        "path": str(voice_b),
                        "language": "en",
                        "source_type": "original_segment",
                        "episode_id": "episode_001",
                        "scene_id": "scene_0002",
                        "segment_id": "seg_002",
                        "duration_seconds": 4.0,
                        "text": "We need one more take.",
                    },
                ],
                "frame_samples": [],
                "video_samples": [],
            }
            cfg = {
                "foundation_training": {
                    "voice_base_model": "openbmb/VoxCPM2",
                    "use_local_character_voice_models": True,
                    "min_voice_duration_seconds_total": 8.0,
                    "target_voice_duration_seconds_total": 18.0,
                    "min_voice_samples_for_clone": 2,
                },
                "generation": {"target_episode_minutes_min": 20.0, "target_episode_minutes_max": 24.0},
            }

            payload = STEP10TRAIN.build_training_pack(manifest, cfg)

            self.assertEqual(payload["voice_pack"]["sample_count"], 2)
            self.assertGreaterEqual(payload["voice_pack"]["duration_seconds_total"], 8.4)
            self.assertGreater(payload["voice_pack"]["quality_score"], 0.4)
            self.assertTrue(payload["voice_pack"]["clone_ready"])
            self.assertEqual(payload["base_models"]["voice"], "local_character_voice_model")
            self.assertEqual(payload["voice_pack"]["dominant_language"], "en")
            self.assertEqual(payload["voice_pack"]["original_voice_sample_count"], 2)

    def test_refresh_after_manual_review_parse_args(self) -> None:
        with mock.patch("sys.argv", ["20_refresh_after_manual_review.py", "--skip-downloads", "--stop-after-training", "--allow-open-review"]):
            args = STEP16.parse_args()
        self.assertTrue(args.skip_downloads)
        self.assertTrue(args.stop_after_training)
        self.assertTrue(args.allow_open_review)

    def test_planned_refresh_steps_keep_render_before_bible(self) -> None:
        cfg = {
            "foundation_training": {
                "prepare_after_batch": True,
                "auto_train_after_prepare": True,
                "required_before_generate": True,
                "required_before_render": True,
            },
            "adapter_training": {"auto_train_after_foundation": True},
            "fine_tune_training": {"auto_train_after_adapter": True},
            "backend_fine_tune": {"auto_run_after_fine_tune": True},
        }

        planned = STEP16.planned_refresh_steps(cfg, skip_downloads=True, stop_after_training=False)
        step_names = [row[0] for row in planned]

        self.assertLess(step_names.index("13_run_backend_finetunes.py"), step_names.index("14_generate_episode_from_trained_model.py"))
        self.assertLess(step_names.index("16_run_storyboard_backend.py"), step_names.index("17_render_episode.py"))
        self.assertLess(step_names.index("17_render_episode.py"), step_names.index("18_build_series_bible.py"))
        self.assertIn("--skip-downloads", planned[2][2])

    def test_planned_refresh_steps_stop_after_training_excludes_generate_render_block(self) -> None:
        cfg = {
            "foundation_training": {
                "prepare_after_batch": True,
                "auto_train_after_prepare": True,
                "required_before_generate": True,
                "required_before_render": True,
            },
            "adapter_training": {"auto_train_after_foundation": True},
            "fine_tune_training": {"auto_train_after_adapter": True},
            "backend_fine_tune": {"auto_run_after_fine_tune": True},
        }

        planned = STEP16.planned_refresh_steps(cfg, skip_downloads=False, stop_after_training=True)
        step_names = [row[0] for row in planned]

        self.assertIn("13_run_backend_finetunes.py", step_names)
        self.assertNotIn("14_generate_episode_from_trained_model.py", step_names)
        self.assertNotIn("17_render_episode.py", step_names)
        self.assertNotIn("18_build_series_bible.py", step_names)

    def test_planned_refresh_steps_respect_disabled_optional_training_stages(self) -> None:
        cfg = {
            "foundation_training": {
                "prepare_after_batch": False,
                "auto_train_after_prepare": False,
                "required_before_generate": False,
                "required_before_render": False,
            },
            "adapter_training": {"auto_train_after_foundation": False},
            "fine_tune_training": {"auto_train_after_adapter": False},
            "backend_fine_tune": {"auto_run_after_fine_tune": False},
        }

        planned = STEP16.planned_refresh_steps(cfg, skip_downloads=True, stop_after_training=False)
        step_names = [row[0] for row in planned]

        self.assertEqual(
            step_names,
            [
                "07_build_dataset.py",
                "08_train_series_model.py",
                "14_generate_episode_from_trained_model.py",
                "15_generate_storyboard_assets.py",
                "16_run_storyboard_backend.py",
                "17_render_episode.py",
                "18_build_series_bible.py",
            ],
        )

    def test_open_review_item_count_reads_review_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            review_queue = root / "characters" / "review" / "review_queue.json"
            review_queue.parent.mkdir(parents=True, exist_ok=True)
            review_queue.write_text(json.dumps({"items": [{"id": 1}, {"id": 2}, {"id": 3}]}), encoding="utf-8")

            cfg = {"paths": {"review_queue": str(review_queue.relative_to(root)).replace("\\", "/")}}
            with mock.patch("pipeline_common.PROJECT_ROOT", root):
                count = open_review_item_count(cfg)

        self.assertEqual(count, 3)

    def test_open_face_review_item_count_counts_only_unresolved_face_clusters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            char_map = root / "characters" / "maps" / "character_map.json"
            char_map.parent.mkdir(parents=True, exist_ok=True)
            char_map.write_text(
                json.dumps(
                    {
                        "clusters": {
                            "face_001": {"name": "face_001", "ignored": False},
                            "face_002": {"name": "Babe Carano", "ignored": False},
                            "face_003": {"name": "statist", "ignored": False},
                            "face_004": {"name": "noface", "ignored": True},
                        }
                    }
                ),
                encoding="utf-8",
            )

            cfg = {"paths": {"character_map": str(char_map.relative_to(root)).replace("\\", "/")}}
            with mock.patch("pipeline_common.PROJECT_ROOT", root):
                count = open_face_review_item_count(cfg)

        self.assertEqual(count, 1)

    def test_import_single_episode_creates_working_copy_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            inbox_file = temp_root / "Demo.S01E01.mp4"
            episodes_dir = temp_root / "episodes"
            metadata_dir = temp_root / "metadata"
            episodes_dir.mkdir(parents=True, exist_ok=True)
            metadata_dir.mkdir(parents=True, exist_ok=True)
            inbox_file.write_bytes(b"abcdef")

            with mock.patch.object(STEP02, "mark_step_started"), mock.patch.object(STEP02, "mark_step_completed"), mock.patch.object(STEP02, "mark_step_failed"):
                imported = STEP02.import_single_episode(inbox_file, episodes_dir, metadata_dir)

            self.assertTrue(imported)
            self.assertFalse(inbox_file.exists())
            self.assertEqual((episodes_dir / "Demo.S01E01.mp4").read_bytes(), b"abcdef")
            metadata = json.loads((metadata_dir / "Demo.S01E01.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["episode_id"], "Demo.S01E01")
            self.assertEqual(Path(metadata["working_file"]).name, "Demo.S01E01.mp4")

    def test_purge_already_processed_inbox_videos_removes_matching_working_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            inbox_dir = temp_root / "inbox"
            episodes_dir = temp_root / "episodes"
            scene_root = temp_root / "scene_clips"
            metadata_dir = temp_root / "metadata"
            for path in (inbox_dir, episodes_dir, scene_root, metadata_dir):
                path.mkdir(parents=True, exist_ok=True)

            inbox_file = inbox_dir / "Demo.mp4"
            working_file = episodes_dir / "Demo.mp4"
            inbox_file.write_bytes(b"abcdef")
            working_file.write_bytes(b"abcdef")

            removed = STEP02.purge_already_processed_inbox_videos(inbox_dir, episodes_dir, scene_root, metadata_dir)

            self.assertEqual(removed, 1)
            self.assertFalse(inbox_file.exists())

    def test_purge_already_processed_inbox_videos_removes_when_scene_dir_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            inbox_dir = temp_root / "inbox"
            episodes_dir = temp_root / "episodes"
            scene_root = temp_root / "scene_clips"
            metadata_dir = temp_root / "metadata"
            for path in (inbox_dir, episodes_dir, scene_root, metadata_dir):
                path.mkdir(parents=True, exist_ok=True)

            inbox_file = inbox_dir / "Demo.mp4"
            inbox_file.write_bytes(b"abcdef")
            (scene_root / "Demo").mkdir(parents=True, exist_ok=True)

            removed = STEP02.purge_already_processed_inbox_videos(inbox_dir, episodes_dir, scene_root, metadata_dir)

            self.assertEqual(removed, 1)
            self.assertFalse(inbox_file.exists())

    def test_save_autosave_keeps_only_two_latest_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            autosave_root = Path(tmpdir)
            cfg = {}
            with mock.patch.object(STEP99, "resolve_project_path", return_value=autosave_root):
                state = STEP99.default_state()
                STEP99.save_autosave(cfg, state, "first")
                STEP99.save_autosave(cfg, state, "second")
                latest = STEP99.save_autosave(cfg, state, "third")

                snapshots = sorted(autosave_root.glob("autosave_*.json"))
                self.assertEqual(len(snapshots), 2)
                self.assertEqual(snapshots[-1], latest)
                loaded = STEP99.load_latest_autosave(cfg)
                self.assertEqual(loaded["autosave_reason"], "third")

    def test_next_autosave_path_avoids_filename_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            autosave_root = Path(tmpdir)
            with mock.patch.object(STEP99, "autosave_filename", return_value="autosave_fixed.json"):
                first = STEP99.next_autosave_path(autosave_root)
                first.write_text("first", encoding="utf-8")
                second = STEP99.next_autosave_path(autosave_root)

        self.assertEqual(first.name, "autosave_fixed.json")
        self.assertEqual(second.name, "autosave_fixed_001.json")

    def test_build_status_snapshot_tracks_episode_and_global_progress(self) -> None:
        state = STEP99.default_state()
        state["setup_completed"] = True
        state["processed_count"] = 1
        state["processed_episodes"] = ["Episode01"]
        state["episode_steps_completed"] = {
            "Episode01": list(STEP99.EPISODE_STEPS),
            "Episode02": ["02_import_episode.py", "03_split_scenes.py"],
        }
        state["current_phase"] = "episode"
        state["current_episode_name"] = "Episode02"
        state["current_episode_file"] = "Episode02.mp4"
        state["current_step"] = "04_diarize_and_transcribe.py"
        state["global_steps_completed"] = ["08_train_series_model.py"]
        cfg = {"foundation_training": {"required_before_generate": True, "required_before_render": True}}

        snapshot = STEP99.build_status_snapshot(cfg, state, inbox_dir=None)

        self.assertEqual(snapshot["processed_count"], 1)
        episode_rows = {row["episode"]: row for row in snapshot["episode_progress"]}
        self.assertEqual(episode_rows["Episode01"]["status"], "completed")
        self.assertEqual(episode_rows["Episode02"]["status"], "running")
        self.assertEqual(episode_rows["Episode02"]["current_step"], "04_diarize_and_transcribe.py")
        global_rows = {row["step"]: row["status"] for row in snapshot["global_progress"]}
        self.assertEqual(global_rows["08_train_series_model.py"], "completed")
        self.assertEqual(global_rows["09_prepare_foundation_training.py"], "pending")

    def test_build_status_snapshot_uses_stored_global_plan_labels(self) -> None:
        state = STEP99.default_state()
        state["skip_downloads"] = True
        state["global_planned_steps"] = STEP99.serialize_global_step_rows(
            STEP99.global_step_rows(
                {
                    "foundation_training": {
                        "prepare_after_batch": True,
                        "auto_train_after_prepare": True,
                        "required_before_generate": False,
                        "required_before_render": False,
                    }
                },
                skip_downloads=True,
            )
        )
        state["global_steps_completed"] = ["07_build_dataset.py"]
        cfg = {"foundation_training": {"required_before_generate": False, "required_before_render": False}}

        snapshot = STEP99.build_status_snapshot(cfg, state, inbox_dir=None)

        self.assertTrue(snapshot["skip_downloads"])
        global_rows = {row["step"]: row["status"] for row in snapshot["global_progress"]}
        self.assertEqual(global_rows["07_build_dataset.py"], "completed")
        self.assertEqual(global_rows["09_prepare_foundation_training.py --skip-downloads"], "pending")

    def test_completed_global_step_labels_follow_planned_order(self) -> None:
        planned_steps = STEP99.serialize_global_step_rows(
            [
                ("07_build_dataset.py", "Build training dataset from reviewed data", []),
                ("09_prepare_foundation_training.py", "Prepare Foundation Training", ["--skip-downloads"]),
                ("10_train_foundation_models.py", "Train Foundation Models", []),
            ]
        )

        labels = STEP99.completed_global_step_labels(
            planned_steps,
            {"10_train_foundation_models.py", "07_build_dataset.py"},
        )

        self.assertEqual(labels, ["07_build_dataset.py", "10_train_foundation_models.py"])

    def test_record_global_generated_episode_outputs_tracks_latest_finished_episode(self) -> None:
        state = STEP99.default_state()
        outputs = {
            "episode_id": "episode_200",
            "final_render": "C:\\demo\\episode_200.mp4",
            "full_generated_episode": "C:\\demo\\episode_200_full_generated_episode.mp4",
            "production_package": "C:\\demo\\episode_200_production_package.json",
        }

        with mock.patch.object(STEP99, "latest_generated_episode_artifacts", return_value=outputs):
            STEP99.record_global_generated_episode_outputs(state, {}, "17_render_episode.py")

        self.assertEqual(state["latest_generated_episode"]["episode_id"], "episode_200")
        self.assertEqual(state["global_step_outputs"]["17_render_episode.py"]["production_package"], outputs["production_package"])

    def test_generated_episode_completion_summary_marks_missing_backend_tasks(self) -> None:
        summary = generated_episode_completion_summary(
            scene_count=6,
            generated_scene_video_count=2,
            scene_dialogue_audio_count=6,
            scene_master_clip_count=1,
            render_mode="hybrid_generated",
            final_render="C:\\demo\\episode_300.mp4",
            full_generated_episode="",
        )

        self.assertEqual(summary["production_readiness"], "hybrid_generated_episode")
        self.assertEqual(summary["remaining_backend_tasks"], ["generate missing scene videos", "master remaining scene clips", "assemble the full generated episode master"])
        self.assertAlmostEqual(summary["scene_video_completion_ratio"], 0.3333, places=4)
        self.assertAlmostEqual(summary["scene_dialogue_completion_ratio"], 1.0)
        self.assertAlmostEqual(summary["scene_master_completion_ratio"], 0.1667, places=4)

    def test_build_series_bible_payload_includes_readiness_and_remaining_tasks(self) -> None:
        model = {
            "trained_at": "2026-04-18 12:00:00",
            "scene_count": 24,
            "dataset_files": ["demo.jsonl"],
            "characters": [{"name": "Babe Carano", "scene_count": 12, "line_count": 40}],
            "keywords": ["mystery"],
            "scene_library": [{"episode_id": "episode_001", "scene_id": "scene_001", "characters": ["Babe Carano"], "transcript": "demo"}],
        }
        generated_episodes = [
            {
                "episode_id": "episode_301",
                "display_title": "Episode 301",
                "episode_title": "Backend Progress",
                "render_mode": "hybrid_generated",
                "production_readiness": "hybrid_generated_episode",
                "final_render": "C:\\demo\\episode_301.mp4",
                "full_generated_episode": "",
                "production_package": "C:\\demo\\episode_301_production_package.json",
                "render_manifest": "C:\\demo\\episode_301_render_manifest.json",
                "scene_count": 6,
                "generated_scene_video_count": 3,
                "scene_dialogue_audio_count": 6,
                "scene_master_clip_count": 2,
                "scene_video_completion_ratio": 0.5,
                "scene_dialogue_completion_ratio": 1.0,
                "scene_master_completion_ratio": 0.3333,
                "remaining_backend_tasks": ["generate missing scene videos", "master remaining scene clips"],
            }
        ]

        _payload, markdown = STEPBIBLE.build_series_bible_payload(model, generated_episodes)

        self.assertIn("- Production readiness: hybrid_generated_episode", markdown)
        self.assertIn("- Scene video coverage: 50%", markdown)
        self.assertIn("- Scene dialogue coverage: 100%", markdown)
        self.assertIn("- Remaining backend tasks: generate missing scene videos, master remaining scene clips", markdown)

    def test_load_latest_autosave_ignores_completed_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            autosave_root = Path(tmpdir)
            cfg = {}
            with mock.patch.object(STEP99, "resolve_project_path", return_value=autosave_root):
                state = STEP99.default_state()
                state["status"] = "completed"
                STEP99.save_autosave(cfg, state, "done")

                loaded = STEP99.load_latest_autosave(cfg)

                self.assertIsNone(loaded)

    def test_foundation_training_maps_face_cluster_to_named_character(self) -> None:
        row = {
            "characters_visible": ["face_004"],
            "face_clusters": ["face_004"],
            "transcript_segments": [],
        }
        known_clusters = {"face_004": "Babe"}

        names = STEP13.scene_character_names(row, known_clusters)

        self.assertIn("Babe", names)

    def test_split_scene_resolve_episode_file_accepts_name_and_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episodes_dir = Path(tmp)
            scene_root = episodes_dir / "scene_clips"
            scene_root.mkdir()
            target = episodes_dir / "Demo.Folge.S01E01.mp4"
            target.write_bytes(b"demo")

            by_name = STEP03.resolve_episode_file(episodes_dir, scene_root, "Demo.Folge.S01E01.mp4")
            by_stem = STEP03.resolve_episode_file(episodes_dir, scene_root, "Demo.Folge.S01E01")

            self.assertEqual(by_name, target)
            self.assertEqual(by_stem, target)

    def test_split_scene_resolve_episode_file_without_explicit_name_returns_first_working_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episodes_dir = Path(tmp) / "episodes"
            scene_root = Path(tmp) / "scene_clips"
            episodes_dir.mkdir(parents=True, exist_ok=True)
            scene_root.mkdir(parents=True, exist_ok=True)

            processed = episodes_dir / "Episode01.mp4"
            pending = episodes_dir / "Episode02.mp4"
            processed.write_bytes(b"processed")
            pending.write_bytes(b"pending")
            (scene_root / "Episode01").mkdir()

            resolved = STEP03.resolve_episode_file(episodes_dir, scene_root, None)

            self.assertEqual(resolved, processed)

    def test_pending_episode_files_returns_all_working_copies_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episodes_dir = Path(tmp) / "episodes"
            episodes_dir.mkdir(parents=True, exist_ok=True)
            (episodes_dir / "Episode03.mp4").write_bytes(b"3")
            (episodes_dir / "Episode01.mp4").write_bytes(b"1")
            (episodes_dir / "Episode02.mp4").write_bytes(b"2")

            pending = STEP03.pending_episode_files(episodes_dir)

            self.assertEqual([path.name for path in pending], ["Episode01.mp4", "Episode02.mp4", "Episode03.mp4"])

    def test_split_was_completed_successfully_uses_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            out_dir = temp_root / "scene_clips" / "Demo"
            scene_index_root = temp_root / "scene_index"
            out_dir.mkdir(parents=True, exist_ok=True)
            scene_index_root.mkdir(parents=True, exist_ok=True)
            episode = temp_root / "Demo.mp4"
            episode.write_bytes(b"demo")
            for index in range(2):
                (out_dir / f"scene_{index + 1:04d}.mp4").write_bytes(b"clip")
            marker_path = STEP03.split_success_marker_path(scene_index_root, episode.stem)
            STEP03.write_split_success_marker(marker_path, episode, 2, None, "segment_fallback")

            completed, count, source = STEP03.split_was_completed_successfully(
                out_dir,
                scene_index_root / f"{episode.stem}_scenes.csv",
                marker_path,
            )

            self.assertTrue(completed)
            self.assertEqual(count, 2)
            self.assertEqual(source, "marker")

    def test_split_was_completed_successfully_uses_csv_for_legacy_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            out_dir = temp_root / "scene_clips" / "Demo"
            scene_index_root = temp_root / "scene_index"
            out_dir.mkdir(parents=True, exist_ok=True)
            scene_index_root.mkdir(parents=True, exist_ok=True)
            scene_csv = scene_index_root / "Demo_scenes.csv"
            for index in range(3):
                (out_dir / f"scene_{index + 1:04d}.mp4").write_bytes(b"clip")
            scene_csv.write_text(
                "scene_number,start_seconds,end_seconds\n1,0.0,1.0\n2,1.0,2.0\n3,2.0,3.0\n",
                encoding="utf-8",
            )

            completed, count, source = STEP03.split_was_completed_successfully(
                out_dir,
                scene_csv,
                scene_index_root / "Demo_split_success.json",
            )

            self.assertTrue(completed)
            self.assertEqual(count, 3)
            self.assertEqual(source, "csv")

    def test_transcribe_resolve_episode_dir_accepts_name_and_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_root = Path(tmp)
            target = scene_root / "Demo.Folge.S01E01"
            target.mkdir()

            cfg = {"paths": {"speaker_transcripts": "speaker_transcripts", "speaker_segments": "speaker_segments"}}
            with mock.patch.object(STEP04, "resolve_project_path", side_effect=lambda value: scene_root / value):
                by_name = STEP04.resolve_episode_dir(scene_root, "Demo.Folge.S01E01", cfg)
                by_stem = STEP04.resolve_episode_dir(scene_root, "Demo.Folge.S01E01.mp4", cfg)

            self.assertEqual(by_name, target)
            self.assertEqual(by_stem, target)

    def test_dataset_resolve_episode_dir_accepts_name_and_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_root = Path(tmp)
            target = scene_root / "Demo.Folge.S01E01"
            target.mkdir()

            cfg = {"paths": {"datasets_video_training": "datasets"}}
            with mock.patch.object(STEP06, "resolve_project_path", side_effect=lambda value: scene_root / value):
                by_name = STEP06.resolve_episode_dir(scene_root, "Demo.Folge.S01E01", cfg)
                by_stem = STEP06.resolve_episode_dir(scene_root, "Demo.Folge.S01E01.mp4", cfg)

            self.assertEqual(by_name, target)
            self.assertEqual(by_stem, target)

    def test_transcribe_resolve_episode_dir_prefers_next_untranscribed_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_root = Path(tmp) / "scene_clips"
            scene_root.mkdir(parents=True, exist_ok=True)
            done = scene_root / "Episode01"
            pending = scene_root / "Episode02"
            done.mkdir()
            pending.mkdir()
            transcripts_dir = Path(tmp) / "speaker_transcripts"
            segments_dir = Path(tmp) / "speaker_segments" / "Episode01"
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            segments_dir.mkdir(parents=True, exist_ok=True)
            (transcripts_dir / "Episode01_segments.json").write_text(
                json.dumps([{"process_version": STEP04.PROCESS_VERSION}]),
                encoding="utf-8",
            )
            (segments_dir / "_speaker_clusters.json").write_text(
                json.dumps({"process_version": STEP04.PROCESS_VERSION}),
                encoding="utf-8",
            )
            cfg = {"paths": {"speaker_transcripts": "speaker_transcripts", "speaker_segments": "speaker_segments"}}

            with mock.patch.object(STEP04, "resolve_project_path", side_effect=lambda value: Path(tmp) / value):
                resolved = STEP04.resolve_episode_dir(scene_root, None, cfg)

            self.assertEqual(resolved, pending)

    def test_pending_untranscribed_episode_dirs_returns_all_open_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_root = Path(tmp) / "scene_clips"
            scene_root.mkdir(parents=True, exist_ok=True)
            done = scene_root / "Episode01"
            pending_a = scene_root / "Episode02"
            pending_b = scene_root / "Episode03"
            done.mkdir()
            pending_a.mkdir()
            pending_b.mkdir()
            transcripts_dir = Path(tmp) / "speaker_transcripts"
            segments_dir = Path(tmp) / "speaker_segments" / "Episode01"
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            segments_dir.mkdir(parents=True, exist_ok=True)
            (transcripts_dir / "Episode01_segments.json").write_text(
                json.dumps([{"process_version": STEP04.PROCESS_VERSION}]),
                encoding="utf-8",
            )
            (segments_dir / "_speaker_clusters.json").write_text(
                json.dumps({"process_version": STEP04.PROCESS_VERSION}),
                encoding="utf-8",
            )
            cfg = {"paths": {"speaker_transcripts": "speaker_transcripts", "speaker_segments": "speaker_segments"}}

            with mock.patch.object(STEP04, "resolve_project_path", side_effect=lambda value: Path(tmp) / value):
                resolved = STEP04.pending_untranscribed_episode_dirs(scene_root, cfg)

            self.assertEqual(resolved, [pending_a, pending_b])

    def test_transcribe_scene_detects_language_when_config_is_auto(self) -> None:
        class DummyModel:
            def __init__(self) -> None:
                self.kwargs = {}

            def transcribe(self, _path, **kwargs):
                self.kwargs = kwargs
                return {
                    "language": "en",
                    "segments": [
                        {"start": 0.0, "end": 1.2, "text": "Hello there."},
                        {"start": 1.25, "end": 2.4, "text": "We made it."},
                    ],
                }

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "scene.wav"
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(24000)
                handle.writeframes(b"\x00\x00" * 24000)

            model = DummyModel()
            cfg = {
                "transcription": {
                    "language": "auto",
                    "task": "transcribe",
                    "merge_gap_seconds": 0.35,
                    "min_segment_seconds": 0.6,
                }
            }

            payload = STEP04.transcribe_scene(model, wav_path, cfg, use_fp16=False)

        self.assertNotIn("language", model.kwargs)
        self.assertEqual(payload["detected_language"], "en")
        self.assertEqual(payload["segments"][0]["language"], "en")

    def test_transcribe_scene_prefers_filename_language_hint_over_detection(self) -> None:
        class DummyModel:
            def __init__(self) -> None:
                self.kwargs = {}

            def transcribe(self, _path, **kwargs):
                self.kwargs = kwargs
                return {
                    "language": "en",
                    "segments": [
                        {"start": 0.0, "end": 1.2, "text": "Hallo zusammen."},
                    ],
                }

        with tempfile.TemporaryDirectory() as tmp:
            episode_dir = Path(tmp) / "Game.Shakers.S02E09.GERMAN.720p.WEB.H264-BiMBAMBiNO"
            episode_dir.mkdir(parents=True, exist_ok=True)
            wav_path = episode_dir / "scene_0001.wav"
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(24000)
                handle.writeframes(b"\x00\x00" * 24000)

            model = DummyModel()
            cfg = {
                "transcription": {
                    "language": "auto",
                    "task": "transcribe",
                    "merge_gap_seconds": 0.35,
                    "min_segment_seconds": 0.6,
                }
            }

            payload = STEP04.transcribe_scene(model, wav_path, cfg, use_fp16=False)

        self.assertEqual(model.kwargs.get("language"), "de")
        self.assertEqual(payload["detected_language"], "de")
        self.assertEqual(payload["segments"][0]["language"], "de")

    def test_stale_shared_transcription_run_requires_in_progress_without_cache_or_leases(self) -> None:
        self.assertTrue(
            STEP04.stale_shared_transcription_run(
                {"status": "in_progress"},
                completed_scene_ids=[],
                active_scene_leases=[],
            )
        )
        self.assertFalse(
            STEP04.stale_shared_transcription_run(
                {"status": "completed"},
                completed_scene_ids=[],
                active_scene_leases=[],
            )
        )
        self.assertFalse(
            STEP04.stale_shared_transcription_run(
                {"status": "in_progress"},
                completed_scene_ids=["scene_0001"],
                active_scene_leases=[],
            )
        )
        self.assertFalse(
            STEP04.stale_shared_transcription_run(
                {"status": "in_progress"},
                completed_scene_ids=[],
                active_scene_leases=["scene_0001"],
            )
        )

    def test_episode_transcription_completed_rejects_autosave_scene_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_dir = root / "scene_clips" / "Episode01"
            scene_dir.mkdir(parents=True, exist_ok=True)
            (scene_dir / "scene_0001.mp4").write_bytes(b"clip")
            transcripts_dir = root / "speaker_transcripts"
            segments_dir = root / "speaker_segments" / "Episode01"
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            segments_dir.mkdir(parents=True, exist_ok=True)
            (transcripts_dir / "Episode01_segments.json").write_text(
                json.dumps([{"process_version": STEP04.PROCESS_VERSION}]),
                encoding="utf-8",
            )
            (segments_dir / "_speaker_clusters.json").write_text(
                json.dumps({"process_version": STEP04.PROCESS_VERSION}),
                encoding="utf-8",
            )
            cfg = {"paths": {"speaker_transcripts": "speaker_transcripts", "speaker_segments": "speaker_segments"}}
            with mock.patch.object(STEP04, "resolve_project_path", side_effect=lambda value: root / value):
                with mock.patch.object(
                    STEP04,
                    "completed_step_state",
                    return_value={"status": "completed", "process_version": STEP04.PROCESS_VERSION, "scene_count": 2},
                ):
                    self.assertFalse(STEP04.episode_transcription_completed(scene_dir, cfg))

    def test_assign_speaker_clusters_rescues_low_quality_unknown_rows(self) -> None:
        cfg = {
            "transcription": {
                "speaker_cluster_high_quality_min_seconds": 1.0,
                "speaker_cluster_min_segments": 2,
                "speaker_unknown_rescue_margin": 0.12,
                "speaker_unknown_neighbor_margin": 0.16,
            }
        }
        rows = [
            {
                "scene_id": "scene_0001",
                "segment_id": "seg_001",
                "start": 0.0,
                "end": 1.3,
                "text": "Hallo zusammen",
                "voice_embedding": [1.0, 0.0],
            },
            {
                "scene_id": "scene_0002",
                "segment_id": "seg_002",
                "start": 0.0,
                "end": 1.4,
                "text": "Wir bleiben dabei",
                "voice_embedding": [0.98, 0.02],
            },
            {
                "scene_id": "scene_0002",
                "segment_id": "seg_003",
                "start": 1.5,
                "end": 1.8,
                "text": "ja",
                "voice_embedding": [0.9, 0.4],
            },
        ]

        assigned_rows, clusters = STEP04.assign_speaker_clusters(rows, 0.9, cfg)

        rescued = {row["segment_id"]: row["speaker_cluster"] for row in assigned_rows}
        self.assertEqual(rescued["seg_001"], "speaker_001")
        self.assertEqual(rescued["seg_002"], "speaker_001")
        self.assertEqual(rescued["seg_003"], "speaker_001")
        self.assertEqual(len(clusters), 1)

    def test_assign_speaker_clusters_rescues_unknown_rows_with_episode_consensus(self) -> None:
        cfg = {
            "transcription": {
                "speaker_cluster_high_quality_min_seconds": 1.0,
                "speaker_cluster_min_segments": 2,
                "speaker_unknown_rescue_margin": 0.08,
                "speaker_unknown_neighbor_margin": 0.12,
                "speaker_unknown_episode_rescue_margin": 0.12,
                "speaker_unknown_episode_embedding_margin": 0.03,
                "speaker_unknown_episode_min_token_score": 2.0,
                "speaker_unknown_episode_min_token_margin": 0.5,
            }
        }
        rows = [
            {
                "scene_id": "scene_0001",
                "segment_id": "seg_001",
                "start": 0.0,
                "end": 1.3,
                "text": "Das ist unser geheimer Code fuer die App",
                "voice_embedding": [1.0, 0.0],
            },
            {
                "scene_id": "scene_0002",
                "segment_id": "seg_002",
                "start": 0.0,
                "end": 1.4,
                "text": "Der Code aus der App ist wieder kaputt",
                "voice_embedding": [0.98, 0.02],
            },
            {
                "scene_id": "scene_0003",
                "segment_id": "seg_003",
                "start": 0.0,
                "end": 1.2,
                "text": "Meine Bestellung kommt spaeter im Laden an",
                "voice_embedding": [0.0, 1.0],
            },
            {
                "scene_id": "scene_0004",
                "segment_id": "seg_004",
                "start": 0.0,
                "end": 1.2,
                "text": "Im Laden wartet schon die Bestellung",
                "voice_embedding": [0.02, 0.98],
            },
            {
                "scene_id": "scene_0005",
                "segment_id": "seg_005",
                "start": 0.0,
                "end": 0.7,
                "text": "Der geheime Code fuer unsere App bleibt heute intern",
                "voice_embedding": [0.93, 0.07],
            },
        ]

        assigned_rows, clusters = STEP04.assign_speaker_clusters(rows, 0.9, cfg)

        rescued = {row["segment_id"]: row["speaker_cluster"] for row in assigned_rows}
        self.assertEqual(rescued["seg_001"], "speaker_001")
        self.assertEqual(rescued["seg_002"], "speaker_001")
        self.assertEqual(rescued["seg_003"], "speaker_002")
        self.assertEqual(rescued["seg_004"], "speaker_002")
        self.assertEqual(rescued["seg_005"], "speaker_001")
        self.assertEqual(len(clusters), 2)

    def test_episode_consensus_rescue_keeps_unknown_when_text_signal_is_ambiguous(self) -> None:
        cfg = {
            "transcription": {
                "speaker_unknown_episode_rescue_margin": 0.12,
                "speaker_unknown_episode_embedding_margin": 0.08,
                "speaker_unknown_episode_min_token_score": 2.0,
                "speaker_unknown_episode_min_token_margin": 1.1,
            }
        }
        rows = [
            {
                "scene_id": "scene_0001",
                "segment_id": "seg_001",
                "start": 0.0,
                "end": 1.3,
                "text": "Das ist unser geheimer Code fuer die App",
                "voice_embedding": [1.0, 0.0],
                "speaker_cluster": "speaker_001",
            },
            {
                "scene_id": "scene_0002",
                "segment_id": "seg_002",
                "start": 0.0,
                "end": 1.4,
                "text": "Der Code aus der App ist wieder kaputt",
                "voice_embedding": [0.98, 0.02],
                "speaker_cluster": "speaker_001",
            },
            {
                "scene_id": "scene_0003",
                "segment_id": "seg_003",
                "start": 0.0,
                "end": 1.2,
                "text": "Meine Bestellung kommt spaeter im Laden an",
                "voice_embedding": [0.0, 1.0],
                "speaker_cluster": "speaker_002",
            },
            {
                "scene_id": "scene_0004",
                "segment_id": "seg_004",
                "start": 0.0,
                "end": 1.2,
                "text": "Im Laden wartet schon die Bestellung",
                "voice_embedding": [0.02, 0.98],
                "speaker_cluster": "speaker_002",
            },
            {
                "scene_id": "scene_0005",
                "segment_id": "seg_005",
                "start": 0.0,
                "end": 0.7,
                "text": "Code und Bestellung sind heute beide wichtig",
                "voice_embedding": [0.93, 0.07],
                "speaker_cluster": "speaker_unknown",
            },
        ]
        cluster_by_id = {
            "speaker_001": {
                "cluster_id": "speaker_001",
                "centroid": [1.0, 0.0],
                "count": 2,
                "scene_ids": {"scene_0001", "scene_0002"},
                "segments": ["seg_001", "seg_002"],
            },
            "speaker_002": {
                "cluster_id": "speaker_002",
                "centroid": [0.0, 1.0],
                "count": 2,
                "scene_ids": {"scene_0003", "scene_0004"},
                "segments": ["seg_003", "seg_004"],
            },
        }

        rescued_count = STEP04.rescue_unknown_speaker_rows_with_episode_consensus(
            rows,
            cluster_by_id,
            {"speaker_001", "speaker_002"},
            0.9,
            cfg,
        )

        rescued = {row["segment_id"]: row["speaker_cluster"] for row in rows}
        self.assertEqual(rescued_count, 0)
        self.assertEqual(rescued["seg_005"], "speaker_unknown")

    def test_face_linking_resolve_episode_dir_prefers_next_unlinked_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_root = Path(tmp) / "scene_clips"
            scene_root.mkdir(parents=True, exist_ok=True)
            done = scene_root / "Episode01"
            pending = scene_root / "Episode02"
            done.mkdir()
            pending.mkdir()
            linked_dir = Path(tmp) / "linked_segments"
            faces_dir = Path(tmp) / "faces" / "Episode01"
            linked_dir.mkdir(parents=True, exist_ok=True)
            faces_dir.mkdir(parents=True, exist_ok=True)
            (linked_dir / "Episode01_linked_segments.json").write_text(json.dumps([{"segment_id": "x"}]), encoding="utf-8")
            (faces_dir / "Episode01_face_summary.json").write_text(json.dumps([{"scene_id": "scene_0001"}]), encoding="utf-8")
            cfg = {"paths": {"linked_segments": "linked_segments", "faces": "faces"}}

            with mock.patch.object(STEP05, "resolve_project_path", side_effect=lambda value: Path(tmp) / value):
                resolved = STEP05.resolve_episode_dir_for_processing(scene_root, None, cfg)

            self.assertEqual(resolved, pending)

    def test_pending_unlinked_episode_dirs_returns_all_open_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_root = Path(tmp) / "scene_clips"
            scene_root.mkdir(parents=True, exist_ok=True)
            done = scene_root / "Episode01"
            pending_a = scene_root / "Episode02"
            pending_b = scene_root / "Episode03"
            done.mkdir()
            pending_a.mkdir()
            pending_b.mkdir()
            linked_dir = Path(tmp) / "linked_segments"
            faces_dir = Path(tmp) / "faces" / "Episode01"
            linked_dir.mkdir(parents=True, exist_ok=True)
            faces_dir.mkdir(parents=True, exist_ok=True)
            (linked_dir / "Episode01_linked_segments.json").write_text(json.dumps([{"segment_id": "x"}]), encoding="utf-8")
            (faces_dir / "Episode01_face_summary.json").write_text(json.dumps([{"scene_id": "scene_0001"}]), encoding="utf-8")
            cfg = {"paths": {"linked_segments": "linked_segments", "faces": "faces"}}

            with mock.patch.object(STEP05, "resolve_project_path", side_effect=lambda value: Path(tmp) / value):
                resolved = STEP05.pending_unlinked_episode_dirs(scene_root, cfg)

            self.assertEqual(resolved, [pending_a, pending_b])

    def test_episode_face_linking_completed_requires_current_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_dir = root / "scene_clips" / "Episode01"
            scene_dir.mkdir(parents=True, exist_ok=True)
            linked_dir = root / "linked_segments"
            faces_dir = root / "faces" / "Episode01"
            linked_dir.mkdir(parents=True, exist_ok=True)
            faces_dir.mkdir(parents=True, exist_ok=True)
            (linked_dir / "Episode01_linked_segments.json").write_text(json.dumps([{"segment_id": "x"}]), encoding="utf-8")
            (faces_dir / "Episode01_face_summary.json").write_text(json.dumps([{"scene_id": "scene_0001"}]), encoding="utf-8")
            (faces_dir / "_face_linking_success.json").write_text(
                json.dumps(
                    {
                        "process_version": STEP05.PROCESS_VERSION,
                        "linked_row_count": 2,
                        "face_scene_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            cfg = {"paths": {"linked_segments": "linked_segments", "faces": "faces"}}

            with mock.patch.object(STEP05, "resolve_project_path", side_effect=lambda value: root / value):
                self.assertFalse(STEP05.episode_face_linking_completed(scene_dir, cfg))

    def test_dataset_resolve_episode_dir_prefers_next_missing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_root = Path(tmp) / "scene_clips"
            scene_root.mkdir(parents=True, exist_ok=True)
            done = scene_root / "Episode01"
            pending = scene_root / "Episode02"
            done.mkdir()
            pending.mkdir()
            datasets_dir = Path(tmp) / "datasets"
            datasets_dir.mkdir(parents=True, exist_ok=True)
            (datasets_dir / "Episode01_dataset.json").write_text(json.dumps([{"scene_id": "scene_0001"}]), encoding="utf-8")
            cfg = {"paths": {"datasets_video_training": "datasets"}}

            with mock.patch.object(STEP06, "resolve_project_path", side_effect=lambda value: Path(tmp) / value):
                resolved = STEP06.resolve_episode_dir(scene_root, None, cfg)

            self.assertEqual(resolved, pending)

    def test_pending_undataseted_episode_dirs_returns_all_open_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_root = Path(tmp) / "scene_clips"
            scene_root.mkdir(parents=True, exist_ok=True)
            done = scene_root / "Episode01"
            pending_a = scene_root / "Episode02"
            pending_b = scene_root / "Episode03"
            done.mkdir()
            pending_a.mkdir()
            pending_b.mkdir()
            datasets_dir = Path(tmp) / "datasets"
            datasets_dir.mkdir(parents=True, exist_ok=True)
            (datasets_dir / "Episode01_dataset.json").write_text(json.dumps([{"scene_id": "scene_0001"}]), encoding="utf-8")
            cfg = {"paths": {"datasets_video_training": "datasets"}}

            with mock.patch.object(STEP06, "resolve_project_path", side_effect=lambda value: Path(tmp) / value):
                resolved = STEP06.pending_undataseted_episode_dirs(scene_root, cfg)

            self.assertEqual(resolved, [pending_a, pending_b])

    def test_episode_dataset_completed_requires_current_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_dir = root / "scene_clips" / "Episode01"
            scene_dir.mkdir(parents=True, exist_ok=True)
            datasets_dir = root / "datasets"
            datasets_dir.mkdir(parents=True, exist_ok=True)
            (datasets_dir / "Episode01_dataset.json").write_text(
                json.dumps([{"scene_id": "scene_0001"}]),
                encoding="utf-8",
            )
            (datasets_dir / "Episode01_dataset_manifest.json").write_text(
                json.dumps({"process_version": 0, "scene_count": 1}),
                encoding="utf-8",
            )
            cfg = {"paths": {"datasets_video_training": "datasets"}}

            with mock.patch.object(STEP06, "resolve_project_path", side_effect=lambda value: root / value):
                self.assertFalse(STEP06.episode_dataset_completed(scene_dir, cfg))

    def test_foundation_training_status_detects_outdated_training(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(ROOT / "ai_series_project" / "tmp"), ignore_cleanup_errors=True) as tmp:
            temp_root = Path(tmp)
            checkpoints_rel = Path("tmp") / temp_root.name / "checkpoints"
            model_rel = Path("tmp") / temp_root.name / "series_model.json"
            checkpoints_dir = ROOT / "ai_series_project" / checkpoints_rel
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            summary_path = checkpoints_dir / "foundation_training_summary.json"
            model_path = ROOT / "ai_series_project" / model_rel
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_text("{}", encoding="utf-8")
            summary_path.write_text(json.dumps({"characters": []}), encoding="utf-8")
            old_time = model_path.stat().st_mtime - 30
            os.utime(summary_path, (old_time, old_time))

            cfg = {
                "paths": {
                    "foundation_checkpoints": checkpoints_rel.as_posix(),
                    "series_model": model_rel.as_posix(),
                },
                "foundation_training": {"required_before_generate": True, "required_before_render": True},
                "cloning": {"require_trained_voice_models": True},
            }

            status = foundation_training_status(cfg)

            self.assertTrue(status["summary_exists"])
            self.assertFalse(status["summary_new_enough"])

    def test_ensure_foundation_training_ready_requires_character_packs(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(ROOT / "ai_series_project" / "tmp"), ignore_cleanup_errors=True) as tmp:
            temp_root = Path(tmp)
            checkpoints_rel = Path("tmp") / temp_root.name / "checkpoints"
            model_rel = Path("tmp") / temp_root.name / "series_model.json"
            checkpoints_dir = ROOT / "ai_series_project" / checkpoints_rel
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            pack_path = checkpoints_dir / "babe" / "foundation_pack.json"
            pack_path.parent.mkdir(parents=True, exist_ok=True)
            pack_path.write_text(
                json.dumps({"voice_pack": {"sample_count": 2}, "video_pack": {"sample_count": 3}, "image_pack": {"sample_count": 4}}),
                encoding="utf-8",
            )
            summary_path = checkpoints_dir / "foundation_training_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "characters": [
                            {
                                "character": "Babe",
                                "pack_path": str(pack_path),
                                "voice_samples": 2,
                                "video_samples": 3,
                                "frame_samples": 4,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            model_path = ROOT / "ai_series_project" / model_rel
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_text("{}", encoding="utf-8")
            newer_time = summary_path.stat().st_mtime + 30
            os.utime(model_path, (newer_time - 60, newer_time - 60))
            os.utime(summary_path, (newer_time, newer_time))

            cfg = {
                "paths": {
                    "foundation_checkpoints": checkpoints_rel.as_posix(),
                    "series_model": model_rel.as_posix(),
                },
                "foundation_training": {"required_before_generate": True, "required_before_render": True},
                "cloning": {"require_trained_voice_models": True},
            }

            ensure_foundation_training_ready(cfg, characters=["Babe"])
            with self.assertRaises(RuntimeError):
                ensure_foundation_training_ready(cfg, characters=["Kenzie"], for_render=True)

    def test_ensure_foundation_training_ready_requires_clone_ready_only_for_render(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(ROOT / "ai_series_project" / "tmp"), ignore_cleanup_errors=True) as tmp:
            temp_root = Path(tmp)
            checkpoints_rel = Path("tmp") / temp_root.name / "checkpoints"
            model_rel = Path("tmp") / temp_root.name / "series_model.json"
            checkpoints_dir = ROOT / "ai_series_project" / checkpoints_rel
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            pack_path = checkpoints_dir / "babe" / "foundation_pack.json"
            pack_path.parent.mkdir(parents=True, exist_ok=True)
            pack_path.write_text(
                json.dumps(
                    {
                        "voice_pack": {"sample_count": 3, "clone_ready": False},
                        "video_pack": {"sample_count": 3},
                        "image_pack": {"sample_count": 4},
                    }
                ),
                encoding="utf-8",
            )
            summary_path = checkpoints_dir / "foundation_training_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "characters": [
                            {
                                "character": "Babe",
                                "pack_path": str(pack_path),
                                "voice_samples": 3,
                                "video_samples": 3,
                                "frame_samples": 4,
                                "voice_clone_ready": False,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            model_path = ROOT / "ai_series_project" / model_rel
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_text("{}", encoding="utf-8")
            newer_time = summary_path.stat().st_mtime + 30
            os.utime(model_path, (newer_time - 60, newer_time - 60))
            os.utime(summary_path, (newer_time, newer_time))

            cfg = {
                "paths": {
                    "foundation_checkpoints": checkpoints_rel.as_posix(),
                    "series_model": model_rel.as_posix(),
                },
                "foundation_training": {"required_before_generate": True, "required_before_render": True},
                "cloning": {"require_trained_voice_models": True},
            }

            ensure_foundation_training_ready(cfg, characters=["Babe"])
            with self.assertRaises(RuntimeError):
                ensure_foundation_training_ready(cfg, characters=["Babe"], for_render=True)

    def test_adapter_training_status_detects_outdated_training(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(ROOT / "ai_series_project" / "tmp"), ignore_cleanup_errors=True) as tmp:
            temp_root = Path(tmp)
            adapter_rel = Path("tmp") / temp_root.name / "adapters"
            checkpoints_rel = Path("tmp") / temp_root.name / "checkpoints"
            adapter_dir = ROOT / "ai_series_project" / adapter_rel
            checkpoint_dir = ROOT / "ai_series_project" / checkpoints_rel
            adapter_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            foundation_summary = checkpoint_dir / "foundation_training_summary.json"
            adapter_summary = adapter_dir / "adapter_training_summary.json"
            foundation_summary.write_text(json.dumps({"characters": []}), encoding="utf-8")
            adapter_summary.write_text(json.dumps({"characters": []}), encoding="utf-8")
            old_time = foundation_summary.stat().st_mtime - 30
            os.utime(adapter_summary, (old_time, old_time))

            cfg = {
                "paths": {
                    "foundation_checkpoints": checkpoints_rel.as_posix(),
                    "foundation_adapters": adapter_rel.as_posix(),
                },
                "adapter_training": {"required_before_generate": False, "required_before_render": False},
            }

            status = adapter_training_status(cfg)

            self.assertTrue(status["summary_exists"])
            self.assertFalse(status["summary_new_enough"])

    def test_build_image_adapter_marks_ready_with_enough_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            sample_path = temp_root / "sample.png"
            image = Image.new("RGB", (32, 32), color=(120, 80, 40))
            image.save(sample_path)
            manifest = {
                "frame_samples": [{"path": str(sample_path)} for _ in range(8)],
            }
            cfg = {"adapter_training": {"min_image_samples": 8, "image_histogram_bins": 16, "image_thumbnail_size": 32}}

            payload = STEP17.build_image_adapter(manifest, cfg)

            self.assertEqual(payload["sample_count"], 8)
            self.assertTrue(payload["ready"])
            self.assertGreater(payload["feature_dim"], 0)

    def test_adapter_train_parse_args_supports_force(self) -> None:
        with mock.patch("sys.argv", ["11_train_adapter_models.py", "--character", "Babe", "--force"]):
            args = STEP17.parse_args()

        self.assertEqual(args.character, "Babe")
        self.assertTrue(args.force)

    def test_global_steps_to_run_includes_adapter_step_when_enabled(self) -> None:
        cfg = {
            "foundation_training": {
                "prepare_after_batch": True,
                "auto_train_after_prepare": True,
                "required_before_generate": True,
                "required_before_render": True,
            },
            "adapter_training": {"auto_train_after_foundation": True},
        }

        steps = STEP99.global_steps_to_run(cfg)

        self.assertIn("11_train_adapter_models.py", steps)
        self.assertLess(steps.index("10_train_foundation_models.py"), steps.index("11_train_adapter_models.py"))
        self.assertLess(steps.index("11_train_adapter_models.py"), steps.index("14_generate_episode_from_trained_model.py"))
        self.assertLess(steps.index("14_generate_episode_from_trained_model.py"), steps.index("15_generate_storyboard_assets.py"))
        self.assertLess(steps.index("15_generate_storyboard_assets.py"), steps.index("17_render_episode.py"))
        self.assertLess(steps.index("17_render_episode.py"), steps.index("18_build_series_bible.py"))

    def test_planned_preview_steps_builds_bible_once_after_batch(self) -> None:
        cfg = {
            "foundation_training": {"required_before_generate": False, "required_before_render": False},
            "adapter_training": {"auto_train_after_foundation": True},
            "fine_tune_training": {"auto_train_after_adapter": True},
            "backend_fine_tune": {"auto_run_after_fine_tune": True},
        }

        steps = STEP19PREVIEW.planned_preview_steps(cfg, 3)

        self.assertEqual(steps.count("14_generate_episode_from_trained_model.py"), 3)
        self.assertEqual(steps.count("17_render_episode.py"), 3)
        self.assertEqual(steps.count("18_build_series_bible.py"), 1)
        self.assertEqual(steps[-1], "18_build_series_bible.py")

    def test_planned_preview_steps_respects_prepare_without_forced_training(self) -> None:
        cfg = {
            "foundation_training": {
                "prepare_after_batch": True,
                "auto_train_after_prepare": False,
                "required_before_generate": False,
                "required_before_render": False,
            },
            "adapter_training": {"auto_train_after_foundation": True},
            "fine_tune_training": {"auto_train_after_adapter": True},
            "backend_fine_tune": {"auto_run_after_fine_tune": True},
        }

        steps = STEP19PREVIEW.planned_preview_steps(cfg, 1)

        self.assertIn("09_prepare_foundation_training.py", steps)
        self.assertNotIn("10_train_foundation_models.py", steps)
        self.assertNotIn("11_train_adapter_models.py", steps)
        self.assertNotIn("12_train_fine_tune_models.py", steps)
        self.assertNotIn("13_run_backend_finetunes.py", steps)
        self.assertLess(steps.index("09_prepare_foundation_training.py"), steps.index("14_generate_episode_from_trained_model.py"))

    def test_planned_preview_steps_skips_nested_training_when_foundation_train_disabled(self) -> None:
        cfg = {
            "foundation_training": {
                "prepare_after_batch": True,
                "auto_train_after_prepare": False,
                "required_before_generate": False,
                "required_before_render": False,
            },
            "adapter_training": {"auto_train_after_foundation": True},
            "fine_tune_training": {"auto_train_after_adapter": True},
            "backend_fine_tune": {"auto_run_after_fine_tune": True},
        }

        flags = STEP19PREVIEW.training_stage_flags(cfg)
        self.assertEqual(flags, (True, False, True, True, True))

        steps = STEP19PREVIEW.planned_preview_steps(cfg, 2)
        self.assertEqual(steps.count("09_prepare_foundation_training.py"), 1)
        self.assertEqual(steps.count("10_train_foundation_models.py"), 0)
        self.assertEqual(steps.count("11_train_adapter_models.py"), 0)
        self.assertEqual(steps.count("12_train_fine_tune_models.py"), 0)
        self.assertEqual(steps.count("13_run_backend_finetunes.py"), 0)
        self.assertEqual(steps.count("14_generate_episode_from_trained_model.py"), 2)

    def test_preview_parse_args_supports_skip_downloads(self) -> None:
        with mock.patch("sys.argv", ["19_generate_finished_episodes.py", "--count", "4", "--skip-downloads"]):
            args = STEP19PREVIEW.parse_args()

        self.assertEqual(args.count, 4)
        self.assertTrue(args.skip_downloads)
        self.assertFalse(args.endless)

    def test_preview_parse_args_defaults_to_one_finished_episode(self) -> None:
        with mock.patch("sys.argv", ["19_generate_finished_episodes.py"]):
            args = STEP19PREVIEW.parse_args()

        self.assertEqual(args.count, 1)
        self.assertFalse(args.endless)
        self.assertFalse(STEP19PREVIEW.preview_endless_mode(args))

    def test_preview_parse_args_count_zero_enables_endless_mode(self) -> None:
        with mock.patch("sys.argv", ["19_generate_finished_episodes.py", "--count", "0"]):
            args = STEP19PREVIEW.parse_args()

        self.assertEqual(args.count, 0)
        self.assertFalse(args.endless)
        self.assertTrue(STEP19PREVIEW.preview_endless_mode(args))

    def test_preview_parse_args_supports_explicit_endless_flag(self) -> None:
        with mock.patch("sys.argv", ["19_generate_finished_episodes.py", "--count", "3", "--endless"]):
            args = STEP19PREVIEW.parse_args()

        self.assertEqual(args.count, 3)
        self.assertTrue(args.endless)
        self.assertTrue(STEP19PREVIEW.preview_endless_mode(args))

    def test_preview_main_blocks_only_on_face_review_cases(self) -> None:
        reporter = mock.Mock()
        args = argparse.Namespace(count=1, endless=False, skip_downloads=False)

        with mock.patch.object(STEP19PREVIEW, "rerun_in_runtime"), mock.patch.object(
            STEP19PREVIEW, "parse_args", return_value=args
        ), mock.patch.object(
            STEP19PREVIEW, "load_config", return_value={}
        ), mock.patch.object(
            STEP19PREVIEW, "training_plan_rows", return_value=[]
        ), mock.patch.object(
            STEP19PREVIEW, "planned_preview_steps",
            return_value=[
                "14_generate_episode_from_trained_model.py",
                "15_generate_storyboard_assets.py",
                "16_run_storyboard_backend.py",
                "17_render_episode.py",
                "18_build_series_bible.py",
            ],
        ), mock.patch.object(
            STEP19PREVIEW, "open_face_review_item_count", return_value=0
        ), mock.patch.object(
            STEP19PREVIEW, "latest_episode_id", side_effect=[None, "episode_001"]
        ), mock.patch.object(
            STEP19PREVIEW, "generated_episode_artifacts",
            return_value={
                "episode_id": "episode_001",
                "final_render": "C:\\demo\\episode_001_final.mp4",
                "full_generated_episode": "C:\\demo\\episode_001_full_generated_episode.mp4",
                "production_package": "C:\\demo\\episode_001_production_package.json",
                "render_manifest": "C:\\demo\\episode_001_render_manifest.json",
            },
        ), mock.patch.object(
            STEP19PREVIEW, "run_step"
        ), mock.patch.object(
            STEP19PREVIEW, "mark_step_started"
        ), mock.patch.object(
            STEP19PREVIEW, "mark_step_completed"
        ) as mark_completed, mock.patch.object(
            STEP19PREVIEW, "mark_step_failed"
        ), mock.patch.object(
            STEP19PREVIEW, "LiveProgressReporter", return_value=reporter
        ), mock.patch.object(
            STEP19PREVIEW, "headline"
        ), mock.patch.object(
            STEP19PREVIEW, "ok"
        ):
            STEP19PREVIEW.main()

        completed_payload = mark_completed.call_args.args[2]
        self.assertEqual(completed_payload["generated_episodes"], ["episode_001"])
        self.assertEqual(completed_payload["generated_episode_outputs"][0]["episode_id"], "episode_001")
        self.assertTrue(
            completed_payload["latest_generated_episode"]["full_generated_episode"].endswith(
                "episode_001_full_generated_episode.mp4"
            )
        )

    def test_planned_preview_steps_for_endless_mode_include_one_generation_cycle_and_bible(self) -> None:
        cfg = {}

        steps = STEP19PREVIEW.planned_preview_steps_for_mode(cfg, 0, endless=True)

        self.assertEqual(steps.count("14_generate_episode_from_trained_model.py"), 1)
        self.assertEqual(steps.count("15_generate_storyboard_assets.py"), 1)
        self.assertEqual(steps.count("16_run_storyboard_backend.py"), 1)
        self.assertEqual(steps.count("17_render_episode.py"), 1)
        self.assertEqual(steps.count("18_build_series_bible.py"), 1)

    def test_preview_parent_label_marks_endless_runs(self) -> None:
        self.assertEqual(STEP19PREVIEW.preview_parent_label(0, True), "Endless")
        self.assertEqual(STEP19PREVIEW.preview_parent_label(4, False), "Count: 4")

    def test_training_plan_rows_add_skip_downloads_only_for_prepare_step(self) -> None:
        cfg = {
            "foundation_training": {
                "prepare_after_batch": True,
                "auto_train_after_prepare": True,
                "required_before_generate": False,
                "required_before_render": False,
            },
            "adapter_training": {"auto_train_after_foundation": True},
            "fine_tune_training": {"auto_train_after_adapter": True},
            "backend_fine_tune": {"auto_run_after_fine_tune": True},
        }

        rows = STEP19PREVIEW.training_plan_rows(cfg, skip_downloads=True)
        row_map = {script_name: args for script_name, _label, args in rows}

        self.assertEqual(row_map.get("09_prepare_foundation_training.py"), ["--skip-downloads"])
        self.assertEqual(row_map.get("10_train_foundation_models.py"), [])
        self.assertEqual(row_map.get("11_train_adapter_models.py"), [])

    def test_full_pipeline_parse_args_supports_skip_downloads(self) -> None:
        with mock.patch("sys.argv", ["99_process_next_episode.py", "--skip-downloads"]):
            args = STEP99.parse_args()

        self.assertTrue(args.skip_downloads)

    def test_global_step_rows_add_skip_downloads_only_for_prepare_step(self) -> None:
        cfg = {
            "foundation_training": {
                "prepare_after_batch": True,
                "auto_train_after_prepare": True,
                "required_before_generate": False,
                "required_before_render": False,
            },
            "adapter_training": {"auto_train_after_foundation": True},
            "fine_tune_training": {"auto_train_after_adapter": True},
            "backend_fine_tune": {"auto_run_after_fine_tune": True},
        }

        rows = STEP99.global_step_rows(cfg, skip_downloads=True)
        row_map = {script_name: args for script_name, _label, args in rows}

        self.assertEqual(row_map.get("09_prepare_foundation_training.py"), ["--skip-downloads"])
        self.assertEqual(row_map.get("10_train_foundation_models.py"), [])
        self.assertEqual(row_map.get("11_train_adapter_models.py"), [])

    def test_render_status_markdown_includes_generated_episode_readiness_summary(self) -> None:
        snapshot = {
            "status": "running",
            "updated_at": "2026-04-18T12:00:00Z",
            "autosave_reason": "global:17_render_episode.py",
            "setup_completed": True,
            "skip_downloads": True,
            "processed_count": 3,
            "pending_inbox_count": 0,
            "current_phase": "global",
            "current_episode_name": None,
            "current_step": "17_render_episode.py",
            "episode_progress": [],
            "global_progress": [],
            "latest_generated_episode": {
                "episode_id": "episode_301",
                "display_title": "Backend Progress",
                "render_mode": "hybrid_generated",
                "production_readiness": "hybrid_generated_episode",
                "scene_count": 6,
                "scene_video_completion_ratio": 0.5,
                "scene_dialogue_completion_ratio": 1.0,
                "scene_master_completion_ratio": 0.3333,
                "remaining_backend_tasks": [
                    "generate missing scene videos",
                    "master remaining scene clips",
                ],
                "final_render": "C:\\demo\\episode_301.mp4",
                "full_generated_episode": "",
                "production_package": "C:\\demo\\episode_301_production_package.json",
                "render_manifest": "C:\\demo\\episode_301_render_manifest.json",
            },
        }

        markdown = STEP99.render_status_markdown(snapshot)

        self.assertIn("- Production readiness: hybrid_generated_episode", markdown)
        self.assertIn("- Scene video coverage: 50.0%", markdown)
        self.assertIn("- Scene dialogue coverage: 100.0%", markdown)
        self.assertIn("- Scene master coverage: 33.3%", markdown)
        self.assertIn(
            "- Remaining backend tasks: generate missing scene videos, master remaining scene clips",
            markdown,
        )

    def test_completed_preview_step_labels_limits_to_completed_count(self) -> None:
        planned_steps = [
            "07_build_dataset.py",
            "08_train_series_model.py",
            "14_generate_episode_from_trained_model.py",
            "18_build_series_bible.py",
        ]

        completed = STEP19PREVIEW.completed_step_labels(planned_steps, 2)

        self.assertEqual(completed, ["07_build_dataset.py", "08_train_series_model.py"])

    def test_fine_tune_training_status_detects_outdated_training(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(ROOT / "ai_series_project" / "tmp")) as tmp:
            temp_root = Path(tmp)
            adapter_rel = Path("tmp") / temp_root.name / "adapters"
            finetune_rel = Path("tmp") / temp_root.name / "finetunes"
            adapter_dir = ROOT / "ai_series_project" / adapter_rel
            finetune_dir = ROOT / "ai_series_project" / finetune_rel
            adapter_dir.mkdir(parents=True, exist_ok=True)
            finetune_dir.mkdir(parents=True, exist_ok=True)
            adapter_summary = adapter_dir / "adapter_training_summary.json"
            finetune_summary = finetune_dir / "fine_tune_training_summary.json"
            adapter_summary.write_text(json.dumps({"characters": []}), encoding="utf-8")
            finetune_summary.write_text(json.dumps({"characters": []}), encoding="utf-8")
            old_time = adapter_summary.stat().st_mtime - 30
            os.utime(finetune_summary, (old_time, old_time))

            cfg = {
                "paths": {
                    "foundation_adapters": adapter_rel.as_posix(),
                    "foundation_finetunes": finetune_rel.as_posix(),
                },
                "fine_tune_training": {"required_before_generate": False, "required_before_render": False},
            }

            status = fine_tune_training_status(cfg)

            self.assertTrue(status["summary_exists"])
            self.assertFalse(status["summary_new_enough"])

    def test_fine_tune_parse_args_supports_force(self) -> None:
        with mock.patch("sys.argv", ["12_train_fine_tune_models.py", "--character", "Babe", "--force"]):
            args = STEP18.parse_args()

        self.assertEqual(args.character, "Babe")
        self.assertTrue(args.force)

    def test_build_fine_tune_profile_marks_ready_with_modalities(self) -> None:
        adapter_row = {
            "character": "Babe",
            "profile_path": "C:\\demo\\adapter_profile.json",
            "modalities_ready": ["image", "video", "voice"],
            "voice_quality_score": 0.72,
            "voice_duration_seconds": 14.5,
            "voice_clone_ready": True,
        }
        adapter_payload = {"slug": "babe", "priority": True}
        cfg = {
            "fine_tune_training": {
                "min_modalities_ready": 1,
                "target_steps_image": 100,
                "target_steps_video": 50,
                "target_steps_voice": 25,
            }
        }

        payload = STEP18.build_fine_tune_profile(adapter_row, adapter_payload, cfg)

        self.assertTrue(payload["training_ready"])
        self.assertEqual(payload["target_steps"]["image"], 100)
        self.assertEqual(payload["target_steps"]["video"], 50)
        self.assertTrue(payload["voice_clone_ready"])
        self.assertEqual(payload["voice_quality_score"], 0.72)

    def test_adapter_training_status_requires_clone_ready_only_for_render(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(ROOT / "ai_series_project" / "tmp"), ignore_cleanup_errors=True) as tmp:
            temp_root = Path(tmp)
            adapter_rel = Path("tmp") / temp_root.name / "adapters"
            checkpoints_rel = Path("tmp") / temp_root.name / "checkpoints"
            adapter_dir = ROOT / "ai_series_project" / adapter_rel
            checkpoint_dir = ROOT / "ai_series_project" / checkpoints_rel
            adapter_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            foundation_summary = checkpoint_dir / "foundation_training_summary.json"
            foundation_summary.write_text(json.dumps({"characters": []}), encoding="utf-8")
            profile_path = adapter_dir / "babe" / "adapter_profile.json"
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profile_path.write_text(
                json.dumps(
                    {
                        "training_ready": True,
                        "modalities_ready": ["voice"],
                        "modalities": {"voice": {"sample_count": 4, "clone_ready": False}},
                    }
                ),
                encoding="utf-8",
            )
            summary_path = adapter_dir / "adapter_training_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "characters": [
                            {
                                "character": "Babe",
                                "profile_path": str(profile_path),
                                "training_ready": True,
                                "modalities_ready": ["voice"],
                                "voice_samples": 4,
                                "voice_clone_ready": False,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            newer_time = summary_path.stat().st_mtime + 30
            os.utime(foundation_summary, (newer_time - 60, newer_time - 60))
            os.utime(summary_path, (newer_time, newer_time))
            cfg = {
                "paths": {
                    "foundation_checkpoints": checkpoints_rel.as_posix(),
                    "foundation_adapters": adapter_rel.as_posix(),
                },
                "adapter_training": {"required_before_generate": True, "required_before_render": True},
            }

            status_generate = adapter_training_status(cfg, characters=["Babe"])
            status_render = adapter_training_status(cfg, characters=["Babe"], require_voice_clone=True)

            self.assertEqual(status_generate["weak_characters"], [])
            self.assertEqual(status_render["weak_characters"], ["Babe"])

    def test_global_steps_to_run_includes_fine_tune_step_when_enabled(self) -> None:
        cfg = {
            "foundation_training": {
                "prepare_after_batch": True,
                "auto_train_after_prepare": True,
                "required_before_generate": True,
                "required_before_render": True,
            },
            "adapter_training": {"auto_train_after_foundation": True},
            "fine_tune_training": {"auto_train_after_adapter": True},
        }

        steps = STEP99.global_steps_to_run(cfg)

        self.assertIn("12_train_fine_tune_models.py", steps)
        self.assertLess(steps.index("11_train_adapter_models.py"), steps.index("12_train_fine_tune_models.py"))
        self.assertLess(steps.index("12_train_fine_tune_models.py"), steps.index("14_generate_episode_from_trained_model.py"))

    def test_backend_fine_tune_status_detects_outdated_runs(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(ROOT / "ai_series_project" / "tmp")) as tmp:
            temp_root = Path(tmp)
            fine_rel = Path("tmp") / temp_root.name / "finetunes"
            backend_rel = Path("tmp") / temp_root.name / "backend"
            fine_dir = ROOT / "ai_series_project" / fine_rel
            backend_dir = ROOT / "ai_series_project" / backend_rel
            fine_dir.mkdir(parents=True, exist_ok=True)
            backend_dir.mkdir(parents=True, exist_ok=True)
            fine_summary = fine_dir / "fine_tune_training_summary.json"
            backend_summary = backend_dir / "backend_fine_tune_summary.json"
            fine_summary.write_text(json.dumps({"characters": []}), encoding="utf-8")
            backend_summary.write_text(json.dumps({"characters": []}), encoding="utf-8")
            old_time = fine_summary.stat().st_mtime - 30
            os.utime(backend_summary, (old_time, old_time))

            cfg = {
                "paths": {
                    "foundation_finetunes": fine_rel.as_posix(),
                    "foundation_backend_runs": backend_rel.as_posix(),
                },
                "backend_fine_tune": {"required_before_generate": False, "required_before_render": False},
            }

            status = backend_fine_tune_status(cfg)

            self.assertTrue(status["summary_exists"])
            self.assertFalse(status["summary_new_enough"])

    def test_backend_parse_args_supports_force(self) -> None:
        with mock.patch("sys.argv", ["13_run_backend_finetunes.py", "--character", "Babe", "--force"]):
            args = STEP19.parse_args()

        self.assertEqual(args.character, "Babe")
        self.assertTrue(args.force)

    def test_build_backend_run_profile_maps_modalities_to_backends(self) -> None:
        row = {
            "character": "Babe",
            "fine_tune_path": "C:\\demo\\fine_tune_profile.json",
            "modalities_ready": ["image", "video", "voice"],
            "target_steps": {"image": 100, "video": 50, "voice": 25},
            "completed_steps": {"image": 100, "video": 50, "voice": 25},
            "voice_quality_score": 0.77,
            "voice_duration_seconds": 16.0,
            "voice_clone_ready": True,
        }
        fine_payload = {"slug": "babe", "priority": True}
        cfg = {
            "backend_fine_tune": {
                "image_backend": "lora-image",
                "video_backend": "motion-adapter",
                "voice_backend": "speaker-adapter",
            },
            "paths": {"foundation_backend_runs": "tmp/backend_runs_profile"},
        }
        with tempfile.TemporaryDirectory(dir=str(ROOT / "ai_series_project" / "tmp"), ignore_cleanup_errors=True) as tmp:
            temp_root = Path(tmp)
            backend_rel = Path("tmp") / temp_root.name / "backend_runs_profile"
            cfg["paths"]["foundation_backend_runs"] = backend_rel.as_posix()

            payload = STEP19.build_backend_run_profile(row, fine_payload, cfg)

            self.assertTrue(payload["training_ready"])
            self.assertEqual(payload["backends"]["image"]["backend"], "lora-image")
            self.assertEqual(payload["backends"]["video"]["backend"], "motion-adapter")
            self.assertEqual(payload["backends"]["voice"]["backend"], "speaker-adapter")
            self.assertTrue(payload["backends"]["voice"]["voice_clone_ready"])
            self.assertTrue(Path(payload["backends"]["image"]["artifacts"]["job_path"]).exists())
            self.assertTrue(Path(payload["backends"]["image"]["artifacts"]["bundle_path"]).exists())
            self.assertTrue(Path(payload["backends"]["image"]["artifacts"]["weights_path"]).exists())

    def test_backend_fine_tune_status_marks_missing_artifacts_as_weak(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(ROOT / "ai_series_project" / "tmp")) as tmp:
            temp_root = Path(tmp)
            fine_rel = Path("tmp") / temp_root.name / "finetunes"
            backend_rel = Path("tmp") / temp_root.name / "backend"
            fine_dir = ROOT / "ai_series_project" / fine_rel
            backend_dir = ROOT / "ai_series_project" / backend_rel
            fine_dir.mkdir(parents=True, exist_ok=True)
            backend_dir.mkdir(parents=True, exist_ok=True)

            fine_summary = fine_dir / "fine_tune_training_summary.json"
            backend_summary = backend_dir / "backend_fine_tune_summary.json"
            fine_summary.write_text(json.dumps({"characters": []}), encoding="utf-8")

            backend_run = backend_dir / "babe" / "backend_fine_tune_run.json"
            backend_run.parent.mkdir(parents=True, exist_ok=True)
            backend_run.write_text(
                json.dumps(
                    {
                        "process_version": STEP19.PROCESS_VERSION,
                        "character": "Babe",
                        "training_ready": True,
                        "modalities_ready": ["image"],
                        "backends": {
                            "image": {
                                "backend": "lora-image",
                                "ready": True,
                                "artifacts": {
                                    "job_path": str(backend_run.parent / "image" / "training_job.json"),
                                    "bundle_path": str(backend_run.parent / "image" / "model_bundle.json"),
                                    "weights_path": str(backend_run.parent / "image" / "image_weights.bin"),
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            backend_summary.write_text(
                json.dumps(
                    {
                        "characters": [
                            {
                                "character": "Babe",
                                "backend_run_path": str(backend_run),
                                "training_ready": True,
                                "modalities_ready": ["image"],
                                "backends": {
                                    "image": {
                                        "backend": "lora-image",
                                        "ready": True,
                                        "artifacts": {
                                            "job_path": str(backend_run.parent / "image" / "training_job.json"),
                                            "bundle_path": str(backend_run.parent / "image" / "model_bundle.json"),
                                            "weights_path": str(backend_run.parent / "image" / "image_weights.bin"),
                                        },
                                    }
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            cfg = {
                "paths": {
                    "foundation_finetunes": fine_rel.as_posix(),
                    "foundation_backend_runs": backend_rel.as_posix(),
                },
                "backend_fine_tune": {"required_before_generate": False, "required_before_render": False},
            }

            status = backend_fine_tune_status(cfg, characters=["Babe"])

            self.assertIn("Babe", status["weak_characters"])

    def test_backend_fine_tune_status_rebases_old_artifact_paths_after_workspace_move(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(ROOT / "ai_series_project" / "tmp")) as tmp:
            temp_root = Path(tmp)
            fine_rel = Path("tmp") / temp_root.name / "finetunes"
            backend_rel = Path("tmp") / temp_root.name / "backend"
            fine_dir = ROOT / "ai_series_project" / fine_rel
            backend_dir = ROOT / "ai_series_project" / backend_rel
            fine_dir.mkdir(parents=True, exist_ok=True)
            backend_dir.mkdir(parents=True, exist_ok=True)

            fine_summary = fine_dir / "fine_tune_training_summary.json"
            backend_summary = backend_dir / "backend_fine_tune_summary.json"
            fine_summary.write_text(json.dumps({"characters": []}), encoding="utf-8")

            artifact_dir = backend_dir / "babe" / "image"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "training_job.json").write_text("{}", encoding="utf-8")
            (artifact_dir / "model_bundle.json").write_text("{}", encoding="utf-8")
            (artifact_dir / "image_weights.bin").write_bytes(b"demo")

            old_prefix = Path("C:/Old/Workspace/KI Serien Training/ai_series_project")
            backend_run = backend_dir / "babe" / "backend_fine_tune_run.json"
            backend_run.write_text(
                json.dumps(
                    {
                        "process_version": STEP19.PROCESS_VERSION,
                        "character": "Babe",
                        "training_ready": True,
                        "modalities_ready": ["image"],
                        "backends": {
                            "image": {
                                "backend": "lora-image",
                                "ready": True,
                                "artifacts": {
                                    "job_path": str(old_prefix / "tmp" / temp_root.name / "backend" / "babe" / "image" / "training_job.json"),
                                    "bundle_path": str(old_prefix / "tmp" / temp_root.name / "backend" / "babe" / "image" / "model_bundle.json"),
                                    "weights_path": str(old_prefix / "tmp" / temp_root.name / "backend" / "babe" / "image" / "image_weights.bin"),
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            backend_summary.write_text(
                json.dumps(
                    {
                        "characters": [
                            {
                                "character": "Babe",
                                "backend_run_path": str(old_prefix / "tmp" / temp_root.name / "backend" / "babe" / "backend_fine_tune_run.json"),
                                "training_ready": True,
                                "modalities_ready": ["image"],
                                "backends": {
                                    "image": {
                                        "backend": "lora-image",
                                        "ready": True,
                                        "artifacts": {
                                            "job_path": str(old_prefix / "tmp" / temp_root.name / "backend" / "babe" / "image" / "training_job.json"),
                                            "bundle_path": str(old_prefix / "tmp" / temp_root.name / "backend" / "babe" / "image" / "model_bundle.json"),
                                            "weights_path": str(old_prefix / "tmp" / temp_root.name / "backend" / "babe" / "image" / "image_weights.bin"),
                                        },
                                    }
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            cfg = {
                "paths": {
                    "foundation_finetunes": fine_rel.as_posix(),
                    "foundation_backend_runs": backend_rel.as_posix(),
                },
                "backend_fine_tune": {"required_before_generate": False, "required_before_render": False},
            }

            status = backend_fine_tune_status(cfg, characters=["Babe"])

            self.assertEqual(status["missing_characters"], [])
            self.assertEqual(status["weak_characters"], [])

    def test_global_steps_to_run_includes_backend_step_when_enabled(self) -> None:
        cfg = {
            "foundation_training": {
                "prepare_after_batch": True,
                "auto_train_after_prepare": True,
                "required_before_generate": True,
                "required_before_render": True,
            },
            "adapter_training": {"auto_train_after_foundation": True},
            "fine_tune_training": {"auto_train_after_adapter": True},
            "backend_fine_tune": {"auto_run_after_fine_tune": True},
        }

        steps = STEP99.global_steps_to_run(cfg)

        self.assertIn("13_run_backend_finetunes.py", steps)
        self.assertLess(steps.index("12_train_fine_tune_models.py"), steps.index("13_run_backend_finetunes.py"))
        self.assertLess(steps.index("13_run_backend_finetunes.py"), steps.index("14_generate_episode_from_trained_model.py"))

    def test_foundation_pack_completed_requires_ready_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir) / "foundation_pack.json"
            pack_path.write_text(
                json.dumps({"process_version": STEP10TRAIN.PROCESS_VERSION, "training_ready": True}),
                encoding="utf-8",
            )
            self.assertTrue(STEP10TRAIN.foundation_pack_completed(pack_path))
            pack_path.write_text(
                json.dumps({"process_version": 0, "training_ready": True}),
                encoding="utf-8",
            )
            self.assertFalse(STEP10TRAIN.foundation_pack_completed(pack_path))

    def test_render_output_ready_requires_nonempty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = Path(tmpdir) / "empty.mp4"
            full_path = Path(tmpdir) / "full.mp4"
            empty_path.write_bytes(b"")
            full_path.write_bytes(b"demo")
            self.assertFalse(STEP10.render_output_ready(empty_path))
            self.assertTrue(STEP10.render_output_ready(full_path))

    def test_render_and_bible_scripts_use_matching_step_metadata_names(self) -> None:
        render_source = (ROOT / "17_render_episode.py").read_text(encoding="utf-8")
        bible_source = (ROOT / "18_build_series_bible.py").read_text(encoding="utf-8")

        self.assertIn('mark_step_started("17_render_episode"', render_source)
        self.assertIn('mark_step_completed(\n            "17_render_episode"', render_source)
        self.assertIn('mark_step_failed("17_render_episode"', render_source)
        self.assertNotIn('mark_step_started("18_render_episode"', render_source)
        self.assertNotIn('mark_step_failed("18_render_episode"', render_source)

        self.assertIn('mark_step_started("18_build_series_bible"', bible_source)
        self.assertIn('mark_step_completed(\n                "18_build_series_bible"', bible_source)
        self.assertIn('mark_step_failed("18_build_series_bible"', bible_source)
        self.assertNotIn('mark_step_started("17_build_series_bible"', bible_source)
        self.assertNotIn('mark_step_failed("17_build_series_bible"', bible_source)


if __name__ == "__main__":
    unittest.main()
