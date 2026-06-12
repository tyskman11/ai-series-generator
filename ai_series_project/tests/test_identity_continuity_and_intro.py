from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


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


class IdentityContinuityAndIntroTests(unittest.TestCase):
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

    def test_finished_episode_defaults_require_fixed_intro(self) -> None:
        self.assertTrue(pipeline_common.DEFAULT_CONFIG["season_intro"]["enabled"])
        self.assertTrue(
            pipeline_common.DEFAULT_CONFIG["season_intro"]["require_in_finished_episode_mode"]
        )
        self.assertTrue(pipeline_common.DEFAULT_CONFIG["season_intro"]["auto_generate_if_missing"])

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


if __name__ == "__main__":
    unittest.main()
