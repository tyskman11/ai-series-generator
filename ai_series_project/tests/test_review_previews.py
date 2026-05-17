from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = PROJECT_DIR


def load_module(filename: str, module_name: str):
    target = ROOT / filename if filename.startswith("support_scripts/") else SCRIPT_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP06 = load_module("05_review_unknowns.py", "step06_review_previews")


class ReviewPreviewTests(unittest.TestCase):
    def test_preview_files_rebase_stored_preview_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_dir = Path(tmpdir)
            image_path = preview_dir / "face_001_crop.jpg"
            image_path.write_text("placeholder", encoding="utf-8")
            payload = {"preview_dir": "B:/old/workspace/ai_series_project/characters/previews/face_001"}

            with mock.patch.object(STEP06, "resolve_stored_project_path", return_value=preview_dir):
                self.assertEqual(STEP06.preview_files(payload), [image_path])

    def test_create_face_review_sheet_writes_pair_contact_sheet(self) -> None:
        try:
            from PIL import Image
        except Exception:
            self.skipTest("Pillow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            preview_dir = Path(tmpdir)
            context_path = preview_dir / "scene_001_context.jpg"
            crop_path = preview_dir / "scene_001_crop.jpg"
            Image.new("RGB", (120, 90), "navy").save(context_path)
            Image.new("RGB", (60, 60), "white").save(crop_path)

            sheet_path = STEP06.create_face_review_sheet("face_001", {"preview_dir": str(preview_dir)})

            self.assertIsNotNone(sheet_path)
            assert sheet_path is not None
            self.assertTrue(sheet_path.exists())
            self.assertGreater(sheet_path.stat().st_size, 0)

    def test_create_face_review_html_writes_browser_preview(self) -> None:
        try:
            from PIL import Image
        except Exception:
            self.skipTest("Pillow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            preview_dir = Path(tmpdir) / "previews"
            review_dir = Path(tmpdir) / "review"
            preview_dir.mkdir(parents=True, exist_ok=True)
            review_dir.mkdir(parents=True, exist_ok=True)
            context_path = preview_dir / "scene_001_context.jpg"
            crop_path = preview_dir / "scene_001_crop.jpg"
            Image.new("RGB", (120, 90), "navy").save(context_path)
            Image.new("RGB", (60, 60), "white").save(crop_path)

            with mock.patch.object(STEP06, "resolve_project_path", return_value=review_dir):
                html_path = STEP06.create_face_review_html("face_001", {"preview_dir": str(preview_dir)})

            self.assertIsNotNone(html_path)
            assert html_path is not None
            self.assertTrue(html_path.exists())
            html_text = html_path.read_text(encoding="utf-8")
            self.assertIn("face_001 Preview", html_text)
            self.assertIn("data:image/jpeg;base64,", html_text)

    def test_open_preview_file_uses_windows_default_viewer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_path = Path(tmpdir) / "preview.jpg"
            preview_path.write_text("placeholder", encoding="utf-8")

            with mock.patch.object(STEP06, "current_os", return_value="windows"), mock.patch.object(
                STEP06,
                "windows_shell_open",
                return_value=True,
            ) as shell_open, mock.patch.object(
                STEP06.subprocess,
                "Popen",
            ) as popen:
                self.assertTrue(STEP06.open_preview_file(preview_path))

            shell_open.assert_called_once_with(preview_path)
            popen.assert_not_called()

    def test_selected_preview_images_prefers_crop_before_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_dir = Path(tmpdir)
            context_path = preview_dir / "scene_001_context.jpg"
            crop_path = preview_dir / "scene_001_crop.jpg"
            context_path.write_text("context", encoding="utf-8")
            crop_path.write_text("crop", encoding="utf-8")

            targets = STEP06.selected_preview_images({"preview_dir": str(preview_dir)})

            self.assertEqual(targets, [crop_path, context_path])

    def test_preview_open_targets_prefers_montage_over_raw_crop(self) -> None:
        try:
            from PIL import Image
        except Exception:
            self.skipTest("Pillow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            preview_dir = Path(tmpdir)
            context_path = preview_dir / "scene_001_context.jpg"
            crop_path = preview_dir / "scene_001_crop.jpg"
            Image.new("RGB", (120, 90), "navy").save(context_path)
            Image.new("RGB", (60, 60), "white").save(crop_path)

            targets = STEP06.preview_open_targets("face_001", {"preview_dir": str(preview_dir)})

            self.assertEqual(len(targets), 1)
            self.assertEqual(targets[0].name, "face_001_montage.jpg")

    def test_preview_open_targets_falls_back_to_raw_images_when_montage_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_dir = Path(tmpdir)
            context_path = preview_dir / "scene_001_context.jpg"
            crop_path = preview_dir / "scene_001_crop.jpg"
            context_path.write_text("context", encoding="utf-8")
            crop_path.write_text("crop", encoding="utf-8")

            with mock.patch.object(STEP06, "create_face_review_sheet", side_effect=RuntimeError("broken image")):
                targets = STEP06.preview_open_targets("face_001", {"preview_dir": str(preview_dir)})

            self.assertEqual(targets, [crop_path])

    def test_materialize_local_preview_bundle_copies_images_for_local_gui(self) -> None:
        try:
            from PIL import Image
        except Exception:
            self.skipTest("Pillow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "nas_source"
            local_root = Path(tmpdir) / "local_cache"
            source_dir.mkdir(parents=True, exist_ok=True)
            local_root.mkdir(parents=True, exist_ok=True)
            crop_path = source_dir / "scene_001_crop.jpg"
            context_path = source_dir / "scene_001_context.jpg"
            Image.new("RGB", (64, 64), "white").save(crop_path)
            Image.new("RGB", (120, 80), "navy").save(context_path)

            with mock.patch.object(STEP06.tempfile, "gettempdir", return_value=str(local_root)):
                bundle = STEP06.materialize_local_preview_bundle("face_001", {"preview_dir": str(source_dir)})

            local_images = bundle.get("local_images", [])
            self.assertTrue(local_images)
            self.assertTrue(all(isinstance(path, Path) and path.exists() for path in local_images))
            self.assertTrue(all(str(path).startswith(str(local_root)) for path in local_images))
            preview_window_image = bundle.get("preview_window_image")
            self.assertIsInstance(preview_window_image, Path)
            assert isinstance(preview_window_image, Path)
            self.assertTrue(str(preview_window_image).startswith(str(local_root)))
            self.assertEqual(preview_window_image.name, "face_001_montage.jpg")
            open_targets = bundle.get("open_targets", [])
            self.assertTrue(open_targets)
            self.assertEqual(open_targets, [preview_window_image])

    def test_open_preview_targets_counts_each_opened_image(self) -> None:
        paths = [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]
        with mock.patch.object(STEP06, "open_preview_file", return_value=True) as open_file:
            self.assertEqual(STEP06.open_preview_targets(paths), 1)
        open_file.assert_called_once_with(paths[0])

    def test_terminal_clickable_path_returns_file_uri(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "scene_001_crop.jpg"
            image_path.write_text("crop", encoding="utf-8")
            uri = STEP06.terminal_clickable_path(image_path)
            self.assertTrue(uri.startswith("file:"))
            self.assertIn("scene_001_crop.jpg", uri)

    def test_review_previews_are_enabled_by_default_and_can_be_disabled(self) -> None:
        with mock.patch("sys.argv", ["05_review_unknowns.py"]):
            self.assertTrue(STEP06.parse_args().open_previews)

        with mock.patch("sys.argv", ["05_review_unknowns.py", "--no-open-previews"]):
            self.assertFalse(STEP06.parse_args().open_previews)

        with mock.patch("sys.argv", ["05_review_unknowns.py", "--offline"]):
            self.assertTrue(STEP06.parse_args().offline)

        with mock.patch("sys.argv", ["05_review_unknowns.py", "--no-internet-lookup"]):
            parsed = STEP06.parse_args()
            self.assertTrue(parsed.no_internet_lookup)
            self.assertFalse(parsed.offline)

        with mock.patch("sys.argv", ["05_review_unknowns.py", "--edit-names"]):
            self.assertTrue(STEP06.parse_args().edit_names)

        with mock.patch("sys.argv", ["05_review_unknowns.py", "--deep-online-face-lookup"]):
            self.assertTrue(STEP06.parse_args().deep_online_face_lookup)

        with mock.patch("sys.argv", ["05_review_unknowns.py", "--show-queue", "--queue-limit", "0"]):
            self.assertEqual(STEP06.parse_args().queue_limit, 0)

    def test_internet_face_lookup_allows_deep_public_image_lookup_by_default(self) -> None:
        self.assertTrue(STEP06.internet_lookup_config({"character_detection": {}})["face_lookup_public_image_allow_slow_torch"])

    def test_gui_preview_unavailable_on_headless_linux(self) -> None:
        with mock.patch.object(STEP06, "current_os", return_value="linux"), mock.patch.dict(
            os.environ,
            {"DISPLAY": "", "WAYLAND_DISPLAY": ""},
            clear=False,
        ):
            self.assertFalse(STEP06.gui_preview_available())

    def test_internet_lookup_completes_partial_character_name_and_keeps_alias(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {"name": "Babe", "embedding": [1.0, 0.0], "samples": 2},
                "face_002": {"name": "Babe", "embedding": [0.98, 0.02], "samples": 1},
            },
            "aliases": {},
        }
        cfg = {
            "character_detection": {
                "internet_name_lookup": True,
            }
        }

        with mock.patch.object(
            STEP06,
            "internet_name_candidates",
            return_value=[
                {
                    "label": "Babe Carano",
                    "confidence": 0.95,
                    "source": "wikidata",
                    "url": "https://www.wikidata.org/wiki/Q-test",
                }
            ],
        ):
            summary = STEP06.enrich_existing_character_names_from_internet(cfg, char_map)

        self.assertEqual(summary["renamed"], 1)
        self.assertEqual(char_map["clusters"]["face_001"]["name"], "Babe Carano")
        self.assertEqual(char_map["clusters"]["face_002"]["name"], "Babe Carano")
        self.assertIn("babe", char_map["clusters"]["face_001"]["aliases"])
        self.assertIn("babe", char_map["aliases"])

    def test_internet_lookup_does_not_rename_below_minimum_confidence(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {"name": "Triple G", "embedding": [1.0, 0.0], "samples": 2},
            },
            "aliases": {},
        }

        with mock.patch.object(
            STEP06,
            "internet_name_candidates",
            return_value=[
                {
                    "label": "Babe & Triple G",
                    "confidence": 0.94,
                    "source": "fandom",
                }
            ],
        ):
            summary = STEP06.enrich_existing_character_names_from_internet({"character_detection": {}}, char_map)

        self.assertEqual(summary["renamed"], 0)
        self.assertEqual(char_map["clusters"]["face_001"]["name"], "Triple G")

    def test_internet_lookup_renames_at_ninety_five_percent_confidence(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {"name": "Babe", "embedding": [1.0, 0.0], "samples": 2},
            },
            "aliases": {},
        }

        with mock.patch.object(
            STEP06,
            "internet_name_candidates",
            return_value=[
                {
                    "label": "Babe Carano",
                    "confidence": 0.95,
                    "source": "wikidata",
                }
            ],
        ):
            summary = STEP06.enrich_existing_character_names_from_internet({"character_detection": {}}, char_map)

        self.assertEqual(summary["renamed"], 1)
        self.assertEqual(char_map["clusters"]["face_001"]["name"], "Babe Carano")

    def test_low_confidence_internet_rename_is_rolled_back(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {
                    "name": "Babe & Triple G",
                    "internet_name_lookup": {
                        "previous_name": "Triple G",
                        "resolved_name": "Babe & Triple G",
                        "confidence": 0.74,
                    },
                },
            },
            "aliases": {},
        }

        summary = STEP06.rollback_low_confidence_internet_names({"character_detection": {}}, char_map)

        self.assertEqual(summary["restored"], 1)
        self.assertEqual(char_map["clusters"]["face_001"]["name"], "Triple G")

    def test_low_confidence_internet_history_is_cleared_when_already_restored(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {
                    "name": "Triple G",
                    "internet_name_lookup": {
                        "previous_name": "Triple G",
                        "resolved_name": "Babe & Triple G",
                        "confidence": 0.74,
                    },
                },
            },
            "aliases": {},
        }

        summary = STEP06.rollback_low_confidence_internet_names({"character_detection": {}}, char_map)

        self.assertEqual(summary["restored"], 0)
        self.assertEqual(summary["cleared_history"], 1)
        self.assertNotIn("internet_name_lookup", char_map["clusters"]["face_001"])
        self.assertEqual(char_map["clusters"]["face_001"]["internet_name_lookup_rejected"][0]["rejected_name"], "Babe & Triple G")

    def test_online_face_lookup_can_use_http_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "face_001_crop.jpg"
            image_path.write_bytes(b"fake-jpeg")

            class FakeResponse:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self) -> bytes:
                    return b'{"matches":[{"name":"Babe Carano","confidence":1.0,"source":"mock-api"}]}'

            captured: dict[str, str] = {}

            def fake_urlopen(request, timeout=0):
                captured["url"] = request.full_url
                captured["body"] = request.data.decode("utf-8")
                return FakeResponse()

            cfg = {"character_detection": {"internet_face_lookup_url": "https://lookup.example/api"}}
            with mock.patch.object(STEP06.urllib.request, "urlopen", side_effect=fake_urlopen):
                matches = STEP06.run_online_face_lookup_http(
                    "https://lookup.example/api",
                    image_path,
                    "face_001",
                    cfg,
                )

        self.assertEqual(matches[0]["label"], "Babe Carano")
        self.assertEqual(matches[0]["confidence"], 1.0)
        self.assertEqual(captured["url"], "https://lookup.example/api")
        self.assertIn("image_base64", captured["body"])

    def test_builtin_public_image_lookup_scores_local_embeddings_without_api_key(self) -> None:
        payload = {"embedding": [1.0, 0.0, 0.0]}
        bank = [
            {"label": "Babe Carano", "embedding": [0.99, 0.01, 0.0], "url": "https://example.test/babe.jpg"},
            {"label": "Kenzie Bell", "embedding": [0.1, 0.9, 0.0], "url": "https://example.test/kenzie.jpg"},
        ]
        cfg = {
            "character_detection": {
                "internet_face_lookup_public_image_min_similarity": 0.72,
                "internet_face_lookup_public_image_min_margin": 0.05,
            }
        }

        matches = STEP06.run_builtin_public_image_face_lookup(payload, bank, cfg)

        self.assertEqual(matches[0]["label"], "Babe Carano")
        self.assertEqual(matches[0]["confidence"], 1.0)
        self.assertEqual(matches[0]["source"], "builtin_public_image_embedding")

    def test_name_editor_rename_updates_face_and_speaker_names(self) -> None:
        char_map = {
            "clusters": {
                "face_001": {"name": "Babe", "priority": True},
                "face_002": {"name": "Babe"},
            },
            "aliases": {},
        }
        voice_map = {"clusters": {"speaker_001": {"name": "Babe", "auto_named": True}}, "aliases": {}}

        summary = STEP06.rename_name_everywhere(char_map, voice_map, "Babe", "Babe Carano", priority=True)

        self.assertEqual(summary, {"faces": 2, "speakers": 1})
        self.assertEqual(char_map["clusters"]["face_001"]["name"], "Babe Carano")
        self.assertEqual(char_map["clusters"]["face_002"]["name"], "Babe Carano")
        self.assertEqual(voice_map["clusters"]["speaker_001"]["name"], "Babe Carano")
        self.assertFalse(voice_map["clusters"]["speaker_001"]["auto_named"])

    def test_auto_match_known_faces_can_rescue_background_statist_clusters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            char_map = {
                "clusters": {
                    "face_known": {
                        "name": "Babe Carano",
                        "embedding": [1.0, 0.0, 0.0],
                        "samples": 4,
                        "scene_count": 5,
                        "detection_count": 20,
                    },
                    "face_statist": {
                        "name": "statist",
                        "embedding": [0.99, 0.01, 0.0],
                        "samples": 1,
                        "scene_count": 1,
                        "detection_count": 2,
                    },
                },
                "aliases": {},
            }
            voice_map = {"clusters": {}, "aliases": {}}
            cfg = {
                "paths": {"linked_segments": str(Path(tmpdir) / "linked")},
                "character_detection": {
                    "review_known_face_threshold": 0.70,
                    "review_known_face_margin": 0.02,
                    "review_known_face_min_consensus": 2,
                    "review_known_face_strong_match_threshold": 0.84,
                    "review_match_background_faces": True,
                },
            }

            summary = STEP06.auto_match_known_faces(cfg, char_map, voice_map, include_background=True)

        self.assertEqual(summary["matched"], 1)
        self.assertNotIn("face_statist", char_map["clusters"])
        self.assertIn("face_statist", char_map["clusters"]["face_known"]["merged_cluster_ids"])

    def test_speaker_assignment_uses_direct_speaker_face_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            linked_root = Path(tmpdir) / "linked"
            linked_root.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "speaker_cluster": "speaker_001",
                    "speaker_face_cluster": "face_babe",
                    "visible_face_clusters": ["face_babe", "face_other"],
                },
                {
                    "speaker_cluster": "speaker_001",
                    "speaker_face_cluster": "face_babe",
                    "visible_face_clusters": ["face_other"],
                },
            ]
            STEP06.write_json(linked_root / "episode_linked_segments.json", rows)
            char_map = {
                "clusters": {
                    "face_babe": {"name": "Babe Carano"},
                    "face_other": {"name": "Kenzie Bell"},
                }
            }
            voice_map = {"clusters": {"speaker_001": {"name": "speaker_001", "auto_named": True}}, "aliases": {}}
            cfg = {
                "paths": {"linked_segments": str(linked_root)},
                "character_detection": {
                    "speaker_face_cluster_vote_weight": 4.0,
                    "speaker_single_visible_vote_weight": 1.0,
                    "speaker_face_link_min_votes": 8.0,
                    "speaker_face_link_min_share": 0.45,
                    "speaker_face_link_min_margin": 3.0,
                },
            }

            summary = STEP06.auto_link_speakers_from_single_visible_faces(cfg, char_map, voice_map)

        self.assertEqual(summary["matched"], 1)
        self.assertEqual(voice_map["clusters"]["speaker_001"]["linked_face_cluster"], "face_babe")
        self.assertEqual(voice_map["clusters"]["speaker_001"]["name"], "Babe Carano")

    def test_review_queue_summary_counts_repeated_unresolved_ids(self) -> None:
        items = [
            {
                "scene_id": "scene_001",
                "speaker_name": "speaker_001",
                "visible_character_names": ["Babe Carano", "face_001"],
            },
            {
                "scene_id": "scene_001",
                "speaker_name": "Babe Carano",
                "visible_character_names": ["face_001", "face_002"],
            },
        ]

        summary = STEP06.review_queue_summary(items)

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["speaker_open"], 1)
        self.assertEqual(summary["visible_open"], 3)
        self.assertEqual(summary["top_speakers"][0], ("speaker_001", 1))
        self.assertEqual(summary["top_visible"][0], ("face_001", 2))

    def test_hydrate_face_clusters_adds_queue_referenced_preview_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            previews = root / "previews"
            missing_preview = previews / "face_999"
            existing_preview = previews / "face_known"
            missing_preview.mkdir(parents=True)
            existing_preview.mkdir(parents=True)
            (missing_preview / "scene_001_crop.jpg").write_text("image", encoding="utf-8")
            queue_path = root / "review_queue.json"
            STEP06.write_json(
                queue_path,
                {
                    "items": [
                        {
                            "visible_face_clusters": ["face_999"],
                            "visible_character_names": ["face_999"],
                        }
                    ]
                },
            )
            cfg = {
                "paths": {
                    "character_previews": str(previews),
                    "review_queue": str(queue_path),
                }
            }
            char_map = {"clusters": {"face_known": {"name": "Babe Carano"}}, "aliases": {}}

            added = STEP06.hydrate_face_clusters_from_previews(cfg, char_map)

        self.assertEqual(added, 1)
        self.assertIn("face_999", char_map["clusters"])
        self.assertEqual(char_map["clusters"]["face_999"]["samples"], 1)


if __name__ == "__main__":
    unittest.main()


