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

    def test_gui_preview_unavailable_on_headless_linux(self) -> None:
        with mock.patch.object(STEP06, "current_os", return_value="linux"), mock.patch.dict(
            os.environ,
            {"DISPLAY": "", "WAYLAND_DISPLAY": ""},
            clear=False,
        ):
            self.assertFalse(STEP06.gui_preview_available())


if __name__ == "__main__":
    unittest.main()


