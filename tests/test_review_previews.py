from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP06 = load_module("06_review_unknowns.py", "step06_review_previews")


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

    def test_open_preview_file_uses_windows_default_viewer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_path = Path(tmpdir) / "preview.jpg"
            preview_path.write_text("placeholder", encoding="utf-8")

            with mock.patch.object(STEP06, "current_os", return_value="windows"), mock.patch.object(
                STEP06.os,
                "startfile",
                create=True,
            ) as startfile:
                self.assertTrue(STEP06.open_preview_file(preview_path))

            startfile.assert_called_once_with(str(preview_path))

    def test_gui_preview_unavailable_on_headless_linux(self) -> None:
        with mock.patch.object(STEP06, "current_os", return_value="linux"), mock.patch.dict(
            os.environ,
            {"DISPLAY": "", "WAYLAND_DISPLAY": ""},
            clear=False,
        ):
            self.assertFalse(STEP06.gui_preview_available())


if __name__ == "__main__":
    unittest.main()
