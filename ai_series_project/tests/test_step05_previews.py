from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from support_scripts import pipeline_common

ROOT = PROJECT_DIR


def load_module(filename: str, module_name: str):
    target = ROOT / filename if filename.startswith("support_scripts/") else SCRIPT_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP05 = load_module("04_link_faces_and_speakers.py", "step05_previews")


class Step05PreviewTests(unittest.TestCase):
    def test_open_file_default_uses_windows_shell_execute_for_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_path = Path(tmpdir) / "preview.html"
            preview_path.write_text("<html></html>", encoding="utf-8")
            shell32 = SimpleNamespace(ShellExecuteW=mock.Mock(return_value=33))
            fake_windll = SimpleNamespace(shell32=shell32)

            with mock.patch.object(pipeline_common, "current_os", return_value="windows"), mock.patch.object(
                pipeline_common.ctypes,
                "windll",
                fake_windll,
                create=True,
            ), mock.patch.object(
                pipeline_common.subprocess,
                "Popen",
            ) as popen:
                pipeline_common.open_file_default(preview_path)

            shell32.ShellExecuteW.assert_called_once()
            popen.assert_not_called()

    def test_create_preview_html_writes_embedded_browser_preview(self) -> None:
        try:
            from PIL import Image
        except Exception:
            self.skipTest("Pillow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            preview_dir = Path(tmpdir)
            crop_path = preview_dir / "scene_001_crop.jpg"
            context_path = preview_dir / "scene_001_context.jpg"
            Image.new("RGB", (64, 64), "white").save(crop_path)
            Image.new("RGB", (120, 80), "navy").save(context_path)

            html_path = STEP05.create_preview_html("face_001", [crop_path, context_path])

            self.assertIsNotNone(html_path)
            assert html_path is not None
            self.assertTrue(html_path.exists())
            html_text = html_path.read_text(encoding="utf-8")
            self.assertIn("face_001 Preview", html_text)
            self.assertIn("data:image/jpeg;base64,", html_text)

    def test_preview_open_targets_prefers_html_then_images_then_montage(self) -> None:
        try:
            from PIL import Image
        except Exception:
            self.skipTest("Pillow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            preview_dir = Path(tmpdir)
            crop_path = preview_dir / "scene_001_crop.jpg"
            context_path = preview_dir / "scene_001_context.jpg"
            montage_path = preview_dir / "face_001_montage.jpg"
            Image.new("RGB", (64, 64), "white").save(crop_path)
            Image.new("RGB", (120, 80), "navy").save(context_path)
            Image.new("RGB", (120, 120), "gray").save(montage_path)

            targets = STEP05.preview_open_targets("face_001", [crop_path, context_path], montage_path)

            self.assertGreaterEqual(len(targets), 3)
            self.assertEqual(targets[0].suffix, ".html")
            self.assertEqual(targets[1:3], [crop_path, context_path])
            self.assertEqual(targets[-1], montage_path)

    def test_ask_name_opens_preferred_targets_and_prints_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_dir = Path(tmpdir)
            crop_path = preview_dir / "scene_001_crop.jpg"
            crop_path.write_text("placeholder", encoding="utf-8")
            html_path = preview_dir / "face_001_preview.html"
            html_path.write_text("<html></html>", encoding="utf-8")
            montage_path = preview_dir / "face_001_montage.jpg"
            montage_path.write_text("placeholder", encoding="utf-8")

            with mock.patch.object(STEP05, "create_contact_sheet", return_value=montage_path), mock.patch.object(
                STEP05,
                "preview_open_targets",
                return_value=[html_path, crop_path, montage_path],
            ), mock.patch.object(STEP05, "open_file_default") as open_file, mock.patch(
                "builtins.input",
                return_value="Babe Carano",
            ):
                result = STEP05.ask_name("character", "face_001", [crop_path], auto_open=True)

            self.assertEqual(result, "Babe Carano")
            self.assertEqual(open_file.call_args_list[0].args[0], html_path)
            self.assertEqual(open_file.call_args_list[1].args[0], crop_path)
            self.assertEqual(open_file.call_args_list[2].args[0], montage_path)


if __name__ == "__main__":
    unittest.main()



