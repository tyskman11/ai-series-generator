from __future__ import annotations

import importlib.util
import tempfile
import unittest
from unittest import mock
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "59_prepare_quality_backends.py"
SPEC = importlib.util.spec_from_file_location("step59_quality_backends", MODULE_PATH)
STEP59 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(STEP59)


class QualityBackendAssetTests(unittest.TestCase):
    def test_main_reruns_in_runtime_before_work(self) -> None:
        with mock.patch.object(STEP59, "rerun_in_runtime") as rerun_mock:
            with mock.patch.object(STEP59, "parse_args", return_value=mock.Mock(force=False, skip_downloads=True, print_plan=True)):
                with mock.patch.object(STEP59, "headline"):
                    with mock.patch.object(STEP59, "load_config", return_value={}):
                        with mock.patch.object(STEP59, "prepare_quality_backend_assets", return_value=[]):
                            with mock.patch.object(STEP59, "write_summary", return_value=Path("summary.json")):
                                with mock.patch.object(STEP59, "ok"):
                                    STEP59.main()
        rerun_mock.assert_called_once_with(str(MODULE_PATH))

    def test_target_required_files_ready_requires_listed_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "main.py").write_text("print('ok')", encoding="utf-8")
            target = {
                "target_dir": str(root),
                "required_files": ["main.py", "server.py"],
            }
            self.assertFalse(STEP59.target_required_files_ready(target))
            (root / "server.py").write_text("print('ok')", encoding="utf-8")
            self.assertTrue(STEP59.target_required_files_ready(target))

    def test_resolve_target_action_detects_incomplete_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "main.py").write_text("print('ok')", encoding="utf-8")
            cache_dir = root / ".cache" / "huggingface" / "download"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "weights.incomplete").write_text("partial", encoding="utf-8")
            target = {
                "kind": "huggingface",
                "target_dir": str(root),
                "required_files": ["main.py"],
            }
            self.assertEqual(STEP59.resolve_target_action(target, "abc123"), "repair")

    def test_resolve_target_action_detects_current_when_revision_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "main.py").write_text("print('ok')", encoding="utf-8")
            STEP59.write_asset_metadata(
                {"target_dir": str(root)},
                {"revision": "abc123"},
            )
            target = {
                "kind": "git",
                "target_dir": str(root),
                "required_files": ["main.py"],
            }
            self.assertEqual(STEP59.resolve_target_action(target, "abc123"), "current")

    def test_default_quality_backend_asset_targets_include_project_local_comfyui(self) -> None:
        cfg = {
            "foundation_training": {
                "image_base_model": "stabilityai/stable-diffusion-xl-base-1.0",
                "video_base_model": "Lightricks/LTX-Video",
                "voice_base_model": "openbmb/VoxCPM2",
                "lipsync_model": "wav2lip",
            },
            "cloning": {
                "xtts_model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
            },
        }
        targets = STEP59.default_quality_backend_asset_targets(cfg)
        comfy = next(target for target in targets if target["name"] == "comfyui")
        self.assertEqual(comfy["target_dir"], "tools/quality_backends/comfyui")
        self.assertTrue(any(target["name"] == "video_base_model" for target in targets))

    def test_fetch_remote_git_revision_uses_github_api_when_git_missing(self) -> None:
        target = {
            "repo_url": "https://github.com/comfyanonymous/ComfyUI.git",
            "ref": "master",
        }
        with mock.patch.object(STEP59, "git_command_available", return_value=False):
            with mock.patch.object(STEP59, "github_request_json", return_value={"sha": "abc123def456"}):
                revision = STEP59.fetch_remote_git_revision(target)
        self.assertEqual(revision, "abc123def456")

    def test_download_and_extract_github_archive_replaces_target_without_git(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target_dir = root / "comfyui"
            target = {
                "repo_url": "https://github.com/comfyanonymous/ComfyUI.git",
                "ref": "master",
                "target_dir": str(target_dir),
            }

            def fake_unpack_archive(_: str, destination: str) -> None:
                extracted = Path(destination) / "ComfyUI-master"
                extracted.mkdir(parents=True, exist_ok=True)
                (extracted / "main.py").write_text("print('ok')", encoding="utf-8")
                (extracted / "server.py").write_text("print('ok')", encoding="utf-8")

            class FakeResponse:
                def __init__(self) -> None:
                    self._sent = False

                def __enter__(self) -> "FakeResponse":
                    return self

                def __exit__(self, exc_type, exc, tb) -> None:
                    return None

                def read(self, size: int = -1) -> bytes:
                    if self._sent:
                        return b""
                    self._sent = True
                    return b"zip-bytes"

            with mock.patch.object(STEP59, "urlopen", return_value=FakeResponse()):
                with mock.patch.object(STEP59.shutil, "unpack_archive", side_effect=fake_unpack_archive):
                    STEP59.download_and_extract_github_archive(target)

            self.assertTrue((target_dir / "main.py").exists())
            self.assertTrue((target_dir / "server.py").exists())


if __name__ == "__main__":
    unittest.main()
