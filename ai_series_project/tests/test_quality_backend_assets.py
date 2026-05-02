from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path


MODULE_PATH = PROJECT_DIR / "support_scripts/prepare_quality_backends.py"
SPEC = importlib.util.spec_from_file_location("step59_quality_backends", MODULE_PATH)
STEP59 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(STEP59)

VOICE_BACKEND_PATH = PROJECT_DIR / "tools/quality_backends/project_local_voice_backend.py"
if str(VOICE_BACKEND_PATH.parent) not in sys.path:
    sys.path.insert(0, str(VOICE_BACKEND_PATH.parent))
VOICE_SPEC = importlib.util.spec_from_file_location("project_local_voice_backend", VOICE_BACKEND_PATH)
VOICE_BACKEND = importlib.util.module_from_spec(VOICE_SPEC)
assert VOICE_SPEC and VOICE_SPEC.loader
VOICE_SPEC.loader.exec_module(VOICE_BACKEND)


class QualityBackendAssetTests(unittest.TestCase):
    def test_project_local_voice_backend_collects_reference_audio_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_path = root / "audio" / "original.wav"
            original_path.parent.mkdir(parents=True, exist_ok=True)
            original_path.write_bytes(b"audio")
            sample_dir = root / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_path = sample_dir / "sample_01.wav"
            sample_path.write_bytes(b"sample")
            model_path = root / "voice_model.json"
            model_path.write_text(
                '{"reference_audio": "%s", "sample_paths": ["%s"]}' % (str(original_path).replace("\\", "\\\\"), str(sample_path).replace("\\", "\\\\")),
                encoding="utf-8",
            )

            line = {
                "voice_profile": {
                    "voice_model_path": str(model_path),
                    "reference_audio": str(original_path),
                },
                "original_voice_reference": {"audio_path": str(original_path)},
                "reference_segments": [{"audio_path": str(sample_path)}],
                "reference_audio_candidates": [str(sample_path)],
                "candidate_sample_dirs": [str(sample_dir)],
            }
            references = VOICE_BACKEND.collect_reference_audio_paths(line)

            self.assertEqual([path.name for path in references[:2]], ["original.wav", "sample_01.wav"])

    def test_support_scripts_bootstrap_project_dir_for_direct_execution(self) -> None:
        configure_source = (PROJECT_DIR / "support_scripts/configure_quality_backends.py").read_text(encoding="utf-8")
        prepare_source = (PROJECT_DIR / "support_scripts/prepare_quality_backends.py").read_text(encoding="utf-8")

        self.assertIn("PROJECT_DIR = Path(__file__).resolve().parents[1]", configure_source)
        self.assertIn("sys.path.insert(0, str(PROJECT_DIR))", configure_source)
        self.assertIn("PROJECT_DIR = Path(__file__).resolve().parents[1]", prepare_source)
        self.assertIn("sys.path.insert(0, str(PROJECT_DIR))", prepare_source)

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

    def test_run_git_command_always_allows_safe_directory(self) -> None:
        with mock.patch.object(STEP59.subprocess, "run", return_value=mock.Mock(returncode=0)) as run_mock:
            STEP59.run_git_command(["status"], target_dir=Path(r"\\server\share\repo"), check=False)
        command = run_mock.call_args.args[0]
        self.assertEqual(command[:3], ["git", "-c", "safe.directory=*"])

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

            real_move = shutil.move

            def fake_move(src: str, dst: str) -> str:
                real_move(src, dst)
                return dst

            with mock.patch.object(STEP59, "project_archive_staging_dir", return_value=root / "staging"):
                with mock.patch.object(STEP59, "urlopen", return_value=FakeResponse()):
                    with mock.patch.object(STEP59.shutil, "unpack_archive", side_effect=fake_unpack_archive):
                        with mock.patch.object(STEP59.shutil, "move", side_effect=fake_move):
                            STEP59.download_and_extract_github_archive(target)

            self.assertTrue((target_dir / "main.py").exists())
            self.assertTrue((target_dir / "server.py").exists())

    def test_ensure_hf_target_stages_locally_for_windows_unc_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            real_target = Path(temp_dir) / "nas-target"
            target = {
                "kind": "huggingface",
                "repo_id": "example/model",
                "target_dir": str(real_target),
                "required_files": ["weights.safetensors"],
            }

            def fake_snapshot_download(
                *,
                repo_id: str,
                local_dir: str,
                token: str | None,
                revision: str | None,
                max_workers: int,
            ) -> str:
                self.assertEqual(repo_id, "example/model")
                self.assertEqual(max_workers, 1)
                self.assertEqual(STEP59.os.environ.get("HF_HUB_DISABLE_XET"), "1")
                self.assertEqual(STEP59.os.environ.get("HF_XET_HIGH_PERFORMANCE"), "0")
                local_path = Path(local_dir)
                self.assertTrue(local_path.exists())
                (local_path / "weights.safetensors").write_text("ok", encoding="utf-8")
                cache_dir = local_path / ".cache" / "huggingface" / "download"
                cache_dir.mkdir(parents=True, exist_ok=True)
                (cache_dir / "entry.metadata").write_text("commit_hash: abc123\n", encoding="utf-8")
                return str(local_path)

            with mock.patch.object(STEP59, "is_windows_unc_path", return_value=True):
                with mock.patch.object(STEP59, "load_hf_snapshot_download", return_value=fake_snapshot_download):
                    with mock.patch.object(STEP59, "cleanup_incomplete_download_files", return_value=0):
                        with mock.patch.object(STEP59, "infer_local_hf_revision", return_value="abc123"):
                            result = STEP59.ensure_hf_target(target, "abc123", "", "download")

            self.assertEqual(result["transport"], "huggingface-local-stage")
            self.assertTrue((real_target / "weights.safetensors").exists())
            self.assertNotIn("HF_HUB_DISABLE_XET", STEP59.os.environ)
            self.assertNotIn("HF_XET_HIGH_PERFORMANCE", STEP59.os.environ)

    def test_load_hf_snapshot_download_sets_xet_env_before_import(self) -> None:
        fake_snapshot_download = object()
        imported_env: dict[str, str | None] = {}
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "huggingface_hub":
                imported_env["HF_HUB_DISABLE_XET"] = STEP59.os.environ.get("HF_HUB_DISABLE_XET")
                imported_env["HF_XET_HIGH_PERFORMANCE"] = STEP59.os.environ.get("HF_XET_HIGH_PERFORMANCE")
                return mock.Mock(snapshot_download=fake_snapshot_download)
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch.object(STEP59, "ensure_runtime_package"):
            with mock.patch.dict(STEP59.sys.modules, {}, clear=False):
                with mock.patch("builtins.__import__", side_effect=fake_import):
                    loaded = STEP59.load_hf_snapshot_download()

        self.assertIs(loaded, fake_snapshot_download)
        self.assertEqual(imported_env["HF_HUB_DISABLE_XET"], "1")
        self.assertEqual(imported_env["HF_XET_HIGH_PERFORMANCE"], "0")
        self.assertNotIn("HF_HUB_DISABLE_XET", STEP59.os.environ)
        self.assertNotIn("HF_XET_HIGH_PERFORMANCE", STEP59.os.environ)

    def test_ensure_git_target_falls_back_to_archive_when_checkout_is_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = Path(temp_dir) / "comfyui"
            target = {
                "name": "comfyui",
                "kind": "git",
                "repo_url": "https://github.com/comfyanonymous/ComfyUI.git",
                "ref": "master",
                "target_dir": str(target_dir),
                "required_files": ["main.py"],
            }

            def fake_run_git_command(args, *, target_dir=None, check=False, capture_output=False):
                if args[:1] == ["fetch"] or args[:1] == ["checkout"]:
                    raise subprocess.CalledProcessError(returncode=128, cmd=["git", *args])
                return mock.Mock(stdout="", returncode=0)

            with mock.patch.object(STEP59, "git_command_available", return_value=True):
                with mock.patch.object(STEP59, "run_git_command", side_effect=fake_run_git_command):
                    with mock.patch.object(STEP59, "refresh_git_target_via_archive", return_value="abc123") as refresh_mock:
                        result = STEP59.ensure_git_target(target, "abc123", "update")

            refresh_mock.assert_called_once()
            self.assertEqual(result["revision"], "abc123")


if __name__ == "__main__":
    unittest.main()



