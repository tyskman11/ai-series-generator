from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STEP17 = load_module("17_render_episode.py", "step17_season_asset_test")
STEP25 = load_module("25_generate_season_assets.py", "step25_season_asset_test")


class SeasonAssetGenerationTests(unittest.TestCase):
    def test_season_asset_retry_policy_defaults_to_unlimited_until_gate_passes(self) -> None:
        retry_until_pass, max_cycles = STEP25.season_asset_retry_policy(
            {"release_mode": {"retry_until_pass": True, "max_auto_retry_cycles": 0}},
            argparse.Namespace(max_quality_cycles=None),
        )

        self.assertTrue(retry_until_pass)
        self.assertEqual(max_cycles, 0)

    def test_season_asset_retry_policy_recovers_from_invalid_legacy_value(self) -> None:
        retry_until_pass, max_cycles = STEP25.season_asset_retry_policy(
            {"release_mode": {"retry_until_pass": True, "max_auto_retry_cycles": "not-a-number"}},
            argparse.Namespace(max_quality_cycles=None),
        )

        self.assertTrue(retry_until_pass)
        self.assertEqual(max_cycles, 0)

    def test_season_asset_retry_stops_for_fallback_backend(self) -> None:
        blocker = STEP25.season_asset_retry_blocker(
            {"source_origin": "backend_generated", "fallback_used": True},
            {"blockers": ["a fallback backend was used"]},
        )

        self.assertIn("fallback backend", blocker)

    def test_season_asset_retry_regenerates_once_then_accepts_passing_gate(self) -> None:
        class Reporter:
            def update(self, *_args, **_kwargs) -> None:
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calls = {"count": 0}

            def materialize(*_args, **_kwargs):
                calls["count"] += 1
                asset_root = root / "generation/season_assets/season_01/intro"
                asset_root.mkdir(parents=True, exist_ok=True)
                manifest = asset_root / "intro_manifest.json"
                image_manifest = asset_root / "image_manifest.json"
                video_manifest = asset_root / "video_manifest.json"
                manifest.write_text("{}", encoding="utf-8")
                image_manifest.write_text("{}", encoding="utf-8")
                video_manifest.write_text("{}", encoding="utf-8")
                video = asset_root / "intro.mp4"
                if calls["count"] > 1:
                    video.write_bytes(b"real-video")
                return {
                    "season_id": "season_01",
                    "canonical_video": str(video),
                    "canonical_sha256": hashlib.sha256(video.read_bytes()).hexdigest() if video.exists() else "",
                    "manifest": str(manifest),
                    "source_origin": "backend_generated",
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

            render_module = type("RenderModule", (), {"materialize_season_asset": staticmethod(materialize)})()
            with mock.patch.object(STEP25, "resolve_project_path", side_effect=lambda relative: root / relative):
                asset, report, _report_json, _report_md = STEP25.render_season_asset_until_quality_gate(
                    render_module,
                    {},
                    {"season_id": "season_01"},
                    "ffmpeg",
                    season_id="season_01",
                    asset_kind="intro",
                    render_cfg={},
                    reporter=Reporter(),
                    overall_base=0.0,
                    force=False,
                    retry_until_pass=True,
                    max_quality_cycles=0,
                )

            self.assertTrue(Path(asset["canonical_video"]).exists())

        self.assertEqual(calls["count"], 2)
        self.assertTrue(report["passed"])
        self.assertEqual(report["attempt"], 2)

    def test_generated_outro_package_uses_single_identity_safe_hero_shot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            package = STEP17.build_generated_season_intro_package(
                {"season_outro": {"generated_duration_seconds": 8.0}},
                {"active_character_groups": [{"label": "Test Series"}], "focus_characters": ["Babe"], "scenes": []},
                "season_01",
                {},
                Path(tmpdir),
                asset_kind="outro",
            )

        scene = package["scene_package"]
        self.assertEqual(scene["scene_id"], "season_outro")
        self.assertEqual(scene["video_generation"]["mode"], "generated_season_outro_video")
        self.assertEqual(scene["shot_packages"][0]["characters_visible"], [])
        self.assertLessEqual(len(scene["shot_packages"][1]["characters_visible"]), 1)

    def test_season_asset_gate_accepts_hash_locked_backend_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "intro.mp4"
            manifest = root / "intro_manifest.json"
            image_manifest = root / "image.json"
            video_manifest = root / "video.json"
            for path in (image_manifest, video_manifest, manifest):
                path.write_text("{}", encoding="utf-8")
            video.write_bytes(b"real-motion-video")
            payload = {
                "season_id": "season_01",
                "canonical_video": str(video),
                "canonical_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "manifest": str(manifest),
                "source_origin": "backend_generated",
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

            report = STEP25.season_asset_quality_gate(payload, "intro")

        self.assertTrue(report["passed"])

    def test_season_asset_gate_rejects_fallback_or_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "outro.mp4"
            video.write_bytes(b"not-a-fallback")
            payload = {
                "season_id": "season_01",
                "canonical_video": str(video),
                "canonical_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "manifest": str(root / "missing.json"),
                "source_origin": "backend_generated",
                "fallback_used": True,
            }

            report = STEP25.season_asset_quality_gate(payload, "outro")

        self.assertFalse(report["passed"])
        self.assertIn("asset manifest is missing", report["blockers"])
        self.assertIn("a fallback backend was used", report["blockers"])

    def test_episode_render_defaults_do_not_require_or_autogenerate_intro(self) -> None:
        from support_scripts.pipeline_common import DEFAULT_CONFIG

        intro = DEFAULT_CONFIG["season_intro"]
        self.assertFalse(intro["require_in_finished_episode_mode"])
        self.assertFalse(intro["auto_generate_if_missing"])
        self.assertIn("season_outro", DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
