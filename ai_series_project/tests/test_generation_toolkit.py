from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import unittest
from unittest import mock

from support_scripts import generation_toolkit


class GenerationToolkitTests(unittest.TestCase):
    def test_all_optional_tools_are_registered_for_generation_context(self) -> None:
        optional_root = PROJECT_DIR / "support_scripts" / "optional_tools"
        disk_tools = {path.name for path in optional_root.glob("*.py") if path.name != "__init__.py"}
        registered = {row["script"] for row in generation_toolkit.tool_manifest()}

        self.assertEqual(disk_tools, registered)

    def test_post_render_invocations_include_episode_and_character_tools(self) -> None:
        cfg = {"generation_toolkit": {"max_characters": 2}}

        with mock.patch.object(generation_toolkit, "top_character_names", return_value=["Babe", "Kenzie"]):
            invocations, skipped = generation_toolkit.build_tool_invocations(cfg, "post_render", "folge_42")

        scripts = [row["script"] for row in invocations]
        self.assertIn("24_mood_analyzer.py", scripts)
        self.assertIn("52_voice_emotion_cloning.py", scripts)
        self.assertIn("--episode-id", invocations[scripts.index("24_mood_analyzer.py")]["args"])
        emotion_rows = [row for row in invocations if row["script"] == "52_voice_emotion_cloning.py"]
        self.assertEqual(len(emotion_rows), 2)
        self.assertTrue(all("--source-episode" in row["args"] for row in emotion_rows))
        self.assertEqual(skipped, [])

    def test_manual_only_tools_stay_manifested_but_do_not_auto_run(self) -> None:
        invocations, skipped = generation_toolkit.build_tool_invocations({}, "manual", "folge_42")

        self.assertEqual(invocations, [])
        skipped_scripts = {row["script"] for row in skipped}
        self.assertIn("25_merge_episodes.py", skipped_scripts)
        self.assertIn("54_backup_project.py", skipped_scripts)
        self.assertIn("55_restore_project.py", skipped_scripts)


if __name__ == "__main__":
    unittest.main()
