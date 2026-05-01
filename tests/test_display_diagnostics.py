from __future__ import annotations

import os
import unittest
from unittest import mock

from support_scripts import pipeline_common


class DisplayDiagnosticsTests(unittest.TestCase):
    def test_interactive_display_state_marks_headless_linux_without_display(self) -> None:
        with mock.patch.object(pipeline_common, "current_os", return_value="linux"), mock.patch.object(
            pipeline_common,
            "is_interactive_session",
            return_value=True,
        ), mock.patch.dict(
            os.environ,
            {"DISPLAY": "", "WAYLAND_DISPLAY": "", "SSH_CONNECTION": ""},
            clear=False,
        ):
            state = pipeline_common.interactive_display_state()

        self.assertEqual(state["os"], "linux")
        self.assertTrue(state["interactive_console"])
        self.assertFalse(state["gui_available"])
        self.assertIn("DISPLAY", state["reason"])

    def test_interactive_display_state_marks_windows_gui_available_for_console_run(self) -> None:
        with mock.patch.object(pipeline_common, "current_os", return_value="windows"), mock.patch.object(
            pipeline_common,
            "is_interactive_session",
            return_value=True,
        ), mock.patch.dict(
            os.environ,
            {"SESSIONNAME": "Console", "SSH_CONNECTION": ""},
            clear=False,
        ):
            state = pipeline_common.interactive_display_state()

        self.assertEqual(state["os"], "windows")
        self.assertTrue(state["interactive_console"])
        self.assertTrue(state["gui_available"])
        self.assertEqual(state["session_name"], "Console")

    def test_print_interactive_display_diagnostics_warns_when_gui_is_required_but_missing(self) -> None:
        state = {
            "os": "linux",
            "interactive_console": True,
            "gui_available": False,
            "display": "",
            "wayland_display": "",
            "session_name": "",
            "ssh_session": True,
            "reason": "Linux run without DISPLAY/WAYLAND_DISPLAY.",
        }
        with mock.patch.object(pipeline_common, "interactive_display_state", return_value=state), mock.patch.object(
            pipeline_common,
            "info",
        ) as info_mock, mock.patch.object(
            pipeline_common,
            "warn",
        ) as warn_mock:
            result = pipeline_common.print_interactive_display_diagnostics(
                "06_review_unknowns.py",
                require_gui=True,
            )

        self.assertEqual(result, state)
        info_mock.assert_called_once()
        warn_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()



