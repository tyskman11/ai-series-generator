from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import pipeline_common
from pipeline_common import PROJECT_ROOT, normalize_portable_project_paths, portable_project_path, resolve_stored_project_path


class PortablePathTests(unittest.TestCase):
    def test_portable_project_path_rebases_project_absolute_path(self) -> None:
        absolute_path = PROJECT_ROOT / "characters" / "previews" / "face_001"
        self.assertEqual(portable_project_path(absolute_path), "characters/previews/face_001")

    def test_portable_project_path_keeps_external_absolute_path(self) -> None:
        external_path = Path("Z:/external/assets/preview.jpg")
        self.assertEqual(portable_project_path(external_path), str(external_path))

    def test_portable_project_path_rebases_linux_absolute_path_via_stored_path_resolver(self) -> None:
        with mock.patch.object(
            pipeline_common,
            "resolve_stored_project_path",
            return_value=PROJECT_ROOT / "characters" / "previews" / "face_999",
        ):
            self.assertEqual(
                portable_project_path("/volume1/shared/ai_series_project/characters/previews/face_999"),
                "characters/previews/face_999",
            )

    def test_resolve_stored_project_path_rebases_linux_absolute_project_path(self) -> None:
        rebased = resolve_stored_project_path(
            "/volume1/shared/work/ai_series_project/characters/previews/face_777/example.jpg"
        )
        self.assertEqual(
            rebased,
            PROJECT_ROOT / "characters" / "previews" / "face_777" / "example.jpg",
        )

    def test_normalize_portable_project_paths_rewrites_nested_known_fields(self) -> None:
        payload = {
            "clusters": {
                "face_001": {
                    "preview_dir": str(PROJECT_ROOT / "characters" / "previews" / "face_001"),
                    "merged_preview_dirs": [
                        str(PROJECT_ROOT / "characters" / "previews" / "face_001"),
                        str(PROJECT_ROOT / "characters" / "previews" / "face_002"),
                    ],
                }
            },
            "items": [
                {
                    "speaker_reference_frames": [
                        str(PROJECT_ROOT / "data" / "processed" / "faces" / "episode_a" / "scene_001.jpg")
                    ]
                }
            ],
        }

        normalized = normalize_portable_project_paths(payload)

        self.assertEqual(normalized["clusters"]["face_001"]["preview_dir"], "characters/previews/face_001")
        self.assertEqual(
            normalized["clusters"]["face_001"]["merged_preview_dirs"],
            ["characters/previews/face_001", "characters/previews/face_002"],
        )
        self.assertEqual(
            normalized["items"][0]["speaker_reference_frames"],
            ["data/processed/faces/episode_a/scene_001.jpg"],
        )


if __name__ == "__main__":
    unittest.main()
