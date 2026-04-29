from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PIPELINE = load_module("pipeline_common.py", "pipeline_common_step99_test")
STEP99 = load_module("99_process_next_episode.py", "step99_process_test")


class ProcessNextEpisodeTests(unittest.TestCase):
    def test_step99_imports_shared_project_root(self) -> None:
        self.assertTrue(hasattr(STEP99, "PROJECT_ROOT"))
        self.assertEqual(STEP99.PROJECT_ROOT, PIPELINE.PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
