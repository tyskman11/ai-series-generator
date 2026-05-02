from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import importlib.util
import unittest
from pathlib import Path


ROOT = PROJECT_DIR


def load_module(filename: str, module_name: str):
    target = ROOT / filename if filename.startswith("support_scripts/") else SCRIPT_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PIPELINE = load_module("support_scripts/pipeline_common.py", "pipeline_common_step99_test")
STEP99 = load_module("57_process_next_episode.py", "step99_process_test")


class ProcessNextEpisodeTests(unittest.TestCase):
    def test_step99_imports_shared_project_root(self) -> None:
        self.assertTrue(hasattr(STEP99, "PROJECT_ROOT"))
        self.assertEqual(STEP99.PROJECT_ROOT, PIPELINE.PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()



