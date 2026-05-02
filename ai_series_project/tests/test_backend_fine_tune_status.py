from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = PROJECT_DIR


def load_module(filename: str, module_name: str):
    target = ROOT / filename if filename.startswith("support_scripts/") else SCRIPT_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PIPELINE = load_module("support_scripts/pipeline_common.py", "pipeline_common_backend_status")
STEP50 = load_module("12_run_backend_finetunes.py", "step50_backend_runs")


class BackendFineTuneStatusTests(unittest.TestCase):
    def test_backend_status_uses_newer_run_files_when_summary_is_stale(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(ROOT / "tmp")) as tmp:
            temp_root = Path(tmp)
            fine_rel = Path("tmp") / temp_root.name / "finetunes"
            backend_rel = Path("tmp") / temp_root.name / "backend"
            fine_dir = ROOT / fine_rel
            backend_dir = ROOT / backend_rel
            fine_dir.mkdir(parents=True, exist_ok=True)
            backend_dir.mkdir(parents=True, exist_ok=True)

            fine_summary = fine_dir / "fine_tune_training_summary.json"
            backend_summary = backend_dir / "backend_fine_tune_summary.json"
            run_dir = backend_dir / "babe"
            artifact_dir = run_dir / "image"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "training_job.json").write_text("{}", encoding="utf-8")
            (artifact_dir / "model_bundle.json").write_text("{}", encoding="utf-8")
            (artifact_dir / "image_weights.bin").write_bytes(b"demo")

            fine_summary.write_text(json.dumps({"characters": []}), encoding="utf-8")
            backend_summary.write_text(json.dumps({"characters": []}), encoding="utf-8")

            old_time = fine_summary.stat().st_mtime - 30
            os.utime(backend_summary, (old_time, old_time))

            run_path = run_dir / "backend_fine_tune_run.json"
            run_path.write_text(
                json.dumps(
                    {
                        "process_version": STEP50.PROCESS_VERSION,
                        "character": "Babe",
                        "slug": "babe",
                        "training_ready": True,
                        "modalities_ready": ["image"],
                        "backends": {
                            "image": {
                                "backend": "lora-image",
                                "ready": True,
                                "artifacts": {
                                    "job_path": str(artifact_dir / "training_job.json"),
                                    "bundle_path": str(artifact_dir / "model_bundle.json"),
                                    "weights_path": str(artifact_dir / "image_weights.bin"),
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            new_time = fine_summary.stat().st_mtime + 30
            os.utime(run_path, (new_time, new_time))

            cfg = {
                "paths": {
                    "foundation_finetunes": fine_rel.as_posix(),
                    "foundation_backend_runs": backend_rel.as_posix(),
                },
                "backend_fine_tune": {"required_before_generate": False, "required_before_render": False},
            }

            status = PIPELINE.backend_fine_tune_status(cfg, characters=["Babe"])

        self.assertTrue(status["summary_exists"])
        self.assertTrue(status["summary_new_enough"])
        self.assertEqual(status["missing_characters"], [])
        self.assertEqual(status["weak_characters"], [])

    def test_collect_existing_backend_summary_rows_aggregates_all_run_files(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(ROOT / "tmp")) as tmp:
            temp_root = Path(tmp)
            backend_rel = Path("tmp") / temp_root.name / "backend"
            backend_dir = ROOT / backend_rel
            backend_dir.mkdir(parents=True, exist_ok=True)

            for character_name, slug in (("Babe", "babe"), ("Kenzie", "kenzie")):
                run_dir = backend_dir / slug
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "backend_fine_tune_run.json").write_text(
                    json.dumps(
                        {
                            "process_version": STEP50.PROCESS_VERSION,
                            "character": character_name,
                            "slug": slug,
                            "training_ready": True,
                            "modalities_ready": ["image"],
                            "backends": {},
                        }
                    ),
                    encoding="utf-8",
                )

            cfg = {"paths": {"foundation_backend_runs": backend_rel.as_posix()}}

            with mock.patch.object(STEP50, "load_step_autosave", return_value={}):
                rows = STEP50.collect_existing_backend_summary_rows(cfg)

        self.assertEqual([row["character"] for row in rows], ["Babe", "Kenzie"])


if __name__ == "__main__":
    unittest.main()



