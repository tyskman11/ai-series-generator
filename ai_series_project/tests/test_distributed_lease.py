from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from support_scripts.pipeline_common import acquire_distributed_lease, schedule_worker_task


class DistributedLeaseTests(unittest.TestCase):
    def test_distributed_lease_allows_takeover_when_same_host_owner_pid_is_gone(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lease_root = Path(tmpdir)
            lease_path = lease_root / "scene_001.json"
            lease_root.mkdir(parents=True, exist_ok=True)
            lease_path.write_text(
                json.dumps(
                    {
                        "owner_id": "master-pc-14404-deadbeef",
                        "heartbeat_at": 100.0,
                        "expires_at": 999999.0,
                        "meta": {
                            "hostname": "MASTER-PC",
                            "pid": 14404,
                            "worker_id": "master-pc-14404-deadbeef",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch("support_scripts.pipeline_common.distributed_hostname_token", return_value="master-pc"), mock.patch(
                "support_scripts.pipeline_common.process_id_active",
                return_value=False,
            ):
                takeover = acquire_distributed_lease(lease_root, "scene_001", "worker-b", 30.0)

            self.assertIsNotNone(takeover)
            assert takeover is not None
            self.assertEqual(takeover["owner_id"], "worker-b")

    def test_worker_scheduler_prefers_backend_ready_low_latency_gpu_worker(self) -> None:
        scheduled = schedule_worker_task(
            [
                {
                    "worker_id": "cpu-nas",
                    "has_gpu": False,
                    "available_memory_mb": 64000,
                    "storage_latency_ms": 4.0,
                },
                {
                    "worker_id": "gpu-slow-storage",
                    "has_gpu": True,
                    "available_memory_mb": 16384,
                    "ready_backend_runners": ["shot_video"],
                    "backend_health_score": 0.9,
                    "storage_latency_ms": 180.0,
                },
                {
                    "worker_id": "gpu-ready",
                    "has_gpu": True,
                    "available_memory_mb": 24576,
                    "ready_backend_runners": ["shot_video"],
                    "package_capabilities": {"quality_generation": True},
                    "backend_health_score": 0.95,
                    "storage_latency_ms": 14.0,
                },
            ],
            {
                "gpu_required": True,
                "min_memory_mb": 12288,
                "required_backend_runner": "shot_video",
                "required_package": "quality_generation",
                "preferred_step": "shot_video",
            },
        )

        self.assertTrue(scheduled["scheduled"])
        self.assertEqual(scheduled["worker"]["worker_id"], "gpu-ready")
        self.assertEqual(scheduled["rejected_workers"][0]["worker"]["worker_id"], "cpu-nas")
        self.assertIn("required backend ready", scheduled["selection_reason"])


if __name__ == "__main__":
    unittest.main()


