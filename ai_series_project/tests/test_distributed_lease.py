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

from support_scripts.pipeline_common import acquire_distributed_lease


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


if __name__ == "__main__":
    unittest.main()



