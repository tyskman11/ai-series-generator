from __future__ import annotations

import http.client
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from support_scripts import web_manager as WEB


def load_review_step():
    path = SCRIPT_ROOT / "05_review_unknowns.py"
    spec = importlib.util.spec_from_file_location("step05_web_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def load_backend_common():
    path = PROJECT_DIR / "tools" / "quality_backends" / "backend_common.py"
    spec = importlib.util.spec_from_file_location("web_resource_backend_common", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class FakeService:
    def __init__(self, media_path: Path) -> None:
        self.media = media_path
        self.mutations: list[tuple[str, list[str], str, bool]] = []
        self.started_resources: dict = {}

    def public_overview(self):
        return {
            "status": {"active": False, "progress_percent": 0},
            "statistics": {"generated_episode_count": 2},
            "server_time": "now",
        }

    def status(self):
        return {"active": False, "progress_percent": 0, "current_step": ""}

    def overview(self):
        return {**self.public_overview(), "status": self.status(), "episodes": [], "assets": []}

    def storage(self, _limit=240):
        return []

    def read_json_database(self, record_id):
        return {"record": {"record_id": record_id}, "text": "{}\n"}

    def save_json_database(self, record_id, text):
        return {"path": record_id, "text": text}

    def media_path(self, _kind, _record_id):
        return self.media

    def latest_pipeline_log(self):
        return {"path": "", "text": "test log"}

    def start_pipeline(self, resources=None):
        self.started_resources = resources or {}
        return {"started": True, "pid": 123}

    def stop_pipeline(self):
        return {"stopped": True, "pid": 123}

    def review_overview(self, include_named=False):
        return {"faces": [], "known_names": [], "voice_clusters": [], "queue_summary": {}, "offline": True}

    def review_preview_path(self, _cluster_id, _sample=0):
        return self.media

    def assign_review_face(self, cluster_id, name, priority=False):
        return {"cluster_id": cluster_id, "name": name, "priority": priority}

    def rename_review_name(self, old_name, new_name, priority=False):
        return {"old_name": old_name, "new_name": new_name, "priority": priority}

    def mutate_episodes(self, ids, action, dry_run=False):
        self.mutations.append(("episodes", ids, action, dry_run))
        return {"action": action}

    def mutate_assets(self, ids, action, dry_run=False):
        self.mutations.append(("assets", ids, action, dry_run))
        return {"action": action}

    def mutate_storage(self, ids, action, dry_run=False):
        self.mutations.append(("storage", ids, action, dry_run))
        return {"action": action}


class WebManagerTests(unittest.TestCase):
    def credentials(self, root: Path, username: str = "admin", password: str = "safe-password-123") -> dict:
        path = root / "admin_credentials.json"
        with mock.patch.object(WEB, "AUTH_PATH", path):
            WEB.configure_admin_credentials(username, password)
            return WEB.load_admin_credentials()

    def start_server(self, service: FakeService, credentials: dict):
        server = WEB.ThreadingHTTPServer(("127.0.0.1", 0), WEB.make_handler(service, credentials))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def login(self, connection: http.client.HTTPConnection, username="admin", password="safe-password-123") -> str:
        body = json.dumps({"username": username, "password": password})
        connection.request("POST", "/api/auth/login", body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        response.read()
        return cookie

    def test_range_header_supports_explicit_open_and_suffix_ranges(self) -> None:
        self.assertEqual(WEB.parse_range_header("bytes=2-5", 10), (2, 5))
        self.assertEqual(WEB.parse_range_header("bytes=7-", 10), (7, 9))
        self.assertEqual(WEB.parse_range_header("bytes=-3", 10), (7, 9))
        self.assertIsNone(WEB.parse_range_header("", 10))
        with self.assertRaises(ValueError):
            WEB.parse_range_header("bytes=20-30", 10)

    def test_admin_credentials_are_salted_hashed_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            password = "safe-password-123"
            path = Path(tmpdir) / "admin_credentials.json"
            with mock.patch.object(WEB, "AUTH_PATH", path):
                WEB.configure_admin_credentials("admin", password)
                credentials = WEB.load_admin_credentials()
            raw = path.read_text(encoding="utf-8")
            self.assertNotIn(password, raw)
            self.assertTrue(WEB.verify_admin_credentials(credentials, "admin", password))
            self.assertFalse(WEB.verify_admin_credentials(credentials, "admin", "wrong-password"))

    def test_public_statistics_need_no_login_but_admin_api_does(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            media = Path(tmpdir) / "episode.mp4"
            media.write_bytes(b"0123456789")
            service = FakeService(media)
            server, thread = self.start_server(service, self.credentials(Path(tmpdir)))
            connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
            try:
                connection.request("GET", "/api/public/overview")
                response = connection.getresponse()
                payload = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertEqual(payload["statistics"]["generated_episode_count"], 2)

                connection.request("GET", "/api/status")
                response = connection.getresponse()
                self.assertEqual(response.status, 401)
                response.read()
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_login_and_mutation_header_protection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            media = Path(tmpdir) / "episode.mp4"
            media.write_bytes(b"0123456789")
            service = FakeService(media)
            server, thread = self.start_server(service, self.credentials(Path(tmpdir)))
            connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
            try:
                cookie = self.login(connection)
                mutation = json.dumps({"action": "delete", "ids": ["folge_01"], "confirmation": "DELETE"})
                connection.request("POST", "/api/episodes/mutate", body=mutation, headers={"Cookie": cookie})
                response = connection.getresponse()
                self.assertEqual(response.status, 403)
                response.read()

                connection.request(
                    "POST",
                    "/api/episodes/mutate",
                    body=mutation,
                    headers={"Cookie": cookie, "X-Series-Web": "1", "Content-Type": "application/json"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                response.read()
                self.assertEqual(service.mutations, [("episodes", ["folge_01"], "delete", False)])
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_authenticated_media_is_inline_and_supports_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            media = Path(tmpdir) / "episode.mp4"
            media.write_bytes(b"0123456789")
            service = FakeService(media)
            server, thread = self.start_server(service, self.credentials(Path(tmpdir)))
            connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
            try:
                cookie = self.login(connection)
                connection.request("GET", "/api/media?kind=episode&id=folge_01", headers={"Cookie": cookie, "Range": "bytes=2-5"})
                response = connection.getresponse()
                self.assertEqual(response.status, 206)
                self.assertTrue(response.getheader("Content-Disposition").startswith("inline"))
                self.assertEqual(response.getheader("Cache-Control"), "no-store")
                self.assertEqual(response.read(), b"2345")
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_pipeline_start_is_local_numbered_and_network_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project = root / "ai_series_project"
            script = root / "24_process_next_episode.py"
            support = project / "support_scripts"
            support.mkdir(parents=True)
            runner = support / "resource_limited_pipeline.py"
            runner.write_text("print('resource controller')\n", encoding="utf-8")
            script.write_text("print('pipeline')\n", encoding="utf-8")
            manager = SimpleNamespace(build_live_generation_status=lambda _cfg: SimpleNamespace(to_dict=lambda: {"active": False}))
            process = mock.Mock(pid=321)
            process.poll.return_value = None
            with mock.patch.object(WEB, "SCRIPT_ROOT", root), mock.patch.object(
                WEB, "PROJECT_DIR", project
            ), mock.patch.object(WEB, "PIPELINE_LOG_ROOT", project / "logs"), mock.patch.object(
                WEB.subprocess, "Popen", return_value=process
            ) as popen:
                result = WEB.WebManagerService(manager, {}).start_pipeline(
                    {"profile": "custom", "cpu_percent": 50, "gpu_memory_percent": 0, "priority": "low"}
                )
            self.assertTrue(result["started"])
            command = popen.call_args.args[0]
            environment = popen.call_args.kwargs["env"]
            self.assertEqual(Path(command[1]), runner)
            self.assertEqual(Path(command[2]), script)
            self.assertEqual(environment["SERIES_DISABLE_NETWORK"], "1")
            self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
            self.assertEqual(environment["SERIES_WEB_CPU_PERCENT"], "50")
            self.assertEqual(environment["SERIES_GPU_MEMORY_PERCENT"], "0")
            self.assertEqual(environment["SERIES_WEB_PROCESS_PRIORITY"], "low")
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(result["resources"]["cpu_threads"], max(1, round((WEB.os.cpu_count() or 1) * 0.5)))
            self.assertNotIn("shell", popen.call_args.kwargs)

    def test_resource_profiles_are_normalized_and_bounded(self) -> None:
        balanced = WEB.normalize_resource_settings({})
        self.assertEqual(balanced["profile"], "balanced")
        self.assertEqual(balanced["cpu_percent"], 65)
        self.assertEqual(balanced["gpu_memory_percent"], 75)
        custom = WEB.normalize_resource_settings(
            {"profile": "custom", "cpu_percent": 3, "gpu_memory_percent": 150, "priority": "invalid"}
        )
        self.assertEqual(custom["cpu_percent"], 10)
        self.assertEqual(custom["gpu_memory_percent"], 100)
        self.assertEqual(custom["priority"], "normal")
        self.assertGreaterEqual(custom["cpu_threads"], 1)

    def test_torch_backends_apply_thread_and_gpu_memory_budget(self) -> None:
        backend_common = load_backend_common()

        class FakeCuda:
            def __init__(self):
                self.fraction = None

            @staticmethod
            def is_available():
                return True

            def set_per_process_memory_fraction(self, fraction, device):
                self.fraction = (fraction, device)

        class FakeTorch:
            def __init__(self):
                self.cuda = FakeCuda()
                self.threads = 0
                self.interop_threads = 0

            def set_num_threads(self, value):
                self.threads = value

            def set_num_interop_threads(self, value):
                self.interop_threads = value

        fake_torch = FakeTorch()
        with mock.patch.dict(
            backend_common.os.environ,
            {"SERIES_CPU_THREADS": "3", "SERIES_GPU_MEMORY_PERCENT": "60"},
            clear=False,
        ):
            result = backend_common.apply_torch_resource_limits(fake_torch)
        self.assertEqual(fake_torch.threads, 3)
        self.assertEqual(fake_torch.interop_threads, 3)
        self.assertEqual(fake_torch.cuda.fraction, (0.6, 0))
        self.assertTrue(result["gpu_memory_limit_applied"])

    def test_pipeline_start_endpoint_forwards_resource_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            media = Path(tmpdir) / "episode.mp4"
            media.write_bytes(b"0123456789")
            service = FakeService(media)
            server, thread = self.start_server(service, self.credentials(Path(tmpdir)))
            connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
            try:
                cookie = self.login(connection)
                resources = {"profile": "eco", "cpu_percent": 35, "gpu_memory_percent": 50, "priority": "low"}
                connection.request(
                    "POST",
                    "/api/pipeline/start",
                    body=json.dumps({"resources": resources}),
                    headers={"Cookie": cookie, "X-Series-Web": "1", "Content-Type": "application/json"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                response.read()
                self.assertEqual(service.started_resources, resources)
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_browser_worker_claims_frame_task_and_saves_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "ai_series_project"
            frame = project / "generation" / "storyboard_assets" / "folge_01" / "scene_0001" / "frame.png"
            frame.parent.mkdir(parents=True)
            frame.write_bytes(b"frame-bytes")
            service = WEB.WebManagerService(SimpleNamespace(), {})
            with mock.patch.object(WEB, "PROJECT_DIR", project):
                registered = service.register_browser_worker(
                    {"profile": "balanced", "cpu_intensity": 60, "hardware_concurrency": 8, "webgpu_available": True}
                )
                worker_id = registered["worker_id"]
                queued = service.queue_browser_frame_checks(8)
                claimed = service.claim_browser_task(worker_id)["task"]
                self.assertIsNotNone(claimed)
                self.assertEqual(service.browser_task_input_path(worker_id, claimed["task_id"]), frame)
                completed = service.complete_browser_task(
                    worker_id,
                    claimed["task_id"],
                    {
                        "status": "success",
                        "result": {
                            "width": 1920,
                            "height": 1080,
                            "sample_width": 640,
                            "sample_height": 360,
                            "mean_luma": 0.52,
                            "luma_stddev": 0.19,
                            "edge_score": 0.17,
                            "dark_ratio": 0.01,
                            "bright_ratio": 0.02,
                        },
                    },
                )
                report = json.loads(service._browser_metrics_path().read_text(encoding="utf-8"))
            self.assertEqual(queued["queued"], 1)
            self.assertTrue(completed["saved"])
            self.assertEqual(report["metrics"][0]["metric"], "browser_frame_quality_score")
            self.assertGreater(report["metrics"][0]["score"], 0.8)

    def test_pwa_worker_assets_are_public_but_api_remains_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            media = Path(tmpdir) / "episode.mp4"
            media.write_bytes(b"0123456789")
            service = FakeService(media)
            server, thread = self.start_server(service, self.credentials(Path(tmpdir)))
            connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
            try:
                connection.request("GET", "/service-worker.js")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertIn("javascript", response.getheader("Content-Type"))
                response.read()
                connection.request("GET", "/api/browser-worker/status")
                response = connection.getresponse()
                self.assertEqual(response.status, 401)
                response.read()
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_step05_is_offline_by_default_and_online_is_explicit(self) -> None:
        step05 = load_review_step()
        default_args = SimpleNamespace(offline=False, online=False)
        online_args = SimpleNamespace(offline=False, online=True)
        self.assertFalse(step05.online_review_lookup_enabled(default_args, {}))
        self.assertTrue(step05.online_review_lookup_enabled(online_args, {}))
        self.assertFalse(step05.online_review_lookup_enabled(online_args, {"SERIES_DISABLE_NETWORK": "1"}))
        with self.assertRaises(RuntimeError):
            step05.online_review_lookup_enabled(SimpleNamespace(offline=True, online=True), {})


if __name__ == "__main__":
    unittest.main()
