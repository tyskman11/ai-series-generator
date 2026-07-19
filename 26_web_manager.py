#!/usr/bin/env python3
"""Start the NAS-backed browser dashboard without changing the desktop GUI."""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_ROOT / "ai_series_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import gui as manager
from support_scripts.web_manager import configure_admin_credentials, run_web_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the read-only statistics and authenticated NAS project manager.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address. Default: all interfaces on this host.")
    parser.add_argument("--port", type=int, default=8765, help="TCP port. Default: 8765")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser on this host.")
    parser.add_argument("--configure-admin", action="store_true", help="Create or replace the project-local administrator login.")
    return parser.parse_args()


def configure_admin() -> None:
    username = os.environ.get("SERIES_WEB_ADMIN_USER", "").strip()
    password = os.environ.get("SERIES_WEB_ADMIN_PASSWORD", "")
    if not username:
        username = input("Administrator username: ").strip()
    if not password:
        password = getpass.getpass("Administrator password: ")
        repeated = getpass.getpass("Repeat password: ")
        if password != repeated:
            raise RuntimeError("The entered passwords do not match.")
    path = configure_admin_credentials(username, password)
    print(f"Administrator credentials saved as a salted password hash: {path}")


def main() -> None:
    args = parse_args()
    if args.configure_admin:
        configure_admin()
        return
    cfg = manager.load_manager_config()
    run_web_manager(
        manager,
        cfg,
        host=str(args.host or "0.0.0.0"),
        port=int(args.port or 8765),
        open_browser=not bool(args.no_browser),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Web manager stopped.")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
