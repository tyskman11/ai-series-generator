#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "ai_series_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import argparse
import json
import shutil
from pathlib import Path

from support_scripts.pipeline_common import (
    headline,
    info,
    ok,
    error,
    load_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore project from backup.")
    parser.add_argument("--backup", required=True, help="Backup folder to restore.")
    parser.add_argument("--target", help="Target project folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    headline("Restore Project")
    cfg = load_config()
    
    backup_path = Path(args.backup)
    meta_path = backup_path / "backup_meta.json"
    
    if not meta_path.exists():
        error("No backup metadata found")
        return
    
    meta = json.loads(meta_path.read_text())
    print(f"Backup date: {meta.get('backup_date')}")
    
    target_root = Path(args.target) if args.target else Path.cwd() / "ai_series_project"
    
    for item in backup_path.iterdir():
        if item.is_dir() and item.name != "backup_meta.json":
            dst = target_root / item.name
            info(f"Restoring {item.name}...")
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
    
    ok(f"Restore complete: {target_root}")


if __name__ == "__main__":
    main()
