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
from datetime import datetime
from pathlib import Path

from support_scripts.pipeline_common import (
    headline,
    info,
    ok,
    load_config,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backup project to cloud/NAS.")
    parser.add_argument("--target", required=True, help="Backup target path.")
    parser.add_argument("--include-models", action="store_true", help="Include trained models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    headline("Backup Project")
    cfg = load_config()
    project_root = Path(cfg.get("project_root", Path.cwd()) / "ai_series_project")
    
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    
    backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_root = target / backup_name
    
    dirs_to_backup = ["data", "characters", "generation", "series_bible", "configs"]
    if args.include_models:
        dirs_to_backup.extend(["training"])
    
    for dir_name in dirs_to_backup:
        src = project_root / dir_name
        if src.exists():
            dst = backup_root / dir_name
            info(f"Copying {dir_name}...")
            shutil.copytree(src, dst, dirs_exist_ok=True)
    
    meta = {
        "backup_date": datetime.now().isoformat(),
        "source": str(project_root),
        "include_models": args.include_models,
    }
    (backup_root / "backup_meta.json").write_text(json.dumps(meta, indent=2))
    
    ok(f"Backup complete: {backup_root}")


if __name__ == "__main__":
    main()
