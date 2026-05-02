#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from support_scripts.pipeline_common import (
    headline,
    ok,
    info,
    load_config,
    resolve_project_path,
    read_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive old episodes with retention policy."
    )
    parser.add_argument(
        "--retention-days", type=int, default=90,
        help="Days to keep episodes before archiving."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be archived without archiving."
    )
    parser.add_argument(
        "--archive-path",
        help="Custom archive destination path."
    )
    return parser.parse_args()


def get_episode_age_days(episode_path: Path) -> int:
    try:
        mtime = datetime.fromtimestamp(episode_path.stat().st_mtime)
        age = datetime.now() - mtime
        return age.days
    except Exception:
        return 0


def list_episodes_for_archive(
    packages_dir: Path,
    retention_days: int
) -> list[tuple[Path, int]]:
    archived = []
    
    for ep_dir in packages_dir.iterdir():
        if not ep_dir.is_dir():
            continue
        
        age = get_episode_age_days(ep_dir)
        
        if age >= retention_days:
            archived.append((ep_dir, age))
    
    return sorted(archived, key=lambda x: x[1], reverse=True)


def archive_episode(
    episode_path: Path,
    archive_root: Path,
    age_days: int,
    dry_run: bool
) -> bool:
    archive_dest = archive_root / episode_path.name
    
    if dry_run:
        info(f"Would archive: {episode_path.name} (age: {age_days} days)")
        return True
    
    try:
        archive_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(episode_path), str(archive_dest))
        
        manifest = {
            "original_path": str(episode_path),
            "archived_at": datetime.now().isoformat(),
            "age_days": age_days,
        }
        
        manifest_path = archive_dest / "archive_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        
        ok(f"Archived: {episode_path.name}")
        return True
    except Exception as e:
        info(f"Failed to archive {episode_path.name}: {e}")
        return False


def cleanup_empty_dirs(packages_dir: Path) -> None:
    for d in packages_dir.iterdir():
        if d.is_dir() and not any(d.iterdir()):
            try:
                d.rmdir()
            except Exception:
                pass


def main() -> None:
    args = parse_args()
    headline("Episode Archive")
    cfg = load_config()
    
    packages_dir = resolve_project_path("generation/final_episode_packages")
    archive_root = Path(args.archive_path) if args.archive_path else packages_dir / "archive"
    
    episodes = list_episodes_for_archive(packages_dir, args.retention_days)
    
    if not episodes:
        info("No episodes to archive")
        return
    
    info(f"Found {len(episodes)} episodes older than {args.retention_days} days")
    
    if args.dry_run:
        for ep, age in episodes:
            info(f"  -> {ep.name} ({age} days)")
    else:
        archived_count = 0
        for ep, age in episodes:
            if archive_episode(ep, archive_root, age, args.dry_run):
                archived_count += 1
        
        cleanup_empty_dirs(packages_dir)
        
        ok(f"Archived {archived_count} episodes")


if __name__ == "__main__":
    main()
