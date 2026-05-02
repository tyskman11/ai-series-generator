#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
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
        description="Review workflow queue system."
    )
    parser.add_argument(
        "--episode-id", help="Episode ID to add to queue."
    )
    parser.add_argument(
        "--action", default="list",
        choices=["list", "add", "approve", "reject", "complete"],
        help="Queue action."
    )
    parser.add_argument(
        "--priority", type=int, default=5,
        help="Priority (1=highest, 10=lowest)."
    )
    parser.add_argument(
        "--notes",
        help="Review notes."
    )
    return parser.parse_args()


QUEUE_FILE = "review_queue.json"

QUEUE_STATES = ["pending", "in_review", "approved", "rejected", "completed"]


def load_queue() -> dict:
    queue_path = resolve_project_path(f"generation/{QUEUE_FILE}")
    
    if queue_path.exists():
        return read_json(queue_path, {"queue": [], "history": []})
    
    return {"queue": [], "history": []}


def save_queue(queue: dict) -> None:
    queue_path = resolve_project_path(f"generation/{QUEUE_FILE}")
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps(queue, indent=2), encoding="utf-8")


def add_to_queue(episode_id: str, priority: int, notes: str) -> None:
    queue = load_queue()
    
    entry = {
        "episode_id": episode_id,
        "priority": priority,
        "notes": notes or "",
        "status": "pending",
        "added_at": datetime.now().isoformat(),
    }
    
    queue["queue"].append(entry)
    queue["queue"].sort(key=lambda x: (x["priority"], x["added_at"]))
    
    save_queue(queue)
    ok(f"Added {episode_id} to review queue")


def list_queue() -> None:
    queue = load_queue()
    
    if not queue["queue"]:
        info("Review queue is empty")
        return
    
    info(f"Review Queue ({len(queue['queue'])} items):")
    for i, item in enumerate(queue["queue"], 1):
        print(f"  {i}. [{item['priority']}] {item['episode_id']} - {item['status']}")
        if item.get("notes"):
            print(f"     Notes: {item['notes']}")


def update_status(episode_id: str, new_status: str) -> None:
    queue = load_queue()
    
    found = False
    for item in queue["queue"]:
        if item["episode_id"] == episode_id:
            item["status"] = new_status
            item["updated_at"] = datetime.now().isoformat()
            found = True
    
    if not found:
        info(f"Episode {episode_id} not in queue")
        return
    
    if new_status in ["approved", "rejected", "completed"]:
        queue["history"].extend([i for i in queue["queue"] if i["episode_id"] == episode_id])
        queue["queue"] = [i for i in queue["queue"] if i["episode_id"] != episode_id]
    
    save_queue(queue)
    ok(f"Updated {episode_id} to {new_status}")


def main() -> None:
    args = parse_args()
    headline("Review Queue")
    cfg = load_config()
    
    if args.action == "list":
        list_queue()
    elif args.action == "add":
        if not args.episode_id:
            info("--episode-id required for add action")
            return
        add_to_queue(args.episode_id, args.priority, args.notes or "")
    elif args.action == "approve":
        if not args.episode_id:
            info("--episode-id required")
            return
        update_status(args.episode_id, "approved")
    elif args.action == "reject":
        if not args.episode_id:
            info("--episode-id required")
            return
        update_status(args.episode_id, "rejected")
    elif args.action == "complete":
        if not args.episode_id:
            info("--episode-id required")
            return
        update_status(args.episode_id, "completed")


if __name__ == "__main__":
    main()
