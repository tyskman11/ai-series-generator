#!/usr/bin/env python3
from __future__ import annotations

import argparse

from pipeline_common import (
    add_shared_worker_arguments,
    distributed_item_lease,
    distributed_step_runtime_root,
    ensure_project_structure,
    error,
    headline,
    info,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    rerun_in_runtime,
    shared_worker_id_for_args,
    shared_workers_enabled_for_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create project structure and configuration")
    add_shared_worker_arguments(parser)
    return parser.parse_args()


def main() -> None:
    rerun_in_runtime()
    args = parse_args()
    headline("Create Project Structure")
    cfg = ensure_project_structure(write_config_file=True)
    worker_id = shared_worker_id_for_args(args)
    shared_workers = shared_workers_enabled_for_args(cfg, args)
    mark_step_started("01_setup_project", "global")
    if shared_workers:
        info(f"Shared NAS workers: enabled ({worker_id})")
    lease_manager = distributed_item_lease(
        root=distributed_step_runtime_root("01_setup_project", "global"),
        lease_name="global",
        cfg=cfg,
        worker_id=worker_id,
        enabled=shared_workers,
        meta={"step": "01_setup_project", "scope": "global", "worker_id": worker_id},
    )
    acquired = lease_manager.__enter__()
    if not acquired:
        info("Project setup is already running on another worker.")
        lease_manager.__exit__(None, None, None)
        return
    try:
        ensure_project_structure(write_config_file=True)
        mark_step_completed("01_setup_project", "global")
        ok("Project structure and configuration are ready.")
    except Exception as exc:
        mark_step_failed("01_setup_project", str(exc), "global")
        raise
    finally:
        lease_manager.__exit__(None, None, None)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise

