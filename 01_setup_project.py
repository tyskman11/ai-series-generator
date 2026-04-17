#!/usr/bin/env python3
from __future__ import annotations

from pipeline_common import (
    ensure_project_structure,
    error,
    headline,
    mark_step_completed,
    mark_step_failed,
    mark_step_started,
    ok,
    rerun_in_runtime,
)


def main() -> None:
    rerun_in_runtime()
    headline("Projektstruktur erstellen")
    mark_step_started("01_setup_project", "global")
    try:
        ensure_project_structure(write_config_file=True)
        mark_step_completed("01_setup_project", "global")
        ok("Projektstruktur und Konfiguration sind bereit.")
    except Exception as exc:
        mark_step_failed("01_setup_project", str(exc), "global")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
