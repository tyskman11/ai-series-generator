#!/usr/bin/env python3
from __future__ import annotations

from pipeline_common import ensure_project_structure, error, headline, ok, rerun_in_runtime


def main() -> None:
    rerun_in_runtime()
    headline("Projektstruktur erstellen")
    ensure_project_structure(write_config_file=True)
    ok("Projektstruktur und Konfiguration sind bereit.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
