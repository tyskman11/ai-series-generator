#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "ai_series_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from support_scripts.manage_character_relationships import main as relationship_main
from support_scripts.pipeline_common import error, rerun_in_runtime


def main() -> None:
    rerun_in_runtime()
    if len(sys.argv) == 1:
        sys.argv.append("--gui")
    relationship_main()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error(str(exc))
        raise
