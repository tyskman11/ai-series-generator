#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from backend_common import print_runtime_error
from project_local_voice_backend import main as xtts_main


if __name__ == "__main__":
    try:
        raise SystemExit(xtts_main())
    except Exception as exc:
        raise SystemExit(print_runtime_error(exc))
