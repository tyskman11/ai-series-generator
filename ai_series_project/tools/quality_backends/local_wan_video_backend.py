#!/usr/bin/env python3
"""Entry point for the approved project-local Wan video runner.

The implementation remains in the older module during the compatibility
migration so existing manifests can still resume, but new configurations only
invoke this Wan-named entry point.
"""
from __future__ import annotations

from local_ltx_video_backend import main


if __name__ == "__main__":
    raise SystemExit(main())
