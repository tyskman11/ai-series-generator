import os
import py_compile
import sys
from pathlib import Path

EXCLUDE_DIRS = {"__pycache__", "runtime", ".venv", "venv", "ai_series_project"}

def collect_py_files(root: Path) -> list[str]:
    result: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Exclude unwanted directories in-place to prevent os.walk from descending
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                result.append(str(Path(dirpath) / fn))
    return sorted(result)

files = collect_py_files(Path.cwd())
errors = []
for f in files:
    try:
        py_compile.compile(f, doraise=True)
    except py_compile.PyCompileError as e:
        errors.append(str(e))

if errors:
    for e in errors:
        print(f"ERROR: {e}")
    sys.exit(1)
else:
    print(f"{len(files)} files OK")
    sys.exit(0)
