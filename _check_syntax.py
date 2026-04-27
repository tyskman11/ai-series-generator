import py_compile
import glob
import sys

files = glob.glob("*.py")
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
