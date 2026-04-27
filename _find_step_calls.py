import re
import glob

# Finde alle mark_step Aufrufe in nummerierten Skripten
files = sorted(glob.glob("[0-9]*.py"))
results = {}
for f in files:
    content = open(f, encoding="utf-8").read()
    started = re.findall(r'mark_step_started\s*\(\s*["\']([^"\']+)["\']', content)
    completed = re.findall(r'mark_step_completed\s*\(\s*["\']([^"\']+)["\']', content)
    failed = re.findall(r'mark_step_failed\s*\(\s*["\']([^"\']+)["\']', content)
    if started or completed or failed:
        results[f] = {
            "started": started,
            "completed": completed,
            "failed": failed
        }

for f, data in results.items():
    print(f"\n{f}:")
    print(f"  started:   {data['started']}")
    print(f"  completed: {data['completed']}")
    print(f"  failed:    {data['failed']}")
