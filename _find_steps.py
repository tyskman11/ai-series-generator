content = open("pipeline_common.py", encoding="utf-8").read()
# Suche nach mark_step_started, mark_step_completed, mark_step_failed
import re
matches = re.findall(r'mark_step_(started|completed|failed)\s*\(\s*["\']([^"\']+)["\']', content)
print(f"Found {len(matches)} mark_step calls in pipeline_common.py")
for m in matches[:20]:
    print(f"  {m[0]}: {m[1]}")

# Suche nach STEP-Definitionen
step_defs = re.findall(r'(?:^|\n)\s*(?:STEP|step)[_\s]+[=\[]', content)
print(f"\nStep definitions: {len(step_defs)}")

# Suche nach completed_step_state oder similar
idx = content.find("completed_step")
if idx >= 0:
    print(f"\ncompleted_step found at {idx}")
    print(content[max(0,idx-100):idx+400])
