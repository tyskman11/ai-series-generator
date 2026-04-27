content = open("pipeline_common.py", encoding="utf-8").read()
idx = content.find("GLOBAL_STEP")
if idx >= 0:
    print(content[max(0,idx-200):idx+800])
else:
    print("GLOBAL_STEP not found")
    # try alternative
    idx2 = content.find("STEP_")
    if idx2 >= 0:
        print(content[max(0,idx2-200):idx2+800])
