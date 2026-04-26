import sys
start = int(sys.argv[1])
end = int(sys.argv[2])
with open("08_train_series_model.py", encoding="utf-8") as f:
    lines = f.readlines()
print("".join(lines[start:end]))
