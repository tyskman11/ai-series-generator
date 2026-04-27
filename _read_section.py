lines = open('pipeline_common.py', encoding='utf-8').readlines()
for i, l in enumerate(lines[3920:4010], start=3921):
    print(f'{i}: {l.rstrip()}')
