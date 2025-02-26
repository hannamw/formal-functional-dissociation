#%%
from typing import Dict, List 
import re
import json

d:Dict[str, List[str]] = {}
expect_new = True
skip = 0
with open('battig_updated.txt', 'r') as f:
    file = f.readlines()

for i, line in enumerate(file):
    if expect_new:
        current_category = line.split('.')[1][3:].strip().lower()
        d[current_category] = []
        expect_new = False
        skip = 2
    elif line == '\n':
        expect_new = True
    elif skip:
        skip -=1 
    else:
        if i < len(file) - 1 and file[i+1][0] == ' ':
            continue

        item = line.strip('\t').split('\t')[0].strip()
        item = re.sub('\(.*\)', '', item).lower()
        d[current_category].append(item)

with open('battig_parsed.json', 'w') as g:
    json.dump(d, g)