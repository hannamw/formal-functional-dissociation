#%%
import json
from collections import Counter, defaultdict

with open('counterfact.json', 'r') as f:
    data = json.load(f)
# %%
c = Counter(item['requested_rewrite']['prompt'] for item in data)
c2 = Counter(item['requested_rewrite']['relation_id'] for item in data)
# %%
d = defaultdict(set)
for item in data:
    d[item['requested_rewrite']['relation_id']].add(item['requested_rewrite']['prompt'])
# %%
[item['requested_rewrite']['subject'] for item in data if item['requested_rewrite']['relation_id'] == 'P407']
# %%
[item['requested_rewrite']['subject'] for item in data if item['requested_rewrite']['relation_id'] == 'P37']
# %%
[item['requested_rewrite']['subject'] for item in data if item['requested_rewrite']['relation_id'] == 'P39']
# %%
[item['requested_rewrite']['subject'] for item in data if item['requested_rewrite']['relation_id'] == 'P136']
# %%
[item['requested_rewrite']['subject'] for item in data if item['requested_rewrite']['relation_id'] == 'P106']
# %%
[item['requested_rewrite'] for item in data if item['requested_rewrite']['relation_id'] == 'P1303']
# %%
[item['requested_rewrite'] for item in data if item['requested_rewrite']['relation_id'] == 'P27']
# %%
[item['requested_rewrite'] for item in data if item['requested_rewrite']['relation_id'] == 'P37']
# %%
[item['requested_rewrite'] for item in data if item['requested_rewrite']['relation_id'] == 'P106']
# %%
[item['requested_rewrite'] for item in data if item['requested_rewrite']['relation_id'] == 'P276'][10:50]
# %%
