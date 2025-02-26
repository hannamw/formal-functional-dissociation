# %%
import random 
import json
from collections import Counter, defaultdict

import pandas as pd
from transformers import AutoTokenizer
import torch 
from tqdm import tqdm 

import nodebox_linguistics_extended as nle 

from hypernymy_utils import format_hyponym
# model_name = 'gpt2-medium'
model_name = 'EleutherAI/pythia-160m'
model_name_noslash = model_name.split('/')[-1]
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open('battig_manual_filtered.json', 'r') as g:
    d = json.load(g)

token_length_mapping = defaultdict(lambda: defaultdict(list))
token_length_dict = defaultdict(Counter)
for hypernym, hyponyms in d.items():
    for hyponym in hyponyms:
        hyponym = format_hyponym(hyponym, hypernym)
        token_length = len(tokenizer(f' {hyponym}', add_special_tokens=False).input_ids)
        token_length_dict[hypernym][token_length] += 1
        token_length_mapping[hypernym][token_length].append(hyponym)
# %%
# generate sets for each 
valid_length_sets = defaultdict(set)
for hypernym, c in token_length_dict.items():
    keys = list(c.keys())
    for i, k in enumerate(keys):
        if c[k] > 1:
            valid_length_sets[hypernym].add(frozenset([k, k]))

        if i != len(keys) - 1:
            for k2 in keys[i+1:]:
                valid_length_sets[hypernym].add(frozenset([k, k2]))
# now reverse it ugh
all_frozensets = set(s for sets in valid_length_sets.values() for s in sets)
frozensets_hypernyms_mapping = {fset:[hypernym for hypernym, valid_sets in valid_length_sets.items() if fset in valid_sets]  for fset in all_frozensets}

for fset, l in frozensets_hypernyms_mapping.items():
    assert len(l) > 1, f"found fset without proper alternatives {fset}: {l}"

# %%
with open('valid_answers.json', 'r') as f:
    valid_answers = json.load(f)

valid_answers = {k: [x[0] for x in v] for k, v in valid_answers.items()}

dataset = {k:[] for k in ['clean', 'h1', 'h2', 'hypernym', 'answers', 'answers_idx', 'corrupted', 'c1', 'c2', 'corrupted_hypernym', 'corrupted_answers', 'corrupted_answers_idx']}

def total_cross_product(xs):
    return [(xs[i], xs[j]) for i in range(len(xs)) for j in range(len(xs)) if i != j]

for hypernym, hyponyms in tqdm(list(d.items())):
    for h1, h2 in total_cross_product(hyponyms):
        h1 = format_hyponym(h1, hypernym)
        h2 = format_hyponym(h2, hypernym)
        h1_length = len(tokenizer(f' {h1}', add_special_tokens=False).input_ids)
        h2_length = len(tokenizer(f' {h2}', add_special_tokens=False).input_ids)
        lens = frozenset([h1_length, h2_length])
        clean = f' {h1} and {h2} are two types of'
        answers = valid_answers[hypernym]
        answers_idx = [tokenizer(f' {answer.strip()}').input_ids[0] for answer in answers]

        valid_corrupted_hypernyms = frozensets_hypernyms_mapping[lens]
        corrupted_hypernym = hypernym
        while corrupted_hypernym == hypernym:
            corrupted_hypernym = random.choice(valid_corrupted_hypernyms)

        c1 = random.choice(token_length_mapping[corrupted_hypernym][h1_length])
        c2 = c1 
        while c2 == c1:
            c2 = random.choice(token_length_mapping[corrupted_hypernym][h2_length])
        corrupted = f' {c1} and {c2} are two types of'
        corrupted_answers = valid_answers[corrupted_hypernym]
        corrupted_answers_idx = [tokenizer(f' {answer.strip()}', add_special_tokens=False).input_ids[0] for answer in corrupted_answers]

        for k, v in zip(['clean', 'h1', 'h2', 'hypernym', 'answers', 'answers_idx', 'corrupted', 'c1', 'c2', 'corrupted_hypernym', 'corrupted_answers', 'corrupted_answers_idx'], [clean, h1, h2, hypernym, answers, answers_idx, corrupted, c1, c2, corrupted_hypernym, corrupted_answers, corrupted_answers_idx]):
            dataset[k].append(v)

df = pd.DataFrame.from_dict(dataset)
df.to_csv(f'dataset_{model_name_noslash}.csv', index=False)
# %%
