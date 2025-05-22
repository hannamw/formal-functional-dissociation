#%%
from functools import partial 
from pathlib import Path 
from argparse import ArgumentParser
import json

import pandas as pd
import numpy as np
import torch
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
from tqdm import tqdm

from eap.evaluate import evaluate_baseline

from dataset import EAPDataset
from metrics import get_metric, task_to_defaults

parser = ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='google/gemma-2-2b')
parser.add_argument('--head', type=int, default=500)
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()

model_name = args.model
model_name_noslash = model_name.split('/')[-1]
model = HookedTransformer.from_pretrained(model_name,center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device='cuda',
    dtype=torch.float16
)

model_to_batch_size = {
    'google/gemma-2-2b': 10,
    'google/gemma-2-9b': 1, # :(
    'meta-llama/Meta-Llama-3-8B': 5,
    'Qwen/Qwen2.5-7B': 6,
    'mistralai/Mistral-7B-v0.3': 6,
    'allenai/OLMo-7B-hf': 6,
    
}
model_batch_size = model_to_batch_size[model_name]
tasks = ['ioi', 
         'fact-retrieval-comma', 
         'gendered-pronoun', 
         'sva', 
         'entity-tracking', 
         'colored-objects', # evaluate starting from here
         'npi', 
         'hypernymy-comma', 
         'fact-retrieval-rev', 
         'greater-than-multitoken',
         'echo',
         'wug'
         ]
if 'llama' in model_name:
    tasks += ['math', 'math-add', 'math-sub', 'math-mul'] 
    tasks += ['counterfact-citizen_of', 'counterfact-official_language', 'counterfact-has_profession', 'counterfact-plays_instrument']
    tasks += ['fact-retrieval-comma-purefunc', 'greater-than-multitoken-purefunc', 'colored-objects-purefunc', 'entity-tracking-purefunc']
    

# if 'gemma' in model_name:
#     tasks += ['sva-multilingual-en', 'sva-multilingual-nl', 'sva-multilingual-de', 'sva-multilingual-fr', 'fact-retrieval-rev-multilingual-en', 
#               'fact-retrieval-rev-multilingual-nl', 'fact-retrieval-rev-multilingual-de', 'fact-retrieval-rev-multilingual-fr']

print("Evaluating", model_name)
accuracies = {}
accuracy_path = Path(f'results/EAP-IG-inputs/faithfulness/{model_name_noslash}/json')
accuracy_path.mkdir(exist_ok=True, parents=True)
if Path(accuracy_path / 'accuracies.json').exists():
    with open(accuracy_path / 'accuracies.json') as f:
        accuracies = json.load(f)

for task in tasks:
    if task in accuracies and not args.overwrite:
        print(f"Skipping {task} as it is already evaluated")
        continue

    metric_name, batch_size_multiplier = task_to_defaults[task]
    batch_size = int(model_batch_size * batch_size_multiplier)
    print("Evaluating", task)
    ds = EAPDataset(task, model_name)
    np.random.seed(42)
    ds.shuffle()
    ds.head(args.head)
    dataloader = ds.to_dataloader(batch_size)
    eval_dataloader = ds.to_dataloader(int(3 * batch_size))
    
    accuracy_metric_name = 'accuracy-prob' if 'prob' in metric_name else 'accuracy'
    accuracy_metric = get_metric(accuracy_metric_name, task, model=model)

    baseline = evaluate_baseline(model, eval_dataloader, partial(accuracy_metric, mean=False, loss=False)).mean().item()
    accuracies[task] = baseline

with open(accuracy_path / 'accuracies.json', 'w') as f:
    json.dump(accuracies, f)
