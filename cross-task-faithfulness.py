#%%
import json
from json.decoder import JSONDecodeError
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm
import pandas as pd
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_graph

from dataset import EAPDataset
from metrics import get_metric, task_to_defaults
# %%
parser = ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='google/gemma-2-2b')
parser.add_argument('--level', type=str, default='edge')
parser.add_argument('--method', type=str, default='EAP-IG-inputs')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--head', type=int, default=500)
parser.add_argument('--threshold', type=float, default=0.85)

model_to_batch_size = {
    'google/gemma-2-2b': 20/2,
    'google/gemma-2-9b': 1, # :(
    'meta-llama/Meta-Llama-3-8B': 4,
    'Qwen/Qwen2-7B': 5,
    'Qwen/Qwen2.5-7B': 5,
    'allenai/OLMo-7B-hf': 5,
    'allenai/OLMo-1B-hf': 35,
    'mistralai/Mistral-7B-v0.3': 5,
    'Qwen/Qwen2-1.5B': 25,
    'Qwen/Qwen2.5-1.5B': 25,
}

args = parser.parse_args()
model_name = args.model 
method = args.method
level = args.level
assert level in ['edge', 'neuron', 'node']

model_name_noslash = model_name.split('/')[-1]
model = HookedTransformer.from_pretrained(model_name,center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device='cuda',
)
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

model_batch_size = model_to_batch_size[model_name]

task_info = {}
task_graphs = {}
for file in tqdm(Path(f'graphs/{method}/{model_name_noslash}').iterdir()):
    if file.suffix == '.pt':
        task = file.stem
        if level == 'edge' and ('_node' in task or '_neuron' in task):
            continue
        if level != 'edge':
            if f'_{level}' not in task:
                continue
            task = task.replace(f'_{level}', '')
        try:
            csv_file = f'results/{method}/faithfulness/{model_name_noslash}/csv/{task}.csv' if level=='edge' else f'results/{method}/faithfulness/{model_name_noslash}/csv/{task}_{level}.csv'
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f'No faithfulness csv for {task}')
            continue
        faithfulnesses = ((df['faithfulness'] - df['corrupted_baseline'])/(df['baseline'] - df['corrupted_baseline'])).to_numpy()
        requested_edge_counts = df['requested_edges'].to_numpy()
        if np.any(faithfulnesses >= args.threshold):
            edge_count = requested_edge_counts[faithfulnesses >= args.threshold][0]
            faithfulness = df['faithfulness'][faithfulnesses >= args.threshold].to_list()[0]
        else:
            edge_count = requested_edge_counts[-1]
            faithfulness = df['faithfulness'].to_list()[-1]
        try:
            g = Graph.from_pt(file)
        except AssertionError as e:
            print("Got badly formatted circuit for", task)
            raise e

        g.apply_topn(edge_count, absolute=True, level=level)
        task_graphs[task] = g
        task_info[task] = {'baseline': df['baseline'].to_list()[0], 'corrupted_baseline': df['corrupted_baseline'].to_list()[0], 'requested_edges': int(edge_count), 'actual_edges': g.count_included_edges(), f'{task}_circuit': float(faithfulness)}

print(f'Found {len(task_info)} tasks with circuits for {model_name} ({level})')
Path(f'results/{method}/cross-task/{model_name_noslash}/json').mkdir(exist_ok=True, parents=True)
# populate with previously-computed cross-task faithfulness
json_file = f'results/{method}/cross-task/{model_name_noslash}/json/circuit_info.json' if level == 'edge' else f'results/{method}/cross-task/{model_name_noslash}/json/circuit_info_{level}.json'
if not args.overwrite and Path(json_file).exists():
    try:
        with open(json_file, 'r') as f:
            old_task_info = json.load(f)
        for old_task, old_info in old_task_info.items():
            if old_task in task_info:
                for k, v in old_info.items():
                    if k not in task_info[old_task]:
                        task_info[old_task][k] = v
    except JSONDecodeError:
        print(f'Failed to load old circuit info for {model_name}')
        pass

for task1 in tqdm(task_info.keys()):
    dataset = EAPDataset(task1, model_name)
    if args.head is not None:
        dataset.head(args.head)
    metric_name, batch_size_multiplier = task_to_defaults[task1]
    batch_size = int(model_batch_size * batch_size_multiplier)
    dataloader = dataset.to_dataloader(batch_size)
    metric = get_metric(metric_name, task1, model=model)
    for task2 in task_info.keys():
        if f'{task2}_circuit' in task_info[task1]:
            #print(f'Skipping {task2} run on {task1}; already found in {str(json_file)}')
            continue
        graph = task_graphs[task2]
        perf = evaluate_graph(model, graph, dataloader, partial(metric, mean=False, loss=False),quiet=True).mean().item()
        task_info[task1][f'{task2}_circuit'] = perf

with open(json_file, 'w') as f:
    json.dump(task_info, f)