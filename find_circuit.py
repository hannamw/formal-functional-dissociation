#%%
from functools import partial 
from pathlib import Path 
from argparse import ArgumentParser
import os

import pandas as pd
import numpy as np
import torch
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
from tqdm import tqdm

from eap.graph import Graph
from eap.attribute import attribute
from eap.evaluate import evaluate_graph, evaluate_baseline

from dataset import EAPDataset
from metrics import get_metric, task_to_defaults

parser = ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='google/gemma-2-2b')
parser.add_argument('--method', type=str, default='EAP-IG-inputs') #clean-corrupted
parser.add_argument('--head', type=int, default=500)
parser.add_argument('--threshold', type=float, default=0.85)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('-t', '--task', type=str, default=None)
args = parser.parse_args()

model_name = args.model
method = args.method
model_name_noslash = model_name.split('/')[-1]
model = HookedTransformer.from_pretrained(model_name,center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device='cuda',
    dtype=torch.float16
)
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

model_to_batch_size = {
    'google/gemma-2-2b': 10,
    'google/gemma-2-9b': 1, # :(
    'meta-llama/Meta-Llama-3-8B': 4,
    'Qwen/Qwen2-7B': 5,
    'Qwen/Qwen2.5-7B': 5,
    'mistralai/Mistral-7B-v0.3': 5,
    'Qwen/Qwen2-1.5B': 25,
    'Qwen/Qwen2.5-1.5B': 25,
    'allenai/OLMo-1B-hf': 35,
    'allenai/OLMo-7B-hf': 5,
    'Qwen/Qwen2-0.5B': 75,
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
    tasks += ['math', 'math-add', 'math-sub', 'math-mul'] + ['counterfact-citizen_of', 'counterfact-official_language', 'counterfact-has_profession', 'counterfact-plays_instrument']
    
if 'gemma' in model_name:
    tasks += ['sva-multilingual-en', 'sva-multilingual-nl', 'sva-multilingual-de', 'sva-multilingual-fr', 'fact-retrieval-rev-multilingual-en', 'fact-retrieval-rev-multilingual-nl', 'fact-retrieval-rev-multilingual-de', 'fact-retrieval-rev-multilingual-fr']

if args.task is not None:
    tasks = [args.task]

for task in tasks:
    circuit_savepath = f'graphs/{method}/{model_name_noslash}/{task}.pt'
    df_savepath = f'results/{method}/faithfulness/{model_name_noslash}/csv/{task}.csv'
    if Path(circuit_savepath).exists() and Path(df_savepath).exists() and not args.overwrite:
        print(f"Skipping {task} as {circuit_savepath} exists")
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
    attribution_metric = get_metric(metric_name, task, model=model)
    task_metric = get_metric(metric_name, task, model=model)

    baseline = evaluate_baseline(model, eval_dataloader, partial(task_metric, mean=False, loss=False)).mean().item()
    corrupted_baseline = evaluate_baseline(model, eval_dataloader, partial(task_metric, mean=False, loss=False), run_corrupted=True).mean().item()
    
    # Instantiate a graph with a model
    g = Graph.from_model(model)

    # Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)
    attribute(model, g, dataloader, partial(attribution_metric, mean=True, loss=True), method=method, ig_steps=5)

    # Apply a threshold
    g.apply_topn(10000, absolute=True)
    g.prune()
    #gz = g.to_graphviz()
    #gz.draw(f'images/{model_name_noslash}/{task}.png', prog='dot')
    Path(f'graphs/{method}/{model_name_noslash}').mkdir(exist_ok=True, parents=True)
    g.to_pt(circuit_savepath)
    
    n_edges = []
    n_requested_edges = []
    results = []
    target_faithfulness = args.threshold
    stop_faithfulness = 0.95
    s = 0
    e = int(0.025 * len(g.edges))
    step = max(1, int(0.003 * len(g.edges)))
    e2 = int(0.05 * len(g.edges))
    step2 = max(1, int(0.005 * len(g.edges)))
    first_steps = list(range(s,e + 1, step))
    later_steps = list(range(e,e2 + 1, step2))
    steps = first_steps + later_steps
    
    def run_graph(graph: Graph, n_edges):
        graph.apply_topn(n_edges, absolute=True)
        graph.prune()
        n = graph.count_included_edges()
        r = evaluate_graph(model, graph, eval_dataloader, partial(task_metric, mean=False, loss=False), quiet=True).mean().item()
        return n, r
    
    # initial_steps
    for i in tqdm(steps, desc='initial steps'):
        n_requested_edges.append(i)
        n, r = run_graph(g, i)
        n_edges.append(n)
        results.append(r)
        current_faithfulness = (r - corrupted_baseline)/(baseline - corrupted_baseline)
        if current_faithfulness > stop_faithfulness:
            print("broke")
            break
    
    # continuation steps if we didn't hit the target faithfulness
    if current_faithfulness < target_faithfulness:
        steps = range(e2, int(0.1 * len(g.edges)), int(0.01 * len(g.edges)))
        for i in tqdm(steps, desc='continuation steps'):
            n_requested_edges.append(i)
            n, r = run_graph(g, i)
            n_edges.append(n)
            results.append(r)
            current_faithfulness = (r - corrupted_baseline)/(baseline - corrupted_baseline)
            if current_faithfulness > stop_faithfulness:
                print("broke")
                break
    
        # continuation steps if we didn't hit the target faithfulness
        if current_faithfulness < target_faithfulness:
            steps = range(int(0.1 * len(g.edges)), len(g.edges) + 1, int(0.1 * len(g.edges)))
            for i in tqdm(steps, desc='continuation steps pt 2'):
                n_requested_edges.append(i)
                n, r = run_graph(g, i)
                n_edges.append(n)
                results.append(r)
                current_faithfulness = (r - corrupted_baseline)/(baseline - corrupted_baseline)
                if current_faithfulness > stop_faithfulness:
                    print("broke")
                    break
            
    # zoom in steps
    requested_edge_array = np.array(n_requested_edges)
    edge_array = np.array(n_edges)
    result_array = np.array(results)
    result_array = (result_array - corrupted_baseline)/(baseline - corrupted_baseline)
    
    min_val = requested_edge_array[result_array < target_faithfulness - 0.05][-1] if np.any(result_array < target_faithfulness - 0.05) else 0
    max_val = requested_edge_array[result_array > target_faithfulness + 0.05][0] if np.any(result_array > target_faithfulness + 0.05) else len(g.edges)
    steps = range(min_val + ((max_val - min_val) // 10), max_val + 1, (max_val - min_val) // 10)
    for i in tqdm(steps, desc='zoom in steps'):
        n_requested_edges.append(i)
        n, r = run_graph(g, i)
        n_edges.append(n)
        results.append(r)

    n_requested_edges = np.array(n_requested_edges)
    n_edges = np.array(n_edges)
    results = np.array(results)
    edge_order = np.argsort(n_edges)
    n_requested_edges = n_requested_edges[edge_order].tolist()
    n_edges = n_edges[edge_order].tolist()
    results = results[edge_order].tolist()

    
    d = {'baseline':[baseline] * len(n_edges), 
        'corrupted_baseline':[corrupted_baseline] * len(n_edges),
        'requested_edges': n_requested_edges,
        'edges': n_edges,
        'faithfulness': results}

    df = pd.DataFrame.from_dict(d)
    Path(f'results/{method}/faithfulness/{model_name_noslash}/csv').mkdir(exist_ok=True, parents=True)
    df.to_csv(f'results/{method}/faithfulness/{model_name_noslash}/csv/{task}.csv', index=False)
    
    fig, ax = plt.subplots()
    ax.plot(n_requested_edges, [baseline] * len(n_requested_edges), linestyle='dotted', label='clean baseline')
    ax.plot(n_requested_edges, [corrupted_baseline] * len(n_requested_edges), linestyle='dotted', label='corrupted baseline')
    ax.plot(n_edges, results, label='Faithfulness')
    ax.legend()
    ax.set_xlabel(f'Edges included (/{len(g.edges)})')
    ax.set_ylabel(f'{metric_name}')
    ax.set_title(f'{task} EAP-IG ({model_name_noslash})')
    fig.show()

    Path(f'results/{method}/faithfulness/{model_name_noslash}/png').mkdir(exist_ok=True, parents=True)
    Path(f'results/{method}/faithfulness/{model_name_noslash}/pdf').mkdir(exist_ok=True, parents=True)
    fig.savefig(f'results/{method}/faithfulness/{model_name_noslash}/png/{task}.png')
    fig.savefig(f'results/{method}/faithfulness/{model_name_noslash}/pdf/{task}.pdf')
# %%
