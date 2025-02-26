#%%
from typing import Dict
import time
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import numpy as np

from eap.virtual_graph import VirtualGraph
from eap.utils import display_name_dict
from plotting_utils import make_whole_fig

# %%
method='EAP-IG-inputs'
level = 'edge'

models = ['meta-llama/Meta-Llama-3-8B', 'Qwen/Qwen2.5-7B', 'allenai/OLMo-7B-hf', 'mistralai/Mistral-7B-v0.3', 'google/gemma-2-2b']
real_task_names = ['gendered-pronoun', 'sva', 'npi', 'hypernymy-comma', 'wug'] + ['ioi', 'colored-objects', 'entity-tracking', 'greater-than-multitoken', 'fact-retrieval-comma', 'fact-retrieval-rev']
all_ious = []
all_recalls = []    
all_edge_proportions = []

for model_name in models:
    model_name_noslash = model_name.split('/')[-1]
    jaccard_path = Path(f'results/{method}/jaccard/{model_name_noslash}')
    jaccard_path.mkdir(exist_ok=True, parents=True)

    threshold = 0.85
    for sub in ['png', 'pdf', 'csv', 'json']:
        (jaccard_path / sub).mkdir(exist_ok=True, parents=True) 

    graphs: Dict[str, VirtualGraph] = {}
    s = time.time()
    for file in tqdm(Path(f'../../../graphs/{method}/{model_name_noslash}').iterdir()):
        if file.suffix == '.pt':
            task = file.stem
            if level == 'edge':
                if '_node' in task or '_neuron' in task:
                    continue
                csv_file = f'../faithfulness/{model_name_noslash}/csv/{task}.csv'
                graph_file = f'../../../graphs/{method}/{model_name_noslash}/{task}.pt'
            else:
                if f'_{level}' not in task:
                    continue
                task = task.replace(f'_{level}', '')
                csv_file = f'../faithfulness/{model_name_noslash}/csv/{task}_{level}.csv'
                graph_file = f'../../../graphs/{method}/{model_name_noslash}/{task}_{level}.pt'
            try:
                df = pd.read_csv(csv_file)
            except FileNotFoundError:
                print(f'{model_name_noslash}: No faithfulness csv for {task}')
                continue
            graphs[task] = (VirtualGraph.from_pt(graph_file))
            faithfulnesses = ((df['faithfulness'] - df['corrupted_baseline'])/(df['baseline'] - df['corrupted_baseline'])).to_numpy()
            requested_edge_counts = df['requested_edges'].to_numpy()
            if np.any(faithfulnesses >= threshold):
                edge_count = requested_edge_counts[faithfulnesses >= threshold][0]
            else:
                edge_count = requested_edge_counts[-1]
            try:
                graphs[task].apply_topn(edge_count, True, level=level)
            except Exception as e:
                print(f'Error in {task} (filepath: {file}): {e}')
                raise e

    # arbitrarily grab a task with a graph
    intersection_graph_task = list(graphs.keys())[0]
    graph_file = f'../../../graphs/{method}/{model_name_noslash}/{task}.pt' if level == 'edge' else f'graphs/{method}/{model_name_noslash}/{task}_{level}.pt'


    intersection_graph = VirtualGraph.from_pt(graph_file)
    if level == 'edge':
        intersection_mat = intersection_graph.in_graph
    elif level == 'node':
        intersection_mat = intersection_graph.nodes_in_graph
    elif level == 'neuron':
        intersection_mat = intersection_graph.neurons_in_graph

    intersection_mat[:] = True

    for graph in graphs.values():
        mat = graph.in_graph if level == 'edge' else graph.nodes_in_graph if level == 'node' else graph.neurons_in_graph
        intersection_mat &= mat

    for graph in graphs.values():
        mat = graph.in_graph if level == 'edge' else graph.nodes_in_graph if level == 'node' else graph.neurons_in_graph
        mat &= ~intersection_mat

    ious = np.zeros((len(real_task_names), len(real_task_names)))
    recalls = np.zeros((len(real_task_names), len(real_task_names)))

    edge_proportions = []
    for i, t1 in enumerate(real_task_names):
        mat1 = graphs[t1].in_graph if level == 'edge' else graphs[t1].nodes_in_graph if level == 'node' else graphs[t1].neurons_in_graph
        edge_proportions.append(mat1.float().sum() / mat1.numel())
        for j, t2 in enumerate(real_task_names):
            mat2 = graphs[t2].in_graph if level == 'edge' else graphs[t2].nodes_in_graph if level == 'node' else graphs[t2].neurons_in_graph
            ious[i,j] = (mat1 & mat2).float().sum() / (mat1 | mat2).float().sum()
            recalls[i,j] = (mat1 & mat2).float().sum() / mat1.float().sum()
    all_ious.append(ious)
    all_recalls.append(recalls)
    all_edge_proportions.append(edge_proportions)
#%%
all_ious_models = np.array(all_ious).mean(axis=0)
all_recalls_models = np.array(all_recalls).mean(axis=0)
all_edge_proportions_models = np.array(all_edge_proportions).mean(axis=0)

display_names = [display_name_dict[x] for x in real_task_names]
all_edge_percents = [f'{x*100:.2f}%' for x in all_edge_proportions_models]
all_iou_fig, all_iou_ax = make_whole_fig(all_ious_models, display_names, all_edge_percents, "Edge Intersection over Union", 'IoU')
all_recall_fig, all_recall_ax = make_whole_fig(all_recalls_models, display_names, all_edge_percents, "Edge Recall", 'Recall')
# %%
