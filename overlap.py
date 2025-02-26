#%%
from typing import Dict
import time
from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm
import pandas as pd
import numpy as np
from overlap_utils_json import make_comparison_heatmap

from eap.virtual_graph import VirtualGraph

# %%
parser = ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='google/gemma-2-2b')
parser.add_argument('--method', type=str, default='EAP-IG-inputs')
parser.add_argument('--level', type=str, default='edge')
args = parser.parse_args()

model_name = args.model # 'google/gemma-2-2b'
method = args.method
model_name_noslash = model_name.split('/')[-1]
level = args.level
jaccard_path = Path(f'results/{method}/jaccard/{model_name_noslash}')
jaccard_path.mkdir(exist_ok=True, parents=True)

threshold = 0.85
for sub in ['png', 'pdf', 'csv', 'json']:
    (jaccard_path / sub).mkdir(exist_ok=True, parents=True) 

graphs: Dict[str, VirtualGraph] = {}
s = time.time()
for file in tqdm(Path(f'graphs/{method}/{model_name_noslash}').iterdir()):
    if file.suffix == '.pt':
        task = file.stem
        if level == 'edge':
            if '_node' in task or '_neuron' in task:
                continue
            csv_file = f'results/{method}/faithfulness/{model_name_noslash}/csv/{task}.csv'
            graph_file = f'graphs/{method}/{model_name_noslash}/{task}.pt'
        else:
            if f'_{level}' not in task:
                continue
            task = task.replace(f'_{level}', '')
            csv_file = f'results/{method}/faithfulness/{model_name_noslash}/csv/{task}_{level}.csv'
            graph_file = f'graphs/{method}/{model_name_noslash}/{task}_{level}.pt'
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
print('Done loading! Took', time.time() - s, 'seconds')

make_comparison_heatmap(graphs, jaccard_path, level=level)

# %%
