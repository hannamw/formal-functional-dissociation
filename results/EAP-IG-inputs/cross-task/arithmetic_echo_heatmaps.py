#%%
from functools import partial
from typing import Tuple
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager, FontProperties
import seaborn as sns

from eap.utils import display_name_dict
from pyfonts import load_font
from plotting_utils import make_whole_fig as make_whole_fig_
make_whole_fig = partial(make_whole_fig_, size=(8, 2.9))

# load font
font = load_font(
   font_url="https://github.com/dolbydu/font/blob/master/Serif/Palatino/Palatino%20Linotype.ttf?raw=true",
)

fontManager.addfont(font.get_file())
sns.set(font=font.get_name())
matplotlib.rcParams['font.family'] = font.get_name()

filename = 'averages'

real_task_names = ['gendered-pronoun', 'sva', 'npi', 'hypernymy-comma', 'wug'] + ['ioi', 'colored-objects', 'entity-tracking', 'greater-than-multitoken', 'fact-retrieval-comma', 'fact-retrieval-rev'] + ['math', 'echo']


model_name ='meta-llama/Meta-Llama-3-8B'
model_name_noslash = model_name.split('/')[-1]
display_names = [display_name_dict[name] for name in real_task_names]

with open(f'{model_name_noslash}/json/circuit_info.json') as f:
    data = json.load(f)

arr = np.array([[data[t1][f'{t2}_circuit'] for t2 in real_task_names] for t1 in real_task_names])
clean_baselines = np.array([data[t]['baseline'] for t in real_task_names]).reshape(-1,1)
corrupted_baselines = np.array([data[t]['corrupted_baseline'] for t in real_task_names]).reshape(-1,1)
task_circuit_baselines = np.array([data[t][f'{t}_circuit'] for t in real_task_names]).reshape(-1,1)

# normalize by task-specific baseline/corrupted baseline
normalized = (arr - corrupted_baselines)/(clean_baselines - corrupted_baselines)

# normalize by task-specific circuit baseline performance (lower than overall baseline) / corrupted_baseline
task_circuit_normalized = (arr - corrupted_baselines)/(task_circuit_baselines - corrupted_baselines)

task_circuit_normalized = task_circuit_normalized[-2:]

with open(f'../jaccard/{model_name_noslash}/json/edge_ious.json') as f:
    edge_ious = json.load(f)
iou_z = np.array([[edge_ious[f'{t1}_{t2}'] for t2 in ['math', 'echo']] for t1 in real_task_names])

with open(f'../jaccard/{model_name_noslash}/json/edge_counts.json') as f:
    edge_counts = json.load(f)


edge_counts_by_task = [edge_counts[name] for name in real_task_names]
edge_proportion = [x / edge_counts['whole_graph'] for x in edge_counts_by_task]
edge_percents = [f'{x*100:.2f}%' for x in edge_proportion]

model_name_pretty = "Llama-3-8B"
iou_fig, _= make_whole_fig(iou_z.T, display_names, edge_percents[-2:], f"Edge Intersection over Union ({model_name_pretty})", 'IoU')
iou_fig.show()
iou_fig.savefig(f'paper-plots/edge_iou_arithmetic_echo.pdf')


faithfulness_fig, _ = make_whole_fig(task_circuit_normalized, display_names, edge_percents[-2:], f"Cross-Task Faithfulness ({model_name_pretty})", 'Faithfulness',)
faithfulness_fig.show()
#faithfulness_fig.savefig(f'paper-plots/cross_task_faithfulness.pdf')
# %%
