#%%
import json

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import fontManager, FontProperties
import seaborn as sns
from scipy.stats import pearsonr

from eap.utils import display_name_dict
from pyfonts import load_font
from plotting_utils import make_whole_fig, color_labels

# load font
font = load_font(
   font_url="https://github.com/dolbydu/font/blob/master/Serif/Palatino/Palatino%20Linotype.ttf?raw=true",
)

fontManager.addfont(font.get_file())
sns.set(font=font.get_name())
matplotlib.rcParams['font.family'] = font.get_name()


level = 'neuron'
filename = 'averages'

real_task_names = ['gendered-pronoun', 'sva', 'npi', 'hypernymy-comma', 'wug'] + ['ioi', 'colored-objects', 'entity-tracking', 'greater-than-multitoken', 'fact-retrieval-comma', 'fact-retrieval-rev']


all_ious = []
all_recalls = []
all_cross_task_faithfulnesses = []
all_cross_task_faithfulnesses_normalized = []
all_weighted_recalls = []
all_absolute_weighted_recalls = []
all_proportions = []

for model_name in ['meta-llama/Meta-Llama-3-8B', 'Qwen/Qwen2.5-7B', 'allenai/OLMo-7B-hf', 'mistralai/Mistral-7B-v0.3', 'google/gemma-2-2b']:
    model_name_noslash = model_name.split('/')[-1]
    display_names = [display_name_dict[name] for name in real_task_names]

    with open(f'{model_name_noslash}/json/circuit_info_{level}.json') as f:
        data = json.load(f)

    arr = np.array([[data[t1][f'{t2}_circuit'] for t2 in real_task_names] for t1 in real_task_names])
    clean_baselines = np.array([data[t]['baseline'] for t in real_task_names]).reshape(-1,1)
    corrupted_baselines = np.array([data[t]['corrupted_baseline'] for t in real_task_names]).reshape(-1,1)
    task_circuit_baselines = np.array([data[t][f'{t}_circuit'] for t in real_task_names]).reshape(-1,1)

    # normalize by task-specific baseline/corrupted baseline
    normalized = (arr - corrupted_baselines)/(clean_baselines - corrupted_baselines)

    # normalize by task-specific circuit baseline performance (lower than overall baseline) / corrupted_baseline
    task_circuit_normalized = (arr - corrupted_baselines)/(task_circuit_baselines - corrupted_baselines)


    with open(f'../jaccard/{model_name_noslash}/json/overlap_analysis_{level}.json') as f:
        overlap_analysis_dict = json.load(f)

    ious = overlap_analysis_dict[f'{level}_ious']
    iou_z = np.array([[ious[f'{t1}_{t2}'] for t2 in real_task_names] for t1 in real_task_names])

    recalls = overlap_analysis_dict[f'{level}_recalls']
    recall_z = np.array([[recalls[f'{t1}_{t2}'] for t2 in real_task_names] for t1 in real_task_names])

    weighted_recalls = overlap_analysis_dict[f'weighted_{level}_overlaps']
    weighted_recall_z = np.array([[weighted_recalls[f'{t1}_{t2}'] for t2 in real_task_names] for t1 in real_task_names])

    absolute_weighted_recalls = overlap_analysis_dict[f'absolute_weighted_{level}_overlaps']
    absolute_weighted_recall_z = np.array([[absolute_weighted_recalls[f'{t1}_{t2}'] for t2 in real_task_names] for t1 in real_task_names])

    counts = overlap_analysis_dict[f'{level}_counts']

    counts_by_task = [counts[name] for name in real_task_names]
    all_proportions.append([x / counts['whole_graph'] for x in counts_by_task])
    
    all_ious.append(iou_z)
    all_recalls.append(recall_z)
    all_weighted_recalls.append(weighted_recall_z)
    all_absolute_weighted_recalls.append(absolute_weighted_recall_z)
    all_cross_task_faithfulnesses.append(task_circuit_normalized)
    all_cross_task_faithfulnesses_normalized.append(normalized)

    iou_fig, _ = make_whole_fig(iou_z, display_names, counts_by_task, f"{level.capitalize()} Intersection over Union ({model_name_noslash})", 'IoU')
    iou_fig.savefig(f'paper-plots-models/{level}_iou_{model_name_noslash}.pdf')

    recall_fig, _ = make_whole_fig(recall_z, display_names,counts_by_task, f"{level.capitalize()} Recall ({model_name_noslash})", "Recall")
    recall_fig.savefig(f'paper-plots-models/{level}_recall_{model_name_noslash}.pdf')

    faithfulness_fig, _ = make_whole_fig(task_circuit_normalized, display_names, counts_by_task, f"({level.capitalize()}) Cross-Task Faithfulness ({model_name_noslash})", 'Faithfulness',)
    faithfulness_fig.savefig(f'paper-plots-models/{level}_cross_task_faithfulness_{model_name_noslash}.pdf')


all_ious = np.stack(all_ious).mean(0)
all_recalls = np.stack(all_recalls).mean(0)
all_weighted_recalls = np.stack(all_weighted_recalls).mean(0)
all_absolute_weighted_recalls = np.stack(all_absolute_weighted_recalls).mean(0)
all_cross_task_faithfulnesses = np.stack(all_cross_task_faithfulnesses).mean(0)
all_cross_task_faithfulnesses_normalized = np.stack(all_cross_task_faithfulnesses_normalized).mean(0)
all_proportions = np.stack(all_proportions).mean(0)

all_percents = [f'{x*100:.2f}%' for x in all_proportions]
all_iou_fig, ax = make_whole_fig(all_ious, display_names, all_percents, f"{level.capitalize()} Intersection over Union", 'IoU', cbar=(level == 'neuron'), tasks_y=(level == 'node'))
title_artist = ax.get_title()
all_iou_fig.savefig(f'paper-plots/{level}_iou_all.pdf', bbox_inches='tight')

all_recall_fig, _ = make_whole_fig(all_recalls, display_names, all_percents, f"{level.capitalize()} Recall", "Recall", xlabel='Hypothesis Task', ylabel='Reference Task', cbar=(level == 'neuron'), tasks_y=(level == 'node'))
all_recall_fig.savefig(f'paper-plots/{level}_recall_all.pdf', bbox_inches='tight')

all_faithfulness_fig, _ = make_whole_fig(all_cross_task_faithfulnesses, display_names, all_percents, f"{level.capitalize()}-Circuit Cross-Task Faithfulness", 'Faithfulness', xlabel='Hypothesis Task', ylabel='Reference Task', cbar=(level == 'neuron'), tasks_y=(level == 'node'))
all_faithfulness_fig.savefig(f'paper-plots/{level}_cross_task_faithfulness_all.pdf', bbox_inches='tight')

all_weighted_recall_fig, _ = make_whole_fig(all_weighted_recalls, display_names, all_percents, f"{level.capitalize()} Weighted Recall", "Weighted Recall", xlabel='Hypothesis Task', ylabel='Reference Task')
#all_weighted_recall_fig.savefig(f'paper-plots/{level}_weighted_recall_all.pdf')

all_absolute_weighted_recall_fig, _ = make_whole_fig(all_absolute_weighted_recalls, display_names, all_percents, f"{level.capitalize()} Absolute Weighted Recall", "Absolute Weighted Recall", xlabel='Hypothesis Task', ylabel='Reference Task')
#all_absolute_weighted_recall_fig.savefig(f'paper-plots/{level}_absolute_weighted_recall_all.pdf')

#%%
def plot_dendrogram(model, ax, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, ax=ax, **kwargs)

# setting distance_threshold=0 ensures we compute the full tree.
def make_dendrogram(data, filename, metric, xlabel='Task', save=True):
    fig, ax = plt.subplots(figsize=(6,6))
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(data)
    ax.set_title(f"Hierarchical Clustering Dendrogram\nfor {metric}", fontsize=18)
    plot_dendrogram(model, ax, leaf_label_func=lambda i: display_name_dict[real_task_names[i]], )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Distance")
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(),rotation=60, ha='right', fontsize=13)
    color_labels(ax)
    fig.tight_layout()
    fig.show()
    if save:
        fig.savefig(f'paper-plots/{filename}.pdf')
# %%
make_dendrogram(all_ious, f'{level}_iou_dendrogram', f'{level.capitalize()} IoU')
make_dendrogram(all_recalls, f'{level}_recall_dendrogram_ref', f'{level.capitalize()} Recall', xlabel='Reference Task')
make_dendrogram(all_recalls.T, f'{level}_recall_dendrogram_hyp', f'{level.capitalize()} Recall', xlabel='Hypothesis Task')
make_dendrogram(all_cross_task_faithfulnesses, f'{level}_cross_task_faithfulness_dendrogram_ref', f'({level.capitalize()}) Cross-Task Faithfulness', xlabel='Reference Task')
make_dendrogram(all_cross_task_faithfulnesses.T, f'{level}_cross_task_faithfulness_dendrogram_hyp', f'({level.capitalize()}) Cross-Task Faithfulness', xlabel='Hypothesis Task')
# %%
