#%%
import json

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import fontManager
import seaborn as sns

from eap.utils import display_name_dict
from plotting_utils import make_whole_fig, color_labels
from pyfonts import load_font

# load font
font = load_font(
   font_url="https://github.com/dolbydu/font/blob/master/Serif/Palatino/Palatino%20Linotype.ttf?raw=true",
)

fontManager.addfont(font.get_file())
sns.set(font=font.get_name())
matplotlib.rcParams['font.family'] = font.get_name()

filename = 'averages'

real_task_names = ['gendered-pronoun', 'sva', 'npi', 'hypernymy-comma', 'wug'] + ['ioi', 'colored-objects', 'entity-tracking', 'greater-than-multitoken', 'fact-retrieval-comma', 'fact-retrieval-rev']

subplot_titles = ["Node Intersection over Union", "Edge Intersection over Union", "Cross-Task Faithfulness"]

# def is_formal(x):
#     return any(formal_task in x for formal_task in [display_name_dict[y] for y in ['gendered-pronoun', 'sva', 'hypernymy-comma', 'npi', 'wug']])

# def color_labels(ax, x=True, y=True):
#     if x:
#         for label in ax.get_xticklabels():
#             if is_formal(label.get_text()):
#                 label.set_color('blue')
#     if y:
#         for label in ax.get_yticklabels():
#             if is_formal(label.get_text()):
#                 label.set_color('blue')

# def make_whole_fig(data, graphs_names, counts, title, colorbar_title, xlabel='Task', ylabel='Task') -> Tuple[Figure, Axes]:
#     if counts is not None:
#         y_label = [f'{name} ({count})' for name, count in zip(graphs_names, counts)]
#     else:
#         y_label = graphs_names

#     fig, ax = plt.subplots()
#     fig:Figure = fig
#     ax:Axes = ax
#     sns.heatmap(data, cmap='Blues', cbar_kws={'label':colorbar_title}, ax=ax)
#     ax.set_xticklabels(graphs_names, rotation=45, ha='right', font=font)
#     ax.set_yticklabels(y_label, rotation=0, font=font)

#     ax.set_title(title, font=font, fontsize=18)
#     ax.set_xlabel(xlabel, font=font)
#     ax.set_ylabel(ylabel, font=font)

#     color_labels(ax)

#     if len(y_label) > 5:
#         ax.axhline(5, color='black')
#     ax.axvline(5, color='black') 

#     fig.tight_layout()
#     return fig, ax

all_ious = []
all_recalls = []
all_cross_task_faithfulnesses = []
all_weighted_edge_recalls = []
all_absolute_weighted_edge_recalls = []
all_geds = []
all_edge_proportions = []

for model_name in ['meta-llama/Meta-Llama-3-8B', 'Qwen/Qwen2.5-7B', 'allenai/OLMo-7B-hf', 'mistralai/Mistral-7B-v0.3', 'google/gemma-2-2b']:
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


    with open(f'../jaccard/{model_name_noslash}/json/overlap_analysis_edge.json') as f:
        overlap_analysis_dict = json.load(f)

    ious = overlap_analysis_dict[f'edge_ious']
    iou_z = np.array([[ious[f'{t1}_{t2}'] for t2 in real_task_names] for t1 in real_task_names])

    recalls = overlap_analysis_dict[f'edge_recalls']
    recall_z = np.array([[recalls[f'{t1}_{t2}'] for t2 in real_task_names] for t1 in real_task_names])

    geds = overlap_analysis_dict[f'geds']
    ged_z = np.array([[geds[f'{t1}_{t2}'] for t2 in real_task_names] for t1 in real_task_names])

    weighted_edge_recalls = overlap_analysis_dict[f'weighted_edge_overlaps']
    weighted_edge_recalls_z = np.array([[weighted_edge_recalls[f'{t1}_{t2}'] for t2 in real_task_names] for t1 in real_task_names])

    absolute_weighted_edge_recalls = overlap_analysis_dict[f'absolute_weighted_edge_overlaps']
    absolute_weighted_edge_recalls_z = np.array([[absolute_weighted_edge_recalls[f'{t1}_{t2}'] for t2 in real_task_names] for t1 in real_task_names])

    edge_counts = overlap_analysis_dict[f'edge_counts']

    edge_counts_by_task = [edge_counts[name] for name in real_task_names]
    all_edge_proportions.append([x / edge_counts['whole_graph'] for x in edge_counts_by_task])
    
    all_ious.append(iou_z)
    all_recalls.append(recall_z)
    all_geds.append(ged_z)
    all_weighted_edge_recalls.append(weighted_edge_recalls_z)
    all_absolute_weighted_edge_recalls.append(absolute_weighted_edge_recalls_z)
    all_cross_task_faithfulnesses.append(task_circuit_normalized)

    iou_fig, _ = make_whole_fig(iou_z, display_names, edge_counts_by_task, f"Edge Intersection over Union ({model_name_noslash})", 'IoU')
    iou_fig.savefig(f'paper-plots-models/edge_iou_{model_name_noslash}.pdf')

    recall_fig, _ = make_whole_fig(recall_z, display_names,edge_counts_by_task, f"Edge Recall ({model_name_noslash})", "Recall")
    recall_fig.savefig(f'paper-plots-models/edge_recall_{model_name_noslash}.pdf')

    faithfulness_fig, _ = make_whole_fig(task_circuit_normalized, display_names, edge_counts_by_task, f"Cross-Task Faithfulness ({model_name_noslash})", 'Faithfulness',)
    faithfulness_fig.savefig(f'paper-plots-models/cross_task_faithfulness_{model_name_noslash}.pdf')


all_ious = np.stack(all_ious).mean(0)
all_recalls = np.stack(all_recalls).mean(0)
all_cross_task_faithfulnesses = np.stack(all_cross_task_faithfulnesses).mean(0)
all_edge_proportions = np.stack(all_edge_proportions).mean(0)
all_geds = np.stack(all_geds).mean(0)
all_weighted_edge_recalls = np.stack(all_weighted_edge_recalls).mean(0)
all_absolute_weighted_edge_recalls = np.stack(all_absolute_weighted_edge_recalls).mean(0)
#%%
iou_df = pd.DataFrame(all_ious, index=real_task_names, columns=real_task_names)
recall_df = pd.DataFrame(all_recalls, index=real_task_names, columns=real_task_names)
all_cross_task_faithfulnesses_df = pd.DataFrame(all_cross_task_faithfulnesses, index=real_task_names, columns=real_task_names)
weighted_edge_recalls_df = pd.DataFrame(all_weighted_edge_recalls, index=real_task_names, columns=real_task_names)
absolute_weighted_edge_recalls_df = pd.DataFrame(all_absolute_weighted_edge_recalls, index=real_task_names, columns=real_task_names)

iou_df.to_csv('averaged_data/edge_iou.csv')
recall_df.to_csv('averaged_data/edge_recall.csv')
all_cross_task_faithfulnesses_df.to_csv('averaged_data/edge_cross_task_faithfulness.csv')
weighted_edge_recalls_df.to_csv('averaged_data/eweighted_edge_recalls.csv')
absolute_weighted_edge_recalls_df.to_csv('averaged_data/absolute_weighted_edge_recalls.csv')


all_edge_percents = [f'{x*100:.2f}%' for x in all_edge_proportions]
all_iou_fig, all_iou_ax = make_whole_fig(all_ious, display_names, all_edge_percents, "Edge Intersection over Union", 'IoU')
all_iou_fig.savefig(f'paper-plots/edge_iou_all.pdf')

# for txt in all_iou_ax.get_yticklabels():
#     if 'SVA' in txt.get_text():
#         txt.set_fontweight('bold')
#     text = txt.get_text()
#     text_no_paren = text.split(' (')[0]
#     txt.set_text(text_no_paren)

# for txt in all_iou_ax.get_xticklabels():
#     if 'Greater-Than' in txt.get_text():
#         txt.set_fontweight('bold')

all_iou_ax.set_xlabel('')
#all_iou_ax.set_xticklabels([])
all_iou_ax.set_ylabel('')
all_iou_ax.set_yticklabels([txt.get_text().split(' (')[0] for txt in all_iou_ax.get_yticklabels()])
#all_iou_fig.set_size_inches(6.4, 3.6)
all_iou_fig.tight_layout()
all_iou_fig.savefig(f'paper-plots/edge_iou_all.svg')

all_recall_fig, _ = make_whole_fig(all_recalls, display_names, all_edge_percents, "Edge Recall", "Recall", xlabel='Hypothesis Task', ylabel='Reference Task')
all_recall_fig.savefig(f'paper-plots/edge_recall_all.pdf')

all_faithfulness_fig, all_faithfulness_ax = make_whole_fig(all_cross_task_faithfulnesses, display_names, all_edge_percents, "Edge-Circuit Cross-Task Faithfulness", 'Faithfulness', xlabel='Hypothesis Task', ylabel='Reference Task')
all_faithfulness_fig.savefig(f'paper-plots/cross_task_faithfulness_all.pdf')

# for txt in all_faithfulness_ax.get_yticklabels():
#     if 'SVA' in txt.get_text():
#         txt.set_fontweight('bold')
#     text = txt.get_text()
#     text_no_paren = text.split(' (')[0]
#     txt.set_text(text_no_paren)

# for txt in all_faithfulness_ax.get_xticklabels():
#     if 'Greater-Than' in txt.get_text():
#         txt.set_fontweight('bold')

all_faithfulness_ax.set_xlabel('')
all_faithfulness_ax.set_ylabel('')
all_faithfulness_ax.set_yticklabels([txt.get_text().split(' (')[0] for txt in all_iou_ax.get_yticklabels()])
all_faithfulness_fig.tight_layout()
#print(all_faithfulness_fig.get_size_inches())
all_faithfulness_fig.savefig(f'paper-plots/cross_task_faithfulness_all.svg')


all_weighted_edge_recalls_fig, _ = make_whole_fig(all_weighted_edge_recalls, display_names, all_edge_percents, "Weighted Edge Recall", 'Recall', xlabel='Hypothesis Task', ylabel='Reference Task')
#all_weighted_edge_recalls_fig.savefig(f'paper-plots/weighted_edge_recall_all.pdf')

all_absolute_weighted_edge_recalls_fig, _ = make_whole_fig(all_absolute_weighted_edge_recalls, display_names, all_edge_percents, "Absolute Weighted Edge Recall", 'Recall', xlabel='Hypothesis Task', ylabel='Reference Task')
#all_absolute_weighted_edge_recalls_fig.savefig(f'paper-plots/absolute_weighted_edge_recall_all.pdf')

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
    plot_dendrogram(model, ax, leaf_label_func=lambda i: display_name_dict[real_task_names[i]])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Distance")
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(),rotation=60, ha='right', fontsize=13)
    color_labels(ax)
    fig.tight_layout()
    if save:
        fig.savefig(f'paper-plots/{filename}.pdf')
# %%
make_dendrogram(all_ious, 'edge_iou_dendrogram', 'Edge IoU')
make_dendrogram(all_recalls, 'edge_recall_dendrogram_ref', 'Edge Recall', xlabel='Reference Task')
make_dendrogram(all_recalls.T, 'edge_recall_dendrogram_hyp', 'Edge Recall', xlabel='Hypothesis Task')
make_dendrogram(all_cross_task_faithfulnesses, 'cross_task_faithfulness_dendrogram_ref', 'Cross-Task Faithfulness', xlabel='Reference Task')
make_dendrogram(all_cross_task_faithfulnesses.T, 'cross_task_faithfulness_dendrogram_hyp', 'Cross-Task Faithfulness', xlabel='Hypothesis Task')
# %%
