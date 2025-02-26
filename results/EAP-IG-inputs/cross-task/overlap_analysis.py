#%%
import pickle
import time
from pathlib import Path 
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.axes import Axes
from matplotlib.font_manager import fontManager
import seaborn as sns


from eap.graph import Graph, AttentionNode, MLPNode, InputNode, LogitNode
from pyfonts import load_font

# load font
font = load_font(
   font_url="https://github.com/dolbydu/font/blob/master/Serif/Palatino/Palatino%20Linotype.ttf?raw=true",
)

fontManager.addfont(font.get_file())
sns.set(font=font.get_name())
matplotlib.rcParams['font.family'] = font.get_name()

# %%
info_dicts = []
figs, figs2, figs3 = [], [], []
heats, heats2, heats3 = [], [], []
model_names = [
    'Meta-Llama-3-8B',
    'Mistral-7B-v0.3',
    'Qwen2.5-7B',
    'gemma-2-2b',
    'allenai/OLMo-7B-hf'
]

for model_name in model_names:
    info_dict = {}
    print("Loading", model_name)
    model_name_noslash = model_name.split('/')[-1]
    jaccard_path = Path(f'../jaccard/{model_name_noslash}')
    jaccard_path.mkdir(exist_ok=True, parents=True)

    threshold = 0.85
    for sub in ['png', 'pdf', 'csv']:
        (jaccard_path / sub).mkdir(exist_ok=True, parents=True) 
        
    real_task_names = ['ioi', 
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

    real_graphs = []
    edge_counts = []
    s = time.time()
    for task in real_task_names:
        print('Loading', task)
        real_graphs.append(Graph.from_pt(f'../../../graphs/EAP-IG-inputs/{model_name_noslash}/{task}.pt'))
        df = pd.read_csv(f'../faithfulness/{model_name_noslash}/csv/{task}.csv')
        faithfulnesses = ((df['faithfulness'] - df['corrupted_baseline'])/(df['baseline'] - df['corrupted_baseline'])).to_numpy()
        requested_edge_counts = df['requested_edges'].to_numpy()
        if np.any(faithfulnesses >= threshold):
            edge_count = requested_edge_counts[faithfulnesses >= threshold][0]
        else:
            edge_count = requested_edge_counts[-1]
        edge_counts.append(edge_count)
        real_graphs[-1].apply_topn(edge_count, True)
    print('Done loading! Took', time.time() - s, 'seconds')

    graphs_names = real_task_names
    graphs = real_graphs
    
    working_graph = Graph.from_pt(f'../../../graphs/EAP-IG-inputs/{model_name_noslash}/{task}.pt')
    
    for graph in graphs:
        working_graph.in_graph &= graph.in_graph

    for graph in graphs:
        working_graph.nodes_in_graph &= graph.nodes_in_graph
    
    
    info_dict['n_layers'] = working_graph.cfg.n_layers

    start_layers = []
    end_layers = []
    for edge in working_graph.edges.values():
        if edge.in_graph:
            if isinstance(edge.parent, InputNode):
                start_layers.append(0)
            else:
                start_layers.append(edge.parent.layer + 1)
            
            if isinstance(edge.child, LogitNode):
                end_layers.append(edge.child.layer + 1)
            else:
                end_layers.append(edge.child.layer)
                
    plt.hist(start_layers, alpha=0.5, label='Start Layers')
    plt.hist(end_layers, alpha=0.5, label='End Layers')
    plt.legend(loc='upper right')
    plt.title('Start and End Layers of Edges Shared Between All Circuits')

    info_dict['layers'] = (start_layers, end_layers)
    
    # Create a mapping from node type to a readable name
    node_type_mapping = {
        "<class 'eap.graph.AttentionNode'>": 'Attn',
        "<class 'eap.graph.MLPNode'>": 'MLP',
        "<class 'eap.graph.InputNode'>": 'Input',
        "<class 'eap.graph.LogitNode'>": 'Logits'
    }

    # Count the occurrences of each parent-child node type grouping
    c = Counter()
    for edge in working_graph.edges.values():
        if edge.in_graph:
            parent = node_type_mapping.get(str(type(edge.parent)), 'Unknown')
            child = node_type_mapping.get(str(type(edge.child)), 'Unknown')
            c[(parent, child)] += 1

    # Prepare data for the pie chart
    labels = [f'{parent} -> {child}' for parent, child in c.keys()]
    sizes = list(c.values())


    # Plot the pie chart
    plt.figure(figsize=(10, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Parent-Child Node Type Groupings')
    plt.show()
    # Create a 3x3 heatmap
    heatmap_data = np.zeros((3, 3))

    # Map node types to indices
    parent_node_type_indices = {
        'Input': 0,
        'Attn': 1,
        'MLP': 2
    }

    child_node_type_indices = {
        'Attn': 0,
        'MLP': 1,
        'Logits': 2
    }

    # Fill the heatmap data
    for (parent, child), count in c.items():
        if parent in parent_node_type_indices and child in child_node_type_indices:
            parent_idx = parent_node_type_indices[parent]
            child_idx = child_node_type_indices[child]
            heatmap_data[parent_idx, child_idx] = count

    info_dict['heatmap_data'] = heatmap_data
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Count')
    plt.xticks(ticks=np.arange(3), labels=['Attn', 'MLP', 'Logits'])
    plt.yticks(ticks=np.arange(3), labels=['Input', 'Attn', 'MLP'])
    plt.xlabel('Child Node Type')
    plt.ylabel('Parent Node Type')
    plt.title('Parent-Child Node Type Groupings Heatmap')
    plt.show()

    edge_spans = []
    for edge in working_graph.edges.values():
        if edge.in_graph:
            edge_spans.append(edge.child.layer - edge.parent.layer)
    plt.hist(edge_spans)

    info_dict['edge_spans'] = edge_spans

    info_dicts.append(info_dict)
#%% 
#with open('averaged_data/info_dicts.pkl', 'wb') as f:
#    pickle.dump(info_dicts, f)

#%%
info_dicts = pickle.load(open('averaged_data/info_dicts.pkl', 'rb'))
#%%
heatmap_data = np.stack([info_dict['heatmap_data'] / info_dict['heatmap_data'].sum() for info_dict in info_dicts]).mean(0)
fig, ax = plt.subplots(figsize=(8, 6))
ax: Axes = ax
sns.heatmap(heatmap_data, cmap='Blues', cbar_kws={'label':'Percentage'}, annot=True, fmt=".0%",annot_kws={"size": 18}, ax=ax)
ax.set_xticklabels(['Attn', 'MLP', 'Logits'], fontsize=14)
ax.set_yticklabels(['Input', 'Attn', 'MLP'], fontsize=14)

ax.set_xlabel('Child Node Type', fontsize=16)
ax.set_ylabel('Parent Node Type', fontsize=16)
ax.set_title('Types of Edges Shared by All Circuits', fontsize=18)
fig.tight_layout()
fig.savefig('paper-plots/parent_child_node_type.pdf')
#%%
start_layers = np.concatenate([np.array(info_dict['layers'][0])/info_dict['n_layers'] for info_dict in info_dicts])
end_layers = np.concatenate([np.array(info_dict['layers'][1])/info_dict['n_layers'] for info_dict in info_dicts])
histogram_fig, histogram_ax = plt.subplots(figsize=(8, 7))
histogram_ax.hist(start_layers, alpha=0.5, label='Start Layers')
histogram_ax.hist(end_layers, alpha=0.5, label='End Layers')
histogram_ax.set_xlabel('Layer (Normalized)', fontsize=16)
histogram_ax.set_ylabel('Count', fontsize=16)
histogram_ax.legend(loc='upper right', fontsize=16)
histogram_ax.set_title('Start and End Layers of Edges Shared by All Circuits', fontsize=18)
histogram_fig.tight_layout()
histogram_fig.savefig('paper-plots/start_end_layers.pdf')
# %%
layers_spanned = np.concatenate([info_dict['edge_spans'] for info_dict in info_dicts])
plt.hist(layers_spanned, bins=20)
# %%
