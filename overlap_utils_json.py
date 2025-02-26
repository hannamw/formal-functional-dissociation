#%%
from typing import List, Dict
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from scipy.stats import hypergeom

from eap.graph import Graph


def make_graph_hm(z, graphs_names, title, counts=None, colorbar_title='IoU', return_heatmap=False):
    if counts is not None:
        y_label = [f'{name} ({count})' for name, count in zip(graphs_names, counts)]
    else:
        y_label = graphs_names
    heat = go.Heatmap(z=z,
                  x=graphs_names,
                  y=y_label,
                  colorbar={"title": colorbar_title},
                  xgap=1, ygap=1,
                  colorscale='Blues',
                  colorbar_thickness=20,
                  colorbar_ticklen=3,
                   )
    layout = go.Layout(title_text=title, title_x=0.5, 
                    width=600, height=600,
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    yaxis_autorange='reversed')
    
    fig=go.Figure(data=[heat], layout=layout)   
    if return_heatmap:
        return fig, heat     
    return fig

def graph_analysis(g1: Graph, g2: Graph):
    edge_intersection = (g1.in_graph & g2.in_graph).float().sum()
    edge_union = (g1.in_graph | g2.in_graph).float().sum()
    x = edge_intersection
    M  = g1.real_edge_mask.sum() #len(g1.edges) 
    n = g1.in_graph.float().sum()
    N = g2.in_graph.float().sum()
    iou_edge = edge_intersection / edge_union
    p_edge = 1 - hypergeom.cdf(x, M, n, N)

    node_intersection = (g1.nodes_in_graph[1:-1] & g2.nodes_in_graph[1:-1]).float().sum() 
    node_union = (g1.nodes_in_graph[1:-1] | g2.nodes_in_graph[1:-1]).float().sum() 
    #print(len(node_intersection), len(node_union), len(node_intersection) / len(node_union))
    x = node_intersection
    M = g1.n_forward #len(g1.nodes) - 2
    n = g1.nodes_in_graph[1:-1].float().sum()
    N = g2.nodes_in_graph[1:-1].float().sum()
    p_node = 1 - hypergeom.cdf(x, M, n, N)
    iou_node = node_intersection / node_union
    #print(1 - hypergeom.cdf(x, M, N, n))

    # directional measures:
    edge_overlap = edge_intersection / g1.count_included_edges()
    node_overlap = node_intersection / g1.count_included_nodes()

    if g1.scores is not None:
        weighted_edge_overlap = g1.scores[g2.in_graph].sum() / g1.scores.sum()
        absolute_weighted_edge_overlap = g1.scores[g2.in_graph].abs().sum() / g1.scores.abs().sum()
    else:
        weighted_edge_overlap = torch.tensor(0)
        absolute_weighted_edge_overlap = torch.tensor(0)

    if g1.nodes_scores is not None:
        weighted_node_overlap = g1.nodes_scores[g2.nodes_in_graph].sum() / g1.nodes_scores.sum()
        absolute_weighted_node_overlap = g1.nodes_scores[g2.nodes_in_graph].abs().sum() / g1.nodes_scores.abs().sum()
    else:
        weighted_node_overlap = torch.tensor(0)
        absolute_weighted_node_overlap = torch.tensor(0)

    edge_overlap_min = edge_intersection / min(g1.count_included_edges(), g2.count_included_edges())
    node_overlap_min = node_intersection / min(g1.count_included_nodes(), g2.count_included_nodes())

    g1_real_scores = g1.scores.flatten()[g1.real_edge_mask.flatten()]
    g2_real_scores = g2.scores.flatten()[g2.real_edge_mask.flatten()]
    graph_edit_distance = F.cosine_similarity(g1_real_scores, g2_real_scores, dim=0) #(g1_real_scores - g2_real_scores).abs().sum()
    
    g1_z_scores = (g1_real_scores - g1_real_scores.mean())/g1_real_scores.std()
    g2_z_scores = (g2_real_scores - g2_real_scores.mean())/g2_real_scores.std()
    z_scored_graph_edit_distance = F.cosine_similarity(g1_z_scores, g2_z_scores, dim=0) # (g1_z_scores - g2_z_scores).abs().sum()

    g1_maxmin_scores = (g1_real_scores - g1_real_scores.min())/(g1_real_scores.max() - g1_real_scores.min())
    g2_maxmin_scores = (g2_real_scores - g2_real_scores.min())/(g2_real_scores.max() - g2_real_scores.min())
    maxmin_graph_edit_distance = (g1_maxmin_scores - g2_maxmin_scores).abs().sum()
    cosine_similarity =  F.cosine_similarity(g1_maxmin_scores, g2_maxmin_scores, dim=0)


    return p_edge, iou_edge, p_node, iou_node, edge_overlap, node_overlap, edge_overlap_min, node_overlap_min, graph_edit_distance, z_scored_graph_edit_distance, maxmin_graph_edit_distance, cosine_similarity, weighted_edge_overlap, absolute_weighted_edge_overlap, weighted_node_overlap, absolute_weighted_node_overlap

def neuron_analysis(g1: Graph, g2: Graph):
    intersection = (g1.neurons_in_graph & g2.neurons_in_graph).float().sum()
    union = (g1.neurons_in_graph | g2.neurons_in_graph).float().sum()
    x = intersection
    M  = g1.n_forward * g1.cfg['d_model'] #len(g1.edges) 
    n = g1.neurons_in_graph.float().sum()
    N = g2.neurons_in_graph.float().sum()
    iou = intersection / union
    p_iou = 1 - hypergeom.cdf(x, M, n, N)


    # directional measures:
    recall = intersection / g1.neurons_in_graph.float().sum()

    iom = intersection / min(g1.neurons_in_graph.float().sum(), g2.neurons_in_graph.float().sum())

    weighted_overlap = g1.neurons_scores[g2.neurons_in_graph].sum() / g1.neurons_scores.sum()
    absolute_weighted_overlap = g1.neurons_scores[g2.neurons_in_graph].abs().sum() / g1.neurons_scores.abs().sum()

    return p_iou, iou, recall, iom, weighted_overlap, absolute_weighted_overlap

def make_array(d: Dict[str, float], names:List[str], lower_only:bool=False):
    n = len(names)
    arr = np.zeros((n, n))
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            arr[i, j] = d[name1, name2]
    if lower_only:
        arr[np.triu_indices(n)] = np.nan
    return arr

def make_comparison_heatmap(graphs: Dict[str, Graph], path: Path, level='edge'):
    edge_counts = {}
    node_counts = {}
    graphs_names = list(graphs.keys())

    edge_ious, node_ious, edge_recalls, node_recalls, edge_ioms, node_ioms, geds, geds_z, geds_minmax, cosine_similarities = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    weighted_edge_overlaps, absolute_weighted_edge_overlaps, weighted_node_overlaps, absolute_weighted_node_overlaps = {}, {}, {}, {}

    if level == 'neuron':
        neuron_counts, neuron_ious, neuron_recalls, neuron_ioms, weighted_neuron_overlaps, absolute_weighted_neuron_overlaps = {}, {}, {}, {}, {}, {}
    
    for n1, g1 in graphs.items():
        edge_counts[n1] = g1.count_included_edges()
        node_counts[n1] = g1.count_included_nodes()
        if level == 'neuron':
            neuron_counts[n1] = g1.neurons_in_graph.float().sum().item()
        for n2, g2 in graphs.items():
            results = graph_analysis(g1,g2)
            p_edge, iou_edge, p_node, iou_node, edge_overlap, node_overlap, edge_overlap_min, node_overlap_min, ged, z_ged, m_ged, cosine_similarity, weighted_edge_overlap, absolute_weighted_edge_overlap, weighted_node_overlap, absolute_weighted_node_overlap = (result.item() for result in results)
            edge_ious[n1, n2] = iou_edge
            node_ious[n1, n2] = iou_node
            edge_recalls[n1, n2] = edge_overlap 
            node_recalls[n1, n2] = node_overlap
            edge_ioms[n1, n2] = edge_overlap_min 
            node_ioms[n1, n2] = node_overlap_min
            geds[n1, n2] = ged
            geds_z[n1, n2] = z_ged
            geds_minmax[n1, n2] = m_ged
            cosine_similarities[n1, n2] = cosine_similarity
            weighted_edge_overlaps[n1, n2] = weighted_edge_overlap
            absolute_weighted_edge_overlaps[n1, n2] = absolute_weighted_edge_overlap
            weighted_node_overlaps[n1, n2] = weighted_node_overlap
            absolute_weighted_node_overlaps[n1, n2] = absolute_weighted_node_overlap

            if level == 'neuron':
                results = neuron_analysis(g1,g2)
                p_iou, iou, recall, iom, weighted_overlap, absolute_weighted_overlap = (result.item() for result in results)
                neuron_ious[n1, n2] = iou
                neuron_recalls[n1, n2] = recall
                neuron_ioms[n1, n2] = iom
                weighted_neuron_overlaps[n1, n2] = weighted_overlap
                absolute_weighted_neuron_overlaps[n1, n2] = absolute_weighted_overlap

    # g1 here is just an arbitrary graph for the model; it doesn't matter which circuit we choose
    edge_counts['whole_graph'] = g1.real_edge_mask.sum().item()
    node_counts['whole_graph'] = g1.n_forward

    output_dict = {}
    for metric_dict, metric_name in zip([edge_ious, node_ious, edge_ioms, node_ioms, edge_recalls, node_recalls, geds, geds_z, geds_minmax, cosine_similarities, edge_counts, node_counts, weighted_edge_overlaps, absolute_weighted_edge_overlaps, weighted_node_overlaps, absolute_weighted_node_overlaps], ['edge_ious', 'node_ious', 'edge_ioms', 'node_ioms', 'edge_recalls', 'node_recalls', 'geds', 'geds_z', 'geds_minmax', 'cosine_similarities', 'edge_counts', 'node_counts', 'weighted_edge_overlaps', 'absolute_weighted_edge_overlaps', 'weighted_node_overlaps', 'absolute_weighted_node_overlaps']):
        if 'counts' in metric_name:
            reformatted_metric_dict = metric_dict
        else:
            reformatted_metric_dict = {f'{k[0]}_{k[1]}': v for k, v in metric_dict.items()}
        output_dict[metric_name] = reformatted_metric_dict

    if level == 'neuron':
        neuron_counts['whole_graph'] = g1.n_forward * g1.cfg['d_model']
        for metric_dict, metric_name in zip([neuron_counts, neuron_ious, neuron_recalls, neuron_ioms, weighted_neuron_overlaps, absolute_weighted_neuron_overlaps], ['neuron_counts', 'neuron_ious', 'neuron_recalls', 'neuron_ioms', 'weighted_neuron_overlaps', 'absolute_weighted_neuron_overlaps']):
            if 'counts' in metric_name:
                reformatted_metric_dict = metric_dict
            else:
                reformatted_metric_dict = {f'{k[0]}_{k[1]}': v for k, v in metric_dict.items()}
            output_dict[metric_name] = reformatted_metric_dict

    
    filename = 'overlap_analysis'
    try:
        with open(path / f'json/{filename}_{level}.json', 'w') as f:
            json.dump(output_dict, f)
    except Exception as e:
        print(f'Failed to save {filename}_{level} to json')
        raise e