#%%
from typing import List
from pathlib import Path

import pandas as pd
import numpy as np
import torch.nn.functional as F
import plotly.graph_objects as go
from scipy.stats import hypergeom

from eap.graph import Graph

display_name_dict = {
    'ioi': 'IOI', 
    'greater-than': "Greater-Than (GT)", 
    "greater-than-multitoken": "Greater-Than",
    'greater-than-price': "GT (Price)", 
    'greater-than-sequence': "GT (Sequence)", 
    'gender-bias': 'Gender-Bias', 
    'gendered-pronoun': 'Gendered Pronoun',
    'math': 'Math',
    'math-add': 'Math (Addition)',
    'math-sub': 'Math (Subtraction)',
    'math-mul': 'Math (Multiplication',
    'sva': 'SVA', 
    'fact-retrieval-comma': 'Capital-Country', 
    'fact-retrieval-rev': 'Country-Capital', 
    'hypernymy-comma': 'Hypernymy (,)',
    'hypernymy': 'Hypernymy',
    'npi': 'NPI',
    'colored-objects': 'Colored Objects',
    'entity-tracking': 'Entity Tracking'
}


def make_graph_hm(z, hovertext, graphs_names, title, counts=None, colorbar_title='IoU', return_heatmap=False):
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
                  hovertext=hovertext,
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
    maxmin_graph_edit_distance = F.cosine_similarity(g1_maxmin_scores, g2_maxmin_scores, dim=0) # (g1_maxmin_scores - g2_maxmin_scores).abs().sum()

    return p_edge, iou_edge, p_node, iou_node, edge_overlap, node_overlap, edge_overlap_min, node_overlap_min, graph_edit_distance, z_scored_graph_edit_distance, maxmin_graph_edit_distance

def make_comparison_heatmap(graphs: List[Graph], graphs_names: List[str], edge_thresholds: List[float], title:str, path: Path):
    edge_counts = []
    node_counts = []
    for graph, n_edges in zip(graphs, edge_thresholds):
        graph.apply_topn(n_edges, absolute=True)
        graph.prune()

    pes = np.zeros((len(graphs), len(graphs)))
    ies = np.zeros((len(graphs), len(graphs)))
    pns = np.zeros((len(graphs), len(graphs)))
    ins = np.zeros((len(graphs), len(graphs)))
    eos = np.zeros((len(graphs), len(graphs)))
    nos = np.zeros((len(graphs), len(graphs)))
    ems = np.zeros((len(graphs), len(graphs)))
    nms = np.zeros((len(graphs), len(graphs)))
    geds = np.zeros((len(graphs), len(graphs)))
    zged = np.zeros((len(graphs), len(graphs)))
    mged = np.zeros((len(graphs), len(graphs)))

    #edge_ious, node_ious, edge_recalls, node_recalls, edge_ioms, node_ioms, geds
    
    for i, (g1, n1) in enumerate(zip(graphs, graphs_names)):
        edge_counts.append(g1.count_included_edges())
        node_counts.append(g1.count_included_nodes())
        for j, (g2, n2) in enumerate(zip(graphs, graphs_names)):
            p_edge, iou_edge, p_node, iou_node, edge_overlap, node_overlap, edge_overlap_min, node_overlap_min, ged, z_ged, m_ged = graph_analysis(g1,g2)
            pes[i,j] = p_edge
            ies[i,j] = iou_edge
            pns[i,j] = p_node
            ins[i,j] = iou_node
            eos[i,j] = edge_overlap 
            nos[i,j] = node_overlap
            ems[i,j] = edge_overlap_min 
            nms[i,j] = node_overlap_min
            geds[i,j] = ged
            zged[i,j] = z_ged
            mged[i,j] = m_ged

    display_names = [display_name_dict[name] for name in graphs_names]

    edge_iou_df = pd.DataFrame.from_dict({gn: ies[:, i] for i, gn in enumerate(graphs_names)})
    edge_iou_df['edge_count'] = edge_counts
    edge_iou_p_df = pd.DataFrame.from_dict({gn: pes[:, i] for i, gn in enumerate(graphs_names)})

    edge_iou_df.to_csv(path / 'csv/edge_ious.csv', index=False)
    edge_iou_p_df.to_csv(path / 'csv/edge_iou_ps.csv', index=False)

    ius = np.triu_indices(len(graphs))
    ies[ius] = np.nan
    pes[ius] = np.nan

    fig = make_graph_hm(ies, pes, display_names, f'Edge Intersection over Union (85%)', edge_counts)
    fig.write_image(path / f'png/{title}_edges.png')
    fig.write_image(path / f'pdf/{title}_edges.pdf')
    
    
    fig = make_graph_hm(pes, ies, display_names, f'Edge p-value (85%)')
    fig.write_image(path / f'png/{title}_edges_p.png')
    fig.write_image(path / f'pdf/{title}_edges_p.pdf')


    node_iou_df = pd.DataFrame.from_dict({gn: ins[:, i] for i, gn in enumerate(graphs_names)})
    node_iou_df['node_count'] = node_counts
    node_iou_p_df = pd.DataFrame.from_dict({gn: pns[:, i] for i, gn in enumerate(graphs_names)})

    node_iou_df.to_csv(path / 'csv/node_ious.csv', index=False)
    node_iou_p_df.to_csv(path / 'csv/node_iou_ps.csv', index=False)

    ius = np.triu_indices(len(graphs))
    ins[ius] = np.nan
    pns[ius] = np.nan

    fig = make_graph_hm(ins, pns, display_names, f'Node Intersection over Union (85%)', node_counts)
    fig.write_image(path / f'png/{title}_nodes.png')
    fig.write_image(path / f'pdf/{title}_nodes.pdf')

    fig = make_graph_hm(pns, ins, display_names, f'Node p-value (85%)')
    fig.write_image(path / f'png/{title}_nodes_p.png')
    fig.write_image(path / f'pdf/{title}_nodes_p.pdf')

    edge_overlap_df = pd.DataFrame.from_dict({gn: eos[:, i] for i, gn in enumerate(graphs_names)})
    edge_overlap_df['edge_count'] = edge_counts
    node_overlap_df = pd.DataFrame.from_dict({gn: nos[:, i] for i, gn in enumerate(graphs_names)})
    node_overlap_df['node_count'] = node_counts

    edge_overlap_df.to_csv(path / 'csv/edge_overlap.csv', index=False)
    node_overlap_df.to_csv(path / 'csv/node_overlap.csv', index=False)

    edge_overlap_min_df = pd.DataFrame.from_dict({gn: ems[:, i] for i, gn in enumerate(graphs_names)})
    edge_overlap_min_df['edge_count'] = edge_counts
    node_overlap_min_df = pd.DataFrame.from_dict({gn: nms[:, i] for i, gn in enumerate(graphs_names)})
    node_overlap_min_df['node_count'] = node_counts

    edge_overlap_min_df.to_csv(path / 'csv/edge_overlap_min.csv', index=False)
    node_overlap_min_df.to_csv(path / 'csv/node_overlap_min.csv', index=False)

    fig = make_graph_hm(eos, pes, display_names, f'Edge Overlap', edge_counts, colorbar_title='Recall')
    fig.write_image(path / f'png/{title}_edge_overlap.png')
    fig.write_image(path / f'pdf/{title}_edge_overlap.pdf')

    fig = make_graph_hm(nos, ins, display_names, f'Node Overlap', node_counts, colorbar_title='Recall')
    fig.write_image(path / f'png/{title}_node_overlap.png')
    fig.write_image(path / f'pdf/{title}_node_overlap.pdf')

    for name, ged_array in zip(['ged', 'z_scored_ged', 'minmax_ged'], [geds, zged, mged]):
        ged_df = pd.DataFrame.from_dict({gn: ged_array[:, i] for i, gn in enumerate(graphs_names)})
        ged_df.to_csv(path / f'csv/{name}.csv', index=False)
        fig = make_graph_hm(ged_array, np.zeros_like(ged_array), display_names, f'{name}', edge_counts, colorbar_title='GED')
        fig.write_image(path / f'png/{title}_{name}.png')
        fig.write_image(path / f'pdf/{title}_{name}.pdf')