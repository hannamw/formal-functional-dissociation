from typing import Tuple 

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns

from eap.utils import display_name_dict

def is_formal(x):
    return any(formal_task in x for formal_task in [display_name_dict[y] for y in ['gendered-pronoun', 'sva', 'hypernymy-comma', 'npi', 'wug']])

def color_labels(ax:Axes, x=True, y=True):
    if x:
        for label in ax.get_xticklabels():
            if is_formal(label.get_text()):
                label.set_color('blue')
    if y:
        for label in ax.get_yticklabels():
            if is_formal(label.get_text()):
                label.set_color('blue')


def make_whole_fig(data, graphs_names, counts, title, colorbar_title, xlabel='Task', ylabel='Task', size=None, cbar=True, tasks_y=True) -> Tuple[Figure, Axes]:
    short_graphs_names = graphs_names[-len(counts):]
    if counts is not None:
        y_label = [f'{name} ({count})' for name, count in zip(short_graphs_names, counts)]
    else:
        y_label = short_graphs_names

    if size:
        fig, ax = plt.subplots(figsize=size)
    else:
        fig, ax = plt.subplots()
    fig:Figure = fig
    ax:Axes = ax
    if cbar:
        sns.heatmap(data, cmap='Blues', cbar_kws={'label':colorbar_title, 'shrink':1.0}, ax=ax)
    else:
        sns.heatmap(data, cmap='Blues', cbar=False, ax=ax)
    ax.set_xticklabels(graphs_names, rotation=45, ha='right')
    ax.set_yticklabels(y_label, rotation=0)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel)
    if tasks_y:
        ax.set_ylabel(ylabel)

    color_labels(ax)

    if len(y_label) > 10:
        ax.axhline(5, color='black')
    ax.axvline(5, color='black') 

    fig.tight_layout()
    return fig, ax