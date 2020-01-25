import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def plot_line(ax, a, a_label, b=None, b_label=None):
    """Plot line a against line b."""
    ax.plot(a.index, a, color='#1f77b4', label=a_label)
    if b is not None:
        if isinstance(b, float) or isinstance(b, int):
            ax.plot(a.index, [b] * len(a.index), color='#1f77b4', linestyle='--', label=b_label)
        else:
            ax.plot(a.index, b, color='#1f77b4', linestyle='--', label=b_label)
        ax.fill_between(a.index, a, b, where=a > b, facecolor='#add8e6', interpolate=True)
        ax.fill_between(a.index, a, b, where=a < b, facecolor='#ffcccb', interpolate=True)

def plot_markers(ax, a, pos_mask, neg_mask, pos_label='Buy', neg_label='Sell'):
    """Plot positive and negative markers on top of line a."""
    pos_idx = np.argwhere(pos_mask).transpose()[0]
    neg_idx = np.argwhere(neg_mask).transpose()[0]
    ax.plot(a.index[pos_idx], a[pos_idx], '^', color='lime', markersize=10, label=pos_label)
    ax.plot(a.index[neg_idx], a[neg_idx], 'v', color='orangered', markersize=10, label=neg_label)