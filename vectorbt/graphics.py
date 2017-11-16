import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt


# Colormaps
###########

def discrete_cmap(bounds, colors):
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def continuous_cmap(colors, midpoint=None, N=6):
    # Create continuous cmap with/without midpoint
    cmap = mcolors.LinearSegmentedColormap.from_list('cont_cmap', colors, N=N)
    if midpoint:
        norm = MidpointNormalize(midpoint=midpoint)
    else:
        norm = mcolors.Normalize()
    return cmap, norm


# Linechart
###########

def plot_line(sr, benchmark=None):
    fig, ax = plt.subplots()

    ax.plot(sr, color='darkgrey')
    if benchmark is not None:
        if isinstance(benchmark, float) or isinstance(benchmark, int):
            pass
        else:
            ax.plot(benchmark, color='lightgrey')
        ax.fill_between(sr.index, sr, benchmark, where=sr > benchmark, facecolor='limegreen', interpolate=True)
        ax.fill_between(sr.index, sr, benchmark, where=sr < benchmark, facecolor='gold', interpolate=True)
    ax.plot(sr.idxmax(), sr.max(), marker='x', markersize=10, color='black')
    ax.plot(sr.idxmin(), sr.min(), marker='x', markersize=10, color='black')
    plt.show()


# Histogram
###########

def plot_hist(sr, cmap=None, norm=mcolors.Normalize(), bins=50):
    fig, ax = plt.subplots()

    hist, bins = np.histogram(sr, bins=bins)
    width = np.diff(bins)
    center = (bins[:-1] + bins[1:]) / 2
    if cmap:
        colors = cmap(norm(bins))
    else:
        colors = 'darkgrey'
    ax.bar(center, hist, color=colors, align='center', width=width)
    plt.show()
