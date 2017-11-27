import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def from_srmap(srmap, reducer, multiprocess=False):
    """Apply KPI and pack into Series"""
    from vectorbt.optimizer.gridsearch import mapper

    params, series = zip(*srmap.items())
    reduced = mapper.map(reducer, list(series), multiprocess=multiprocess)
    params = list(params)
    nummap_sr = pd.Series(reduced, index=params)
    bounds(nummap_sr)
    return nummap_sr


def bounds(nummap_sr):
    """Min and max"""
    sorted_sr = nummap_sr.dropna().sort_values()
    print("min %s: %s" % (str(sorted_sr.index[0]), str(sorted_sr.iloc[0])))
    print("max %s: %s" % (str(sorted_sr.index[-1]), str(sorted_sr.iloc[-1])))


def compare_quantiles(nummap_sr, benchmark_sr):
    print(pd.DataFrame([nummap_sr.describe(), benchmark_sr.describe()], index=['nummap', 'benchmark']))

    perc_index = range(0, 101, 5)
    nummap_perc_sr = pd.Series({x: np.nanpercentile(nummap_sr, x) for x in perc_index})
    benchmark_perc_sr = pd.Series({x: np.nanpercentile(benchmark_sr, x) for x in perc_index})

    fig, ax = plt.subplots()
    ax.plot(benchmark_perc_sr, color='lightgrey')
    ax.plot(nummap_perc_sr, color='darkgrey')
    ax.fill_between(perc_index,
                    nummap_perc_sr,
                    benchmark_perc_sr,
                    where=nummap_perc_sr > benchmark_perc_sr,
                    facecolor='lime',
                    interpolate=True)
    ax.fill_between(perc_index,
                    nummap_perc_sr,
                    benchmark_perc_sr,
                    where=nummap_perc_sr < benchmark_perc_sr,
                    facecolor='orangered',
                    interpolate=True)
    diff_sr = nummap_perc_sr - benchmark_perc_sr
    ax.plot(diff_sr.idxmax(), nummap_perc_sr.loc[diff_sr.idxmax()], marker='x', markersize=10, color='black')
    ax.plot(diff_sr.idxmin(), nummap_perc_sr.loc[diff_sr.idxmin()], marker='x', markersize=10, color='black')
    plt.show()


def compare_hists(nummap_sr, benchmark_sr, bins, cmap, norm):
    print(pd.DataFrame([nummap_sr.describe(), benchmark_sr.describe()], index=['nummap', 'benchmark']))

    sr_min = np.min([nummap_sr.min(), benchmark_sr.min()])
    sr_max = np.max([nummap_sr.max(), benchmark_sr.max()])
    bins = np.linspace(sr_min, sr_max, bins)
    width = np.diff(bins)
    center = (bins[:-1] + bins[1:]) / 2
    nummap_hist, _ = np.histogram(nummap_sr.dropna().values, bins=bins)
    benchmark_hist, _ = np.histogram(benchmark_sr.dropna().values, bins=bins)

    def plot_hist(hist):
        fig, ax = plt.subplots()
        ax.bar(center, hist, color=cmap(norm(bins)), align='center', width=width)
        ax.set_ylim(0, np.max([nummap_hist.max(), benchmark_hist.max()]))
        plt.show()

    plot_hist(nummap_hist)
    plot_hist(benchmark_hist)
