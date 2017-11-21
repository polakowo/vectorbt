import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from vectorbt.optimizer.gridsearch import mapper


def from_seriesmap(seriesmap, reducer):
    """Apply KPI and pack into Series"""
    reduced = mapper.map(reducer, list(seriesmap.values()))
    params = list(seriesmap.keys())
    nummap_sr = pd.Series(dict(zip(params, reduced)))
    bounds(nummap_sr)
    return nummap_sr


def bounds(nummap_sr):
    """Min and max"""
    sorted_sr = nummap_sr.dropna().sort_values()
    print("min %s: %s" % (str(sorted_sr.index[0]), str(sorted_sr.iloc[0])))
    print("max %s: %s" % (str(sorted_sr.index[-1]), str(sorted_sr.iloc[-1])))


def compare(nummap_a_sr, nummap_b_sr):
    """Compare distributions of KPI maps"""
    info_df = pd.DataFrame()  # contains general info for printing
    perc_index = range(0, 101, 5)
    perc_df = pd.DataFrame(index=perc_index)  # contains percentiles for drawing

    for i, nummap_sr in enumerate([nummap_a_sr, nummap_b_sr]):
        info_df[i] = nummap_sr.describe()
        perc_df[i] = [np.nanpercentile(nummap_sr, x) for x in perc_index]

    print(info_df.transpose())

    fig, ax = plt.subplots()
    ax.plot(perc_df[0], color='lightgrey')
    ax.plot(perc_df[1], color='darkgrey')
    ax.fill_between(perc_index,
                    perc_df[1],
                    perc_df[0],
                    where=perc_df[1] > perc_df[0],
                    facecolor='lime',
                    interpolate=True)
    ax.fill_between(perc_index,
                    perc_df[1],
                    perc_df[0],
                    where=perc_df[1] < perc_df[0],
                    facecolor='orangered',
                    interpolate=True)
    diff_df = perc_df[1] - perc_df[0]
    ax.plot(diff_df.idxmax(), perc_df.loc[diff_df.idxmax(), 1], marker='x', markersize=10, color='black')
    ax.plot(diff_df.idxmin(), perc_df.loc[diff_df.idxmin(), 1], marker='x', markersize=10, color='black')
    plt.show()
