from timeit import default_timer as timer

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


##########
### L3 ###
##########

def from_eqdmap(eqdmap, kpi_func):
    """
    Apply KPI on equity diffs map

    :param kpi_func: kpi (e.g., from vectorbt.indicators)
    :return: kpi series indexed by parameters
    """
    print("%s-kpimap" % kpi_func.__name__)
    longest_sr = sorted(list(eqdmap.items()), key=lambda x: -len(x[1]))[0][1]
    t1 = timer()
    kpi_func(longest_sr)
    t2 = timer()
    print("calcs: %d (~%.2fs)" % (len(eqdmap), len(eqdmap) * (t2 - t1)))
    kpimap_sr = pd.Series({params: kpi_func(returns_sr) if len(returns_sr.index) > 0 else np.nan
                           for params, returns_sr in eqdmap.items()})
    print_bounds(kpimap_sr)
    print("passed. %.2fs" % (timer() - t1))
    return kpimap_sr


def bounds(kpimap_sr):
    # Bounds of series (min and max)
    return kpimap_sr.dropna().sort_values().iloc[[0, -1]]


def print_bounds(kpimap_sr):
    kpimap_bounds = bounds(kpimap_sr)
    print("min %s: %s" % (str(kpimap_bounds.index[0]), str(kpimap_bounds.iloc[0])))
    print("max %s: %s" % (str(kpimap_bounds.index[-1]), str(kpimap_bounds.iloc[-1])))


def compare(kpimap_a_sr, kpimap_b_sr):
    # Compare distributions of KPI maps
    info_df = pd.DataFrame()  # contains general info for printing
    perc_index = range(0, 101, 5)
    perc_df = pd.DataFrame(index=perc_index)  # contains percentiles for drawing

    for i, kpimap_sr in enumerate([kpimap_a_sr, kpimap_b_sr]):
        info_df[i] = kpimap_sr.describe()
        perc_df[i] = [np.nanpercentile(kpimap_sr, x) for x in perc_index]

    print(info_df.transpose())

    fig, ax = plt.subplots()

    ax.plot(perc_df[0], color='lightgrey')
    ax.plot(perc_df[1], color='darkgrey')
    ax.fill_between(perc_index,
                    perc_df[1],
                    perc_df[0],
                    where=perc_df[1] > perc_df[0],
                    facecolor='limegreen',
                    interpolate=True)
    ax.fill_between(perc_index,
                    perc_df[1],
                    perc_df[0],
                    where=perc_df[1] < perc_df[0],
                    facecolor='gold',
                    interpolate=True)
    diff_df = perc_df[1] - perc_df[0]
    ax.plot(diff_df.idxmax(), perc_df.loc[diff_df.idxmax(), 1], marker='x', markersize=10, color='black')
    ax.plot(diff_df.idxmin(), perc_df.loc[diff_df.idxmin(), 1], marker='x', markersize=10, color='black')
    plt.show()
