import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def from_eqdmap(eqdmap, kpi_func):
    # Apply KPI on each vector of returns in the map
    kpimap = {params: kpi_func(returns_sr) if len(returns_sr.index) > 0 else np.nan
            for params, returns_sr in eqdmap.items()}
    print(shape(kpimap))
    return kpimap


def reduce(kpimap):
    # Flatten all KPIs and describe them
    return pd.Series(list(kpimap.values())).describe()


def shape(kpimap):
    # Min and max of a map
    kpimap_sr = pd.Series(kpimap).dropna().sort_values()
    sorted_map = list(zip(kpimap_sr.index, kpimap_sr))
    return {
        'min': sorted_map[0],
        'max': sorted_map[-1]
    }


def compare(kpimap_a, kpimap_b):
    # Compare maps on one single KPI
    info_df = pd.DataFrame() # contains general info for printing
    perc_index = range(0, 101, 5)
    perc_df = pd.DataFrame(index=perc_index) # contains percentiles for drawing

    for i, kpimap in enumerate([kpimap_a, kpimap_b]):
        info_df[i] = reduce(kpimap)
        perc_df[i] = [np.nanpercentile(list(kpimap.values()), x) for x in perc_index]

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
