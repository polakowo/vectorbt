from timeit import default_timer as timer

import pandas as pd

from vectorbt.optimizer.gridsearch import kpimap


##########
### L4 ###
##########

def from_kpimap(kpimap_sr, min_score, max_score, reverse=False):
    """
    Convert each number in KPI series into [min_score, max_score]

    :param sr: series of numbers
    :param min_score: minimum of target scale
    :param max_score: maximum
    :param reverse: reverse scale
    :return: score series indexed by parameters
    """
    scoremap_sr = kpimap_sr.copy()
    old_range = scoremap_sr.max() - scoremap_sr.min()
    new_range = max_score - min_score
    if old_range == 0:
        scoremap_sr *= 0
        scoremap_sr += min_score
    else:
        scoremap_sr = (scoremap_sr - scoremap_sr.min()) * new_range / old_range + min_score
    if reverse:
        scoremap_sr = min_score + max_score - scoremap_sr
    scoremap_sr.fillna(min_score, inplace=True)
    return scoremap_sr


def from_kpimaps(kpimaps, weights, reversed):
    """
    Combine multiple weighted KPI series into a single score series

    :param retmap: returns keyed by params
    :param kpi_funcs: list of KPIs
    :param weights: list of weights
    :param reversed: list of booleans
    :return: score series indexed by parameters
    """
    t = timer()
    min_score, max_score = 1, 100
    if sum(weights) != 1:
        print("Sum of weights must be 1.")
        return
    scoremap_sr = pd.Series(0, index=kpimaps[0].index)
    for i, kpimap_sr in enumerate(kpimaps):
        scoremap_sr += from_kpimap(kpimap_sr, min_score, max_score, reverse=reversed[i]) * weights[i]
    print("%d-%d-scoremap" % (min_score, max_score))
    kpimap.print_bounds(scoremap_sr)
    print("passed. %.2fs" % (timer() - t))
    return scoremap_sr
