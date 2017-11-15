import pandas as pd

from vectorbt.optimizer.linmap import kpimap


def score(sr, min_score, max_score, reverse=False):
    """
    For each number, build a score from [min_score, max_score]
    :param sr: series of numbers
    :param min_score: minimum of target scale
    :param max_score: maximum
    :param reverse: reverse scale
    :return: series of scores
    """
    old_range = sr.max() - sr.min()
    new_range = max_score - min_score
    if old_range == 0:
        sr *= 0
        sr += min_score
    else:
        sr = (sr - sr.min()) * new_range / old_range + min_score
    if reverse:
        sr = min_score + max_score - sr
    sr.fillna(min_score, inplace=True)
    return sr


def shape(scoremap):
    # Min and max of a map
    scoremap_sr = pd.Series(scoremap).dropna().sort_values()
    sorted_map = list(zip(scoremap_sr.index, scoremap_sr))
    return {
        'min': sorted_map[0],
        'max': sorted_map[-1]
    }


def from_eqdmap(eqdmap, kpi_funcs, weights, reversed):
    """
    For each vector, build a score from the weighted KPIs
    :param retmap: returns keyed by params
    :param kpi_funcs: list of KPIs
    :param weights: list of weights
    :param reversed: list of booleans
    :return: scores keyed by params
    """
    if sum(weights) != 1:
        print("Sum of weights must be 1.")
        return
    scoremap_sr = pd.Series(0, index=eqdmap.keys())
    for i, kpi_func in enumerate(kpi_funcs):
        kpi_sr = pd.Series(kpimap.from_eqdmap(eqdmap, kpi_func))
        scoremap_sr += score(kpi_sr, 1, 100, reverse=reversed[i]) * weights[i]
    scoremap = scoremap_sr.to_dict()
    print(shape(scoremap))
    return scoremap
