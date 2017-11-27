from timeit import default_timer as timer

import pandas as pd


def from_nummap(nummap_sr, min_score, max_score, reversed=False):
    """Convert each number in KPI series into [min_score, max_score]"""
    scoremap_sr = nummap_sr.copy()
    old_range = scoremap_sr.max() - scoremap_sr.min()
    new_range = max_score - min_score
    if old_range == 0:
        scoremap_sr *= 0
        scoremap_sr += min_score
    else:
        scoremap_sr = (scoremap_sr - scoremap_sr.min()) * new_range / old_range + min_score
    if reversed:
        scoremap_sr = min_score + max_score - scoremap_sr
    scoremap_sr.fillna(min_score, inplace=True)
    return scoremap_sr


def from_nummaps(nummaps, weights, reversed):
    """Combine multiple weighted KPI series into a single score series"""
    from vectorbt.optimizer.gridsearch import nummap

    t = timer()
    min_score, max_score = 1, 100
    if sum(weights) != 1:
        print("Sum of weights must be 1.")
        return
    scoremap_sr = pd.Series(0, index=nummaps[0].index)
    for i, nummap_sr in enumerate(nummaps):
        scoremap_sr += from_nummap(nummap_sr, min_score, max_score, reversed=reversed[i]) * weights[i]
    print("done. %.2fs" % (timer() - t))
    nummap.bounds(scoremap_sr)
    return scoremap_sr
