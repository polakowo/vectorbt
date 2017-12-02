import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def from_signals(rate_sr, entries, exits):
    """Generate positions from entry and exit bit-vectors"""
    # Merge
    merged = entries - exits - entries * exits
    nz_idx = np.flatnonzero(merged)
    if len(nz_idx) == 0:
        return pd.Series(0, index=rate_sr.index)
    mask_first = merged[nz_idx] != np.insert(merged[nz_idx[:-1]], 0, 0)
    nz_idx = nz_idx[mask_first]
    # Always buy first
    if merged[nz_idx[0]] == -1:
        nz_idx = nz_idx[1:]
    return pd.Series(merged[nz_idx], index=rate_sr.index[nz_idx])


def random(rate_sr, n):
    """Generate random positions"""
    import random

    if n == 0:
        return pd.Series()
    idx = sorted(random.sample(range(len(rate_sr.index)), n))
    positions = pd.Series(index=rate_sr.index[idx])
    positions.iloc[0::2] = 1
    if n > 1:
        positions.iloc[1::2] = -1
    return positions


def plot(rate_sr, pos_sr):
    bought = rate_sr.loc[pos_sr.index[0::2]]
    sold = rate_sr.loc[pos_sr.index[1::2]]
    if pos_sr.iloc[-1] == 1:
        sold.loc[rate_sr.index[-1]] = rate_sr.iloc[-1]
    stats = pd.Series((sold.values - bought.values)).describe()
    print(pd.DataFrame(stats).transpose())
    fig, ax = plt.subplots()

    ax.plot(rate_sr, c='darkgrey')

    # Draw position markers
    ax.plot(bought.index, bought, '^', color='lime',
            markeredgecolor='darkgreen', markersize=8, markeredgewidth=1)
    ax.plot(sold.index, sold, 'v', color='orangered',
            markeredgecolor='darkred', markersize=8, markeredgewidth=1)
    plt.show()

