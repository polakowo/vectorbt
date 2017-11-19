import pandas as pd
from matplotlib import pyplot as plt
from vectorbt import graphics


def from_vectors(rate_sr, entry_vector, exit_vector):
    """
    Generate positions from short/long signals

    :param entry_vector: bit vector of long signals
    :param exit_vector: bit vector of short signals
    :return: bit vector of positions
    """
    # Merge vectors
    signals = entry_vector - exit_vector - entry_vector * exit_vector
    # Always sell at the end
    signals[-1] = -1
    signal_sr = pd.Series(signals, index=rate_sr.index)
    signal_sr = signal_sr.iloc[signal_sr.nonzero()]
    # Generate positions
    pos_sr = signal_sr[signal_sr != signal_sr.shift()]
    # Always buy at the beginning
    if len(pos_sr.index)%2 != 0:
        pos_sr = pos_sr.iloc[1:]
    # Positions are always even, starting with long, ending with short
    return pos_sr


def on_hold(rate_sr):
    # Positions on hold
    return pd.Series([1, -1], index=rate_sr.index[[0, -1]])


def plot(rate_sr, pos_sr):
    diffs = rate_sr.loc[pos_sr.index].diff().fillna(0)
    print(pd.DataFrame(diffs.describe()).transpose())
    fig, ax = plt.subplots()

    ax.plot(rate_sr, c='darkgrey')

    # Draw position markers
    purchase_dates = pos_sr.index[0::2]
    sale_dates = pos_sr.index[1::2]
    ax.plot(purchase_dates, rate_sr.loc[purchase_dates].values, '^', color='lime',
            markeredgecolor='darkgreen', markersize=8, markeredgewidth=1)
    ax.plot(sale_dates, rate_sr.loc[sale_dates].values, 'v', color='orangered',
            markeredgecolor='darkred', markersize=8, markeredgewidth=1)
    plt.show()

