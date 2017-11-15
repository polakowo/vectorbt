import pandas as pd
from matplotlib import pyplot as plt


def from_vectors(rate_sr, entry_vector, exit_vector):
    # Generate positions from short/long signals
    # Merge vectors
    signals = entry_vector - exit_vector - entry_vector * exit_vector
    # Always sell at the end
    signals[-1] = -1
    signal_sr = pd.Series(signals, index=rate_sr.index)
    signal_sr = signal_sr.iloc[signal_sr.nonzero()]
    # Generate positions
    pos_sr = signal_sr.diff()
    pos_sr = signal_sr.iloc[pos_sr.nonzero()]
    # Always buy at the beginning
    if len(pos_sr.index)%2 != 0:
        pos_sr = pos_sr.iloc[1:]
    return pos_sr


def on_hold(rate_sr):
    return pd.Series([1, -1], index=rate_sr.index[[0, -1]])


def plot(rate_sr, pos_sr):
    fig, ax = plt.subplots()

    ax.plot(rate_sr, c='darkgrey')

    # Draw position markers
    purchase_dates = pos_sr.index[0::2]
    ax.scatter(purchase_dates,
               rate_sr.loc[purchase_dates].values,
               marker='^',
               c='darkgreen',
               s=50,
               zorder=100)
    sale_dates = pos_sr.index[1::2]
    ax.scatter(sale_dates,
               rate_sr.loc[sale_dates].values,
               marker='v',
               c='darkred',
               s=50,
               zorder=100)
    plt.show()
