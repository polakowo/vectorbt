import pandas as pd

from vectorbt import graphics


def from_positions(rate_sr, pos_sr, fees):
    """Generate position returns"""
    diffs = rate_sr.loc[pos_sr.index].pct_change().iloc[1::2].values
    pos_ret_sr = pos_sr * 0
    pos_ret_sr.iloc[1::2] = diffs
    pos_ret_sr += 1
    pos_ret_sr *= 1 - fees
    pos_ret_sr = pos_ret_sr.cumprod()
    pos_ret_sr = pos_ret_sr.pct_change().fillna(-fees)
    return pos_ret_sr


def on_hold(rate_sr, fees):
    """Generate hold returns"""
    pos_sr = pd.Series([1, -1], index=rate_sr.index[[0, -1]])
    return from_positions(rate_sr, pos_sr, fees)


def plot(pos_ret_sr):
    graphics.plot_line(pos_ret_sr, benchmark=0)
