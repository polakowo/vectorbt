import pandas as pd
from vectorbt import graphics


def from_positions(rate_sr, pos_sr, fees):
    # Generate returns from positions
    diffs = rate_sr.loc[pos_sr.index].pct_change().iloc[1::2].values
    pos_ret_sr = pos_sr * 0
    pos_ret_sr.iloc[1::2] = diffs
    pos_ret_sr += 1
    pos_ret_sr *= 1 - fees
    pos_ret_sr = pos_ret_sr.cumprod()
    pos_ret_sr = pos_ret_sr.pct_change().fillna(-fees)
    return pos_ret_sr


def on_hold(rate_sr, fees):
    # Generate returns from holding
    pos_ret_sr = pd.Series()
    pos_ret_sr.loc[rate_sr.index[0]] = -fees
    pos_ret_sr.loc[rate_sr.index[-1]] = (rate_sr.iloc[-1] - rate_sr.iloc[0]) / rate_sr.iloc[0]
    return pos_ret_sr

def plot(pos_ret_sr):
    graphics.plot_line(pos_ret_sr, benchmark=0)
