import numpy as np
import pandas as pd

from vectorbt import graphics


def on_positions(pos_ret_sr):
    """Equity on positions"""
    return (pos_ret_sr + 1).cumprod()


def diff_on_positions(pos_ret_sr):
    """Equity diffs on positions (absolute returns)"""
    return on_positions(pos_ret_sr) - on_positions(pos_ret_sr).shift().fillna(1)


def from_returns(rate_sr, pos_ret_sr):
    """
    Generate equity in base and quote currency from position returns

    :param pos_ret_sr: position returns (both short/long positions)
    :return: dataframe
    """
    quote_sr = np.cumprod(pos_ret_sr + 1)
    quote_sr *= rate_sr.loc[pos_ret_sr.index[0]]
    quote_sr /= rate_sr.loc[quote_sr.index]
    # Hold and cash periods
    pos_sr = pd.Series([1, -1] * (len(pos_ret_sr.index) // 2), index=pos_ret_sr.index)
    hold_mask = pos_sr.reindex(rate_sr.index).ffill() == 1
    hold_rates = rate_sr.loc[hold_mask]
    cash_rates = rate_sr.loc[~hold_mask]
    hold_sr = quote_sr.iloc[0::2].reindex(hold_rates.index).ffill() * hold_rates
    cash_sr = (quote_sr.iloc[1::2].reindex(cash_rates.index) * cash_rates).ffill()
    # Fill dataframe
    equity_df = hold_sr.append(cash_sr).sort_index().to_frame('base')
    equity_df['quote'] = equity_df['base'] / rate_sr
    return equity_df


def plot(rate_sr, equity_df):
    print("base")
    graphics.plot_line(equity_df['base'], benchmark=rate_sr)
    print("quote")
    graphics.plot_line(equity_df['quote'], benchmark=rate_sr * 0 + 1)
