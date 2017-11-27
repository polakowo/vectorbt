import pandas as pd
import numpy as np

def from_positions(rate_sr, pos_sr, fees, slippage):
    """Generate base equity from positions"""
    from vectorbt import array

    rates = rate_sr.values
    positions = pos_sr.reindex(rate_sr.index).fillna(0).values
    pos_idx = np.flatnonzero(positions)
    if len(pos_idx) == 0:
        return pd.Series(index=rate_sr.index)
    returns_mask = array.ffill(positions)
    # Returns are always shifted by 1
    returns_mask = array.fshift(returns_mask, 1, 0)
    returns_mask = returns_mask == 1
    equity = np.ones(len(positions))
    equity[returns_mask] += array.pct_change(rates)[returns_mask]
    # Fees in %
    equity[pos_idx] *= 1 - fees
    # Slippage in %
    if isinstance(slippage, pd.Series):
        equity[pos_idx] *= 1 - slippage.iloc[pos_idx].values
    else:
        equity[pos_idx] *= 1 - slippage
    equity = np.cumprod(equity) * rates[0]
    # NaN before first position
    equity[:pos_idx[0]] = np.nan
    # Attach index
    return pd.Series(equity, index=rate_sr.index)


def to_quote(rate_sr, equity_sr):
    """Generate quote equity"""
    return equity_sr / rate_sr


def plot(rate_sr, equity_sr):
    from vectorbt import graphics

    print("base")
    graphics.plot_line(equity_sr, benchmark=rate_sr)
    print("quote")
    graphics.plot_line(to_quote(rate_sr, equity_sr), benchmark=rate_sr * 0 + 1)
