import pandas as pd
import numpy as np

def from_positions(rate_sr, pos_sr, investment, fees, slippage):
    """
    Generate base equity from positions
    Initial purchase is exactly 1 (e.g., in BTC), fees and slippage covered
    """
    from vectorbt import array

    rates = rate_sr.values
    positions = pos_sr.reindex(rate_sr.index).fillna(0).values
    pos_idx = np.flatnonzero(positions)
    if len(pos_idx) == 0:
        return pd.Series(index=rate_sr.index)
    
    # Calculate returns
    returns_mask = array.ffill(positions)
    # Returns are always shifted by 1
    returns_mask = array.fshift(returns_mask, 1, 0)
    returns_mask = returns_mask == 1
    equity = np.ones(len(positions))
    equity[returns_mask] += array.pct_change(rates)[returns_mask]
    
    # Apply costs
    equity[pos_idx] *= 1 - fees
    if isinstance(slippage, pd.Series):
        equity[pos_idx] *= 1 - slippage.iloc[pos_idx].values
    else:
        equity[pos_idx] *= 1 - slippage
    
    # Apply investment
    if investment is None:
        investment = rates[0]
    equity = np.cumprod(equity) * investment
    
    # NaN before first position
    equity[:pos_idx[0]] = np.nan
    # Attach index
    return pd.Series(equity, index=rate_sr.index)


def to_quote(rate_sr, equity_sr):
    """Generate quote equity"""
    return equity_sr / rate_sr


def plot(rate_sr, equity_sr):
    from vectorbt import graphics

    investment = equity_sr.bfill().iloc[0]
    hold_sr = (rate_sr.pct_change().fillna(0) + 1).cumprod() * investment
    print("base: equity - hold")
    graphics.plot_line(equity_sr, benchmark=hold_sr)
    print("quote: equity - hold")
    graphics.plot_line(to_quote(rate_sr, equity_sr), benchmark=to_quote(rate_sr, hold_sr))
