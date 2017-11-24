import pandas as pd


# Equity on position
####################

def base_on_positions(rate_sr, posret_sr):
    """Base equity on positions if bought 1.0 quote"""
    return (posret_sr + 1).cumprod() * rate_sr.loc[posret_sr.index[0]]


def quote_on_positions(rate_sr, posret_sr):
    """Quote equity on positions"""
    return base_on_positions(rate_sr, posret_sr) / rate_sr.loc[posret_sr.index]


# Equity diffs
##############

def base_diffs(rate_sr, posret_sr):
    """Absolute base returns"""
    base_sr = base_on_positions(rate_sr, posret_sr)
    return (base_sr - base_sr.shift().fillna(1)).iloc[1::2]


def quote_diffs(rate_sr, posret_sr):
    """Absolute quote returns"""
    quote_sr = quote_on_positions(rate_sr, posret_sr)
    return (quote_sr - quote_sr.shift().fillna(1)).iloc[2::2]


# Equity
########

def base(rate_sr, posret_sr):
    """Base equity for each index in rate_sr"""
    quote_sr = quote_on_positions(rate_sr, posret_sr)
    pos_sr = pd.Series([1, -1] * (len(posret_sr.index) // 2), index=posret_sr.index)
    hold_mask = pos_sr.reindex(rate_sr.index).ffill() == 1
    hold_rates = rate_sr.loc[hold_mask]
    cash_rates = rate_sr.loc[~hold_mask]
    hold_sr = quote_sr.iloc[0::2].reindex(hold_rates.index).ffill() * hold_rates
    cash_sr = (quote_sr.iloc[1::2].reindex(cash_rates.index) * cash_rates).ffill()
    return hold_sr.append(cash_sr).sort_index()


def quote(rate_sr, posret_sr):
    """Quote equity for each index in rate_sr"""
    return base(rate_sr, posret_sr) / rate_sr


def plot_base(rate_sr, base_sr):
    from vectorbt import graphics
    print("base")
    graphics.plot_line(base_sr, benchmark=rate_sr)


def plot_quote(rate_sr, quote_sr):
    from vectorbt import graphics
    print("quote")
    graphics.plot_line(quote_sr, benchmark=rate_sr * 0 + 1)
