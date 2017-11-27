import numpy as np
import pandas as pd


def above(rate_sr, benchmark_sr):
    """Rate above benchmark"""
    return np.where(rate_sr > benchmark_sr, 1, 0)


def below(rate_sr, benchmark_sr):
    return np.where(rate_sr < benchmark_sr, 1, 0)


def raising(rate_sr, n):
    """nst raise in a row"""
    from vectorbt import bitvector

    raised = (rate_sr.diff() > 0).astype(int).reindex(rate_sr.index).fillna(0)
    return bitvector.from_nst(raised, n)


def dropping(rate_sr, n):
    return raising(-rate_sr, n)


def is_max(rate_sr, window):
    """Rate is equal to window's max"""
    return (rate_sr == rate_sr.rolling(window=window).max()).astype(int).values


def is_min(rate_sr, window):
    return (rate_sr == rate_sr.rolling(window=window).min()).astype(int).values


def depending(rate_sr, on_vector, signal_func, wait=0):
    """For each source signal generate a target signal (e.g., stop loss)"""
    from vectorbt import bitvector

    idx = []
    vector_idx = np.flatnonzero(on_vector)
    # Every two adjacent signals in the vector become a bin
    bins = list(zip(vector_idx, np.append(vector_idx[1:], None)))
    for x, z in bins:
        x += wait
        # Apply signal function on the bin space only
        y = signal_func(rate_sr.iloc[x:z])
        if y is not None:
            idx.append(x + y)
    return bitvector.from_idx(len(rate_sr.index), idx)


def cross_depending(rate_sr, entry_func, exit_func, wait=0):
    """Generate signals one after another iteratively"""
    from vectorbt import bitvector

    idx = [entry_func(rate_sr)]
    while True:
        i = idx[-1] + wait
        if len(idx) % 2 == 0:  # exit or entry?
            j = entry_func(rate_sr.iloc[i:])
        else:
            j = exit_func(rate_sr.iloc[i:])
        if j is not None:
            idx.append(i + j)
        else:
            break
    entries = bitvector.from_idx(len(rate_sr.index), idx[0::2])
    exits = bitvector.from_idx(len(rate_sr.index), idx[1::2])
    return entries, exits


def random(rate_sr, n, excl_vector=None):
    """Random vector"""
    from vectorbt import bitvector
    import random

    if excl_vector is None:
        # Pick signals not in excl_vector
        idx = random.sample(range(len(rate_sr.index)), n)
    else:
        entries = np.flatnonzero(excl_vector)
        non_entries = np.flatnonzero(excl_vector == 0)
        idx = np.random.choice(non_entries[non_entries > entries[0]], n, replace=True)
    randv = bitvector.from_idx(len(rate_sr.index), idx)
    return randv


###############################
# Trend confirmation patterns #
###############################


# Dual MA Crossover
###################

def DMAC_entries(fast_ma_sr, slow_ma_sr):
    """Entry once fast MA over slow MA (or band)"""
    return above(fast_ma_sr, slow_ma_sr)


def DMAC_exits(fast_ma_sr, slow_ma_sr):
    return below(fast_ma_sr, slow_ma_sr)


# MA Convergence/Divergence
###########################

def MACD_entries(macd_sr, signal_sr):
    """Entry once MACD higher than signal line"""
    return above(macd_sr, signal_sr)


def MACD_exits(macd_sr, signal_sr):
    return below(macd_sr, signal_sr)


def MACD_histdrop_entries(hist_sr, ndrops):
    """Entry market once there is N negative but raising bars in a row"""
    return raising(hist_sr[hist_sr < 0], ndrops)


def MACD_histdrop_exits(hist_sr, ndrops):
    return dropping(hist_sr[hist_sr > 0], ndrops)


# Period min/max
################

def max_signals(rate_sr, window):
    """Entry market once its period's max"""
    return is_max(rate_sr, window)


def min_signals(rate_sr, window):
    return is_max(rate_sr, window)


###########################
# Trend reversal patterns #
###########################

# Bollinger Bands
#################

def BB_entries(rate_sr, lower_band_sr):
    """Entry market once oversold"""
    return below(rate_sr, lower_band_sr)


def BB_exits(rate_sr, upper_band_sr):
    return above(rate_sr, upper_band_sr)


# RSI
#####

def RSI_entries(rsi_sr, lower_bound):
    """Entry market once oversold"""
    return below(rsi_sr, lower_bound)


def RSI_exits(rsi_sr, upper_bound):
    return above(rsi_sr, upper_bound)


# Bullish/Bearish Engulfing Pattern
###################################

def BEP_entries(open_sr, high_sr, low_sr, close_sr, full=False, amount=1):
    """Entry once hollow body completely engulfs the previous filled body/candlestick"""
    hollow = close_sr.values[1:] > open_sr.values[1:]
    if full:
        last_candle = high_sr.values[:-1] - low_sr.values[:-1]
    else:
        last_candle = open_sr.values[:-1] - close_sr.values[:-1]
    candle = close_sr.values[1:] - open_sr.values[1:]
    engulfing = candle > amount * last_candle
    entries = np.insert((hollow & engulfing).astype(int), 0, 0)
    # Close is unknown (future) data -> shift vector
    entries = np.insert(entries[1:], 0, 0)
    return entries


def BEP_exits(open_sr, high_sr, low_sr, close_sr, full=False, amount=1):
    return BEP_entries(close_sr, high_sr, low_sr, open_sr, full=full, amount=amount)


###################
# Risk limitation #
###################

# Stop loss
###########

def stoploss_exit(rate_sr, stop):
    """Index of the first rate below the stop"""
    if isinstance(stop, pd.Series):
        stop = stop.loc[rate_sr.index[0]]
    stops = np.flatnonzero(below(rate_sr, stop))
    return stops[0] if len(stops) > 0 else None


def stoploss_exits(rate_sr, entries, stop):
    """Apply stop loss on each entry signal"""
    return depending(rate_sr, entries, lambda sr: stoploss_exit(sr, stop))


# Trailing stop
###############

def trailstop_exit(rate_sr, trail):
    rollmax_sr = rate_sr.rolling(window=len(rate_sr.index), min_periods=1).max()
    # Trail is either absolute number or series of absolute numbers
    # To set trail in %, multiply % with rate_sr beforehand to get absolute numbers
    if isinstance(trail, pd.Series):
        changing_sr = rollmax_sr.iloc[rollmax_sr.pct_change().nonzero()]
        stop_sr = (changing_sr - trail).reindex(rate_sr.index).ffill()
    else:
        stop_sr = rollmax_sr - trail
    sellstops = np.flatnonzero(below(rate_sr, stop_sr))
    sellstop = sellstops[0] if len(sellstops) > 0 else None
    return sellstop


def trailstop_exits(rate_sr, entries, trail):
    return depending(rate_sr, entries, lambda sr: trailstop_exit(sr, trail))
