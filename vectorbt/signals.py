import random

import numpy as np
import pandas as pd


def above(rate_sr, benchmark_sr):
    """Rate above benchmark"""
    return np.where(rate_sr > benchmark_sr, 1, 0)


def below(rate_sr, benchmark_sr):
    return np.where(rate_sr < benchmark_sr, 1, 0)


def raising(rate_sr, n):
    from vectorbt import vector
    """nst raise in a row"""
    v = (rate_sr.diff() > 0).astype(int).reindex(rate_sr.index).fillna(0)
    return vector.from_nst(v, n)


def dropping(rate_sr, n):
    return raising(-rate_sr, n)


def is_max(rate_sr, window):
    """Rate is equal to window's max"""
    return (rate_sr == rate_sr.rolling(window=window).max()).astype(int).values


def is_min(rate_sr, window):
    return (rate_sr == rate_sr.rolling(window=window).min()).astype(int).values


def depending_vector(rate_sr, on_vector, signal_func, wait=0):
    from vectorbt import vector
    """For each source signal generate a target signal (e.g., stop loss)"""
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
    v = vector.from_idx(len(rate_sr.index), idx)
    return v


def depending_vectors(rate_sr, entry_func, exit_func, wait=0):
    from vectorbt import vector
    """Generate signals one after another iteratively"""
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
    evector = vector.from_idx(len(rate_sr.index), idx[0::2])
    xvector = vector.from_idx(len(rate_sr.index), idx[1::2])
    return evector, xvector


##########
# Random #
##########

def random_vector(rate_sr, n, excl_vector=None):
    from vectorbt import vector
    """Random vector"""
    if excl_vector is None:
        # Pick signals not in excl_vector
        idx = random.sample(range(len(rate_sr.index)), n)
    else:
        entries = np.flatnonzero(excl_vector)
        non_entries = np.flatnonzero(excl_vector == 0)
        idx = np.random.choice(non_entries[non_entries > entries[0]], n, replace=True)
    randv = vector.from_idx(len(rate_sr.index), idx)
    return randv


###############################
# Trend confirmation patterns #
###############################


# Dual MA Crossover
###################

def DMAC_evector(fast_ma_sr, slow_ma_sr):
    """Entry once fast MA over slow MA (or band)"""
    return above(fast_ma_sr, slow_ma_sr)


def DMAC_xvector(fast_ma_sr, slow_ma_sr):
    return below(fast_ma_sr, slow_ma_sr)


# MA Convergence/Divergence
###########################

def MACD_evector(macd_sr, signal_sr):
    """Entry once MACD higher than signal line"""
    return above(macd_sr, signal_sr)


def MACD_xvector(macd_sr, signal_sr):
    return below(macd_sr, signal_sr)


def MACD_histdrop_evector(hist_sr, ndrops):
    """Entry market once there is N negative but raising bars in a row"""
    return raising(hist_sr[hist_sr < 0], ndrops)


def MACD_histdrop_xvector(hist_sr, ndrops):
    return dropping(hist_sr[hist_sr > 0], ndrops)


# Period min/max
################

def max_vector(rate_sr, window):
    """Entry market once its period's max"""
    return is_max(rate_sr, window)


def min_vector(rate_sr, window):
    return is_max(rate_sr, window)


###########################
# Trend reversal patterns #
###########################

# Bollinger Bands
#################

def BB_evector(rate_sr, lower_band_sr):
    """Entry market once oversold"""
    return below(rate_sr, lower_band_sr)


def BB_xvector(rate_sr, upper_band_sr):
    return above(rate_sr, upper_band_sr)


# RSI
#####

def RSI_evector(rsi_sr, lower_bound):
    """Entry market once oversold"""
    return below(rsi_sr, lower_bound)


def RSI_xvector(rsi_sr, upper_bound):
    return above(rsi_sr, upper_bound)


# Bullish/Bearish Engulfing Pattern
###################################

def BEP_evector(open_sr, high_sr, low_sr, close_sr, full=False, amount=1):
    """Entry once hollow body completely engulfs the previous filled body/candlestick"""
    hollow = close_sr.values[1:] > open_sr.values[1:]
    if full:
        last_candle = high_sr.values[:-1] - low_sr.values[:-1]
    else:
        last_candle = open_sr.values[:-1] - close_sr.values[:-1]
    candle = close_sr.values[1:] - open_sr.values[1:]
    engulfing = candle > amount * last_candle
    evector = np.insert((hollow & engulfing).astype(int), 0, 0)
    # Close is unknown (future) data -> shift vector
    evector = np.insert(evector[1:], 0, 0)
    return evector


def BEP_xvector(open_sr, high_sr, low_sr, close_sr, full=False, amount=1):
    return BEP_evector(close_sr, high_sr, low_sr, open_sr, full=full, amount=amount)


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


def stoploss_xvector(rate_sr, evector, stop):
    """Apply stop loss on each entry signal"""
    return depending_vector(rate_sr, evector, lambda sr: stoploss_exit(sr, stop))


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


def trailstop_xvector(rate_sr, evector, trail):
    return depending_vector(rate_sr, evector, lambda sr: trailstop_exit(sr, trail))
