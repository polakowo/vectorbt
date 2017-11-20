import random

import numpy as np
import pandas as pd


def reduce_vector(vector):
    # Pick first entry from each sequence of entries
    return np.insert((np.diff(vector) == 1).astype(int), 0, vector[0])


# Dual moving average crossover
###############################


def dmac_entry_vector(rate_sr, fast_ma_sr, slow_ma_sr, th=(0, 0)):
    """We require provision of MA beforehand, so we don't need to recalculate it every time"""
    return np.where(fast_ma_sr - slow_ma_sr > th[0] * rate_sr, 1, 0)


def dmac_exit_vector(rate_sr, fast_ma_sr, slow_ma_sr, th=(0, 0)):
    return np.where(fast_ma_sr - slow_ma_sr < -th[0] * rate_sr, 1, 0)


# Moving average envelope
#########################

def mae_entry_vector(rate_sr, ma_sr, envelope):
    return np.where(rate_sr - (1 + envelope) * ma_sr > 0, 1, 0)


def mae_exit_vector(rate_sr, ma_sr, envelope):
    return np.where(rate_sr - (1 - envelope) * ma_sr < 0, 1, 0)


# MACD crossover
################

def macd_co_entry_vector(macd_sr, signal_sr):
    return np.where(macd_sr - signal_sr > 0, 1, 0)


def macd_co_exit_vector(macd_sr, signal_sr):
    return np.where(macd_sr - signal_sr < 0, 1, 0)


# MACD histogram drops
######################

def macd_hist_entry_vector(hist_sr, ndrops):
    vector = (hist_sr[hist_sr < 0].diff() > 0).astype(int).reindex(hist_sr.index).fillna(0)
    grouped = reduce_vector(vector).cumsum()
    cum_drops = (vector.groupby(grouped).cumsum() >= ndrops).astype(int)
    return vector * cum_drops


def macd_hist_exit_vector(hist_sr, ndrops):
    """Entry market once there is N dropping bars in a row"""
    vector = (hist_sr[hist_sr > 0].diff() < 0).astype(int).reindex(hist_sr.index).fillna(0)
    grouped = reduce_vector(vector).cumsum()
    cum_drops = (vector.groupby(grouped).cumsum() >= ndrops).astype(int)
    return vector * cum_drops


# Random
########

def random_entry_vector(rate_sr, n):
    indexes = random.sample(range(len(rate_sr.index)), n)
    vector = np.zeros(len(rate_sr.index))
    vector[indexes] = 1
    return vector


def random_exit_vector(rate_sr, entry_vector, n):
    # Needs clear entry points
    entry_vector = reduce_vector(entry_vector)
    entries = np.flatnonzero(entry_vector)
    non_entries = np.flatnonzero(entry_vector == 0)
    indexes = np.random.choice(non_entries[non_entries > entries[0]], n, replace=True)
    vector = np.zeros(len(rate_sr.index))
    vector[indexes] = 1
    return vector


# Turtle
########

def turtle_entry_vector(rate_sr, window):
    return (rate_sr == rate_sr.rolling(window=window).max()).astype(int).values


def turtle_exit_vector(rate_sr, window):
    return (rate_sr == rate_sr.rolling(window=window).min()).astype(int).values


# Trailing stop
###############

def apply_trail(roll_sr, trail):
    """Apply trail to rolling series"""
    # Trail is in %
    if isinstance(trail, float) and 0 < abs(trail) < 1:
        stop_sr = roll_sr * (1 + trail)
    # Trail is an absolute number
    elif isinstance(trail, float) or isinstance(trail, int):
        stop_sr = roll_sr + trail
    # Trail is a series of absolute numbers
    elif isinstance(trail, pd.Series):
        changing_sr = roll_sr.iloc[roll_sr.pct_change().fillna(0).nonzero()]
        stop_sr = (changing_sr + trail).reindex(roll_sr.index).ffill()
    else:
        raise Exception("Trail must be either number or pd.Series")
    return stop_sr


def trailstop_entry(rate_sr, trail):
    rollmin_sr = rate_sr.rolling(window=len(rate_sr.index), min_periods=1).min()
    stop_sr = apply_trail(rollmin_sr, trail)
    sellstops = np.flatnonzero(np.where(rate_sr > stop_sr, 1, 0))
    sellstop = sellstops[0] if len(sellstops) > 0 else None
    return sellstop


def trailstop_exit(rate_sr, trail):
    rollmax_sr = rate_sr.rolling(window=len(rate_sr.index), min_periods=1).max()
    stop_sr = apply_trail(rollmax_sr, -trail)
    sellstops = np.flatnonzero(np.where(rate_sr < stop_sr, 1, 0))
    sellstop = sellstops[0] if len(sellstops) > 0 else None
    return sellstop


def traverse_trailstops(rate_sr, entry_trail, exit_trail):
    """In case both vectors are calculated using trailing stop and thus depending"""
    trailstops = [0]
    entry_vector = np.zeros(len(rate_sr.index))
    exit_vector = np.zeros(len(rate_sr.index))
    while True:
        if len(trailstops) % 2 == 0:  # exit or entry?
            i = trailstops[-1] + 1  # exit excluded
            j = trailstop_entry(rate_sr.iloc[i:], entry_trail)
            if j is not None:
                trailstops.append(i + j)  # index adjusted to rate_sr
                entry_vector[i + j] = 1
            else:
                break
        else:
            i = trailstops[-1]  # entry included
            j = trailstop_exit(rate_sr.iloc[i:], exit_trail)
            if j is not None:
                trailstops.append(i + j)
                exit_vector[i + j] = 1
            else:
                break
    return entry_vector, exit_vector


def trailstop_entry_vector(rate_sr, exit_vector, trail):
    """
    Exit vector needed
    Exit resets entry -> vectorized solution possible -> divide and conquer
    """
    # Needs clear exit points
    exit_vector = reduce_vector(exit_vector)
    groups = rate_sr.groupby(np.cumsum(exit_vector))
    rel_entry_pos = groups.apply(lambda x: trailstop_entry(x, trail)).values
    abs_exit_pos = np.insert(np.flatnonzero(exit_vector), 0, 0)
    abs_entry_pos = rel_entry_pos + abs_exit_pos
    abs_entry_pos = abs_entry_pos[~np.isnan(abs_entry_pos)]
    abs_entry_pos = abs_entry_pos.astype(int)
    vector = np.zeros(len(rate_sr.index))
    vector[abs_entry_pos] = 1
    return vector


def trailstop_exit_vector(rate_sr, entry_vector, trail):
    """
    Entry vector needed
    Entry doesn't reset exit -> vectorized solution not possible -> iterate
    """
    # Needs clear entry points
    entry_vector = reduce_vector(entry_vector)
    entries = np.flatnonzero(entry_vector)
    exits = []
    while True:
        if len(exits) > 0:
            # Entries do not reset trailing stops -> next entry after exit
            entries = entries[entries > exits[-1]]
        if len(entries) == 0:
            break
        entry = entries[0]
        exit = trailstop_exit(rate_sr.iloc[entry:], trail)
        if exit is None:
            break
        exits.append(entry + exit)
    vector = np.zeros(len(rate_sr.index))
    vector[exits] = 1
    return vector


# Bollinger Bands
#################

def bb_entry_vector(rate_sr, lower_band_sr):
    # Oversold
    return np.where(rate_sr < lower_band_sr, 1, 0)


def bb_exit_vector(rate_sr, upper_band_sr):
    # Overbought
    return np.where(rate_sr > upper_band_sr, 1, 0)


# RSI
#####

def rsi_entry_vector(rsi_sr, lower_bound):
    # Oversold
    return np.where(rsi_sr < lower_bound, 1, 0)


def rsi_exit_vector(rsi_sr, upper_bound):
    # Overbought
    return np.where(rsi_sr > upper_bound, 1, 0)
