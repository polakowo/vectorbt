import random

import numpy as np
import pandas as pd


def reduce_vector(vector):
    # Pick first entry from each sequence of entries
    return np.insert((np.diff(vector) == 1).astype(int), 0, vector[0])


# Dual moving average
#####################

def sma(sr, window):
    # SMA
    return sr.rolling(window=window, min_periods=1).mean()


def ema(sr, span):
    # EMA
    return sr.ewm(span=span, adjust=False, min_periods=1).mean()


def ma_entry_vector(rate_sr, fast_ma_sr, slow_ma_sr, th=(0, 0)):
    # Pass: the first time the fast MA is above the slow MA by threshold
    return reduce_vector(np.where(fast_ma_sr - slow_ma_sr > th[0] * rate_sr, 1, 0))


def ma_exit_vector(rate_sr, fast_ma_sr, slow_ma_sr, th=(0, 0)):
    # Pass: the first time the fast MA is below the slow MA by threshold
    return reduce_vector(np.where(fast_ma_sr - slow_ma_sr < -th[0] * rate_sr, 1, 0))


# Random
########

def random_entry_vector(rate_sr, n):
    # Pass: any random date
    indexes = random.sample(range(len(rate_sr.index)), n)
    vector = np.zeros(len(rate_sr.index))
    vector[indexes] = 1
    return reduce_vector(vector)


def random_exit_vector(rate_sr, entry_vector, n):
    # Pass: any random date between entries
    entries = np.flatnonzero(entry_vector)
    non_entries = np.flatnonzero(entry_vector == 0)
    indexes = np.random.choice(non_entries[non_entries > entries[0]], n, replace=True)
    vector = np.zeros(len(rate_sr.index))
    vector[indexes] = 1
    return reduce_vector(vector)


# Turtle
########

def turtle_entry_vector(rate_sr, window):
    # Pass: first rate being max of the window
    return reduce_vector((rate_sr == rate_sr.rolling(window=window).max()).astype(int).values)


def turtle_exit_vector(rate_sr, window):
    # Pass: first rate being min of the window
    return reduce_vector((rate_sr == rate_sr.rolling(window=window).min()).astype(int).values)


# Trailing stop
###############

def true_range(ohlc_df):
    df = pd.DataFrame()
    df[0] = ohlc_df['high'] - ohlc_df['low']
    df[1] = ohlc_df['high'] - ohlc_df['close'].shift()
    df[2] = ohlc_df['low'] - ohlc_df['close'].shift()
    df = df.abs()
    return df.max(axis=1)


def avg_true_range(ohlc_df, ma_func, window, multiplier):
    return ma_func(true_range(ohlc_df), window) * multiplier


def apply_trail(roll_sr, trail):
    # Apply trail to rolling series
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
    # Pass: Rate is higher than trailing stop
    rollmin_sr = rate_sr.rolling(window=len(rate_sr.index), min_periods=1).min()
    stop_sr = apply_trail(rollmin_sr, trail)
    sellstops = np.flatnonzero(np.where(rate_sr > stop_sr, 1, 0))
    sellstop = sellstops[0] if len(sellstops) > 0 else None
    return sellstop


def trailstop_exit(rate_sr, trail):
    # Pass: Rate is lower than trailing stop
    rollmax_sr = rate_sr.rolling(window=len(rate_sr.index), min_periods=1).max()
    stop_sr = apply_trail(rollmax_sr, -trail)
    sellstops = np.flatnonzero(np.where(rate_sr < stop_sr, 1, 0))
    sellstop = sellstops[0] if len(sellstops) > 0 else None
    return sellstop


def traverse_trailstops(rate_sr, entry_trail, exit_trail):
    # In case both vectors are calculated using trailing stop and thus depending
    trailstops = [0]
    entry_vector = np.zeros(len(rate_sr.index))
    exit_vector = np.zeros(len(rate_sr.index))
    while True:
        if len(trailstops) % 2 == 0:  # exit or entry?
            i = trailstops[-1] + 1  # exit excluded
            j = trailstop_entry(rate_sr.iloc[i:], entry_trail)
            if j is not None:
                trailstops.append(i + j)  # index adjusted to rate_sr
                entry_vector[i+j] = 1
            else:
                break
        else:
            i = trailstops[-1]  # entry included
            j = trailstop_exit(rate_sr.iloc[i:], exit_trail)
            if j is not None:
                trailstops.append(i + j)
                exit_vector[i+j] = 1
            else:
                break
    return reduce_vector(entry_vector), reduce_vector(exit_vector)


def trailstop_entry_vector(rate_sr, exit_vector, trail):
    # Exit vector needed
    # Exit resets entry -> vectorized solution possible -> divide and conquer
    groups = rate_sr.groupby(np.cumsum(exit_vector))
    rel_entry_pos = groups.apply(lambda x: trailstop_entry(x, trail)).values
    abs_exit_pos = np.insert(np.flatnonzero(exit_vector), 0, 0)
    abs_entry_pos = rel_entry_pos+abs_exit_pos
    abs_entry_pos = abs_entry_pos[~np.isnan(abs_entry_pos)]
    abs_entry_pos = abs_entry_pos.astype(int)
    vector = np.zeros(len(rate_sr.index))
    vector[abs_entry_pos] = 1
    return reduce_vector(vector)


def trailstop_exit_vector(rate_sr, entry_vector, trail):
    # Entry vector needed
    # Entry doesn't reset exit -> vectorized solution not possible -> iterate)
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
    return reduce_vector(vector)


# Bollinger bands
#################

def bbounds(rate_sr, window, std_n):
    rollmean_sr = rate_sr.rolling(window=window, min_periods=1).mean()
    rollstd_sr = rate_sr.rolling(window=window, min_periods=1).std()
    upper_band_sr = rollmean_sr + std_n * rollstd_sr
    lower_band_sr = rollmean_sr - std_n * rollstd_sr
    return upper_band_sr, lower_band_sr


def bbounds_entry_vector(rate_sr, upper_band_sr):
    # Pass: Rate is above the upper bollinger band
    return reduce_vector(np.where(rate_sr > upper_band_sr, 1, 0))


def bbounds_exit_vector(rate_sr, lower_band_sr):
    # Pass: Rate is below the lower bollinger band
    return reduce_vector(np.where(rate_sr < lower_band_sr, 1, 0))
