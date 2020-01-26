import pandas as pd
import numpy as np

from vectorbt.utils.array import *
from vectorbt.signals import Signals
from vectorbt.timeseries import TimeSeries


class MA():
    """The SMA is a technical indicator for determining if an asset price 
    will continue or reverse a bull or bear trend. The SMA is calculated as 
    the arithmetic average of an asset's price over some period.

    The EMA is a moving average that places a greater weight and 
    significance on the most recent data points."""

    def __init__(self, ts, fast_window, slow_window, ewm=False):
        if not isinstance(ts, TimeSeries): 
            raise TypeError("Argument ts is not TimeSeries")

        self.ts = ts
        if ewm:
            self.fast_ma = ewma(ts, window=fast_window)
            self.slow_ma = ewma(ts, window=slow_window)
        else:
            self.fast_ma = rolling_mean(ts, window=fast_window)
            self.slow_ma = rolling_mean(ts, window=slow_window)
        self.fast_ma[:slow_window] = 0
        self.slow_ma[:slow_window] = 0

    def entries(self):
        return Signals(self.fast_ma > self.slow_ma, index=self.ts.index)

    def exits(self):
        return Signals(self.fast_ma < self.slow_ma, index=self.ts.index)


class MACD():
    """MACD, short for moving average convergence/divergence."""

    def __init__(self, ts, fast_window, slow_window, signal_window):
        if not isinstance(ts, TimeSeries): 
            raise TypeError("Argument ts is not TimeSeries")

        self.ts = ts
        fast_ewma = ewma(ts, window=fast_window)
        slow_ewma = ewma(ts, window=slow_window)
        self.macd = fast_ewma - slow_ewma
        self.signal = ewma(self.macd, window=signal_window)
        self.hist = self.macd - self.signal

    def entries(self):
        return Signals(self.macd > self.signal, index=self.ts.index)

    def exits(self):
        return Signals(self.macd < self.signal, index=self.ts.index)

    def hist_entries(self, n):
        # n-st raise in a row while hist being negative -> entry
        diff = np.diff(self.hist, prepend=self.hist[0])
        return Signals((self.hist < 0) & (diff > 0), index=self.ts.index)

    def hist_exits(self, n):
        # n-st drop in a row while hist being positive -> exit
        diff = np.diff(self.hist, prepend=self.hist[0])
        return Signals((self.hist > 0) & (diff < 0), index=self.ts.index)


class BB():
    """Bollinger Bands® are volatility bands placed above and below a moving average."""

    def __init__(self, ts, window, std_n):
        if not isinstance(ts, TimeSeries): 
            raise TypeError("Argument ts is not TimeSeries")

        self.ts = ts
        rollmean = rolling_mean(ts, window=window)
        rollstd = rolling_std(ts, window=window)
        self.lower_band = rollmean - std_n * rollstd
        self.upper_band = rollmean + std_n * rollstd

    def entries(self):
        return Signals(self.ts < self.lower_band, index=self.ts.index)

    def exits(self):
        return Signals(self.ts > self.upper_band, index=self.ts.index)


class RSI():
    """The relative strength index (RSI) is a momentum indicator that 
    measures the magnitude of recent price changes to evaluate overbought 
    or oversold conditions in the price of a stock or other asset."""

    def __init__(self, ts, window):
        if not isinstance(ts, TimeSeries): 
            raise TypeError("Argument ts is not TimeSeries")

        delta = np.diff(ts)
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        rs = ewma(up, window=window) / ewma(np.abs(down), window=window)
        self.rsi = 100 - 100 / (1 + rs)

    def entries(self, lower_bound):
        return Signals(self.rsi < lower_bound, index=self.ts.index)

    def exits(self, upper_bound):
        return Signals(self.rsi > upper_bound, index=self.ts.index)


class BEP():
    """Bullish/Bearish Engulfing Pattern:
    Entry once hollow body completely engulfs the previous filled body/candlestick"""

    def __init__(self, open_ts, high_ts, low_ts, close_ts):
        if not isinstance(open_ts, TimeSeries): 
            raise TypeError("Argument open_ts is not TimeSeries")
        if not isinstance(high_ts, TimeSeries): 
            raise TypeError("Argument high_ts is not TimeSeries")
        if not isinstance(low_ts, TimeSeries): 
            raise TypeError("Argument low_ts is not TimeSeries")
        if not isinstance(close_ts, TimeSeries): 
            raise TypeError("Argument close_ts is not TimeSeries")

        self.hollow = close_ts[1:] > open_ts[1:]
        self.last_candle = close_ts[:-1] - open_ts[:-1]
        self.candle = close_ts[1:] - open_ts[1:]

    def entries(self, amount):
        engulfing = self.candle > amount * self.last_candle
        entries = np.insert(self.hollow & engulfing, 0, False)
        # Close is unknown (future) data -> shift vector
        entries = np.insert(entries[1:], 0, False)
        return Signals(entries, index=entries.index)

    def exits(self, amount):
        engulfing = self.candle < amount * self.last_candle
        exits = np.insert(self.hollow & engulfing, 0, False)
        # Close is unknown (future) data -> shift vector
        exits = np.insert(exits[1:], 0, False)
        return Signals(exits, index=entries.index)


#####################
# Risk minimization #
#####################

class StopLoss():
    """A stop-loss is designed to limit an investor's loss on a security position. 
    Setting a stop-loss order for 10% below the price at which you bought the stock 
    will limit your loss to 10%."""

    def __init__(self, ts, entries):
        if not isinstance(ts, TimeSeries): 
            raise TypeError("Argument ts is not TimeSeries")
        if not isinstance(entries, Signals): 
            raise TypeError("Argument entries is not Signals")

        self.ts = ts
        self.entries = entries

    def exits(self, stop):

        def exit_func(ts, prev_idx=None, next_idx=None):
            """Index of the first event when ts went below the stop."""

            nonlocal stop
            # stop-loss order is being made when the stock is bought (at entry signal)
            if isinstance(stop, np.ndarray):
                # Stop is an absolute value
                stop_value = stop[prev_idx]
            else:
                # Stop is in % below the ts
                stop_value = (1 - stop) * ts[prev_idx]
            stop_idxs = np.flatnonzero(ts < stop_value)
            range_mask = stop_idxs > prev_idx
            if next_idx is not None:
                range_mask = range_mask & (stop_idxs < next_idx)
            stop_idxs = stop_idxs[range_mask]
            return stop_idxs[0] if len(stop_idxs) > 0 else None

        return Signals.generate_exits(self.ts, self.entries, exit_func)


class TrailingStop():
    """A Trailing Stop order is a stop order that can be set at a defined percentage 
    or amount away from the current market price. The main difference between a regular 
    stop loss and a trailing stop is that the trailing stop moves as the price moves."""

    def __init__(self, ts, entries):
        if not isinstance(ts, TimeSeries): 
            raise TypeError("Argument ts is not TimeSeries")
        if not isinstance(entries, Signals): 
            raise TypeError("Argument entries is not Signals")
        
        self.ts = ts
        self.entries = entries

    def exits(self, stop):

        def exit_func(ts, prev_idx=None, next_idx=None):
            """Index of the first event when ts went beyond the stop."""
            nonlocal stop

            # propagate peaks from entry using expanding max
            peak_from_entry = rolling_max(ts[prev_idx:])
            dummy = np.zeros_like(ts[:prev_idx])
            # bring to the old shape
            peak = np.insert(peak_from_entry, 0, dummy)
            if isinstance(stop, np.ndarray):
                # Stop is an absolute value
                # Each element in stop_values gets assigned to the stop of the last peak
                raising_idxs = np.flatnonzero(pct_change(peak))
                stop_values = ffill(stop[raising_idxs])
            else:
                # Stop is in % below the peak ts
                stop_values = (1 - stop) * peak

            stop_idxs = np.flatnonzero(ts < stop_values)
            range_mask = stop_idxs > prev_idx
            if next_idx is not None:
                range_mask = range_mask & (stop_idxs < next_idx)
            stop_idxs = stop_idxs[range_mask]
            return stop_idxs[0] if len(stop_idxs) > 0 else None

        return Signals.generate_exits(self.ts, self.entries, exit_func)
