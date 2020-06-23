"""Common functions and classes."""

import numpy as np
import pandas as pd
from datetime import timedelta

from vectorbt.utils import checks
from vectorbt.utils.array_wrapper import ArrayWrapper

DatetimeTypes = (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)


def to_time_units(obj, freq_delta):
    """Multiply each element with `freq_delta` to get result in time units."""
    checks.assert_not_none(freq_delta)

    total_seconds = pd.Timedelta(freq_delta).total_seconds()

    def to_td(x):
        return timedelta(seconds=x * total_seconds) if ~np.isnan(x) else np.nan

    to_td = np.vectorize(to_td, otypes=[np.object])
    if checks.is_pandas(obj):
        return obj.vbt.wrap(to_td(obj.values))
    arr = np.asarray(obj)
    if arr.ndim == 0:
        return to_td(arr.item())
    return to_td(arr)


def freq_delta(freq):
    """Return delta of frequency."""
    if isinstance(freq, str) and not freq[0].isdigit():
        # Otherwise "ValueError: unit abbreviation w/o a number"
        freq = '1' + freq
    return pd.to_timedelta(freq)


class TSArrayWrapper(ArrayWrapper):
    """Introduces methods for wrapping time series on top of `vectorbt.utils.reshape_fns.ArrayWrapper`."""

    def __init__(self, freq=None, **kwargs):
        ArrayWrapper.__init__(self, **kwargs)

        # Set index frequency
        self._freq = None
        if freq is not None:
            self._freq = freq_delta(freq)
        elif isinstance(self.index, DatetimeTypes):
            if self.index.freq is not None:
                self._freq = freq_delta(self.index.freq)
            elif self.index.inferred_freq is not None:
                self._freq = freq_delta(self.index.inferred_freq)

    @property
    def freq(self):
        """Index frequency."""
        return self._freq

    def to_time_units(self, a):
        """Convert array to time units."""
        if self.freq is None:
            raise Exception("Couldn't parse the frequency of index. You must set `freq`.")
        return to_time_units(a, self.freq)

    def wrap_reduced(self, a, time_units=False, index=None):
        """Wrap result of reduction.

        If `time_units` is set, calls `vectorbt.timeseries.common.to_time_units`."""
        if time_units:
            a = self.to_time_units(a)
        return ArrayWrapper.wrap_reduced(self, a, index=index)
