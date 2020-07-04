"""Common functions and classes."""

import numpy as np
import pandas as pd

from vectorbt.utils import checks
from vectorbt.base.array_wrapper import ArrayWrapper

DatetimeTypes = (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)


def freq_delta(freq):
    """Return delta of frequency."""
    if isinstance(freq, str) and not freq[0].isdigit():
        # Otherwise "ValueError: unit abbreviation w/o a number"
        freq = '1' + freq
    return pd.to_timedelta(freq)


def to_time_units(obj, freq):
    """Multiply each element with `freq_delta` to get result in time units."""
    if not checks.is_array(obj):
        obj = np.asarray(obj)
    return obj * freq_delta(freq).to_timedelta64()


class TSArrayWrapper(ArrayWrapper):
    """Introduces methods for wrapping time series on top of `vectorbt.base.array_wrapper.ArrayWrapper`."""

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

        If `time_units` is set, calls `vectorbt.tseries.common.to_time_units`."""
        if time_units:
            a = self.to_time_units(a)
        return ArrayWrapper.wrap_reduced(self, a, index=index)
