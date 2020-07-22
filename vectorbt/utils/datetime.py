"""Datetime utilities."""

import numpy as np
import pandas as pd

from vectorbt.utils import checks

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
    return obj * freq_delta(freq)