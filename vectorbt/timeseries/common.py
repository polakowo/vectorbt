"""Common functions and classes."""

import numpy as np
import pandas as pd
from datetime import timedelta


def to_time_units(obj, time_delta):
    """Multiply each element with `time_delta` to get result in time units."""
    total_seconds = pd.Timedelta(time_delta).total_seconds()
    def to_td(x): return timedelta(seconds=x * total_seconds) if ~np.isnan(x) else np.nan
    to_td = np.vectorize(to_td, otypes=[np.object])
    return obj.vbt.wrap_array(to_td(obj.vbt.to_array()))
