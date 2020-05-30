import numpy as np
import pandas as pd
from datetime import timedelta

from vectorbt.utils import reshape_fns, checks


def to_time_units(obj, time_delta):
    """Multiply each element with `time_delta` to get result in time units."""
    total_seconds = pd.Timedelta(time_delta).total_seconds()
    def to_td(x): return timedelta(seconds=x * total_seconds) if ~np.isnan(x) else np.nan
    to_td = np.vectorize(to_td, otypes=[np.object])
    return obj.vbt.wrap(to_td(obj.values))


class TSArrayWrapper(reshape_fns.ArrayWrapper):
    """Introduces methods for wrapping time series on top of `vectorbt.utils.reshape_fns.ArrayWrapper`."""

    @property
    def timedelta(self):
        """Return time delta of the index frequency."""
        checks.assert_type(self.index, (pd.DatetimeIndex, pd.PeriodIndex))

        if self.index.freq is not None:
            return pd.to_timedelta(pd.tseries.frequencies.to_offset(self.index.freq))
        elif self.index.inferred_freq is not None:
            return pd.to_timedelta(pd.tseries.frequencies.to_offset(self.index.inferred_freq))
        return (self.index[1:] - self.index[:-1]).min()

    def wrap_reduced(self, a, time_units=False, index=None):
        """Wrap result of reduction.

        If `time_units` is set, calls `vectorbt.timeseries.common.to_time_units`."""
        if a.ndim == 1:
            # Each column reduced to a single value
            a_obj = pd.Series(a, index=self.columns)
            if time_units:
                if isinstance(time_units, bool):
                    time_units = self.timedelta
                a_obj = to_time_units(a_obj, time_units)
            if self.ndim > 1:
                return a_obj
            return a_obj.iloc[0]
        else:
            # Each column reduced to an array
            if index is None:
                index = pd.Index(range(a.shape[0]))
            a_obj = self.wrap(a, index=index)
            if time_units:
                if isinstance(time_units, bool):
                    time_units = self.timedelta
                a_obj = to_time_units(a_obj, time_units)
            return a_obj
