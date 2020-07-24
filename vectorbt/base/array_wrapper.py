"""Class for wrapping NumPy arrays into Series/DataFrames."""

import numpy as np
import pandas as pd

from vectorbt.utils import checks
from vectorbt.utils.datetime import freq_delta, DatetimeTypes, to_time_units
from vectorbt.base import index_fns, reshape_fns


class ArrayWrapper:
    """Class that stores index, columns and shape metadata for wrapping NumPy arrays."""

    def __init__(self, index=None, columns=None, ndim=None, freq=None):
        if index is not None and not isinstance(index, pd.Index):
            index = pd.Index(index)
        if columns is not None and not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        self._index = index
        self._columns = columns
        self._ndim = ndim
        self._freq = None
        if freq is not None:
            self._freq = freq_delta(freq)
        elif isinstance(self.index, DatetimeTypes):
            if self.index.freq is not None:
                self._freq = freq_delta(self.index.freq)
            elif self.index.inferred_freq is not None:
                self._freq = freq_delta(self.index.inferred_freq)

    @property
    def index(self):
        """Index."""
        return self._index

    @property
    def columns(self):
        """Columns."""
        return self._columns

    @property
    def ndim(self):
        """Number of dimensions."""
        return self._ndim

    @property
    def name(self):
        """Name."""
        if self.ndim == 1:
            if self.columns[0] == 0:
                return None
            return self.columns[0]
        return None

    @property
    def shape(self):
        """Shape."""
        if self.ndim == 1:
            return len(self.index),
        return len(self.index), len(self.columns)

    @property
    def freq(self):
        """Index frequency."""
        return self._freq

    def to_time_units(self, a):
        """Convert array to time units."""
        if self.freq is None:
            raise ValueError("Couldn't parse the frequency of index. You must set `freq`.")
        return to_time_units(a, self.freq)

    @classmethod
    def from_obj(cls, obj, *args, **kwargs):
        """Derive metadata from an object."""
        index = index_fns.get_index(obj, 0)
        columns = index_fns.get_index(obj, 1)
        ndim = obj.ndim
        return cls(*args, index=index, columns=columns, ndim=ndim, **kwargs)

    def wrap(self, a, index=None, columns=None, ndim=None, dtype=None):
        """Wrap a NumPy array using the stored metadata."""
        checks.assert_ndim(a, (1, 2))

        a = np.asarray(a)
        if ndim is None:
            ndim = self.ndim
        if ndim is not None:
            a = reshape_fns.soft_to_ndim(a, self.ndim)
        if index is None:
            index = self.index
        if columns is None:
            columns = self.columns
        if columns is not None and len(columns) == 1:
            name = columns[0]
            if name == 0:  # was a Series before
                name = None
        else:
            name = None

        # Perform checks
        if index is not None:
            checks.assert_same_shape(a, index, axis=(0, 0))
        if a.ndim == 2 and columns is not None:
            checks.assert_same_shape(a, columns, axis=(1, 0))

        if a.ndim == 1:
            return pd.Series(a, index=index, name=name, dtype=dtype)
        return pd.DataFrame(a, index=index, columns=columns, dtype=dtype)

    def wrap_reduced(self, a, index=None, time_units=False):
        """Wrap result of reduction.

        Argument `index` can be passed when reducing to an array of values (vs. one value) per column.

        If `time_units` is set, calls `to_time_units`."""
        checks.assert_not_none(self.ndim)

        a = np.asarray(a)
        if time_units:
            a = self.to_time_units(a)
        if a.ndim == 0:
            # Scalar value
            if time_units:
                return pd.to_timedelta(a.item())
            return a.item()
        elif a.ndim == 1:
            if self.ndim == 1:
                if a.shape[0] == 1:
                    # Scalar value
                    if time_units:
                        return pd.to_timedelta(a[0])
                    return a[0]
                # Array per series
                name = self.columns[0]
                if name == 0:  # was a Series before
                    name = None
                return pd.Series(a, index=index, name=name)
            # Value per column
            return pd.Series(a, index=self.columns)
        if self.ndim == 1:
            # Array per series
            name = self.columns[0]
            if name == 0:  # was a Series before
                name = None
            return pd.Series(a[:, 0], index=index, name=name)
        # Value per column
        return pd.DataFrame(a, index=index, columns=self.columns)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if not pd.Index.equals(self.index, other.index):
            return False
        if not pd.Index.equals(self.columns, other.columns):
            return False
        if self.ndim != other.ndim:
            return False
        return True
