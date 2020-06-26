"""Class for wrapping NumPy arrays into Series/DataFrames."""

import numpy as np
import pandas as pd

from vectorbt.utils import checks
from vectorbt.base import index_fns, reshape_fns


class ArrayWrapper:
    """Class that stores index, columns and shape metadata for wrapping NumPy arrays."""

    def __init__(self, index=None, columns=None, ndim=None):
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        self._index = index
        self._columns = columns
        self._ndim = ndim

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
    def shape(self):
        """Shape."""
        if self.ndim == 1:
            return len(self.index),
        return len(self.index), len(self.columns)

    @classmethod
    def from_obj(cls, obj, *args, **kwargs):
        """Derive metadata from an object."""
        index = index_fns.get_index(obj, 0)
        columns = index_fns.get_index(obj, 1)
        ndim = obj.ndim
        return cls(*args, index=index, columns=columns, ndim=ndim, **kwargs)

    def wrap(self, arg, index=None, columns=None, ndim=None, dtype=None):
        """Wrap a NumPy array using the stored metadata."""
        arg = np.asarray(arg)
        if ndim is None:
            ndim = self.ndim
        if ndim is not None:
            arg = reshape_fns.soft_broadcast_to_ndim(arg, self.ndim)
        if index is None:
            index = self.index
        if columns is None:
            columns = self.columns
        if columns is not None and len(columns) == 1:
            name = columns[0]
        else:
            name = None

        # Perform checks
        if index is not None:
            checks.assert_same_shape(arg, index, axis=(0, 0))
        if arg.ndim == 2 and columns is not None:
            checks.assert_same_shape(arg, columns, axis=(1, 0))

        if arg.ndim == 1:
            return pd.Series(arg, index=index, name=name, dtype=dtype)
        return pd.DataFrame(arg, index=index, columns=columns, dtype=dtype)

    def wrap_reduced(self, a, index=None):
        """Wrap result of reduction."""
        if a.ndim == 0:
            return a
        elif a.ndim == 1:
            # Each column reduced to a single value
            a_obj = pd.Series(a, index=self.columns)
            if self.ndim > 1:
                return a_obj
            return a_obj.iloc[0]
        else:
            # Each column reduced to an array
            if index is None:
                index = pd.Index(range(a.shape[0]))
            a_obj = self.wrap(a, index=index)
            return a_obj
