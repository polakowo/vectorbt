"""Class for wrapping NumPy arrays into Series/DataFrames."""

import numpy as np
import pandas as pd
import warnings

from vectorbt.utils import checks
from vectorbt.utils.config import Configured
from vectorbt.utils.datetime import freq_delta, DatetimeTypes, to_time_units
from vectorbt.utils.array import get_ranges_arr
from vectorbt.base import index_fns, reshape_fns
from vectorbt.base.indexing import IndexingError, PandasIndexer
from vectorbt.base.column_grouper import ColumnGrouper


def indexing_on_wrapper_meta(obj, pd_indexing_func, index=None, columns=None,
                             column_only_select=None, group_select=None):
    """Perform indexing on `ArrayWrapper` and also return indexing metadata.

    Takes into account column grouping.

    Set `column_only_select` to True to index the array wrapper as a Series of columns.
    This way, selection of index (axis 0) can be avoided. Set `group_select` to True
    to select groups rather than columns. Takes effect only if grouping is enabled.

    !!! note
        If `column_only_select` is True, make sure to index the array wrapper
        as a Series of columns rather than a DataFrame. For example, the operation
        `.iloc[:, :2]` should become `.iloc[:2]`. Operations are not allowed if the
        object is already a Series and thus has only one column/group."""
    from vectorbt import defaults

    if column_only_select is None:
        column_only_select = obj.column_only_select
    if column_only_select is None:
        column_only_select = defaults.array_wrapper['column_only_select']
    if group_select is None:
        group_select = obj.group_select
    if group_select is None:
        group_select = defaults.array_wrapper['group_select']
    if index is None:
        index = obj.index
    if columns is None:
        if group_select:
            columns = obj.grouper.get_columns()
        else:
            columns = obj.columns
    if group_select:
        # Groups as columns
        i_wrapper = ArrayWrapper(index, columns, obj.grouped_ndim)
    else:
        # Columns as columns
        i_wrapper = ArrayWrapper(index, columns, obj.ndim)
    n_rows = len(index)
    n_cols = len(columns)

    if column_only_select:
        if i_wrapper.ndim == 1:
            raise IndexingError("Columns only: One column already selected")
        col_mapper = i_wrapper.wrap_reduced(np.arange(n_cols), columns=columns)
        try:
            col_mapper = pd_indexing_func(col_mapper)
        except pd.core.indexing.IndexingError as err:
            warnings.warn("Columns only: Make sure to treat this object "
                          "as a Series of columns rather than a DataFrame", stacklevel=2)
            raise err
        if checks.is_series(col_mapper):
            new_columns = col_mapper.index
            col_idxs = col_mapper.values
            new_ndim = 2
        else:
            new_columns = columns[[col_mapper]]
            col_idxs = col_mapper
            new_ndim = 1
        new_index = index
        idx_idxs = np.arange(len(index))
    else:
        idx_mapper = i_wrapper.wrap(
            np.broadcast_to(np.arange(n_rows)[:, None], (n_rows, n_cols)),
            index=index,
            columns=columns
        )
        idx_mapper = pd_indexing_func(idx_mapper)
        if i_wrapper.ndim == 1:
            if not checks.is_series(idx_mapper):
                raise IndexingError("Selection of a scalar is not allowed")
            idx_idxs = idx_mapper.values
            col_idxs = 0
        else:
            col_mapper = i_wrapper.wrap(
                np.broadcast_to(np.arange(n_cols), (n_rows, n_cols)), 
                index=index, 
                columns=columns
            )
            col_mapper = pd_indexing_func(col_mapper)
            if checks.is_frame(idx_mapper):
                idx_idxs = idx_mapper.values[:, 0]
                col_idxs = col_mapper.values[0]
            elif checks.is_series(idx_mapper):
                one_col = np.all(col_mapper.values == col_mapper.values.item(0))
                one_idx = np.all(idx_mapper.values == idx_mapper.values.item(0))
                if one_col and one_idx:
                    # One index and one column selected, multiple times
                    raise IndexingError("Must select at least two unique indices in one of both axes")
                elif one_col:
                    # One column selected
                    idx_idxs = idx_mapper.values
                    col_idxs = col_mapper.values[0]
                elif one_idx:
                    # One index selected
                    idx_idxs = idx_mapper.values[0]
                    col_idxs = col_mapper.values
            else:
                raise IndexingError("Selection of a scalar is not allowed")
        new_index = index_fns.get_index(idx_mapper, 0)
        new_columns = index_fns.get_index(idx_mapper, 1)
        new_ndim = idx_mapper.ndim

    if obj.grouper.group_by is not None:
        # Grouping enabled
        if np.asarray(idx_idxs).ndim == 0:
            raise IndexingError("Flipping index and columns is not allowed")

        if group_select:
            # Selection based on groups
            # Get indices of columns corresponding to selected groups
            group_idxs = col_idxs
            group_idxs_arr = reshape_fns.to_1d(group_idxs)
            group_start_idxs = obj.grouper.get_group_start_idxs()[group_idxs_arr]
            group_end_idxs = obj.grouper.get_group_end_idxs()[group_idxs_arr]
            ungrouped_col_idxs = get_ranges_arr(group_start_idxs, group_end_idxs)
            ungrouped_columns = obj.columns[ungrouped_col_idxs]
            if new_ndim == 1 and len(ungrouped_columns) == 1:
                ungrouped_ndim = 1
                ungrouped_col_idxs = ungrouped_col_idxs[0]
            else:
                ungrouped_ndim = 2

            # Get indices of selected groups corresponding to the new columns
            # We could do obj.group_by[ungrouped_col_idxs] but indexing operation may have changed the labels
            group_lens = obj.grouper.get_group_lens()[group_idxs_arr]
            ungrouped_group_idxs = np.full(len(ungrouped_columns), 0)
            ungrouped_group_idxs[group_lens[:-1]] = 1
            ungrouped_group_idxs = np.cumsum(ungrouped_group_idxs)

            return obj.copy(
                index=new_index,
                columns=ungrouped_columns,
                ndim=ungrouped_ndim,
                grouped_ndim=new_ndim,
                group_by=new_columns[ungrouped_group_idxs]
            ), idx_idxs, group_idxs, ungrouped_col_idxs

        # Selection based on columns
        col_idxs_arr = reshape_fns.to_1d(col_idxs)
        return obj.copy(
            index=new_index,
            columns=new_columns,
            ndim=new_ndim,
            grouped_ndim=None,
            group_by=obj.grouper.group_by[col_idxs_arr]
        ), idx_idxs, col_idxs, col_idxs

    # Grouping disabled
    return obj.copy(
        index=new_index,
        columns=new_columns,
        ndim=new_ndim,
        grouped_ndim=None,
        group_by=None
    ), idx_idxs, col_idxs, col_idxs


def array_wrapper_indexing_func(obj, pd_indexing_func, **kwargs):
    """Perform indexing on `ArrayWrapper`"""
    return indexing_on_wrapper_meta(obj, pd_indexing_func, **kwargs)[0]


class ArrayWrapper(Configured, PandasIndexer):
    """Class that stores index, columns and shape metadata for wrapping NumPy arrays.

    If the underlying object is a Series, pass `[sr.name]` as `columns`.

    !!! note
        This class is meant to be immutable. To change any attribute, use `ArrayWrapper.copy`."""

    def __init__(self, index, columns, ndim, freq=None, column_only_select=None, group_select=None,
                 grouped_ndim=None, group_by=None, allow_enable=True, allow_disable=True, allow_modify=True):
        Configured.__init__(
            self,
            index=index,
            columns=columns,
            ndim=ndim,
            freq=freq,
            column_only_select=column_only_select,
            group_select=group_select,
            grouped_ndim=grouped_ndim,
            group_by=group_by,
            allow_enable=allow_enable,
            allow_disable=allow_disable,
            allow_modify=allow_modify
        )

        checks.assert_not_none(index)
        checks.assert_not_none(columns)
        checks.assert_not_none(ndim)
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)

        self._index = index
        self._columns = columns
        self._ndim = ndim
        self._freq = freq
        self._column_only_select = column_only_select
        self._group_select = group_select
        self._grouper = ColumnGrouper(
            columns,
            group_by=group_by,
            allow_enable=allow_enable,
            allow_disable=allow_disable,
            allow_modify=allow_modify
        )
        self._grouped_ndim = grouped_ndim

        PandasIndexer.__init__(
            self,
            array_wrapper_indexing_func,
            column_only_select=column_only_select,
            group_select=group_select
        )

    @classmethod
    def from_obj(cls, obj, *args, **kwargs):
        """Derive metadata from an object."""
        index = index_fns.get_index(obj, 0)
        columns = index_fns.get_index(obj, 1)
        ndim = obj.ndim
        return cls(index, columns, ndim, *args, **kwargs)

    @property
    def index(self):
        """Index."""
        return self._index

    @property
    def columns(self):
        """Columns."""
        return self._columns

    @property
    def name(self):
        """Name."""
        if self.ndim == 1:
            if self.columns[0] == 0:
                return None
            return self.columns[0]
        return None

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

    @property
    def shape_2d(self):
        """Shape as if the object was two-dimensional."""
        if self.ndim == 1:
            return self.shape[0], 1
        return self.shape

    @property
    def freq(self):
        """Index frequency."""
        from vectorbt import defaults

        freq = self._freq
        if freq is None:
            freq = defaults.array_wrapper['freq']
        if freq is not None:
            return freq_delta(freq)
        if isinstance(self.index, DatetimeTypes):
            if self.index.freq is not None:
                try:
                    return freq_delta(self.index.freq)
                except ValueError as e:
                    warnings.warn(repr(e), stacklevel=2)
            if self.index.inferred_freq is not None:
                try:
                    return freq_delta(self.index.inferred_freq)
                except ValueError as e:
                    warnings.warn(repr(e), stacklevel=2)
        return freq

    @property
    def column_only_select(self):
        """Whether to perform indexing on columns only."""
        return self._column_only_select

    @property
    def group_select(self):
        """Whether to perform indexing on groups."""
        return self._group_select

    @property
    def grouper(self):
        """Column grouper."""
        return self._grouper

    @property
    def grouped_ndim(self):
        """Number of dimensions under column grouping."""
        if self._grouped_ndim is None:
            if self.grouper.is_grouped():
                return 2 if self.grouper.get_group_count() > 1 else 1
            return self.ndim
        return self._grouped_ndim

    def regroup(self, group_by):
        """Regroup this object."""
        if self.grouper.is_grouping_changed(group_by=group_by):
            self.grouper.check_group_by(group_by=group_by)
            return self.copy(group_by=group_by)
        return self

    def to_time_units(self, a):
        """Convert array to time units."""
        if self.freq is None:
            raise ValueError("Couldn't parse the frequency of index. You must set `freq`.")
        return to_time_units(a, self.freq)

    def wrap(self, a, index=None, columns=None, dtype=None, collapse=None, **kwargs):
        """Wrap a NumPy array using the stored metadata."""
        checks.assert_ndim(a, (1, 2))
        group_by = self.grouper.resolve_group_by(**kwargs)

        a = np.asarray(a)
        a = reshape_fns.soft_to_ndim(a, self.ndim)
        if index is None:
            index = self.index
        if columns is None:
            columns = self.grouper.get_columns(**kwargs)
        if collapse is None:
            collapse = group_by is not None and group_by is not False and self.grouped_ndim == 1
        if columns is not None and len(columns) == 1:
            name = columns[0]
            if name == 0:  # was a Series before
                name = None
        else:
            name = None

        # Perform checks
        if index is not None:
            checks.assert_shape_equal(a, index, axis=(0, 0))
        if a.ndim == 2 and columns is not None:
            checks.assert_shape_equal(a, columns, axis=(1, 0))

        if a.ndim == 1:
            return pd.Series(a, index=index, name=name, dtype=dtype)
        if a.ndim == 2 and a.shape[1] == 1 and collapse:
            return pd.Series(a[:, 0], index=index, name=name, dtype=dtype)
        return pd.DataFrame(a, index=index, columns=columns, dtype=dtype)

    def wrap_reduced(self, a, index=None, columns=None, time_units=False, collapse=None, **kwargs):
        """Wrap result of reduction.

        `index` can be set when reducing to an array of values (vs. one value) per column.
        `columns` can be set to override object's default columns.

        If `time_units` is set, calls `to_time_units`."""
        checks.assert_not_none(self.ndim)
        group_by = self.grouper.resolve_group_by(**kwargs)
        if columns is None:
            columns = self.grouper.get_columns(**kwargs)
        if collapse is None:
            collapse = group_by is not None and group_by is not False and self.grouped_ndim == 1

        a = np.asarray(a)
        if time_units:
            a = self.to_time_units(a)
        if a.ndim == 0:
            # Scalar per Series/DataFrame
            if time_units:
                return pd.to_timedelta(a.item())
            return a.item()
        if a.ndim == 1:
            if self.ndim == 1 or (self.ndim == 2 and len(columns) == 1 and collapse):
                if a.shape[0] == 1:
                    # Scalar per Series/DataFrame with one column
                    if time_units:
                        return pd.to_timedelta(a[0])
                    return a[0]
                # Array per Series
                name = columns[0]
                if name == 0:  # was a Series before
                    name = None
                return pd.Series(a, index=index, name=name)
            # Scalar per column in a DataFrame
            if index is None:
                index = columns
            return pd.Series(a, index=index)
        if self.ndim == 1:
            # Array per Series
            name = columns[0]
            if name == 0:  # was a Series before
                name = None
            return pd.Series(a[:, 0], index=index, name=name)
        # Array per column in a DataFrame
        return pd.DataFrame(a, index=index, columns=columns)

