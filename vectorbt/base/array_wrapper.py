"""Class for wrapping NumPy arrays into Series/DataFrames.

vectorbt's functionality is based upon the ability to perform the most essential pandas operations
using NumPy+Numba stack. One has to convert the Series/DataFrame into the NumPy format, perform
the computation, and put the array back into the pandas format. The last step is done using
`ArrayWrapper`.

It stores metadata of the original pandas object and offers methods `wrap` and `wrap_reduced`
for wrapping NumPy arrays to match the stored metadata as closest as possible.

```python-repl
>>> import numpy as np
>>> import pandas as pd
>>> import vectorbt as vbt
>>> from vectorbt.base.array_wrapper import ArrayWrapper

>>> aw = ArrayWrapper(index=['x', 'y', 'z'], columns=['a', 'b', 'c'], ndim=2)
>>> aw._config
{
    'columns': Index(['a', 'b', 'c'], dtype='object'),
    'group_select': None,
    'ndim': 2,
    'freq': None,
    'column_only_select': None,
    'grouped_ndim': None,
    'index': ['x', 'y', 'z'],
    'allow_modify': True,
    'allow_enable': True,
    'group_by': None,
    'allow_disable': True
}

>>> np.random.seed(42)
>>> a = np.random.uniform(size=(3, 3))
>>> aw.wrap(a)
          a         b         c
x  0.374540  0.950714  0.731994
y  0.598658  0.156019  0.155995
z  0.058084  0.866176  0.601115

>>> aw.wrap_reduced(np.sum(a, axis=0))
a    1.031282
b    1.972909
c    1.489103
dtype: float64
```

It can also be indexed as a regular pandas object and integrates `vectorbt.base.column_grouper.ColumnGrouper`:

```python-repl
>>> aw.loc['x':'y', 'a']._config
{
    'columns': Index(['a'], dtype='object'),
    'group_select': None,
    'ndim': 1,
    'freq': None,
    'column_only_select': None,
    'grouped_ndim': None,
    'index': Index(['x', 'y'], dtype='object'),
    'allow_modify': True,
    'allow_enable': True,
    'group_by': None,
    'allow_disable': True
}

>>> aw.regroup(np.array([0, 0, 1]))._config
{
    'columns': Index(['a', 'b', 'c'], dtype='object'),
    'group_select': None,
    'ndim': 2,
    'freq': None,
    'column_only_select': None,
    'grouped_ndim': None,
    'index': ['x', 'y', 'z'],
    'allow_modify': True,
    'allow_enable': True,
    'group_by': array([0, 0, 1]),
    'allow_disable': True
}
```

Class `Wrapping` is a convenience class meant to be subclassed by classes that do not want to subclass
`ArrayWrapper` but rather use it as an attribute (which is a better SE design pattern anyway!)."""

import numpy as np
import pandas as pd
import warnings

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.config import Configured, merge_dicts
from vectorbt.utils.datetime import freq_to_timedelta, DatetimeIndexes
from vectorbt.utils.array import get_ranges_arr
from vectorbt.utils.decorators import cached_method
from vectorbt.base import index_fns, reshape_fns
from vectorbt.base.indexing import IndexingError, PandasIndexer
from vectorbt.base.column_grouper import ColumnGrouper
from vectorbt.base.reshape_fns import to_pd_array

ArrayWrapperT = tp.TypeVar("ArrayWrapperT", bound="ArrayWrapper")
IndexingMetaT = tp.Tuple[ArrayWrapperT, tp.MaybeArray, tp.MaybeArray, tp.Array1d]


class ArrayWrapper(Configured, PandasIndexer):
    """Class that stores index, columns and shape metadata for wrapping NumPy arrays.
    Tightly integrated with `vectorbt.base.column_grouper.ColumnGrouper`.

    If the underlying object is a Series, pass `[sr.name]` as `columns`.

    `**kwargs` are passed to `vectorbt.base.column_grouper.ColumnGrouper`.

    !!! note
        This class is meant to be immutable. To change any attribute, use `ArrayWrapper.copy`.

        Use methods that begin with `get_` to get group-aware results."""

    def __init__(self,
                 index: tp.IndexLike,
                 columns: tp.IndexLike,
                 ndim: int,
                 freq: tp.Optional[tp.FrequencyLike] = None,
                 column_only_select: tp.Optional[bool] = None,
                 group_select: tp.Optional[bool] = None,
                 grouped_ndim: tp.Optional[int] = None,
                 **kwargs) -> None:
        config = dict(
            index=index,
            columns=columns,
            ndim=ndim,
            freq=freq,
            column_only_select=column_only_select,
            group_select=group_select,
            grouped_ndim=grouped_ndim,
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
        self._grouper = ColumnGrouper(columns, **kwargs)
        self._grouped_ndim = grouped_ndim

        PandasIndexer.__init__(self)
        Configured.__init__(self, **merge_dicts(config, self._grouper._config))

    @cached_method
    def indexing_func_meta(self: ArrayWrapperT,
                           pd_indexing_func: tp.PandasIndexingFunc,
                           index: tp.Optional[tp.IndexLike] = None,
                           columns: tp.Optional[tp.IndexLike] = None,
                           column_only_select: tp.Optional[bool] = None,
                           group_select: tp.Optional[bool] = None,
                           group_by: tp.GroupByLike = None) -> IndexingMetaT:
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
        from vectorbt._settings import settings
        array_wrapper_cfg = settings['array_wrapper']

        if column_only_select is None:
            column_only_select = self.column_only_select
        if column_only_select is None:
            column_only_select = array_wrapper_cfg['column_only_select']
        if group_select is None:
            group_select = self.group_select
        if group_select is None:
            group_select = array_wrapper_cfg['group_select']
        _self = self.regroup(group_by)
        group_select = group_select and _self.grouper.is_grouped()
        if index is None:
            index = _self.index
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        if columns is None:
            if group_select:
                columns = _self.grouper.get_columns()
            else:
                columns = _self.columns
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        if group_select:
            # Groups as columns
            i_wrapper = ArrayWrapper(index, columns, _self.get_ndim())
        else:
            # Columns as columns
            i_wrapper = ArrayWrapper(index, columns, _self.ndim)
        n_rows = len(index)
        n_cols = len(columns)

        if column_only_select:
            if i_wrapper.ndim == 1:
                raise IndexingError("Columns only: Attempting to select a column on a Series")
            try:
                col_mapper = pd_indexing_func(i_wrapper.wrap_reduced(np.arange(n_cols), columns=columns))
            except pd.core.indexing.IndexingError as e:
                warnings.warn("Columns only: Make sure to treat this object "
                              "as a Series of columns rather than a DataFrame", stacklevel=2)
                raise e
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
            idx_mapper = pd_indexing_func(i_wrapper.wrap(
                np.broadcast_to(np.arange(n_rows)[:, None], (n_rows, n_cols)),
                index=index,
                columns=columns
            ))
            if i_wrapper.ndim == 1:
                if not checks.is_series(idx_mapper):
                    raise IndexingError("Selection of a scalar is not allowed")
                idx_idxs = idx_mapper.values
                col_idxs = 0
            else:
                col_mapper = pd_indexing_func(i_wrapper.wrap(
                    np.broadcast_to(np.arange(n_cols), (n_rows, n_cols)),
                    index=index,
                    columns=columns
                ))
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
                        raise IndexingError
                else:
                    raise IndexingError("Selection of a scalar is not allowed")
            new_index = index_fns.get_index(idx_mapper, 0)
            if not isinstance(idx_idxs, np.ndarray):
                # One index selected
                new_columns = index[[idx_idxs]]
            elif not isinstance(col_idxs, np.ndarray):
                # One column selected
                new_columns = columns[[col_idxs]]
            else:
                new_columns = index_fns.get_index(idx_mapper, 1)
            new_ndim = idx_mapper.ndim

        if _self.grouper.is_grouped():
            # Grouping enabled
            if np.asarray(idx_idxs).ndim == 0:
                raise IndexingError("Flipping index and columns is not allowed")

            if group_select:
                # Selection based on groups
                # Get indices of columns corresponding to selected groups
                group_idxs = col_idxs
                group_idxs_arr = reshape_fns.to_1d(group_idxs)
                group_start_idxs = _self.grouper.get_group_start_idxs()[group_idxs_arr]
                group_end_idxs = _self.grouper.get_group_end_idxs()[group_idxs_arr]
                ungrouped_col_idxs = get_ranges_arr(group_start_idxs, group_end_idxs)
                ungrouped_columns = _self.columns[ungrouped_col_idxs]
                if new_ndim == 1 and len(ungrouped_columns) == 1:
                    ungrouped_ndim = 1
                    ungrouped_col_idxs = ungrouped_col_idxs[0]
                else:
                    ungrouped_ndim = 2

                # Get indices of selected groups corresponding to the new columns
                # We could do _self.group_by[ungrouped_col_idxs] but indexing operation may have changed the labels
                group_lens = _self.grouper.get_group_lens()[group_idxs_arr]
                ungrouped_group_idxs = np.full(len(ungrouped_columns), 0)
                ungrouped_group_idxs[group_lens[:-1]] = 1
                ungrouped_group_idxs = np.cumsum(ungrouped_group_idxs)

                return _self.copy(
                    index=new_index,
                    columns=ungrouped_columns,
                    ndim=ungrouped_ndim,
                    grouped_ndim=new_ndim,
                    group_by=new_columns[ungrouped_group_idxs]
                ), idx_idxs, group_idxs, ungrouped_col_idxs

            # Selection based on columns
            col_idxs_arr = reshape_fns.to_1d(col_idxs)
            return _self.copy(
                index=new_index,
                columns=new_columns,
                ndim=new_ndim,
                grouped_ndim=None,
                group_by=_self.grouper.group_by[col_idxs_arr]
            ), idx_idxs, col_idxs, col_idxs

        # Grouping disabled
        return _self.copy(
            index=new_index,
            columns=new_columns,
            ndim=new_ndim,
            grouped_ndim=None,
            group_by=None
        ), idx_idxs, col_idxs, col_idxs

    def indexing_func(self: ArrayWrapperT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> ArrayWrapperT:
        """Perform indexing on `ArrayWrapper`"""
        return self.indexing_func_meta(pd_indexing_func, **kwargs)[0]

    @classmethod
    def from_obj(cls: tp.Type[ArrayWrapperT], obj: tp.ArrayLike, *args, **kwargs) -> ArrayWrapperT:
        """Derive metadata from an object."""
        pd_obj = to_pd_array(obj)
        index = index_fns.get_index(pd_obj, 0)
        columns = index_fns.get_index(pd_obj, 1)
        ndim = pd_obj.ndim
        return cls(index, columns, ndim, *args, **kwargs)

    @classmethod
    def from_shape(cls: tp.Type[ArrayWrapperT], shape: tp.Shape, *args, **kwargs) -> ArrayWrapperT:
        """Derive metadata from shape."""
        index = pd.RangeIndex(start=0, step=1, stop=shape[0])
        columns = pd.RangeIndex(start=0, step=1, stop=shape[1] if len(shape) > 1 else 1)
        ndim = len(shape)
        return cls(index, columns, ndim, *args, **kwargs)

    @property
    def index(self) -> tp.Index:
        """Index."""
        return self._index

    @property
    def columns(self) -> tp.Index:
        """Columns."""
        return self._columns

    def get_columns(self, group_by: tp.GroupByLike = None) -> tp.Index:
        """Get group-aware `ArrayWrapper.columns`."""
        return self.resolve(group_by=group_by).columns

    @property
    def name(self) -> tp.Any:
        """Name."""
        if self.ndim == 1:
            if self.columns[0] == 0:
                return None
            return self.columns[0]
        return None

    def get_name(self, group_by: tp.GroupByLike = None) -> tp.Any:
        """Get group-aware `ArrayWrapper.name`."""
        return self.resolve(group_by=group_by).name

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._ndim

    def get_ndim(self, group_by: tp.GroupByLike = None) -> int:
        """Get group-aware `ArrayWrapper.ndim`."""
        return self.resolve(group_by=group_by).ndim

    @property
    def shape(self) -> tp.Shape:
        """Shape."""
        if self.ndim == 1:
            return len(self.index),
        return len(self.index), len(self.columns)

    def get_shape(self, group_by: tp.GroupByLike = None) -> tp.Shape:
        """Get group-aware `ArrayWrapper.shape`."""
        return self.resolve(group_by=group_by).shape

    @property
    def shape_2d(self) -> tp.Shape:
        """Shape as if the object was two-dimensional."""
        if self.ndim == 1:
            return self.shape[0], 1
        return self.shape

    def get_shape_2d(self, group_by: tp.GroupByLike = None) -> tp.Shape:
        """Get group-aware `ArrayWrapper.shape_2d`."""
        return self.resolve(group_by=group_by).shape_2d

    @property
    def freq(self) -> tp.Optional[pd.Timedelta]:
        """Index frequency."""
        from vectorbt._settings import settings
        array_wrapper_cfg = settings['array_wrapper']

        freq = self._freq
        if freq is None:
            freq = array_wrapper_cfg['freq']
        if freq is not None:
            return freq_to_timedelta(freq)
        if isinstance(self.index, DatetimeIndexes):
            if self.index.freq is not None:
                try:
                    return freq_to_timedelta(self.index.freq)
                except ValueError as e:
                    warnings.warn(repr(e), stacklevel=2)
            if self.index.inferred_freq is not None:
                try:
                    return freq_to_timedelta(self.index.inferred_freq)
                except ValueError as e:
                    warnings.warn(repr(e), stacklevel=2)
        return freq

    def to_time_units(self, a: tp.MaybeArray[float]) -> tp.Union[pd.Timedelta, tp.Array]:
        """Convert array to time units."""
        if self.freq is None:
            raise ValueError("Couldn't parse the frequency of index. You must set `freq`.")
        return a * self.freq

    @property
    def column_only_select(self) -> tp.Optional[bool]:
        """Whether to perform indexing on columns only."""
        return self._column_only_select

    @property
    def group_select(self) -> tp.Optional[bool]:
        """Whether to perform indexing on groups."""
        return self._group_select

    @property
    def grouper(self) -> ColumnGrouper:
        """Column grouper."""
        return self._grouper

    @property
    def grouped_ndim(self) -> int:
        """Number of dimensions under column grouping."""
        if self._grouped_ndim is None:
            if self.grouper.is_grouped():
                return 2 if self.grouper.get_group_count() > 1 else 1
            return self.ndim
        return self._grouped_ndim

    def regroup(self: ArrayWrapperT, group_by: tp.GroupByLike, **kwargs) -> ArrayWrapperT:
        """Regroup this object.

        Only creates a new instance if grouping has changed, otherwise returns itself."""
        if self.grouper.is_grouping_changed(group_by=group_by):
            self.grouper.check_group_by(group_by=group_by)
            grouped_ndim = None
            if self.grouper.is_grouped(group_by=group_by):
                if not self.grouper.is_group_count_changed(group_by=group_by):
                    grouped_ndim = self.grouped_ndim
            return self.copy(grouped_ndim=grouped_ndim, group_by=group_by, **kwargs)
        return self  # important for keeping cache

    @cached_method
    def resolve(self: ArrayWrapperT, group_by: tp.GroupByLike = None, **kwargs) -> ArrayWrapperT:
        """Resolve this object.

        Replaces columns and other metadata with groups."""
        _self = self.regroup(group_by=group_by, **kwargs)
        if _self.grouper.is_grouped():
            return _self.copy(
                columns=_self.grouper.get_columns(),
                ndim=_self.grouped_ndim,
                grouped_ndim=None,
                group_by=None
            )
        return _self  # important for keeping cache

    def wrap(self,
             a: tp.ArrayLike,
             index: tp.Optional[tp.IndexLike] = None,
             columns: tp.Optional[tp.IndexLike] = None,
             dtype: tp.Optional[tp.PandasDTypeLike] = None,
             group_by: tp.GroupByLike = None) -> tp.SeriesFrame:
        """Wrap a NumPy array using the stored metadata."""
        checks.assert_ndim(a, (1, 2))
        _self = self.resolve(group_by=group_by)

        if index is None:
            index = _self.index
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        if columns is None:
            columns = _self.columns
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        if len(columns) == 1:
            name = columns[0]
            if name == 0:  # was a Series before
                name = None
        else:
            name = None

        arr = np.asarray(a)
        arr = reshape_fns.soft_to_ndim(arr, self.ndim)
        checks.assert_shape_equal(arr, index, axis=(0, 0))
        if arr.ndim == 2:
            checks.assert_shape_equal(arr, columns, axis=(1, 0))
        if arr.ndim == 1:
            return pd.Series(arr, index=index, name=name, dtype=dtype)
        if arr.ndim == 2:
            if arr.shape[1] == 1 and _self.ndim == 1:
                return pd.Series(arr[:, 0], index=index, name=name, dtype=dtype)
            return pd.DataFrame(arr, index=index, columns=columns, dtype=dtype)
        raise ValueError(f"{arr.ndim}-d input is not supported")

    def wrap_reduced(self,
                     a: tp.ArrayLike,
                     name_or_index: tp.NameIndex = None,
                     columns: tp.Optional[tp.IndexLike] = None,
                     dtype: tp.Optional[tp.PandasDTypeLike] = None,
                     group_by: tp.GroupByLike = None,
                     time_units: bool = False) -> tp.MaybeSeriesFrame:
        """Wrap result of reduction.

        `name_or_index` can be the name of the resulting series if reducing to a scalar per column,
        or the index of the resulting series/dataframe if reducing to an array per column.
        `columns` can be set to override object's default columns.

        If `time_units` is set, calls `to_time_units`."""
        checks.assert_not_none(self.ndim)
        _self = self.resolve(group_by=group_by)

        if columns is None:
            columns = _self.columns
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)

        arr = np.asarray(a)
        if dtype is not None:
            try:
                arr = arr.astype(dtype)
            except Exception as e:
                warnings.warn(repr(e), stacklevel=2)
        if time_units:
            arr = _self.to_time_units(arr)
            dtype = None
        if arr.ndim == 0:
            # Scalar per Series/DataFrame
            if time_units:
                return pd.to_timedelta(arr.item())
            return arr.item()
        if arr.ndim == 1:
            if _self.ndim == 1:
                if arr.shape[0] == 1:
                    # Scalar per Series/DataFrame with one column
                    if time_units:
                        return pd.to_timedelta(arr[0])
                    return arr[0]
                # Array per Series
                sr_name = columns[0]
                if sr_name == 0:  # was arr Series before
                    sr_name = None
                return pd.Series(arr, index=name_or_index, name=sr_name, dtype=dtype)
            # Scalar per column in arr DataFrame
            return pd.Series(arr, index=columns, name=name_or_index, dtype=dtype)
        if arr.ndim == 2:
            if arr.shape[1] == 1 and _self.ndim == 1:
                # Array per Series
                sr_name = columns[0]
                if sr_name == 0:  # was arr Series before
                    sr_name = None
                return pd.Series(arr[:, 0], index=name_or_index, name=sr_name, dtype=dtype)
            # Array per column in arr DataFrame
            return pd.DataFrame(arr, index=name_or_index, columns=columns, dtype=dtype)
        raise ValueError(f"{arr.ndim}-d input is not supported")

    def dummy(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.SeriesFrame:
        """Create a dummy Series/DataFrame."""
        _self = self.resolve(group_by=group_by)
        return _self.wrap(np.empty(_self.shape), **kwargs)


WrappingT = tp.TypeVar("WrappingT", bound="Wrapping")


class Wrapping(Configured, PandasIndexer):
    """Class that uses `ArrayWrapper` globally."""

    def __init__(self, wrapper: ArrayWrapper, **kwargs) -> None:
        checks.assert_type(wrapper, ArrayWrapper)
        self._wrapper = wrapper

        Configured.__init__(self, wrapper=wrapper, **kwargs)
        PandasIndexer.__init__(self)

    def indexing_func(self: WrappingT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> WrappingT:
        """Perform indexing on `Wrapping`."""
        return self.copy(wrapper=self.wrapper.indexing_func(pd_indexing_func, **kwargs))

    @property
    def wrapper(self) -> ArrayWrapper:
        """Array wrapper."""
        return self._wrapper

    def regroup(self: WrappingT, group_by: tp.GroupByLike, **kwargs) -> WrappingT:
        """Regroup this object.

        Only creates a new instance if grouping has changed, otherwise returns itself.

        `**kwargs` will be passed to `ArrayWrapper.regroup`."""
        if self.wrapper.grouper.is_grouping_changed(group_by=group_by):
            self.wrapper.grouper.check_group_by(group_by=group_by)
            return self.copy(wrapper=self.wrapper.regroup(group_by, **kwargs))
        return self  # important for keeping cache

    def select_one(self: WrappingT, column: tp.Any = None, group_by: tp.GroupByLike = None, **kwargs) -> WrappingT:
        """Select one column/group."""
        _self = self.regroup(group_by, **kwargs)
        if column is not None:
            if _self.wrapper.grouper.is_grouped():
                if column not in _self.wrapper.get_columns():
                    raise KeyError(f"Group '{column}' not found")
            else:
                if column not in _self.wrapper.columns:
                    raise KeyError(f"Column '{column}' not found")
            return _self[column]
        if not _self.wrapper.grouper.is_grouped():
            if _self.wrapper.ndim == 1:
                return _self
            raise TypeError("Only one column is allowed. Use indexing or column argument.")
        if _self.wrapper.grouped_ndim == 1:
            return _self
        raise TypeError("Only one group is allowed. Use indexing or column argument.")

    @staticmethod
    def select_one_from_obj(obj: tp.SeriesFrame, wrapper: ArrayWrapper, column: tp.Any = None) -> tp.MaybeSeries:
        """Select one column/group from a pandas object."""
        if column is not None:
            if wrapper.grouper.is_grouped():
                if column not in wrapper.get_columns():
                    raise KeyError(f"Group '{column}' not found")
            else:
                if column not in wrapper.columns:
                    raise KeyError(f"Column '{column}' not found")
            return obj[column]
        if not wrapper.grouper.is_grouped():
            if wrapper.ndim == 1:
                return obj
            raise TypeError("Only one column is allowed. Use indexing or column argument.")
        if wrapper.grouped_ndim == 1:
            return obj
        raise TypeError("Only one group is allowed. Use indexing or column argument.")
