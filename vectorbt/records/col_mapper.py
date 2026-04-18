# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Class for mapping column arrays."""

from vectorbt import _typing as tp
from vectorbt.base.array_wrapper import ArrayWrapper, Wrapping
from vectorbt.base.reshape_fns import to_1d_array
from vectorbt.records import dispatch
from vectorbt.utils.decorators import cached_property, cached_method


class ColumnMapper(Wrapping):
    """Used by `vectorbt.records.base.Records` and `vectorbt.records.mapped_array.MappedArray`
    classes to make use of column and group metadata."""

    def __init__(self, wrapper: ArrayWrapper, col_arr: tp.Array1d, engine: tp.Optional[str] = None, **kwargs) -> None:
        Wrapping.__init__(self, wrapper, col_arr=col_arr, engine=engine, **kwargs)
        self._wrapper = wrapper
        self._col_arr = col_arr
        self._engine = engine

    @property
    def engine(self) -> tp.Optional[str]:
        """Engine preference for dispatch functions."""
        return self._engine

    def _col_idxs_meta(self, col_idxs: tp.Array1d, engine: tp.Optional[str] = None) -> tp.Tuple[tp.Array1d, tp.Array1d]:
        """Get metadata of column indices.

        Returns element indices and new column array.
        Automatically decides whether to use column range or column map."""
        engine = engine if engine is not None else self.engine
        if self.is_sorted(engine=engine):
            new_indices, new_col_arr = dispatch.col_range_select(
                self.col_range, to_1d_array(col_idxs), engine=engine
            )  # faster
        else:
            new_indices, new_col_arr = dispatch.col_map_select(self.col_map, to_1d_array(col_idxs), engine=engine)
        return new_indices, new_col_arr

    @property
    def wrapper(self) -> ArrayWrapper:
        """Array wrapper."""
        return self._wrapper

    @property
    def col_arr(self) -> tp.Array1d:
        """Column array."""
        return self._col_arr

    @cached_method
    def get_col_arr(self, group_by: tp.GroupByLike = None) -> tp.Array1d:
        """Get group-aware column array."""
        group_arr = self.wrapper.grouper.get_groups(group_by=group_by)
        if group_arr is not None:
            col_arr = group_arr[self.col_arr]
        else:
            col_arr = self.col_arr
        return col_arr

    @cached_property
    def col_range(self) -> tp.ColRange:
        """Column index.

        Faster than `ColumnMapper.col_map` but only compatible with sorted columns.
        More suited for records."""
        return dispatch.col_range(self.col_arr, len(self.wrapper.columns), engine=self.engine)

    @cached_method
    def get_col_range(self, group_by: tp.GroupByLike = None, engine: tp.Optional[str] = None) -> tp.ColRange:
        """Get group-aware column range."""
        engine = engine if engine is not None else self.engine
        if not self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.col_range
        col_arr = self.get_col_arr(group_by=group_by)
        columns = self.wrapper.get_columns(group_by=group_by)
        return dispatch.col_range(col_arr, len(columns), engine=engine)

    @cached_property
    def col_map(self) -> tp.ColMap:
        """Column map.

        More flexible than `ColumnMapper.col_range`.
        More suited for mapped arrays."""
        return dispatch.col_map(self.col_arr, len(self.wrapper.columns), engine=self.engine)

    @cached_method
    def get_col_map(self, group_by: tp.GroupByLike = None, engine: tp.Optional[str] = None) -> tp.ColMap:
        """Get group-aware column map."""
        engine = engine if engine is not None else self.engine
        if not self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.col_map
        col_arr = self.get_col_arr(group_by=group_by)
        columns = self.wrapper.get_columns(group_by=group_by)
        return dispatch.col_map(col_arr, len(columns), engine=engine)

    @cached_method
    def is_sorted(self, engine: tp.Optional[str] = None) -> bool:
        """Check whether column array is sorted."""
        engine = engine if engine is not None else self.engine
        return dispatch.is_col_sorted(self.col_arr, engine=engine)
