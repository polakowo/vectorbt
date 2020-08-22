"""Class for grouping index."""

import numpy as np
import pandas as pd
from numba import njit

from vectorbt.utils.decorators import cached_method
from vectorbt.utils.array import is_sorted
from vectorbt.utils import checks
from vectorbt.base.reshape_fns import to_1d
from vectorbt.base.index_fns import select_levels
from vectorbt.base.indexing import PandasIndexer


def group_by_to_index(index, group_by):
    """Convert mapper `group_by` to `pd.Index`.

    !!! note
        Index and mapper must have the same length."""
    if isinstance(group_by, (int, str, tuple, list)):
        group_by = select_levels(index, group_by)
    if not isinstance(group_by, pd.Index):
        group_by = pd.Index(group_by)
    if len(group_by) != len(index):
        raise ValueError("group_by and index must have the same length")
    return group_by


def get_groups_and_index(index, group_by):
    """Return array of group indices pointing to the original index, and grouped index.
    """
    if group_by is not None:
        group_by = group_by_to_index(index, group_by)
        groups, index = pd.factorize(group_by)
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        if isinstance(group_by, pd.MultiIndex):
            index.names = group_by.names
        elif isinstance(group_by, (pd.Index, pd.Series)):
            index.name = group_by.name
        if not is_sorted(groups):
            raise ValueError("Groups must be coherent and sorted")
        return groups, index
    return np.arange(len(index)), index


@njit(cache=True)
def get_group_counts_nb(groups):
    """Return count per group."""
    result = np.empty(groups.shape[0], dtype=np.int_)
    j = 0
    prev_group = -1
    run_count = 0
    for i in range(groups.shape[0]):
        cur_group = groups[i]
        if cur_group < prev_group:
            raise ValueError("Groups must be coherent and sorted")
        if cur_group != prev_group:
            if prev_group != -1:
                # Process previous group
                result[j] = run_count
                j += 1
                run_count = 0
            prev_group = cur_group
        run_count += 1
        if i == groups.shape[0] - 1:
            # Process last group
            result[j] = run_count
            j += 1
            run_count = 0
    return result[:j]


def _indexing_func(obj, pd_indexing_func):
    """Perform indexing on `IndexGrouper`."""
    range_sr = pd.Series(np.arange(len(obj.index)), index=obj.index)
    range_sr = pd_indexing_func(range_sr)
    idx_arr = to_1d(range_sr, raw=True)
    index = obj.index[idx_arr]
    new_group_by = obj.group_by[idx_arr]
    return obj.__class__(
        index,
        group_by=new_group_by,
        allow_change=obj.allow_change,
        allow_enable=obj.allow_enable,
        allow_disable=obj.allow_disable
    )


class IndexGrouper(PandasIndexer):
    """Class that exposes methods to group index.

    `group_by` can be integer (level by position), string (level by name), tuple or list
    (multiple levels), index or series (named index with groups), or NumPy array (raw groups).

    Set `allow_change` to `False` to prohibit changing groups (you can still change their labels).
    Set `allow_enable` to `False` to prohibit grouping if `IndexGrouper.group_by` is `None`.
    Set `allow_disable` to `False` to prohibit disabling of grouping if `IndexGrouper.group_by` is not `None`.

    !!! note
        Columns must build groups that are coherent and sorted."""

    def __init__(self, index, group_by=None, allow_change=True, allow_enable=True, allow_disable=True):
        self.index = index

        if group_by is False:
            # Disable completely
            group_by = None
        if group_by is not None:
            group_by = group_by_to_index(index, group_by)
        self.group_by = group_by

        # Everything is allowed by default
        self.allow_change = allow_change
        self.allow_enable = allow_enable
        self.allow_disable = allow_disable

        PandasIndexer.__init__(self, _indexing_func)

    def get_group_by(self, group_by=None, allow_change=None, allow_enable=None, allow_disable=None):
        """Get `group_by` from either object variable or keyword argument."""
        if allow_change is None:
            allow_change = self.allow_change
        if allow_enable is None:
            allow_enable = self.allow_enable
        if allow_disable is None:
            allow_disable = self.allow_disable
        if group_by is not None:
            if not allow_change:
                if self.group_by is not None and group_by is not False:
                    if not np.array_equal(self.get_groups(), get_groups_and_index(self.index, group_by)[0]):
                        raise ValueError("Changing groups is not allowed")
            if not allow_enable:
                if self.group_by is None and group_by is not False:
                    raise ValueError("Enabling grouping is not allowed")
            if not allow_disable:
                if self.group_by is not None and group_by is False:
                    raise ValueError("Disabling grouping is not allowed")
        if group_by is None:
            group_by = self.group_by
        if group_by is False:
            # Disable completely
            group_by = None
        if group_by is not None:
            group_by = group_by_to_index(self.index, group_by)
        return group_by

    @cached_method
    def get_groups_and_index(self, **kwargs):
        """See `get_groups_and_index`."""
        group_by = self.get_group_by(**kwargs)
        return get_groups_and_index(self.index, group_by)

    def get_groups(self, **kwargs):
        """Return groups array."""
        return self.get_groups_and_index(**kwargs)[0]

    def get_index(self, **kwargs):
        """Return grouped index."""
        return self.get_groups_and_index(**kwargs)[1]

    @cached_method
    def get_group_counts(self, **kwargs):
        """See get_group_counts_nb."""
        group_by = self.get_group_by(**kwargs)
        if group_by is None:
            # No grouping
            return np.full(len(self.index), 1)
        groups = self.get_groups(**kwargs)
        return get_group_counts_nb(groups)

    @cached_method
    def get_group_first_idxs(self, **kwargs):
        """Get first index of each group as an array."""
        group_by = self.get_group_by(**kwargs)
        if group_by is None:
            # No grouping
            return np.arange(len(self.index))
        group_counts = self.get_group_counts(**kwargs)
        return np.cumsum(group_counts) - group_counts
    
    @cached_method
    def get_group_last_idxs(self, **kwargs):
        """Get last index of each group as an array."""
        group_by = self.get_group_by(**kwargs)
        if group_by is None:
            # No grouping
            return np.arange(len(self.index))
        group_counts = self.get_group_counts(**kwargs)
        return np.cumsum(group_counts) - 1

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if not checks.is_equal(self.index, other.index, pd.Index.equals):
            return False
        if not checks.is_equal(self.group_by, other.group_by, pd.Index.equals):
            return False
        if self.allow_change != other.allow_change:
            return False
        if self.allow_enable != other.allow_enable:
            return False
        if self.allow_disable != other.allow_disable:
            return False
        return True
