"""Class for grouping index."""

import numpy as np
import pandas as pd
from numba import njit
from numba.typed import Dict

from vectorbt.utils.decorators import cached_method
from vectorbt.utils.array import is_sorted
from vectorbt.utils import checks
from vectorbt.base.reshape_fns import to_1d
from vectorbt.base.index_fns import select_levels
from vectorbt.base.indexing import PandasIndexer


def group_by_to_index(index, group_by):
    """Convert mapper to `pd.Index`.

    `group_by` can be integer (level by position), string (level by name), tuple or list
    (multiple levels), index or series (named index with groups), or NumPy array (raw groups).

    !!! note
        Index and mapper must have the same length."""
    if isinstance(group_by, (int, str, tuple, list)):
        group_by = select_levels(index, group_by)
    if not isinstance(group_by, pd.Index):
        group_by = pd.Index(group_by)
    checks.assert_same_len(index, group_by)
    return group_by


def group_index(index, group_by, return_dict=False, nb_compatible=False, assert_sorted=False):
    """Group index by some mapper.

    By default, returns an array of group indices pointing to the original index, and the new index.
    Set `return_dict` to `True` to return a dict instead of array.
    Set `nb_compatible` to `True` to make the dict Numba-compatible (Dict out of arrays).
    Set `assert_sorted` to `True` to verify that group indices are increasing.
    """
    group_by = group_by_to_index(index, group_by)
    group_arr, new_index = pd.factorize(group_by)
    if not isinstance(new_index, pd.Index):
        new_index = pd.Index(new_index)
    if isinstance(group_by, pd.MultiIndex):
        new_index.names = group_by.names
    elif isinstance(group_by, (pd.Index, pd.Series)):
        new_index.name = group_by.name
    if assert_sorted:
        if not is_sorted(group_arr):
            raise ValueError("Group indices are not increasing. Use .sort_values() on the index.")
    if return_dict:
        groups = dict()
        for i, idx in enumerate(group_arr):
            if idx not in groups:
                groups[idx] = []
            groups[idx].append(i)
        if nb_compatible:
            numba_groups = Dict()
            for k, v in groups.items():
                numba_groups[k] = np.array(v)
            return numba_groups, new_index
        return groups, new_index
    return group_arr, new_index


@njit
def count_per_group_nb(group_arr):
    """Return count per group."""
    result = np.empty(group_arr.shape[0], dtype=np.int_)
    j = 0
    prev_group = -1
    run_count = 0
    for i in range(group_arr.shape[0]):
        cur_group = group_arr[i]
        if cur_group < prev_group:
            raise ValueError("group_arr must be sorted")
        if cur_group != prev_group:
            if prev_group != -1:
                # Process previous group
                result[j] = run_count
                j += 1
                run_count = 0
            prev_group = cur_group
        run_count += 1
        if i == group_arr.shape[0] - 1:
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
    new_index = obj.index[idx_arr]
    new_group_by = obj.group_by[idx_arr]
    return obj.__class__(new_index, group_by=new_group_by)


class IndexGrouper(PandasIndexer):
    """Class that exposes methods to group index."""

    def __init__(self, index, group_by=None):
        self.index = index
        self.group_by = None
        self.group_by = self.resolve_group_by(group_by=group_by)

        PandasIndexer.__init__(self, _indexing_func)

    def resolve_group_by(self, group_by=None):
        """Resolve `group_by` from either object variable or keyword argument."""
        if group_by is None:
            group_by = self.group_by
        if isinstance(group_by, bool) and not group_by:
            # Disable completely
            group_by = None
        if group_by is not None:
            group_by = group_by_to_index(self.index, group_by)
        return group_by

    @cached_method
    def group_index(self, group_by=None, new_index=None, **kwargs):
        """See `group_index`."""
        group_by = self.resolve_group_by(group_by=group_by)
        if group_by is not None:
            group_arr, _new_index = group_index(self.index, group_by, **kwargs)
        else:
            group_arr = None
            _new_index = self.index
        if new_index is None:
            new_index = _new_index
        return group_arr, new_index

    def get_group_arr(self, **kwargs):
        return self.group_index(**kwargs)[0]

    def get_new_index(self, **kwargs):
        return self.group_index(**kwargs)[1]

    @cached_method
    def get_group_counts(self, **kwargs):
        group_arr, new_index = self.group_index(**kwargs)
        if group_arr is None:
            return np.full(len(new_index), 1)
        return count_per_group_nb(group_arr)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if not checks.is_equal(self.index, other.index, pd.Index.equals):
            return False
        if not checks.is_equal(self.group_by, other.group_by, pd.Index.equals):
            return False
        return True
