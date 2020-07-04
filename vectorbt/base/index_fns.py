"""Functions for working with index/columns."""

import numpy as np
import pandas as pd
from collections.abc import Iterable

from vectorbt import defaults
from vectorbt.utils import checks


def get_index(arg, axis):
    """Get index of `arg` by `axis`."""
    checks.assert_type(arg, (pd.Series, pd.DataFrame))
    checks.assert_value_in(axis, (0, 1))

    if axis == 0:
        return arg.index
    else:
        if checks.is_series(arg):
            if arg.name is not None:
                return pd.Index([arg.name])
            return pd.Index([0])  # same as how pandas does it
        else:
            return arg.columns


def index_from_values(values, name=None):
    """Create a new `pd.Index` with `name` by parsing an iterable `values`.

    Each in `values` will correspond to an element in the new index."""
    checks.assert_type(values, Iterable)

    value_names = []
    for i, v in enumerate(values):
        v = np.asarray(v)
        if np.all(v == v.item(0)):
            value_names.append(v.item(0))
        else:
            value_names.append('mix_%d' % i)
    return pd.Index(value_names, name=name)


def repeat_index(index, n):
    """Repeat each element in `index` `n` times."""
    if not isinstance(index, pd.Index):
        index = pd.Index(index)
    if pd.Index.equals(index, pd.RangeIndex(start=0, stop=len(index), step=1)):  # ignore simple ranges without name
        return pd.RangeIndex(start=0, stop=n, step=1)
    return np.repeat(index, n)


def tile_index(index, n):
    """Tile the whole `index` `n` times."""
    if not isinstance(index, pd.Index):
        index = pd.Index(index)
    if pd.Index.equals(index, pd.RangeIndex(start=0, stop=len(index), step=1)):  # ignore simple ranges without name
        return pd.RangeIndex(start=0, stop=n, step=1)
    if isinstance(index, pd.MultiIndex):
        return pd.MultiIndex.from_tuples(np.tile(index, n), names=index.names)
    return pd.Index(np.tile(index, n), name=index.name)


def stack_indexes(*indexes):
    """Stack each index in `indexes` on top of each other, from top to bottom."""
    new_index = indexes[0]
    for i in range(1, len(indexes)):
        index1, index2 = new_index, indexes[i]
        checks.assert_same_shape(index1, index2)
        if not isinstance(index1, pd.MultiIndex):
            index1 = pd.MultiIndex.from_arrays([index1])
        if not isinstance(index2, pd.MultiIndex):
            index2 = pd.MultiIndex.from_arrays([index2])

        levels = []
        for i in range(index1.nlevels):
            levels.append(index1.get_level_values(i))
        for i in range(index2.nlevels):
            levels.append(index2.get_level_values(i))

        new_index = pd.MultiIndex.from_arrays(levels)
    return new_index


def combine_indexes(*indexes, ignore_single=None):
    """Combine each index in `indexes` using Cartesian product.

    If `ignore_single` is `True`, ignores indexes/columns with one value. If both are with one
    value, will keep both regardless of `ignore_single`.

    For defaults, see `vectorbt.defaults.broadcasting`."""
    if ignore_single is None:
        ignore_single = defaults.broadcasting['ignore_single']

    new_index = indexes[0]
    for i in range(1, len(indexes)):
        index1, index2 = new_index, indexes[i]
        if not isinstance(index1, pd.Index):
            index1 = pd.Index(index1)
        if not isinstance(index2, pd.Index):
            index2 = pd.Index(index2)

        if ignore_single:
            if len(index1) > 1 or len(index2) > 1:
                if len(index1) == 1:
                    new_index = index2
                    continue
                if len(index2) == 1:
                    new_index = index1
                    continue
        tuples1 = np.repeat(index1.to_numpy(), len(index2))
        tuples2 = np.tile(index2.to_numpy(), len(index1))

        if isinstance(index1, pd.MultiIndex):
            index1 = pd.MultiIndex.from_tuples(tuples1, names=index1.names)
        else:
            index1 = pd.Index(tuples1, name=index1.name)
        if isinstance(index2, pd.MultiIndex):
            index2 = pd.MultiIndex.from_tuples(tuples2, names=index2.names)
        else:
            index2 = pd.Index(tuples2, name=index2.name)

        new_index = stack_indexes(index1, index2)
    return new_index


def drop_levels(index, levels):
    """Softly drop `levels` in `index` by their name/position."""
    if not isinstance(index, pd.MultiIndex):
        return index

    levels_to_drop = []
    if not isinstance(levels, (tuple, list)):
        levels = [levels]
    for level in levels:
        if level in index.names:
            if level not in levels_to_drop:
                levels_to_drop.append(level)
        elif isinstance(level, int):
            if 0 <= level < index.nlevels or level == -1:
                if level not in levels_to_drop:
                    levels_to_drop.append(level)
    if len(levels_to_drop) < index.nlevels:
        # Drop only if there will be some indexes left
        return index.droplevel(levels_to_drop)
    return index


def rename_levels(index, name_dict):
    """Rename levels in `index` by `name_dict`."""
    for k, v in name_dict.items():
        if isinstance(index, pd.MultiIndex):
            if k in index.names:
                index = index.rename(v, level=k)
        else:
            if index.name == k:
                index.name = v
    return index


def select_levels(index, level_names):
    """Build a new index by selecting one or multiple `level_names` from `index`."""
    checks.assert_type(index, pd.MultiIndex)

    if isinstance(level_names, (list, tuple)):
        levels = [index.get_level_values(level_name) for level_name in level_names]
        return pd.MultiIndex.from_arrays(levels)
    else:
        return index.get_level_values(level_names)


def drop_redundant_levels(index):
    """Drop levels in `index` that either have a single value or a range from 0 to n."""
    if not isinstance(index, pd.MultiIndex):
        return index
    if len(index) == 1:
        return index

    levels_to_drop = []
    for i, level in enumerate(index.levels):
        if len(level) == 1:
            levels_to_drop.append(i)
        elif level.name is None and (level == np.arange(len(level))).all():  # basic range
            if len(index.get_level_values(i)) == len(level):
                levels_to_drop.append(i)
    # Remove redundant levels only if there are some non-redundant levels left
    if len(levels_to_drop) < index.nlevels:
        return index.droplevel(levels_to_drop)
    return index


def drop_duplicate_levels(index, keep='last'):
    """Drop levels in `index` with the same name and values.

    Set `keep` to 'last' to keep last levels, otherwise 'first'."""
    if not isinstance(index, pd.MultiIndex):
        return index

    levels = []
    levels_to_drop = []
    if keep == 'first':
        r = range(0, index.nlevels)
    elif keep == 'last':
        r = range(index.nlevels-1, -1, -1)  # loop backwards
    for i in r:
        level = (index.levels[i].name, tuple(index.get_level_values(i).to_numpy().tolist()))
        if level not in levels:
            levels.append(level)
        else:
            levels_to_drop.append(i)
    return index.droplevel(levels_to_drop)


def align_index_to(index1, index2):
    """Align `index1` to have the same shape of `index2`.

    Returns integer indices of occurrences and None if aligning not needed.

    The second one must contain all levels from the first (and can have some more). In all these levels, 
    both must share the same elements."""
    if not isinstance(index1, pd.MultiIndex):
        index1 = pd.MultiIndex.from_arrays([index1])
    if not isinstance(index2, pd.MultiIndex):
        index2 = pd.MultiIndex.from_arrays([index2])
    if index1.duplicated().any():
        raise Exception("Duplicates index values are not allowed for the first index")

    if pd.Index.equals(index1, index2):
        return pd.IndexSlice[:]
    if len(index1) <= len(index2):
        if len(index1) == 1:
            return pd.IndexSlice[np.tile([0])]
        js = []
        for i in range(index1.nlevels):
            for j in range(index2.nlevels):
                if index1.names[i] == index2.names[j]:
                    if np.array_equal(index1.levels[i], index2.levels[j]):
                        js.append(j)
                        break
        if index1.nlevels == len(js):
            new_index = pd.MultiIndex.from_arrays([index2.get_level_values(j) for j in js])
            xsorted = np.argsort(index1)
            ypos = np.searchsorted(index1[xsorted], new_index)
            return pd.IndexSlice[xsorted[ypos]]

    raise Exception("Indexes could not be aligned together")
