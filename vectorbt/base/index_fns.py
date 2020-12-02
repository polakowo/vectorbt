"""Functions for working with index/columns."""

import numpy as np
import pandas as pd
from numba import njit
from collections.abc import Iterable

from vectorbt.utils import checks


def get_index(arg, axis):
    """Get index of `arg` by `axis`."""
    checks.assert_type(arg, (pd.Series, pd.DataFrame))
    checks.assert_in(axis, (0, 1))

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
    if checks.is_default_index(index):  # ignore simple ranges without name
        return pd.RangeIndex(start=0, stop=len(index) * n, step=1)
    return np.repeat(index, n)


def tile_index(index, n):
    """Tile the whole `index` `n` times."""
    if not isinstance(index, pd.Index):
        index = pd.Index(index)
    if checks.is_default_index(index):  # ignore simple ranges without name
        return pd.RangeIndex(start=0, stop=len(index) * n, step=1)
    if isinstance(index, pd.MultiIndex):
        return pd.MultiIndex.from_tuples(np.tile(index, n), names=index.names)
    return pd.Index(np.tile(index, n), name=index.name)


def stack_indexes(*indexes, drop_duplicates=None, keep=None, drop_redundant=None):
    """Stack each index in `indexes` on top of each other, from top to bottom."""
    from vectorbt import settings

    if drop_duplicates is None:
        drop_duplicates = settings.broadcasting['drop_duplicates']
    if drop_redundant is None:
        drop_redundant = settings.broadcasting['drop_redundant']

    levels = []
    for i in range(len(indexes)):
        index = indexes[i]
        if not isinstance(index, pd.MultiIndex):
            if not isinstance(index, pd.Index):
                index = pd.Index(index)
            levels.append(index)
        else:
            for j in range(index.nlevels):
                levels.append(index.get_level_values(j))

    new_index = pd.MultiIndex.from_arrays(levels)
    if drop_duplicates:
        new_index = drop_duplicate_levels(new_index, keep=keep)
    if drop_redundant:
        new_index = drop_redundant_levels(new_index)
    return new_index


def combine_indexes(*indexes, **kwargs):
    """Combine each index in `indexes` using Cartesian product.

    Keyword arguments will be passed to `stack_indexes`."""
    new_index = indexes[0]
    for i in range(1, len(indexes)):
        index1, index2 = new_index, indexes[i]
        if not isinstance(index1, pd.Index):
            index1 = pd.Index(index1)
        if not isinstance(index2, pd.Index):
            index2 = pd.Index(index2)

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

        new_index = stack_indexes(index1, index2, **kwargs)
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
    return index.get_level_values(level_names)


def drop_redundant_levels(index):
    """Drop levels in `index` that either have a single unnamed value or a range from 0 to n."""
    if not isinstance(index, pd.MultiIndex):
        return index

    levels_to_drop = []
    for i in range(index.nlevels):
        if len(index) > 1 and len(index.levels[i]) == 1 and index.levels[i].name is None:
            levels_to_drop.append(i)
        elif checks.is_default_index(index.get_level_values(i)):
            levels_to_drop.append(i)
    # Remove redundant levels only if there are some non-redundant levels left
    if len(levels_to_drop) < index.nlevels:
        return index.droplevel(levels_to_drop)
    return index


def drop_duplicate_levels(index, keep=None):
    """Drop levels in `index` with the same name and values.

    Set `keep` to 'last' to keep last levels, otherwise 'first'."""
    from vectorbt import settings

    if keep is None:
        keep = settings.broadcasting['keep']
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


@njit
def _align_index_to_nb(a, b):
    """Return indices required to align `a` to `b`."""
    idxs = np.empty(b.shape[0], dtype=np.int_)
    g = 0
    for i in range(b.shape[0]):
        for j in range(a.shape[0]):
            if b[i] == a[j]:
                idxs[g] = j
                g += 1
                break
    return idxs


def align_index_to(index1, index2):
    """Align `index1` to have the same shape as `index2` if they have any levels in common.

    Returns index slice for the aligning."""
    if not isinstance(index1, pd.MultiIndex):
        index1 = pd.MultiIndex.from_arrays([index1])
    if not isinstance(index2, pd.MultiIndex):
        index2 = pd.MultiIndex.from_arrays([index2])
    if pd.Index.equals(index1, index2):
        return pd.IndexSlice[:]

    # Build map between levels in first and second index
    mapper = {}
    for i in range(index1.nlevels):
        for j in range(index2.nlevels):
            name1 = index1.names[i]
            name2 = index2.names[j]
            if name1 == name2:
                if set(index2.levels[j]).issubset(set(index1.levels[i])):
                    if i in mapper:
                        raise ValueError(f"There are multiple candidate levels with name {name1} in second index")
                    mapper[i] = j
                    continue
                if name1 is not None:
                    raise ValueError(f"Level {name1} in second index contains values not in first index")
    if len(mapper) == 0:
        raise ValueError("Can't find common levels to align both indexes")

    # Factorize first to be accepted by Numba
    factorized = []
    for k, v in mapper.items():
        factorized.append(pd.factorize(pd.concat((
            index1.get_level_values(k).to_series(),
            index2.get_level_values(v).to_series()
        )))[0])
    stacked = np.transpose(np.stack(factorized))
    indices1 = stacked[:len(index1)]
    indices2 = stacked[len(index1):]
    if len(np.unique(indices1, axis=0)) != len(indices1):
        raise ValueError("Duplicated values in first index are not allowed")

    # Try to tile
    if len(index2) % len(index1) == 0:
        tile_times = len(index2) // len(index1)
        index1_tiled = np.tile(indices1, (tile_times, 1))
        if np.array_equal(index1_tiled, indices2):
            return pd.IndexSlice[np.tile(np.arange(len(index1)), tile_times)]

    # Do element-wise comparison
    unique_indices = np.unique(stacked, axis=0, return_inverse=True)[1]
    unique1 = unique_indices[:len(index1)]
    unique2 = unique_indices[len(index1):]
    return pd.IndexSlice[_align_index_to_nb(unique1, unique2)]


def align_indexes(*indexes):
    """Align multiple indexes to each other."""
    max_len = max(map(len, indexes))
    indices = []
    for i in range(len(indexes)):
        index_i = indexes[i]
        if len(index_i) == max_len:
            indices.append(pd.IndexSlice[:])
        else:
            for j in range(len(indexes)):
                index_j = indexes[j]
                if len(index_j) == max_len:
                    try:
                        indices.append(align_index_to(index_i, index_j))
                        break
                    except ValueError:
                        pass
            if len(indices) < i + 1:
                raise ValueError(f"Index at position {i} could not be aligned")
    return indices


def pick_levels(index, required_levels=[], optional_levels=[]):
    """Pick optional and required levels and return their indices.

    Raises an exception if index has less or more levels than expected."""
    checks.assert_type(index, pd.MultiIndex)

    n_opt_set = len(list(filter(lambda x: x is not None, optional_levels)))
    n_req_set = len(list(filter(lambda x: x is not None, required_levels)))
    n_levels_left = index.nlevels - n_opt_set
    if n_req_set < len(required_levels):
        if n_levels_left != len(required_levels):
            n_expected = len(required_levels) + n_opt_set
            raise ValueError(f"Expected {n_expected} levels, found {index.nlevels}")

    levels_left = list(range(index.nlevels))

    # Pick optional levels
    _optional_levels = []
    for level in optional_levels:
        if level is not None:
            checks.assert_type(level, (int, str))
            if isinstance(level, str):
                level = index.names.index(level)
            if level < 0:
                level = index.nlevels + level
            levels_left.remove(level)
        _optional_levels.append(level)

    # Pick required levels
    _required_levels = []
    for level in required_levels:
        if level is not None:
            checks.assert_type(level, (int, str))
            if isinstance(level, str):
                level = index.names.index(level)
            if level < 0:
                level = index.nlevels + level
            levels_left.remove(level)
        _required_levels.append(level)
    for i, level in enumerate(_required_levels):
        if level is None:
            _required_levels[i] = levels_left.pop(0)

    return _required_levels, _optional_levels

