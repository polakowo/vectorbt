"""Numba-compiled 1-dim and 2-dim functions.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0).

    All functions passed as argument should be Numba-compiled.

    Records should retain the order they were created in."""

import numpy as np
from numba import njit

from vectorbt.utils.array import insert_argsort_nb


# ############# Indexing (general) ############# #


@njit(cache=True)
def col_range_nb(col_arr, n_cols):
    """Build column range for sorted column array.

    Creates a 2-dim array with first column being start indices (inclusive) and
    second column being end indices (exclusive)."""
    col_range = np.full((n_cols, 2), -1, dtype=np.int_)
    last_col = -1

    for r in range(col_arr.shape[0]):
        col = col_arr[r]
        if col < last_col:
            raise ValueError("col_arr must be sorted")
        if col != last_col:
            if last_col != -1:
                col_range[last_col, 1] = r
            col_range[col, 0] = r
            last_col = col
        if r == col_arr.shape[0] - 1:
            col_range[col, 1] = r + 1
    return col_range


@njit(cache=True)
def col_map_nb(col_arr, n_cols):
    """Build a map between columns and their indices.

    Returns an array with first axis being columns and second axis being indices,
    and an array with number of elements filled for each column; note that elements
    after this number are empty.

    Works well for unsorted column arrays."""
    col_idxs_out = np.empty((n_cols, len(col_arr)), dtype=np.int_)
    col_ns_out = np.full((n_cols,), 0, dtype=np.int_)

    for r in range(col_arr.shape[0]):
        col = col_arr[r]
        col_idxs_out[col, col_ns_out[col]] = r
        col_ns_out[col] += 1
    return col_idxs_out[:, :np.max(col_ns_out)], col_ns_out


@njit(cache=True)
def sort_col_map_nb(col_map, idx_arr):
    """Extends `col_map_nb` with sorting against `idx_arr` within each column."""
    col_idxs, col_ns = col_map
    col_idxs = np.copy(col_idxs)

    for col in range(col_idxs.shape[0]):
        n = col_ns[col]
        idxs = col_idxs[col][:n]
        insert_argsort_nb(idx_arr[idxs], idxs)
    return col_idxs, col_ns


# ############# Indexing (mapped arrays) ############# #

@njit(cache=True)
def mapped_col_range_select_nb(col_range, new_cols):
    """Perform indexing on a sorted array using column range `col_range`.

    Returns indices of elements corresponding to columns in `new_cols` and a new column array."""
    col_range = col_range[new_cols]
    new_n = np.sum(col_range[:, 1] - col_range[:, 0])
    indices_out = np.empty(new_n, dtype=np.int_)
    col_arr_out = np.empty(new_n, dtype=np.int_)
    j = 0

    for c in range(new_cols.shape[0]):
        from_r = col_range[c, 0]
        to_r = col_range[c, 1]
        if from_r == -1 or to_r == -1:
            continue
        rang = np.arange(from_r, to_r)
        indices_out[j:j + rang.shape[0]] = rang
        col_arr_out[j:j + rang.shape[0]] = c
        j += rang.shape[0]
    return indices_out, col_arr_out


@njit(cache=True)
def mapped_col_map_select_nb(col_map, new_cols):
    """Same as `mapped_col_range_select_nb` but using column map `col_map`."""
    col_idxs, col_ns = col_map
    total_count = np.sum(col_ns[new_cols])
    indices_out = np.empty((total_count,), dtype=np.int_)
    col_arr_out = np.empty((total_count,), dtype=np.int_)
    j = 0

    for new_col_i in range(len(new_cols)):
        new_col = new_cols[new_col_i]
        n = col_ns[new_col]
        idxs = col_idxs[new_col][:n]
        indices_out[j:j + len(idxs)] = idxs
        col_arr_out[j:j + len(idxs)] = new_col_i
        j += len(idxs)
    return indices_out, col_arr_out


# ############# Indexing (records) ############# #


@njit(cache=True)
def record_col_range_select_nb(records, col_range, new_cols):
    """Perform indexing on sorted records using column range `col_range`.

    Returns new records."""
    col_range = col_range[new_cols]
    new_n = np.sum(col_range[:, 1] - col_range[:, 0])
    out = np.empty(new_n, dtype=records.dtype)
    j = 0

    for c in range(new_cols.shape[0]):
        from_r = col_range[c, 0]
        to_r = col_range[c, 1]
        if from_r == -1 or to_r == -1:
            continue
        col_records = np.copy(records[from_r:to_r])
        col_records['col'][:] = c  # don't forget to assign new column indices
        out[j:j + col_records.shape[0]] = col_records
        j += col_records.shape[0]
    return out


@njit(cache=True)
def record_col_map_select_nb(records, col_map, new_cols):
    """Same as `record_col_range_select_nb` but using column map `col_map`."""
    col_idxs, col_ns = col_map
    out = np.empty((np.sum(col_ns[new_cols]),), dtype=records.dtype)
    j = 0

    for new_col_i in range(len(new_cols)):
        new_col = new_cols[new_col_i]
        col_n = col_ns[new_col]
        col_records = np.copy(records[col_idxs[new_col][:col_n]])
        col_records['col'][:] = new_col_i
        out[j:j + len(col_records)] = col_records
        j += len(col_records)
    return out


# ############# Sorting ############# #


@njit(cache=True)
def is_col_sorted_nb(col_arr):
    """Check whether the column array is sorted."""
    for i in range(len(col_arr) - 1):
        if col_arr[i + 1] < col_arr[i]:
            return False
    return True


@njit(cache=True)
def is_col_idx_sorted_nb(col_arr, idx_arr):
    """Check whether the column and index arrays are sorted."""
    for i in range(len(col_arr) - 1):
        if col_arr[i + 1] < col_arr[i]:
            return False
        if col_arr[i + 1] == col_arr[i] and idx_arr[i + 1] < idx_arr[i]:
            return False
    return True


# ############# Mapping (mapped arrays) ############# #


@njit
def mapped_to_mask_nb(mapped_arr, col_range, inout_map_func_nb, *args):
    """Map mapped array to a mask.

    Returns the same shape as `mapped_arr`.

    `inout_map_func_nb` should accept an empty boolean array of the column that should be written,
    index of the column, a mapped array of the column, and `*args`, and return nothing."""
    in_out = np.full(mapped_arr.shape[0], False, dtype=np.bool_)

    for col in range(col_range.shape[0]):
        from_r = col_range[col, 0]
        to_r = col_range[col, 1]
        if from_r == -1 or to_r == -1:
            continue
        inout_map_func_nb(in_out[from_r:to_r], col, mapped_arr[from_r:to_r], *args)
    return in_out


@njit(cache=True)
def top_n_inout_map_nb(in_out, col, mapped_arr, n):
    """`inout_map_func_nb` that returns mask of top N elements."""
    # TODO: np.argpartition
    top_n_idxs = np.argsort(mapped_arr)[-n:]
    in_out[top_n_idxs] = True


@njit(cache=True)
def bottom_n_inout_map_nb(in_out, col, mapped_arr, n):
    """`inout_map_func_nb` that returns mask of bottom N elements."""
    bottom_n_idxs = np.argsort(mapped_arr)[:n]
    in_out[bottom_n_idxs] = True


# ############# Mapping (records) ############# #


@njit
def map_records_nb(records, map_func_nb, *args):
    """Map each record to a scalar value.

    `map_func_nb` should accept a single record and `*args`, and return a scalar value."""
    out = np.empty(records.shape[0], dtype=np.float_)

    for r in range(records.shape[0]):
        out[r] = map_func_nb(records[r], *args)
    return out


# ############# Converting to matrix (mapped arrays) ############# #


@njit(cache=True)
def mapped_matrix_compatible_nb(col_arr, idx_arr, target_shape):
    """Check whether mapped array can be converted to a matrix without positional conflicts."""
    temp = np.zeros(target_shape)

    for i in range(len(col_arr)):
        if temp[idx_arr[i], col_arr[i]] > 0:
            return False
        temp[idx_arr[i], col_arr[i]] = 1
    return True


@njit(cache=True)
def mapped_to_matrix_nb(mapped_arr, col_arr, idx_arr, target_shape, default_val):
    """Convert mapped array to the matrix form."""
    out = np.full(target_shape, default_val, dtype=np.float_)

    for r in range(mapped_arr.shape[0]):
        out[idx_arr[r], col_arr[r]] = mapped_arr[r]
    return out


# ############# Reducing (mapped arrays) ############# #

@njit
def reduce_mapped_nb(mapped_arr, col_range, default_val, reduce_func_nb, *args):
    """Reduce mapped array by column to a scalar value.

    Faster than `mapped_to_matrix_nb` and `vbt.*` used together, and also
    requires less memory. But does not take advantage of caching.

    `reduce_func_nb` should accept index of the column, mapped array and `*args`,
    and return a scalar value."""
    out = np.full(col_range.shape[0], default_val, dtype=np.float_)

    for col in range(col_range.shape[0]):
        from_r = col_range[col, 0]
        to_r = col_range[col, 1]
        if from_r == -1 or to_r == -1:
            continue
        out[col] = reduce_func_nb(col, mapped_arr[from_r:to_r], *args)
    return out


@njit
def reduce_mapped_to_idx_nb(mapped_arr, col_range, idx_arr, default_val, reduce_func_nb, *args):
    """Reduce mapped array by column to an index.

    Same as `reduce_mapped_nb` except `idx_arr` should be passed.

    !!! note
        Must return integers or raise an exception."""
    out = np.full(col_range.shape[0], default_val, dtype=np.float_)

    for col in range(col_range.shape[0]):
        from_r = col_range[col, 0]
        to_r = col_range[col, 1]
        if from_r == -1 or to_r == -1:
            continue
        col_out = reduce_func_nb(col, mapped_arr[from_r:to_r], *args)
        out[col] = idx_arr[from_r:to_r][col_out]
    return out


@njit
def reduce_mapped_to_array_nb(mapped_arr, col_range, default_val, reduce_func_nb, *args):
    """Reduce mapped array by column to an array.

    `reduce_func_nb` same as for `reduce_mapped_nb` but should return an array."""
    col0, from_r0, to_r0 = -1, -1, -1
    for col in range(col_range.shape[0]):
        from_r = col_range[col, 0]
        to_r = col_range[col, 1]
        if from_r != -1 and to_r != -1:
            col0, from_r0, to_r0 = col, from_r, to_r
            break

    col_out = reduce_func_nb(col0, mapped_arr[from_r0:to_r0], *args)
    out = np.full((col_out.shape[0], col_range.shape[0]), default_val, dtype=np.float_)
    out[:, col0] = col_out

    for col in range(col0 + 1, col_range.shape[0]):
        from_r = col_range[col, 0]
        to_r = col_range[col, 1]
        if from_r == -1 or to_r == -1:
            continue
        out[:, col] = reduce_func_nb(col, mapped_arr[from_r:to_r], *args)
    return out


@njit
def reduce_mapped_to_idx_array_nb(mapped_arr, col_range, idx_arr, default_val, reduce_func_nb, *args):
    """Reduce mapped array by column to an index array.

    Same as `reduce_mapped_to_array_nb` except `idx_arr` should be passed.

    !!! note
        Must return integers or raise an exception."""
    col0, from_r0, to_r0 = -1, -1, -1
    for col in range(col_range.shape[0]):
        from_r = col_range[col, 0]
        to_r = col_range[col, 1]
        if from_r != -1 and to_r != -1:
            col0, from_r0, to_r0 = col, from_r, to_r
            break

    col_out = reduce_func_nb(col0, mapped_arr[from_r0:to_r0], *args)
    out = np.full((col_out.shape[0], col_range.shape[0]), default_val, dtype=np.float_)
    out[:, col0] = idx_arr[from_r0:to_r0][col_out]

    for col in range(col0 + 1, col_range.shape[0]):
        from_r = col_range[col, 0]
        to_r = col_range[col, 1]
        if from_r == -1 or to_r == -1:
            continue
        col_out = reduce_func_nb(col, mapped_arr[from_r:to_r], *args)
        out[:, col] = idx_arr[from_r:to_r][col_out]
    return out


@njit(cache=True)
def mapped_value_counts_nb(mapped_codes, col_range):
    """Get value counts of an already factorized mapped array."""
    last_code = np.max(mapped_codes)
    out = np.full((last_code + 1, col_range.shape[0]), 0, dtype=np.int_)

    for col in range(col_range.shape[0]):
        from_r = col_range[col, 0]
        to_r = col_range[col, 1]
        if from_r == -1 or to_r == -1:
            continue
        for r in range(from_r, to_r):
            out[mapped_codes[r], col] += 1
    return out


@njit(cache=True)
def stack_mapped_nb(mapped_arr, col_range, default_val):
    """Stack mapped array."""
    col_max_n = np.max(col_range[:, 1] - col_range[:, 0])
    out = np.full((col_max_n, col_range.shape[0]), default_val, dtype=np.float_)

    for col in range(col_range.shape[0]):
        from_r = col_range[col, 0]
        to_r = col_range[col, 1]
        if from_r == -1 or to_r == -1:
            continue
        arr = mapped_arr[from_r:to_r]
        out[:len(arr), col] = arr
    return out
