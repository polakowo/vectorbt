"""Numba-compiled 1-dim and 2-dim functions.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0).

    All functions passed as argument should be Numba-compiled.

    Records should retain the order they were created in."""

import numpy as np
from numba import njit


# ############# Indexing (records) ############# #

@njit(cache=True)
def record_col_index_nb(records, n_cols):
    """Index columns of `records`.

    Creates a 2-dim array with first column being start indices (inclusive) and
    second column being end indices (exclusive)."""
    # Record start and end indices for each column
    # Instead of doing np.flatnonzero and masking, this is much faster
    col_index = np.full((n_cols, 2), -1, dtype=np.int_)
    last_col = -1
    for r in range(records.shape[0]):
        col = records['col'][r]
        if col < last_col:
            raise ValueError("records must be sorted")
        if col != last_col:
            if last_col != -1:
                col_index[last_col, 1] = r
            col_index[col, 0] = r
            last_col = col
        if r == records.shape[0] - 1:
            col_index[col, 1] = r + 1
    return col_index


@njit(cache=True)
def select_record_cols_nb(records, col_index, new_cols):
    """Select columns of `records` given column indices `col_index`."""
    col_index = col_index[new_cols]
    new_n = np.sum(col_index[:, 1] - col_index[:, 0])
    out = np.empty(new_n, dtype=records.dtype)
    j = 0
    for c in range(new_cols.shape[0]):
        from_i = col_index[c, 0]
        to_i = col_index[c, 1]
        if from_i == -1 or to_i == -1:
            continue
        col_records = np.copy(records[from_i:to_i])
        col_records['col'][:] = c  # don't forget to assign new column indices
        out[j:j + col_records.shape[0]] = col_records
        j += col_records.shape[0]
    return out


# ############# Indexing (mapped arrays) ############# #


@njit(cache=True)
def mapped_col_index_nb(mapped_arr, col_arr, n_cols):
    """Identical to `record_col_index_nb`, but for mapped arrays."""
    col_index = np.full((n_cols, 2), -1, dtype=np.int_)
    last_col = -1
    for r in range(mapped_arr.shape[0]):
        col = col_arr[r]
        if col < last_col:
            raise ValueError("col_arr must be sorted")
        if col != last_col:
            if last_col != -1:
                col_index[last_col, 1] = r
            col_index[col, 0] = r
            last_col = col
        if r == mapped_arr.shape[0] - 1:
            col_index[col, 1] = r + 1
    return col_index


@njit(cache=True)
def select_mapped_cols_nb(col_arr, col_index, new_cols):
    """Return indices of elements corresponding to columns in `new_cols`.

    In contrast to `select_record_cols_nb`, returns new indices and new column array."""
    col_index = col_index[new_cols]
    new_n = np.sum(col_index[:, 1] - col_index[:, 0])
    mapped_arr_result = np.empty(new_n, dtype=np.int_)
    col_arr_result = np.empty(new_n, dtype=np.int_)
    j = 0
    for c in range(new_cols.shape[0]):
        from_i = col_index[c, 0]
        to_i = col_index[c, 1]
        if from_i == -1 or to_i == -1:
            continue
        rang = np.arange(from_i, to_i)
        mapped_arr_result[j:j + rang.shape[0]] = rang
        col_arr_result[j:j + rang.shape[0]] = c
        j += rang.shape[0]
    return mapped_arr_result, col_arr_result


# ############# Mapping (records) ############# #


@njit
def map_records_nb(records, map_func_nb, *args):
    """Map each record to a scalar value.

    `map_func_nb` should accept a single record and `*args`, and return a scalar value."""
    out = np.empty(records.shape[0], dtype=np.float_)
    for r in range(records.shape[0]):
        out[r] = map_func_nb(records[r], *args)
    return out


# ############# Mapping (mapped arrays) ############# #


@njit
def mapped_to_mask_nb(mapped_arr, col_arr, inout_map_func_nb, *args):
    """Map mapped array to a mask.

    Returns the same shape as `mapped_arr`.

    `inout_map_func_nb` should accept an empty boolean array of the column that should be written,
    index of the column, a mapped array of the column, and `*args`, and return nothing."""
    in_out = np.full(mapped_arr.shape[0], False, dtype=np.bool_)
    from_r = 0
    last_col = -1

    for r in range(mapped_arr.shape[0]):
        col = col_arr[r]
        if col < last_col:
            raise ValueError("col_arr must be sorted")
        if col != last_col:
            if last_col != -1:
                # At the beginning of second column do reduce on the first
                inout_map_func_nb(in_out[from_r:r], last_col, mapped_arr[from_r:r], *args)
            from_r = r
            last_col = col
        if r == len(mapped_arr) - 1:
            inout_map_func_nb(in_out[from_r:r + 1], col, mapped_arr[from_r:r + 1], *args)
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


# ############# Converting to matrix (mapped arrays) ############# #


@njit(cache=True)
def mapped_to_matrix_nb(mapped_arr, col_arr, idx_arr, target_shape, default_val):
    """Convert mapped array to the matrix form.

    !!! note
        Will raise an error if there are multiple values pointing to the same matrix element."""

    out = np.full(target_shape, default_val, dtype=np.float_)
    last_idx = -1
    last_col = -1
    for r in range(mapped_arr.shape[0]):
        cur_idx = idx_arr[r]
        cur_col = col_arr[r]
        if cur_col == last_col:
            if cur_idx == last_idx:
                raise ValueError("Multiple values are pointing to the same matrix element")
            if cur_idx < last_idx:
                raise ValueError("col_arr and idx_arr must be sorted")
        if cur_col < last_col:
            raise ValueError("col_arr and idx_arr must be sorted")
        out[cur_idx, cur_col] = mapped_arr[r]
        last_idx = cur_idx
        last_col = cur_col
    return out


# ############# Reducing (mapped arrays) ############# #

@njit
def reduce_mapped_nb(mapped_arr, col_arr, n_cols, default_val, reduce_func_nb, *args):
    """Reduce mapped array by column to a scalar value.

    Faster than `mapped_to_matrix_nb` and `vbt.*` used together, and also
    requires less memory. But does not take advantage of caching.

    `reduce_func_nb` should accept index of the column, mapped array and `*args`,
    and return a scalar value."""
    out = np.full(n_cols, default_val, dtype=np.float_)
    from_r = 0
    last_col = -1

    for r in range(mapped_arr.shape[0]):
        col = col_arr[r]
        if col < last_col:
            raise ValueError("col_arr must be sorted")
        if col != last_col:
            if last_col != -1:
                # At the beginning of second column do reduce on the first
                out[last_col] = reduce_func_nb(last_col, mapped_arr[from_r:r], *args)
            from_r = r
            last_col = col
        if r == len(mapped_arr) - 1:
            out[col] = reduce_func_nb(col, mapped_arr[from_r:r + 1], *args)
    return out


@njit
def reduce_mapped_to_idx_nb(mapped_arr, col_arr, idx_arr, n_cols, default_val, reduce_func_nb, *args):
    """Reduce mapped array by column to an index.

    Same as `reduce_mapped_nb` except `idx_arr` should be passed.

    !!! note
        Must return integers or raise an exception."""
    out = np.full(n_cols, default_val, dtype=np.float_)
    from_r = 0
    last_col = -1

    for r in range(mapped_arr.shape[0]):
        col = col_arr[r]
        if col < last_col:
            raise ValueError("col_arr must be sorted")
        if col != last_col:
            if last_col != -1:
                # At the beginning of second column do reduce on the first
                col_result = reduce_func_nb(last_col, mapped_arr[from_r:r], *args)
                out[last_col] = idx_arr[from_r:r][col_result]
            from_r = r
            last_col = col
        if r == len(mapped_arr) - 1:
            col_result = reduce_func_nb(col, mapped_arr[from_r:r + 1], *args)
            out[col] = idx_arr[from_r:r + 1][col_result]
    return out


@njit
def reduce_mapped_to_array_nb(mapped_arr, col_arr, n_cols, default_val, reduce_func_nb, *args):
    """Reduce mapped array by column to an array.

    `reduce_func_nb` same as for `reduce_mapped_nb` but should return an array."""
    col_out = reduce_func_nb(col_arr[0], mapped_arr[0:1], *args)
    out = np.full((col_out.shape[0], n_cols), default_val, dtype=np.float_)
    from_r = 0
    last_col = -1

    for r in range(mapped_arr.shape[0]):
        col = col_arr[r]
        if col < last_col:
            raise ValueError("col_arr must be sorted")
        if col != last_col:
            if last_col != -1:
                # At the beginning of second column do reduce on the first
                out[:, last_col] = reduce_func_nb(last_col, mapped_arr[from_r:r], *args)
            from_r = r
            last_col = col
        if r == len(mapped_arr) - 1:
            out[:, col] = reduce_func_nb(col, mapped_arr[from_r:r + 1], *args)
    return out


@njit
def reduce_mapped_to_idx_array_nb(mapped_arr, col_arr, idx_arr, n_cols, default_val, reduce_func_nb, *args):
    """Reduce mapped array by column to an index array.

    Same as `reduce_mapped_to_array_nb` except `idx_arr` should be passed.

    !!! note
        Must return integers or raise an exception."""
    col_out = reduce_func_nb(col_arr[0], mapped_arr[0:1], *args)
    out = np.full((col_out.shape[0], n_cols), default_val, dtype=np.float_)
    from_r = 0
    last_col = -1

    for r in range(mapped_arr.shape[0]):
        col = col_arr[r]
        if col < last_col:
            raise ValueError("col_arr must be sorted")
        if col != last_col:
            if last_col != -1:
                # At the beginning of second column do reduce on the first
                col_out = reduce_func_nb(last_col, mapped_arr[from_r:r], *args)
                out[:, last_col] = idx_arr[from_r:r][col_out]
            from_r = r
            last_col = col
        if r == len(mapped_arr) - 1:
            col_out = reduce_func_nb(col, mapped_arr[from_r:r + 1], *args)
            out[:, col] = idx_arr[from_r:r + 1][col_out]
    return out


@njit(cache=True)
def stack_mapped_nb(mapped_arr, col_arr, n_cols, col_index, default_val):
    """Stack mapped array."""
    col_max_n = np.max(col_index[:, 1] - col_index[:, 0])
    out = np.full((col_max_n, n_cols), default_val, dtype=np.float_)
    last_col = -1
    i = 0

    for r in range(mapped_arr.shape[0]):
        col = col_arr[r]
        if col < last_col:
            raise ValueError("col_arr must be sorted")
        if col != last_col:
            last_col = col
            i = 0
        out[i, col] = mapped_arr[r]
        i += 1
    return out


@njit(cache=True)
def mapped_value_counts_nb(mapped_codes, col_arr, n_cols):
    """Get value counts of an already factorized mapped array."""
    last_code = np.max(mapped_codes)
    out = np.full((last_code + 1, n_cols), 0, dtype=np.int_)
    last_col = -1

    for r in range(mapped_codes.shape[0]):
        col = col_arr[r]
        if col < last_col:
            raise ValueError("col_arr must be sorted")
        if col != last_col:
            last_col = col
        out[mapped_codes[r], col] += 1
    return out
