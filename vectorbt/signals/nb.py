"""Numba-compiled 1-dim and 2-dim functions.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function. 
    Data is processed along index (axis 0).
    
    All functions passed as argument should be Numba-compiled.

    Returned indices must be absolute."""

from numba import njit
import numpy as np

from vectorbt.base.reshape_fns import flex_choose_i_and_col_nb, flex_select_nb
from vectorbt.signals.enums import StopPosition


# ############# Signal generation ############# #


@njit
def generate_nb(shape, choice_func_nb, *args):
    """Create a boolean matrix of `shape` and pick signals using `choice_func_nb`.

    `choice_func_nb` must accept index of the current column `col`, index of the start
    of the range `from_i`, index of the end of the range `to_i`, and `*args`.
    It must return an array of indices from `[from_i, to_i)` (can be empty).

    Example:
        ```python-repl
        >>> from numba import njit
        >>> import numpy as np
        >>> from vectorbt.signals.nb import generate_nb

        >>> @njit
        ... def choice_func_nb(col, from_i, to_i):
        ...     return np.array([from_i + col])

        >>> generate_nb((5, 3), choice_func_nb)
        [[ True False False]
         [False  True False]
         [False False  True]
         [False False False]
         [False False False]]
        ```"""
    out = np.full(shape, False, dtype=np.bool_)

    for col in range(out.shape[1]):
        idxs = choice_func_nb(col, 0, shape[0], *args)
        out[idxs, col] = True
    return out


@njit
def generate_ex_nb(entries, exit_choice_func_nb, *args):
    """Pick exit signals using `exit_choice_func_nb` after each signal in `entries`.

    `exit_choice_func_nb` is same as for `generate_nb`."""
    exits = np.full_like(entries, False)

    for col in range(entries.shape[1]):
        entry_idxs = np.flatnonzero(entries[:, col])
        for i in range(entry_idxs.shape[0]):
            # Calculate the range to choose from
            prev_idx = entry_idxs[i]
            if i < entry_idxs.shape[0] - 1:
                next_idx = entry_idxs[i + 1]
            else:
                next_idx = entries.shape[0]
            if prev_idx < entries.shape[0] - 1:
                if next_idx - prev_idx > 1:
                    # Run the UDF
                    from_i = prev_idx + 1
                    to_i = next_idx
                    idxs = exit_choice_func_nb(col, from_i, to_i, *args)
                    if np.any(idxs < from_i) or np.any(idxs >= to_i):
                        raise ValueError("Returned indices are out of bounds")
                    exits[idxs, col] = True
    return exits


@njit
def generate_enex_nb(shape, entry_choice_func_nb, entry_args, exit_choice_func_nb, exit_args):
    """Pick entry signals using `entry_choice_func_nb` and exit signals using 
    `exit_choice_func_nb` iteratively.

    `entry_choice_func_nb` and `exit_choice_func_nb` are same as for `generate_nb`.
    `entry_args` and `exit_args` must be tuples that will be unpacked and passed to
    each function respectively.

    If any function returns multiple values, only the first value will be picked."""
    entries = np.full(shape, False)
    exits = np.full(shape, False)

    for col in range(shape[1]):
        prev_idx = -1
        i = 0
        while prev_idx < shape[0] - 1:
            from_i = prev_idx + 1
            to_i = shape[0]
            if i % 2 == 0:
                # Cannot assign two functions to a var in numba
                idxs = entry_choice_func_nb(col, from_i, to_i, *entry_args)
                a = entries
            else:
                idxs = exit_choice_func_nb(col, from_i, to_i, *exit_args)
                a = exits
            if len(idxs) == 0:
                break
            next_idx = idxs[0]
            if next_idx < from_i or next_idx >= to_i:
                raise ValueError("Returned index is out of bounds")
            a[next_idx, col] = True
            prev_idx = next_idx
            i += 1
    return entries, exits


# ############# Random ############# #


@njit(cache=True)
def shuffle_1d_nb(a, seed=None):
    """Shuffle each column in `a`.

    Specify seed to make output deterministic."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.permutation(a)


@njit(cache=True)
def shuffle_nb(a, seed=None):
    """2-dim version of `shuffle_1d_nb`."""
    if seed is not None:
        np.random.seed(seed)
    out = np.empty_like(a, dtype=np.bool_)

    for col in range(a.shape[1]):
        out[:, col] = np.random.permutation(a[:, col])
    return out


@njit(cache=True)
def rand_choice_nb(col, from_i, to_i, n):
    """`choice_func_nb` to randomly pick `n` values from range `[from_i, to_i)`."""
    return np.random.choice(np.arange(from_i, to_i), size=n, replace=False)


@njit
def generate_rand_nb(shape, n, seed=None):
    """Create a boolean matrix of `shape` and pick `n` signals randomly.

    Specify seed to make output deterministic.
    See `rand_choice_nb`."""
    if seed is not None:
        np.random.seed(seed)
    return generate_nb(shape, rand_choice_nb, n)


@njit(cache=True)
def rand_by_prob_choice_nb(col, from_i, to_i, probs, return_first):
    """`choice_func_nb` to randomly pick values from range `[from_i, to_i)` with probabilities `probs`.

    `probs` must be a 1-dim array."""
    out = np.empty(to_i - from_i, dtype=np.int_)
    j = 0
    for i in np.arange(from_i, to_i):
        if np.random.uniform(0, 1) <= probs[i, col]:
            out[j] = i
            j += 1
            if return_first:
                break
    return out[:j]


@njit
def generate_rand_by_prob_nb(shape, probs, seed=None):
    """Create a boolean matrix of `shape` and pick signals randomly by probabilities `probs`.

    `probs` must be a 2-dim array of shape `shape`.
    Specify seed to make output deterministic.

    See `rand_by_prob_choice_nb`."""
    if seed is not None:
        np.random.seed(seed)
    return generate_nb(shape, rand_by_prob_choice_nb, probs, False)


# ############# Random exits ############# #

@njit
def generate_rand_ex_nb(entries, seed=None):
    """Pick an exit after each entry in `entries`.

    Specify seed to make output deterministic."""
    if seed is not None:
        np.random.seed(seed)
    return generate_ex_nb(entries, rand_choice_nb, 1)


@njit
def generate_rand_ex_by_prob_nb(entries, probs, seed=None):
    """Pick an exit after each entry in `entries` by probabilities `probs`.

    `probs` must be a 2-dim array of shape `shape`.
    Specify seed to make output deterministic."""
    if seed is not None:
        np.random.seed(seed)
    return generate_ex_nb(entries, rand_by_prob_choice_nb, probs, True)


@njit
def generate_rand_enex_nb(shape, n, seed=None):
    """Pick `n` entries and the same number of exits one after another.

    Specify seed to make output deterministic."""
    entries = np.full(shape, False)
    exits = np.full(shape, False)
    both = generate_rand_nb(shape, n * 2, seed=seed)

    for col in range(both.shape[1]):
        both_idxs = np.flatnonzero(both[:, col])
        entries[both_idxs[0::2], col] = True
        exits[both_idxs[1::2], col] = True

    return entries, exits


@njit
def generate_rand_enex_by_prob_nb(shape, entry_probs, exit_probs, seed=None):
    """Pick entries by probabilities `entry_probs` and exits by probabilities `exit_probs` one after another.

    `entry_probs` and `exit_probs` must be 2-dim arrays of shape `shape`.
    Specify seed to make output deterministic."""
    if seed is not None:
        np.random.seed(seed)
    return generate_enex_nb(
        shape,
        rand_by_prob_choice_nb, (entry_probs, True),
        rand_by_prob_choice_nb, (exit_probs, True)
    )


# ############# Stop exits ############# #


@njit(cache=True)
def first_choice_nb(col, from_i, to_i, a):
    """`choice_func_nb` that returns the index of the first signal in `a`."""
    out = np.empty((1,), dtype=np.int_)
    for i in range(from_i, to_i):
        if a[i, col]:
            out[0] = i
            return out
    return out[:0]  # empty


@njit(cache=True)
def stop_choice_nb(col, from_i, to_i, ts, _stop, stop_pos, first, temp_int, is_2d):
    """`choice_func_nb` that returns the indices of `ts` that hit the stop."""
    stop = np.asarray(_stop)
    flex_i1, flex_col1 = flex_choose_i_and_col_nb(ts, is_2d)
    flex_i2, flex_col2 = flex_choose_i_and_col_nb(stop, is_2d)
    j = 0
    min_i = max_i = from_i - 1
    min_val = max_val = flex_select_nb(from_i - 1, col, ts, flex_i1, flex_col1, is_2d)

    for i in range(from_i, to_i):
        _ts = flex_select_nb(i, col, ts, flex_i1, flex_col1, is_2d)
        if _ts < min_val:  # keep track of min
            min_i = i
            min_val = _ts
        elif _ts > max_val:  # keep track of max
            max_i = i
            max_val = _ts
        if stop_pos == StopPosition.ExpMin:  # defined at min
            ts_val = flex_select_nb(min_i, col, ts, flex_i1, flex_col1, is_2d)
            stop_val = flex_select_nb(min_i, col, stop, flex_i2, flex_col2, is_2d)
        elif stop_pos == StopPosition.ExpMax:  # defined at max
            ts_val = flex_select_nb(max_i, col, ts, flex_i1, flex_col1, is_2d)
            stop_val = flex_select_nb(max_i, col, stop, flex_i2, flex_col2, is_2d)
        elif stop_pos == StopPosition.Entry:  # defined at entry
            ts_val = flex_select_nb(from_i - 1, col, ts, flex_i1, flex_col1, is_2d)
            stop_val = flex_select_nb(from_i - 1, col, stop, flex_i2, flex_col2, is_2d)
        else:
            raise ValueError("Comparison value not supported")

        # Is this exit signal?
        if stop_val > 0:
            exit_signal = _ts >= (1 + stop_val) * ts_val
        else:
            exit_signal = _ts <= (1 + stop_val) * ts_val
        if exit_signal:
            temp_int[j] = i
            j += 1
            if first:
                return temp_int[:1]
    return temp_int[:j]


@njit
def generate_stop_ex_nb(entries, ts, stop, stop_pos, first, is_2d):
    """Generate stop exits using `generate_ex_nb`.

    Args:
        entries (array_like): 2-dim boolean array of entry signals.
        ts (array_like): 2-dim time series array such as price.
        stop (float or array_like): One or more stop values (per row/column/element).
        stop_pos (StopPosition): See `vectorbt.signals.enums.StopPosition`.
        first (bool): If True, selects the first signal, otherwise returns the whole sequence.
        is_2d (bool): See `vectorbt.base.reshape_fns.flex_choose_i_and_col_nb`.

    !!! note
        `ts` and `stop` use flexible broadcasting.

    Example:
        Generate trailing stop loss and take profit signals for 10%.
        ```python-repl
        >>> import numpy as np
        >>> from vectorbt.signals.nb import generate_stop_ex_nb
        >>> from vectorbt.signals.enums import StopPosition

        >>> entries = np.asarray([False, True, False, False, False])[:, None]
        >>> ts = np.asarray([1, 2, 3, 2, 1])[:, None]

        >>> generate_stop_ex_nb(entries, ts, np.asarray(-0.1),
        ...     StopPosition.ExpMax, True, True)
        array([[False],
               [False],
               [False],
               [ True],
               [False]])

        >>> generate_stop_ex_nb(entries, ts, np.asarray(0.1),
        ...     StopPosition.Entry, True, True)
        array([[False],
               [False],
               [ True],
               [False],
               [False]])
        ```"""
    temp_int = np.empty((entries.shape[0],), dtype=np.int_)
    return generate_ex_nb(entries, stop_choice_nb, ts, stop, stop_pos, first, temp_int, is_2d)


@njit
def generate_stop_ex_iter_nb(entries, ts, stop, stop_pos, is_2d):
    """Generate stop loss exits iteratively using `generate_enex_nb`.

    Returns two arrays: new entries and exits.

    For arguments, see `generate_sl_ex_nb`."""
    temp_int = np.empty((entries.shape[0],), dtype=np.int_)
    return generate_enex_nb(
        entries.shape,
        first_choice_nb, (entries,),
        stop_choice_nb, (ts, stop, stop_pos, True, temp_int, is_2d)
    )


# ############# Map and reduce ############# #


@njit
def map_reduce_between_nb(a, map_func_nb, map_args, reduce_func_nb, reduce_args):
    """Map using `map_func_nb` and reduce using `reduce_func_nb` each consecutive
    pair of signals in `a`.

    Applies `map_func_nb` on each range `[from_i, to_i)`. Must accept index of the current column,
    index of the start of the range `from_i`, index of the end of the range `to_i`, and `*map_args`.

    Applies `reduce_func_nb` on all mapper results in a column. Must accept index of the
    current column, the array of results from `map_func_nb` for that column, and `*reduce_args`.

    Example:
        ```python-repl
        >>> import numpy as np
        >>> from numba import njit
        >>> from vectorbt.signals.nb import map_reduce_between_nb

        >>> @njit
        ... def map_func_nb(col, from_i, to_i):
        ...     return to_i - from_i
        >>> @njit
        ... def reduce_func_nb(col, map_res):
        ...     return np.nanmean(map_res)
        >>> a = np.asarray([False, True, True, False, True])[:, None]

        >>> map_reduce_between_nb(a, map_func_nb, (), reduce_func_nb, ())
        array([1.5])
        ```"""
    out = np.full(a.shape[1], np.nan, dtype=np.float_)

    for col in range(a.shape[1]):
        a_idxs = np.flatnonzero(a[:, col])
        if a_idxs.shape[0] > 1:
            map_res = np.empty(a_idxs.shape[0])
            k = 0
            for j in range(1, a_idxs.shape[0]):
                from_i = a_idxs[j - 1]
                to_i = a_idxs[j]
                map_res[k] = map_func_nb(col, from_i, to_i, *map_args)
                k += 1
            if k > 0:
                out[col] = reduce_func_nb(col, map_res[:k], *reduce_args)
    return out


@njit
def map_reduce_between_two_nb(a, b, map_func_nb, map_args, reduce_func_nb, reduce_args):
    """Map using `map_func_nb` and reduce using `reduce_func_nb` each consecutive
    pair of signals between `a` and `b`.

    Iterates over `b`, and for each found signal, looks for the preceding signal in `a`.

    `map_func_nb` and `reduce_func_nb` are same as for `map_reduce_between_nb`."""
    out = np.full((a.shape[1],), np.nan, dtype=np.float_)

    for col in range(a.shape[1]):
        a_idxs = np.flatnonzero(a[:, col])
        if a_idxs.shape[0] > 0:
            b_idxs = np.flatnonzero(b[:, col])
            if b_idxs.shape[0] > 0:
                map_res = np.empty(b_idxs.shape)
                k = 0
                for j, to_i in enumerate(b_idxs):
                    valid_a_idxs = a_idxs[a_idxs < to_i]
                    if len(valid_a_idxs) > 0:
                        from_i = valid_a_idxs[-1]  # preceding in a
                        map_res[k] = map_func_nb(col, from_i, to_i, *map_args)
                        k += 1
                if k > 0:
                    out[col] = reduce_func_nb(col, map_res[:k], *reduce_args)
    return out


@njit
def map_reduce_partitions_nb(a, map_func_nb, map_args, reduce_func_nb, reduce_args):
    """Map using `map_func_nb` and reduce using `reduce_func_nb` each partition of signals in `a`.

    `map_func_nb` and `reduce_func_nb` are same as for `map_reduce_between_nb`."""
    out = np.full(a.shape[1], np.nan, dtype=np.float_)

    for col in range(a.shape[1]):
        is_partition = False
        from_i = -1
        map_res = np.empty(a.shape[0])
        k = 0
        for i in range(a.shape[0]):
            if a[i, col]:
                if not is_partition:
                    from_i = i
                is_partition = True
            elif is_partition:
                to_i = i
                map_res[k] = map_func_nb(col, from_i, to_i, *map_args)
                k += 1
                is_partition = False
            if i == a.shape[0] - 1:
                if is_partition:
                    to_i = a.shape[0]
                    map_res[k] = map_func_nb(col, from_i, to_i, *map_args)
                    k += 1
        if k > 0:
            out[col] = reduce_func_nb(col, map_res[:k], *reduce_args)
    return out


@njit(cache=True)
def distance_map_nb(col, from_i, to_i):
    """Distance mapper."""
    return to_i - from_i


@njit(cache=True)
def mean_reduce_nb(col, a):
    """Average reducer."""
    return np.nanmean(a)


# ############# Ranking ############# #


@njit(cache=True)
def rank_1d_nb(a, reset_by=None, after_false=False, allow_gaps=False):
    """Rank signals in each partition.

    Partition is some number of signals in a row. You can reset partitions by signals from
    `reset_by` (must have the same shape). If `after_false` is True, the first partition must
    come after at least one False value. If `allow_gaps` is True, ignores gaps between partitions.

    Example:
        ```python-repl
        >>> import numpy as np
        >>> from vectorbt.signals.nb import rank_1d_nb

        >>> signals = np.asarray([True, True, False, True, True])
        >>> reset_by = np.asarray([False, True, False, False, True])

        >>> rank_1d_nb(signals)
        [1 2 0 1 2]
        >>> rank_1d_nb(signals, after_false=True)
        [0 0 0 1 2]
        >>> rank_1d_nb(signals, allow_gaps=True)
        [1 2 0 3 4]
        >>> rank_1d_nb(signals, allow_gaps=True, reset_by=reset_by)
        [1 1 0 2 1]
        ```"""
    out = np.zeros(a.shape, dtype=np.int_)

    false_seen = not after_false
    inc = 0
    for i in range(a.shape[0]):
        if reset_by is not None:
            if reset_by[i]:
                # Signal in b_ref resets rank
                false_seen = not after_false
                inc = 0
        if a[i]:
            if false_seen:
                inc += 1
                out[i] = inc
        else:
            false_seen = True
            if not allow_gaps:
                inc = 0
    return out


@njit(cache=True)
def rank_nb(a, reset_by=None, after_false=False, allow_gaps=False):
    """2-dim version of `rank_1d_nb`."""
    out = np.zeros(a.shape, dtype=np.int_)

    for col in range(a.shape[1]):
        out[:, col] = rank_1d_nb(
            a[:, col],
            None if reset_by is None else reset_by[:, col],
            after_false=after_false,
            allow_gaps=allow_gaps
        )
    return out


@njit(cache=True)
def rank_partitions_1d_nb(a, reset_by=None, after_false=False):
    """Rank partitions of signals.

    For keyword arguments, see `rank_nb`.

    Example:
        ```python-repl
        >>> import numpy as np
        >>> from vectorbt.signals.nb import rank_partitions_1d_nb

        >>> signals = np.asarray([True, True, False, True, True])
        >>> reset_by = np.asarray([False, True, False, False, True])

        >>> rank_partitions_1d_nb(signals)
        [1 1 0 2 2]
        >>> rank_partitions_1d_nb(signals, after_false=True)
        [0 0 0 1 1]
        >>> rank_partitions_1d_nb(signals, reset_by=reset_by)
        [1 1 0 2 1]
        ```"""
    out = np.zeros(a.shape, dtype=np.int_)

    false_seen = not after_false
    first_seen = False
    inc = 0
    for i in range(a.shape[0]):
        if reset_by is not None:
            if reset_by[i]:
                # Signal in b_ref resets rank
                false_seen = not after_false
                first_seen = False
                inc = 0
        if a[i]:
            if false_seen:
                if not first_seen:
                    inc += 1
                    first_seen = True
                out[i] = inc
        else:
            false_seen = True
            first_seen = False
    return out


@njit(cache=True)
def rank_partitions_nb(a, reset_by=None, after_false=False):
    """2-dim version of `rank_partitions_1d_nb`."""
    out = np.zeros(a.shape, dtype=np.int_)

    for col in range(a.shape[1]):
        out[:, col] = rank_partitions_1d_nb(
            a[:, col],
            None if reset_by is None else reset_by[:, col],
            after_false=after_false
        )
    return out


# ############# Boolean operations ############# #

# Boolean operations are natively supported by pandas
# You can, for example, perform Signals_1 & Signals_2 to get logical AND of both arrays
# NOTE: We don't implement backward operations to avoid look-ahead bias!

@njit(cache=True)
def fshift_1d_nb(a, n):
    """Shift forward `a` by `n` positions."""
    out = np.empty_like(a, dtype=np.bool_)
    out[:n] = False
    out[n:] = a[:-n]
    return out


@njit(cache=True)
def fshift_nb(a, n):
    """2-dim version of `fshift_1d_nb`."""
    return fshift_1d_nb(a, n)
