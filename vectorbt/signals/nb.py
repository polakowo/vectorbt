"""Numba-compiled 1-dim and 2-dim functions.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function. 
    Data is processed along index (axis 0).
    
    All functions passed as argument must be Numba-compiled."""

from numba import njit
import numpy as np

from vectorbt.base import combine_fns
from vectorbt.generic import nb as generic_nb


# ############# Signal generation ############# #


@njit
def generate_nb(shape, choice_func_nb, *args):
    """Create a boolean matrix of `shape` and pick signals using `choice_func_nb`.

    `choice_func_nb` must accept index of the current column `col`, index of the start
    of the range `from_i`, index of the end of the range `to_i`, and `*args`.
    It must return an array of indices from `[from_i, to_i)` (can be empty).

    !!! note
        Returned indices must be absolute.

    Example:
        ```python-repl
        >>> from numba import njit
        >>> import numpy as np
        >>> from vectorbt.signals.nb import generate_nb

        >>> @njit
        ... def choice_func_nb(col, from_i, to_i):
        ...     return np.array([from_i + col])

        >>> print(generate_nb((5, 3), choice_func_nb))
        [[ True False False]
         [False  True False]
         [False False  True]
         [False False False]
         [False False False]]
        ```"""
    result = np.full(shape, False, dtype=np.bool_)

    for col in range(result.shape[1]):
        idxs = choice_func_nb(col, 0, shape[0], *args)
        result[idxs, col] = True
    return result


@njit
def generate_after_nb(a, choice_func_nb, *args):
    """Pick signals using `choice_func_nb` after each signal in `a`.

    `choice_func_nb` is same as for `generate_nb`."""
    result = np.full_like(a, False)

    for col in range(a.shape[1]):
        a_idxs = np.flatnonzero(a[:, col])
        for i in range(a_idxs.shape[0]):
            # Calculate the range to choose from
            prev_idx = a_idxs[i]
            if i < a_idxs.shape[0] - 1:
                next_idx = a_idxs[i + 1]
            else:
                next_idx = a.shape[0]
            if prev_idx < a.shape[0] - 1:
                if next_idx - prev_idx > 1:
                    # Run the UDF
                    from_i = prev_idx + 1
                    to_i = next_idx
                    idxs = choice_func_nb(col, from_i, to_i, *args)
                    if np.any(idxs < from_i) or np.any(idxs >= to_i):
                        raise ValueError("Returned indices are out of bounds")
                    result[idxs, col] = True
    return result


@njit
def generate_iteratively_nb(shape, choice_func1_nb, choice_func2_nb, args_1, args_2):
    """Pick signals using `choice_func1_nb` and `choice_func2_nb` one after another.

    `choice_func1_nb` and `choice_func2_nb` are same as for `generate_nb`.
    `args_1` and `args_2` must be tuples that will be unpacked and passed to each function respectively.

    If any function returns multiple values, only the first value will be picked."""
    result1 = np.full(shape, False)
    result2 = np.full(shape, False)

    for col in range(shape[1]):
        prev_idx = -1
        i = 0
        while prev_idx < shape[0] - 1:
            from_i = prev_idx + 1
            to_i = shape[0]
            if i % 2 == 0:
                # Cannot assign two functions to a var in numba
                idxs = choice_func1_nb(col, from_i, to_i, *args_1)
                a = result1
            else:
                idxs = choice_func2_nb(col, from_i, to_i, *args_2)
                a = result2
            if len(idxs) == 0:
                break
            next_idx = idxs[0]
            if next_idx < from_i or next_idx >= to_i:
                raise ValueError("Returned index is out of bounds")
            a[next_idx, col] = True
            prev_idx = next_idx
            i += 1
    return result1, result2


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
    result = np.empty_like(a, dtype=np.bool_)

    for col in range(a.shape[1]):
        result[:, col] = np.random.permutation(a[:, col])
    return result


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
    result = np.empty(to_i - from_i, dtype=np.int_)
    j = 0
    for i in np.arange(from_i, to_i):
        if np.random.uniform(0, 1) <= probs[i, col]:
            result[j] = i
            j += 1
            if return_first:
                break
    return result[:j]


@njit
def generate_rand_by_prob_nb(shape, probs, seed=None):
    """Create a boolean matrix of `shape` and pick signals randomly by probabilities `probs`.

    `probs` must be a 2-dim array of shape `shape`.
    Specify seed to make output deterministic.

    See `rand_by_prob_choice_nb`."""
    if seed is not None:
        np.random.seed(seed)
    return generate_nb(shape, rand_by_prob_choice_nb, probs, False)


# ############# Exits ############# #

@njit
def generate_rand_ex_nb(entries, seed=None):
    """Pick an exit after each entry in `entries`.

    Specify seed to make output deterministic."""
    if seed is not None:
        np.random.seed(seed)
    return generate_after_nb(entries, rand_choice_nb, 1)


@njit
def generate_rand_ex_by_prob_nb(entries, probs, seed=None):
    """Pick an exit after each entry in `entries` by probabilities `probs`.

    `probs` must be a 2-dim array of shape `shape`.
    Specify seed to make output deterministic."""
    if seed is not None:
        np.random.seed(seed)
    return generate_after_nb(entries, rand_by_prob_choice_nb, probs, True)


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
    return generate_iteratively_nb(
        shape,
        rand_by_prob_choice_nb,
        rand_by_prob_choice_nb,
        (entry_probs, True),
        (exit_probs, True)
    )


@njit(cache=True)
def true_choice_nb(col, from_i, to_i, a):
    """`choice_func_nb` that returns the indices of signals in `a`."""
    return from_i + np.flatnonzero(a[from_i:to_i, col])


@njit(cache=True)
def sl_choice_nb(col, from_i, to_i, ts, stop, trailing, first):
    """`choice_func_nb` that returns the first index of `ts` being below the stop defined at `from_i-1`."""
    ts = ts[from_i - 1:to_i, col]
    stop = stop[from_i - 1:to_i, col]
    if trailing:
        # Propagate the maximum value from the entry using expanding max
        stop = (1 - stop) * generic_nb.expanding_max_1d_nb(ts)
        # Get the absolute index of the first ts being below that stop
        exits = from_i + np.flatnonzero(ts[1:] <= stop[1:])
    else:
        exits = from_i + np.flatnonzero(ts[1:] <= (1 - stop[0]) * ts[0])
    if first:
        return exits[:1]
    return exits


@njit(cache=True)
def tp_choice_nb(col, from_i, to_i, ts, stop, first):
    """`choice_func_nb` that returns the first index of `ts` being above the stop defined at `from_i-1`."""
    ts = ts[from_i - 1:to_i, col]
    stop = stop[from_i - 1:to_i, col]
    exits = from_i + np.flatnonzero(ts[1:] >= (1 + stop[0]) * ts[0])
    if first:
        return exits[:1]
    return exits


@njit
def sl_apply_nb(i, entries, ts, stops, trailing, first):
    """`apply_func_nb` for stop loss used in `vectorbt.base.combine_fns.apply_and_concat_one_nb`."""
    return generate_after_nb(entries, sl_choice_nb, ts, stops[i, :, :], trailing, first)


@njit
def sl_iter_apply_nb(i, entries, ts, stops, trailing, first):
    """`apply_func_nb` for iterative stop loss used in `vectorbt.base.combine_fns.apply_and_concat_one_nb`."""
    return generate_iteratively_nb(
        entries.shape,
        true_choice_nb,
        sl_choice_nb,
        (entries,), (ts, stops[i, :, :], trailing, first))


@njit
def tp_apply_nb(i, entries, ts, stops, first):
    """`apply_func_nb` for take profit used in `vectorbt.base.combine_fns.apply_and_concat_one_nb`."""
    return generate_after_nb(entries, tp_choice_nb, ts, stops[i, :, :], first)


@njit
def tp_iter_apply_nb(i, entries, ts, stops, first):
    """`apply_func_nb` for iterative take profit used in `vectorbt.base.combine_fns.apply_and_concat_one_nb`."""
    return generate_iteratively_nb(
        entries.shape,
        true_choice_nb,
        tp_choice_nb,
        (entries,), (ts, stops[i, :, :], first))


@njit
def generate_sl_ex_nb(entries, ts, stops, trailing=False, first=True):
    """Generate (trailing) stop loss exits using `generate_after_nb`.

    For each signal in `entries`, find the first value in `ts` that is below the (trailing) stop.

    Args:
        entries (array_like): 2-dim boolean array of entry signals.
        ts (array_like): 2-dim time series array such as price.
        stops (array_like): 3-dim array of stop values.

            !!! note
                `stops` must be a 3D array - an array out of 2-dim arrays each of `ts` shape.
                Each of these arrays will correspond to a different stop configuration.
        trailing (bool): If `True`, uses trailing stop, otherwise constant stop.
        first (bool): If `True`, selects the first signal, otherwise returns the whole sequence.

    Example:
        ```python-repl
        >>> import numpy as np
        >>> from vectorbt.signals.nb import generate_sl_ex_nb
        >>> from vectorbt.base.reshape_fns import broadcast_to_array_of

        >>> entries = np.asarray([False, True, False, False, False])[:, None]
        >>> ts = np.asarray([1, 2, 3, 2, 1])[:, None]
        >>> stops = broadcast_to_array_of([0.1, 0.5], ts)

        >>> print(generate_sl_ex_nb(entries, ts, stops,
        ...     trailing=True, first=True))
        [[False False]
         [False False]
         [False False]
         [ True False]
         [False  True]]
        ```"""
    return combine_fns.apply_and_concat_one_nb(
        len(stops), sl_apply_nb, entries, ts, stops, trailing, first)


@njit
def generate_sl_ex_iter_nb(entries, ts, stops, trailing=False):
    """Generate (trailing) stop loss exits iteratively using `generate_iteratively_nb`.

    Returns two arrays: new entries and exits.

    For arguments, see `generate_sl_ex_nb`."""
    return combine_fns.apply_and_concat_multiple_nb(
        len(stops), sl_iter_apply_nb, entries, ts, stops, trailing, True)


@njit
def generate_tp_ex_nb(entries, ts, stops, first=True):
    """Generate take profit exits using `generate_after_nb`.

    For arguments, see `generate_sl_ex_nb`."""
    return combine_fns.apply_and_concat_one_nb(
        len(stops), tp_apply_nb, entries, ts, stops, first)


@njit
def generate_tp_ex_iter_nb(entries, ts, stops):
    """Generate take profit exits iteratively using `generate_iteratively_nb`.

    For arguments, see `generate_sl_ex_nb`."""
    return combine_fns.apply_and_concat_multiple_nb(
        len(stops), tp_iter_apply_nb, entries, ts, stops, True)


# ############# Map and reduce ############# #


@njit
def map_reduce_between_nb(a, map_func_nb, reduce_func_nb, *args):
    """Map using `map_func_nb` and reduce using `reduce_func_nb` each consecutive
    pair of signals in `a`.

    Applies `map_func_nb` on each range `[from_i, to_i)`. Must accept index of the current column,
    index of the start of the range `from_i`, index of the end of the range `to_i`, and `*args`.

    Applies `reduce_func_nb` on all mapper results in a column. Must accept index of the
    current column, the array of results from `map_func_nb` for that column, and `*args`.

    !!! note
        Returned indices must be absolute.

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

        >>> print(map_reduce_between_nb(a, map_func_nb, reduce_func_nb))
        [1.5]
        ```"""
    result = np.full(a.shape[1], np.nan, dtype=np.float_)

    for col in range(a.shape[1]):
        a_idxs = np.flatnonzero(a[:, col])
        if a_idxs.shape[0] > 1:
            map_res = np.empty(a_idxs.shape[0])
            k = 0
            for j in range(1, a_idxs.shape[0]):
                from_i = a_idxs[j - 1]
                to_i = a_idxs[j]
                map_res[k] = map_func_nb(col, from_i, to_i, *args)
                k += 1
            if k > 0:
                result[col] = reduce_func_nb(col, map_res[:k], *args)
    return result


@njit
def map_reduce_between_two_nb(a, b, map_func_nb, reduce_func_nb, *args):
    """Map using `map_func_nb` and reduce using `reduce_func_nb` each consecutive
    pair of signals between `a` and `b`.

    Iterates over `b`, and for each found signal, looks for the preceding signal in `a`.

    `map_func_nb` and `reduce_func_nb` are same as for `map_reduce_between_nb`."""
    result = np.full((a.shape[1],), np.nan, dtype=np.float_)

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
                        map_res[k] = map_func_nb(col, from_i, to_i, *args)
                        k += 1
                if k > 0:
                    result[col] = reduce_func_nb(col, map_res[:k], *args)
    return result


@njit
def map_reduce_partitions_nb(a, map_func_nb, reduce_func_nb, *args):
    """Map using `map_func_nb` and reduce using `reduce_func_nb` each partition of signals in `a`.

    `map_func_nb` and `reduce_func_nb` are same as for `map_reduce_between_nb`."""
    result = np.full(a.shape[1], np.nan, dtype=np.float_)

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
                map_res[k] = map_func_nb(col, from_i, to_i, *args)
                k += 1
                is_partition = False
            if i == a.shape[0] - 1:
                if is_partition:
                    to_i = a.shape[0]
                    map_res[k] = map_func_nb(col, from_i, to_i, *args)
                    k += 1
        if k > 0:
            result[col] = reduce_func_nb(col, map_res[:k], *args)
    return result


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
    `reset_by` (must have the same shape). If `after_false` is `True`, the first partition must
    come after at least one `False` value. If `allow_gaps` is `True`, ignores gaps between partitions.

    Example:
        ```python-repl
        >>> import numpy as np
        >>> from vectorbt.signals.nb import rank_1d_nb

        >>> signals = np.asarray([True, True, False, True, True])
        >>> reset_by = np.asarray([False, True, False, False, True])

        >>> print(rank_1d_nb(signals))
        [1 2 0 1 2]
        >>> print(rank_1d_nb(signals, after_false=True))
        [0 0 0 1 2]
        >>> print(rank_1d_nb(signals, allow_gaps=True))
        [1 2 0 3 4]
        >>> print(rank_1d_nb(signals, allow_gaps=True, reset_by=reset_by))
        [1 1 0 2 1]
        ```"""
    result = np.zeros(a.shape, dtype=np.int_)

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
                result[i] = inc
        else:
            false_seen = True
            if not allow_gaps:
                inc = 0
    return result


@njit(cache=True)
def rank_nb(a, reset_by=None, after_false=False, allow_gaps=False):
    """2-dim version of `rank_1d_nb`."""
    result = np.zeros(a.shape, dtype=np.int_)

    for col in range(a.shape[1]):
        result[:, col] = rank_1d_nb(
            a[:, col],
            None if reset_by is None else reset_by[:, col],
            after_false=after_false,
            allow_gaps=allow_gaps
        )
    return result


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

        >>> print(rank_partitions_1d_nb(signals))
        [1 1 0 2 2]
        >>> print(rank_partitions_1d_nb(signals, after_false=True))
        [0 0 0 1 1]
        >>> print(rank_partitions_1d_nb(signals, reset_by=reset_by))
        [1 1 0 2 1]
        ```"""
    result = np.zeros(a.shape, dtype=np.int_)

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
                result[i] = inc
        else:
            false_seen = True
            first_seen = False
    return result


@njit(cache=True)
def rank_partitions_nb(a, reset_by=None, after_false=False):
    """2-dim version of `rank_partitions_1d_nb`."""
    result = np.zeros(a.shape, dtype=np.int_)

    for col in range(a.shape[1]):
        result[:, col] = rank_partitions_1d_nb(
            a[:, col],
            None if reset_by is None else reset_by[:, col],
            after_false=after_false
        )
    return result


# ############# Boolean operations ############# #

# Boolean operations are natively supported by pandas
# You can, for example, perform Signals_1 & Signals_2 to get logical AND of both arrays
# NOTE: We don't implement backward operations to avoid look-ahead bias!

@njit(cache=True)
def fshift_1d_nb(a, n):
    """Shift forward `a` by `n` positions."""
    result = np.empty_like(a, dtype=np.bool_)
    result[:n] = False
    result[n:] = a[:-n]
    return result


@njit(cache=True)
def fshift_nb(a, n):
    """2-dim version of `fshift_1d_nb`."""
    return fshift_1d_nb(a, n)
