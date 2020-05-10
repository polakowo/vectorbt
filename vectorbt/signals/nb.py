"""Numba-compiled functions for signals.

!!! note
    `vectorbt` treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function. 
    Data is processed along index (axis 0)."""

from numba import njit, f8, i8, b1, optional
import numpy as np
from vectorbt import timeseries
from vectorbt.utils import combine_fns

# ############# Signal generation ############# #


@njit
def generate_nb(shape, choice_func_nb, *args):
    """Create a boolean matrix of `shape` and pick `True` values using `choice_func_nb`.

    `choice_func_nb` must accept index of the current column `col`, index of the start 
    of the range `from_i`, index of the end of the range `to_i`, and `*args`. 
    It must return an array of indices from `[from_i, to_i]` (can be empty).

    !!! note
        All indices must be absolute.

        `choice_func_nb` must be Numba-compiled.

    Example:
        ```python-repl
        >>> from numba import njit
        >>> from vectorbt.signals.nb import generate_nb

        >>> @njit
        ... def choice_func_nb(col, from_i, to_i):
        ...     return from_i + col

        >>> print(generate_nb((5, 3), choice_func_nb))
        [[ True False False]
         [False  True False]
         [False False  True]
         [False False False]
         [False False False]]
        ```"""
    result = np.full(shape, False, dtype=b1)

    for col in range(result.shape[1]):
        idxs = choice_func_nb(col, 0, shape[0]-1, *args)
        result[idxs, col] = True
    return result


@njit
def generate_after_nb(a, choice_func_nb, *args):
    """Pick `True` values using `choice_func_nb` after each `True` in `a`.

    `choice_func_nb` is same as for `generate_nb`."""
    result = np.full_like(a, False)

    for col in range(a.shape[1]):
        a_idxs = np.flatnonzero(a[:, col])
        for i in range(a_idxs.shape[0]):
            # Calculate the range to choose from
            prev_idx = a_idxs[i]
            if i < a_idxs.shape[0] - 1:
                next_idx = a_idxs[i+1]
            else:
                next_idx = a.shape[0]
            if prev_idx < a.shape[0] - 1:
                if next_idx - prev_idx > 1:
                    # Run the UDF
                    idxs = choice_func_nb(col, prev_idx+1, next_idx-1, *args)
                    result[idxs, col] = True
    return result


@njit
def generate_iteratively_nb(shape, choice_func1_nb, choice_func2_nb, *args):
    """Pick `True` values using `choice_func1_nb` and `choice_func2_nb` one after another.

    `choice_func1_nb` and `choice_func2_nb` are same as for `generate_nb`."""
    result1 = np.full(shape, False)
    result2 = np.full(shape, False)

    for col in range(shape[1]):
        prev_idx = -1
        i = 0
        while prev_idx < shape[0] - 1:
            if i % 2 == 0:
                # Cannot assign two functions to a var in numba
                idxs = choice_func1_nb(col, prev_idx+1, shape[0]-1, *args)
                a = result1
            else:
                idxs = choice_func2_nb(col, prev_idx+1, shape[0]-1, *args)
                a = result2
            if len(idxs) == 0:
                break
            a[idxs, col] = True
            prev_idx = np.flatnonzero(a[:, col])[-1]
            i += 1
    return result1, result2


# ############# Random ############# #


@njit(cache=True)
def random_prob_choice_1d_nb(a, prob_b):
    """Create a random sample from `a` with probabilities `prob_b`.

    `prob_b` must be of the same shape as `a`."""
    return a[np.searchsorted(np.cumsum(prob_b), np.random.random(), side="right")]


@njit(cache=True)
def shuffle_nb(a, seed=None):
    """Shuffle each column in `a`. 

    Specify seed to make output deterministic."""
    if seed is not None:
        np.random.seed(seed)
    result = np.empty_like(a, dtype=b1)

    for col in range(a.shape[1]):
        result[:, col] = np.random.permutation(a[:, col])
    return result


@njit(cache=True)
def random_choice_func_nb(col, from_i, to_i, n_range, n_prob, min_space):
    """`choice_func_nb` to randomly pick values from range `[from_i, to_i]`.

    The size of the sample will also be picked randomly from `n_range` with probabilities `n_prob`.
    Separate generated signals apart by `min_space` positions.

    `n_range` must be of same shape as `n_prob`."""
    from_range = np.arange(from_i, to_i+1)
    if min_space is not None:
        # Pick at every (min_space+1)-th position
        from_range = from_range[np.random.randint(0, min_space+1)::min_space+1]
    if n_prob is None:
        # Pick size from n_range
        size = np.random.choice(n_range)
    else:
        # Pick size from n_range with probabilities n_prob
        size = random_prob_choice_1d_nb(n_range, n_prob)
    # Sample should not be larger than population
    size = min(len(from_range), size)
    return np.random.choice(from_range, size=size, replace=False)


@njit
def generate_random_nb(shape, n_range, n_prob=None, min_space=None, seed=None):
    """Create a boolean matrix of `shape` and pick `True` values randomly.

    Specify seed to make output deterministic.
    See `random_choice_func_nb`."""
    if seed is not None:
        np.random.seed(seed)
    return generate_nb(shape, random_choice_func_nb, n_range, n_prob, min_space)


@njit
def generate_random_after_nb(a, n_range, n_prob=None, min_space=None, seed=None):
    """Pick `True` values randomly after each `True` in `a`.

    See `generate_random_nb`."""
    if seed is not None:
        np.random.seed(seed)
    return generate_after_nb(a, random_choice_func_nb, n_range, n_prob, min_space)

# ############# Stop-loss ############# #


@njit(cache=True)
def stop_loss_choice_func_nb(col, from_i, to_i, ts, stop, trailing):
    """Return index of the first `ts` value that is below `stop` defined at `from_i-1`."""
    # Set range starting from last entry until next entry
    ts = ts[from_i-1:to_i+1, col]
    stop = stop[from_i-1:to_i+1, col]
    if trailing:
        # Propagate the maximum value from the entry using expanding max
        peak_ts = timeseries.nb.expanding_max_1d_nb(ts)
        stop = (1 - stop) * peak_ts
        # Get the absolute index of the first value below that stop
        return from_i + np.flatnonzero(ts[1:] < stop[1:])[:1]
    else:
        # Calculate the stop
        stop_val = (1 - stop[0]) * ts[0]
        # Get the absolute index of the first value below that stop
        return from_i + np.flatnonzero(ts[1:] < stop_val)[:1]


@njit
def stop_loss_apply_func_nb(i, entries, ts, stops, trailing):
    """`apply_func_nb` for `vectorbt.utils.combine_fns.apply_and_concat_one_nb` for stop loss."""
    return generate_after_nb(entries, stop_loss_choice_func_nb, ts, stops[i, :, :], trailing)


@njit
def generate_stop_loss_nb(entries, ts, stops, trailing):
    """For each `True` in `entries`, find the first value in `ts` that is below the (trailing) stop.

    `stops` must be a 3D array - an array out of 2-dim arrays each of `ts` shape. Each of 
    these arrays will correspond to a different stop loss configuration. Set `trailing` to
    `True` to use trailing stop.
    
    Example:
        ```python-repl
        >>> import numpy as np
        >>> from numba import njit
        >>> from vectorbt.signals.nb import generate_stop_loss_nb
        >>> from vectorbt.utils.reshape_fns import broadcast_to_array_of

        >>> entries = np.asarray([False, True, False, False, False])[:, None]
        >>> ts = np.asarray([1, 2, 3, 2, 1])[:, None]
        >>> stops = broadcast_to_array_of([0.1, 0.5], ts)

        >>> print(generate_stop_loss_nb(entries, ts, stops, True))
        [[False False]
         [False False]
         [False False]
         [ True False]
         [False  True]]
        ```"""
    return combine_fns.apply_and_concat_one_nb(len(stops), stop_loss_apply_func_nb, entries, ts, stops, trailing)

# ############# Map and reduce ############# #


@njit
def map_reduce_between_nb(a, map_func_nb, reduce_func_nb, *args):
    """Map using `map_func_nb` and reduce using `reduce_func_nb` each consecutive 
    pair of `True` values in `a`.

    Applies `map_func_nb` on each pair `(prev_i, next_i)`. Must accept index of the current column, 
    index of previous `True`, index of next `True`, and `*args`. 

    Applies `reduce_func_nb` on all mapper results in a column. Must accept index of the 
    current column, the array of results from `map_func_nb` for that column, and `*args`.

    !!! note
        All indices must be absolute.

        `map_func_nb` and `reduce_func_nb` must be Numba-compiled.

    Example:
        ```python-repl
        >>> import numpy as np
        >>> from numba import njit
        >>> from vectorbt.signals.nb import map_reduce_between_nb

        >>> @njit
        ... def map_func_nb(col, prev_i, next_i):
        ...     return next_i - prev_i
        >>> @njit
        ... def reduce_func_nb(col, map_res):
        ...     return np.nanmean(map_res)
        >>> a = np.asarray([False, True, True, False, True])[:, None]

        >>> print(map_reduce_between_nb(a, map_func_nb, reduce_func_nb))
        [1.5]
        ```"""
    result = np.full(a.shape[1], np.nan, dtype=f8)

    for col in range(a.shape[1]):
        a_idxs = np.flatnonzero(a[:, col])
        if a_idxs.shape[0] > 1:
            map_res = np.empty(a_idxs.shape[0])
            k = 0
            for j in range(1, a_idxs.shape[0]):
                prev_i = a_idxs[j-1]
                next_i = a_idxs[j]
                map_res[k] = map_func_nb(col, prev_i, next_i, *args)
                k += 1
            if k > 0:
                result[col] = reduce_func_nb(col, map_res[:k], *args)
    return result


@njit
def map_reduce_between_two_nb(a, b, map_func_nb, reduce_func_nb, *args):
    """Map using `map_func_nb` and reduce using `reduce_func_nb` each consecutive 
    pair of `True` values in `a` and `b`.

    Applies `map_func_nb` on each pair `(prev_i, prev_i <= b < next_i)`.
    `map_func_nb` and `reduce_func_nb` are same as for `map_reduce_between_nb`."""
    result = np.full((a.shape[1],), np.nan, dtype=f8)

    for col in range(a.shape[1]):
        a_idxs = np.flatnonzero(a[:, col])
        if a_idxs.shape[0] > 0:
            b_idxs = np.flatnonzero(b[:, col])
            if b_idxs.shape[0] > 0:
                map_res = np.empty(b_idxs.shape)
                k = 0
                for j, b_i in enumerate(b_idxs):
                    valid_a_idxs = a_idxs[b_i >= a_idxs]
                    if len(valid_a_idxs) > 0:
                        a_i = valid_a_idxs[-1]  # last preceding a
                        map_res[k] = map_func_nb(col, a_i, b_i, *args)
                        k += 1
                if k > 0:
                    result[col] = reduce_func_nb(col, map_res[:k], *args)
    return result


@njit(cache=True)
def distance_map_nb(col, prev_i, next_i):
    """Distance mapper."""
    return next_i - prev_i


@njit(cache=True)
def mean_reduce_nb(col, a):
    """Average reducer."""
    return np.nanmean(a)

# ############# Ranking ############# #


@njit(cache=True)
def rank_nb(a, reset_by=None, after_false=False, allow_gaps=False):
    """Rank over each partition of `True` values.

    Partition is some number of `True` values in a row. You can reset partitions by `True` values 
    from `reset_by` (must have the same shape). If `after_false` is `True`, the first partition 
    must come after at least one `False`. If `allow_gaps` is `True`, ignores gaps between partitions."""
    result = np.zeros(a.shape, dtype=i8)

    for col in range(a.shape[1]):
        false_seen = ~after_false
        inc = 0
        for i in range(a.shape[0]):
            if reset_by is not None:
                if reset_by[i, col]:
                    # Signal in b_ref resets rank
                    false_seen = ~after_false
                    inc = 0
            if a[i, col] and false_seen:
                inc += 1
                result[i, col] = inc
            else:
                false_seen = True
                if not allow_gaps:
                    inc = 0
    return result

# ############# Boolean operations ############# #

# Boolean operations are natively supported by pandas
# You can, for example, perform Signals_1 & Signals_2 to get logical AND of both arrays
# NOTE: We don't implement backward operations to avoid look-ahead bias!


@njit(cache=True)
def fshift_nb(a, n):
    """Shift forward `a` by `n` positions."""
    result = np.empty_like(a, dtype=b1)
    result[:n, :] = False
    result[n:, :] = a[:-n, :]
    return result
