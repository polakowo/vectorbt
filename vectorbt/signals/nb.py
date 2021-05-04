"""Numba-compiled functions.

Provides an arsenal of Numba-compiled functions that are used by accessors
and in many other parts of the backtesting pipeline, such as technical indicators.
These only accept NumPy arrays and other Numba-compatible types.

```python-repl
>>> import numpy as np
>>> import vectorbt as vbt

>>> # vectorbt.signals.nb.rank_1d_nb
>>> vbt.signals.nb.rank_1d_nb(np.array([False, True, True, True, False]))
array([0, 1, 2, 3, 0])
```

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function. 
    Data is processed along index (axis 0).
    
    All functions passed as argument should be Numba-compiled.

    Returned indices should be absolute."""

from numba import njit
import numpy as np

from vectorbt import _typing as tp
from vectorbt.utils.array import uniform_summing_to_one_nb, rescale_float_to_int_nb
from vectorbt.base.reshape_fns import flex_select_auto_nb
from vectorbt.signals.enums import StopType


# ############# Generation ############# #


@njit
def generate_nb(shape: tp.Shape, choice_func_nb: tp.SignalChoiceFunc, *args) -> tp.Array2d:
    """Create a boolean matrix of `shape` and pick signals using `choice_func_nb`.

    `choice_func_nb` should accept index of the start of the range `from_i`,
    index of the end of the range `to_i`, index of the column `col`, and `*args`.
    It should return an array of indices from `[from_i, to_i)` (can be empty).

    ## Example

    ```python-repl
    >>> from numba import njit
    >>> import numpy as np
    >>> from vectorbt.signals.nb import generate_nb

    >>> @njit
    ... def choice_func_nb(from_i, to_i, col):
    ...     return np.array([from_i + col])

    >>> generate_nb((5, 3), choice_func_nb)
    [[ True False False]
     [False  True False]
     [False False  True]
     [False False False]
     [False False False]]
    ```
    """
    out = np.full(shape, False, dtype=np.bool_)

    for col in range(out.shape[1]):
        idxs = choice_func_nb(0, shape[0], col, *args)
        out[idxs, col] = True
    return out


@njit
def generate_ex_nb(entries: tp.Array2d, wait: int, exit_choice_func_nb: tp.SignalChoiceFunc, *args) -> tp.Array2d:
    """Pick exit signals using `exit_choice_func_nb` after each signal in `entries`.

    Set `wait` to a number of ticks to wait before placing exits.

    !!! note
        Setting `wait` to 0 or False may result in two signals at one tick.

    `exit_choice_func_nb` is same as for `generate_nb`."""
    exits = np.full_like(entries, False)

    for col in range(entries.shape[1]):
        entry_idxs = np.flatnonzero(entries[:, col])
        for i in range(entry_idxs.shape[0]):
            # Calculate the range to choose from
            from_i = entry_idxs[i] + wait
            if i < entry_idxs.shape[0] - 1:
                to_i = entry_idxs[i + 1]
            else:
                to_i = entries.shape[0]
            if to_i > from_i:
                # Run the UDF
                idxs = exit_choice_func_nb(from_i, to_i, col, *args)
                if np.any(idxs < from_i) or np.any(idxs >= to_i):
                    raise ValueError("Returned indices are out of bounds")
                exits[idxs, col] = True
    return exits


@njit
def generate_enex_nb(shape: tp.Shape,
                     entry_wait: int,
                     exit_wait: int,
                     entry_choice_func_nb: tp.SignalChoiceFunc,
                     entry_args: tp.Args,
                     exit_choice_func_nb: tp.SignalChoiceFunc,
                     exit_args: tp.Args) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Pick entry signals using `entry_choice_func_nb` and exit signals using 
    `exit_choice_func_nb` iteratively.

    Set `entry_wait`/`exit_wait` to a number of ticks to wait before placing entries/exits.

    !!! note
        Setting `entry_wait` or `exit_wait` to 0 or False may result in two signals at one tick.

    `entry_choice_func_nb` and `exit_choice_func_nb` are same as for `generate_nb`.
    `entry_args` and `exit_args` should be tuples that will be unpacked and passed to
    each function respectively.

    If any function returns multiple values, only the first value will be picked."""
    entries = np.full(shape, False)
    exits = np.full(shape, False)
    if entry_wait == 0 and exit_wait == 0:
        raise ValueError("entry_wait and exit_wait cannot be both 0")

    for col in range(shape[1]):
        prev_prev_i = -2
        prev_i = -1
        i = 0
        while True:
            to_i = shape[0]
            # Cannot assign two functions to a var in numba
            if i % 2 == 0:
                if i == 0:
                    from_i = 0
                else:
                    from_i = prev_i + entry_wait
                if from_i >= to_i:
                    break
                idxs = entry_choice_func_nb(from_i, to_i, col, *entry_args)
                a = entries
            else:
                from_i = prev_i + exit_wait
                if from_i >= to_i:
                    break
                idxs = exit_choice_func_nb(from_i, to_i, col, *exit_args)
                a = exits
            if len(idxs) == 0:
                break
            found_i = idxs[0]
            if found_i == prev_i == prev_prev_i:
                raise ValueError("Infinite loop detected")
            if found_i < from_i or found_i >= to_i:
                raise ValueError("Returned index is out of bounds")
            a[found_i, col] = True
            prev_prev_i = prev_i
            prev_i = found_i
            i += 1
    return entries, exits


# ############# Filtering ############# #


@njit(cache=True)
def clean_enex_1d_nb(entries: tp.Array1d,
                     exits: tp.Array1d,
                     entry_first: bool) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Clean entry and exit arrays by picking the first signal out of each.

    Entry signal must be picked first. If both signals are present, selects none."""
    entries_out = np.full(entries.shape, False, dtype=np.bool_)
    exits_out = np.full(exits.shape, False, dtype=np.bool_)

    phase = -1
    for i in range(entries.shape[0]):
        if entries[i] and exits[i]:
            continue
        if entries[i]:
            if phase == -1 or phase == 0:
                phase = 1
                entries_out[i] = True
        if exits[i]:
            if (not entry_first and phase == -1) or phase == 1:
                phase = 0
                exits_out[i] = True

    return entries_out, exits_out


@njit(cache=True)
def clean_enex_nb(entries: tp.Array2d,
                  exits: tp.Array2d,
                  entry_first: bool) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """2-dim version of `clean_enex_1d_nb`."""
    entries_out = np.empty(entries.shape, dtype=np.bool_)
    exits_out = np.empty(exits.shape, dtype=np.bool_)

    for col in range(entries.shape[1]):
        entries_out[:, col], exits_out[:, col] = clean_enex_1d_nb(entries[:, col], exits[:, col], entry_first)
    return entries_out, exits_out


# ############# Random ############# #


@njit(cache=True)
def rand_choice_nb(from_i: int, to_i: int, col: int, n: tp.MaybeArray[int]) -> tp.Array1d:
    """`choice_func_nb` to randomly pick `n` values from range `[from_i, to_i)`.

    `n` uses flexible indexing."""
    ns = np.asarray(n)
    return from_i + np.random.choice(to_i - from_i, size=flex_select_auto_nb(0, col, ns, True), replace=False)


@njit
def generate_rand_nb(shape: tp.Shape, n: tp.MaybeArray[int], seed: tp.Optional[int] = None) -> tp.Array2d:
    """Create a boolean matrix of `shape` and pick a number of signals randomly.

    Specify `seed` to make output deterministic.

    See `rand_choice_nb`."""
    if seed is not None:
        np.random.seed(seed)
    return generate_nb(shape, rand_choice_nb, n)


@njit(cache=True)
def rand_by_prob_choice_nb(from_i: int,
                           to_i: int,
                           col: int,
                           prob: tp.MaybeArray[float],
                           first: bool,
                           temp_idx_arr: tp.Array1d,
                           flex_2d: bool) -> tp.Array1d:
    """`choice_func_nb` to randomly pick values from range `[from_i, to_i)` with probability `prob`.

    `prob` uses flexible indexing."""
    probs = np.asarray(prob)
    j = 0
    for i in range(from_i, to_i):
        if np.random.uniform(0, 1) < flex_select_auto_nb(i, col, probs, flex_2d):  # [0, 1)
            temp_idx_arr[j] = i
            j += 1
            if first:
                break
    return temp_idx_arr[:j]


@njit
def generate_rand_by_prob_nb(shape: tp.Shape,
                             prob: tp.MaybeArray[float],
                             flex_2d: bool,
                             seed: tp.Optional[int] = None) -> tp.Array2d:
    """Create a boolean matrix of `shape` and pick signals randomly by probability `prob`.

    `prob` should be a 2-dim array of shape `shape`.
    Specify `seed` to make output deterministic.

    See `rand_by_prob_choice_nb`."""
    if seed is not None:
        np.random.seed(seed)
    temp_idx_arr = np.empty((shape[0],), dtype=np.int_)
    return generate_nb(shape, rand_by_prob_choice_nb, prob, False, temp_idx_arr, flex_2d)


# ############# Random exits ############# #

@njit
def generate_rand_ex_nb(entries: tp.Array2d, wait: int, seed: tp.Optional[int] = None) -> tp.Array2d:
    """Pick an exit after each entry in `entries`.

    Specify `seed` to make output deterministic."""
    if seed is not None:
        np.random.seed(seed)
    return generate_ex_nb(entries, wait, rand_choice_nb, np.full(entries.shape[1], 1))


@njit
def generate_rand_ex_by_prob_nb(entries: tp.Array2d,
                                prob: tp.MaybeArray[float],
                                wait: int,
                                flex_2d: bool,
                                seed: tp.Optional[int] = None) -> tp.Array2d:
    """Pick an exit after each entry in `entries` by probability `prob`.

    `prob` should be a 2-dim array of shape `shape`.
    Specify `seed` to make output deterministic."""
    if seed is not None:
        np.random.seed(seed)
    temp_idx_arr = np.empty((entries.shape[0],), dtype=np.int_)
    return generate_ex_nb(entries, wait, rand_by_prob_choice_nb, prob, True, temp_idx_arr, flex_2d)


@njit
def generate_rand_enex_nb(shape: tp.Shape,
                          n: tp.MaybeArray[int],
                          entry_wait: int,
                          exit_wait: int,
                          seed: tp.Optional[int] = None) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Pick a number of entries and the same number of exits one after another.

    Respects `entry_wait` and `exit_wait` constraints through a number of tricks.
    Tries to mimic a uniform distribution as much as possible.

    The idea is the following: with constraints, there is some fixed amount of total
    space required between first entry and last exit. Upscale this space in a way that
    distribution of entries and exit is similar to a uniform distribution. This means
    randomizing the position of first entry, last exit, and all signals between them.

    `n` uses flexible indexing.
    Specify `seed` to make output deterministic."""
    if seed is not None:
        np.random.seed(seed)
    entries = np.full(shape, False)
    exits = np.full(shape, False)
    if entry_wait == 0 and exit_wait == 0:
        raise ValueError("entry_wait and exit_wait cannot be both 0")
    ns = np.asarray(n)

    if entry_wait == 1 and exit_wait == 1:
        # Basic case
        both = generate_rand_nb(shape, ns * 2, seed=None)
        for col in range(both.shape[1]):
            both_idxs = np.flatnonzero(both[:, col])
            entries[both_idxs[0::2], col] = True
            exits[both_idxs[1::2], col] = True
    else:
        for col in range(shape[1]):
            _n = flex_select_auto_nb(0, col, ns, True)
            if _n == 1:
                entry_idx = np.random.randint(0, shape[0] - exit_wait)
                entries[entry_idx, col] = True
            else:
                # Minimum range between two entries
                min_range = entry_wait + exit_wait

                # Minimum total range between first and last entry
                min_total_range = min_range * (_n - 1)
                if shape[0] < min_total_range + exit_wait + 1:
                    raise ValueError("Cannot take a larger sample than population")

                # We should decide how much space should be allocate before first and after last entry
                # Maximum space outside of min_total_range
                max_free_space = shape[0] - min_total_range - 1

                # If min_total_range is tiny compared to max_free_space, limit it
                # otherwise we would have huge space before first and after last entry
                # Limit it such as distribution of entries mimics uniform
                free_space = min(max_free_space, 3 * shape[0] // (_n + 1))

                # What about last exit? it requires exit_wait space
                free_space -= exit_wait

                # Now we need to distribute free space among three ranges:
                # 1) before first, 2) between first and last added to min_total_range, 3) after last
                # We do 2) such that min_total_range can freely expand to maximum
                # We allocate twice as much for 3) as for 1) because an exit is missing
                rand_floats = uniform_summing_to_one_nb(6)
                chosen_spaces = rescale_float_to_int_nb(rand_floats, (0, free_space), free_space)
                first_idx = chosen_spaces[0]
                last_idx = shape[0] - np.sum(chosen_spaces[-2:]) - exit_wait - 1

                # Selected range between first and last entry
                total_range = last_idx - first_idx

                # Maximum range between two entries within total_range
                max_range = total_range - (_n - 2) * min_range

                # Select random ranges within total_range
                rand_floats = uniform_summing_to_one_nb(_n - 1)
                chosen_ranges = rescale_float_to_int_nb(rand_floats, (min_range, max_range), total_range)

                # Translate them into entries
                entry_idxs = np.empty(_n, dtype=np.int_)
                entry_idxs[0] = first_idx
                entry_idxs[1:] = chosen_ranges
                entry_idxs = np.cumsum(entry_idxs)
                entries[entry_idxs, col] = True

        # Generate exits
        for col in range(shape[1]):
            entry_idxs = np.flatnonzero(entries[:, col])
            for j in range(len(entry_idxs)):
                entry_i = entry_idxs[j] + exit_wait
                if j < len(entry_idxs) - 1:
                    exit_i = entry_idxs[j + 1] - entry_wait
                else:
                    exit_i = entries.shape[0] - 1
                i = np.random.randint(exit_i - entry_i + 1)
                exits[entry_i + i, col] = True
    return entries, exits


def rand_enex_apply_nb(input_shape: tp.Shape,
                       n: tp.MaybeArray[int],
                       entry_wait: int,
                       exit_wait: int) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """`apply_func_nb` that calls `generate_rand_enex_nb`."""
    return generate_rand_enex_nb(input_shape, n, entry_wait, exit_wait)


@njit
def generate_rand_enex_by_prob_nb(shape: tp.Shape,
                                  entry_prob: tp.MaybeArray[float],
                                  exit_prob: tp.MaybeArray[float],
                                  entry_wait: int,
                                  exit_wait: int,
                                  flex_2d: bool,
                                  seed: tp.Optional[int] = None) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Pick entries by probability `entry_prob` and exits by probability `exit_prob` one after another.

    `entry_prob` and `exit_prob` should be 2-dim arrays of shape `shape`.
    Specify `seed` to make output deterministic."""
    if seed is not None:
        np.random.seed(seed)
    temp_idx_arr = np.empty((shape[0],), dtype=np.int_)
    return generate_enex_nb(
        shape,
        entry_wait, exit_wait,
        rand_by_prob_choice_nb, (entry_prob, True, temp_idx_arr, flex_2d),
        rand_by_prob_choice_nb, (exit_prob, True, temp_idx_arr, flex_2d)
    )


# ############# Stop exits ############# #


@njit(cache=True)
def first_choice_nb(from_i: int, to_i: int, col: int, a: tp.Array2d) -> tp.Array1d:
    """`choice_func_nb` that returns the index of the first signal in `a`."""
    out = np.empty((1,), dtype=np.int_)
    for i in range(from_i, to_i):
        if a[i, col]:
            out[0] = i
            return out
    return out[:0]  # empty


@njit(cache=True)
def stop_choice_nb(from_i: int,
                   to_i: int,
                   col: int,
                   ts: tp.Array,
                   stop: tp.MaybeArray[float],
                   trailing: tp.MaybeArray[bool],
                   wait: int,
                   first: bool,
                   temp_idx_arr: tp.Array1d,
                   flex_2d: bool) -> tp.Array1d:
    """`choice_func_nb` that returns the indices of the stop being hit.

    Args:
        from_i (int): Index to start generation from (inclusive).
        to_i (int): Index to run generation to (exclusive).
        col (int): Current column.
        ts (array of float): 2-dim time series array such as price.
        stop (float or array_like): Stop value for stop loss.

            Can be per frame, column, row, or element-wise. Set to 0. to disable.
        trailing (bool or array_like): Whether to use trailing stop.

            Can be per frame, column, row, or element-wise.
        wait (int): Number of ticks to wait before placing exits.

            Setting False or 0 may result in two signals at one tick.
        first (bool): Whether to stop as soon as the first exit signal is found.
        temp_idx_arr (array of int): Empty integer array used to temporarily store indices.
        flex_2d (bool): See `vectorbt.base.reshape_fns.flex_choose_i_and_col_nb`."""
    stops = np.asarray(stop)
    trailings = np.asarray(trailing)

    j = 0
    min_i = max_i = init_i = from_i - wait
    init_ts = flex_select_auto_nb(init_i, col, ts, flex_2d)
    init_stop = flex_select_auto_nb(init_i, col, stops, flex_2d)
    init_trailing = flex_select_auto_nb(init_i, col, trailings, flex_2d)
    max_high = min_low = init_ts

    for i in range(from_i, to_i):
        if init_trailing:
            if init_stop > 0:
                # Trailing stop buy
                last_stop = flex_select_auto_nb(min_i, col, stops, flex_2d)
                curr_stop_price = min_low * (1 + abs(last_stop))
            elif init_stop < 0:
                # Trailing stop sell
                last_stop = flex_select_auto_nb(max_i, col, stops, flex_2d)
                curr_stop_price = max_high * (1 - abs(last_stop))
        else:
            curr_stop_price = init_ts * (1 + init_stop)

        # Check if stop price is within bar
        curr_ts = flex_select_auto_nb(i, col, ts, flex_2d)
        exit_signal = False
        if init_stop > 0:
            exit_signal = curr_ts >= curr_stop_price
        elif init_stop < 0:
            exit_signal = curr_ts <= curr_stop_price
        if exit_signal:
            temp_idx_arr[j] = i
            j += 1
            if first:
                return temp_idx_arr[:1]

        # Keep track of lowest low and highest high if trailing
        if init_trailing:
            if curr_ts < min_low:
                min_i = i
                min_low = curr_ts
            elif curr_ts > max_high:
                max_i = i
                max_high = curr_ts
    return temp_idx_arr[:j]


@njit
def generate_stop_ex_nb(entries: tp.Array2d,
                        ts: tp.Array,
                        stop: tp.MaybeArray[float],
                        trailing: tp.MaybeArray[bool],
                        wait: int,
                        first: bool,
                        flex_2d: bool) -> tp.Array2d:
    """Generate using `generate_ex_nb` and `stop_choice_nb`.

    ## Example

    Generate trailing stop loss and take profit signals for 10%.
    ```python-repl
    >>> import numpy as np
    >>> from vectorbt.signals.nb import generate_stop_ex_nb

    >>> entries = np.asarray([False, True, False, False, False])[:, None]
    >>> ts = np.asarray([1, 2, 3, 2, 1])[:, None]

    >>> generate_stop_ex_nb(entries, ts, -0.1, True, 1, True, True)
    array([[False],
           [False],
           [False],
           [ True],
           [False]])

    >>> generate_stop_ex_nb(entries, ts, 0.1, False, 1, True, True)
    array([[False],
           [False],
           [ True],
           [False],
           [False]])
    ```
    """
    temp_idx_arr = np.empty((entries.shape[0],), dtype=np.int_)
    return generate_ex_nb(entries, wait, stop_choice_nb, ts, stop, trailing, wait, first, temp_idx_arr, flex_2d)


@njit
def generate_stop_ex_iter_nb(entries: tp.Array2d,
                             ts: tp.Array,
                             stop: tp.MaybeArray[float],
                             trailing: tp.MaybeArray[bool],
                             entry_wait: int,
                             exit_wait: int,
                             flex_2d: bool) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Generate iteratively using `generate_enex_nb` and `stop_choice_nb`.

    Returns two arrays: new entries and exits."""
    temp_idx_arr = np.empty((entries.shape[0],), dtype=np.int_)
    return generate_enex_nb(
        entries.shape,
        entry_wait, exit_wait,
        first_choice_nb, (entries,),
        stop_choice_nb, (ts, stop, trailing, exit_wait, True, temp_idx_arr, flex_2d)
    )


@njit(cache=True)
def ohlc_stop_choice_nb(from_i: int,
                        to_i: int,
                        col: int,
                        open: tp.Array,
                        high: tp.Array,
                        low: tp.Array,
                        close: tp.Array,
                        hit_price_out: tp.Array2d,
                        stop_type_out: tp.Array2d,
                        sl_stop: tp.MaybeArray[float],
                        ts_stop: tp.MaybeArray[float],
                        tp_stop: tp.MaybeArray[float],
                        is_open_safe: bool,
                        wait: int,
                        first: bool,
                        temp_idx_arr: tp.Array1d,
                        flex_2d: bool) -> tp.Array1d:
    """`choice_func_nb` that returns the indices of the stop price being hit within OHLC.

    Compared to `stop_choice_nb`, takes into account the whole bar, can check for both
    (trailing) stop loss and take profit simultaneously, and tracks hit price and stop type.

    !!! note
        We don't have intra-candle data. If there was a huge price fluctuation in both directions,
        we can't determine whether SL was triggered before TP and vice versa. So some assumptions
        need to be made: 1) trailing stop can only be based on previous close/high, and
        2) we pessimistically assume that SL comes before TS and TP.
    
    Args:
        col (int): Current column.
        from_i (int): Index to start generation from (inclusive).
        to_i (int): Index to run generation to (exclusive).
        open (array of float): Entry price such as open or previous close.
        high (array of float): High price.
        low (array of float): Low price.
        close (array of float): Close price.
        hit_price_out (array of float): Array where hit price of each exit will be stored.
        stop_type_out (array of int): Array where stop type of each exit will be stored.

            0 for stop loss, 1 for take profit.
        sl_stop (float or array_like): Percentage value for stop loss.

            Can be per frame, column, row, or element-wise. Set to 0 to disable.
        ts_stop (bool or array_like): Percentage value for trailing stop.

            Can be per frame, column, row, or element-wise.
        tp_stop (float or array_like): Percentage value for take profit.

            Can be per frame, column, row, or element-wise. Set to 0 to disable.
        is_open_safe (bool): Whether entry price comes right at or before open.

            If True and wait is 0, can use high/low at entry bar. Otherwise uses only close.
        wait (int): Number of ticks to wait before placing exits.

            Setting False or 0 may result in entry and exit signal at one bar.
        first (bool): Whether to stop as soon as the first exit signal is found.
        temp_idx_arr (array of int): Empty integer array used to temporarily store indices.
        flex_2d (bool): See `vectorbt.base.reshape_fns.flex_choose_i_and_col_nb`.
    """
    sl_stops = np.asarray(sl_stop)
    ts_stops = np.asarray(ts_stop)
    tp_stops = np.asarray(tp_stop)

    init_i = from_i - wait
    init_open = flex_select_auto_nb(init_i, col, open, flex_2d)
    init_sl_stop = abs(flex_select_auto_nb(init_i, col, sl_stops, flex_2d))
    init_ts_stop = abs(flex_select_auto_nb(init_i, col, ts_stops, flex_2d))
    init_tp_stop = abs(flex_select_auto_nb(init_i, col, tp_stops, flex_2d))
    max_i = init_i
    max_p = init_open
    j = 0

    for i in range(from_i, to_i):
        # Calculate stop price
        if init_sl_stop > 0:
            curr_sl_stop_price = init_open * (1 - init_sl_stop)
        if init_ts_stop > 0:
            max_ts_stop = abs(flex_select_auto_nb(max_i, col, ts_stops, flex_2d))
            curr_ts_stop_price = max_p * (1 - max_ts_stop)
        if init_tp_stop > 0:
            curr_tp_stop_price = init_open * (1 + init_tp_stop)

        # Check if stop price is within bar
        if i > init_i or is_open_safe:
            # is_open_safe means open is either open or any other price before it
            # so it's safe to use high/low at entry tick
            curr_high = flex_select_auto_nb(i, col, high, flex_2d)
            curr_low = flex_select_auto_nb(i, col, low, flex_2d)
        else:
            # Otherwise, we can only use close price at entry tick
            curr_close = flex_select_auto_nb(i, col, close, flex_2d)
            curr_high = curr_low = curr_close

        exit_signal = False
        if init_sl_stop > 0:
            if curr_low <= curr_sl_stop_price:
                exit_signal = True
                hit_price_out[i, col] = curr_sl_stop_price
                stop_type_out[i, col] = StopType.StopLoss
        if not exit_signal and init_ts_stop > 0:
            if curr_low <= curr_ts_stop_price:
                exit_signal = True
                hit_price_out[i, col] = curr_ts_stop_price
                stop_type_out[i, col] = StopType.TrailStop
        if not exit_signal and init_tp_stop > 0:
            if curr_high >= curr_tp_stop_price:
                exit_signal = True
                hit_price_out[i, col] = curr_tp_stop_price
                stop_type_out[i, col] = StopType.TakeProfit
        if exit_signal:
            temp_idx_arr[j] = i
            j += 1
            if first:
                return temp_idx_arr[:1]

        # Keep track of highest high if trailing
        if init_ts_stop > 0:
            if curr_high > max_p:
                max_i = i
                max_p = curr_high

    return temp_idx_arr[:j]


@njit
def generate_ohlc_stop_ex_nb(entries: tp.Array2d,
                             open: tp.Array,
                             high: tp.Array,
                             low: tp.Array,
                             close: tp.Array,
                             hit_price_out: tp.Array2d,
                             stop_type_out: tp.Array2d,
                             sl_stop: tp.MaybeArray[float],
                             ts_stop: tp.MaybeArray[float],
                             tp_stop: tp.MaybeArray[float],
                             is_open_safe: bool,
                             wait: int,
                             first: bool,
                             flex_2d: bool) -> tp.Array2d:
    """Generate using `generate_ex_nb` and `ohlc_stop_choice_nb`.

    ## Example

    Generate trailing stop loss and take profit signals for 10%.
    Illustrates how exit signal can be generated within the same tick as entry.
    ```python-repl
    >>> import numpy as np
    >>> from vectorbt.signals.nb import generate_ohlc_stop_ex_nb

    >>> entries = np.asarray([True, False, False, False, False])[:, None]
    >>> open_p = np.asarray([10, 11, 12, 11, 10])[:, None]
    >>> high_p = open_p + 1
    >>> low_p = open_p - 1
    >>> close_p = open_p
    >>> hit_p_out = np.empty_like(entries, dtype=np.float_)
    >>> stop_type_out = np.empty_like(entries, dtype=np.int_)
    >>> sl_stop = 0.1
    >>> ts_stop = 0.1
    >>> tp_stop = 0.1
    >>> is_entry_p_safe = True
    >>> first = True
    >>> flex_2d = True

    >>> generate_ohlc_stop_ex_nb(
    ...     entries, open_p, high_p, low_p, close_p,
    ...     hit_p_out, stop_type_out, sl_stop, tp_stop, tp_stop,
    ...     is_entry_p_safe, 0, first, flex_2d)
    array([[ True],  <<< exit
           [False],
           [False],
           [False],
           [False]])

    >>> hit_p_out
    array([[9.0e+000],  <<< exit
           [5.4e-323],
           [5.9e-323],
           [5.4e-323],
           [4.9e-323]])

    >>> stop_type_out
    array([[ 0],  <<< exit
           [11],
           [12],
           [11],
           [10]])
    ```
    """
    temp_idx_arr = np.empty((entries.shape[0],), dtype=np.int_)
    return generate_ex_nb(
        entries, wait, ohlc_stop_choice_nb,
        open, high, low, close, hit_price_out, stop_type_out,
        sl_stop, ts_stop, tp_stop, is_open_safe, wait, first, temp_idx_arr, flex_2d
    )


@njit
def generate_ohlc_stop_ex_iter_nb(entries: tp.Array2d,
                                  open: tp.Array,
                                  high: tp.Array,
                                  low: tp.Array,
                                  close: tp.Array,
                                  hit_price_out: tp.Array2d,
                                  stop_type_out: tp.Array2d,
                                  sl_stop: tp.MaybeArray[float],
                                  ts_stop: tp.MaybeArray[float],
                                  tp_stop: tp.MaybeArray[float],
                                  is_open_safe: bool,
                                  entry_wait: int,
                                  exit_wait: int,
                                  first: bool,
                                  flex_2d: bool) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Generate iteratively using `generate_enex_nb` and `ohlc_stop_choice_nb`.

    Returns two arrays: new entries and exits."""
    temp_idx_arr = np.empty((entries.shape[0],), dtype=np.int_)
    return generate_enex_nb(
        entries.shape,
        entry_wait, exit_wait,
        first_choice_nb, (entries,),
        ohlc_stop_choice_nb, (
            open, high, low, close, hit_price_out, stop_type_out,
            sl_stop, ts_stop, tp_stop, is_open_safe, exit_wait, first, temp_idx_arr, flex_2d
        )
    )


# ############# Map and reduce ############# #


@njit
def map_reduce_between_nb(a: tp.Array2d,
                          map_func_nb: tp.SignalMapFunc,
                          map_args: tp.Args,
                          reduce_func_nb: tp.SignalReduceFunc,
                          reduce_args: tp.Args) -> tp.Array1d:
    """Map using `map_func_nb` and reduce using `reduce_func_nb` each consecutive
    pair of signals in `a`.

    Applies `map_func_nb` on each range `[from_i, to_i)`. Must accept index of the start of the
    range `from_i`, index of the end of the range `to_i`, index of the column `col`, and `*map_args`.

    Applies `reduce_func_nb` on all mapper results in a column. Must accept index of the column,
    the array of results from `map_func_nb` for that column, and `*reduce_args`.

    ## Example

    ```python-repl
    >>> import numpy as np
    >>> from numba import njit
    >>> from vectorbt.signals.nb import map_reduce_between_nb

    >>> @njit
    ... def map_func_nb(from_i, to_i, col):
    ...     return to_i - from_i
    >>> @njit
    ... def reduce_func_nb(col, map_res):
    ...     return np.nanmean(map_res)
    >>> a = np.asarray([False, True, True, False, True])[:, None]

    >>> map_reduce_between_nb(a, map_func_nb, (), reduce_func_nb, ())
    array([1.5])
    ```
    """
    out = np.full((a.shape[1],), np.nan, dtype=np.float_)

    for col in range(a.shape[1]):
        a_idxs = np.flatnonzero(a[:, col])
        if a_idxs.shape[0] > 1:
            map_res = np.empty(a_idxs.shape[0])
            k = 0
            for j in range(1, a_idxs.shape[0]):
                from_i = a_idxs[j - 1]
                to_i = a_idxs[j]
                map_res[k] = map_func_nb(from_i, to_i, col, *map_args)
                k += 1
            if k > 0:
                out[col] = reduce_func_nb(col, map_res[:k], *reduce_args)
    return out


@njit
def map_reduce_between_two_nb(a: tp.Array2d,
                              b: tp.Array2d,
                              map_func_nb: tp.SignalMapFunc,
                              map_args: tp.Args,
                              reduce_func_nb: tp.SignalReduceFunc,
                              reduce_args: tp.Args) -> tp.Array1d:
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
                        map_res[k] = map_func_nb(from_i, to_i, col, *map_args)
                        k += 1
                if k > 0:
                    out[col] = reduce_func_nb(col, map_res[:k], *reduce_args)
    return out


@njit
def map_reduce_partitions_nb(a: tp.Array2d,
                             map_func_nb: tp.SignalMapFunc,
                             map_args: tp.Args,
                             reduce_func_nb: tp.SignalReduceFunc,
                             reduce_args: tp.Args) -> tp.Array1d:
    """Map using `map_func_nb` and reduce using `reduce_func_nb` each partition of signals in `a`.

    `map_func_nb` and `reduce_func_nb` are same as for `map_reduce_between_nb`."""
    out = np.full((a.shape[1],), np.nan, dtype=np.float_)

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
                map_res[k] = map_func_nb(from_i, to_i, col, *map_args)
                k += 1
                is_partition = False
            if i == a.shape[0] - 1:
                if is_partition:
                    to_i = a.shape[0]
                    map_res[k] = map_func_nb(from_i, to_i, col, *map_args)
                    k += 1
        if k > 0:
            out[col] = reduce_func_nb(col, map_res[:k], *reduce_args)
    return out


@njit(cache=True)
def distance_map_nb(from_i: int, to_i: int, col: int) -> int:
    """Distance mapper."""
    return to_i - from_i


@njit(cache=True)
def mean_reduce_nb(col: int, a: tp.Array1d) -> float:
    """Average reducer."""
    return np.nanmean(a)


# ############# Ranking ############# #


@njit(cache=True)
def rank_1d_nb(a: tp.Array1d,
               reset_by: tp.Optional[tp.Array1d] = None,
               after_false: bool = False,
               allow_gaps: bool = False) -> tp.Array1d:
    """Rank signals in each partition.

    Partition is some number of signals in a row. You can reset partitions by signals from
    `reset_by` (should have the same shape). If `after_false` is True, the first partition should
    come after at least one False value. If `allow_gaps` is True, ignores gaps between partitions.

    ## Example

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
    ```
    """
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
def rank_nb(a: tp.Array2d,
            reset_by: tp.Optional[tp.Array2d] = None,
            after_false: bool = False,
            allow_gaps: bool = False) -> tp.Array2d:
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
def rank_partitions_1d_nb(a: tp.Array1d,
                          reset_by: tp.Optional[tp.Array1d] = None,
                          after_false: bool = False) -> tp.Array1d:
    """Rank partitions of signals.

    For keyword arguments, see `rank_nb`.

    ## Example

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
    ```
    """
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
def rank_partitions_nb(a: tp.Array2d,
                       reset_by: tp.Optional[tp.Array2d] = None,
                       after_false: bool = False) -> tp.Array2d:
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

@njit(cache=True)
def fshift_1d_nb(a: tp.Array1d, n: int) -> tp.Array1d:
    """Shift forward `a` by `n` positions."""
    out = np.empty_like(a, dtype=np.bool_)
    out[:n] = False
    out[n:] = a[:-n]
    return out


@njit(cache=True)
def fshift_nb(a: tp.Array2d, n: int) -> tp.Array2d:
    """2-dim version of `fshift_1d_nb`."""
    return fshift_1d_nb(a, n)
