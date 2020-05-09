"""Numba-compiled functions for calculating portfolio value.

!!! note
    `vectorbt` treats matrices as first-class citizens and expects input arrays to be
    2D, unless function has suffix `_1d` or is meant to be input to another function. 
    Data is processed along index (axis 0)."""

import numpy as np
from numba import njit, b1, i1, i8, f8
from numba.core.types import UniTuple

from vectorbt import timeseries


@njit
def portfolio_np(ts, init_capital, fees, slippage, order_func_np, *args):
    """Calculate portfolio value in cash and shares based on `order_func_np`.

    Incorporates price `ts`, initial `init_capital`, and basic transaction costs `fees`
    and `slippage`. Returns running cash, shares, paid fees and paid slippage.

    `slippage` must be in % of price and `fees` in % of transaction volume.

    At each time point `i`, runs the function `order_func_np` to get the exact amount of 
    shares to buy/sell. Tries then to fulfill that order. If unsuccessful due to insufficient 
    cash/shares, always buys/sells the available fraction. 

    `order_func_np` must accept index of the current column `col`, the time step `i`,
    the amount of cash `run_cash` and shares `run_shares` held at the time step `i`, and `*args`.
    It must return a positive/negative number to buy/sell. Return 0 to do nothing.

    !!! note
        `order_func_np` must be Numba-compiled.

    Example:
        Calculate portfolio value for buy-and-hold strategy:
        ```python-repl
        >>> import numpy as np
        >>> from numba import njit
        >>> from vectorbt.portfolio.nb import portfolio_np

        >>> price = np.asarray([
        ...     [1, 5, 1],
        ...     [2, 4, 2],
        ...     [3, 3, 3],
        ...     [4, 2, 2],
        ...     [5, 1, 1]
        ... ])
        >>> @njit
        ... def order_func_nb(col, i, run_cash, run_shares):
        ...     return 1 if i == 0 else 0

        >>> cash, shares, paid_fees, paid_slippage = \\
        ...     portfolio_np(price, 100, 0.1, 0.1, order_func_nb)
        >>> print(cash)
        [[98.79  93.95  98.79]
         [100.   100.   100. ]
         [100.   100.   100. ]
         [100.   100.   100. ]
         [100.   100.   100. ]]
        >>> print(shares)
        [[1. 1. 1.]
         [0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]]
        >>> print(paid_fees)
        [[0.11 0.55 0.11]
         [0.   0.   0.  ]
         [0.   0.   0.  ]
         [0.   0.   0.  ]
         [0.   0.   0.  ]]
        >>> print(paid_slippage)
        [[0.1 0.5 0.1]
         [0.  0.  0. ]
         [0.  0.  0. ]
         [0.  0.  0. ]
         [0.  0.  0. ]]
        ```
    """
    cash = np.empty_like(ts, dtype=f8)
    shares = np.empty_like(ts, dtype=f8)
    paid_fees = np.zeros_like(ts, dtype=f8)
    paid_slippage = np.zeros_like(ts, dtype=f8)

    for col in range(ts.shape[1]):
        run_cash = init_capital
        run_shares = 0
        for i in range(ts.shape[0]):
            volume = order_func_np(col, i, run_cash, run_shares, *args)  # the amount of shares to buy/sell
            if volume > 0:
                # Buy volume
                run_cash, run_shares, paid_fees[i, col], paid_slippage[i, col] = buy_volume(
                    run_cash, run_shares, volume, ts[i, col], fees[i, col], slippage[i, col])
            elif volume < 0:
                # Sell volume
                run_cash, run_shares, paid_fees[i, col], paid_slippage[i, col] = sell_volume(
                    run_cash, run_shares, volume, ts[i, col], fees[i, col], slippage[i, col])
            cash[i, col], shares[i, col] = run_cash, run_shares

    return cash, shares, paid_fees, paid_slippage


@njit(cache=True)
def buy_volume(run_cash, run_shares, volume, ts, fees, slippage):
    """Buy volume and return updated `run_cash` and `run_shares`, but also paid fees and slippage."""
    # Slippage in % applies on price
    adj_price = ts * (1 + slippage)
    req_cash = volume * adj_price
    # Fees in % applies on transaction volume
    req_cash_wcom = req_cash * (1 + fees)
    if req_cash_wcom <= run_cash:
        # Sufficient cash
        new_run_shares = run_shares + volume
        new_run_cash = run_cash - req_cash_wcom
        paid_fees = req_cash_wcom - req_cash
    else:
        # Insufficient cash, volume will be less than requested
        # For fees of 10%, you can buy shares for 90.9$ to spend 100$ in total
        run_cash_wcom = run_cash / (1 + fees)
        new_run_shares = run_shares + run_cash_wcom / adj_price
        new_run_cash = 0
        paid_fees = run_cash - run_cash_wcom
    # Difference in equity is the total cost of transaction = paid_fees + paid_slippage
    old_equity = run_cash + ts * run_shares
    new_equity = new_run_cash + ts * new_run_shares
    paid_slippage = old_equity - new_equity - paid_fees
    return new_run_cash, new_run_shares, paid_fees, paid_slippage


@njit(cache=True)
def sell_volume(run_cash, run_shares, volume, ts, fees, slippage):
    """Sell volume and return updated `run_cash` and `run_shares`, but also paid fees and slippage."""
    # Slippage in % applies on price
    adj_price = ts * (1 - slippage)
    # If insufficient volume, sell what's left
    adj_shares = min(run_shares, abs(volume))
    adj_cash = adj_shares * adj_price
    # Fees in % applies on transaction volume
    adj_cash_wcom = adj_cash * (1 - fees)
    new_run_shares = run_shares - adj_shares
    new_run_cash = run_cash + adj_cash_wcom
    paid_fees = adj_cash - adj_cash_wcom
    # Difference in equity is the total cost of transaction = paid_fees + paid_slippage
    old_equity = run_cash + ts * run_shares
    new_equity = new_run_cash + ts * new_run_shares
    paid_slippage = old_equity - new_equity - paid_fees
    return new_run_cash, new_run_shares, paid_fees, paid_slippage


@njit(cache=True)
def signals_order_func_np(col, i, run_cash, run_shares, entries, exits, volume):
    """`order_func_np` that buys/sells based on signals `entries` and `exits`.

    At each entry/exit it buys/sells `volume` shares."""
    if entries[i, col] and not exits[i, col]:
        return volume[i, col]
    if not entries[i, col] and exits[i, col]:
        return -volume[i, col]
    return 0.


@njit(cache=True)
def volume_order_func_np(col, i, run_cash, run_shares, volume, is_target):
    """`order_func_np` that buys/sells amount specified in `volume`.

    If `is_target` is `True`, will buy/sell the difference between current and target volume."""
    if is_target:
        return volume[i, col] - run_shares
    else:
        return volume[i, col]


OPEN = 0
"""Open position."""
CLOSED = 1
"""Closed position."""


@njit
def map_positions_nb(positions, pos_type, map_func_nb, *args):
    """Apply `map_func_nb` on each position in `positions`.

    `pos_type` can be either `None`, `OPEN` or `CLOSED`.

    For each position in range `[entry_i, exit_i]`, `map_func_nb` must return a number
    that is then stored either at index `exit_i` or the last index if position is still open.
    `map_func_nb` must accept index of the current column `col`, index of the entry `entry_i`,
    index of the exit `exit_i`, and `*args`. The index `exit_i` will be `None` if position is open.

    !!! note
        `order_func_np` must be Numba-compiled.

    Example:
        Map each closed position to its duration:
        ```python-repl
        >>> import numpy as np
        >>> from numba import njit
        >>> from vectorbt.portfolio.nb import map_positions_nb

        >>> @njit
        ... def map_func_nb(col, entry_i, exit_i):
        ...     if exit_i is not None:
        ...         return exit_i - entry_i
        ...     return np.nan # ignore open positions
        >>> positions = np.asarray([
        ...     [1, 0, 1],
        ...     [2, 1, 2],
        ...     [0, 2, 3],
        ...     [1, 0, 4],
        ...     [0, 1, 5]
        ... ])
        >>> print(map_positions_nb(positions, None, map_func_nb))
        [[nan nan nan]
         [nan nan nan]
         [ 2. nan nan]
         [nan  2. nan]
         [ 1. nan nan]]
        ```"""
    result = np.full_like(positions, np.nan, dtype=f8)

    for col in range(positions.shape[1]):
        entry_i = 0
        in_market = positions[0, col] > 0
        for i in range(1, positions.shape[0]):
            if in_market and positions[i, col] == 0:
                if pos_type is None or pos_type == CLOSED:
                    result[i, col] = map_func_nb(col, entry_i, i, *args)
                in_market = False
            elif not in_market and positions[i, col] > 0:
                entry_i = i
                in_market = True
            if in_market and i == positions.shape[0] - 1:  # unrealized
                if pos_type is None or pos_type == OPEN:
                    result[i, col] = map_func_nb(col, entry_i, None, *args)
    return result


@njit(cache=True)
def get_position_equities_nb(col, entry_i, exit_i, ts, cash, shares, init_capital):
    """Get equity before purchase at `entry_i` and after sale at `exit_i`.

    !!! note
        The index `exit_i` will be `None` if position is still open."""
    if entry_i == 0:
        equity_before = init_capital
    else:
        # We can't use equity at time entry_i, since it already has purchase cost applied
        # Instead apply price at entry_i to the cash and shares immediately before purchase
        equity_before = cash[entry_i-1, col] + shares[entry_i-1, col] * ts[entry_i, col]
    if exit_i is not None:
        equity_after = cash[exit_i, col] + shares[exit_i, col] * ts[exit_i, col]
    else:
        # A bit optimistic, since it doesn't include sale cost
        equity_after = cash[ts.shape[0]-1, col] + shares[ts.shape[0]-1, col] * ts[ts.shape[0]-1, col]
    return equity_before, equity_after


@njit(cache=True)
def pnl_map_func_nb(*args):
    """`map_func_nb` that returns P/L of the position`.

    Based on `get_position_equities_nb`."""
    equity_before, equity_after = get_position_equities_nb(*args)
    return equity_after - equity_before


@njit(cache=True)
def returns_map_func_nb(*args):
    """`map_func_nb` that returns return of the position`.

    Based on `get_position_equities_nb`."""
    equity_before, equity_after = get_position_equities_nb(*args)
    return equity_after / equity_before - 1


@njit(cache=True)
def status_map_func_nb(col, entry_i, exit_i):
    """`map_func_nb` that returns whether the position is open or closed."""
    if exit_i is None:
        return 0
    return 1


@njit(cache=True)
def duration_map_func_nb(col, entry_i, exit_i, shape):
    """`map_func_nb` that returns duration of the position."""
    if exit_i is None:
        return shape[0] - entry_i
    return exit_i - entry_i

@njit
def filter_map_results(map_results, filter_func_nb, *args):
    """Filter map results using `filter_func_nb`.

    Applies `filter_func_nb` on all map results in a column. Must accept index
    of the current column, index of the current element, the element itself, and `*args`.

    !!! note
        `filter_func_nb` must be Numba-compiled.

    Example:
        Filter out the positions of duration less than 2:
        ```python-repl
        >>> import numpy as np
        >>> from numba import njit
        >>> from vectorbt.portfolio.nb import filter_map_results

        >>> @njit
        ... def filter_func_nb(col, i, map_result):
        ...     return map_result > 1
        >>> map_results = np.asarray([
        ...     [np.nan, np.nan, np.nan],
        ...     [np.nan, np.nan, np.nan],
        ...     [2., np.nan, np.nan],
        ...     [np.nan, 2., np.nan],
        ...     [1., np.nan, np.nan]
        ... ])
        >>> print(filter_map_results(map_results, filter_func_nb))
        [[nan nan nan]
         [nan nan nan]
         [ 2. nan nan]
         [nan  2. nan]
         [nan nan nan]]
        ```"""
    result = np.copy(map_results)

    for col in range(result.shape[1]):
        idxs = np.flatnonzero(~np.isnan(map_results[:, col]))
        for i in idxs:
            if ~filter_func_nb(col, i, map_results[i, col], *args):
                result[i, col] = np.nan
    return result

@njit(cache=True)
def winning_filter_func_nb(col, i, map_result, pnl):
    """`filter_func_nb` that includes only winning positions."""
    return pnl[i, col] > 0


@njit(cache=True)
def losing_filter_func_nb(col, i, map_result, pnl):
    """`filter_func_nb` that includes only losing positions."""
    return pnl[i, col] < 0


@njit
def reduce_map_results(map_results, reduce_func_nb, *args):
    """Reduce map results of each column into a single value using `reduce_func_nb`.

    Applies `reduce_func_nb` on all map results in a column. Must accept index
    of the current column, the array of map results for that column, and `*args`.

    !!! note
        `reduce_func_nb` must be Numba-compiled.

    Example:
        Return the average duration of the positions in each column:
        ```python-repl
        >>> import numpy as np
        >>> from numba import njit
        >>> from vectorbt.portfolio.nb import reduce_map_results

        >>> @njit
        ... def reduce_func_nb(col, map_results):
        ...     return np.nanmean(map_results)
        >>> map_results = np.asarray([
        ...     [np.nan, np.nan, np.nan],
        ...     [np.nan, np.nan, np.nan],
        ...     [2., np.nan, np.nan],
        ...     [np.nan, 2., np.nan],
        ...     [1., np.nan, np.nan]
        ... ])
        >>> print(reduce_map_results(map_results, reduce_func_nb))
        [1.5 2.  nan]
        ```"""
    result = np.full(map_results.shape[1], np.nan, dtype=f8)

    for col in range(map_results.shape[1]):
        filled = map_results[~np.isnan(map_results[:, col]), col]
        if len(filled) > 0:
            result[col] = reduce_func_nb(col, filled, *args)
    return result


@njit(cache=True)
def pos_sum_reduce_func_nb(col, map_results):
    """`reduce_func_nb` that sums up positive results."""
    return np.sum(map_results[map_results > 0])


@njit(cache=True)
def neg_sum_reduce_func_nb(col, map_results):
    """`reduce_func_nb` that sums up negative results.

    Returns an absolute number."""
    return np.abs(np.sum(map_results[map_results < 0]))


@njit(cache=True)
def pos_mean_reduce_func_nb(col, map_results):
    """`reduce_func_nb` that takes average of positive results."""
    wins = map_results[map_results > 0]
    if len(wins) > 0:
        return np.mean(wins)
    return 0.


@njit(cache=True)
def neg_mean_reduce_func_nb(col, map_results):
    """`reduce_func_nb` that takes average of negative results.

    Returns an absolute number."""
    losses = map_results[map_results < 0]
    if len(losses) > 0:
        return np.abs(np.mean(losses))
    return 0.


@njit(cache=True)
def pos_rate_reduce_func_nb(col, map_results):
    """`reduce_func_nb` that returns fraction of positive results."""
    return len(map_results[map_results > 0]) / len(map_results)


@njit(cache=True)
def neg_rate_reduce_func_nb(col, map_results):
    """`reduce_func_nb` that returns fraction of negative results."""
    return len(map_results[map_results < 0]) / len(map_results)


@njit(cache=True)
def open_cnt_reduce_func_nb(col, map_results):
    """`reduce_func_nb` that counts open positions."""
    return len(map_results[map_results == OPEN])


@njit(cache=True)
def closed_cnt_reduce_func_nb(col, map_results):
    """`reduce_func_nb` that counts closed positions."""
    return len(map_results[map_results == CLOSED])


@njit(cache=True)
def cnt_reduce_func_nb(col, map_results):
    """`reduce_func_nb` that counts all results."""
    return len(map_results)


@njit(cache=True)
def mean_reduce_func_nb(col, map_results):
    """`reduce_func_nb` that returns average of all results."""
    return np.mean(map_results)


@njit(cache=True)
def sum_reduce_func_nb(col, map_results):
    """`reduce_func_nb` that sums up all results."""
    return np.sum(map_results)


@njit
def mult_reduce_map_results(map_results, mult_reduce_func_nb, *args):
    """Reduce map results of each column into an array using `mult_reduce_func_nb`.

    `mult_reduce_func_nb` same as for `reduce_map_results` but must return an array.

    !!! note
        * `mult_reduce_func_nb` must be Numba-compiled
        * Output of `mult_reduce_func_nb` must be strictly homogeneous"""
    from_col = -1
    for col in range(map_results.shape[1]):
        filled = map_results[~np.isnan(map_results[:, col]), col]
        if len(filled) > 0:
            result0 = mult_reduce_func_nb(col, filled, *args)
            from_col = col
            break
    if from_col == -1:
        raise ValueError("All map results are NA")
    result = np.full((result0.shape[0], map_results.shape[1]), np.nan, dtype=f8)
    for col in range(from_col, map_results.shape[1]):
        filled = map_results[~np.isnan(map_results[:, col]), col]
        if len(filled) > 0:
            result[:, col] = mult_reduce_func_nb(col, filled, *args)
    return result


@njit(cache=True)
def describe_mult_reduce_func_nb(col, map_results, percentiles):
    """`mult_reduce_func_nb` that describes results using statistics."""
    result = np.empty(5 + len(percentiles), dtype=f8)
    result[0] = len(map_results)
    result[1] = np.mean(map_results)
    rcount = max(len(map_results) - 1, 0)
    if rcount == 0:
        result[2] = np.nan
    else:
        result[2] = np.std(map_results) * np.sqrt(len(map_results) / rcount)
    result[3] = np.min(map_results)
    for i, percentile in enumerate(percentiles):
        result[4:-1] = np.percentile(map_results, percentiles * 100)
    result[4+len(percentiles)] = np.max(map_results)
    return result


@njit(cache=True)
def is_accumulated_1d_nb(positions):
    """Detect accumulation, that is, position is being increased/decreased gradually."""
    for i in range(1, positions.shape[0]):
        if (positions[i-1] > 0 and positions[i] > 0) or \
                (positions[i-1] < 0 and positions[i] < 0):
            if positions[i-1] != positions[i]:
                return True
    return False


@njit(cache=True)
def is_accumulated_nb(positions):
    """2D version of `is_accumulated_1d_nb`."""
    result = np.empty(positions.shape[1], b1)
    for col in range(positions.shape[1]):
        result[col] = is_accumulated_1d_nb(positions[:, col])
    return result


@njit(cache=True)
def total_return_apply_func_nb(returns):
    """Calculate total return from returns."""
    return timeseries.nb.product_1d_nb(returns + 1) - 1
