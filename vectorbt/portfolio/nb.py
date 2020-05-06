"""Numba-compiled functions for calculating portfolio value.

!!! note
    `vectorbt` treats matrices as first-class citizens and expects input arrays to be
    2D, unless function has suffix `_1d` or is meant to be input to another function. 
    Data is processed along index (axis 0)."""

import numpy as np
from numba import njit, b1, i1, i8, f8
from numba.core.types import UniTuple


@njit
def portfolio_np(ts, investment, slippage, commission, order_func_np, *args):
    """Calculate portfolio value in cash and shares based on `order_func_np`.

    Incorporates price `ts`, initial `investment`, and basic transaction costs `slippage`
    and `commission` to calculate the running cash and shares. At each time point `i`,
    runs the function `order_func_np` to get the exact amount of shares to buy/sell.
    Tries then to fulfill that order. If unsuccessful due to insufficient cash/shares,
    always buys/sells the available fraction.

    `slippage` must be in % of price and `commission` in % of transaction volume.

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

        >>> cash, shares = portfolio_np(price, 100, 0, 0.1, order_func_nb)
        >>> print(cash)
        [[98.9 94.5 98.9]
         [98.9 94.5 98.9]
         [98.9 94.5 98.9]
         [98.9 94.5 98.9]
         [98.9 94.5 98.9]]
        >>> print(shares)
        [[1. 1. 1.]
         [1. 1. 1.]
         [1. 1. 1.]
         [1. 1. 1.]
         [1. 1. 1.]]
        >>> equity = cash + shares * price
        >>> print(equity)
        [[ 99.9  99.5  99.9]
         [100.9  98.5 100.9]
         [101.9  97.5 101.9]
         [102.9  96.5 100.9]
         [103.9  95.5  99.9]]
        ```
    """
    cash = np.empty_like(ts, dtype=f8)
    shares = np.empty_like(ts, dtype=f8)

    for col in range(ts.shape[1]):
        run_cash = investment
        run_shares = 0
        for i in range(ts.shape[0]):
            volume = order_func_np(col, i, run_cash, run_shares, *args)  # the amount of shares to buy/sell
            if volume > 0:
                # Buy volume
                adj_price = ts[i, col] * (1 + slippage)  # slippage in % applies on price
                req_cash = volume * adj_price
                req_cash *= (1 + commission)  # commission in % applies on transaction volume
                if req_cash <= run_cash:  # sufficient cash
                    run_shares += volume
                    run_cash -= req_cash
                else:  # insufficient cash, volume will be less than requested
                    # For commission of 10%, you can buy shares for 90.9$ to spend 100$ in total
                    adj_cash = run_cash / (1 + commission)
                    run_shares += adj_cash / adj_price
                    run_cash = 0
            elif volume < 0:
                # Sell volume
                adj_price = ts[i, col] * (1 - slippage)
                adj_shares = min(run_shares, abs(volume))
                adj_cash = adj_shares * adj_price
                adj_cash *= (1 - commission)
                run_shares -= adj_shares
                run_cash += adj_cash
            cash[i, col] = run_cash
            shares[i, col] = run_shares

    return cash, shares


@njit(cache=True)
def signals_order_func_np(col, i, run_cash, run_shares, entries, exits, volume):
    """`order_func_np` to buy/sell based on signals `entries` and `exits`.

    At each entry/exit it buys/sells `volume` shares."""
    if entries[i, col] and not exits[i, col]:
        return volume[i, col]
    if not entries[i, col] and exits[i, col]:
        return -volume[i, col]
    return 0.


@njit(cache=True)
def volume_order_func_np(col, i, run_cash, run_shares, volume, is_target):
    """`order_func_np` to buy/sell amount specified in `volume`.

    If `is_target` is `True`, will buy/sell the difference between current and target volume."""
    if is_target:
        return volume[i, col] - run_shares
    else:
        return volume[i, col]


@njit
def map_positions_nb(positions, map_func_nb, *args):
    """Apply `map_func_nb` on each position in `positions`.

    For each position in range `[entry_i, exit_i]`, `map_func_nb` must return a number
    that is then stored either at index `exit_i` or the last index if position is still open.
    `map_func_nb` must accept index of the current column `col`, index of the entry `entry_i`,
    index of the exit `exit_i`, and `*args`. The index `exit_i` will be `None` if position is open.

    !!! note
        `order_func_np` must be Numba-compiled.

    Example:
        Calculate length of each closed position:
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
        >>> print(map_positions_nb(positions, map_func_nb))
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
                result[i, col] = map_func_nb(col, entry_i, i, *args)
                in_market = False
            elif not in_market and positions[i, col] > 0:
                entry_i = i
                in_market = True
            if in_market and i == positions.shape[0] - 1:  # unrealized
                result[i, col] = map_func_nb(col, entry_i, None, *args)
    return result


@njit(cache=True)
def get_position_equities_nb(col, entry_i, exit_i, ts, cash, shares, investment):
    """Get equity before purchase at `entry_i` and after sale at `exit_i`.

    !!! note
        The index `exit_i` will be `None` if position is still open."""
    if entry_i == 0:
        equity_before = investment
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
    """`map_func_nb` to calculate the P/L of a position`.

    Based on `get_position_equities_nb`."""
    equity_before, equity_after = get_position_equities_nb(*args)
    return equity_after - equity_before


@njit(cache=True)
def returns_map_func_nb(*args):
    """`map_func_nb` to calculate the return of a position`.

    Based on `get_position_equities_nb`."""
    equity_before, equity_after = get_position_equities_nb(*args)
    return equity_after / equity_before - 1


@njit
def reduce_map_results(map_results, reduce_func_nb, *args):
    """Reduce results of `map_positions_nb` using `reduce_func_nb`.

    Applies `reduce_func_nb` on all mapper results in a column. Must accept index
    of the current column, the array of mapper results for that column, and `*args`.

    !!! note
        `reduce_func_nb` must be Numba-compiled.

    Example:
        Calculate length of each closed position:
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
def win_sum_reduce_func_nb(col, map_results):
    """`reduce_func_nb` to sum up profits."""
    return np.sum(map_results[map_results > 0])


@njit(cache=True)
def loss_sum_reduce_func_nb(col, map_results):
    """`reduce_func_nb` to sum up losses.

    Returns an absolute number."""
    return np.abs(np.sum(map_results[map_results < 0]))


@njit(cache=True)
def win_mean_reduce_func_nb(col, map_results):
    """`reduce_func_nb` to take average of profits."""
    wins = map_results[map_results > 0]
    if len(wins) > 0:
        return np.mean(wins)
    return 0.


@njit(cache=True)
def loss_mean_reduce_func_nb(col, map_results):
    """`reduce_func_nb` to take average of losses.

    Returns an absolute number."""
    losses = map_results[map_results < 0]
    if len(losses) > 0:
        return np.abs(np.mean(losses))
    return 0.


@njit(cache=True)
def win_rate_reduce_func_nb(col, map_results):
    """`reduce_func_nb` to calculate win rate."""
    return len(map_results[map_results > 0]) / len(map_results)


@njit(cache=True)
def loss_rate_reduce_func_nb(col, map_results):
    """`reduce_func_nb` to calculate loss rate."""
    return len(map_results[map_results < 0]) / len(map_results)


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
