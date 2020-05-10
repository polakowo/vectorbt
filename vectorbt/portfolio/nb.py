"""Numba-compiled functions for portfolio.

!!! note
    `vectorbt` treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function. 
    Data is processed along index (axis 0)."""

import numpy as np
from numba import njit, b1, i1, i8, f8
from numba.core.types import UniTuple

from vectorbt import timeseries

# ############# Portfolio ############# #


@njit
def portfolio_nb(price, init_capital, fees, slippage, order_func_nb, *args):
    """Calculate portfolio value in cash and shares based on `order_func_nb`.

    Incorporates price `price`, initial `init_capital`, and basic transaction costs `fees`
    and `slippage`. Returns running cash, shares, paid fees and paid slippage.

    `slippage` must be in % of price and `fees` in % of transaction amount, and
    both must have the same shape as `price`.

    At each time point `i`, runs the function `order_func_nb` to get the exact amount of 
    shares to order. Tries then to fulfill that order. If unsuccessful due to insufficient 
    cash/shares, always orders the available fraction. 

    `order_func_nb` must accept index of the current column `col`, the time step `i`,
    the amount of cash `run_cash` and shares `run_shares` held at the time step `i`, and `*args`.
    It must return a positive/negative number to buy/sell. Return 0 to do nothing.

    !!! note
        `order_func_nb` must be Numba-compiled.

    Example:
        Calculate portfolio value for buy-and-hold strategy:
        ```python-repl
        >>> import numpy as np
        >>> from numba import njit
        >>> from vectorbt.portfolio.nb import portfolio_nb

        >>> price = np.asarray([
        ...     [1, 5, 1],
        ...     [2, 4, 2],
        ...     [3, 3, 3],
        ...     [4, 2, 2],
        ...     [5, 1, 1]
        ... ])
        >>> @njit
        ... def order_func_nb(col, i, run_cash, run_shares):
        ...     return 1 if i == 0 else 0 # buy and hold

        >>> cash, shares, paid_fees, paid_slippage = \\
        ...     portfolio_nb(price, 100, 0.1, 0.1, order_func_nb)
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
    cash = np.empty_like(price, dtype=f8)
    shares = np.empty_like(price, dtype=f8)
    paid_fees = np.zeros_like(price, dtype=f8)
    paid_slippage = np.zeros_like(price, dtype=f8)

    for col in range(price.shape[1]):
        run_cash = init_capital
        run_shares = 0
        for i in range(price.shape[0]):
            amount = order_func_nb(col, i, run_cash, run_shares, *args)  # the amount of shares to order
            if amount > 0:
                # Buy amount
                run_cash, run_shares, paid_fees[i, col], paid_slippage[i, col] = buy(
                    run_cash, run_shares, amount, price[i, col], fees[i, col], slippage[i, col])
            elif amount < 0:
                # Sell amount
                run_cash, run_shares, paid_fees[i, col], paid_slippage[i, col] = sell(
                    run_cash, run_shares, amount, price[i, col], fees[i, col], slippage[i, col])
            cash[i, col], shares[i, col] = run_cash, run_shares

    return cash, shares, paid_fees, paid_slippage


@njit(cache=True)
def buy(run_cash, run_shares, amount, price, fees, slippage):
    """Buy `amount` of shares and return updated `run_cash` and `run_shares`, but also paid fees and slippage."""
    # Slippage in % applies on price
    adj_price = price * (1 + slippage)
    req_cash = amount * adj_price
    # Fees in % applies on transaction amount
    req_cash_wcom = req_cash * (1 + fees)
    if req_cash_wcom <= run_cash:
        # Sufficient cash
        new_run_shares = run_shares + amount
        new_run_cash = run_cash - req_cash_wcom
        paid_fees = req_cash_wcom - req_cash
    else:
        # Insufficient cash, amount will be less than requested
        # For fees of 10%, you can buy shares for 90.9$ to spend 100$ in total
        run_cash_wcom = run_cash / (1 + fees)
        new_run_shares = run_shares + run_cash_wcom / adj_price
        new_run_cash = 0
        paid_fees = run_cash - run_cash_wcom
    # Difference in equity is the total cost of transaction = paid_fees + paid_slippage
    old_equity = run_cash + price * run_shares
    new_equity = new_run_cash + price * new_run_shares
    if slippage == 0:
        paid_slippage = 0. # otherwise you will get numbers such as 7.105427e-15
    else:
        paid_slippage = old_equity - new_equity - paid_fees
    return new_run_cash, new_run_shares, paid_fees, paid_slippage


@njit(cache=True)
def sell(run_cash, run_shares, amount, price, fees, slippage):
    """Sell `amount` of shares and return updated `run_cash` and `run_shares`, but also paid fees and slippage."""
    # Slippage in % applies on price
    adj_price = price * (1 - slippage)
    # If insufficient shares, sell what's left
    adj_shares = min(run_shares, abs(amount))
    adj_cash = adj_shares * adj_price
    # Fees in % applies on transaction amount
    adj_cash_wcom = adj_cash * (1 - fees)
    new_run_shares = run_shares - adj_shares
    new_run_cash = run_cash + adj_cash_wcom
    paid_fees = adj_cash - adj_cash_wcom
    # Difference in equity is the total cost of transaction = paid_fees + paid_slippage
    old_equity = run_cash + price * run_shares
    new_equity = new_run_cash + price * new_run_shares
    if slippage == 0:
        paid_slippage = 0.
    else:
        paid_slippage = old_equity - new_equity - paid_fees
    return new_run_cash, new_run_shares, paid_fees, paid_slippage


@njit(cache=True)
def signals_order_func_nb(col, i, run_cash, run_shares, entries, exits, amount):
    """`order_func_nb` that orders based on signals `entries` and `exits`.

    At each entry/exit it buys/sells `amount` of shares."""
    if entries[i, col] and not exits[i, col]:
        return amount[i, col]
    if not entries[i, col] and exits[i, col]:
        return -amount[i, col]
    return 0.


@njit(cache=True)
def amount_order_func_nb(col, i, run_cash, run_shares, amount, is_target):
    """`order_func_nb` that orders the amount specified in `amount`.

    If `is_target` is `True`, will order the difference between current and target amount."""
    if is_target:
        return amount[i, col] - run_shares
    else:
        return amount[i, col]

# ############# Mappers ############# #


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
        `order_func_nb` must be Numba-compiled.

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
def get_position_equities_nb(col, entry_i, exit_i, price, cash, shares, init_capital):
    """Get equity before purchase at `entry_i` and after sale at `exit_i`.

    !!! note
        The index `exit_i` will be `None` if position is still open."""
    if entry_i == 0:
        equity_before = init_capital
    else:
        # We can't use equity at time entry_i, since it already has purchase cost applied
        # Instead apply price at entry_i to the cash and shares immediately before purchase
        equity_before = cash[entry_i-1, col] + shares[entry_i-1, col] * price[entry_i, col]
    if exit_i is not None:
        equity_after = cash[exit_i, col] + shares[exit_i, col] * price[exit_i, col]
    else:
        # A bit optimistic, since it doesn't include sale cost
        equity_after = cash[price.shape[0]-1, col] + shares[price.shape[0]-1, col] * price[price.shape[0]-1, col]
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

# ############# Filters ############# #


@njit(cache=True)
def winning_filter_func_nb(col, i, map_result, pnl):
    """`filter_func_nb` that includes only winning positions."""
    return pnl[i, col] > 0


@njit(cache=True)
def losing_filter_func_nb(col, i, map_result, pnl):
    """`filter_func_nb` that includes only losing positions."""
    return pnl[i, col] < 0

# ############# Appliers ############# #


@njit(cache=True)
def total_return_apply_func_nb(col, idxs, returns):
    """Calculate total return from returns."""
    return timeseries.nb.product_1d_nb(returns + 1) - 1

# ############# Accumulation ############# #


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
    """2-dim version of `is_accumulated_1d_nb`."""
    result = np.empty(positions.shape[1], b1)
    for col in range(positions.shape[1]):
        result[col] = is_accumulated_1d_nb(positions[:, col])
    return result
