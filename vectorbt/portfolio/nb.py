"""Numba-compiled 1-dim and 2-dim functions.

!!! note
    `vectorbt` treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function. 
    Data is processed along index (axis 0)."""

import numpy as np
from numba import njit, b1, i1, i8, f8
from numba.core.types import UniTuple

from vectorbt import timeseries
from vectorbt.portfolio.const import PositionType

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

        >>> cash, shares, fees_paid, slippage_paid = \\
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
        >>> print(fees_paid)
        [[0.11 0.55 0.11]
         [0.   0.   0.  ]
         [0.   0.   0.  ]
         [0.   0.   0.  ]
         [0.   0.   0.  ]]
        >>> print(slippage_paid)
        [[0.1 0.5 0.1]
         [0.  0.  0. ]
         [0.  0.  0. ]
         [0.  0.  0. ]
         [0.  0.  0. ]]
        ```
    """
    cash = np.empty_like(price, dtype=f8)
    shares = np.empty_like(price, dtype=f8)
    fees_paid = np.zeros_like(price, dtype=f8)
    slippage_paid = np.zeros_like(price, dtype=f8)

    for col in range(price.shape[1]):
        run_cash = init_capital
        run_shares = 0
        for i in range(price.shape[0]):
            amount = order_func_nb(col, i, run_cash, run_shares, *args)  # the amount of shares to order
            if amount > 0:
                # Buy amount
                run_cash, run_shares, fees_paid[i, col], slippage_paid[i, col] = buy(
                    run_cash, run_shares, amount, price[i, col], fees[i, col], slippage[i, col])
            elif amount < 0:
                # Sell amount
                run_cash, run_shares, fees_paid[i, col], slippage_paid[i, col] = sell(
                    run_cash, run_shares, amount, price[i, col], fees[i, col], slippage[i, col])
            cash[i, col], shares[i, col] = run_cash, run_shares

    return cash, shares, fees_paid, slippage_paid


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
        fees_paid = req_cash_wcom - req_cash
    else:
        # Insufficient cash, amount will be less than requested
        # For fees of 10%, you can buy shares for 90.9$ to spend 100$ in total
        run_cash_wcom = run_cash / (1 + fees)
        new_run_shares = run_shares + run_cash_wcom / adj_price
        new_run_cash = 0
        fees_paid = run_cash - run_cash_wcom
    # Difference in equity is the total cost of transaction = fees_paid + slippage_paid
    old_equity = run_cash + price * run_shares
    new_equity = new_run_cash + price * new_run_shares
    if slippage == 0:
        slippage_paid = 0.  # otherwise you will get numbers such as 7.105427e-15
    else:
        slippage_paid = old_equity - new_equity - fees_paid
    return new_run_cash, new_run_shares, fees_paid, slippage_paid


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
    fees_paid = adj_cash - adj_cash_wcom
    # Difference in equity is the total cost of transaction = fees_paid + slippage_paid
    old_equity = run_cash + price * run_shares
    new_equity = new_run_cash + price * new_run_shares
    if slippage == 0:
        slippage_paid = 0.
    else:
        slippage_paid = old_equity - new_equity - fees_paid
    return new_run_cash, new_run_shares, fees_paid, slippage_paid


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


OPEN = PositionType.OPEN
CLOSED = PositionType.CLOSED


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

# ############# Financial risk and performance metrics ############# #

# Functions from empyrical but Numba-compiled


@njit(cache=True)
def cum_returns_1d_nb(returns, starting_value=0):
    """See `empyrical.cum_returns`."""
    if returns.shape[0] < 1:
        return returns.copy()

    result = timeseries.nb.cumprod_1d_nb(returns + 1)
    if starting_value == 0:
        result -= 1
    else:
        result *= starting_value
    return result


@njit(cache=True)
def cum_returns_nb(returns, starting_value=0):
    """2-dim version of `cum_returns_1d_nb`."""
    result = np.empty_like(returns, dtype=f8)
    for col in range(returns.shape[1]):
        result[:, col] = cum_returns_1d_nb(returns[:, col], starting_value=starting_value)
    return result


@njit(cache=True)
def cum_returns_final_1d_nb(returns, starting_value=0):
    """See `empyrical.cum_returns_final`."""
    if returns.shape[0] == 0:
        return np.nan

    result = timeseries.nb.product_1d_nb(returns + 1)
    if starting_value == 0:
        result -= 1
    else:
        result *= starting_value
    return result


@njit(cache=True)
def cum_returns_final_nb(returns, starting_value=0):
    """2-dim version of `cum_returns_final_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=f8)
    for col in range(returns.shape[1]):
        result[col] = cum_returns_final_1d_nb(returns[:, col], starting_value=starting_value)
    return result


@njit(cache=True)
def annualized_return_1d_nb(returns, ann_factor):
    """See `empyrical.annual_return`."""
    if returns.shape[0] < 1:
        return np.nan

    ending_value = cum_returns_final_1d_nb(returns, starting_value=1)
    return ending_value ** (ann_factor / returns.shape[0]) - 1


@njit(cache=True)
def annualized_return_nb(returns, ann_factor):
    """2-dim version of `annualized_return_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=f8)
    for col in range(returns.shape[1]):
        result[col] = annualized_return_1d_nb(returns[:, col], ann_factor)
    return result


@njit(cache=True)
def annualized_volatility_1d_nb(returns, ann_factor, alpha=2.0):
    """See `empyrical.annual_volatility`."""
    if returns.shape[0] < 2:
        return np.nan

    return timeseries.nb.nanstd_1d_nb(returns, ddof=1) * ann_factor ** (1.0 / alpha)


@njit(cache=True)
def annualized_volatility_nb(returns, ann_factor, alpha=2.0):
    """2-dim version of `annualized_volatility_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=f8)
    for col in range(returns.shape[1]):
        result[col] = annualized_volatility_1d_nb(returns[:, col], ann_factor, alpha=alpha)
    return result


@njit(cache=True)
def calmar_ratio_1d_nb(returns, annualized_return, max_drawdown, ann_factor):
    """See `empyrical.calmar_ratio`."""
    if max_drawdown == 0.:
        return np.nan
    return annualized_return / np.abs(max_drawdown)


@njit(cache=True)
def calmar_ratio_nb(returns, annualized_return, max_drawdown, ann_factor):
    """2-dim version of `calmar_ratio_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=f8)
    for col in range(returns.shape[1]):
        result[col] = calmar_ratio_1d_nb(returns[:, col], annualized_return[col], max_drawdown[col], ann_factor)
    return result


@njit(cache=True)
def omega_ratio_1d_nb(returns, ann_factor, risk_free=0., required_return=0.):
    """See `empyrical.omega_ratio`."""
    if returns.shape[0] < 1:
        return np.nan

    if ann_factor == 1:
        return_threshold = required_return
    elif ann_factor <= -1:
        return np.nan
    else:
        return_threshold = (1 + required_return) ** (1. / ann_factor) - 1
    returns_less_thresh = returns - risk_free - return_threshold
    numer = np.sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * np.sum(returns_less_thresh[returns_less_thresh < 0.0])
    if denom == 0.:
        return np.nan
    return numer / denom


@njit(cache=True)
def omega_ratio_nb(returns, ann_factor, risk_free=0., required_return=0.):
    """2-dim version of `omega_ratio_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=f8)
    for col in range(returns.shape[1]):
        result[col] = omega_ratio_1d_nb(
            returns[:, col], ann_factor, risk_free=risk_free, required_return=required_return)
    return result


@njit(cache=True)
def sharpe_ratio_1d_nb(returns, ann_factor, risk_free=0.):
    """See `empyrical.sharpe_ratio`."""
    if returns.shape[0] < 2:
        return np.nan

    returns_risk_adj = returns - risk_free
    mean = np.nanmean(returns_risk_adj)
    std = timeseries.nb.nanstd_1d_nb(returns_risk_adj, ddof=1)
    if std == 0.:
        return np.nan
    return mean / std * np.sqrt(ann_factor)


@njit(cache=True)
def sharpe_ratio_nb(returns, ann_factor, risk_free=0.):
    """2-dim version of `sharpe_ratio_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=f8)
    for col in range(returns.shape[1]):
        result[col] = sharpe_ratio_1d_nb(returns[:, col], ann_factor, risk_free=risk_free)
    return result


@njit(cache=True)
def downside_risk_1d_nb(returns, ann_factor, required_return=0.):
    """See `empyrical.downside_risk`."""
    if returns.shape[0] < 1:
        return np.nan

    adj_returns = returns - required_return
    adj_returns[adj_returns > 0] = 0
    return np.sqrt(np.nanmean(adj_returns ** 2)) * np.sqrt(ann_factor)


@njit(cache=True)
def downside_risk_nb(returns, ann_factor, required_return=0.):
    """2-dim version of `downside_risk_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=f8)
    for col in range(returns.shape[1]):
        result[col] = downside_risk_1d_nb(returns[:, col], ann_factor, required_return=required_return)
    return result


@njit(cache=True)
def sortino_ratio_1d_nb(returns, downside_risk, ann_factor, required_return=0.):
    """See `empyrical.sortino_ratio`."""
    if returns.shape[0] < 2:
        return np.nan

    adj_returns = returns - required_return
    average_annualized_return = np.nanmean(adj_returns) * ann_factor
    if downside_risk == 0.:
        return np.nan
    return average_annualized_return / downside_risk


@njit(cache=True)
def sortino_ratio_nb(returns, downside_risk, ann_factor, required_return=0.):
    """2-dim version of `sortino_ratio_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=f8)
    for col in range(returns.shape[1]):
        result[col] = sortino_ratio_1d_nb(
            returns[:, col], downside_risk[col], ann_factor, required_return=required_return)
    return result


@njit(cache=True)
def information_ratio_1d_nb(returns, factor_returns):
    """See `empyrical.excess_sharpe`."""
    if returns.shape[0] < 2:
        return np.nan

    active_return = returns - factor_returns
    return np.nanmean(active_return) / timeseries.nb.nanstd_1d_nb(active_return, ddof=1)


@njit(cache=True)
def information_ratio_nb(returns, factor_returns):
    """2-dim version of `information_ratio_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=f8)
    for col in range(returns.shape[1]):
        result[col] = information_ratio_1d_nb(returns[:, col], factor_returns[:, col])
    return result


@njit(cache=True)
def beta_1d_nb(returns, factor_returns, risk_free=0.):
    """See `empyrical.beta`."""
    if returns.shape[0] < 1 or factor_returns.shape[0] < 2:
        return np.nan

    independent = np.where(
        np.isnan(returns),
        np.nan,
        factor_returns,
    )
    ind_residual = independent - np.nanmean(independent)
    covariances = np.nanmean(ind_residual * returns)
    ind_residual = ind_residual ** 2
    ind_variances = np.nanmean(ind_residual)
    if ind_variances < 1.0e-30:
        ind_variances = np.nan
    if ind_variances == 0.:
        return np.nan
    return covariances / ind_variances


@njit(cache=True)
def beta_nb(returns, factor_returns, risk_free=0.):
    """2-dim version of `beta_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=f8)
    for col in range(returns.shape[1]):
        result[col] = beta_1d_nb(returns[:, col], factor_returns[:, col], risk_free=risk_free)
    return result


@njit(cache=True)
def alpha_1d_nb(returns, factor_returns, beta, ann_factor, risk_free=0.):
    """See `empyrical.alpha`."""
    if returns.shape[0] < 2:
        return np.nan

    adj_returns = returns - risk_free
    adj_factor_returns = factor_returns - risk_free
    alpha_series = adj_returns - (beta * adj_factor_returns)
    return (np.nanmean(alpha_series) + 1) ** ann_factor - 1


@njit(cache=True)
def alpha_nb(returns, factor_returns, beta, ann_factor, risk_free=0.):
    """2-dim version of `alpha_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=f8)
    for col in range(returns.shape[1]):
        result[col] = alpha_1d_nb(returns[:, col], factor_returns[:, col], beta[col], ann_factor, risk_free=risk_free)
    return result


@njit(cache=True)
def tail_ratio_1d_nb(returns):
    """See `empyrical.tail_ratio`."""
    if returns.shape[0] < 1:
        return np.nan

    returns = returns[~np.isnan(returns)]
    if len(returns) < 1:
        return np.nan
    perc_95 = np.abs(np.percentile(returns, 95))
    perc_5 = np.abs(np.percentile(returns, 5))
    if perc_5 == 0.:
        return np.nan
    return perc_95 / perc_5


@njit(cache=True)
def tail_ratio_nb(returns):
    """2-dim version of `tail_ratio_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=f8)
    for col in range(returns.shape[1]):
        result[col] = tail_ratio_1d_nb(returns[:, col])
    return result


@njit(cache=True)
def value_at_risk_1d_nb(returns, cutoff=0.05):
    """See `empyrical.value_at_risk`."""
    if returns.shape[0] < 1:
        return np.nan

    returns = returns[~np.isnan(returns)]
    if len(returns) < 1:
        return np.nan
    return np.percentile(returns, 100 * cutoff)


@njit(cache=True)
def value_at_risk_nb(returns, cutoff=0.05):
    """2-dim version of `value_at_risk_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=f8)
    for col in range(returns.shape[1]):
        result[col] = value_at_risk_1d_nb(returns[:, col], cutoff=cutoff)
    return result


@njit(cache=True)
def conditional_value_at_risk_1d_nb(returns, cutoff=0.05):
    """See `empyrical.conditional_value_at_risk`."""
    cutoff_index = int((len(returns) - 1) * cutoff)
    return np.mean(np.partition(returns, cutoff_index)[:cutoff_index + 1])


@njit(cache=True)
def conditional_value_at_risk_nb(returns, cutoff=0.05):
    """2-dim version of `conditional_value_at_risk_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=f8)
    for col in range(returns.shape[1]):
        result[col] = conditional_value_at_risk_1d_nb(returns[:, col], cutoff=cutoff)
    return result


@njit(cache=True)
def capture_1d_nb(returns, factor_returns, ann_factor):
    """See `empyrical.capture`."""
    annualized_return1 = annualized_return_1d_nb(returns, ann_factor)
    annualized_return2 = annualized_return_1d_nb(factor_returns, ann_factor)
    if annualized_return2 == 0.:
        return np.nan
    return annualized_return1 / annualized_return2


@njit(cache=True)
def capture_nb(returns, factor_returns, ann_factor):
    """2-dim version of `capture_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=f8)
    for col in range(returns.shape[1]):
        result[col] = capture_1d_nb(returns[:, col], factor_returns[:, col], ann_factor)
    return result


@njit(cache=True)
def up_capture_1d_nb(returns, factor_returns, ann_factor):
    """See `empyrical.up_capture`."""
    returns = returns[factor_returns > 0]
    factor_returns = factor_returns[factor_returns > 0]
    annualized_return1 = annualized_return_1d_nb(returns, ann_factor)
    annualized_return2 = annualized_return_1d_nb(factor_returns, ann_factor)
    if annualized_return2 == 0.:
        return np.nan
    return annualized_return1 / annualized_return2


@njit(cache=True)
def up_capture_nb(returns, factor_returns, ann_factor):
    """2-dim version of `up_capture_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=f8)
    for col in range(returns.shape[1]):
        result[col] = up_capture_1d_nb(returns[:, col], factor_returns[:, col], ann_factor)
    return result


@njit(cache=True)
def down_capture_1d_nb(returns, factor_returns, ann_factor):
    """See `empyrical.down_capture`."""
    returns = returns[factor_returns < 0]
    factor_returns = factor_returns[factor_returns < 0]
    annualized_return1 = annualized_return_1d_nb(returns, ann_factor)
    annualized_return2 = annualized_return_1d_nb(factor_returns, ann_factor)
    if annualized_return2 == 0.:
        return np.nan
    return annualized_return1 / annualized_return2


@njit(cache=True)
def down_capture_nb(returns, factor_returns, ann_factor):
    """2-dim version of `down_capture_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=f8)
    for col in range(returns.shape[1]):
        result[col] = down_capture_1d_nb(returns[:, col], factor_returns[:, col], ann_factor)
    return result

