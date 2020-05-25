"""Numba-compiled 1-dim and 2-dim functions.

!!! note
    `vectorbt` treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function. 
    Data is processed along index (axis 0).
    
    All functions passed as argument must be Numba-compiled."""

import numpy as np
from numba import njit, b1, i1, i8, f8
from numba.core.types import UniTuple

from vectorbt import timeseries
from vectorbt.portfolio.enums import TradeType, PositionStatus, Order, Trade, Position

# ############# Portfolio ############# #


@njit(cache=True)
def buy_nb(run_cash, run_shares, order):
    """Perform a Buy.

    Returns an updated cash and shares balance, the number of shares bought, 
    the price adjusted with slippage, and fees paid."""

    # Compute cash required to complete this order
    adj_price = order.price * (1 + order.slippage)
    req_cash = order.size * adj_price
    adj_req_cash = req_cash * (1 + order.fees) + order.fixed_fees

    if adj_req_cash <= run_cash:
        # Sufficient cash
        adj_size = order.size
        fees_paid = adj_req_cash - req_cash
    else:
        # Insufficient cash, size will be less than requested
        # For fees of 10%, you can buy shares for 90.9$ (adj_cash) to spend 100$ (run_cash) in total
        adj_cash = (run_cash - order.fixed_fees) / (1 + order.fees)

        # Update size and fees
        adj_size = adj_cash / adj_price
        fees_paid = run_cash - adj_cash

    if adj_size > 0.:
        # Update current cash and shares
        run_cash -= adj_size * adj_price + fees_paid
        run_shares += adj_size
    return run_cash, run_shares, adj_size, adj_price, fees_paid


@njit(cache=True)
def sell_nb(run_cash, run_shares, order):
    """Perform a Sell.

    Returns an updated cash and shares balance, the number of shares sold, 
    the price adjusted with slippage, and fees paid."""

    # Compute acquired cash
    adj_price = order.price * (1 - order.slippage)
    adj_size = min(run_shares, abs(order.size))
    cash = adj_size * adj_price

    # Minus costs
    adj_cash = cash * (1 - order.fees) - order.fixed_fees

    # Update fees
    fees_paid = cash - adj_cash

    if adj_size > 0.:
        # Update current cash and shares
        run_cash += adj_size * adj_price - fees_paid
        run_shares -= adj_size
        adj_size *= -1
    return run_cash, run_shares, adj_size, adj_price, fees_paid


@njit(cache=True)
def fill_order_nb(run_cash, run_shares, order):
    """Fill an order."""
    if order.size > 0.:
        return buy_nb(run_cash, run_shares, order)
    if order.size < 0.:
        return sell_nb(run_cash, run_shares, order)
    return run_cash, run_shares, 0., 0., 0.


@njit
def simulate_nb(target_shape, init_capital, order_func_nb, *args):
    """Simulate a portfolio by generating and filling orders.

    Starting with initial capital `init_capital`, iterates over shape `target_shape`, 
    and for each data point, generates an order using `order_func_nb`. Tries then to 
    fulfill that order. If unsuccessful due to insufficient cash/shares, always orders 
    the available fraction. Updates then the current cash and shares balance.

    Returns trade size with direction, trade price, trade fees, cash and shares as time series.

    `order_func_nb` must accept index of the current column `col`, the time step `i`,
    the amount of cash `run_cash` and shares `run_shares` held at the time step `i`, and `*args`.
    Must either return an `Order` tuple or `None` to do nothing.

    !!! warning
        In some cases, passing large arrays as `*args` can negatively impact performance. What can help
        is accessing arrays from `order_func_nb` as non-local variables as we do in the example below.

    Example:
        ```python-repl
        >>> import numpy as np
        >>> from numba import njit
        >>> from vectorbt.portfolio.nb import simulate_nb
        >>> from vectorbt.portfolio.enums import Order

        >>> price = np.asarray([
        ...     [1, 5, 1],
        ...     [2, 4, 2],
        ...     [3, 3, 3],
        ...     [4, 2, 2],
        ...     [5, 1, 1]
        ... ])
        >>> fees = 0.001
        >>> fixed_fees = 1
        >>> slippage = 0.001
        >>> @njit
        ... def order_func_nb(col, i, run_cash, run_shares):
        ...     return Order(1 if i % 2 == 0 else -1, price[i, col], 
        ...         fees=fees, fixed_fees=fixed_fees, slippage=slippage)
        >>> trade_size, trade_price, trade_fees, cash, shares = \\
        ...     simulate_nb(price.shape, 100, order_func_nb)

        >>> print(trade_size)
        [[ 1.  1.  1.]
         [-1. -1. -1.]
         [ 1.  1.  1.]
         [-1. -1. -1.]
         [ 1.  1.  1.]]
        >>> print(trade_price)
        [[1.001 5.005 1.001]
         [1.998 3.996 1.998]
         [3.003 3.003 3.003]
         [3.996 1.998 1.998]
         [5.005 1.001 1.001]]
        >>> print(trade_fees)
        [[1.001001 1.005005 1.001001]
         [1.001998 1.003996 1.001998]
         [1.003003 1.003003 1.003003]
         [1.003996 1.001998 1.001998]
         [1.005005 1.001001 1.001001]]
        >>> print(cash)
        [[97.997999 93.989995 97.997999]
         [98.994001 96.981999 98.994001]
         [94.987998 92.975996 94.987998]
         [97.980002 93.971998 95.984   ]
         [91.969997 91.969997 93.981999]]
        >>> print(shares)
        [[1. 1. 1.]
         [0. 0. 0.]
         [1. 1. 1.]
         [0. 0. 0.]
         [1. 1. 1.]]
        ```
    """
    trade_size = np.full(target_shape, np.nan, dtype=f8)
    trade_price = np.full(target_shape, np.nan, dtype=f8)
    trade_fees = np.full(target_shape, np.nan, dtype=f8)
    cash = np.empty(target_shape, dtype=f8)
    shares = np.empty(target_shape, dtype=f8)

    for col in range(target_shape[1]):
        run_cash = init_capital
        run_shares = 0.

        for i in range(target_shape[0]):
            # Generate the next oder or None to do nothing
            order = order_func_nb(col, i, run_cash, run_shares, *args)
            if order is not None:
                # Fill the order
                run_cash, run_shares, adj_size, adj_price, fees_paid = fill_order_nb(run_cash, run_shares, order)

                # Update matrices
                if adj_size != 0.:
                    trade_size[i, col] = adj_size
                    trade_price[i, col] = adj_price
                    trade_fees[i, col] = fees_paid

            # Populate cash and shares
            cash[i, col], shares[i, col] = run_cash, run_shares

    return trade_size, trade_price, trade_fees, cash, shares


@njit(cache=True)
def simulate_from_signals_nb(target_shape, init_capital, entries, exits, size, entry_price,
                             exit_price, fees, fixed_fees, slippage, accumulate):
    """Adaptation of `simulate_nb` for simulation based on entry and exit signals."""
    trade_size = np.full(target_shape, np.nan, dtype=f8)
    trade_price = np.full(target_shape, np.nan, dtype=f8)
    trade_fees = np.full(target_shape, np.nan, dtype=f8)
    cash = np.empty(target_shape, dtype=f8)
    shares = np.empty(target_shape, dtype=f8)

    for col in range(target_shape[1]):
        run_cash = init_capital
        run_shares = 0.

        for i in range(target_shape[0]):
            # Generate the next oder or None to do nothing
            if entries[i, col] or exits[i, col]:
                order = signals_order_func_nb(
                    run_shares,
                    entries[i, col],
                    exits[i, col],
                    size[i, col],
                    entry_price[i, col],
                    exit_price[i, col],
                    fees[i, col],
                    fixed_fees[i, col],
                    slippage[i, col],
                    accumulate)
                if order is not None:
                    # Fill the order
                    run_cash, run_shares, adj_size, adj_price, fees_paid = fill_order_nb(run_cash, run_shares, order)

                    # Update matrices
                    if adj_size != 0.:
                        trade_size[i, col] = adj_size
                        trade_price[i, col] = adj_price
                        trade_fees[i, col] = fees_paid

            # Populate cash and shares
            cash[i, col], shares[i, col] = run_cash, run_shares

    return trade_size, trade_price, trade_fees, cash, shares


@njit(cache=True)
def signals_order_func_nb(run_shares, entries, exits, size, entry_price,
                          exit_price, fees, fixed_fees, slippage, accumulate):
    """`order_func_nb` of `simulate_from_signals_nb`."""
    if entries and not exits:
        # Buy the amount of shares specified in size (only once if not accumulate)
        if run_shares == 0. or accumulate:
            order_size = abs(size)
            order_price = entry_price
        else:
            return None
    elif not entries and exits:
        # Sell everything
        if run_shares > 0.:
            order_size = -np.inf
            order_price = exit_price
        else:
            return None
    elif entries and exits:
        # Buy the difference between entry and exit size
        order_size = abs(size) - run_shares
        if order_size > 0:
            order_price = entry_price
        elif order_size < 0:
            order_price = exit_price
        else:
            return None
    return Order(
        order_size,
        order_price,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage)


@njit(cache=True)
def simulate_from_orders_nb(target_shape, init_capital, size, price, fees, fixed_fees, slippage, is_target):
    """Adaptation of `simulate_nb` for simulation based on orders."""
    trade_size = np.full(target_shape, np.nan, dtype=f8)
    trade_price = np.full(target_shape, np.nan, dtype=f8)
    trade_fees = np.full(target_shape, np.nan, dtype=f8)
    cash = np.empty(target_shape, dtype=f8)
    shares = np.empty(target_shape, dtype=f8)

    for col in range(target_shape[1]):
        run_cash = init_capital
        run_shares = 0.

        for i in range(target_shape[0]):
            # Generate the next oder or None to do nothing
            order = size_order_func_nb(
                run_shares,
                size[i, col],
                price[i, col],
                fees[i, col],
                fixed_fees[i, col],
                slippage[i, col],
                is_target)
            if order is not None:
                # Fill the order
                run_cash, run_shares, adj_size, adj_price, fees_paid = fill_order_nb(run_cash, run_shares, order)

                # Update matrices
                if adj_size != 0.:
                    trade_size[i, col] = adj_size
                    trade_price[i, col] = adj_price
                    trade_fees[i, col] = fees_paid

            # Populate cash and shares
            cash[i, col], shares[i, col] = run_cash, run_shares

    return trade_size, trade_price, trade_fees, cash, shares


@njit(cache=True)
def size_order_func_nb(run_shares, size, price, fees, fixed_fees, slippage, is_target):
    """`order_func_nb` of `simulate_from_orders_nb`."""
    if is_target:
        order_size = size - run_shares
    else:
        order_size = size
    return Order(
        order_size,
        price,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage)

# ############# Trades ############# #


@njit
def map_trades_nb(price, trade_size, trade_price, trade_fees, map_func_nb, *args):
    """Map each trade to a value using `map_func_nb`. 

    The value is then stored at the index of the trade.

    `map_func_nb` must accept a position of type `vectorbt.portfolio.enums.Trade`.

    Example:
        Map each trade to its value:
        ```python-repl
        >>> import numpy as np
        >>> from numba import njit
        >>> from vectorbt.portfolio.nb import map_trades_nb

        >>> trade_price = np.asarray([1, 2, 3, 4, 5])[:, None]
        >>> trade_size = np.asarray([1, 0, -1, 1, 1])[:, None]
        >>> trade_fees = np.asarray([0, 0, 0, 0, 0])[:, None]
        >>> @njit
        ... def map_func_nb(trade):
        ...     return trade.size * trade.price

        >>> print(map_trades_nb(trade_size, trade_price, trade_fees, map_func_nb))
        [[ 1.]
         [nan]
         [ 3.]
         [ 4.]
         [ 5.]]
        ```"""
    result = np.full_like(trade_size, np.nan, dtype=f8)

    for col in range(trade_size.shape[1]):
        buy_size_sum = 0.
        buy_gross_sum = 0.
        buy_fees_sum = 0.
        position_idx = 0

        for i in range(trade_size.shape[0]):
            sig_size = trade_size[i, col]
            size = abs(sig_size)
            price = trade_price[i, col]
            fees = trade_fees[i, col]

            if ~np.isnan(sig_size) and sig_size != 0.:
                if sig_size > 0.:
                    # Position increased
                    if buy_size_sum == 0.:
                        position_start = i

                    buy_size_sum += size
                    buy_gross_sum += size * price
                    buy_fees_sum += fees

                    # There is no counterpart for a buy operation
                    open_i = np.nan
                    avg_buy_price = np.nan
                    frac_buy_fees = np.nan
                    trade_type = TradeType.Buy
                elif sig_size < 0.:
                    # Counterpart operation for sell is a buy
                    open_i = position_start
                    # Measure average buy price and fees
                    # A size-weighted average over all purchase prices
                    avg_buy_price = buy_gross_sum / buy_size_sum
                    # A size-weighted average over all purchase fees
                    frac_buy_fees = size / buy_size_sum * buy_fees_sum
                    trade_type = TradeType.Sell

                    # Position decreased, previous purchases have now less impact
                    size_fraction = (buy_size_sum - size) / buy_size_sum
                    buy_size_sum *= size_fraction
                    buy_gross_sum *= size_fraction
                    buy_fees_sum *= size_fraction

                # Map trade
                trade = Trade(
                    col=col,
                    size=size,
                    open_i=open_i,
                    open_price=avg_buy_price,
                    open_fees=frac_buy_fees,
                    close_i=i,
                    close_price=price,
                    close_fees=fees,
                    type=trade_type,
                    position_idx=position_idx)
                result[i, col] = map_func_nb(trade, *args)

                if buy_size_sum == 0.:
                    position_idx += 1
    return result


@njit(cache=True)
def trade_type_map_func_nb(trade):
    """`map_func_nb` that returns type of the trade.

    See `vectorbt.portfolio.enums.TradeType`."""
    return trade.type

@njit(cache=True)
def buy_filter_func_nb(col, i, value, trade_type):
    """`filter_func_nb` that includes only buy operations."""
    return trade_type[i, col] == TradeType.Buy


@njit(cache=True)
def sell_filter_func_nb(col, i, value, trade_type):
    """`filter_func_nb` that includes only sell operations."""
    return trade_type[i, col] == TradeType.Sell

# ############# Positions ############# #


@njit
def map_positions_nb(price, trade_size, trade_price, trade_fees, map_func_nb, *args):
    """Map each position to a value using `map_func_nb`. 

    The value is then stored at the last index of the position.

    `map_func_nb` must accept a position of type `vectorbt.portfolio.enums.Position`.

    Example:
        Map each closed position to its duration:
        ```python-repl
        >>> import numpy as np
        >>> from numba import njit
        >>> from vectorbt.portfolio.nb import map_positions_nb
        >>> from vectorbt.portfolio.enums import PositionStatus

        >>> trade_price = price = np.asarray([1, 2, 3, 4, 5])[:, None]
        >>> trade_size = np.asarray([1, 0, -1, 1, 1])[:, None]
        >>> trade_fees = np.asarray([0, 0, 0, 0, 0])[:, None]
        >>> @njit
        ... def map_func_nb(position):
        ...     if position.status == PositionStatus.Closed:
        ...          return position.close_i - position.open_i
        ...     return np.nan

        >>> print(map_positions_nb(price, trade_size, trade_price, trade_fees, map_func_nb))
        [[nan]
         [nan]
         [ 2.]
         [nan]
         [nan]]
        ```"""
    result = np.full_like(price, np.nan, dtype=f8)

    for col in range(price.shape[1]):
        buy_size_sum = 0.
        buy_gross_sum = 0.
        buy_fees_sum = 0.
        sell_size_sum = 0.
        sell_gross_sum = 0.
        sell_fees_sum = 0.
        store_position = False

        for i in range(trade_size.shape[0]):
            sig_tsize = trade_size[i, col]
            tsize = abs(sig_tsize)
            tprice = trade_price[i, col]
            tfees = trade_fees[i, col]

            if ~np.isnan(sig_tsize) and sig_tsize != 0.:
                if sig_tsize > 0.:
                    # Position increased
                    if buy_size_sum == 0.:
                        open_i = i

                    buy_size_sum += tsize
                    buy_gross_sum += tsize * tprice
                    buy_fees_sum += tfees

                elif sig_tsize < 0.:
                    # Position decreased
                    sell_size_sum += tsize
                    sell_gross_sum += tsize * tprice
                    sell_fees_sum += tfees

                if buy_size_sum == sell_size_sum:
                    # Closed position
                    status = PositionStatus.Closed
                    store_position = True

            if i == price.shape[0] - 1 and buy_size_sum > sell_gross_sum:
                # If position hasn't been closed, calculate its unrealized metrics
                sell_size_sum += buy_size_sum
                sell_gross_sum += buy_size_sum * price[i, col]
                # NOTE: We have no information about fees here, so we don't add them
                status = PositionStatus.Open
                store_position = True

            if store_position:
                # Calculate PnL and return
                avg_buy_price = buy_gross_sum / buy_size_sum
                avg_sell_price = sell_gross_sum / sell_size_sum

                # Map position
                position = Position(
                    col=col,
                    size=buy_size_sum,
                    open_i=open_i,
                    open_price=avg_buy_price,
                    open_fees=buy_fees_sum,
                    close_i=i,
                    close_price=avg_sell_price,
                    close_fees=sell_fees_sum,
                    status=status
                )
                result[i, col] = map_func_nb(position, *args)

                # Create a new position
                buy_size_sum = 0.
                buy_gross_sum = 0.
                buy_fees_sum = 0.
                sell_size_sum = 0.
                sell_gross_sum = 0.
                sell_fees_sum = 0.
                store_position = False
    return result


@njit(cache=True)
def pos_status_map_func_nb(position):
    """`map_func_nb` that returns status of the position.

    See `vectorbt.portfolio.enums.PositionStatus`."""
    return position.status


@njit(cache=True)
def open_filter_func_nb(col, i, value, status):
    """`filter_func_nb` that includes only open positions."""
    return status[i, col] == PositionStatus.Open


@njit(cache=True)
def closed_filter_func_nb(col, i, value, status):
    """`filter_func_nb` that includes only closed positions."""
    return status[i, col] == PositionStatus.Closed

# ############# Events ############# #


@njit(cache=True)
def event_duration_map_func_nb(position):
    """`map_func_nb` that returns duration of the event."""
    return position.close_i - position.open_i


@njit(cache=True)
def event_pnl_map_func_nb(trade):
    """`map_func_nb` that returns total PnL of the event."""
    buy_val = trade.size * trade.open_price + trade.open_fees
    sell_val = trade.size * trade.close_price - trade.close_fees
    return sell_val - buy_val


@njit(cache=True)
def event_return_map_func_nb(trade):
    """`map_func_nb` that returns total return of the event."""
    buy_val = trade.size * trade.open_price + trade.open_fees
    sell_val = trade.size * trade.close_price - trade.close_fees
    return (sell_val - buy_val) / buy_val


@njit(cache=True)
def winning_filter_func_nb(col, i, value, pnl):
    """`filter_func_nb` that includes only winning events."""
    return pnl[i, col] > 0


@njit(cache=True)
def losing_filter_func_nb(col, i, value, pnl):
    """`filter_func_nb` that includes only losing events."""
    return pnl[i, col] < 0

# ############# Appliers ############# #


@njit(cache=True)
def total_return_apply_func_nb(col, idxs, returns):
    """Calculate total return from returns."""
    return timeseries.nb.product_1d_nb(returns + 1) - 1

# ############# Accumulation ############# #


@njit(cache=True)
def is_accumulated_1d_nb(trade_size):
    """Detect accumulation, that is, position is being increased/decreased gradually."""
    buy_size_sum = 0.
    for i in range(trade_size.shape[0]):
        if trade_size[i] > 0.:
            buy_size_sum += trade_size[i]
            if buy_size_sum != trade_size[i]:
                return True
        elif trade_size[i] < 0.:
            buy_size_sum += trade_size[i]
            if buy_size_sum != 0.:
                return True
    return False


@njit(cache=True)
def is_accumulated_nb(trade_size):
    """2-dim version of `is_accumulated_1d_nb`."""
    result = np.empty(trade_size.shape[1], b1)
    for col in range(trade_size.shape[1]):
        result[col] = is_accumulated_1d_nb(trade_size[:, col])
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
