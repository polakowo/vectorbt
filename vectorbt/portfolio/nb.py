"""Numba-compiled 1-dim and 2-dim functions.

!!! note
    `vectorbt` treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function. 
    Data is processed along index (axis 0).
    
    All functions passed as argument must be Numba-compiled.
    
    Records must remain in the order they were created."""

import numpy as np
from numba import njit, f8

from vectorbt.portfolio.enums import (
    Order,
    FilledOrder
)
from vectorbt.records.enums import (
    OrderSide,
    order_dt
)

# ############# Simulation ############# #


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

        # Update current cash and shares
        run_cash -= adj_size * adj_price + fees_paid
        run_shares += adj_size
    else:
        # Insufficient cash, size will be less than requested
        # For fees of 10%, you can buy shares for 90.9$ (adj_cash) to spend 100$ (run_cash) in total
        adj_cash = (run_cash - order.fixed_fees) / (1 + order.fees)
        if adj_cash <= 0.:
            # Can't cover
            return run_cash, run_shares, None

        # Update size and feee
        adj_size = adj_cash / adj_price
        fees_paid = run_cash - adj_cash

        # Update current cash and shares
        run_cash = 0.  # numerical stability
        run_shares += adj_size

    # Return filled order
    return run_cash, run_shares, FilledOrder(adj_size, adj_price, fees_paid, OrderSide.Buy)


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
    if adj_cash <= 0.:
        # Can't cover
        return run_cash, run_shares, None

    # Update fees
    fees_paid = cash - adj_cash

    # Update current cash and shares
    run_cash += adj_size * adj_price - fees_paid
    if run_shares <= abs(order.size):
        run_shares = 0.  # numerical stability
    else:
        run_shares -= adj_size
    return run_cash, run_shares, FilledOrder(adj_size, adj_price, fees_paid, OrderSide.Sell)


@njit(cache=True)
def fill_order_nb(run_cash, run_shares, order):
    """Fill an order."""
    if order.size != 0.:
        if order.price <= 0.:
            raise Exception("Price must be greater than zero")
        if order.fees < 0.:
            raise Exception("Fees must be zero or greater")
        if order.fixed_fees < 0.:
            raise Exception("Fixed fees must be zero or greater")
        if order.slippage < 0.:
            raise Exception("Slippage must be zero or greater")
        if order.size > 0.:
            return buy_nb(run_cash, run_shares, order)
        if order.size < 0.:
            return sell_nb(run_cash, run_shares, order)
    return run_cash, run_shares, None


@njit
def simulate_nb(target_shape, init_capital, order_func_nb, *args):
    """Simulate a portfolio by generating and filling orders.

    Starting with initial capital `init_capital`, iterates over shape `target_shape`, 
    and for each data point, generates an order using `order_func_nb`. Tries then to 
    fulfill that order. If unsuccessful due to insufficient cash/shares, always orders 
    the available fraction. Updates then the current cash and shares balance.

    Returns order records of layout `vectorbt.records.enums.order_dt`, but also
    cash and shares as time series.

    `order_func_nb` must accept index of the current column `col`, the time step `i`,
    the amount of cash `run_cash` and shares `run_shares` held at the time step `i`, and `*args`.
    Must either return an `vectorbt.portfolio.enums.Order` tuple or `None` to do nothing.

    !!! warning
        In some cases, passing large arrays as `*args` can negatively impact performance. What can help
        is accessing arrays from `order_func_nb` as non-local variables as we do in the example below.

    Example:
        Simulate a basic buy-and-hold strategy:
        ```python-repl
        >>> import numpy as np
        >>> import pandas as pd
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
        >>> init_capital = np.full(3, 100)
        >>> fees = 0.001
        >>> fixed_fees = 1
        >>> slippage = 0.001
        >>> @njit
        ... def order_func_nb(col, i, run_cash, run_shares):
        ...     return Order(np.inf if i == 0 else 0, price[i, col], 
        ...         fees=fees, fixed_fees=fixed_fees, slippage=slippage)
        >>> order_records, cash, shares = simulate_nb(
        ...     price.shape, init_capital, order_func_nb)

        >>> print(pd.DataFrame.from_records(order_records))
           col  idx       size  price      fees  side
        0    0    0  98.802297  1.001  1.098901     0
        1    1    0  19.760459  5.005  1.098901     0
        2    2    0  98.802297  1.001  1.098901     0
        >>> print(cash)
        [[0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]]
        >>> print(shares)
        [[98.8022966  19.76045932 98.8022966 ]
         [98.8022966  19.76045932 98.8022966 ]
         [98.8022966  19.76045932 98.8022966 ]
         [98.8022966  19.76045932 98.8022966 ]
         [98.8022966  19.76045932 98.8022966 ]]
        ```
    """
    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    j = 0
    cash = np.empty(target_shape, dtype=f8)
    shares = np.empty(target_shape, dtype=f8)

    for col in range(target_shape[1]):
        run_cash = init_capital[col]
        run_shares = 0.

        for i in range(target_shape[0]):
            # Generate the next oder or None to do nothing
            order = order_func_nb(col, i, run_cash, run_shares, *args)

            if order is not None:
                # Fill the order
                run_cash, run_shares, filled_order = fill_order_nb(run_cash, run_shares, order)

                # Add a new record
                if filled_order is not None:
                    order_records[j]['col'] = col
                    order_records[j]['idx'] = i
                    order_records[j]['size'] = filled_order.size
                    order_records[j]['price'] = filled_order.price
                    order_records[j]['fees'] = filled_order.fees
                    order_records[j]['side'] = filled_order.side
                    j += 1

            # Populate cash and shares
            cash[i, col], shares[i, col] = run_cash, run_shares

    return order_records[:j], cash, shares


@njit(cache=True)
def simulate_from_signals_nb(target_shape, init_capital, entries, exits, size, entry_price,
                             exit_price, fees, fixed_fees, slippage, accumulate):
    """Adaptation of `simulate_nb` for simulation based on entry and exit signals."""
    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    j = 0
    cash = np.empty(target_shape, dtype=f8)
    shares = np.empty(target_shape, dtype=f8)

    for col in range(target_shape[1]):
        run_cash = init_capital[col]
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
                    run_cash, run_shares, filled_order = fill_order_nb(run_cash, run_shares, order)

                    # Add a new record
                    if filled_order is not None:
                        order_records[j]['col'] = col
                        order_records[j]['idx'] = i
                        order_records[j]['size'] = filled_order.size
                        order_records[j]['price'] = filled_order.price
                        order_records[j]['fees'] = filled_order.fees
                        order_records[j]['side'] = filled_order.side
                        j += 1

            # Populate cash and shares
            cash[i, col], shares[i, col] = run_cash, run_shares

    return order_records[:j], cash, shares


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
        if run_shares > 0. and not accumulate:
            # If accumulation is turned off, sell everything
            order_size = -np.inf
            order_price = exit_price
        elif run_shares > 0. and accumulate:
            # If accumulation is turned on, sell size
            order_size = -abs(size)
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
    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    j = 0
    cash = np.empty(target_shape, dtype=f8)
    shares = np.empty(target_shape, dtype=f8)

    for col in range(target_shape[1]):
        run_cash = init_capital[col]
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
                run_cash, run_shares, filled_order = fill_order_nb(run_cash, run_shares, order)

                # Add a new record
                if filled_order is not None:
                    order_records[j]['col'] = col
                    order_records[j]['idx'] = i
                    order_records[j]['size'] = filled_order.size
                    order_records[j]['price'] = filled_order.price
                    order_records[j]['fees'] = filled_order.fees
                    order_records[j]['side'] = filled_order.side
                    j += 1

            # Populate cash and shares
            cash[i, col], shares[i, col] = run_cash, run_shares

    return order_records[:j], cash, shares


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

