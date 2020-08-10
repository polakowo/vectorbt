"""Numba-compiled 1-dim and 2-dim functions.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    
    All functions passed as argument must be Numba-compiled.
    
    Records must remain the order they were created in."""

import numpy as np
from numba import njit

from vectorbt.utils.math import is_close_or_less_nb
from vectorbt.base.reshape_fns import flex_choose_i_and_col_nb, flex_select_nb
from vectorbt.portfolio.enums import (
    OrderContext,
    RowContext,
    SizeType,
    Order,
    FilledOrder
)
from vectorbt.records.enums import (
    OrderSide,
    order_dt
)


# ############# Simulation ############# #

@njit(cache=True)
def buy_in_cash_nb(run_cash, run_shares, order_price, order_cash, order_fees, order_fixed_fees, order_slippage):
    """Buy shares for `order_cash` cash.

    Returns an updated cash and shares balance, and `vectorbt.portfolio.enums.FilledOrder`."""
    # Get offered cash
    offer_cash = min(run_cash, order_cash)

    # Get effective cash by subtracting costs
    if is_close_or_less_nb(offer_cash, order_fixed_fees):
        # Can't cover
        return run_cash, run_shares, None
    effect_cash = (offer_cash - order_fixed_fees) * (1 - order_fees)

    # Get price adjusted with slippage
    adj_price = order_price * (1 + order_slippage)

    # Get final size for that cash
    final_size = effect_cash / adj_price

    # Get paid fees
    fees_paid = offer_cash - effect_cash

    # Update current cash and shares
    run_shares += final_size
    if is_close_or_less_nb(run_cash, order_cash):
        run_cash = 0.  # numerical stability
    else:
        run_cash -= offer_cash
    return run_cash, run_shares, FilledOrder(final_size, adj_price, fees_paid, OrderSide.Buy)


@njit(cache=True)
def sell_in_cash_nb(run_cash, run_shares, order_price, order_cash, order_fees, order_fixed_fees, order_slippage):
    """Sell shares for `order_cash` cash.

    Returns an updated cash and shares balance, and `vectorbt.portfolio.enums.FilledOrder`."""
    # Get price adjusted with slippage
    adj_price = order_price * (1 - order_slippage)

    # Get value required to complete this order
    req_value = (order_cash + order_fixed_fees) / (1 - order_fees)

    # Translate this to shares
    req_shares = req_value / adj_price

    if is_close_or_less_nb(req_shares, run_shares):
        # Sufficient shares
        final_size = req_shares
        fees_paid = req_value - order_cash

        # Update current cash and shares
        run_cash += order_cash
        run_shares -= req_shares
    else:
        # Insufficient shares, cash will be less than requested
        final_size = run_shares
        acq_cash = final_size * adj_price

        # Get final cash by subtracting costs
        final_cash = acq_cash * (1 - order_fees)
        if is_close_or_less_nb(run_cash + final_cash, order_fixed_fees):
            # Can't cover
            return run_cash, run_shares, None
        final_cash -= order_fixed_fees

        # Update fees
        fees_paid = acq_cash - final_cash

        # Update current cash and shares
        run_cash += final_cash
        run_shares = 0.
    return run_cash, run_shares, FilledOrder(final_size, adj_price, fees_paid, OrderSide.Sell)


@njit(cache=True)
def buy_in_shares_nb(run_cash, run_shares, order_price, order_size, order_fees, order_fixed_fees, order_slippage):
    """Buy `order_size` shares.

    Returns an updated cash and shares balance, and `vectorbt.portfolio.enums.FilledOrder`."""

    # Get price adjusted with slippage
    adj_price = order_price * (1 + order_slippage)

    # Get cash required to complete this order
    req_cash = order_size * adj_price
    adj_req_cash = req_cash * (1 + order_fees) + order_fixed_fees

    if is_close_or_less_nb(adj_req_cash, run_cash):
        # Sufficient cash
        final_size = order_size
        fees_paid = adj_req_cash - req_cash

        # Update current cash and shares
        run_cash -= order_size * adj_price + fees_paid
        run_shares += final_size
    else:
        # Insufficient cash, size will be less than requested
        if is_close_or_less_nb(run_cash, order_fixed_fees):
            # Can't cover
            return run_cash, run_shares, None

        # For fees of 10% and 1$ per transaction, you can buy shares for 90$ (effect_cash)
        # to spend 100$ (adj_req_cash) in total
        effect_cash = (run_cash - order_fixed_fees) / (1 + order_fees)

        # Update size and fees
        final_size = effect_cash / adj_price
        fees_paid = run_cash - effect_cash

        # Update current cash and shares
        run_cash = 0.  # numerical stability
        run_shares += final_size

    # Return filled order
    return run_cash, run_shares, FilledOrder(final_size, adj_price, fees_paid, OrderSide.Buy)


@njit(cache=True)
def sell_in_shares_nb(run_cash, run_shares, order_price, order_size, order_fees, order_fixed_fees, order_slippage):
    """Sell `order_size` shares.

    Returns an updated cash and shares balance, and `vectorbt.portfolio.enums.FilledOrder`."""

    # Get price adjusted with slippage
    adj_price = order_price * (1 - order_slippage)

    # Compute acquired cash
    final_size = min(run_shares, order_size)
    acq_cash = final_size * adj_price

    # Get final cash by subtracting costs
    final_cash = acq_cash * (1 - order_fees)
    if is_close_or_less_nb(run_cash + final_cash, order_fixed_fees):
        # Can't cover
        return run_cash, run_shares, None
    final_cash -= order_fixed_fees

    # Update fees
    fees_paid = acq_cash - final_cash

    # Update current cash and shares
    run_cash += final_size * adj_price - fees_paid
    if is_close_or_less_nb(run_shares, order_size):
        run_shares = 0.  # numerical stability
    else:
        run_shares -= final_size
    return run_cash, run_shares, FilledOrder(final_size, adj_price, fees_paid, OrderSide.Sell)


@njit(cache=True)
def fill_order_nb(run_cash, run_shares, order):
    """Fill an order."""
    if not np.isnan(order.size) and not np.isnan(order.price):
        if order.price <= 0.:
            raise ValueError("Price must be greater than zero")
        if np.isnan(order.fees) or order.fees < 0.:
            raise ValueError("Fees must be zero or greater")
        if np.isnan(order.fixed_fees) or order.fixed_fees < 0.:
            raise ValueError("Fixed fees must be zero or greater")
        if np.isnan(order.slippage) or order.slippage < 0.:
            raise ValueError("Slippage must be zero or greater")
        if order.size_type == SizeType.Shares \
                or order.size_type == SizeType.TargetShares \
                or order.size_type == SizeType.TargetValue \
                or order.size_type == SizeType.TargetPercent:
            size = order.size
            if order.size_type == SizeType.TargetShares:
                # order.size contains target amount of shares
                size = order.size - run_shares
            elif order.size_type == SizeType.TargetValue:
                # order.size contains value in monetary units of the asset
                target_value = order.size
                current_value = run_shares * order.price
                size = (target_value - current_value) / order.price
            elif order.size_type == SizeType.TargetPercent:
                # order.size contains percentage from current portfolio value
                target_perc = order.size
                current_value = run_shares * order.price
                current_total_value = run_cash + current_value
                target_value = target_perc * current_total_value
                size = (target_value - current_value) / order.price
            if size > 0. and run_cash > 0.:
                return buy_in_shares_nb(
                    run_cash,
                    run_shares,
                    order.price,
                    size,
                    order.fees,
                    order.fixed_fees,
                    order.slippage
                )
            if size < 0. and run_shares > 0.:
                return sell_in_shares_nb(
                    run_cash,
                    run_shares,
                    order.price,
                    abs(size),
                    order.fees,
                    order.fixed_fees,
                    order.slippage
                )
        else:
            cash = order.size
            if order.size_type == SizeType.TargetCash:
                # order.size contains target amount of cash
                cash = run_cash - order.size
            if cash > 0. and run_cash > 0.:
                return buy_in_cash_nb(
                    run_cash,
                    run_shares,
                    order.price,
                    cash,
                    order.fees,
                    order.fixed_fees,
                    order.slippage
                )
            if cash < 0. and run_shares > 0.:
                return sell_in_cash_nb(
                    run_cash,
                    run_shares,
                    order.price,
                    abs(cash),
                    order.fees,
                    order.fixed_fees,
                    order.slippage
                )
    return run_cash, run_shares, None


@njit
def simulate_nb(target_shape, init_capital, order_func_nb, *args):
    """Simulate a portfolio by iterating over columns and generating and filling orders.

    Starting with initial capital `init_capital`, iterates over each column in shape `target_shape`,
    and for each data point, generates an order using `order_func_nb`. Tries then to fulfill that
    order. If unsuccessful due to insufficient cash/shares, orders the available fraction.
    Updates then the current cash and shares balance.

    Returns order records of layout `vectorbt.records.enums.order_dt`, but also
    cash and shares as time series.

    `order_func_nb` must accept the current order context `vectorbt.portfolio.enums.OrderContext`,
    and `*args`. Should either return an `vectorbt.portfolio.enums.Order` tuple or `None` to do nothing.

    !!! note
        This function assumes that all columns are independent of each other. Since iteration
        happens over columns, all columns next to the current one will be empty. Accessing
        these columns will not trigger any errors or warnings, but provide you with arbitrary data
        (see [numpy.empty](https://numpy.org/doc/stable/reference/generated/numpy.empty.html)).

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
        >>> from vectorbt.portfolio.enums import Order, SizeType

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
        ... def order_func_nb(oc):
        ...     return Order(np.inf if oc.i == 0 else 0, SizeType.Shares,
        ...         price[oc.i, oc.col], fees, fixed_fees, slippage)
        >>> order_records, cash, shares = simulate_nb(
        ...     price.shape, init_capital, order_func_nb)

        >>> pd.DataFrame.from_records(order_records)
           col  idx       size  price      fees  side
        0    0    0  98.802297  1.001  1.098901     0
        1    1    0  19.760459  5.005  1.098901     0
        2    2    0  98.802297  1.001  1.098901     0
        >>> cash
        [[0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]]
        >>> shares
        [[98.8022966  19.76045932 98.8022966 ]
         [98.8022966  19.76045932 98.8022966 ]
         [98.8022966  19.76045932 98.8022966 ]
         [98.8022966  19.76045932 98.8022966 ]
         [98.8022966  19.76045932 98.8022966 ]]
        ```
    """
    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    j = 0
    cash = np.empty(target_shape, dtype=np.float_)
    shares = np.empty(target_shape, dtype=np.float_)

    for col in range(target_shape[1]):
        run_cash = float(flex_select_nb(0, col, init_capital, is_2d=True))
        run_shares = 0.

        for i in range(target_shape[0]):
            # Generate the next order or None to do nothing
            order_context = OrderContext(
                col, i,
                target_shape,
                init_capital,
                order_records[:j],
                cash,
                shares,
                run_cash,
                run_shares
            )
            order = order_func_nb(order_context, *args)

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
def none_row_prep_func_nb(row_context, *args):
    """`row_prep_func_nb` that returns an empty tuple."""
    return ()


@njit
def simulate_row_wise_nb(target_shape, init_capital, row_prep_func_nb, order_func_nb, *args):
    """Simulate a portfolio by iterating over rows and generating and filling orders.

    As opposed to `simulate_nb`, iterates using C-like index order, with the rows
    changing fastest, and the columns changing slowest.

    `row_prep_func_nb` must accept the current row context `vectorbt.portfolio.enums.RowContext`,
    and `*args`. Should return a tuple of any content.

    `order_func_nb` must accept the current order context `vectorbt.portfolio.enums.OrderContext`,
    unpacked result of `row_prep_func_nb`, and `*args`. Should either return an
    `vectorbt.portfolio.enums.Order` tuple or `None` to do nothing.

    !!! note
        This function allows sharing information between columns. This allows complex logic
        such as rebalancing.

    Example:
        Simulate random rebalancing. Note, however, that columns do not share the same capital.
        ```python-repl
        >>> import numpy as np
        >>> import pandas as pd
        >>> from numba import njit
        >>> from vectorbt.portfolio.nb import simulate_row_wise_nb
        >>> from vectorbt.portfolio.enums import Order, SizeType

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
        ... def row_prep_func_nb(rc):
        ...     np.random.seed(rc.i)
        ...     w = np.random.uniform(0, 1, size=rc.target_shape[1])
        ...     return (w / np.sum(w),)

        >>> @njit
        ... def order_func_nb(oc, w):
        ...     current_value = oc.run_cash / price[oc.i, oc.col] + oc.run_shares
        ...     target_size = w[oc.col] * current_value
        ...     return Order(target_size - oc.run_shares, SizeType.Shares,
        ...         price[oc.i, oc.col], fees, fixed_fees, slippage)

        >>> order_records, cash, shares = simulate_row_wise_nb(
        ...     price.shape, init_capital, row_prep_func_nb, order_func_nb)

        >>> pd.DataFrame.from_records(order_records)
            col  idx       size  price      fees  side
        0     0    0  29.399155  1.001  1.029429     0
        1     0    1   5.872746  1.998  1.011734     1
        2     0    2   1.855144  2.997  1.005560     1
        3     0    3   6.433713  3.996  1.025709     1
        4     0    4   0.796768  4.995  1.003980     1
        5     1    0   7.662334  5.005  1.038350     0
        6     1    1   6.785973  4.004  1.027171     0
        7     1    2  13.801094  2.997  1.041362     1
        8     1    3  16.265081  2.002  1.032563     0
        9     1    4   4.578725  0.999  1.004574     1
        10    2    0  32.289173  1.001  1.032321     0
        11    2    1  32.282575  1.998  1.064501     1
        12    2    2  23.557854  3.003  1.070744     0
        13    2    3  13.673091  1.998  1.027319     1
        14    2    4  27.049616  1.001  1.027077     0
        >>> cash
        [[ 69.5420172   60.61166607  66.64621673]
         [ 80.26402911  32.41346024 130.08230128]
         [ 84.81833559  72.73397843  58.26732111]
         [109.50174358  39.13872328  84.55883717]
         [112.47761836  42.70829586  56.45509534]]
        >>> shares
        [[2.93991551e+01 7.66233445e+00 3.22891726e+01]
         [2.35264095e+01 1.44483072e+01 6.59749521e-03]
         [2.16712656e+01 6.47212729e-01 2.35644516e+01]
         [1.52375526e+01 1.69122939e+01 9.89136108e+00]
         [1.44407849e+01 1.23335684e+01 3.69409766e+01]]
        ```
    """
    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    j = 0
    cash = np.empty(target_shape, dtype=np.float_)
    shares = np.empty(target_shape, dtype=np.float_)

    for i in range(target_shape[0]):
        # Run a row preparation function and pass the result to each order function
        row_context = RowContext(
            i,
            target_shape,
            init_capital,
            order_records[:j],  # not sorted!
            cash,
            shares
        )
        prep_result = row_prep_func_nb(row_context, *args)

        for col in range(target_shape[1]):
            if i == 0:
                run_cash = float(flex_select_nb(0, col, init_capital, is_2d=True))
                run_shares = 0.
            else:
                run_cash = cash[i - 1, col]
                run_shares = shares[i - 1, col]

            # Generate the next order or None to do nothing
            order_context = OrderContext(
                col, i,
                target_shape,
                init_capital,
                order_records[:j],  # not sorted!
                cash,
                shares,
                run_cash,
                run_shares
            )
            order = order_func_nb(order_context, *prep_result, *args)

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

    # Order records are not sorted yet
    order_records = order_records[:j]
    return order_records[np.argsort(order_records['col'])], cash, shares


@njit(cache=True)
def simulate_from_signals_nb(target_shape, init_capital, entries, exits, size, size_type, entry_price,
                             exit_price, fees, fixed_fees, slippage, accumulate, is_2d=False):
    """Adaptation of `simulate_nb` for simulation based on entry and exit signals."""
    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    j = 0
    cash = np.empty(target_shape, dtype=np.float_)
    shares = np.empty(target_shape, dtype=np.float_)

    # Inputs were not broadcasted -> use flexible indexing
    i1, col1 = flex_choose_i_and_col_nb(entries, is_2d=is_2d)
    i2, col2 = flex_choose_i_and_col_nb(exits, is_2d=is_2d)
    i3, col3 = flex_choose_i_and_col_nb(size, is_2d=is_2d)
    i4, col4 = flex_choose_i_and_col_nb(size_type, is_2d=is_2d)
    i5, col5 = flex_choose_i_and_col_nb(entry_price, is_2d=is_2d)
    i6, col6 = flex_choose_i_and_col_nb(exit_price, is_2d=is_2d)
    i7, col7 = flex_choose_i_and_col_nb(fees, is_2d=is_2d)
    i8, col8 = flex_choose_i_and_col_nb(fixed_fees, is_2d=is_2d)
    i9, col9 = flex_choose_i_and_col_nb(slippage, is_2d=is_2d)

    for col in range(target_shape[1]):
        run_cash = float(flex_select_nb(0, col, init_capital, is_2d=is_2d))
        run_shares = 0.

        for i in range(target_shape[0]):
            # Generate the next oder or None to do nothing
            is_entry = flex_select_nb(i, col, entries, def_i=i1, def_col=col1, is_2d=is_2d)
            is_exit = flex_select_nb(i, col, exits, def_i=i2, def_col=col2, is_2d=is_2d)
            if is_entry or is_exit:
                order = signals_order_func_nb(
                    run_shares,
                    run_cash,
                    is_entry,
                    is_exit,
                    flex_select_nb(i, col, size, def_i=i3, def_col=col3, is_2d=is_2d),
                    flex_select_nb(i, col, size_type, def_i=i4, def_col=col4, is_2d=is_2d),
                    flex_select_nb(i, col, entry_price, def_i=i5, def_col=col5, is_2d=is_2d),
                    flex_select_nb(i, col, exit_price, def_i=i6, def_col=col6, is_2d=is_2d),
                    flex_select_nb(i, col, fees, def_i=i7, def_col=col7, is_2d=is_2d),
                    flex_select_nb(i, col, fixed_fees, def_i=i8, def_col=col8, is_2d=is_2d),
                    flex_select_nb(i, col, slippage, def_i=i9, def_col=col9, is_2d=is_2d),
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
def signals_order_func_nb(run_shares, run_cash, entries, exits, size, size_type, entry_price,
                          exit_price, fees, fixed_fees, slippage, accumulate):
    """`order_func_nb` of `simulate_from_signals_nb`."""
    if size_type != SizeType.Shares and size_type != SizeType.Cash:
        raise ValueError("Only SizeType.Shares and SizeType.Cash are supported")
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
        if size_type == SizeType.Shares:
            order_size = abs(size) - run_shares
        if size_type == SizeType.Cash:
            order_size = run_cash - abs(size)
        if order_size > 0:
            order_price = entry_price
        elif order_size < 0:
            order_price = exit_price
        else:
            return None
    else:
        return None
    return Order(order_size, size_type, order_price, fees, fixed_fees, slippage)


@njit(cache=True)
def simulate_from_orders_nb(target_shape, init_capital, size, size_type, price,
                            fees, fixed_fees, slippage, is_2d=False):
    """Adaptation of `simulate_nb` for simulation based on orders."""
    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    j = 0
    cash = np.empty(target_shape, dtype=np.float_)
    shares = np.empty(target_shape, dtype=np.float_)

    # Inputs were not broadcasted -> use flexible indexing
    i1, col1 = flex_choose_i_and_col_nb(size, is_2d=is_2d)
    i2, col2 = flex_choose_i_and_col_nb(size_type, is_2d=is_2d)
    i3, col3 = flex_choose_i_and_col_nb(price, is_2d=is_2d)
    i4, col4 = flex_choose_i_and_col_nb(fees, is_2d=is_2d)
    i5, col5 = flex_choose_i_and_col_nb(fixed_fees, is_2d=is_2d)
    i6, col6 = flex_choose_i_and_col_nb(slippage, is_2d=is_2d)

    for col in range(target_shape[1]):
        run_cash = float(flex_select_nb(0, col, init_capital, is_2d=is_2d))
        run_shares = 0.

        for i in range(target_shape[0]):
            # Generate the next oder or None to do nothing
            order = Order(
                flex_select_nb(i, col, size, def_i=i1, def_col=col1, is_2d=is_2d),
                flex_select_nb(i, col, size_type, def_i=i2, def_col=col2, is_2d=is_2d),
                flex_select_nb(i, col, price, def_i=i3, def_col=col3, is_2d=is_2d),
                flex_select_nb(i, col, fees, def_i=i4, def_col=col4, is_2d=is_2d),
                flex_select_nb(i, col, fixed_fees, def_i=i5, def_col=col5, is_2d=is_2d),
                flex_select_nb(i, col, slippage, def_i=i6, def_col=col6, is_2d=is_2d),
            )

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

