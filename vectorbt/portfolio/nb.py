"""Numba-compiled 1-dim and 2-dim functions.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    
    All functions passed as argument should be Numba-compiled.
    
    Records should remain the order they were created in."""

import numpy as np
from numba import njit

from vectorbt.utils.math import is_close_or_less_nb
from vectorbt.base.reshape_fns import flex_choose_i_and_col_nb, flex_select_nb
from vectorbt.portfolio.enums import (
    CallSeqType,
    OrderContext,
    GroupRowContext,
    RowContext,
    SizeType,
    AccumulateExitMode,
    ConflictMode,
    Order,
    NoOrder,
    OrderStatus,
    OrderResult,
    RejectedOrder
)
from vectorbt.records.enums import (
    OrderSide,
    order_dt
)


# ############# Simulation ############# #

@njit(cache=True)
def buy_in_cash_nb(cash_now, shares_now, order_price, order_cash, order_fees, order_fixed_fees, order_slippage):
    """Buy shares for `order_cash` cash.

    Returns an updated cash and shares balance, and `vectorbt.portfolio.enums.FilledOrder`."""
    # Get offered cash
    offer_cash = min(cash_now, order_cash)

    # Get effective cash by subtracting costs
    if is_close_or_less_nb(offer_cash, order_fixed_fees):
        # Can't cover
        return cash_now, shares_now, RejectedOrder
    effect_cash = (offer_cash - order_fixed_fees) * (1 - order_fees)

    # Get price adjusted with slippage
    adj_price = order_price * (1 + order_slippage)

    # Get final size for that cash
    final_size = effect_cash / adj_price

    # Get paid fees
    fees_paid = offer_cash - effect_cash

    # Update current cash and shares
    shares_now += final_size
    if is_close_or_less_nb(cash_now, order_cash):
        cash_now = 0.  # numerical stability
    else:
        cash_now -= offer_cash
    return cash_now, shares_now, OrderResult(final_size, adj_price, fees_paid, OrderSide.Buy, OrderStatus.Filled)


@njit(cache=True)
def sell_in_cash_nb(cash_now, shares_now, order_price, order_cash, order_fees, order_fixed_fees, order_slippage):
    """Sell shares for `order_cash` cash.

    Returns an updated cash and shares balance, and `vectorbt.portfolio.enums.FilledOrder`."""
    # Get price adjusted with slippage
    adj_price = order_price * (1 - order_slippage)

    # Get value required to complete this order
    req_value = (order_cash + order_fixed_fees) / (1 - order_fees)

    # Translate this to shares
    req_shares = req_value / adj_price

    if is_close_or_less_nb(req_shares, shares_now):
        # Sufficient shares
        final_size = req_shares
        fees_paid = req_value - order_cash

        # Update current cash and shares
        cash_now += order_cash
        shares_now -= req_shares
    else:
        # Insufficient shares, cash will be less than requested
        final_size = shares_now
        acq_cash = final_size * adj_price

        # Get final cash by subtracting costs
        final_cash = acq_cash * (1 - order_fees)
        if is_close_or_less_nb(cash_now + final_cash, order_fixed_fees):
            # Can't cover
            return cash_now, shares_now, RejectedOrder
        final_cash -= order_fixed_fees

        # Update fees
        fees_paid = acq_cash - final_cash

        # Update current cash and shares
        cash_now += final_cash
        shares_now = 0.
    return cash_now, shares_now, OrderResult(final_size, adj_price, fees_paid, OrderSide.Sell, OrderStatus.Filled)


@njit(cache=True)
def buy_in_shares_nb(cash_now, shares_now, order_price, order_size, order_fees, order_fixed_fees, order_slippage):
    """Buy `order_size` shares.

    Returns an updated cash and shares balance, and `vectorbt.portfolio.enums.FilledOrder`."""

    # Get price adjusted with slippage
    adj_price = order_price * (1 + order_slippage)

    # Get cash required to complete this order
    req_cash = order_size * adj_price
    adj_req_cash = req_cash * (1 + order_fees) + order_fixed_fees

    if is_close_or_less_nb(adj_req_cash, cash_now):
        # Sufficient cash
        final_size = order_size
        fees_paid = adj_req_cash - req_cash

        # Update current cash and shares
        cash_now -= order_size * adj_price + fees_paid
        shares_now += final_size
    else:
        # Insufficient cash, size will be less than requested
        if is_close_or_less_nb(cash_now, order_fixed_fees):
            # Can't cover
            return cash_now, shares_now, RejectedOrder

        # For fees of 10% and 1$ per transaction, you can buy shares for 90$ (effect_cash)
        # to spend 100$ (adj_req_cash) in total
        effect_cash = (cash_now - order_fixed_fees) / (1 + order_fees)

        # Update size and fees
        final_size = effect_cash / adj_price
        fees_paid = cash_now - effect_cash

        # Update current cash and shares
        cash_now = 0.  # numerical stability
        shares_now += final_size

    # Return filled order
    return cash_now, shares_now, OrderResult(final_size, adj_price, fees_paid, OrderSide.Buy, OrderStatus.Filled)


@njit(cache=True)
def sell_in_shares_nb(cash_now, shares_now, order_price, order_size, order_fees, order_fixed_fees, order_slippage):
    """Sell `order_size` shares.

    Returns an updated cash and shares balance, and `vectorbt.portfolio.enums.FilledOrder`."""

    # Get price adjusted with slippage
    adj_price = order_price * (1 - order_slippage)

    # Compute acquired cash
    final_size = min(shares_now, order_size)
    acq_cash = final_size * adj_price

    # Get final cash by subtracting costs
    final_cash = acq_cash * (1 - order_fees)
    if is_close_or_less_nb(cash_now + final_cash, order_fixed_fees):
        # Can't cover
        return cash_now, shares_now, RejectedOrder
    final_cash -= order_fixed_fees

    # Update fees
    fees_paid = acq_cash - final_cash

    # Update current cash and shares
    cash_now += final_size * adj_price - fees_paid
    if is_close_or_less_nb(shares_now, order_size):
        shares_now = 0.  # numerical stability
    else:
        shares_now -= final_size
    return cash_now, shares_now, OrderResult(final_size, adj_price, fees_paid, OrderSide.Sell, OrderStatus.Filled)


@njit(cache=True)
def process_order_nb(cash_now, shares_now, order):
    """Fill an order."""
    if not np.isnan(order.size) and not np.isnan(order.price):
        if not np.isfinite(order.price) or order.price <= 0.:
            raise ValueError("Price must be finite and greater than zero")
        if not np.isfinite(order.fees) or order.fees < 0.:
            raise ValueError("Fees must be finite and zero or greater")
        if not np.isfinite(order.fixed_fees) or order.fixed_fees < 0.:
            raise ValueError("Fixed fees must be finite and zero or greater")
        if not np.isfinite(order.slippage) or order.slippage < 0.:
            raise ValueError("Slippage must be finite and zero or greater")

        if order.size_type == SizeType.Shares \
                or order.size_type == SizeType.TargetShares \
                or order.size_type == SizeType.TargetValue \
                or order.size_type == SizeType.TargetPercent:
            size = order.size
            if order.size_type == SizeType.TargetShares:
                # order.size contains target amount of shares
                size = order.size - shares_now
            elif order.size_type == SizeType.TargetValue:
                # order.size contains value in monetary units of the asset
                target_value = order.size
                current_value = shares_now * order.price
                size = (target_value - current_value) / order.price
            elif order.size_type == SizeType.TargetPercent:
                # order.size contains percentage from current portfolio value
                target_perc = order.size
                current_value = shares_now * order.price
                current_total_value = cash_now + current_value
                target_value = target_perc * current_total_value
                size = (target_value - current_value) / order.price

            if size > 0. and cash_now > 0.:
                if np.isinf(order.size) and np.isinf(cash_now):
                    raise ValueError("Size and current cash cannot be both infinite")

                return buy_in_shares_nb(
                    cash_now,
                    shares_now,
                    order.price,
                    size,
                    order.fees,
                    order.fixed_fees,
                    order.slippage
                )
            if size < 0. and shares_now > 0.:
                if np.isinf(order.size) and np.isinf(shares_now):
                    raise ValueError("Size and current shares cannot be both infinite")

                return sell_in_shares_nb(
                    cash_now,
                    shares_now,
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
                cash = cash_now - order.size

            if cash > 0. and cash_now > 0.:
                if np.isinf(order.size) and np.isinf(cash_now):
                    raise ValueError("Size and current cash cannot be both infinite")

                return buy_in_cash_nb(
                    cash_now,
                    shares_now,
                    order.price,
                    cash,
                    order.fees,
                    order.fixed_fees,
                    order.slippage
                )
            if cash < 0. and shares_now > 0.:
                if np.isinf(order.size) and np.isinf(shares_now):
                    raise ValueError("Size and current shares cannot be both infinite")

                return sell_in_cash_nb(
                    cash_now,
                    shares_now,
                    order.price,
                    abs(cash),
                    order.fees,
                    order.fixed_fees,
                    order.slippage
                )
    return cash_now, shares_now, RejectedOrder


@njit(cache=True)
def check_group_counts(target_shape, group_counts):
    """Check `group_counts`."""
    if np.sum(group_counts) != target_shape[1]:
        raise ValueError("group_counts has incorrect total number of columns")


@njit(cache=True)
def check_group_init_cash(target_shape, group_counts, init_cash, cash_sharing):
    """Check `init_cash`."""
    if cash_sharing:
        if len(init_cash) != len(group_counts):
            raise ValueError("If cash sharing is enabled, init_cash must match the number of groups")
    else:
        if len(init_cash) != target_shape[1]:
            raise ValueError("If cash sharing is disabled, init_cash must match the number of columns")


@njit(cache=True)
def get_record_idx_nb(target_shape, col, i):
    """Get record index by position of order in the matrix."""
    return col * target_shape[0] + i


@njit(cache=True)
def is_grouped_nb(group_counts):
    """Check if columns are grouped, that is, more than one column per group."""
    return np.any(group_counts > 1)


@njit
def shuffle_call_seq_nb(call_seq, group_counts, seed=None):
    """Shuffle the call sequence array."""
    if seed is not None:
        np.random.seed(seed)
    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        for i in range(call_seq.shape[0]):
            np.random.shuffle(call_seq[i, from_col:to_col])
        from_col = to_col

@njit
def build_call_seq_nb(target_shape, group_counts, call_seq_type=CallSeqType.Default, seed=None):
    """Build a new call sequence array."""
    if call_seq_type == CallSeqType.Reversed:
        out = np.full(target_shape[1], 1, dtype=np.int_)
        out[np.cumsum(group_counts)[1:] - group_counts[1:] - 1] -= group_counts[1:]
        out = np.cumsum(out[::-1])[::-1] - 1
        out = out * np.ones((target_shape[0], 1), dtype=np.int_)
        return out
    out = np.full(target_shape[1], 1, dtype=np.int_)
    out[np.cumsum(group_counts)[:-1]] -= group_counts[:-1]
    out = np.cumsum(out) - 1
    out = out * np.ones((target_shape[0], 1), dtype=np.int_)
    if call_seq_type == CallSeqType.Random:
        shuffle_call_seq_nb(out, group_counts, seed=seed)
    return out


def require_call_seq(call_seq):
    """Force the call sequence array to pass our requirements."""
    return np.require(call_seq, dtype=np.int_, requirements=['A', 'O', 'W', 'F'])


def build_call_seq(target_shape, group_counts, call_seq_type=CallSeqType.Default, seed=None):
    """Not compiled but faster version of `build_call_seq_nb`."""
    call_seq = np.full(target_shape[1], 1, dtype=np.int_)
    if call_seq_type == CallSeqType.Reversed:
        call_seq[np.cumsum(group_counts)[1:] - group_counts[1:] - 1] -= group_counts[1:]
        call_seq = np.cumsum(call_seq[::-1])[::-1] - 1
    else:
        call_seq[np.cumsum(group_counts[:-1])] -= group_counts[:-1]
        call_seq = np.cumsum(call_seq) - 1
    call_seq = np.broadcast_to(call_seq, target_shape)
    if call_seq_type == CallSeqType.Random:
        call_seq = require_call_seq(call_seq)
        shuffle_call_seq_nb(call_seq, group_counts, seed=seed)
    return require_call_seq(call_seq)


@njit(cache=True)
def empty_row_prep_nb(rc, *args):
    """`row_prep_func_nb` that returns an empty tuple."""
    return ()


@njit(cache=True)
def default_call_seq_nb(grc, *args):
    """`call_seq_func_nb` that returns the default order from left to right."""
    return grc.call_seq[grc.i, grc.from_col:grc.to_col]


@njit
def simulate_nb(target_shape, group_counts, init_cash, cash_sharing, call_seq, active_mask,
                row_prep_func_nb, row_prep_args, call_seq_func_nb, call_seq_args,
                order_func_nb, *order_args):
    """Simulate a portfolio by generating and filling orders.

    As opposed to `simulate_row_wise_nb`, iterates using column-major order, with the columns
    changing fastest, and the row changing slowest.

    Starting with initial cash `init_cash`, iterates over each group and column over shape `target_shape`,
    and for each data point, generates an order using `order_func_nb`. Tries then to fulfill that
    order. If unsuccessful due to insufficient cash/shares, always orders the available fraction.
    Updates then the current cash and shares balance.

    Returns order records of layout `vectorbt.records.enums.order_dt`, but also cash and shares
    holding at each time step as time series.

    Args:
        target_shape (tuple): Target shape.

            A tuple with exactly two elements: the number of steps and columns.
        group_counts (np.ndarray): Column count per group.

            Even if columns are not grouped, `group_counts` should contain ones - one column per group.
        call_seq (np.ndarray): Default sequence of calls per row and group.

            Should have shape `target_shape` and each value indicate the index of a column in a group.
        active_mask (np.ndarray): Mask of whether a particular row should be executed per group.

            Should have shape `(target_shape[0], group_counts.shape[0])`.
        init_cash (np.ndarray): Initial capital per column, or per group if cash sharing is enabled.

            If `cash_sharing` is `True`, should have shape `(target_shape[0], group_counts.shape[0])`.
            Otherwise, should have shape `target_shape`.
        cash_sharing (bool): Whether to share cash within the same group.
        row_prep_func_nb (function): Row preparation function.

            Can be used for common order preparation tasks, and is executed before each row in each group.
            It should accept the current group row context `vectorbt.portfolio.enums.GroupRowContext`
            and unpacked `row_prep_args`. It should return a tuple of any content, which is passed to both
            `call_seq_func_nb` and `order_func_nb`.

            By default, returns an empty tuple.
        row_prep_args (tuple): Packed arguments passed to `row_prep_func_nb`.
        call_seq_func_nb (function): Call sequence generation function.

            Used to derive the order in which columns are called, and is executed before each row in
            each group and after `row_prep_func_nb`. It should accept the current group row context
            `vectorbt.portfolio.enums.GroupRowContext`, unpacked tuple from `row_prep_func_nb`, and
            unpacked `call_seq_args`. It should return an array of call indices.

            By default, returns the default sequence as in `call_seq`.

            !!! note
                Make sure to not generate any new arrays and not to write to `call_seq`, as it may
                negatively impact performance. You should simply re-order the relevant `call_seq` segment
                and return it. See `default_call_seq_nb`.

                For example, to reverse order, return `rc.call_seq[rc.i, rc.from_col:rc.to_col][::-1]`.
        call_seq_args (tuple): Packed arguments passed to `call_seq_func_nb`.
        order_func_nb (function): Order generation function.

            Used for either generating an order or skipping. Should accept the current order context
            `vectorbt.portfolio.enums.OrderContext`, unpacked tuple from `row_prep_func_nb`, and `*order_args`.
            Should either return `vectorbt.portfolio.enums.Order`, or `vectorbt.portfolio.enums.NoOrder`
            to do nothing.
        *order_args: Arguments passed to `order_func_nb`.

    !!! note
        Broadcasting isn't done automatically: you should either broadcast inputs before passing them
        to `order_func_nb`, or use flexible indexing - `vectorbt.base.reshape_fns.flex_choose_i_and_col_nb`
        together with `vectorbt.base.reshape_fns.flex_select_nb`.

        Also remember that indexing of 2-dim arrays in vectorbt follows that of pandas: `a[i, col]`.

    !!! note
        When simulating a grouped portfolio, order processing happens in row-major order,
        that is, from top to bottom slower (along time axis) and from left to right faster
        (along asset axis). See [Glossary](https://numpy.org/doc/stable/glossary.html).

    !!! warning
        You can only safely access data of columns that are to the left of the current group and
        rows that are to the top of the current row within the same group. Other data points have
        not been processed yet and thus empty. Accessing them, for example their `cash` and `shares`,
        will not trigger any errors or warnings, but provide you with arbitrary data
        (see [np.empty](https://numpy.org/doc/stable/reference/generated/numpy.empty.html)).

    Example:
        Build two groups: one with two columns together sharing 200$ and one with 100$. Then, buy all
        the one day and sell all the other. The column that is called first in each group is
        chosen randomly. Notice how assets fight for funds in the first group.

        ```python-repl
        >>> import numpy as np
        >>> import pandas as pd
        >>> from numba import njit
        >>> from vectorbt.portfolio.nb import simulate_nb, empty_row_prep_nb, build_call_seq
        >>> from vectorbt.portfolio.enums import Order, SizeType

        >>> price = np.array([1., 2., 3., 4., 5.])
        >>> n_cols = 3
        >>> target_shape = (price.shape[0], n_cols)
        >>> group_counts = np.array([2, 1])  # two groups
        >>> init_cash = np.array([200., 100.])  # per group if cash sharing
        >>> cash_sharing = True
        >>> call_seq = build_call_seq(target_shape, group_counts)
        >>> active_mask = np.copy(np.broadcast_to(True, target_shape))
        >>> size = np.inf
        >>> fees = 0.001
        >>> fixed_fees = 1.
        >>> slippage = 0.001
        >>> seed = 42

        >>> @njit
        ... def row_prep_func_nb(grc, seed):
        ...     return (seed + grc.i, size if grc.i % 2 == 0 else -size,)

        >>> @njit
        ... def call_seq_func_nb(grc, row_seed, row_size):
        ...     new_call_seq = grc.call_seq[grc.i, grc.from_col:grc.to_col]
        ...     np.random.seed(row_seed)
        ...     np.random.shuffle(new_call_seq)
        ...     return new_call_seq

        >>> @njit
        ... def order_func_nb(oc, row_seed, row_size):
        ...     return Order(
        ...         row_size,
        ...         SizeType.Shares,
        ...         price[oc.i],
        ...         fees,
        ...         fixed_fees,
        ...         slippage
        ...     )

        >>> order_records, cash, shares = simulate_nb(
        ...     target_shape, group_counts, call_seq, active_mask, init_cash, cash_sharing,
        ...     row_prep_func_nb, (seed,), call_seq_func_nb, (), order_func_nb)

        >>> pd.DataFrame.from_records(order_records)  # sorted
           col  idx        size  price      fees  side
        0    0    4  104.147735  5.005  1.521781     0
        1    1    0  198.602398  1.001  1.199000     0
        2    1    1  198.602398  1.998  1.396808     1
        3    1    2  131.207583  3.003  1.394411     0
        4    1    3  131.207583  3.996  1.524306     1
        5    2    0   98.802198  1.001  1.099000     0
        6    2    1   98.802198  1.998  1.197407     1
        7    2    2   64.939785  3.003  1.195209     0
        8    2    3   64.939785  3.996  1.259499     1
        9    2    4   51.345183  5.005  1.257240     0
        >>> call_seq
        [[1 0 0]
         [1 0 0]
         [1 0 0]
         [0 1 0]
         [0 1 0]]
        >>> shares  # currently holding at each step
        [[  0.         198.6025962   98.8022966 ]
         [  0.           0.           0.        ]
         [  0.         131.20784618  64.93991577]
         [  0.           0.           0.        ]
         [104.14804911   0.          51.34533868]]
        >>> cash  # currently holding at each step
        [[  0.           0.           0.        ]
         [395.41117923 395.41117923 196.20958163]
         [  0.           0.           0.        ]
         [  0.         522.78224677 258.24040352]
         [  0.           0.           0.        ]]
        ```
    """
    check_group_counts(target_shape, group_counts)
    check_group_init_cash(target_shape, group_counts, init_cash, cash_sharing)

    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    record_mask = np.full(target_shape[0] * target_shape[1], False)
    j = 0
    run_cash = init_cash.astype(np.float_)
    run_shares = np.full(target_shape[1], 0., dtype=np.float_)

    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        n_cols = to_col - from_col
        if cash_sharing:
            # Cash per group if cash sharing is enabled
            cash_now = run_cash[group]

        for i in range(target_shape[0]):
            # Is this row active?
            if active_mask[i, group]:
                # Run a function to preprocess this row within this group
                grc = GroupRowContext(
                    i,
                    group,
                    from_col,
                    to_col,
                    target_shape,
                    group_counts,
                    call_seq,
                    active_mask,
                    init_cash,
                    cash_sharing,
                    order_records,
                    record_mask,
                    j,
                    run_cash,
                    run_shares
                )
                prep_out = row_prep_func_nb(grc, *row_prep_args)

                # Run a function to override the default column ordering
                new_call_seq = call_seq_func_nb(grc, *prep_out, *call_seq_args)
                call_seq_changed = False

                for k in range(n_cols):
                    col_i = new_call_seq[k]
                    col = from_col + col_i
                    if col < from_col or col >= to_col:
                        raise ValueError("Index in call_seq is pointing outside of its group")

                    # Register ordering changes
                    if col_i != call_seq[i, from_col + k]:
                        call_seq_changed = True

                    if not cash_sharing:
                        # Cash per column if cash sharing is disabled
                        cash_now = run_cash[col]
                    # Shares per column
                    shares_now = run_shares[col]

                    # Generate the next order
                    oc = OrderContext(
                        i,
                        col,
                        group,
                        from_col,
                        to_col,
                        target_shape,
                        group_counts,
                        call_seq,
                        new_call_seq,
                        k,
                        active_mask,
                        init_cash,
                        cash_sharing,
                        order_records,
                        record_mask,
                        j,
                        run_cash,
                        run_shares,
                        cash_now,
                        shares_now
                    )
                    order = order_func_nb(oc, *prep_out, *order_args)

                    # Process the order
                    cash_now, shares_now, order_result = process_order_nb(cash_now, shares_now, order)

                    if order_result.status == OrderStatus.Filled:
                        # Add a new record
                        r = get_record_idx_nb(target_shape, col, i)
                        order_records[r]['col'] = col
                        order_records[r]['idx'] = i
                        order_records[r]['size'] = order_result.size
                        order_records[r]['price'] = order_result.price
                        order_records[r]['fees'] = order_result.fees
                        order_records[r]['side'] = order_result.side
                        record_mask[r] = True
                        j += 1

                    # Update last cash and shares
                    if cash_sharing:
                        run_cash[group] = cash_now
                    else:
                        run_cash[col] = cash_now
                    run_shares[col] = shares_now

                # Update call sequence if needed
                if call_seq_changed:
                    call_seq[i, from_col:to_col] = new_call_seq

        from_col = to_col

    # Order records are not sorted yet
    return order_records[record_mask]


@njit
def simulate_row_wise_nb(target_shape, group_counts, init_cash, cash_sharing, call_seq, active_mask,
                         row_prep_func_nb, row_prep_args, call_seq_func_nb, call_seq_args,
                         order_func_nb, *order_args):
    """Same as `simulate_nb`, but iterates using row-major order, with the rows
    changing fastest, and the columns changing slowest.

    The main difference is that `active_mask` and `row_prep_func_nb` are now executed
    per whole row rather than per row in the current group. Also, `row_prep_func_nb`
    should accept `vectorbt.portfolio.enums.RowContext` instead of `vectorbt.portfolio.enums.GroupRowContext`.

    !!! warning
        You can only safely access data points that are to the left of the current group and
        rows that are to the top of the current row, but now globally.
        ```
    """
    check_group_counts(target_shape, group_counts)
    check_group_init_cash(target_shape, group_counts, init_cash, cash_sharing)

    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    record_mask = np.full(target_shape[0] * target_shape[1], False)
    j = 0
    run_cash = init_cash.astype(np.float_)
    run_shares = np.full(target_shape[1], 0., dtype=np.float_)

    for i in range(target_shape[0]):
        # Is this row active?
        if active_mask[i]:
            # Run a function to preprocess this entire row
            rc = RowContext(
                i,
                target_shape,
                group_counts,
                call_seq,
                active_mask,
                init_cash,
                cash_sharing,
                order_records,
                record_mask,
                j,
                run_cash,
                run_shares,
            )
            prep_out = row_prep_func_nb(rc, *row_prep_args)

            from_col = 0
            for group in range(len(group_counts)):
                to_col = from_col + group_counts[group]
                n_cols = to_col - from_col

                # Cash per group if cash sharing is enabled
                if cash_sharing:
                    cash_now = run_cash[group]

                # Run a function to override the default column ordering
                grc = GroupRowContext(
                    i,
                    group,
                    from_col,
                    to_col,
                    target_shape,
                    group_counts,
                    call_seq,
                    active_mask,
                    init_cash,
                    cash_sharing,
                    order_records,
                    record_mask,
                    j,
                    run_cash,
                    run_shares,
                )
                new_call_seq = call_seq_func_nb(grc, *prep_out, *call_seq_args)
                call_seq_changed = False

                for k in range(n_cols):
                    col_i = new_call_seq[k]
                    col = from_col + col_i
                    if col < from_col or col >= to_col:
                        raise ValueError("Index in call_seq is pointing outside of its group")

                    # Register ordering changes
                    if col_i != call_seq[i, from_col + k]:
                        call_seq_changed = True

                    if not cash_sharing:
                        # Cash per column if cash sharing is disabled
                        cash_now = run_cash[col]
                    # Shares per column
                    shares_now = run_shares[col]

                    # Generate the next order
                    oc = OrderContext(
                        i,
                        col,
                        group,
                        from_col,
                        to_col,
                        target_shape,
                        group_counts,
                        call_seq,
                        new_call_seq,
                        k,
                        active_mask,
                        init_cash,
                        cash_sharing,
                        order_records,
                        record_mask,
                        j,
                        run_cash,
                        run_shares,
                        cash_now,
                        shares_now
                    )
                    order = order_func_nb(oc, *prep_out, *order_args)

                    # Process the order
                    cash_now, shares_now, order_result = process_order_nb(cash_now, shares_now, order)

                    if order_result.status == OrderStatus.Filled:
                        # Add a new record
                        r = get_record_idx_nb(target_shape, col, i)
                        order_records[r]['col'] = col
                        order_records[r]['idx'] = i
                        order_records[r]['size'] = order_result.size
                        order_records[r]['price'] = order_result.price
                        order_records[r]['fees'] = order_result.fees
                        order_records[r]['side'] = order_result.side
                        record_mask[r] = True
                        j += 1

                    # Update last cash and shares
                    if cash_sharing:
                        run_cash[group] = cash_now
                    else:
                        run_cash[col] = cash_now
                    run_shares[col] = shares_now

                # Update call sequence if needed
                if call_seq_changed:
                    call_seq[i, from_col:to_col] = new_call_seq

                from_col = to_col

    # Order records are not sorted yet
    return order_records[record_mask]


@njit(cache=True)
def simulate_from_signals_nb(target_shape, group_counts, init_cash, cash_sharing, call_seq, entries,
                             exits, size, size_type, entry_price, exit_price, fees, fixed_fees,
                             slippage, accumulate, accumulate_exit_mode, conflict_mode, is_2d):
    """Adaptation of `simulate_nb` for simulation based on entry and exit signals.

    Utilizes flexible broadcasting."""
    check_group_counts(target_shape, group_counts)
    check_group_init_cash(target_shape, group_counts, init_cash, cash_sharing)

    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    record_mask = np.full(target_shape[0] * target_shape[1], False)
    order_matters = is_grouped_nb(group_counts)
    j = 0
    run_cash = init_cash.astype(np.float_)
    run_shares = np.full(target_shape[1], 0., dtype=np.float_)

    # Inputs were not broadcast -> use flexible indexing
    def_i1, def_col1 = flex_choose_i_and_col_nb(entries, is_2d)
    def_i2, def_col2 = flex_choose_i_and_col_nb(exits, is_2d)
    def_i3, def_col3 = flex_choose_i_and_col_nb(size, is_2d)
    def_i4, def_col4 = flex_choose_i_and_col_nb(size_type, is_2d)
    def_i5, def_col5 = flex_choose_i_and_col_nb(entry_price, is_2d)
    def_i6, def_col6 = flex_choose_i_and_col_nb(exit_price, is_2d)
    def_i7, def_col7 = flex_choose_i_and_col_nb(fees, is_2d)
    def_i8, def_col8 = flex_choose_i_and_col_nb(fixed_fees, is_2d)
    def_i9, def_col9 = flex_choose_i_and_col_nb(slippage, is_2d)

    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        n_cols = to_col - from_col
        if cash_sharing:
            # Cash per group if cash sharing is enabled
            cash_now = run_cash[group]

        for i in range(target_shape[0]):
            for k in range(n_cols):
                col = from_col + k
                if order_matters:
                    col = from_col + call_seq[i, col]
                    if col < from_col or col >= to_col:
                        raise ValueError("Index in call_seq is pointing outside of its group")

                if not cash_sharing:
                    # Cash per column if cash sharing is disabled
                    cash_now = run_cash[col]
                # Shares per column
                shares_now = run_shares[col]

                is_entry = flex_select_nb(i, col, entries, def_i1, def_col1, is_2d)
                is_exit = flex_select_nb(i, col, exits, def_i2, def_col2, is_2d)
                if is_entry or is_exit:
                    # Generate the next order
                    order = signals_order_func_nb(
                        shares_now,
                        is_entry,
                        is_exit,
                        flex_select_nb(i, col, size, def_i3, def_col3, is_2d),
                        flex_select_nb(i, col, size_type, def_i4, def_col4, is_2d),
                        flex_select_nb(i, col, entry_price, def_i5, def_col5, is_2d),
                        flex_select_nb(i, col, exit_price, def_i6, def_col6, is_2d),
                        flex_select_nb(i, col, fees, def_i7, def_col7, is_2d),
                        flex_select_nb(i, col, fixed_fees, def_i8, def_col8, is_2d),
                        flex_select_nb(i, col, slippage, def_i9, def_col9, is_2d),
                        accumulate,
                        accumulate_exit_mode,
                        conflict_mode
                    )

                    # Process the order
                    cash_now, shares_now, order_result = process_order_nb(cash_now, shares_now, order)

                    if order_result.status == OrderStatus.Filled:
                        # Add a new record
                        if order_matters:
                            r = get_record_idx_nb(target_shape, col, i)
                        else:
                            r = j
                        order_records[r]['col'] = col
                        order_records[r]['idx'] = i
                        order_records[r]['size'] = order_result.size
                        order_records[r]['price'] = order_result.price
                        order_records[r]['fees'] = order_result.fees
                        order_records[r]['side'] = order_result.side
                        if order_matters:
                            record_mask[r] = True
                        j += 1

                # Update last cash and shares
                if cash_sharing:
                    run_cash[group] = cash_now
                else:
                    run_cash[col] = cash_now
                run_shares[col] = shares_now

        from_col = to_col

    # Order records are not sorted yet
    if order_matters:
        return order_records[record_mask]
    return order_records[:j]


@njit(cache=True)
def signals_order_func_nb(shares_now, is_entry, is_exit, size, size_type, entry_price,
                          exit_price, fees, fixed_fees, slippage, accumulate, accumulate_exit_mode, conflict_mode):
    """`order_func_nb` of `simulate_from_signals_nb`."""
    if size_type != SizeType.Shares and size_type != SizeType.Cash:
        raise ValueError("Only SizeType.Shares and SizeType.Cash are supported")
    if is_entry and not is_exit:
        # Open or increase the position
        if shares_now == 0. or accumulate:
            order_size = abs(size)
            order_price = entry_price
        else:
            return NoOrder
    elif not is_entry and is_exit:
        if shares_now > 0.:
            # If in position
            if not accumulate or (accumulate and accumulate_exit_mode == AccumulateExitMode.Close):
                # Close the position
                order_size = -np.inf
                order_price = exit_price
            elif accumulate and accumulate_exit_mode == AccumulateExitMode.Reduce:
                # Decrease the position
                order_size = -abs(size)
                order_price = exit_price
        else:
            return NoOrder
    elif is_entry and is_exit:
        # Conflict
        if conflict_mode == ConflictMode.Ignore:
            return NoOrder
        if shares_now > 0.:
            # If in position
            if accumulate and accumulate_exit_mode == AccumulateExitMode.Reduce:
                # Selling and buying the same size makes no sense
                return NoOrder
            if conflict_mode == ConflictMode.Exit:
                # Close the position
                order_size = -np.inf
                order_price = exit_price
            elif conflict_mode == ConflictMode.ExitAndEntry:
                # Do not sell and then buy, but buy/sell the difference (less fees)
                if size_type == SizeType.Shares:
                    # Target size in shares
                    order_size = abs(size) - shares_now
                else:
                    # Target size in cash, not to be confused with SizeType.TargetCash
                    # Must be converted to shares the same way as in buy_in_cash_nb
                    # At the end the number of shares must be the same as if we had a clear entry signal
                    order_cash = (abs(size) - fixed_fees) * (1 - fees)
                    target_size = order_cash / (entry_price * (1 + slippage))
                    order_size = target_size - shares_now
                    size_type = SizeType.Shares
                if order_size > 0:
                    order_price = entry_price
                elif order_size < 0:
                    order_price = exit_price
                else:
                    return NoOrder
            else:
                return NoOrder
        else:
            return NoOrder
    else:
        return NoOrder
    return Order(order_size, size_type, order_price, fees, fixed_fees, slippage)


@njit(cache=True)
def simulate_from_orders_nb(target_shape, group_counts, init_cash, cash_sharing, call_seq, size,
                            size_type, price, fees, fixed_fees, slippage, is_2d=False):
    """Adaptation of `simulate_nb` for simulation based on orders.

    Utilizes flexible broadcasting."""
    check_group_counts(target_shape, group_counts)
    check_group_init_cash(target_shape, group_counts, init_cash, cash_sharing)

    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    record_mask = np.full(target_shape[0] * target_shape[1], False)
    order_matters = is_grouped_nb(group_counts)
    j = 0
    run_cash = init_cash.astype(np.float_)
    run_shares = np.full(target_shape[1], 0., dtype=np.float_)

    # Inputs were not broadcast -> use flexible indexing
    def_i1, def_col1 = flex_choose_i_and_col_nb(size, is_2d)
    def_i2, def_col2 = flex_choose_i_and_col_nb(size_type, is_2d)
    def_i3, def_col3 = flex_choose_i_and_col_nb(price, is_2d)
    def_i4, def_col4 = flex_choose_i_and_col_nb(fees, is_2d)
    def_i5, def_col5 = flex_choose_i_and_col_nb(fixed_fees, is_2d)
    def_i6, def_col6 = flex_choose_i_and_col_nb(slippage, is_2d)

    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        n_cols = to_col - from_col
        if cash_sharing:
            # Cash per group if cash sharing is enabled
            cash_now = run_cash[group]

        for i in range(target_shape[0]):
            for k in range(n_cols):
                col = from_col + k
                if order_matters:
                    col = from_col + call_seq[i, col]
                    if col < from_col or col >= to_col:
                        raise ValueError("Index in call_seq is pointing outside of its group")

                if not cash_sharing:
                    # Cash per column if cash sharing is disabled
                    cash_now = run_cash[col]
                # Shares per column
                shares_now = run_shares[col]

                # Generate the next order
                order = Order(
                    flex_select_nb(i, col, size, def_i1, def_col1, is_2d),
                    flex_select_nb(i, col, size_type, def_i2, def_col2, is_2d),
                    flex_select_nb(i, col, price, def_i3, def_col3, is_2d),
                    flex_select_nb(i, col, fees, def_i4, def_col4, is_2d),
                    flex_select_nb(i, col, fixed_fees, def_i5, def_col5, is_2d),
                    flex_select_nb(i, col, slippage, def_i6, def_col6, is_2d),
                )

                # Process the order
                cash_now, shares_now, order_result = process_order_nb(cash_now, shares_now, order)

                if order_result.status == OrderStatus.Filled:
                    # Add a new record
                    if order_matters:
                        r = get_record_idx_nb(target_shape, col, i)
                    else:
                        r = j
                    order_records[r]['col'] = col
                    order_records[r]['idx'] = i
                    order_records[r]['size'] = order_result.size
                    order_records[r]['price'] = order_result.price
                    order_records[r]['fees'] = order_result.fees
                    order_records[r]['side'] = order_result.side
                    if order_matters:
                        record_mask[r] = True
                    j += 1

                # Update last cash and shares
                if cash_sharing:
                    run_cash[group] = cash_now
                else:
                    run_cash[col] = cash_now
                run_shares[col] = shares_now

        from_col = to_col

    # Order records are not sorted yet
    if order_matters:
        return order_records[record_mask]
    return order_records[:j]


# ############# Properties ############# #

@njit(cache=True)
def cash_grouped_nb(cash, group_counts, cash_sharing):
    """Get cash per group."""
    check_group_counts(cash.shape, group_counts)
    if cash_sharing:
        return cash[:, np.cumsum(group_counts) - 1]
    out = np.empty((cash.shape[0], len(group_counts)), dtype=np.float_)
    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        out[:, group] = np.sum(cash[:, from_col:to_col], axis=1)
        from_col = to_col
    return out


@njit(cache=True)
def cash_flow_nb(target_shape, order_records):
    """Get cash flow per column."""
    out = np.full(target_shape, 0., dtype=np.float_)
    for r in range(order_records.shape[0]):
        record = order_records[r]
        if record['side'] == OrderSide.Buy:
            out[record['idx'], record['col']] -= record['size'] * record['price'] + record['fees']
        elif record['side'] == OrderSide.Sell:
            out[record['idx'], record['col']] += record['size'] * record['price'] - record['fees']
    return out


@njit(cache=True)
def cash_flow_grouped_nb(cash_flow, group_counts):
    """Get cash flow per group."""
    check_group_counts(cash_flow.shape, group_counts)
    out = np.empty((cash_flow.shape[0], len(group_counts)), dtype=np.float_)
    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        out[:, group] = np.sum(cash_flow[:, from_col:to_col], axis=1)
        from_col = to_col
    return out


@njit(cache=True)
def grouped_holding_value_nb(price, shares, group_counts):
    """Get holding value per group."""
    check_group_counts(price.shape, group_counts)
    out = np.empty((price.shape[0], len(group_counts)), dtype=np.float_)
    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        out[:, group] = np.sum(shares[:, from_col:to_col] * price[:, from_col:to_col], axis=1)
        from_col = to_col
    return out


@njit(cache=True)
def ungrouped_iter_value_nb(cash, holding_value, group_counts):
    """Get value per asset in the group in simulation order."""
    check_group_counts(cash.shape, group_counts)
    out = np.empty(cash.shape, dtype=np.float_)
    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        n_cols = to_col - from_col
        cash_flat = cash[:, from_col:to_col].flatten()
        holding_value_flat = holding_value[:, from_col:to_col].flatten()
        curr_holding_value = 0.
        # Without correctly treating NaN values, after one NaN all will be NaN
        since_last_nan = n_cols

        for j in range(cash_flat.shape[0]):
            if j >= n_cols:
                prev_j = j - n_cols
                if not np.isnan(holding_value_flat[prev_j]):
                    curr_holding_value -= holding_value_flat[prev_j]
            if np.isnan(holding_value_flat[j]):
                since_last_nan = 0
            else:
                curr_holding_value += holding_value_flat[j]
            if since_last_nan < n_cols:
                out[j // n_cols, from_col + j % n_cols] = np.nan
            else:
                out[j // n_cols, from_col + j % n_cols] = cash_flat[j] + curr_holding_value
            since_last_nan += 1

        from_col = to_col
    return out


@njit(cache=True)
def ungrouped_iter_returns_nb(iter_value, init_value, group_counts):
    """Get returns per asset in the group in simulation order."""
    check_group_counts(iter_value.shape, group_counts)
    out = np.empty(iter_value.shape, dtype=np.float_)
    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        n_cols = to_col - from_col
        iter_value_flat = iter_value[:, from_col:to_col].flatten()
        iter_returns_flat = np.empty(iter_value_flat.shape, dtype=np.float_)
        iter_returns_flat[0] = (iter_value_flat[0] - init_value[group]) / init_value[group]
        iter_returns_flat[1:] = (iter_value_flat[1:] - iter_value_flat[:-1]) / iter_value_flat[:-1]
        out[:, from_col:to_col] = iter_returns_flat.reshape((iter_value.shape[0], n_cols))
        from_col = to_col
    return out


@njit(cache=True)
def grouped_buy_and_hold_return_nb(price, group_counts):
    """Get total return of buy-and-hold per group."""
    check_group_counts(price.shape, group_counts)
    out = np.empty(len(group_counts), dtype=np.float_)
    total_return = (price[-1, :] - price[0, :]) / price[0, :]
    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        n_cols = to_col - from_col
        out[group] = np.sum(total_return[from_col:to_col]) / n_cols
        from_col = to_col
    return out
