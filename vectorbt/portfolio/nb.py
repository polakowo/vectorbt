"""Numba-compiled 1-dim and 2-dim functions.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    
    All functions passed as argument should be Numba-compiled.
    
    Records should remain the order they were created in.

!!! warning
    Accumulation of roundoff error possible.
    See [here](https://en.wikipedia.org/wiki/Round-off_error#Accumulation_of_roundoff_error) for explanation.

    Rounding errors can cause trades and positions to not close properly.

    Example:

        >>> print('%.50f' % 0.1)  # has positive error
        0.10000000000000000555111512312578270211815834045410

        >>> # many buy transactions with positive error -> cannot close position
        >>> sum([0.1 for _ in range(1000000)]) - 100000
        1.3328826753422618e-06

        >>> print('%.50f' % 0.3)  # has negative error
        0.29999999999999998889776975374843459576368331909180

        >>> # many sell transactions with negative error -> cannot close position
        >>> 300000 - sum([0.3 for _ in range(1000000)])
        5.657668225467205e-06

    While vectorbt has implemented tolerance checks when comparing floats for equality (look for 
    lines with "numerical stability" comment), adding/subtracting small amounts large number of
    times may still introduce a noticable error that cannot be corrected post factum.

    To mitigate this issue, avoid repeating lots of micro-transactions of the same sign in favor of
    one bigger transaction. Also, when closing a position, use `-np.inf` instead of a finite number.
"""

import numpy as np
from numba import njit

from vectorbt.utils.math import is_close_or_less_nb
from vectorbt.utils.array import insert_argsort_nb
from vectorbt.base.reshape_fns import flex_choose_i_and_col_nb, flex_select_nb
from vectorbt.generic import nb as generic_nb
from vectorbt.portfolio.enums import (
    SimulationContext,
    GroupContext,
    RowContext,
    SegmentContext,
    OrderContext,
    CallSeqType,
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
def buy_shares_nb(cash_now, shares_now, order_price, order_size, order_fees, order_fixed_fees, order_slippage):
    """Buy shares."""

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
def sell_shares_nb(cash_now, shares_now, order_price, order_size, order_fees, order_fixed_fees, order_slippage):
    """Sell shares."""

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
def process_order_nb(cash_now, shares_now, order, min_size):
    """Fill an order."""
    if np.isnan(order.size) or np.isnan(order.price):
        return cash_now, shares_now, RejectedOrder

    if not np.isfinite(order.price) or order.price <= 0.:
        raise ValueError("Price must be finite and greater than 0")
    if not np.isfinite(order.fees) or order.fees < 0.:
        raise ValueError("Fees must be finite and 0 or greater")
    if not np.isfinite(order.fixed_fees) or order.fixed_fees < 0.:
        raise ValueError("Fixed fees must be finite and 0 or greater")
    if not np.isfinite(order.slippage) or order.slippage < 0.:
        raise ValueError("Slippage must be finite and 0 or greater")
    if not np.isfinite(order.reject_prob) or order.reject_prob < 0. or order.reject_prob > 1.:
        raise ValueError("Rejection probability must be between 0 and 1")
    if not np.isfinite(min_size) or min_size < 0.:
        raise ValueError("Minimum size must be finite and 0 or greater")

    # Convert target size to shares
    size = order.size
    if order.size_type == SizeType.TargetShares:
        # Target amount of shares
        size = order.size - shares_now
    elif order.size_type == SizeType.TargetValue:
        # Value in monetary units of the asset
        target_value = order.size
        current_value = shares_now * order.price
        size = (target_value - current_value) / order.price
    elif order.size_type == SizeType.TargetPercent:
        # Percentage from current value that can be moved by this asset
        # Not to confuse with group value!
        target_perc = order.size
        current_value = shares_now * order.price
        current_total_value = cash_now + current_value
        target_value = target_perc * current_total_value
        size = (target_value - current_value) / order.price

    if abs(size) < min_size:
        return cash_now, shares_now, RejectedOrder

    if order.reject_prob > 0.:
        if np.random.uniform(0, 1) < order.reject_prob:
            return cash_now, shares_now, RejectedOrder

    if size > 0. and cash_now > 0.:
        if np.isinf(size) and np.isinf(cash_now):
            raise ValueError("Size and current cash cannot be both infinite")

        return buy_shares_nb(
            cash_now,
            shares_now,
            order.price,
            size,
            order.fees,
            order.fixed_fees,
            order.slippage
        )
    if size < 0. and shares_now > 0.:
        if np.isinf(size) and np.isinf(shares_now):
            raise ValueError("Size and current shares cannot be both infinite")

        return sell_shares_nb(
            cash_now,
            shares_now,
            order.price,
            abs(size),
            order.fees,
            order.fixed_fees,
            order.slippage
        )

    return cash_now, shares_now, RejectedOrder


@njit(cache=True)
def check_group_counts(group_counts, n_cols):
    """Check `group_counts`."""
    if np.sum(group_counts) != n_cols:
        raise ValueError("group_counts has incorrect total number of columns")


@njit(cache=True)
def check_group_init_cash(group_counts, n_cols, init_cash, cash_sharing):
    """Check `init_cash`."""
    if cash_sharing:
        if len(init_cash) != len(group_counts):
            raise ValueError("If cash sharing is enabled, init_cash must match the number of groups")
    else:
        if len(init_cash) != n_cols:
            raise ValueError("If cash sharing is disabled, init_cash must match the number of columns")


@njit(cache=True)
def get_record_idx_nb(target_shape, col, i):
    """Get record index by position of order in the matrix."""
    return col * target_shape[0] + i


@njit(cache=True)
def is_grouped_nb(group_counts):
    """Check if columm,ns are grouped, that is, more than one column per group."""
    return np.any(group_counts > 1)


@njit(cache=True)
def shuffle_call_seq_nb(call_seq, group_counts):
    """Shuffle the call sequence array."""
    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        for i in range(call_seq.shape[0]):
            np.random.shuffle(call_seq[i, from_col:to_col])
        from_col = to_col


@njit(cache=True)
def build_call_seq_nb(target_shape, group_counts, call_seq_type=CallSeqType.Default):
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
        shuffle_call_seq_nb(out, group_counts)
    return out


def require_call_seq(call_seq):
    """Force the call sequence array to pass our requirements."""
    return np.require(call_seq, dtype=np.int_, requirements=['A', 'O', 'W', 'F'])


def build_call_seq(target_shape, group_counts, call_seq_type=CallSeqType.Default):
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
        shuffle_call_seq_nb(call_seq, group_counts)
    return require_call_seq(call_seq)


@njit(cache=True)
def empty_prep_nb(*args):
    """Preparation function that returns an empty tuple."""
    return ()


@njit(cache=True)
def get_group_value_nb(from_col, to_col, cash_now, last_shares, last_val_price):
    """Get group value."""
    group_value = cash_now
    group_len = to_col - from_col
    for k in range(group_len):
        col = from_col + k
        if last_shares[col] > 0.:
            holding_value = last_shares[col] * last_val_price[col]
            group_value += holding_value
    return group_value


@njit(cache=True)
def get_group_value_ctx_nb(sc_oc):
    """Get group value from context.

    Accepts `vectorbt.portfolio.enums.SegmentContext` and `vectorbt.portfolio.enums.OrderContext`.

    Best called once from `segment_prep_func_nb`.
    To set the valuation price, change `last_val_price` of the context in-place.

    !!! note
        Cash sharing must be enabled."""
    if not sc_oc.cash_sharing:
        raise ValueError("Cash sharing must be enabled")
    return get_group_value_nb(
        sc_oc.from_col,
        sc_oc.to_col,
        sc_oc.last_cash[sc_oc.group],
        sc_oc.last_shares,
        sc_oc.last_val_price
    )


@njit(cache=True)
def get_order_value_nb(order_size, order_size_type, shares_now, price_now, value_now):
    """Get potential value of an order."""
    holding_value_now = shares_now * price_now
    if order_size_type == SizeType.Shares:
        return order_size * price_now
    if order_size_type == SizeType.TargetShares:
        return order_size * price_now - holding_value_now
    if order_size_type == SizeType.TargetValue:
        return order_size - holding_value_now
    if order_size_type == SizeType.TargetPercent:
        return order_size * value_now - holding_value_now
    return np.nan


@njit(cache=True)
def auto_call_seq_ctx_nb(sc, order_size, order_size_type, temp_float_arr):
    """Generate call sequence based on order value dynamically, for example, to rebalance.

    Accepts `vectorbt.portfolio.enums.SegmentContext`.

    Arrays `order_size`, `order_size_type` and `temp_float_arr` should match the number
    of columns in the group. Array `temp_float_arr` should be empty and will contain
    sorted order values after execution.

    Best called once from `segment_prep_func_nb`.

    !!! note
        Cash sharing must be enabled and `call_seq_now` should follow `CallSeqType.Default`."""
    if not sc.cash_sharing:
        raise ValueError("Cash sharing must be enabled")
    group_value_now = get_group_value_ctx_nb(sc)
    group_len = sc.to_col - sc.from_col
    for k in range(group_len):
        if sc.call_seq_now[k] != k:
            raise ValueError("call_seq_now should follow CallSeqType.Default")
        col = sc.from_col + k
        temp_float_arr[k] = get_order_value_nb(
            order_size[k],
            order_size_type[k],
            sc.last_shares[col],
            sc.last_val_price[col],
            group_value_now
        )
    # Sort by order value
    insert_argsort_nb(temp_float_arr, sc.call_seq_now)


@njit
def simulate_nb(target_shape, close, group_counts, init_cash, cash_sharing, call_seq, active_mask, min_size,
                prep_func_nb, prep_args, group_prep_func_nb, group_prep_args, segment_prep_func_nb,
                segment_prep_args, order_func_nb, order_args):
    """Simulate a portfolio by generating and filling orders.

    Starting with initial cash `init_cash`, iterates over each group and column over shape `target_shape`,
    and for each data point, generates an order using `order_func_nb`. Tries then to fulfill that
    order. If unsuccessful due to insufficient cash/shares, always orders the available fraction.
    Updates then the current cash and shares balance.

    Returns order records of layout `vectorbt.records.enums.order_dt`.

    As opposed to `simulate_row_wise_nb`, order processing happens in row-major order, that is,
    from top to bottom slower (along time axis) and from left to right faster (along asset axis).
    See [Glossary](https://numpy.org/doc/stable/glossary.html).

    Args:
        target_shape (tuple): Target shape.

            A tuple with exactly two elements: the number of steps and columns.
        close (np.ndarray): Reference price, such as close.
        
            Should have shape `target_shape`.
        group_counts (np.ndarray): Column count per group.

            Even if columns are not grouped, `group_counts` should contain ones - one column per group.
        init_cash (np.ndarray): Initial capital per column, or per group if cash sharing is enabled.

            If `cash_sharing` is True, should have shape `(target_shape[0], group_counts.shape[0])`.
            Otherwise, should have shape `target_shape`.
        cash_sharing (bool): Whether to share cash within the same group.
        call_seq (np.ndarray): Default sequence of calls per row and group.

            Should have shape `target_shape` and each value indicate the index of a column in a group.

            !!! note
                To use `auto_call_seq_ctx_nb`, should be of `CallSeqType.Default`.
        active_mask (np.ndarray): Mask of whether a particular segment should be executed.

            A segment is simply a sequence of `order_func_nb` calls under the same group and row.

            Should have shape `(target_shape[0], group_counts.shape[0])`.
        min_size (np.ndarray): Minimum size for an order to be accepted.

            Should have shape `(target_shape[1],)`.
        prep_func_nb (callable): Simulation preparation function.

            Can be used for creation of global arrays and setting the seed, and is executed at the
            beginning of the simulation. It should accept `*prep_args`, and return a tuple of any
            content, which is then passed to `group_prep_func_nb`.
        prep_args (tuple): Packed arguments passed to `prep_func_nb`.
        group_prep_func_nb (callable): Group preparation function.

            Executed before each group. Should accept the current group context
            `vectorbt.portfolio.enums.GroupContext`, unpacked tuple from `prep_func_nb`, and
            `*group_prep_args`. Should return a tuple of any content, which is then passed to
            `segment_prep_func_nb`.
        group_prep_args (tuple): Packed arguments passed to `group_prep_func_nb`.
        segment_prep_func_nb (callable): Segment preparation function.

            Executed before each row in a group. Should accept the current segment context
            `vectorbt.portfolio.enums.SegmentContext`, unpacked tuple from `group_prep_func_nb`,
            and `*segment_prep_args`. Should return a tuple of any content, which is then
            passed to `order_func_nb`.

            !!! note
                To change the call sequence of the segment, access `SegmentContext.call_seq_now`
                and change it in-place. Make sure to not generate any new arrays as it may
                negatively impact performance. Assigning `SegmentContext.call_seq_now` is not allowed.

            !!! note
                Use `last_val_price` to manipulate group valuation. By default, `last_val_price`
                contains the last `close` for a column. You can change it in-place.
                The column/group is then valuated after `segment_prep_func_nb`, and the value is
                passed as `value_now` to `order_func_nb` and internally used for converting
                `SizeType.TargetPercent` and `SizeType.TargetValue` to `SizeType.TargetShares`.
        segment_prep_args (tuple): Packed arguments passed to `segment_prep_func_nb`.
        order_func_nb (callable): Order generation function.

            Used for either generating an order or skipping. Should accept the current order context
            `vectorbt.portfolio.enums.OrderContext`, unpacked tuple from `segment_prep_func_nb`, and
            `*order_args`. Should either return `vectorbt.portfolio.enums.Order`, or
            `vectorbt.portfolio.enums.NoOrder` to do nothing.
        order_args (tuple): Arguments passed to `order_func_nb`.

    !!! note
        Broadcasting isn't done automatically: you should either broadcast inputs before passing them
        to `order_func_nb`, or use flexible indexing - `vectorbt.base.reshape_fns.flex_choose_i_and_col_nb`
        together with `vectorbt.base.reshape_fns.flex_select_nb`.

        Also remember that indexing of 2-dim arrays in vectorbt follows that of pandas: `a[i, col]`.

    !!! note
        Function `group_prep_func_nb` is only called if there is at least on active segment in
        the group. Functions `segment_prep_func_nb` and `order_func_nb` are only called if their
        segment is active. If the main task of `group_prep_func_nb` is to activate/deactivate segments,
        all segments should be activated by default to allow `group_prep_func_nb` to be called.

    !!! warning
        You can only safely access data of columns that are to the left of the current group and
        rows that are to the top of the current row within the same group. Other data points have
        not been processed yet and thus empty. Accessing them will not trigger any errors or warnings,
        but provide you with arbitrary data (see [np.empty](https://numpy.org/doc/stable/reference/generated/numpy.empty.html)).

    Example:
        Create a group of three assets together sharing 100$ and simulate an equal-weighted portfolio
        that rebalances every second tick, all without leaving Numba:

        ```python-repl
        >>> import numpy as np
        >>> import pandas as pd
        >>> from numba import njit
        >>> from vectorbt.portfolio.nb import (
        ...     simulate_nb,
        ...     build_call_seq,
        ...     auto_call_seq_ctx_nb,
        ...     share_flow_nb,
        ...     shares_nb,
        ...     holding_value_ungrouped_nb
        ... )
        >>> from vectorbt.portfolio.enums import Order, SizeType

        >>> target_shape = (5, 3)
        >>> np.random.seed(42)
        >>> price = np.random.uniform(1, 10, size=target_shape)
        >>> group_counts = np.array([3])  # group of three columns
        >>> init_cash = np.array([100.])  # one capital per group
        >>> cash_sharing = True
        >>> call_seq = build_call_seq(target_shape, group_counts)  # will be overridden
        >>> active_mask = np.array([True, False, True, False, True])[:, None]
        >>> active_mask = np.copy(np.broadcast_to(active_mask, target_shape))
        >>> min_size = np.copy(np.broadcast_to(1e-4, (target_shape[1],)))
        >>> fees = 0.001
        >>> fixed_fees = 1.
        >>> slippage = 0.001
        >>> reject_prob = 0.

        >>> @njit
        ... def prep_func_nb(simc):  # do nothing
        ...     print('preparing simulation')
        ...     return ()

        >>> @njit
        ... def group_prep_func_nb(gc):  # create arrays
        ...     print('\\tpreparing group', gc.group)
        ...     # Try to create new arrays as rarely as possible
        ...     order_size = np.empty(gc.group_len, dtype=np.float_)
        ...     order_size_type = np.empty(gc.group_len, dtype=np.int_)
        ...     temp_float_arr = np.empty(gc.group_len, dtype=np.float_)
        ...     return order_size, order_size_type, temp_float_arr

        >>> @njit
        ... def segment_prep_func_nb(sc, order_size, order_size_type, temp_float_arr):  # rebalance
        ...     print('\\t\\tpreparing segment', sc.i, '(row)')
        ...     for k in range(sc.group_len):
        ...         col = sc.from_col + k
        ...         order_size[k] = 1 / sc.group_len
        ...         order_size_type[k] = SizeType.TargetPercent
        ...         # In this example last seen price is now, just for illustration
        ...         sc.last_val_price[col] = price[sc.i, col]
        ...     # Reorder call sequence such that selling orders come first and buying last
        ...     auto_call_seq_ctx_nb(sc, order_size, order_size_type, temp_float_arr)
        ...     return order_size, order_size_type

        >>> @njit
        ... def order_func_nb(oc, order_size, order_size_type):  # place an order
        ...     print('\\t\\t\\trunning order', oc.call_idx, 'at column', oc.col)
        ...     col_i = oc.call_seq_now[oc.call_idx]  # or col - from_col
        ...     return Order(
        ...         order_size[col_i],
        ...         order_size_type[col_i],
        ...         price[oc.i, oc.col],
        ...         fees, fixed_fees, slippage, reject_prob
        ...     )

        >>> order_records = simulate_nb(
        ...     target_shape,
        ...     price,
        ...     group_counts,
        ...     init_cash,
        ...     cash_sharing,
        ...     call_seq,
        ...     active_mask,
        ...     min_size,
        ...     prep_func_nb, (),
        ...     group_prep_func_nb, (),
        ...     segment_prep_func_nb, (),
        ...     order_func_nb, ())
        preparing simulation
            preparing group 0
                preparing segment 0 (row)
                    running order 0 at column 0
                    running order 1 at column 1
                    running order 2 at column 2
                preparing segment 2 (row)
                    running order 0 at column 1
                    running order 1 at column 2
                    running order 2 at column 0
                preparing segment 4 (row)
                    running order 0 at column 0
                    running order 1 at column 2
                    running order 2 at column 1

        >>> pd.DataFrame.from_records(order_records)  # sorted
           col  idx       size     price      fees  side
        0    0    0   7.626262  4.375232  1.033367     0
        1    0    2   5.210115  1.524275  1.007942     0
        2    0    4   7.899568  8.483492  1.067016     1
        3    1    0   3.488053  9.565985  1.033367     0
        4    1    2   0.920352  8.786790  1.008087     1
        5    1    4  10.713236  2.913963  1.031218     0
        6    2    0   3.972040  7.595533  1.030170     0
        7    2    2   0.448747  6.403625  1.002874     1
        8    2    4  12.378281  2.639061  1.032667     0

        >>> call_seq
        array([[0, 1, 2],
               [0, 1, 2],
               [1, 2, 0],
               [0, 1, 2],
               [0, 2, 1]])

        >>> share_flow = share_flow_nb(target_shape, order_records)
        >>> shares = shares_nb(share_flow)
        >>> holding_value = holding_value_ungrouped_nb(price, shares)
        >>> pd.DataFrame(holding_value).vbt.scatter()
        ```

        ![](/vectorbt/docs/img/simulate_nb.png)

        Note that the last order in a group with cash sharing is always disadvantaged
        as it has a bit less funds than the previous orders due to costs, which are not
        included when valuating the group.
    """
    check_group_counts(group_counts, target_shape[1])
    check_group_init_cash(group_counts, target_shape[1], init_cash, cash_sharing)

    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    record_mask = np.full(target_shape[0] * target_shape[1], False)
    j = 0
    last_cash = init_cash.astype(np.float_)
    last_shares = np.full(target_shape[1], 0., dtype=np.float_)
    last_val_price = np.full_like(last_shares, np.nan, dtype=np.float_)

    # Run a function to prepare the simulation
    simc = SimulationContext(
        target_shape,
        close,
        group_counts,
        init_cash,
        cash_sharing,
        call_seq,
        active_mask,
        min_size,
        order_records,
        record_mask,
        last_cash,
        last_shares,
        last_val_price
    )
    prep_out = prep_func_nb(simc, *prep_args)

    from_col = 0
    for group in range(len(group_counts)):
        # Is this group active?
        if np.any(active_mask[:, group]):
            to_col = from_col + group_counts[group]
            group_len = to_col - from_col

            # Run a function to preprocess this entire group
            gc = GroupContext(
                target_shape,
                close,
                group_counts,
                init_cash,
                cash_sharing,
                call_seq,
                active_mask,
                min_size,
                order_records,
                record_mask,
                last_cash,
                last_shares,
                last_val_price,
                group,
                group_len,
                from_col,
                to_col,
                j
            )
            group_prep_out = group_prep_func_nb(gc, *prep_out, *group_prep_args)

            for i in range(target_shape[0]):
                # Is this row segment active?
                if active_mask[i, group]:
                    # Update valuation price
                    if i > 0:
                        for col in range(from_col, to_col):
                            last_val_price[col] = close[i - 1, col]

                    # Run a function to preprocess this group within this row
                    call_seq_now = call_seq[i, from_col:to_col]
                    sc = SegmentContext(
                        target_shape,
                        close,
                        group_counts,
                        init_cash,
                        cash_sharing,
                        call_seq,
                        active_mask,
                        min_size,
                        order_records,
                        record_mask,
                        last_cash,
                        last_shares,
                        last_val_price,
                        i,
                        group,
                        group_len,
                        from_col,
                        to_col,
                        j,
                        call_seq_now
                    )
                    segment_prep_out = segment_prep_func_nb(sc, *group_prep_out, *segment_prep_args)

                    # Get running values per group
                    if cash_sharing:
                        cash_now = last_cash[group]
                        value_now = get_group_value_nb(from_col, to_col, cash_now, last_shares, last_val_price)

                    for k in range(group_len):
                        col_i = call_seq_now[k]
                        if col_i >= group_len:
                            raise ValueError("Call index exceeds bounds of the group")
                        col = from_col + col_i

                        # Get running values per column
                        shares_now = last_shares[col]
                        val_price_now = last_val_price[col]
                        if not cash_sharing:
                            cash_now = last_cash[col]
                            value_now = cash_now
                            if shares_now > 0.:
                                value_now += shares_now * val_price_now

                        # Generate the next order
                        oc = OrderContext(
                            target_shape,
                            close,
                            group_counts,
                            init_cash,
                            cash_sharing,
                            call_seq,
                            active_mask,
                            min_size,
                            order_records,
                            record_mask,
                            last_cash,
                            last_shares,
                            last_val_price,
                            i,
                            group,
                            group_len,
                            from_col,
                            to_col,
                            j,
                            call_seq_now,
                            col,
                            k,
                            cash_now,
                            shares_now,
                            val_price_now,
                            value_now
                        )
                        order = order_func_nb(oc, *segment_prep_out, *order_args)

                        # Convert target value or percent into target shares
                        _size = order.size
                        _size_type = order.size_type
                        if order.size_type == SizeType.TargetPercent:
                            if not np.isnan(order.size):
                                if np.isnan(val_price_now):
                                    raise ValueError("Valuation price is NaN")
                                if np.isnan(value_now):
                                    raise ValueError("Value of the group is NaN")
                            _size = order.size * value_now / val_price_now
                            _size_type = SizeType.TargetShares
                        elif order.size_type == SizeType.TargetValue:
                            if not np.isnan(order.size):
                                if np.isnan(val_price_now):
                                    raise ValueError("Valuation price is NaN")
                            _size = order.size / val_price_now
                            _size_type = SizeType.TargetShares
                        order = Order(
                            _size,
                            _size_type,
                            order.price,
                            order.fees,
                            order.fixed_fees,
                            order.slippage,
                            order.reject_prob
                        )

                        # Process the order
                        cash_now, shares_now, order_result = process_order_nb(
                            cash_now, shares_now, order, min_size[col])

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

                        # Now becomes last
                        if cash_sharing:
                            last_cash[group] = cash_now
                        else:
                            last_cash[col] = cash_now
                        last_shares[col] = shares_now

            from_col = to_col

    # Order records are not sorted yet
    return order_records[record_mask]


@njit
def simulate_row_wise_nb(target_shape, close, group_counts, init_cash, cash_sharing, call_seq,
                         active_mask, min_size, prep_func_nb, prep_args, row_prep_func_nb, row_prep_args,
                         segment_prep_func_nb, segment_prep_args, order_func_nb, order_args):
    """Same as `simulate_nb`, but iterates using row-major order, with the rows
    changing fastest, and the columns/groups changing slowest.

    The main difference is that instead of `group_prep_func_nb` it now exposes `row_prep_func_nb`,
    which is executed per entire row. It should accept `vectorbt.portfolio.enums.RowContext`.

    !!! note
        Function `row_prep_func_nb` is only called if there is at least on active segment in
        the row. Functions `segment_prep_func_nb` and `order_func_nb` are only called if their
        segment is active. If the main task of `row_prep_func_nb` is to activate/deactivate segments,
        all segments should be activated by default to allow `row_prep_func_nb` to be called.

    !!! warning
        You can only safely access data points that are to the left of the current group and
        rows that are to the top of the current row.

    Example:
        Running the same example as in `simulate_nb` but replacing `group_prep_func_nb` for
        `row_prep_func_nb` gives the same results but now the following call hierarchy:
        ```plaintext
        preparing simulation
            preparing row 0
                preparing segment 0 (group)
                    running order 0 at column 0
                    running order 1 at column 1
                    running order 2 at column 2
            preparing row 2
                preparing segment 0 (group)
                    running order 0 at column 1
                    running order 1 at column 2
                    running order 2 at column 0
            preparing row 4
                preparing segment 0 (group)
                    running order 0 at column 0
                    running order 1 at column 2
                    running order 2 at column 1
        ```

        Note, however, that we cannot create NumPy arrays per group anymore as there is no
        `group_prep_func_nb`, so you would need to move this part to `prep_func_nb`,
        make arrays wider, and use only the part of the array that corresponds to the current group.
    """
    check_group_counts(group_counts, target_shape[1])
    check_group_init_cash(group_counts, target_shape[1], init_cash, cash_sharing)

    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    record_mask = np.full(target_shape[0] * target_shape[1], False)
    j = 0
    last_cash = init_cash.astype(np.float_)
    last_shares = np.full(target_shape[1], 0., dtype=np.float_)
    last_val_price = np.full_like(last_shares, np.nan, dtype=np.float_)

    # Run a function to prepare the simulation
    simc = SimulationContext(
        target_shape,
        close,
        group_counts,
        init_cash,
        cash_sharing,
        call_seq,
        active_mask,
        min_size,
        order_records,
        record_mask,
        last_cash,
        last_shares,
        last_val_price
    )
    prep_out = prep_func_nb(simc, *prep_args)

    for i in range(target_shape[0]):
        # Is this row active?
        if np.any(active_mask[i, :]):
            # Update valuation price
            if i > 0:
                for col in range(target_shape[1]):
                    last_val_price[col] = close[i - 1, col]

            # Run a function to preprocess this entire row
            rc = RowContext(
                target_shape,
                close,
                group_counts,
                init_cash,
                cash_sharing,
                call_seq,
                active_mask,
                min_size,
                order_records,
                record_mask,
                last_cash,
                last_shares,
                last_val_price,
                i,
                j
            )
            row_prep_out = row_prep_func_nb(rc, *prep_out, *row_prep_args)

            from_col = 0
            for group in range(len(group_counts)):
                # Is this group segment active?
                if active_mask[i, group]:
                    to_col = from_col + group_counts[group]
                    group_len = to_col - from_col

                    # Run a function to preprocess this row within this group
                    call_seq_now = call_seq[i, from_col:to_col]
                    sc = SegmentContext(
                        target_shape,
                        close,
                        group_counts,
                        init_cash,
                        cash_sharing,
                        call_seq,
                        active_mask,
                        min_size,
                        order_records,
                        record_mask,
                        last_cash,
                        last_shares,
                        last_val_price,
                        i,
                        group,
                        group_len,
                        from_col,
                        to_col,
                        j,
                        call_seq_now
                    )
                    segment_prep_out = segment_prep_func_nb(sc, *row_prep_out, *segment_prep_args)

                    # Get running values per group
                    if cash_sharing:
                        cash_now = last_cash[group]
                        value_now = get_group_value_nb(from_col, to_col, cash_now, last_shares, last_val_price)

                    for k in range(group_len):
                        col_i = call_seq_now[k]
                        if col_i >= group_len:
                            raise ValueError("Call index exceeds bounds of the group")
                        col = from_col + col_i

                        # Get running values per column
                        shares_now = last_shares[col]
                        val_price_now = last_val_price[col]
                        if not cash_sharing:
                            cash_now = last_cash[col]
                            value_now = cash_now
                            if shares_now > 0.:
                                value_now += shares_now * val_price_now

                        # Generate the next order
                        oc = OrderContext(
                            target_shape,
                            close,
                            group_counts,
                            init_cash,
                            cash_sharing,
                            call_seq,
                            active_mask,
                            min_size,
                            order_records,
                            record_mask,
                            last_cash,
                            last_shares,
                            last_val_price,
                            i,
                            group,
                            group_len,
                            from_col,
                            to_col,
                            j,
                            call_seq_now,
                            col,
                            k,
                            cash_now,
                            shares_now,
                            val_price_now,
                            value_now
                        )
                        order = order_func_nb(oc, *segment_prep_out, *order_args)

                        # Convert target value or percent into target shares
                        _size = order.size
                        _size_type = order.size_type
                        if order.size_type == SizeType.TargetPercent:
                            if not np.isnan(order.size):
                                if np.isnan(val_price_now):
                                    raise ValueError("Valuation price is NaN")
                                if np.isnan(value_now):
                                    raise ValueError("Value of the group is NaN")
                            _size = order.size * value_now / val_price_now
                            _size_type = SizeType.TargetShares
                        elif order.size_type == SizeType.TargetValue:
                            if not np.isnan(order.size):
                                if np.isnan(val_price_now):
                                    raise ValueError("Valuation price is NaN")
                            _size = order.size / val_price_now
                            _size_type = SizeType.TargetShares
                        order = Order(
                            _size,
                            _size_type,
                            order.price,
                            order.fees,
                            order.fixed_fees,
                            order.slippage,
                            order.reject_prob
                        )

                        # Process the order
                        cash_now, shares_now, order_result = process_order_nb(
                            cash_now, shares_now, order, min_size[col])

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

                        # Now becomes last
                        if cash_sharing:
                            last_cash[group] = cash_now
                        else:
                            last_cash[col] = cash_now
                        last_shares[col] = shares_now

                    from_col = to_col

    # Order records are not sorted yet
    return order_records[record_mask]


@njit(cache=True)
def simulate_from_signals_nb(target_shape, group_counts, init_cash, call_seq, entries, exits, size,
                             entry_price, exit_price, fees, fixed_fees, slippage, reject_prob, min_size,
                             accumulate, accumulate_exit_mode, conflict_mode, is_2d):
    """Adaptation of `simulate_nb` for simulation based on entry and exit signals.

    Utilizes flexible broadcasting.

    !!! note
        Should be only grouped if cash sharing is enabled."""
    check_group_counts(group_counts, target_shape[1])
    cash_sharing = is_grouped_nb(group_counts)
    check_group_init_cash(group_counts, target_shape[1], init_cash, cash_sharing)

    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    record_mask = np.full(target_shape[0] * target_shape[1], False)
    j = 0
    last_cash = init_cash.astype(np.float_)
    last_shares = np.full(target_shape[1], 0., dtype=np.float_)

    # Inputs were not broadcast -> use flexible indexing
    flex_i1, flex_col1 = flex_choose_i_and_col_nb(entries, is_2d)
    flex_i2, flex_col2 = flex_choose_i_and_col_nb(exits, is_2d)
    flex_i3, flex_col3 = flex_choose_i_and_col_nb(size, is_2d)
    flex_i4, flex_col4 = flex_choose_i_and_col_nb(entry_price, is_2d)
    flex_i5, flex_col5 = flex_choose_i_and_col_nb(exit_price, is_2d)
    flex_i6, flex_col6 = flex_choose_i_and_col_nb(fees, is_2d)
    flex_i7, flex_col7 = flex_choose_i_and_col_nb(fixed_fees, is_2d)
    flex_i8, flex_col8 = flex_choose_i_and_col_nb(slippage, is_2d)
    flex_i9, flex_col9 = flex_choose_i_and_col_nb(reject_prob, is_2d)

    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        group_len = to_col - from_col

        # Get running values per group
        if cash_sharing:
            cash_now = last_cash[group]

        for i in range(target_shape[0]):
            for k in range(group_len):
                col = from_col + k
                if cash_sharing:
                    col_i = call_seq[i, col]
                    if col_i >= group_len:
                        raise ValueError("Call index exceeds bounds of the group")
                    col = from_col + col_i

                # Get running values per column
                if not cash_sharing:
                    cash_now = last_cash[col]
                shares_now = last_shares[col]

                is_entry = flex_select_nb(i, col, entries, flex_i1, flex_col1, is_2d)
                is_exit = flex_select_nb(i, col, exits, flex_i2, flex_col2, is_2d)
                if is_entry or is_exit:
                    # Generate the next order
                    order = signals_order_func_nb(
                        shares_now,
                        is_entry,
                        is_exit,
                        flex_select_nb(i, col, size, flex_i3, flex_col3, is_2d),
                        flex_select_nb(i, col, entry_price, flex_i4, flex_col4, is_2d),
                        flex_select_nb(i, col, exit_price, flex_i5, flex_col5, is_2d),
                        flex_select_nb(i, col, fees, flex_i6, flex_col6, is_2d),
                        flex_select_nb(i, col, fixed_fees, flex_i7, flex_col7, is_2d),
                        flex_select_nb(i, col, slippage, flex_i8, flex_col8, is_2d),
                        flex_select_nb(i, col, reject_prob, flex_i9, flex_col9, is_2d),
                        accumulate,
                        accumulate_exit_mode,
                        conflict_mode
                    )

                    # Process the order
                    cash_now, shares_now, order_result = process_order_nb(
                        cash_now, shares_now, order, min_size[col])

                    if order_result.status == OrderStatus.Filled:
                        # Add a new record
                        if cash_sharing:
                            r = get_record_idx_nb(target_shape, col, i)
                        else:
                            r = j
                        order_records[r]['col'] = col
                        order_records[r]['idx'] = i
                        order_records[r]['size'] = order_result.size
                        order_records[r]['price'] = order_result.price
                        order_records[r]['fees'] = order_result.fees
                        order_records[r]['side'] = order_result.side
                        if cash_sharing:
                            record_mask[r] = True
                        j += 1

                # Now becomes last
                if cash_sharing:
                    last_cash[group] = cash_now
                else:
                    last_cash[col] = cash_now
                last_shares[col] = shares_now

        from_col = to_col

    # Order records are not sorted yet
    if cash_sharing:
        return order_records[record_mask]
    return order_records[:j]


@njit(cache=True)
def signals_order_func_nb(shares_now, is_entry, is_exit, size, entry_price, exit_price, fees, fixed_fees,
                          slippage, reject_prob, accumulate, accumulate_exit_mode, conflict_mode):
    """`order_func_nb` of `simulate_from_signals_nb`."""
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
                order_size = abs(size) - shares_now
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
    return Order(
        order_size,
        SizeType.Shares,
        order_price,
        fees,
        fixed_fees,
        slippage,
        reject_prob
    )


@njit(cache=True)
def simulate_from_orders_nb(target_shape, group_counts, init_cash, call_seq, size, size_type,
                            price, fees, fixed_fees, slippage, reject_prob, min_size, val_price,
                            auto_call_seq, is_2d):
    """Adaptation of `simulate_nb` for simulation based on orders.

    Utilizes flexible broadcasting.

    !!! note
        Should be only grouped if cash sharing is enabled.

        If `auto_call_seq` is True, make sure that `call_seq` follows `CallSeqType.Default`."""
    check_group_counts(group_counts, target_shape[1])
    cash_sharing = is_grouped_nb(group_counts)
    check_group_init_cash(group_counts, target_shape[1], init_cash, cash_sharing)

    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    record_mask = np.full(target_shape[0] * target_shape[1], False)
    j = 0
    last_cash = init_cash.astype(np.float_)
    last_shares = np.full(target_shape[1], 0., dtype=np.float_)
    temp_order_value = np.empty(target_shape[1], dtype=np.float_)

    # Inputs were not broadcast -> use flexible indexing
    flex_i1, flex_col1 = flex_choose_i_and_col_nb(size, is_2d)
    flex_i2, flex_col2 = flex_choose_i_and_col_nb(size_type, is_2d)
    flex_i3, flex_col3 = flex_choose_i_and_col_nb(price, is_2d)
    flex_i4, flex_col4 = flex_choose_i_and_col_nb(fees, is_2d)
    flex_i5, flex_col5 = flex_choose_i_and_col_nb(fixed_fees, is_2d)
    flex_i6, flex_col6 = flex_choose_i_and_col_nb(slippage, is_2d)
    flex_i7, flex_col7 = flex_choose_i_and_col_nb(reject_prob, is_2d)
    flex_i8, flex_col8 = flex_choose_i_and_col_nb(val_price, is_2d)

    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        group_len = to_col - from_col

        # Get running values per group
        if cash_sharing:
            cash_now = last_cash[group]

        for i in range(target_shape[0]):
            # Calculate group value and rearrange if cash sharing is enabled
            if cash_sharing:
                # Same as get_group_value_ctx_nb but with flexible indexing
                value_now = cash_now
                for k in range(group_len):
                    col = from_col + k
                    if last_shares[col] > 0.:
                        _val_price = flex_select_nb(i, col, val_price, flex_i8, flex_col8, is_2d)
                        holding_value = last_shares[col] * _val_price
                        value_now += holding_value

                # Dynamically sort by order value -> selling comes first to release funds early
                if auto_call_seq:
                    # Same as sort_by_order_value_ctx_nb but with flexible indexing
                    for k in range(group_len):
                        col = from_col + k
                        _size = flex_select_nb(i, col, size, flex_i1, flex_col1, is_2d)
                        _size_type = flex_select_nb(i, col, size_type, flex_i2, flex_col2, is_2d)
                        _val_price = flex_select_nb(i, col, val_price, flex_i8, flex_col8, is_2d)
                        holding_value_now = last_shares[col] * _val_price

                        if _size_type == SizeType.Shares:
                            temp_order_value[k] = _size * _val_price
                        if _size_type == SizeType.TargetShares:
                            temp_order_value[k] = _size * _val_price - holding_value_now
                        if _size_type == SizeType.TargetValue:
                            temp_order_value[k] = _size - holding_value_now
                        if _size_type == SizeType.TargetPercent:
                            temp_order_value[k] = _size * value_now - holding_value_now

                    # Sort by order value
                    insert_argsort_nb(temp_order_value[:group_len], call_seq[i, from_col:to_col])

            for k in range(group_len):
                if cash_sharing:
                    col_i = call_seq[i, from_col + k]
                    if col_i >= group_len:
                        raise ValueError("Call index exceeds bounds of the group")
                    col = from_col + col_i
                else:
                    col = from_col + k

                # Get running values per column
                shares_now = last_shares[col]
                _val_price = flex_select_nb(i, col, val_price, flex_i8, flex_col8, is_2d)
                if not cash_sharing:
                    cash_now = last_cash[col]
                    value_now = cash_now
                    if shares_now > 0.:
                        value_now += shares_now * _val_price

                # Convert target value or percent into target shares
                _size = flex_select_nb(i, col, size, flex_i1, flex_col1, is_2d)
                _size_type = flex_select_nb(i, col, size_type, flex_i2, flex_col2, is_2d)
                if _size_type == SizeType.TargetPercent:
                    if not np.isnan(_size):
                        if np.isnan(_val_price):
                            raise ValueError("Valuation price is NaN")
                        if np.isnan(value_now):
                            raise ValueError("Value of the group is NaN")
                    _size = _size * value_now / _val_price
                    _size_type = SizeType.TargetShares
                elif _size_type == SizeType.TargetValue:
                    if not np.isnan(_size):
                        if np.isnan(_val_price):
                            raise ValueError("Valuation price is NaN")
                    _size = _size / _val_price
                    _size_type = SizeType.TargetShares

                # Generate the next order
                order = Order(
                    _size,
                    _size_type,
                    flex_select_nb(i, col, price, flex_i3, flex_col3, is_2d),
                    flex_select_nb(i, col, fees, flex_i4, flex_col4, is_2d),
                    flex_select_nb(i, col, fixed_fees, flex_i5, flex_col5, is_2d),
                    flex_select_nb(i, col, slippage, flex_i6, flex_col6, is_2d),
                    flex_select_nb(i, col, reject_prob, flex_i7, flex_col7, is_2d)
                )

                # Process the order
                cash_now, shares_now, order_result = process_order_nb(
                    cash_now, shares_now, order, min_size[col])

                if order_result.status == OrderStatus.Filled:
                    # Add a new record
                    if cash_sharing:
                        r = get_record_idx_nb(target_shape, col, i)
                    else:
                        r = j
                    order_records[r]['col'] = col
                    order_records[r]['idx'] = i
                    order_records[r]['size'] = order_result.size
                    order_records[r]['price'] = order_result.price
                    order_records[r]['fees'] = order_result.fees
                    order_records[r]['side'] = order_result.side
                    if cash_sharing:
                        record_mask[r] = True
                    j += 1

                # Now becomes last
                if cash_sharing:
                    last_cash[group] = cash_now
                else:
                    last_cash[col] = cash_now
                last_shares[col] = shares_now

        from_col = to_col

    # Order records are not sorted yet
    if cash_sharing:
        return order_records[record_mask]
    return order_records[:j]


# ############# Shares ############# #

@njit(cache=True)
def share_flow_nb(target_shape, order_records):
    """Get share flow series per column."""
    out = np.full(target_shape, 0., dtype=np.float_)
    for r in range(order_records.shape[0]):
        record = order_records[r]
        if record['side'] == OrderSide.Buy:
            out[record['idx'], record['col']] += record['size']
        else:
            out[record['idx'], record['col']] -= record['size']
    return out


@njit(cache=True)
def shares_nb(share_flow):
    """Get share series per column."""
    out = np.empty_like(share_flow)
    for col in range(share_flow.shape[1]):
        shares_now = 0.
        for i in range(share_flow.shape[0]):
            flow_value = share_flow[i, col]
            if flow_value < 0 and is_close_or_less_nb(shares_now, -flow_value):
                shares_now = 0.  # numerical stability
            else:
                shares_now += flow_value
            out[i, col] = shares_now
    return out


@njit(cache=True)
def holding_mask_grouped_nb(shares, group_counts):
    """Get holding mask per group."""
    check_group_counts(group_counts, shares.shape[1])

    out = np.empty((shares.shape[0], len(group_counts)), dtype=np.bool_)
    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        out[:, group] = np.sum(shares[:, from_col:to_col], axis=1) > 0
        from_col = to_col
    return out


# ############# Cash ############# #

@njit(cache=True)
def init_cash_grouped_nb(init_cash, group_counts, cash_sharing):
    """Get initial cash per group."""
    if cash_sharing:
        return init_cash
    out = np.empty(group_counts.shape, dtype=np.float_)
    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        cash_sum = 0.
        for col in range(from_col, to_col):
            cash_sum += init_cash[col]
        out[group] = cash_sum
        from_col = to_col
    return out


@njit(cache=True)
def init_cash_ungrouped_nb(init_cash, group_counts, cash_sharing):
    """Get initial cash per column."""
    if not cash_sharing:
        return init_cash
    group_counts_cs = np.cumsum(group_counts)
    out = np.full(group_counts_cs[-1], np.nan, dtype=np.float_)
    out[group_counts_cs - group_counts] = init_cash
    out = generic_nb.ffill_1d_nb(out)
    return out


@njit(cache=True)
def cash_flow_ungrouped_nb(target_shape, order_records):
    """Get cash flow series per column."""
    out = np.full(target_shape, 0., dtype=np.float_)
    for r in range(order_records.shape[0]):
        record = order_records[r]
        if record['side'] == OrderSide.Buy:
            out[record['idx'], record['col']] -= record['size'] * record['price'] + record['fees']
        else:
            out[record['idx'], record['col']] += record['size'] * record['price'] - record['fees']
    return out


@njit(cache=True)
def cash_flow_grouped_nb(cash_flow_ungrouped, group_counts):
    """Get cash flow series per group."""
    check_group_counts(group_counts, cash_flow_ungrouped.shape[1])

    out = np.empty((cash_flow_ungrouped.shape[0], len(group_counts)), dtype=np.float_)
    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        out[:, group] = np.sum(cash_flow_ungrouped[:, from_col:to_col], axis=1)
        from_col = to_col
    return out


@njit(cache=True)
def cash_ungrouped_nb(cash_flow_ungrouped, group_counts, init_cash, call_seq, in_sim_order):
    """Get cash series per column.

    `init_cash` should be grouped if `in_sim_order` is True, and ungrouped otherwise."""
    check_group_counts(group_counts, cash_flow_ungrouped.shape[1])

    out = np.empty_like(cash_flow_ungrouped)
    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        group_len = to_col - from_col
        if in_sim_order:
            cash_now = init_cash[group]
        for i in range(cash_flow_ungrouped.shape[0]):
            for k in range(group_len):
                col = from_col + call_seq[i, from_col + k]
                if not in_sim_order:
                    cash_now = init_cash[col] if i == 0 else out[i - 1, col]
                flow_value = cash_flow_ungrouped[i, col]
                if flow_value < 0 and is_close_or_less_nb(cash_now, -flow_value):
                    cash_now = 0.  # numerical stability
                else:
                    cash_now += flow_value
                out[i, col] = cash_now
        from_col = to_col
    return out


@njit(cache=True)
def cash_grouped_nb(target_shape, cash_flow_grouped, group_counts, init_cash_grouped):
    """Get cash series per group."""
    check_group_counts(group_counts, target_shape[1])

    out = np.empty_like(cash_flow_grouped)
    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        cash_now = init_cash_grouped[group]
        for i in range(cash_flow_grouped.shape[0]):
            flow_value = cash_flow_grouped[i, group]
            if flow_value < 0 and is_close_or_less_nb(cash_now, -flow_value):
                cash_now = 0.  # numerical stability
            else:
                cash_now += flow_value
            out[i, group] = cash_now
        from_col = to_col
    return out


# ############# Performance ############# #


@njit(cache=True)
def holding_value_grouped_nb(close, shares, group_counts):
    """Get holding value series per group."""
    check_group_counts(group_counts, close.shape[1])

    out = np.empty((close.shape[0], len(group_counts)), dtype=np.float_)
    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        out[:, group] = np.sum(shares[:, from_col:to_col] * close[:, from_col:to_col], axis=1)
        from_col = to_col
    return out


@njit(cache=True)
def holding_value_ungrouped_nb(close, shares):
    """Get holding value series per column."""
    return close * shares


@njit(cache=True)
def value_in_sim_order_nb(cash_ungrouped, holding_value_ungrouped, group_counts, call_seq):
    """Get portfolio value series in simulation order."""
    check_group_counts(group_counts, cash_ungrouped.shape[1])

    out = np.empty_like(cash_ungrouped)
    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        group_len = to_col - from_col
        curr_holding_value = 0.
        # Without correctly treating NaN values, after one NaN all will be NaN
        since_last_nan = group_len
        for j in range(cash_ungrouped.shape[0] * group_len):
            i = j // group_len
            col = from_col + call_seq[i, from_col + j % group_len]
            if j >= group_len:
                prev_j = j - group_len
                prev_i = prev_j // group_len
                prev_col = from_col + call_seq[prev_i, from_col + prev_j % group_len]
                if not np.isnan(holding_value_ungrouped[prev_i, prev_col]):
                    curr_holding_value -= holding_value_ungrouped[prev_i, prev_col]
            if np.isnan(holding_value_ungrouped[i, col]):
                since_last_nan = 0
            else:
                curr_holding_value += holding_value_ungrouped[i, col]
            if since_last_nan < group_len:
                out[i, col] = np.nan
            else:
                out[i, col] = cash_ungrouped[i, col] + curr_holding_value
            since_last_nan += 1

        from_col = to_col
    return out


@njit(cache=True)
def value_nb(cash, holding_value):
    """Get portfolio value series per column/group."""
    return cash + holding_value


@njit
def total_profit_ungrouped_nb(target_shape, close, order_records, init_cash_ungrouped):
    """Get total profit per column.

    A much faster version than the one based on `value_nb`."""
    shares = np.full(target_shape[1], 0., dtype=np.float_)
    cash = init_cash_ungrouped.copy()

    prev_col = -1
    prev_i = -1
    for r in range(order_records.shape[0]):
        record = order_records[r]
        col = record['col']
        i = record['idx']

        if col < prev_col:
            raise ValueError("Order records must be sorted")
        if col != prev_col:
            prev_i = -1
        if i < prev_i:
            raise ValueError("Order records must be sorted")

        if record['side'] == OrderSide.Buy:
            shares[col] += record['size']
        else:
            order_size = record['size']
            if is_close_or_less_nb(shares[col], order_size):
                shares[col] = 0.  # numerical stability
            else:
                shares[col] -= order_size
        if record['side'] == OrderSide.Buy:
            order_cash = record['size'] * record['price'] + record['fees']
            if is_close_or_less_nb(cash[col], order_cash):
                cash[col] = 0.  # numerical stability
            else:
                cash[col] -= order_cash
        else:
            cash[col] += record['size'] * record['price'] - record['fees']

    return cash + shares * close[-1, :] - init_cash_ungrouped


@njit
def total_profit_grouped_nb(total_profit_ungrouped, group_counts):
    """Get total profit per group."""
    check_group_counts(group_counts, total_profit_ungrouped.shape[0])

    out = np.empty((len(group_counts),), dtype=np.float_)
    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        out[group] = np.sum(total_profit_ungrouped[from_col:to_col])
        from_col = to_col
    return out


@njit(cache=True)
def final_value_nb(total_profit, init_cash_regrouped):
    """Get total profit per column/group."""
    return total_profit + init_cash_regrouped


@njit(cache=True)
def total_return_nb(total_profit, init_cash_regrouped):
    """Get total return per column/group."""
    return total_profit / init_cash_regrouped


@njit(cache=True)
def active_returns_nb(cash_flow, holding_value):
    """Get active return series per column/group."""
    out = np.empty_like(cash_flow)
    for col in range(cash_flow.shape[1]):
        for i in range(cash_flow.shape[0]):
            input_value = 0. if i == 0 else holding_value[i - 1, col]
            if cash_flow[i, col] < 0.:
                input_value += -cash_flow[i, col]
            if input_value == 0.:
                out[i, col] = 0.
                continue
            output_value = holding_value[i, col]
            if cash_flow[i, col] > 0.:
                output_value += cash_flow[i, col]
            out[i, col] = (output_value - input_value) / input_value
    return out


@njit(cache=True)
def returns_nb(value, init_cash_regrouped):
    """Get portfolio return series per column/group."""
    out = np.empty(value.shape, dtype=np.float_)
    out[0, :] = (value[0, :] - init_cash_regrouped) / init_cash_regrouped
    out[1:, :] = (value[1:, :] - value[:-1, :]) / value[:-1, :]
    return out


@njit(cache=True)
def returns_in_sim_order_nb(value_iso, group_counts, init_cash_grouped, call_seq):
    """Get portfolio return series in simulation order."""
    check_group_counts(group_counts, value_iso.shape[1])

    out = np.empty_like(value_iso)
    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        group_len = to_col - from_col
        input_value = init_cash_grouped[group]
        for j in range(value_iso.shape[0] * group_len):
            i = j // group_len
            col = from_col + call_seq[i, from_col + j % group_len]
            output_value = value_iso[i, col]
            out[i, col] = (output_value - input_value) / input_value
            input_value = output_value
        from_col = to_col
    return out


@njit(cache=True)
def buy_and_hold_return_ungrouped_nb(close):
    """Get total return value of buying and holding per column."""
    return (close[-1, :] - close[0, :]) / close[0, :]


@njit(cache=True)
def buy_and_hold_return_grouped_nb(close, group_counts):
    """Get total return value of buying and holding per group."""
    check_group_counts(group_counts, close.shape[1])

    out = np.empty(len(group_counts), dtype=np.float_)
    total_return = (close[-1, :] - close[0, :]) / close[0, :]
    from_col = 0
    for group in range(len(group_counts)):
        to_col = from_col + group_counts[group]
        group_len = to_col - from_col
        out[group] = np.sum(total_return[from_col:to_col]) / group_len
        from_col = to_col
    return out
