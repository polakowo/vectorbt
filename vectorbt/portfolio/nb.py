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

    While vectorbt has implemented tolerance checks when comparing floats for equality,
    adding/subtracting small amounts large number of times may still introduce a noticable
    error that cannot be corrected post factum.

    To mitigate this issue, avoid repeating lots of micro-transactions of the same sign.
    For example, reduce by `np.inf` or `shares_now` to close a long/short position.

    See `vectorbt.utils.math` for current tolerance values.
"""

import numpy as np
from numba import njit

from vectorbt.utils.math import (
    is_close_or_less_nb,
    is_less_nb,
    add_nb
)
from vectorbt.utils.array import insert_argsort_nb
from vectorbt.base.reshape_fns import flex_select_auto_nb
from vectorbt.generic import nb as generic_nb
from vectorbt.portfolio.enums import (
    SimulationContext,
    GroupContext,
    RowContext,
    SegmentContext,
    OrderContext,
    CallSeqType,
    SizeType,
    ConflictMode,
    Direction,
    Order,
    NoOrder,
    OrderStatus,
    OrderResult,
    IgnoredOrder,
    RejectedOrder,
    RejectedOrderError
)
from vectorbt.records.enums import (
    OrderSide,
    order_dt
)


# ############# Simulation ############# #

@njit(cache=True)
def buy_shares_nb(cash_now, shares_now, price, size, fees, fixed_fees, slippage,
                  min_size, allow_partial, raise_by_reject, direction):
    """Buy shares."""

    # Get optimal order size
    if direction == Direction.Short:
        size = min(-shares_now, size)

    # Get price adjusted with slippage
    adj_price = price * (1 + slippage)

    # Get cash required to complete this order
    req_cash = size * adj_price
    req_fees = req_cash * fees + fixed_fees
    adj_req_cash = req_cash + req_fees

    if is_close_or_less_nb(adj_req_cash, cash_now):
        # Sufficient cash
        final_size = size
        fees_paid = req_fees
        final_cash = adj_req_cash

        # Update current cash
        cash_now = add_nb(cash_now, -final_cash)
    else:
        # Insufficient cash, size will be less than requested
        if not np.isinf(size) and not allow_partial:
            # np.inf doesn't count
            if raise_by_reject:
                raise RejectedOrderError("Order rejected: Partial fill")
            return cash_now, shares_now, RejectedOrder
        if is_close_or_less_nb(cash_now, fixed_fees):
            # Can't fill
            if raise_by_reject:
                raise RejectedOrderError("Order rejected: Insufficient funds")
            return cash_now, shares_now, RejectedOrder

        # For fees of 10% and 1$ per transaction, you can buy shares for 90$ (effect_cash)
        # to spend 100$ (adj_req_cash) in total
        final_cash = (cash_now - fixed_fees) / (1 + fees)

        # Update size and fees
        final_size = final_cash / adj_price
        fees_paid = cash_now - final_cash

        # Update current cash
        cash_now = 0.  # numerical stability

    # Check against minimum size
    if abs(final_size) < min_size:
        if raise_by_reject:
            raise RejectedOrderError("Order rejected: Size is less than minimum allowed")
        return cash_now, shares_now, RejectedOrder

    # Update current shares
    shares_now = add_nb(shares_now, final_size)

    # Return filled order
    return cash_now, \
           shares_now, \
           OrderResult(
               final_size,
               adj_price,
               fees_paid,
               OrderSide.Buy,
               OrderStatus.Filled
           )


@njit(cache=True)
def sell_shares_nb(cash_now, shares_now, price, size, fees, fixed_fees, slippage,
                   min_size, raise_by_reject, direction):
    """Sell shares."""

    # Get optimal order size
    if direction == Direction.Long:
        final_size = min(shares_now, size)
    else:
        final_size = size

    # Check against minimum size
    if abs(final_size) < min_size:
        if raise_by_reject:
            raise RejectedOrderError("Order rejected: Size is less than minimum allowed")
        return cash_now, shares_now, RejectedOrder

    # Get price adjusted with slippage
    adj_price = price * (1 - slippage)

    # Compute acquired cash
    acq_cash = final_size * adj_price

    # Update fees
    fees_paid = acq_cash * fees + fixed_fees

    # Get final cash by subtracting costs
    if is_less_nb(acq_cash, fees_paid):
        # Can't fill
        if raise_by_reject:
            raise RejectedOrderError("Order rejected: Cost outweighs the benefits")
        return cash_now, shares_now, RejectedOrder
    final_cash = acq_cash - fees_paid

    # Update current cash and shares
    cash_now += final_cash
    shares_now = add_nb(shares_now, -final_size)

    # Return filled order
    return cash_now, \
           shares_now, \
           OrderResult(
               final_size,
               adj_price,
               fees_paid,
               OrderSide.Sell,
               OrderStatus.Filled
           )


@njit(cache=True)
def process_order_nb(cash_now, shares_now, val_price_now, value_now, order):
    """Process an order given current cash and share balance.

    Ignores order if size is 0 or size, price, value or valuation price is NaN.

    Args:
        cash_now (float): Cash available to this asset or group with cash sharing.
        shares_now (float): Holdings of this particular asset.
        val_price_now (float): Valuation price for this particular asset.
        value_now (float): Value of this asset or group with cash sharing.
        order (Order): See `vectorbt.portfolio.enums.Order`.
    """
    if np.isnan(order.size) or np.isnan(order.price):
        return cash_now, shares_now, IgnoredOrder

    # Check variables
    if cash_now < 0:
        raise ValueError("cash_now must be greater than 0")
    if not np.isfinite(shares_now):
        raise ValueError("shares_now must be finite")
    if np.isinf(val_price_now) or val_price_now <= 0:
        raise ValueError("val_price_now must be finite and greater than 0")

    # Check order
    if not np.isfinite(order.price) or order.price <= 0:
        raise ValueError("order.price must be finite and greater than 0")
    if not np.isfinite(order.fees) or order.fees < 0:
        raise ValueError("order.fees must be finite and 0 or greater")
    if not np.isfinite(order.fixed_fees) or order.fixed_fees < 0:
        raise ValueError("order.fixed_fees must be finite and 0 or greater")
    if not np.isfinite(order.slippage) or order.slippage < 0:
        raise ValueError("order.slippage must be finite and 0 or greater")
    if not np.isfinite(order.min_size) or order.min_size < 0:
        raise ValueError("order.min_size must be finite and 0 or greater")
    if order.max_size <= 0:
        raise ValueError("order.max_size must be greater than 0")
    if not np.isfinite(order.reject_prob) or order.reject_prob < 0 or order.reject_prob > 1:
        raise ValueError("order.reject_prob must be between 0 and 1")
    if order.allow_partial != 0 and order.allow_partial != 1:
        raise ValueError("order.allow_partial must be boolean")
    if order.raise_by_reject != 0 and order.raise_by_reject != 1:
        raise ValueError("order.raise_by_reject must be boolean")
    if order.direction == Direction.Long and shares_now < 0:
        raise ValueError("shares_now is negative but order.direction is Direction.Long")
    if order.direction == Direction.Short and shares_now > 0:
        raise ValueError("shares_now is positive but order.direction is Direction.Short")

    order_size = order.size
    order_size_type = order.size_type

    if order.direction == Direction.Short:
        # Positive size in short direction should be treated as negative
        order_size *= -1

    if order.direction == Direction.Short or order.direction == Direction.All:
        if order_size < 0 and np.isinf(order_size):
            # Go full short, depends upon current value
            order_size = -1.
            order_size_type = SizeType.TargetPercent

    if order_size_type == SizeType.TargetPercent:
        # Target percentage of current value
        if np.isnan(value_now):
            return cash_now, shares_now, IgnoredOrder

        order_size *= value_now
        order_size_type = SizeType.TargetValue

    if order_size_type == SizeType.TargetValue:
        # Target value
        if np.isnan(val_price_now):
            return cash_now, shares_now, IgnoredOrder

        order_size = order_size / val_price_now
        order_size_type = SizeType.TargetShares

    if order_size_type == SizeType.TargetShares:
        # Target amount of shares
        order_size -= shares_now

    if order_size == 0:
        return cash_now, shares_now, IgnoredOrder

    if abs(order_size) > order.max_size:
        if not order.allow_partial:
            if order.raise_by_reject:
                raise RejectedOrderError("Order rejected: Size is greater than maximum allowed")
            return cash_now, shares_now, RejectedOrder

        order_size = np.sign(order_size) * order.max_size

    if order.reject_prob > 0:
        if np.random.uniform(0, 1) < order.reject_prob:
            if order.raise_by_reject:
                raise RejectedOrderError("Order rejected: Random event")
            return cash_now, shares_now, RejectedOrder

    if order_size > 0:
        if order.direction == Direction.Long or order.direction == Direction.All:
            if np.isinf(order_size) and np.isinf(cash_now):
                raise ValueError("Attempt to go in long direction indefinitely. Set max_size or finite init_cash.")
        else:
            if shares_now == 0:
                return cash_now, shares_now, IgnoredOrder

        return buy_shares_nb(
            cash_now,
            shares_now,
            order.price,
            order_size,
            order.fees,
            order.fixed_fees,
            order.slippage,
            order.min_size,
            order.allow_partial,
            order.raise_by_reject,
            order.direction
        )
    else:
        if order.direction == Direction.Short or order.direction == Direction.All:
            if np.isinf(order_size):
                raise ValueError("Attempt to go in short direction indefinitely. Set max_size or finite init_cash.")
        else:
            if shares_now == 0:
                return cash_now, shares_now, IgnoredOrder

        return sell_shares_nb(
            cash_now,
            shares_now,
            order.price,
            -order_size,
            order.fees,
            order.fixed_fees,
            order.slippage,
            order.min_size,
            order.raise_by_reject,
            order.direction
        )


@njit(cache=True)
def create_order_nb(size=np.nan,
                 size_type=SizeType.Shares,
                 price=np.nan,
                 fees=0.,
                 fixed_fees=0.,
                 slippage=0.,
                 min_size=0.,
                 max_size=np.inf,
                 reject_prob=0.,
                 allow_partial=True,
                 raise_by_reject=False,
                 direction=Direction.All):
    """Convenience function to create an order with some defaults."""

    return Order(
        float(size),
        size_type,
        float(price),
        float(fees),
        float(fixed_fees),
        float(slippage),
        float(min_size),
        float(max_size),
        float(reject_prob),
        bool(allow_partial),
        bool(raise_by_reject),
        direction
    )


@njit(cache=True)
def order_nothing():
    """Convenience function to order nothing."""
    return NoOrder


@njit(cache=True)
def check_group_lens(group_lens, n_cols):
    """Check `group_lens`."""
    if np.sum(group_lens) != n_cols:
        raise ValueError("group_lens has incorrect total number of columns")


@njit(cache=True)
def check_group_init_cash(group_lens, n_cols, init_cash, cash_sharing):
    """Check `init_cash`."""
    if cash_sharing:
        if len(init_cash) != len(group_lens):
            raise ValueError("If cash sharing is enabled, init_cash must match the number of groups")
    else:
        if len(init_cash) != n_cols:
            raise ValueError("If cash sharing is disabled, init_cash must match the number of columns")


@njit(cache=True)
def get_record_idx_nb(target_shape, i, col):
    """Get record index by position of order in the matrix."""
    return col * target_shape[0] + i


@njit(cache=True)
def is_grouped_nb(group_lens):
    """Check if columm,ns are grouped, that is, more than one column per group."""
    return np.any(group_lens > 1)


@njit(cache=True)
def shuffle_call_seq_nb(call_seq, group_lens):
    """Shuffle the call sequence array."""
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        for i in range(call_seq.shape[0]):
            np.random.shuffle(call_seq[i, from_col:to_col])
        from_col = to_col


@njit(cache=True)
def build_call_seq_nb(target_shape, group_lens, call_seq_type=CallSeqType.Default):
    """Build a new call sequence array."""
    if call_seq_type == CallSeqType.Reversed:
        out = np.full(target_shape[1], 1, dtype=np.int_)
        out[np.cumsum(group_lens)[1:] - group_lens[1:] - 1] -= group_lens[1:]
        out = np.cumsum(out[::-1])[::-1] - 1
        out = out * np.ones((target_shape[0], 1), dtype=np.int_)
        return out
    out = np.full(target_shape[1], 1, dtype=np.int_)
    out[np.cumsum(group_lens)[:-1]] -= group_lens[:-1]
    out = np.cumsum(out) - 1
    out = out * np.ones((target_shape[0], 1), dtype=np.int_)
    if call_seq_type == CallSeqType.Random:
        shuffle_call_seq_nb(out, group_lens)
    return out


def require_call_seq(call_seq):
    """Force the call sequence array to pass our requirements."""
    return np.require(call_seq, dtype=np.int_, requirements=['A', 'O', 'W', 'F'])


def build_call_seq(target_shape, group_lens, call_seq_type=CallSeqType.Default):
    """Not compiled but faster version of `build_call_seq_nb`."""
    call_seq = np.full(target_shape[1], 1, dtype=np.int_)
    if call_seq_type == CallSeqType.Reversed:
        call_seq[np.cumsum(group_lens)[1:] - group_lens[1:] - 1] -= group_lens[1:]
        call_seq = np.cumsum(call_seq[::-1])[::-1] - 1
    else:
        call_seq[np.cumsum(group_lens[:-1])] -= group_lens[:-1]
        call_seq = np.cumsum(call_seq) - 1
    call_seq = np.broadcast_to(call_seq, target_shape)
    if call_seq_type == CallSeqType.Random:
        call_seq = require_call_seq(call_seq)
        shuffle_call_seq_nb(call_seq, group_lens)
    return require_call_seq(call_seq)


@njit(cache=True)
def empty_prep_nb(context, *args):
    """Preparation function that forwards received arguments down the stack."""
    return args


@njit(cache=True)
def get_group_value_nb(from_col, to_col, cash_now, last_shares, last_val_price):
    """Get group value."""
    group_value = cash_now
    group_len = to_col - from_col
    for k in range(group_len):
        col = from_col + k
        if last_shares[col] != 0:
            group_value += last_shares[col] * last_val_price[col]
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
def get_order_value_nb(size, size_type, shares_now, val_price_now, value_now, direction):
    """Get potential value of an order."""
    if direction == Direction.Short:
        size *= -1
    holding_value_now = shares_now * val_price_now
    if size_type == SizeType.Shares:
        return size * val_price_now
    if size_type == SizeType.TargetShares:
        return size * val_price_now - holding_value_now
    if size_type == SizeType.TargetValue:
        return size - holding_value_now
    if size_type == SizeType.TargetPercent:
        return size * value_now - holding_value_now
    return np.nan


@njit(cache=True)
def auto_call_seq_ctx_nb(sc, size, size_type, direction, temp_float_arr):
    """Generate call sequence based on order value dynamically, for example, to rebalance.

    Accepts `vectorbt.portfolio.enums.SegmentContext`.

    Arrays `size`, `size_type`, `direction` and `temp_float_arr` should match the number
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
            size[k],
            size_type[k],
            sc.last_shares[col],
            sc.last_val_price[col],
            group_value_now,
            direction[k]
        )
    # Sort by order value
    insert_argsort_nb(temp_float_arr, sc.call_seq_now)


@njit
def simulate_nb(target_shape, close, group_lens, init_cash, cash_sharing, call_seq, active_mask,
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
        group_lens (np.ndarray): Column count per group.

            Even if columns are not grouped, `group_lens` should contain ones - one column per group.
        init_cash (np.ndarray): Initial capital per column, or per group if cash sharing is enabled.

            If `cash_sharing` is True, should have shape `(target_shape[0], group_lens.shape[0])`.
            Otherwise, should have shape `target_shape`.
        cash_sharing (bool): Whether to share cash within the same group.
        call_seq (np.ndarray): Default sequence of calls per row and group.

            Should have shape `target_shape` and each value indicate the index of a column in a group.

            !!! note
                To use `auto_call_seq_ctx_nb`, should be of `CallSeqType.Default`.
        active_mask (np.ndarray): Mask of whether a particular segment should be executed.

            A segment is simply a sequence of `order_func_nb` calls under the same group and row.

            Should have shape `(target_shape[0], group_lens.shape[0])`.
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
        ...     create_order_nb,
        ...     simulate_nb,
        ...     build_call_seq,
        ...     auto_call_seq_ctx_nb,
        ...     share_flow_nb,
        ...     shares_nb,
        ...     holding_value_ungrouped_nb
        ... )
        >>> from vectorbt.portfolio.enums import SizeType, Direction

        >>> @njit
        ... def prep_func_nb(simc):  # do nothing
        ...     print('preparing simulation')
        ...     return ()

        >>> @njit
        ... def group_prep_func_nb(gc):
        ...     '''Define empty arrays for each group.'''
        ...     print('\\tpreparing group', gc.group)
        ...     # Try to create new arrays as rarely as possible
        ...     size = np.empty(gc.group_len, dtype=np.float_)
        ...     size_type = np.empty(gc.group_len, dtype=np.int_)
        ...     direction = np.empty(gc.group_len, dtype=np.int_)
        ...     temp_float_arr = np.empty(gc.group_len, dtype=np.float_)
        ...     return size, size_type, direction, temp_float_arr

        >>> @njit
        ... def segment_prep_func_nb(sc, size, size_type, direction, temp_float_arr):
        ...     '''Perform rebalancing at each segment.'''
        ...     print('\\t\\tpreparing segment', sc.i, '(row)')
        ...     for k in range(sc.group_len):
        ...         col = sc.from_col + k
        ...         size[k] = 1 / sc.group_len
        ...         size_type[k] = SizeType.TargetPercent
        ...         direction[k] = Direction.Long  # long positions only
        ...         # Here we use order price instead of previous close to valuate the assets
        ...         sc.last_val_price[col] = sc.close[sc.i, col]
        ...     # Reorder call sequence such that selling orders come first and buying last
        ...     auto_call_seq_ctx_nb(sc, size, size_type, direction, temp_float_arr)
        ...     return size, size_type, direction

        >>> @njit
        ... def order_func_nb(oc, size, size_type, direction, fees, fixed_fees, slippage):
        ...     '''Place an order.'''
        ...     print('\\t\\t\\trunning order', oc.call_idx, 'at column', oc.col)
        ...     col_i = oc.call_seq_now[oc.call_idx]  # or col - from_col
        ...     return create_order_nb(
        ...         size=size[col_i],
        ...         size_type=size_type[col_i],
        ...         price=oc.close[oc.i, oc.col],
        ...         fees=fees, fixed_fees=fixed_fees, slippage=slippage,
        ...         direction=direction[col_i]
        ...     )

        >>> target_shape = (5, 3)
        >>> np.random.seed(42)
        >>> close = np.random.uniform(1, 10, size=target_shape)
        >>> group_lens = np.array([3])  # one group of three columns
        >>> init_cash = np.array([100.])  # one capital per group
        >>> cash_sharing = True
        >>> call_seq = build_call_seq(target_shape, group_lens)  # will be overridden
        >>> active_mask = np.array([True, False, True, False, True])[:, None]
        >>> active_mask = np.copy(np.broadcast_to(active_mask, target_shape))
        >>> fees = 0.001
        >>> fixed_fees = 1.
        >>> slippage = 0.001

        >>> order_records = simulate_nb(
        ...     target_shape,
        ...     close,
        ...     group_lens,
        ...     init_cash,
        ...     cash_sharing,
        ...     call_seq,
        ...     active_mask,
        ...     prep_func_nb, (),
        ...     group_prep_func_nb, (),
        ...     segment_prep_func_nb, (),
        ...     order_func_nb, (fees, fixed_fees, slippage))
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
        >>> holding_value = holding_value_ungrouped_nb(close, shares)
        >>> pd.DataFrame(holding_value).vbt.scatter()
        ```

        ![](/vectorbt/docs/img/simulate_nb.png)

        Note that the last order in a group with cash sharing is always disadvantaged
        as it has a bit less funds than the previous orders due to costs, which are not
        included when valuating the group.
    """
    check_group_lens(group_lens, target_shape[1])
    check_group_init_cash(group_lens, target_shape[1], init_cash, cash_sharing)

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
        group_lens,
        init_cash,
        cash_sharing,
        call_seq,
        active_mask,
        order_records,
        record_mask,
        last_cash,
        last_shares,
        last_val_price
    )
    prep_out = prep_func_nb(simc, *prep_args)

    from_col = 0
    for group in range(len(group_lens)):
        # Is this group active?
        if np.any(active_mask[:, group]):
            to_col = from_col + group_lens[group]
            group_len = to_col - from_col

            # Run a function to preprocess this entire group
            gc = GroupContext(
                target_shape,
                close,
                group_lens,
                init_cash,
                cash_sharing,
                call_seq,
                active_mask,
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
                        group_lens,
                        init_cash,
                        cash_sharing,
                        call_seq,
                        active_mask,
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
                            if shares_now != 0:
                                value_now += shares_now * val_price_now

                        # Generate the next order
                        oc = OrderContext(
                            target_shape,
                            close,
                            group_lens,
                            init_cash,
                            cash_sharing,
                            call_seq,
                            active_mask,
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

                        # Process the order
                        cash_now, shares_now, order_result = process_order_nb(
                            cash_now, shares_now, val_price_now, value_now, order)

                        if order_result.status == OrderStatus.Filled:
                            # Add a new record
                            r = get_record_idx_nb(target_shape, i, col)
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
def simulate_row_wise_nb(target_shape, close, group_lens, init_cash, cash_sharing, call_seq,
                         active_mask, prep_func_nb, prep_args, row_prep_func_nb, row_prep_args,
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
    check_group_lens(group_lens, target_shape[1])
    check_group_init_cash(group_lens, target_shape[1], init_cash, cash_sharing)

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
        group_lens,
        init_cash,
        cash_sharing,
        call_seq,
        active_mask,
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
                group_lens,
                init_cash,
                cash_sharing,
                call_seq,
                active_mask,
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
            for group in range(len(group_lens)):
                # Is this group segment active?
                if active_mask[i, group]:
                    to_col = from_col + group_lens[group]
                    group_len = to_col - from_col

                    # Run a function to preprocess this row within this group
                    call_seq_now = call_seq[i, from_col:to_col]
                    sc = SegmentContext(
                        target_shape,
                        close,
                        group_lens,
                        init_cash,
                        cash_sharing,
                        call_seq,
                        active_mask,
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
                            if shares_now != 0:
                                value_now += shares_now * val_price_now

                        # Generate the next order
                        oc = OrderContext(
                            target_shape,
                            close,
                            group_lens,
                            init_cash,
                            cash_sharing,
                            call_seq,
                            active_mask,
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

                        # Process the order
                        cash_now, shares_now, order_result = process_order_nb(
                            cash_now, shares_now, val_price_now, value_now, order)

                        if order_result.status == OrderStatus.Filled:
                            # Add a new record
                            r = get_record_idx_nb(target_shape, i, col)
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
def simulate_from_orders_nb(target_shape, group_lens, init_cash, call_seq, auto_call_seq,
                            size, size_type, price, fees, fixed_fees, slippage, min_size, max_size,
                            reject_prob, allow_partial, raise_by_reject, direction, val_price, flex_2d):
    """Adaptation of `simulate_nb` for simulation based on orders.

    Utilizes flexible broadcasting.

    !!! note
        Should be only grouped if cash sharing is enabled.

        If `auto_call_seq` is True, make sure that `call_seq` follows `CallSeqType.Default`."""
    check_group_lens(group_lens, target_shape[1])
    cash_sharing = is_grouped_nb(group_lens)
    check_group_init_cash(group_lens, target_shape[1], init_cash, cash_sharing)

    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    record_mask = np.full(target_shape[0] * target_shape[1], False)
    j = 0
    last_cash = init_cash.astype(np.float_)
    last_shares = np.full(target_shape[1], 0., dtype=np.float_)
    temp_order_value = np.empty(target_shape[1], dtype=np.float_)

    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
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
                    if last_shares[col] != 0:
                        _val_price = flex_select_auto_nb(i, col, val_price, flex_2d)
                        value_now += last_shares[col] * _val_price

                # Dynamically sort by order value -> selling comes first to release funds early
                if auto_call_seq:
                    # Same as sort_by_order_value_ctx_nb but with flexible indexing
                    for k in range(group_len):
                        col = from_col + k
                        temp_order_value[k] = get_order_value_nb(
                            flex_select_auto_nb(i, col, size, flex_2d),
                            flex_select_auto_nb(i, col, size_type, flex_2d),
                            last_shares[col],
                            flex_select_auto_nb(i, col, val_price, flex_2d),
                            value_now,
                            flex_select_auto_nb(i, col, direction, flex_2d)
                        )

                    # Sort by order value
                    insert_argsort_nb(temp_order_value[:group_len], call_seq[i, from_col:to_col])

            for k in range(group_len):
                col = from_col + k
                if cash_sharing:
                    col_i = call_seq[i, col]
                    if col_i >= group_len:
                        raise ValueError("Call index exceeds bounds of the group")
                    col = from_col + col_i

                # Get running values per column
                shares_now = last_shares[col]
                val_price_now = flex_select_auto_nb(i, col, val_price, flex_2d)
                if not cash_sharing:
                    cash_now = last_cash[col]
                    value_now = cash_now
                    if shares_now != 0:
                        value_now += shares_now * val_price_now

                # Generate the next order
                order = Order(
                    flex_select_auto_nb(i, col, size, flex_2d),
                    flex_select_auto_nb(i, col, size_type, flex_2d),
                    flex_select_auto_nb(i, col, price, flex_2d),
                    flex_select_auto_nb(i, col, fees, flex_2d),
                    flex_select_auto_nb(i, col, fixed_fees, flex_2d),
                    flex_select_auto_nb(i, col, slippage, flex_2d),
                    flex_select_auto_nb(i, col, min_size, flex_2d),
                    flex_select_auto_nb(i, col, max_size, flex_2d),
                    flex_select_auto_nb(i, col, reject_prob, flex_2d),
                    flex_select_auto_nb(i, col, allow_partial, flex_2d),
                    flex_select_auto_nb(i, col, raise_by_reject, flex_2d),
                    flex_select_auto_nb(i, col, direction, flex_2d)
                )

                # Process the order
                cash_now, shares_now, order_result = process_order_nb(
                    cash_now, shares_now, val_price_now, value_now, order)

                if order_result.status == OrderStatus.Filled:
                    # Add a new record
                    if cash_sharing:
                        r = get_record_idx_nb(target_shape, i, col)
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
def signals_get_size_nb(shares_now,
                        is_entry, is_exit,
                        long_size, short_size,
                        long_close_first, short_close_first,
                        long_accumulate, short_accumulate,
                        conflict_mode, direction):
    """Get order size given signals."""
    order_size = 0.
    abs_shares_now = abs(shares_now)
    abs_long_size = abs(long_size)
    abs_short_size = abs(short_size)

    if is_entry and is_exit:
        # Conflict
        if conflict_mode == ConflictMode.Entry:
            # Ignore exit signal
            is_exit = False
        elif conflict_mode == ConflictMode.Exit:
            # Ignore entry signal
            is_entry = False
        elif conflict_mode == ConflictMode.Opposite:
            # Take opposite signal from the position we are in
            if direction == Direction.All:
                if shares_now > 0:
                    is_entry = False
                elif shares_now < 0:
                    is_exit = False
            else:
                if shares_now != 0:
                    is_entry = False

    if is_entry and not is_exit:
        if direction == Direction.All:
            # Behaves like Direction.Long
            if long_accumulate:
                order_size = abs_long_size
            else:
                if shares_now < 0:
                    # Reverse short position
                    if short_close_first:
                        order_size = abs_shares_now
                    else:
                        order_size = abs_shares_now + abs_long_size
                elif shares_now == 0:
                    # Open long position
                    order_size = abs_long_size
        elif direction == Direction.Long:
            if shares_now == 0 or long_accumulate:
                # Open or increase long position
                order_size = abs_long_size
        else:
            if shares_now == 0 or short_accumulate:
                # Open or increase short position
                order_size = -abs_short_size

    elif not is_entry and is_exit:
        if direction == Direction.All:
            # Behaves like Direction.Short
            if short_accumulate:
                order_size = -abs_short_size
            else:
                if shares_now > 0:
                    # Reverse long position
                    if long_close_first:
                        order_size = -abs_shares_now
                    else:
                        order_size = -abs_shares_now - abs_short_size
                elif shares_now == 0:
                    # Open short position
                    order_size = -abs_short_size
        elif direction == Direction.Short:
            if shares_now < 0:
                if long_accumulate:
                    # Reduce short position
                    order_size = abs_long_size
                else:
                    # Close short position
                    order_size = abs_shares_now
        else:
            if shares_now > 0:
                if short_accumulate:
                    # Reduce long position
                    order_size = -abs_short_size
                else:
                    # Close long position
                    order_size = -abs_shares_now
    return order_size


@njit(cache=True)
def simulate_from_signals_nb(target_shape, group_lens, init_cash,
                             call_seq, auto_call_seq,
                             entries, exits,
                             long_size, short_size,
                             long_price, short_price,
                             long_fees, short_fees,
                             long_fixed_fees, short_fixed_fees,
                             long_slippage, short_slippage,
                             long_min_size, short_min_size,
                             long_max_size, short_max_size,
                             long_reject_prob, short_reject_prob,
                             long_allow_partial, short_allow_partial,
                             long_raise_by_reject, short_raise_by_reject,
                             long_close_first, short_close_first,
                             long_accumulate, short_accumulate,
                             conflict_mode, direction,
                             val_price, flex_2d):
    """Adaptation of `simulate_nb` for simulation based on entry and exit signals.

    Utilizes flexible broadcasting.

    !!! note
        Should be only grouped if cash sharing is enabled."""
    check_group_lens(group_lens, target_shape[1])
    cash_sharing = is_grouped_nb(group_lens)
    check_group_init_cash(group_lens, target_shape[1], init_cash, cash_sharing)

    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    record_mask = np.full(target_shape[0] * target_shape[1], False)
    j = 0
    last_cash = init_cash.astype(np.float_)
    last_shares = np.full(target_shape[1], 0., dtype=np.float_)
    order_size = np.empty(target_shape[1], dtype=np.float_)
    temp_order_value = np.empty(target_shape[1], dtype=np.float_)

    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_len = to_col - from_col

        # Get running values per group
        if cash_sharing:
            cash_now = last_cash[group]

        for i in range(target_shape[0]):

            # Get size and value of each order
            for k in range(group_len):
                col = from_col + k  # order doesn't matter
                _order_size = signals_get_size_nb(
                    last_shares[col],
                    flex_select_auto_nb(i, col, entries, flex_2d),
                    flex_select_auto_nb(i, col, exits, flex_2d),
                    flex_select_auto_nb(i, col, long_size, flex_2d),
                    flex_select_auto_nb(i, col, short_size, flex_2d),
                    flex_select_auto_nb(i, col, long_close_first, flex_2d),
                    flex_select_auto_nb(i, col, short_close_first, flex_2d),
                    flex_select_auto_nb(i, col, long_accumulate, flex_2d),
                    flex_select_auto_nb(i, col, short_accumulate, flex_2d),
                    flex_select_auto_nb(i, col, conflict_mode, flex_2d),
                    flex_select_auto_nb(i, col, direction, flex_2d)
                )  # already takes into account direction
                order_size[col] = _order_size

                if cash_sharing:
                    if _order_size > 0:
                        order_price = flex_select_auto_nb(i, col, long_price, flex_2d)
                    elif _order_size < 0:
                        order_price = flex_select_auto_nb(i, col, short_price, flex_2d)
                    else:
                        order_price = 0.
                    temp_order_value[k] = _order_size * order_price

            if cash_sharing:
                # Dynamically sort by order value -> selling comes first to release funds early
                if auto_call_seq:
                    insert_argsort_nb(temp_order_value[:group_len], call_seq[i, from_col:to_col])

                # Same as get_group_value_ctx_nb but with flexible indexing
                value_now = cash_now
                for k in range(group_len):
                    col = from_col + k
                    if last_shares[col] != 0:
                        _val_price = flex_select_auto_nb(i, col, val_price, flex_2d)
                        value_now += last_shares[col] * _val_price

            for k in range(group_len):
                col = from_col + k
                if cash_sharing:
                    col_i = call_seq[i, col]
                    if col_i >= group_len:
                        raise ValueError("Call index exceeds bounds of the group")
                    col = from_col + col_i

                # Get running values per column
                shares_now = last_shares[col]
                val_price_now = flex_select_auto_nb(i, col, val_price, flex_2d)
                if not cash_sharing:
                    cash_now = last_cash[col]
                    value_now = cash_now
                    if shares_now != 0:
                        value_now += shares_now * val_price_now

                # Generate the next order
                _order_size = order_size[col]  # already takes into account direction
                if _order_size != 0:
                    if _order_size > 0:  # long order
                        _direction = flex_select_auto_nb(i, col, direction, flex_2d)
                        if _direction == Direction.Short:
                            _order_size *= -1  # must reverse for process_order_nb
                        order = create_order_nb(
                            size=_order_size,
                            size_type=SizeType.Shares,
                            price=flex_select_auto_nb(i, col, long_price, flex_2d),
                            fees=flex_select_auto_nb(i, col, long_fees, flex_2d),
                            fixed_fees=flex_select_auto_nb(i, col, long_fixed_fees, flex_2d),
                            slippage=flex_select_auto_nb(i, col, long_slippage, flex_2d),
                            min_size=flex_select_auto_nb(i, col, long_min_size, flex_2d),
                            max_size=flex_select_auto_nb(i, col, long_max_size, flex_2d),
                            reject_prob=flex_select_auto_nb(i, col, long_reject_prob, flex_2d),
                            allow_partial=flex_select_auto_nb(i, col, long_allow_partial, flex_2d),
                            raise_by_reject=flex_select_auto_nb(i, col, long_raise_by_reject, flex_2d),
                            direction=_direction
                        )
                    else:  # short order
                        _direction = flex_select_auto_nb(i, col, direction, flex_2d)
                        if _direction == Direction.Short:
                            _order_size *= -1
                        order = create_order_nb(
                            size=_order_size,
                            size_type=SizeType.Shares,
                            price=flex_select_auto_nb(i, col, short_price, flex_2d),
                            fees=flex_select_auto_nb(i, col, short_fees, flex_2d),
                            fixed_fees=flex_select_auto_nb(i, col, short_fixed_fees, flex_2d),
                            slippage=flex_select_auto_nb(i, col, short_slippage, flex_2d),
                            min_size=flex_select_auto_nb(i, col, short_min_size, flex_2d),
                            max_size=flex_select_auto_nb(i, col, short_max_size, flex_2d),
                            reject_prob=flex_select_auto_nb(i, col, short_reject_prob, flex_2d),
                            allow_partial=flex_select_auto_nb(i, col, short_allow_partial, flex_2d),
                            raise_by_reject=flex_select_auto_nb(i, col, short_raise_by_reject, flex_2d),
                            direction=_direction
                        )

                    # Process the order
                    cash_now, shares_now, order_result = process_order_nb(
                        cash_now, shares_now, val_price_now, value_now, order)

                    if order_result.status == OrderStatus.Filled:
                        # Add a new record
                        if cash_sharing:
                            r = get_record_idx_nb(target_shape, i, col)
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
            shares_now = add_nb(shares_now, flow_value)
            out[i, col] = shares_now
    return out


@njit(cache=True)
def i_group_sum_reduce_nb(i, group, a):
    """Sum reducer for grouped columns."""
    return np.sum(a)


@njit
def pos_mask_nb(shares, group_lens):
    """Get number of columns in position for each row and group."""
    return generic_nb.reduce_grouped_row_wise_nb(shares != 0, group_lens, i_group_sum_reduce_nb) > 0


@njit
def long_pos_mask_nb(shares, group_lens):
    """Get number of columns in long position for each row and group."""
    return generic_nb.reduce_grouped_row_wise_nb(shares > 0, group_lens, i_group_sum_reduce_nb) > 0


@njit
def short_pos_mask_nb(shares, group_lens):
    """Get number of columns in short position for each row and group."""
    return generic_nb.reduce_grouped_row_wise_nb(shares < 0, group_lens, i_group_sum_reduce_nb) > 0


@njit(cache=True)
def group_mean_reduce_nb(group, a):
    """Mean reducer for grouped columns."""
    return np.mean(a)


@njit
def pos_duration_nb(shares, group_lens):
    """Get duration of position for each row and group."""
    return generic_nb.reduce_grouped_nb(shares != 0, group_lens, group_mean_reduce_nb)


@njit
def long_pos_duration_nb(shares, group_lens):
    """Get duration of long position for each row and group."""
    return generic_nb.reduce_grouped_nb(shares > 0, group_lens, group_mean_reduce_nb)


@njit
def short_pos_duration_nb(shares, group_lens):
    """Get duration of short position for each row and group."""
    return generic_nb.reduce_grouped_nb(shares < 0, group_lens, group_mean_reduce_nb)


# ############# Cash ############# #

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
def cash_flow_grouped_nb(cash_flow_ungrouped, group_lens):
    """Get cash flow series per group."""
    check_group_lens(group_lens, cash_flow_ungrouped.shape[1])

    out = np.empty((cash_flow_ungrouped.shape[0], len(group_lens)), dtype=np.float_)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        out[:, group] = np.sum(cash_flow_ungrouped[:, from_col:to_col], axis=1)
        from_col = to_col
    return out


@njit(cache=True)
def init_cash_grouped_nb(init_cash, group_lens, cash_sharing):
    """Get initial cash per group."""
    if cash_sharing:
        return init_cash
    out = np.empty(group_lens.shape, dtype=np.float_)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        cash_sum = 0.
        for col in range(from_col, to_col):
            cash_sum += init_cash[col]
        out[group] = cash_sum
        from_col = to_col
    return out


@njit(cache=True)
def init_cash_ungrouped_nb(init_cash, group_lens, cash_sharing):
    """Get initial cash per column."""
    if not cash_sharing:
        return init_cash
    group_lens_cs = np.cumsum(group_lens)
    out = np.full(group_lens_cs[-1], np.nan, dtype=np.float_)
    out[group_lens_cs - group_lens] = init_cash
    out = generic_nb.ffill_1d_nb(out)
    return out


@njit(cache=True)
def cash_ungrouped_nb(cash_flow_ungrouped, group_lens, init_cash, call_seq, in_sim_order):
    """Get cash series per column.

    `init_cash` should be grouped if `in_sim_order` is True, and ungrouped otherwise."""
    check_group_lens(group_lens, cash_flow_ungrouped.shape[1])

    out = np.empty_like(cash_flow_ungrouped)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_len = to_col - from_col
        if in_sim_order:
            cash_now = init_cash[group]
        for i in range(cash_flow_ungrouped.shape[0]):
            for k in range(group_len):
                col = from_col + call_seq[i, from_col + k]
                if not in_sim_order:
                    cash_now = init_cash[col] if i == 0 else out[i - 1, col]
                flow_value = cash_flow_ungrouped[i, col]
                cash_now = add_nb(cash_now, flow_value)
                out[i, col] = cash_now
        from_col = to_col
    return out


@njit(cache=True)
def cash_grouped_nb(target_shape, cash_flow_grouped, group_lens, init_cash_grouped):
    """Get cash series per group."""
    check_group_lens(group_lens, target_shape[1])

    out = np.empty_like(cash_flow_grouped)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        cash_now = init_cash_grouped[group]
        for i in range(cash_flow_grouped.shape[0]):
            flow_value = cash_flow_grouped[i, group]
            cash_now = add_nb(cash_now, flow_value)
            out[i, group] = cash_now
        from_col = to_col
    return out


# ############# Performance ############# #


@njit(cache=True)
def holding_value_grouped_nb(close, shares, group_lens):
    """Get holding value series per group."""
    check_group_lens(group_lens, close.shape[1])

    out = np.empty((close.shape[0], len(group_lens)), dtype=np.float_)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        out[:, group] = np.sum(shares[:, from_col:to_col] * close[:, from_col:to_col], axis=1)
        from_col = to_col
    return out


@njit(cache=True)
def holding_value_ungrouped_nb(close, shares):
    """Get holding value series per column."""
    return close * shares


@njit(cache=True)
def value_in_sim_order_nb(cash_ungrouped, holding_value_ungrouped, group_lens, call_seq):
    """Get portfolio value series in simulation order."""
    check_group_lens(group_lens, cash_ungrouped.shape[1])

    out = np.empty_like(cash_ungrouped)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
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


@njit(cache=True)
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

        # Fill shares
        if record['side'] == OrderSide.Buy:
            order_size = record['size']
            shares[col] = add_nb(shares[col], order_size)
        else:
            order_size = record['size']
            shares[col] = add_nb(shares[col], -order_size)

        # Fill cash
        if record['side'] == OrderSide.Buy:
            order_cash = record['size'] * record['price'] + record['fees']
            cash[col] = add_nb(cash[col], -order_cash)
        else:
            order_cash = record['size'] * record['price'] - record['fees']
            cash[col] = add_nb(cash[col], order_cash)

    return cash + shares * close[-1, :] - init_cash_ungrouped


@njit(cache=True)
def total_profit_grouped_nb(total_profit_ungrouped, group_lens):
    """Get total profit per group."""
    check_group_lens(group_lens, total_profit_ungrouped.shape[0])

    out = np.empty((len(group_lens),), dtype=np.float_)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
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
def get_return_nb(input_value, output_value):
    """Get return from input and output value."""
    if input_value == 0:
        if output_value == 0:
            return 0.
        return np.inf * np.sign(output_value)
    return_value = (output_value - input_value) / input_value
    if input_value < 0:
        return_value *= -1
    return return_value


@njit(cache=True)
def returns_nb(value, init_cash_regrouped):
    """Get portfolio return series per column/group."""
    out = np.empty(value.shape, dtype=np.float_)
    for col in range(out.shape[1]):
        input_value = init_cash_regrouped[col]
        for i in range(out.shape[0]):
            output_value = value[i, col]
            out[i, col] = get_return_nb(input_value, output_value)
            input_value = output_value
    return out


@njit(cache=True)
def returns_in_sim_order_nb(value_iso, group_lens, init_cash_grouped, call_seq):
    """Get portfolio return series in simulation order."""
    check_group_lens(group_lens, value_iso.shape[1])

    out = np.empty_like(value_iso)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_len = to_col - from_col
        input_value = init_cash_grouped[group]
        for j in range(value_iso.shape[0] * group_len):
            i = j // group_len
            col = from_col + call_seq[i, from_col + j % group_len]
            output_value = value_iso[i, col]
            out[i, col] = get_return_nb(input_value, output_value)
            input_value = output_value
        from_col = to_col
    return out


@njit(cache=True)
def active_returns_nb(cash_flow, holding_value):
    """Get active return series per column/group."""
    out = np.empty_like(cash_flow)
    for col in range(cash_flow.shape[1]):
        for i in range(cash_flow.shape[0]):
            input_value = 0. if i == 0 else holding_value[i - 1, col]
            output_value = holding_value[i, col] + cash_flow[i, col]
            out[i, col] = get_return_nb(input_value, output_value)
    return out


@njit(cache=True)
def buy_and_hold_return_ungrouped_nb(close):
    """Get total return value of buying and holding per column."""
    return (close[-1, :] - close[0, :]) / close[0, :]


@njit(cache=True)
def buy_and_hold_return_grouped_nb(close, group_lens):
    """Get total return value of buying and holding per group."""
    check_group_lens(group_lens, close.shape[1])

    out = np.empty(len(group_lens), dtype=np.float_)
    total_return = (close[-1, :] - close[0, :]) / close[0, :]
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_len = to_col - from_col
        out[group] = np.sum(total_return[from_col:to_col]) / group_len
        from_col = to_col
    return out
