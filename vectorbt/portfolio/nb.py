"""Numba-compiled 1-dim and 2-dim functions.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    
    All functions passed as argument should be Numba-compiled.
    
    Records should retain the order they were created in.

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
    is_close_nb,
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
    Order,
    NoOrder,
    OrderStatus,
    OrderSide,
    StatusInfo,
    OrderResult,
    RejectedOrderError,
    Direction,
    order_dt,
    TradeDirection,
    TradeStatus,
    trade_dt,
    position_dt,
    log_dt
)


# ############# Simulation ############# #

@njit(cache=True)
def fill_req_log_nb(cash_now, shares_now, val_price_now, value_now, order, log_record):
    """Fill log record on order request."""
    log_record['cash_now'] = cash_now
    log_record['shares_now'] = shares_now
    log_record['val_price_now'] = val_price_now
    log_record['value_now'] = value_now
    log_record['size'] = order.size
    log_record['size_type'] = order.size_type
    log_record['direction'] = order.direction
    log_record['price'] = order.price
    log_record['fees'] = order.fees
    log_record['fixed_fees'] = order.fixed_fees
    log_record['slippage'] = order.slippage
    log_record['min_size'] = order.min_size
    log_record['max_size'] = order.max_size
    log_record['reject_prob'] = order.reject_prob
    log_record['close_first'] = order.close_first
    log_record['allow_partial'] = order.allow_partial
    log_record['raise_reject'] = order.raise_reject
    log_record['log'] = order.log


@njit(cache=True)
def fill_res_log_nb(new_cash, new_shares, order_result, log_record):
    """Fill log record on order result."""
    log_record['new_cash'] = new_cash
    log_record['new_shares'] = new_shares
    log_record['res_size'] = order_result.size
    log_record['res_price'] = order_result.price
    log_record['res_fees'] = order_result.fees
    log_record['res_side'] = order_result.side
    log_record['res_status'] = order_result.status
    log_record['res_status_info'] = order_result.status_info


@njit(cache=True)
def order_not_filled_nb(cash_now, shares_now, status, status_info, log_record, log):
    """Return `cash_now`, `shares_now` and `OrderResult` for order that hasn't been filled."""
    order_result = OrderResult(np.nan, np.nan, np.nan, -1, status, status_info)
    if log:
        fill_res_log_nb(cash_now, shares_now, order_result, log_record)
    return cash_now, shares_now, order_result


@njit(cache=True)
def buy_shares_nb(cash_now, shares_now, size, direction, price, fees, fixed_fees, slippage,
                  min_size, allow_partial, raise_reject, log_record, log):
    """Buy shares."""

    # Get optimal order size
    if direction == Direction.ShortOnly:
        adj_size = min(-shares_now, size)
    else:
        adj_size = size

    # Get price adjusted with slippage
    adj_price = price * (1 + slippage)

    # Get cash required to complete this order
    req_cash = adj_size * adj_price
    req_fees = req_cash * fees + fixed_fees
    adj_req_cash = req_cash + req_fees

    if is_close_or_less_nb(adj_req_cash, cash_now):
        # Sufficient cash
        final_size = adj_size
        fees_paid = req_fees
        final_cash = adj_req_cash

        # Update current cash
        new_cash = add_nb(cash_now, -final_cash)
    else:
        # Insufficient cash, size will be less than requested
        if is_close_or_less_nb(cash_now, fixed_fees):
            # Can't fill
            if raise_reject:
                raise RejectedOrderError("Order rejected: Not enough cash to cover fees")
            return order_not_filled_nb(
                cash_now, shares_now,
                OrderStatus.Rejected, StatusInfo.CantCoverFees,
                log_record, log)

        # For fees of 10% and 1$ per transaction, you can buy shares for 90$ (effect_cash)
        # to spend 100$ (adj_req_cash) in total
        final_cash = (cash_now - fixed_fees) / (1 + fees)

        # Update size and fees
        final_size = final_cash / adj_price
        fees_paid = cash_now - final_cash

        # Update current cash
        new_cash = 0.  # numerical stability

    # Check against minimum size
    if abs(final_size) < min_size:
        if raise_reject:
            raise RejectedOrderError("Order rejected: Final size is less than minimum allowed")
        return order_not_filled_nb(
            cash_now, shares_now,
            OrderStatus.Rejected, StatusInfo.MinSizeNotReached,
            log_record, log)

    # Check against partial fill (np.inf doesn't count)
    if np.isfinite(size) and is_less_nb(final_size, size) and not allow_partial:
        if raise_reject:
            raise RejectedOrderError("Order rejected: Final size is less than requested")
        return order_not_filled_nb(
            cash_now, shares_now,
            OrderStatus.Rejected, StatusInfo.PartialFill,
            log_record, log)

    # Update current shares
    new_shares = add_nb(shares_now, final_size)

    # Return filled order
    order_result = OrderResult(
        final_size,
        adj_price,
        fees_paid,
        OrderSide.Buy,
        OrderStatus.Filled,
        -1
    )
    if log:
        fill_res_log_nb(new_cash, new_shares, order_result, log_record)
    return new_cash, new_shares, order_result


@njit(cache=True)
def sell_shares_nb(cash_now, shares_now, size, direction, price, fees, fixed_fees, slippage,
                   min_size, allow_partial, raise_reject, log_record, log):
    """Sell shares."""

    # Get optimal order size
    if direction == Direction.LongOnly:
        final_size = min(shares_now, size)
    else:
        final_size = size

    # Check against minimum size
    if abs(final_size) < min_size:
        if raise_reject:
            raise RejectedOrderError("Order rejected: Final size is less than minimum allowed")
        return order_not_filled_nb(
            cash_now, shares_now,
            OrderStatus.Rejected, StatusInfo.MinSizeNotReached,
            log_record, log)

    # Check against partial fill
    if np.isfinite(size) and is_less_nb(final_size, size) and not allow_partial:
        # np.inf doesn't count
        if raise_reject:
            raise RejectedOrderError("Order rejected: Final size is less than requested")
        return order_not_filled_nb(
            cash_now, shares_now,
            OrderStatus.Rejected, StatusInfo.PartialFill,
            log_record, log)

    # Get price adjusted with slippage
    adj_price = price * (1 - slippage)

    # Compute acquired cash
    acq_cash = final_size * adj_price

    # Update fees
    fees_paid = acq_cash * fees + fixed_fees

    # Get final cash by subtracting costs
    if is_less_nb(acq_cash, fees_paid):
        # Can't fill
        if raise_reject:
            raise RejectedOrderError("Order rejected: Fees cannot be covered")
        return order_not_filled_nb(
            cash_now, shares_now,
            OrderStatus.Rejected, StatusInfo.CantCoverFees,
            log_record, log)
    final_cash = acq_cash - fees_paid

    # Update current cash and shares
    new_cash = cash_now + final_cash
    new_shares = add_nb(shares_now, -final_size)

    # Return filled order
    order_result = OrderResult(
        final_size,
        adj_price,
        fees_paid,
        OrderSide.Sell,
        OrderStatus.Filled,
        -1
    )
    if log:
        fill_res_log_nb(new_cash, new_shares, order_result, log_record)
    return new_cash, new_shares, order_result


@njit(cache=True)
def process_order_nb(cash_now, shares_now, val_price_now, value_now, order, log_record):
    """Process an order given current cash and share balance.

    Args:
        cash_now (float): Cash available to this asset or group with cash sharing.
        shares_now (float): Holdings of this particular asset.
        val_price_now (float): Valuation price for this particular asset.

            Used to convert `SizeType.TargetValue` to `SizeType.TargetShares`.
        value_now (float): Value of this asset or group with cash sharing.

            Used to convert `SizeType.TargetPercent` to `SizeType.TargetValue`.
        order (Order): See `vectorbt.portfolio.enums.Order`.
        log_record (log_dt): Record of type `vectorbt.portfolio.enums.log_dt`.

    Error is thrown if an input has value that is not expected.
    Order is ignored if its execution has no effect on current balance.
    Order is rejected if an input goes over a limit/restriction.
    """
    if order.log:
        fill_req_log_nb(cash_now, shares_now, val_price_now, value_now, order, log_record)

    if np.isnan(order.size):
        return order_not_filled_nb(
            cash_now, shares_now,
            OrderStatus.Ignored, StatusInfo.SizeNaN,
            log_record, order.log)
    if np.isnan(order.price):
        return order_not_filled_nb(
            cash_now, shares_now,
            OrderStatus.Ignored, StatusInfo.PriceNaN,
            log_record, order.log)

    # Check variables
    if np.isnan(cash_now) or cash_now < 0:
        raise ValueError("cash_now must be greater than 0")
    if not np.isfinite(shares_now):
        raise ValueError("shares_now must be finite")

    # Check order
    if order.direction == Direction.LongOnly and shares_now < 0:
        raise ValueError("shares_now is negative but order.direction is Direction.LongOnly")
    if order.direction == Direction.ShortOnly and shares_now > 0:
        raise ValueError("shares_now is positive but order.direction is Direction.ShortOnly")
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

    order_size = order.size
    order_size_type = order.size_type

    if order.direction == Direction.ShortOnly:
        # Positive size in short direction should be treated as negative
        order_size *= -1

    if order_size_type == SizeType.TargetPercent:
        # Target percentage of current value
        if np.isnan(value_now):
            return order_not_filled_nb(
                cash_now, shares_now,
                OrderStatus.Ignored, StatusInfo.ValueNaN,
                log_record, order.log)
        if value_now <= 0:
            return order_not_filled_nb(
                cash_now, shares_now,
                OrderStatus.Rejected, StatusInfo.ValueZeroNeg,
                log_record, order.log)

        order_size *= value_now
        order_size_type = SizeType.TargetValue

    if order_size_type == SizeType.TargetValue:
        # Target value
        if np.isinf(val_price_now) or val_price_now <= 0:
            raise ValueError("val_price_now must be finite and greater than 0")
        if np.isnan(val_price_now):
            return order_not_filled_nb(
                cash_now, shares_now,
                OrderStatus.Ignored, StatusInfo.ValPriceNaN,
                log_record, order.log)

        order_size = order_size / val_price_now
        order_size_type = SizeType.TargetShares

    if order_size_type == SizeType.TargetShares:
        # Target amount of shares
        order_size -= shares_now

    if order.direction == Direction.ShortOnly or order.direction == Direction.All:
        if order_size < 0 and np.isinf(order_size):
            # Similar to going all long, going all short also depends upon current funds
            # If in short position, also subtract cash that covers this position (1:1)
            # This way, two successive -np.inf operations with same price will trigger only one short
            order_size = -2 * shares_now - cash_now / order.price
            if order_size >= 0:
                if order.raise_reject:
                    raise RejectedOrderError("Order rejected: Not enough cash to short")
                return order_not_filled_nb(
                    cash_now, shares_now,
                    OrderStatus.Rejected, StatusInfo.NoCashShort,
                    log_record, order.log)

    direction = order.direction
    if order.close_first:
        # Close position before reversal, requires second order to open opposite position
        if not is_close_nb(shares_now, 0):
            if shares_now > 0:
                if order_size < 0:
                    # Restrict at bottom
                    direction = Direction.LongOnly
            else:
                if order_size > 0:
                    # Restrict at top
                    direction = Direction.ShortOnly

    if is_close_nb(order_size, 0):
        return order_not_filled_nb(
            cash_now, shares_now,
            OrderStatus.Ignored, StatusInfo.SizeZero,
            log_record, order.log)

    if abs(order_size) > order.max_size:
        if not order.allow_partial:
            if order.raise_reject:
                raise RejectedOrderError("Order rejected: Size is greater than maximum allowed")
            return order_not_filled_nb(
                cash_now, shares_now,
                OrderStatus.Rejected, StatusInfo.MaxSizeExceeded,
                log_record, order.log)

        order_size = np.sign(order_size) * order.max_size

    if order.reject_prob > 0:
        if np.random.uniform(0, 1) < order.reject_prob:
            if order.raise_reject:
                raise RejectedOrderError("Random event happened")
            return order_not_filled_nb(
                cash_now, shares_now,
                OrderStatus.Rejected, StatusInfo.RandomEvent,
                log_record, order.log)

    if order_size > 0:
        if direction == Direction.LongOnly or direction == Direction.All:
            if is_close_nb(cash_now, 0):
                if order.raise_reject:
                    raise RejectedOrderError("Order rejected: Not enough cash to long")
                return order_not_filled_nb(
                    0., shares_now,
                    OrderStatus.Rejected, StatusInfo.NoCashLong,
                    log_record, order.log)
            if np.isinf(order_size) and np.isinf(cash_now):
                raise ValueError("Attempt to go in long direction indefinitely. Set max_size or finite init_cash.")
        else:
            if is_close_nb(shares_now, 0):
                if order.raise_reject:
                    raise RejectedOrderError("Order rejected: No open position to reduce/close")
                return order_not_filled_nb(
                    cash_now, 0.,
                    OrderStatus.Rejected, StatusInfo.NoOpenPosition,
                    log_record, order.log)

        return buy_shares_nb(
            cash_now,
            shares_now,
            order_size,
            direction,
            order.price,
            order.fees,
            order.fixed_fees,
            order.slippage,
            order.min_size,
            order.allow_partial,
            order.raise_reject,
            log_record,
            order.log
        )
    else:
        if direction == Direction.ShortOnly or direction == Direction.All:
            if np.isinf(order_size):
                raise ValueError("Attempt to go in short direction indefinitely. Set max_size or finite init_cash.")
        else:
            if is_close_nb(shares_now, 0):
                if order.raise_reject:
                    raise RejectedOrderError("Order rejected: No open position to reduce/close")
                return order_not_filled_nb(
                    cash_now, 0.,
                    OrderStatus.Rejected, StatusInfo.NoOpenPosition,
                    log_record, order.log)

        return sell_shares_nb(
            cash_now,
            shares_now,
            -order_size,
            direction,
            order.price,
            order.fees,
            order.fixed_fees,
            order.slippage,
            order.min_size,
            order.allow_partial,
            order.raise_reject,
            log_record,
            order.log
        )


@njit(cache=True)
def create_order_nb(size=np.nan,
                    size_type=SizeType.Shares,
                    direction=Direction.All,
                    price=np.nan,
                    fees=0.,
                    fixed_fees=0.,
                    slippage=0.,
                    min_size=0.,
                    max_size=np.inf,
                    reject_prob=0.,
                    close_first=False,
                    allow_partial=True,
                    raise_reject=False,
                    log=False):
    """Convenience function to create an order with some defaults."""

    return Order(
        float(size),
        size_type,
        direction,
        float(price),
        float(fees),
        float(fixed_fees),
        float(slippage),
        float(min_size),
        float(max_size),
        float(reject_prob),
        close_first,
        allow_partial,
        raise_reject,
        log
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
    if direction == Direction.ShortOnly:
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

    Returns order records of layout `vectorbt.portfolio.enums.order_dt` and log records of layout
    `vectorbt.portfolio.enums.log_dt`.

    As opposed to `simulate_row_wise_nb`, order processing happens in row-major order, that is,
    from top to bottom slower (along time axis) and from left to right faster (along asset axis).
    See [Glossary](https://numpy.org/doc/stable/glossary.html).

    Args:
        target_shape (tuple): Target shape.

            A tuple with exactly two elements: the number of steps and columns.
        close (array_like of float): Reference price, such as close.

            Should have shape `target_shape`.
        group_lens (array_like of int): Column count per group.

            Even if columns are not grouped, `group_lens` should contain ones - one column per group.
        init_cash (array_like of float): Initial capital per column, or per group if cash sharing is enabled.

            If `cash_sharing` is True, should have shape `(target_shape[0], group_lens.shape[0])`.
            Otherwise, should have shape `target_shape`.
        cash_sharing (bool): Whether to share cash within the same group.
        call_seq (array_like of int): Default sequence of calls per row and group.

            Should have shape `target_shape` and each value indicate the index of a column in a group.

            !!! note
                To use `auto_call_seq_ctx_nb`, should be of `CallSeqType.Default`.
        active_mask (array_like of bool): Mask of whether a particular segment should be executed.

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

    ## Example

    Create a group of three assets together sharing 100$ and simulate an equal-weighted portfolio
    that rebalances every second tick, all without leaving Numba:

    ```python-repl
    >>> import numpy as np
    >>> import pandas as pd
    >>> from numba import njit
    >>> from vectorbt.generic.plotting import create_scatter
    >>> from vectorbt.records.nb import col_map_nb
    >>> from vectorbt.portfolio.enums import SizeType, Direction
    >>> from vectorbt.portfolio.nb import (
    ...     create_order_nb,
    ...     simulate_nb,
    ...     build_call_seq,
    ...     auto_call_seq_ctx_nb,
    ...     share_flow_nb,
    ...     shares_nb,
    ...     holding_value_nb
    ... )

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
    ...         direction[k] = Direction.LongOnly  # long positions only
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
    ...         direction=direction[col_i],
    ...         price=oc.close[oc.i, oc.col],
    ...         fees=fees, fixed_fees=fixed_fees, slippage=slippage
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

    >>> order_records, log_records = simulate_nb(
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

    >>> pd.DataFrame.from_records(order_records)
       id  idx  col       size     price      fees  side
    0   0    0    0   7.626262  4.375232  1.033367     0
    1   1    0    1   3.488053  9.565985  1.033367     0
    2   2    0    2   3.972040  7.595533  1.030170     0
    3   3    2    1   0.920352  8.786790  1.008087     1
    4   4    2    2   0.448747  6.403625  1.002874     1
    5   5    2    0   5.210115  1.524275  1.007942     0
    6   6    4    0   7.899568  8.483492  1.067016     1
    7   7    4    2  12.378281  2.639061  1.032667     0
    8   8    4    1  10.713236  2.913963  1.031218     0

    >>> call_seq
    array([[0, 1, 2],
           [0, 1, 2],
           [1, 2, 0],
           [0, 1, 2],
           [0, 2, 1]])

    >>> col_map = col_map_nb(order_records['col'], target_shape[1])
    >>> share_flow = share_flow_nb(target_shape, order_records, col_map, Direction.All)
    >>> shares = shares_nb(share_flow)
    >>> holding_value = holding_value_nb(close, shares)
    >>> create_scatter(data=holding_value)
    ```

    ![](/vectorbt/docs/img/simulate_nb.png)

    Note that the last order in a group with cash sharing is always disadvantaged
    as it has a bit less funds than the previous orders due to costs, which are not
    included when valuating the group.
    """
    check_group_lens(group_lens, target_shape[1])
    check_group_init_cash(group_lens, target_shape[1], init_cash, cash_sharing)

    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    ridx = 0
    log_records = np.empty(target_shape[0] * target_shape[1], dtype=log_dt)
    lidx = 0
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
        log_records,
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
                order_records[:ridx],
                log_records[:lidx],
                last_cash,
                last_shares,
                last_val_price,
                group,
                group_len,
                from_col,
                to_col
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
                        order_records[:ridx],
                        log_records[:lidx],
                        last_cash,
                        last_shares,
                        last_val_price,
                        i,
                        group,
                        group_len,
                        from_col,
                        to_col,
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
                            order_records[:ridx],
                            log_records[:lidx],
                            last_cash,
                            last_shares,
                            last_val_price,
                            i,
                            group,
                            group_len,
                            from_col,
                            to_col,
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
                            cash_now, shares_now, val_price_now, value_now, order, log_records[lidx])

                        if order.log:
                            # Add log metadata
                            log_records[lidx]['id'] = lidx
                            log_records[lidx]['idx'] = i
                            log_records[lidx]['col'] = col
                            log_records[lidx]['group'] = group
                            if order_result.status == OrderStatus.Filled:
                                log_records[lidx]['order_id'] = ridx
                            else:
                                log_records[lidx]['order_id'] = -1
                            lidx += 1

                        if order_result.status == OrderStatus.Filled:
                            # Add order metadata
                            order_records[ridx]['id'] = ridx
                            order_records[ridx]['idx'] = i
                            order_records[ridx]['col'] = col
                            order_records[ridx]['size'] = order_result.size
                            order_records[ridx]['price'] = order_result.price
                            order_records[ridx]['fees'] = order_result.fees
                            order_records[ridx]['side'] = order_result.side
                            ridx += 1

                        # Now becomes last
                        if cash_sharing:
                            last_cash[group] = cash_now
                        else:
                            last_cash[col] = cash_now
                        last_shares[col] = shares_now

            from_col = to_col

    return order_records[:ridx], log_records[:lidx]


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

    ## Example

    Running the same example as in `simulate_nb` but replacing `group_prep_func_nb` for
    `row_prep_func_nb` gives the same results but now the following call hierarchy:
    ```python-repl
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
    ridx = 0
    log_records = np.empty(target_shape[0] * target_shape[1], dtype=log_dt)
    lidx = 0
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
        log_records,
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
                order_records[:ridx],
                log_records[:lidx],
                last_cash,
                last_shares,
                last_val_price,
                i
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
                        order_records[:ridx],
                        log_records[:lidx],
                        last_cash,
                        last_shares,
                        last_val_price,
                        i,
                        group,
                        group_len,
                        from_col,
                        to_col,
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
                            order_records[:ridx],
                            log_records[:lidx],
                            last_cash,
                            last_shares,
                            last_val_price,
                            i,
                            group,
                            group_len,
                            from_col,
                            to_col,
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
                            cash_now, shares_now, val_price_now, value_now, order, log_records[lidx])

                        if order.log:
                            # Add log metadata
                            log_records[lidx]['id'] = lidx
                            log_records[lidx]['idx'] = i
                            log_records[lidx]['col'] = col
                            log_records[lidx]['group'] = group
                            if order_result.status == OrderStatus.Filled:
                                log_records[lidx]['order_id'] = ridx
                            else:
                                log_records[lidx]['order_id'] = -1
                            lidx += 1

                        if order_result.status == OrderStatus.Filled:
                            # Add order metadata
                            order_records[ridx]['id'] = ridx
                            order_records[ridx]['idx'] = i
                            order_records[ridx]['col'] = col
                            order_records[ridx]['size'] = order_result.size
                            order_records[ridx]['price'] = order_result.price
                            order_records[ridx]['fees'] = order_result.fees
                            order_records[ridx]['side'] = order_result.side
                            ridx += 1

                        # Now becomes last
                        if cash_sharing:
                            last_cash[group] = cash_now
                        else:
                            last_cash[col] = cash_now
                        last_shares[col] = shares_now

                    from_col = to_col

    return order_records[:ridx], log_records[:lidx]


@njit(cache=True)
def simulate_from_orders_nb(target_shape, group_lens, init_cash, call_seq, auto_call_seq,
                            size, size_type, direction, price, fees, fixed_fees, slippage,
                            min_size, max_size, reject_prob, close_first, allow_partial,
                            raise_reject, log, val_price, flex_2d):
    """Adaptation of `simulate_nb` for simulation based on orders.

    Utilizes flexible broadcasting.

    !!! note
        Should be only grouped if cash sharing is enabled.

        If `auto_call_seq` is True, make sure that `call_seq` follows `CallSeqType.Default`."""
    check_group_lens(group_lens, target_shape[1])
    cash_sharing = is_grouped_nb(group_lens)
    check_group_init_cash(group_lens, target_shape[1], init_cash, cash_sharing)

    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    ridx = 0
    log_records = np.empty(target_shape[0] * target_shape[1], dtype=log_dt)
    lidx = 0
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
                order = create_order_nb(
                    size=flex_select_auto_nb(i, col, size, flex_2d),
                    size_type=flex_select_auto_nb(i, col, size_type, flex_2d),
                    direction=flex_select_auto_nb(i, col, direction, flex_2d),
                    price=flex_select_auto_nb(i, col, price, flex_2d),
                    fees=flex_select_auto_nb(i, col, fees, flex_2d),
                    fixed_fees=flex_select_auto_nb(i, col, fixed_fees, flex_2d),
                    slippage=flex_select_auto_nb(i, col, slippage, flex_2d),
                    min_size=flex_select_auto_nb(i, col, min_size, flex_2d),
                    max_size=flex_select_auto_nb(i, col, max_size, flex_2d),
                    reject_prob=flex_select_auto_nb(i, col, reject_prob, flex_2d),
                    close_first=flex_select_auto_nb(i, col, close_first, flex_2d),
                    allow_partial=flex_select_auto_nb(i, col, allow_partial, flex_2d),
                    raise_reject=flex_select_auto_nb(i, col, raise_reject, flex_2d),
                    log=flex_select_auto_nb(i, col, log, flex_2d)
                )

                # Process the order
                cash_now, shares_now, order_result = process_order_nb(
                    cash_now, shares_now, val_price_now, value_now, order, log_records[lidx])

                if order.log:
                    # Add log metadata
                    log_records[lidx]['id'] = lidx
                    log_records[lidx]['idx'] = i
                    log_records[lidx]['col'] = col
                    log_records[lidx]['group'] = group
                    if order_result.status == OrderStatus.Filled:
                        log_records[lidx]['order_id'] = ridx
                    else:
                        log_records[lidx]['order_id'] = -1
                    lidx += 1

                if order_result.status == OrderStatus.Filled:
                    # Add order metadata
                    order_records[ridx]['id'] = ridx
                    order_records[ridx]['idx'] = i
                    order_records[ridx]['col'] = col
                    order_records[ridx]['size'] = order_result.size
                    order_records[ridx]['price'] = order_result.price
                    order_records[ridx]['fees'] = order_result.fees
                    order_records[ridx]['side'] = order_result.side
                    ridx += 1

                # Now becomes last
                if cash_sharing:
                    last_cash[group] = cash_now
                else:
                    last_cash[col] = cash_now
                last_shares[col] = shares_now

        from_col = to_col

    return order_records[:ridx], log_records[:lidx]


@njit(cache=True)
def signals_get_size_nb(shares_now, is_entry, is_exit, size, accumulate, conflict_mode, direction):
    """Get order size given signals."""
    order_size = 0.
    abs_shares_now = abs(shares_now)
    abs_size = abs(size)

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
            # Behaves like Direction.LongOnly
            if accumulate:
                order_size = abs_size
            else:
                if shares_now < 0:
                    # Reverse short position
                    order_size = abs_shares_now + abs_size
                elif shares_now == 0:
                    # Open long position
                    order_size = abs_size
        elif direction == Direction.LongOnly:
            if shares_now == 0 or accumulate:
                # Open or increase long position
                order_size = abs_size
        else:
            if shares_now == 0 or accumulate:
                # Open or increase short position
                order_size = -abs_size

    elif not is_entry and is_exit:
        if direction == Direction.All:
            # Behaves like Direction.ShortOnly
            if accumulate:
                order_size = -abs_size
            else:
                if shares_now > 0:
                    # Reverse long position
                    order_size = -abs_shares_now - abs_size
                elif shares_now == 0:
                    # Open short position
                    order_size = -abs_size
        elif direction == Direction.ShortOnly:
            if shares_now < 0:
                if accumulate:
                    # Reduce short position
                    order_size = abs_size
                else:
                    # Close short position
                    order_size = abs_shares_now
        else:
            if shares_now > 0:
                if accumulate:
                    # Reduce long position
                    order_size = -abs_size
                else:
                    # Close long position
                    order_size = -abs_shares_now
    return order_size


@njit(cache=True)
def simulate_from_signals_nb(target_shape, group_lens, init_cash, call_seq, auto_call_seq,
                             entries, exits, size, price, fees, fixed_fees, slippage,
                             min_size, max_size, reject_prob, close_first, allow_partial,
                             raise_reject, accumulate, log, conflict_mode, direction,
                             val_price, flex_2d):
    """Adaptation of `simulate_nb` for simulation based on entry and exit signals.

    Utilizes flexible broadcasting.

    !!! note
        Should be only grouped if cash sharing is enabled."""
    check_group_lens(group_lens, target_shape[1])
    cash_sharing = is_grouped_nb(group_lens)
    check_group_init_cash(group_lens, target_shape[1], init_cash, cash_sharing)

    order_records = np.empty(target_shape[0] * target_shape[1], dtype=order_dt)
    ridx = 0
    log_records = np.empty(target_shape[0] * target_shape[1], dtype=log_dt)
    lidx = 0
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
                    flex_select_auto_nb(i, col, size, flex_2d),
                    flex_select_auto_nb(i, col, accumulate, flex_2d),
                    flex_select_auto_nb(i, col, conflict_mode, flex_2d),
                    flex_select_auto_nb(i, col, direction, flex_2d)
                )  # already takes into account direction
                order_size[col] = _order_size

                if cash_sharing:
                    if _order_size == 0:
                        temp_order_value[k] = 0.
                    else:
                        _order_price = flex_select_auto_nb(i, col, price, flex_2d)
                        temp_order_value[k] = _order_size * _order_price

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
                        if _direction == Direction.ShortOnly:
                            _order_size *= -1  # must reverse for process_order_nb
                    else:  # short order
                        _direction = flex_select_auto_nb(i, col, direction, flex_2d)
                        if _direction == Direction.ShortOnly:
                            _order_size *= -1
                    order = create_order_nb(
                        size=_order_size,
                        size_type=SizeType.Shares,
                        direction=_direction,
                        price=flex_select_auto_nb(i, col, price, flex_2d),
                        fees=flex_select_auto_nb(i, col, fees, flex_2d),
                        fixed_fees=flex_select_auto_nb(i, col, fixed_fees, flex_2d),
                        slippage=flex_select_auto_nb(i, col, slippage, flex_2d),
                        min_size=flex_select_auto_nb(i, col, min_size, flex_2d),
                        max_size=flex_select_auto_nb(i, col, max_size, flex_2d),
                        close_first=flex_select_auto_nb(i, col, close_first, flex_2d),
                        reject_prob=flex_select_auto_nb(i, col, reject_prob, flex_2d),
                        allow_partial=flex_select_auto_nb(i, col, allow_partial, flex_2d),
                        raise_reject=flex_select_auto_nb(i, col, raise_reject, flex_2d),
                        log=flex_select_auto_nb(i, col, log, flex_2d)
                    )

                    # Process the order
                    cash_now, shares_now, order_result = process_order_nb(
                        cash_now, shares_now, val_price_now, value_now, order, log_records[lidx])

                    if order.log:
                        # Add log metadata
                        log_records[lidx]['id'] = lidx
                        log_records[lidx]['idx'] = i
                        log_records[lidx]['col'] = col
                        log_records[lidx]['group'] = group
                        if order_result.status == OrderStatus.Filled:
                            log_records[lidx]['order_id'] = ridx
                        else:
                            log_records[lidx]['order_id'] = -1
                        lidx += 1

                    if order_result.status == OrderStatus.Filled:
                        # Add order metadata
                        order_records[ridx]['id'] = ridx
                        order_records[ridx]['idx'] = i
                        order_records[ridx]['col'] = col
                        order_records[ridx]['size'] = order_result.size
                        order_records[ridx]['price'] = order_result.price
                        order_records[ridx]['fees'] = order_result.fees
                        order_records[ridx]['side'] = order_result.side
                        ridx += 1

                # Now becomes last
                if cash_sharing:
                    last_cash[group] = cash_now
                else:
                    last_cash[col] = cash_now
                last_shares[col] = shares_now

        from_col = to_col

    return order_records[:ridx], log_records[:lidx]


# ############# Trades ############# #


@njit(cache=True)
def trade_duration_map_nb(record):
    """`map_func_nb` that returns trade duration."""
    return record['exit_idx'] - record['entry_idx']


@njit(cache=True)
def get_trade_stats_nb(size, entry_price, entry_fees, exit_price, exit_fees, direction):
    """Get trade statistics."""
    entry_val = size * entry_price
    exit_val = size * exit_price
    val_diff = add_nb(exit_val, -entry_val)
    if val_diff != 0 and direction == TradeDirection.Short:
        val_diff *= -1
    pnl = val_diff - entry_fees - exit_fees
    ret = pnl / entry_val
    return pnl, ret


size_zero_neg_err = "Found order with size 0 or less"
price_zero_neg_err = "Found order with price 0 or less"


@njit(cache=True)
def save_trade_nb(record, col,
                  entry_idx, entry_size_sum, entry_gross_sum, entry_fees_sum,
                  exit_idx, exit_size, exit_price, exit_fees,
                  direction, status, position_id):
    """Save trade to the record."""
    # Size-weighted average of price
    entry_price = entry_gross_sum / entry_size_sum

    # Fraction of fees
    size_fraction = exit_size / entry_size_sum
    entry_fees = size_fraction * entry_fees_sum

    # Get P&L and return
    pnl, ret = get_trade_stats_nb(
        exit_size,
        entry_price,
        entry_fees,
        exit_price,
        exit_fees,
        direction
    )

    # Save trade
    record['col'] = col
    record['size'] = exit_size
    record['entry_idx'] = entry_idx
    record['entry_price'] = entry_price
    record['entry_fees'] = entry_fees
    record['exit_idx'] = exit_idx
    record['exit_price'] = exit_price
    record['exit_fees'] = exit_fees
    record['pnl'] = pnl
    record['return'] = ret
    record['direction'] = direction
    record['status'] = status
    record['position_id'] = position_id


@njit(cache=True)
def orders_to_trades_nb(close, order_records, col_map):
    """Find trades and store their information as records to an array.

    ## Example

    Simulate a strategy and find all trades in generated orders:
    ```python-repl
    >>> import numpy as np
    >>> import pandas as pd
    >>> from numba import njit
    >>> from vectorbt.records.nb import col_map_nb
    >>> from vectorbt.portfolio.nb import (
    ...     simulate_nb,
    ...     create_order_nb,
    ...     empty_prep_nb,
    ...     orders_to_trades_nb
    ... )

    >>> @njit
    ... def order_func_nb(oc, order_size, order_price):
    ...     return create_order_nb(
    ...         size=order_size[oc.i, oc.col],
    ...         price=order_price[oc.i, oc.col],
    ...         fees=0.01, slippage=0.01
    ...     )

    >>> order_size = np.asarray([
    ...     [1, -1],
    ...     [0.1, -0.1],
    ...     [-1, 1],
    ...     [-0.1, 0.1],
    ...     [1, -1],
    ...     [-2, 2]
    ... ])
    >>> close = order_price = np.array([
    ...     [1, 6],
    ...     [2, 5],
    ...     [3, 4],
    ...     [4, 3],
    ...     [5, 2],
    ...     [6, 1]
    ... ])
    >>> target_shape = order_size.shape
    >>> group_lens = np.full(target_shape[1], 1)
    >>> init_cash = np.full(target_shape[1], 100)
    >>> cash_sharing = False
    >>> call_seq = np.full(target_shape, 0)
    >>> active_mask = np.full(target_shape, True)

    >>> order_records, log_records = simulate_nb(
    ...     target_shape, close, group_lens,
    ...     init_cash, cash_sharing, call_seq, active_mask,
    ...     empty_prep_nb, (),
    ...     empty_prep_nb, (),
    ...     empty_prep_nb, (),
    ...     order_func_nb, (order_size, order_price))

    >>> col_map = col_map_nb(order_records['col'], target_shape[1])
    >>> trade_records = orders_to_trades_nb(close, order_records, col_map)
    >>> print(pd.DataFrame.from_records(trade_records))
       id  col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
    0   0    0   1.0          0     1.101818    0.011018         2        2.97
    1   1    0   0.1          0     1.101818    0.001102         3        3.96
    2   2    0   1.0          4     5.050000    0.050500         5        5.94
    3   3    0   1.0          5     5.940000    0.059400         5        6.00
    4   4    1   1.0          0     5.850000    0.058500         2        4.04
    5   5    1   0.1          0     5.850000    0.005850         3        3.03
    6   6    1   1.0          4     1.980000    0.019800         5        1.01
    7   7    1   1.0          5     1.010000    0.010100         5        1.00

       exit_fees       pnl    return  direction  status  position_id
    0    0.02970  1.827464  1.658589          0       1            0
    1    0.00396  0.280756  2.548119          0       1            0
    2    0.05940  0.780100  0.154475          0       1            1
    3    0.00000 -0.119400 -0.020101          1       0            2
    4    0.04040  1.711100  0.292496          1       1            3
    5    0.00303  0.273120  0.466872          1       1            3
    6    0.01010  0.940100  0.474798          1       1            4
    7    0.00000 -0.020100 -0.019901          0       0            5
    ```
    """
    col_idxs, col_ns = col_map
    records = np.empty(close.shape[0] * close.shape[1], dtype=trade_dt)
    ridx = 0
    entry_size_sum = 0.
    entry_gross_sum = 0.
    entry_fees_sum = 0.
    position_id = -1

    for col in range(col_idxs.shape[0]):
        n = col_ns[col]
        if n == 0:
            continue
        entry_idx = -1
        direction = -1
        last_id = -1

        for i in range(n):
            r = col_idxs[col][i]
            record = order_records[r]

            if record['id'] < last_id:
                raise ValueError("id must come in ascending order per column")
            last_id = record['id']

            i = record['idx']
            order_size = record['size']
            order_price = record['price']
            order_fees = record['fees']
            order_side = record['side']

            if order_size <= 0.:
                raise ValueError(size_zero_neg_err)
            if order_price <= 0.:
                raise ValueError(price_zero_neg_err)

            if entry_idx == -1:
                # Trade opened
                entry_idx = i
                if order_side == OrderSide.Buy:
                    direction = TradeDirection.Long
                else:
                    direction = TradeDirection.Short
                position_id += 1

                # Reset running vars for a new position
                entry_size_sum = 0.
                entry_gross_sum = 0.
                entry_fees_sum = 0.

            if (direction == TradeDirection.Long and order_side == OrderSide.Buy) \
                    or (direction == TradeDirection.Short and order_side == OrderSide.Sell):
                # Position increased
                entry_size_sum += order_size
                entry_gross_sum += order_size * order_price
                entry_fees_sum += order_fees

            elif (direction == TradeDirection.Long and order_side == OrderSide.Sell) \
                    or (direction == TradeDirection.Short and order_side == OrderSide.Buy):
                if is_close_or_less_nb(order_size, entry_size_sum):
                    # Trade closed
                    if is_close_nb(order_size, entry_size_sum):
                        exit_size = entry_size_sum
                    else:
                        exit_size = order_size
                    exit_price = order_price
                    exit_fees = order_fees
                    exit_idx = i
                    save_trade_nb(
                        records[ridx],
                        col,
                        entry_idx,
                        entry_size_sum,
                        entry_gross_sum,
                        entry_fees_sum,
                        exit_idx,
                        exit_size,
                        exit_price,
                        exit_fees,
                        direction,
                        TradeStatus.Closed,
                        position_id
                    )
                    records[ridx]['id'] = ridx
                    ridx += 1

                    if is_close_nb(order_size, entry_size_sum):
                        # Position closed
                        entry_idx = -1
                        direction = -1
                    else:
                        # Position decreased, previous orders have now less impact
                        size_fraction = (entry_size_sum - order_size) / entry_size_sum
                        entry_size_sum *= size_fraction
                        entry_gross_sum *= size_fraction
                        entry_fees_sum *= size_fraction
                else:
                    # Trade reversed
                    # Close current trade
                    cl_exit_size = entry_size_sum
                    cl_exit_price = order_price
                    cl_exit_fees = cl_exit_size / order_size * order_fees
                    cl_exit_idx = i
                    save_trade_nb(
                        records[ridx],
                        col,
                        entry_idx,
                        entry_size_sum,
                        entry_gross_sum,
                        entry_fees_sum,
                        cl_exit_idx,
                        cl_exit_size,
                        cl_exit_price,
                        cl_exit_fees,
                        direction,
                        TradeStatus.Closed,
                        position_id
                    )
                    records[ridx]['id'] = ridx
                    ridx += 1

                    # Open a new trade
                    entry_size_sum = order_size - cl_exit_size
                    entry_gross_sum = entry_size_sum * order_price
                    entry_fees_sum = order_fees - cl_exit_fees
                    entry_idx = i
                    if direction == TradeDirection.Long:
                        direction = TradeDirection.Short
                    else:
                        direction = TradeDirection.Long
                    position_id += 1

        if entry_idx != -1 and is_less_nb(-entry_size_sum, 0):
            # Trade in the previous column hasn't been closed
            exit_size = entry_size_sum
            exit_price = close[close.shape[0] - 1, col]
            exit_fees = 0.
            exit_idx = close.shape[0] - 1
            save_trade_nb(
                records[ridx],
                col,
                entry_idx,
                entry_size_sum,
                entry_gross_sum,
                entry_fees_sum,
                exit_idx,
                exit_size,
                exit_price,
                exit_fees,
                direction,
                TradeStatus.Open,
                position_id
            )
            records[ridx]['id'] = ridx
            ridx += 1

    return records[:ridx]


# ############# Positions ############# #

@njit(cache=True)
def save_position_nb(record, trade_records):
    """Save position to the record."""
    # Aggregate trades
    col = trade_records['col'][0]
    size = np.sum(trade_records['size'])
    entry_idx = trade_records['entry_idx'][0]
    entry_price = np.sum(trade_records['size'] * trade_records['entry_price']) / size
    entry_fees = np.sum(trade_records['entry_fees'])
    exit_idx = trade_records['exit_idx'][-1]
    exit_price = np.sum(trade_records['size'] * trade_records['exit_price']) / size
    exit_fees = np.sum(trade_records['exit_fees'])
    direction = trade_records['direction'][-1]
    status = trade_records['status'][-1]
    pnl, ret = get_trade_stats_nb(
        size,
        entry_price,
        entry_fees,
        exit_price,
        exit_fees,
        direction
    )

    # Save position
    record['col'] = col
    record['size'] = size
    record['entry_idx'] = entry_idx
    record['entry_price'] = entry_price
    record['entry_fees'] = entry_fees
    record['exit_idx'] = exit_idx
    record['exit_price'] = exit_price
    record['exit_fees'] = exit_fees
    record['pnl'] = pnl
    record['return'] = ret
    record['direction'] = direction
    record['status'] = status


@njit(cache=True)
def copy_trade_record_nb(position_record, trade_record):
    # Save position
    position_record['col'] = trade_record['col']
    position_record['size'] = trade_record['size']
    position_record['entry_idx'] = trade_record['entry_idx']
    position_record['entry_price'] = trade_record['entry_price']
    position_record['entry_fees'] = trade_record['entry_fees']
    position_record['exit_idx'] = trade_record['exit_idx']
    position_record['exit_price'] = trade_record['exit_price']
    position_record['exit_fees'] = trade_record['exit_fees']
    position_record['pnl'] = trade_record['pnl']
    position_record['return'] = trade_record['return']
    position_record['direction'] = trade_record['direction']
    position_record['status'] = trade_record['status']


@njit(cache=True)
def trades_to_positions_nb(trade_records, col_map):
    """Find positions and store their information as records to an array.

    ## Example

    Building upon the example in `orders_to_trades_nb`, convert trades to positions:
    ```python-repl
    >>> from vectorbt.portfolio.nb import trades_to_positions_nb

    >>> col_map = col_map_nb(trade_records['col'], target_shape[1])
    >>> position_records = trades_to_positions_nb(trade_records, col_map)
    >>> pd.DataFrame.from_records(position_records)
       id  col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
    0   0    0   1.1          0     1.101818     0.01212         3    3.060000   
    1   1    0   1.0          4     5.050000     0.05050         5    5.940000   
    2   2    0   1.0          5     5.940000     0.05940         5    6.000000   
    3   3    1   1.1          0     5.850000     0.06435         3    3.948182   
    4   4    1   1.0          4     1.980000     0.01980         5    1.010000   
    5   5    1   1.0          5     1.010000     0.01010         5    1.000000   
    
       exit_fees      pnl    return  direction  status  
    0    0.03366  2.10822  1.739455          0       1  
    1    0.05940  0.78010  0.154475          0       1  
    2    0.00000 -0.11940 -0.020101          1       0  
    3    0.04343  1.98422  0.308348          1       1  
    4    0.01010  0.94010  0.474798          1       1  
    5    0.00000 -0.02010 -0.019901          0       0  
    ```
    """
    col_idxs, col_ns = col_map
    records = np.empty(trade_records.shape[0], dtype=position_dt)
    ridx = 0
    from_r = -1

    for col in range(col_idxs.shape[0]):
        n = col_ns[col]
        if n == 0:
            continue
        last_id = -1
        last_position_id = -1

        for i in range(n):
            r = col_idxs[col][i]
            record = trade_records[r]

            if record['id'] < last_id:
                raise ValueError("id must come in ascending order per column")
            last_id = record['id']

            position_id = record['position_id']

            if position_id != last_position_id:
                if last_position_id != -1:
                    if r - from_r > 1:
                        save_position_nb(records[ridx], trade_records[from_r:r])
                    else:
                        # Speed up
                        copy_trade_record_nb(records[ridx], trade_records[from_r])
                    records[ridx]['id'] = ridx
                    ridx += 1
                from_r = r
                last_position_id = position_id

        if r - from_r > 0:
            save_position_nb(records[ridx], trade_records[from_r:r + 1])
        else:
            # Speed up
            copy_trade_record_nb(records[ridx], trade_records[from_r])
        records[ridx]['id'] = ridx
        ridx += 1

    return records[:ridx]


# ############# Shares ############# #


@njit(cache=True)
def get_long_size_nb(shares_now, new_shares_now):
    """Get long size."""
    if shares_now <= 0 and new_shares_now <= 0:
        return 0.
    if shares_now >= 0 and new_shares_now < 0:
        return -shares_now
    if shares_now < 0 and new_shares_now >= 0:
        return new_shares_now
    return add_nb(new_shares_now, -shares_now)


@njit(cache=True)
def get_short_size_nb(shares_now, new_shares_now):
    """Get short size."""
    if shares_now >= 0 and new_shares_now >= 0:
        return 0.
    if shares_now >= 0 and new_shares_now < 0:
        return -new_shares_now
    if shares_now < 0 and new_shares_now >= 0:
        return shares_now
    return add_nb(shares_now, -new_shares_now)


@njit(cache=True)
def share_flow_nb(target_shape, order_records, col_map, direction):
    """Get share flow series per column. Has opposite sign."""
    col_idxs, col_ns = col_map
    out = np.full(target_shape, 0., dtype=np.float_)

    for col in range(col_idxs.shape[0]):
        n = col_ns[col]
        if n == 0:
            continue
        last_id = -1
        shares_now = 0.

        for i in range(n):
            r = col_idxs[col][i]
            record = order_records[r]

            if record['id'] < last_id:
                raise ValueError("id must come in ascending order per column")
            last_id = record['id']

            i = record['idx']
            side = record['side']
            size = record['size']

            if side == OrderSide.Sell:
                size *= -1
            new_shares_now = add_nb(shares_now, size)
            if direction == Direction.LongOnly:
                out[i, col] += get_long_size_nb(shares_now, new_shares_now)
            elif direction == Direction.ShortOnly:
                out[i, col] += get_short_size_nb(shares_now, new_shares_now)
            else:
                out[i, col] += size
            shares_now = new_shares_now
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
def i_group_any_reduce_nb(i, group, a):
    """Boolean "any" reducer for grouped columns."""
    return np.any(a)


@njit
def pos_mask_grouped_nb(pos_mask, group_lens):
    """Get number of columns in position for each row and group."""
    return generic_nb.squeeze_grouped_nb(pos_mask, group_lens, i_group_any_reduce_nb).astype(np.bool_)


@njit(cache=True)
def group_mean_reduce_nb(group, a):
    """Mean reducer for grouped columns."""
    return np.mean(a)


@njit
def pos_coverage_grouped_nb(pos_mask, group_lens):
    """Get coverage of position for each row and group."""
    return generic_nb.reduce_grouped_nb(pos_mask, group_lens, group_mean_reduce_nb)


# ############# Cash ############# #


@njit(cache=True)
def cash_flow_nb(target_shape, order_records, col_map, short_cash):
    """Get cash flow series per column."""
    col_idxs, col_ns = col_map
    out = np.full(target_shape, 0., dtype=np.float_)

    for col in range(col_idxs.shape[0]):
        n = col_ns[col]
        if n == 0:
            continue
        last_id = -1
        shares_now = 0.
        debt_now = 0.

        for i in range(n):
            r = col_idxs[col][i]
            record = order_records[r]

            if record['id'] < last_id:
                raise ValueError("id must come in ascending order per column")
            last_id = record['id']

            i = record['idx']
            side = record['side']
            size = record['size']
            price = record['price']
            fees = record['fees']
            volume = size * price

            if side == OrderSide.Sell:
                size *= -1
            new_shares_now = add_nb(shares_now, size)
            shorted_size = get_short_size_nb(shares_now, new_shares_now)

            if not short_cash and shorted_size != 0:
                if shorted_size > 0:
                    debt_now += shorted_size * price
                    out[i, col] += add_nb(volume, -2 * shorted_size * price)
                else:
                    if is_close_nb(volume, debt_now):
                        volume = debt_now
                    if volume >= debt_now:
                        out[i, col] += add_nb(2 * debt_now, -volume)
                        debt_now = 0.
                    else:
                        out[i, col] += volume
                        debt_now -= volume
            else:
                if side == OrderSide.Buy:
                    out[i, col] -= volume
                else:
                    out[i, col] += volume
            out[i, col] -= fees
            shares_now = new_shares_now
    return out


@njit(cache=True)
def cash_flow_grouped_nb(cash_flow, group_lens):
    """Get cash flow series per group."""
    check_group_lens(group_lens, cash_flow.shape[1])

    out = np.empty((cash_flow.shape[0], len(group_lens)), dtype=np.float_)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        out[:, group] = np.sum(cash_flow[:, from_col:to_col], axis=1)
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
def init_cash_nb(init_cash, group_lens, cash_sharing):
    """Get initial cash per column."""
    if not cash_sharing:
        return init_cash
    group_lens_cs = np.cumsum(group_lens)
    out = np.full(group_lens_cs[-1], np.nan, dtype=np.float_)
    out[group_lens_cs - group_lens] = init_cash
    out = generic_nb.ffill_1d_nb(out)
    return out


@njit(cache=True)
def cash_nb(cash_flow, group_lens, init_cash):
    """Get cash series per column."""
    check_group_lens(group_lens, cash_flow.shape[1])

    out = np.empty_like(cash_flow)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        for i in range(cash_flow.shape[0]):
            for col in range(from_col, to_col):
                cash_now = init_cash[col] if i == 0 else out[i - 1, col]
                flow_value = cash_flow[i, col]
                out[i, col] = add_nb(cash_now, flow_value)
        from_col = to_col
    return out


@njit(cache=True)
def cash_in_sim_order_nb(cash_flow, group_lens, init_cash_grouped, call_seq):
    """Get cash series in simulation order."""
    check_group_lens(group_lens, cash_flow.shape[1])

    out = np.empty_like(cash_flow)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_len = to_col - from_col
        cash_now = init_cash_grouped[group]
        for i in range(cash_flow.shape[0]):
            for k in range(group_len):
                col = from_col + call_seq[i, from_col + k]
                out[i, col] = add_nb(cash_now, cash_flow[i, col])
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
def holding_value_nb(close, shares):
    """Get holding value series per column."""
    return close * shares


@njit(cache=True)
def holding_value_grouped_nb(holding_value, group_lens):
    """Get holding value series per group."""
    check_group_lens(group_lens, holding_value.shape[1])

    out = np.empty((holding_value.shape[0], len(group_lens)), dtype=np.float_)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        out[:, group] = np.sum(holding_value[:, from_col:to_col], axis=1)
        from_col = to_col
    return out


@njit(cache=True)
def value_in_sim_order_nb(cash, holding_value, group_lens, call_seq):
    """Get portfolio value series in simulation order."""
    check_group_lens(group_lens, cash.shape[1])

    out = np.empty_like(cash)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_len = to_col - from_col
        curr_holding_value = 0.
        # Without correctly treating NaN values, after one NaN all will be NaN
        since_last_nan = group_len
        for j in range(cash.shape[0] * group_len):
            i = j // group_len
            col = from_col + call_seq[i, from_col + j % group_len]
            if j >= group_len:
                last_j = j - group_len
                last_i = last_j // group_len
                last_col = from_col + call_seq[last_i, from_col + last_j % group_len]
                if not np.isnan(holding_value[last_i, last_col]):
                    curr_holding_value -= holding_value[last_i, last_col]
            if np.isnan(holding_value[i, col]):
                since_last_nan = 0
            else:
                curr_holding_value += holding_value[i, col]
            if since_last_nan < group_len:
                out[i, col] = np.nan
            else:
                out[i, col] = cash[i, col] + curr_holding_value
            since_last_nan += 1

        from_col = to_col
    return out


@njit(cache=True)
def value_nb(cash, holding_value):
    """Get portfolio value series per column/group."""
    return cash + holding_value


@njit(cache=True)
def total_profit_nb(target_shape, close, order_records, col_map, init_cash):
    """Get total profit per column.

    A much faster version than the one based on `value_nb`."""
    col_idxs, col_ns = col_map
    shares = np.full(target_shape[1], 0., dtype=np.float_)
    cash = init_cash.copy()

    for col in range(col_idxs.shape[0]):
        n = col_ns[col]
        if n == 0:
            continue
        last_id = -1

        for i in range(n):
            r = col_idxs[col][i]
            record = order_records[r]

            if record['id'] < last_id:
                raise ValueError("id must come in ascending order per column")
            last_id = record['id']

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

    return cash + shares * close[-1, :] - init_cash


@njit(cache=True)
def total_profit_grouped_nb(total_profit, group_lens):
    """Get total profit per group."""
    check_group_lens(group_lens, total_profit.shape[0])

    out = np.empty((len(group_lens),), dtype=np.float_)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        out[group] = np.sum(total_profit[from_col:to_col])
        from_col = to_col
    return out


@njit(cache=True)
def final_value_nb(total_profit, init_cash):
    """Get total profit per column/group."""
    return total_profit + init_cash


@njit(cache=True)
def total_return_nb(total_profit, init_cash):
    """Get total return per column/group."""
    return total_profit / init_cash


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
def returns_nb(value, init_cash):
    """Get portfolio return series per column/group."""
    out = np.empty(value.shape, dtype=np.float_)
    for col in range(out.shape[1]):
        input_value = init_cash[col]
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
def market_value_nb(close, init_cash):
    """Get market value per column."""
    return close / close[0] * init_cash


@njit(cache=True)
def market_value_grouped_nb(close, group_lens, init_cash_grouped):
    """Get market value per group."""
    check_group_lens(group_lens, close.shape[1])

    out = np.empty((close.shape[0], len(group_lens)), dtype=np.float_)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_len = to_col - from_col
        col_init_cash = init_cash_grouped[group] / group_len
        close_norm = close[:, from_col:to_col] / close[0, from_col:to_col]
        out[:, group] = col_init_cash * np.sum(close_norm, axis=1)
        from_col = to_col
    return out


@njit(cache=True)
def total_market_return_nb(market_value):
    """Get total market return per column/group."""
    out = np.empty((market_value.shape[1],), dtype=np.float_)
    for col in range(market_value.shape[1]):
        out[col] = get_return_nb(market_value[0, col], market_value[-1, col])
    return out


@njit(cache=True)
def gross_exposure_nb(holding_value, cash):
    """Get gross exposure per column/group."""
    out = np.empty(holding_value.shape, dtype=np.float_)
    for col in range(out.shape[1]):
        for i in range(out.shape[0]):
            denom = add_nb(holding_value[i, col], cash[i, col])
            if denom == 0:
                out[i, col] = 0.
            else:
                out[i, col] = holding_value[i, col] / denom
    return out
