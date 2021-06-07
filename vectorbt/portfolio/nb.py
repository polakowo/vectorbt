"""Numba-compiled functions.

Provides an arsenal of Numba-compiled functions that are used for portfolio
modeling, such as generating and filling orders. These only accept NumPy arrays and
other Numba-compatible types.

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
    For example, reduce by `np.inf` or `position_now` to close a long/short position.

    See `vectorbt.utils.math` for current tolerance values.
"""

import numpy as np
from numba import njit

from vectorbt import _typing as tp
from vectorbt.utils.math import (
    is_close_nb,
    is_close_or_less_nb,
    is_less_nb,
    add_nb
)
from vectorbt.utils.array import insert_argsort_nb
from vectorbt.base.reshape_fns import flex_select_auto_nb
from vectorbt.generic import nb as generic_nb
from vectorbt.portfolio.enums import *


# ############# Simulation ############# #


@njit(cache=True)
def order_not_filled_nb(status: int, status_info: int) -> OrderResult:
    """Return `OrderResult` for order that hasn't been filled."""
    return OrderResult(np.nan, np.nan, np.nan, -1, status, status_info)


@njit(cache=True)
def buy_nb(exec_state: ExecuteOrderState,
           size: float,
           price: float,
           direction: int = Direction.All,
           fees: float = 0.,
           fixed_fees: float = 0.,
           slippage: float = 0.,
           min_size: float = 0.,
           max_size: float = np.inf,
           allow_partial: bool = True,
           percent: float = np.nan) -> tp.Tuple[ExecuteOrderState, OrderResult]:
    """Buy or/and cover."""

    # Set cash limit
    cash_limit = exec_state.cash
    if not np.isnan(percent):
        # Apply percentage
        cash_limit = min(cash_limit, percent * cash_limit)

    if direction == Direction.LongOnly or direction == Direction.All:
        if cash_limit == 0:
            return exec_state, order_not_filled_nb(OrderStatus.Rejected, StatusInfo.NoCashLong)
        if np.isinf(size) and np.isinf(cash_limit):
            raise ValueError("Attempt to go in long direction infinitely")
    else:
        if exec_state.position == 0:
            return exec_state, order_not_filled_nb(OrderStatus.Rejected, StatusInfo.NoOpenPosition)

    # Get price adjusted with slippage
    adj_price = price * (1 + slippage)

    # Get optimal order size
    if direction == Direction.ShortOnly:
        adj_size = min(-exec_state.position, size)
    else:
        adj_size = size

    if adj_size == 0:
        return exec_state, order_not_filled_nb(OrderStatus.Ignored, StatusInfo.SizeZero)

    if adj_size > max_size:
        if not allow_partial:
            return exec_state, order_not_filled_nb(OrderStatus.Rejected, StatusInfo.MaxSizeExceeded)

        adj_size = max_size

    # Get cash required to complete this order
    req_cash = adj_size * adj_price
    req_fees = req_cash * fees + fixed_fees
    adj_req_cash = req_cash + req_fees

    if is_close_or_less_nb(adj_req_cash, cash_limit):
        # Sufficient amount of cash
        final_size = adj_size
        fees_paid = req_fees
        final_cash = adj_req_cash
    else:
        # Insufficient amount of cash, size will be less than requested

        # For fees of 10% and 1$ per transaction, you can buy for 90$ (new_req_cash)
        # to spend 100$ (cash_limit) in total
        new_req_cash = add_nb(cash_limit, -fixed_fees) / (1 + fees)
        if new_req_cash <= 0:
            return exec_state, order_not_filled_nb(OrderStatus.Rejected, StatusInfo.CantCoverFees)

        final_size = new_req_cash / adj_price
        fees_paid = cash_limit - new_req_cash
        final_cash = cash_limit

    # Check against minimum size
    if is_less_nb(final_size, min_size):
        return exec_state, order_not_filled_nb(OrderStatus.Rejected, StatusInfo.MinSizeNotReached)

    # Check against partial fill (np.inf doesn't count)
    if np.isfinite(size) and is_less_nb(final_size, size) and not allow_partial:
        return exec_state, order_not_filled_nb(OrderStatus.Rejected, StatusInfo.PartialFill)

    # Update current cash balance and position
    new_cash = add_nb(exec_state.cash, -final_cash)
    new_position = add_nb(exec_state.position, final_size)

    # Update current debt and free cash
    if exec_state.position < 0:
        if new_position < 0:
            short_size = final_size
        else:
            short_size = abs(exec_state.position)
        avg_entry_price = exec_state.debt / abs(exec_state.position)
        debt_diff = short_size * avg_entry_price
        new_debt = add_nb(exec_state.debt, -debt_diff)
        new_free_cash = add_nb(exec_state.free_cash + 2 * debt_diff, -final_cash)
    else:
        new_debt = exec_state.debt
        new_free_cash = add_nb(exec_state.free_cash, -final_cash)

    # Return filled order
    order_result = OrderResult(
        final_size,
        adj_price,
        fees_paid,
        OrderSide.Buy,
        OrderStatus.Filled,
        -1
    )
    new_exec_state = ExecuteOrderState(
        cash=new_cash,
        position=new_position,
        debt=new_debt,
        free_cash=new_free_cash
    )
    return new_exec_state, order_result


@njit(cache=True)
def sell_nb(exec_state: ExecuteOrderState,
            size: float,
            price: float,
            direction: int = Direction.All,
            fees: float = 0.,
            fixed_fees: float = 0.,
            slippage: float = 0.,
            min_size: float = 0.,
            max_size: float = np.inf,
            lock_cash: bool = False,
            allow_partial: bool = True,
            percent: float = np.nan) -> tp.Tuple[ExecuteOrderState, OrderResult]:
    """Sell or/and short sell."""

    # Get price adjusted with slippage
    adj_price = price * (1 - slippage)

    # Get optimal order size
    if direction == Direction.LongOnly:
        size_limit = min(exec_state.position, size)
    else:
        if lock_cash or (np.isinf(size) and not np.isnan(percent)):
            # Get the maximum size that can be (short) sold
            long_size = max(exec_state.position, 0)
            long_cash = long_size * adj_price * (1 - fees)
            total_free_cash = add_nb(exec_state.free_cash, long_cash)

            if total_free_cash <= 0:
                if exec_state.position <= 0:
                    # There is nothing to sell, and no free cash to short sell
                    return exec_state, order_not_filled_nb(OrderStatus.Rejected, StatusInfo.NoCashShort)

                # There is position to close, but no free cash to short sell
                max_size_limit = long_size
            else:
                # There is position to close and/or free cash to short sell
                max_short_size = add_nb(total_free_cash, -fixed_fees) / (adj_price * (1 + fees))
                max_size_limit = add_nb(long_size, max_short_size)
                if max_size_limit <= 0:
                    return exec_state, order_not_filled_nb(OrderStatus.Rejected, StatusInfo.CantCoverFees)

            if lock_cash:
                # Size has upper limit
                if np.isinf(size) and not np.isnan(percent):
                    size_limit = min(percent * max_size_limit, max_size_limit)
                    percent = np.nan
                elif not np.isnan(percent):
                    size_limit = min(percent * size, max_size_limit)
                    percent = np.nan
                else:
                    size_limit = min(size, max_size_limit)
            else:  # np.isinf(size) and not np.isnan(percent)
                # Size has no upper limit
                size_limit = max_size_limit
        else:
            size_limit = size

    if not np.isnan(percent):
        # Apply percentage
        size_limit = percent * size_limit

    if size_limit > max_size:
        if not allow_partial:
            return exec_state, order_not_filled_nb(OrderStatus.Rejected, StatusInfo.MaxSizeExceeded)

        size_limit = max_size

    if direction == Direction.ShortOnly or direction == Direction.All:
        if np.isinf(size_limit):
            raise ValueError("Attempt to go in short direction infinitely")
    else:
        if exec_state.position == 0:
            return exec_state, order_not_filled_nb(OrderStatus.Rejected, StatusInfo.NoOpenPosition)

    if is_close_nb(size_limit, 0):
        return exec_state, order_not_filled_nb(OrderStatus.Ignored, StatusInfo.SizeZero)

    # Check against minimum size
    if is_less_nb(size_limit, min_size):
        return exec_state, order_not_filled_nb(OrderStatus.Rejected, StatusInfo.MinSizeNotReached)

    # Check against partial fill
    if np.isfinite(size) and is_less_nb(size_limit, size) and not allow_partial:  # np.inf doesn't count
        return exec_state, order_not_filled_nb(OrderStatus.Rejected, StatusInfo.PartialFill)

    # Get acquired cash
    acq_cash = size_limit * adj_price

    # Update fees
    fees_paid = acq_cash * fees + fixed_fees

    # Get final cash by subtracting costs
    final_cash = add_nb(acq_cash, -fees_paid)
    if final_cash < 0:
        return exec_state, order_not_filled_nb(OrderStatus.Rejected, StatusInfo.CantCoverFees)

    # Update current cash balance and position
    new_cash = exec_state.cash + final_cash
    new_position = add_nb(exec_state.position, -size_limit)

    # Update current debt and free cash
    if new_position < 0:
        if exec_state.position < 0:
            short_size = size_limit
        else:
            short_size = abs(new_position)
        short_value = short_size * adj_price
        new_debt = exec_state.debt + short_value
        free_cash_diff = add_nb(final_cash, -2 * short_value)
        new_free_cash = add_nb(exec_state.free_cash, free_cash_diff)
    else:
        new_debt = exec_state.debt
        new_free_cash = exec_state.free_cash + final_cash

    # Return filled order
    order_result = OrderResult(
        size_limit,
        adj_price,
        fees_paid,
        OrderSide.Sell,
        OrderStatus.Filled,
        -1
    )
    new_exec_state = ExecuteOrderState(
        cash=new_cash,
        position=new_position,
        debt=new_debt,
        free_cash=new_free_cash
    )
    return new_exec_state, order_result


@njit(cache=True)
def execute_order_nb(state: ProcessOrderState, order: Order) -> tp.Tuple[ExecuteOrderState, OrderResult]:
    """Execute an order given the current state.

    Args:
        state (ProcessOrderState): See `vectorbt.portfolio.enums.ProcessOrderState`.
        order (Order): See `vectorbt.portfolio.enums.Order`.

    Error is thrown if an input has value that is not expected.
    Order is ignored if its execution has no effect on current balance.
    Order is rejected if an input goes over a limit/restriction.
    """
    # numerical stability
    cash = state.cash
    if is_close_nb(cash, 0):
        cash = 0.
    position = state.position
    if is_close_nb(position, 0):
        position = 0.
    debt = state.debt
    if is_close_nb(debt, 0):
        debt = 0.
    free_cash = state.free_cash
    if is_close_nb(free_cash, 0):
        free_cash = 0.
    val_price = state.val_price
    if is_close_nb(val_price, 0):
        val_price = 0.
    value = state.value
    if is_close_nb(value, 0):
        value = 0.

    exec_state = ExecuteOrderState(
        cash=cash,
        position=position,
        debt=debt,
        free_cash=free_cash
    )

    if np.isnan(order.size):
        return exec_state, order_not_filled_nb(OrderStatus.Ignored, StatusInfo.SizeNaN)
    if np.isnan(order.price):
        return exec_state, order_not_filled_nb(OrderStatus.Ignored, StatusInfo.PriceNaN)

    # Check execution state
    if np.isnan(cash) or cash < 0:
        raise ValueError("cash cannot be NaN and must be greater than 0")
    if not np.isfinite(position):
        raise ValueError("position must be finite")
    if not np.isfinite(debt) or debt < 0:
        raise ValueError("debt must be finite and 0 or greater")
    if np.isnan(free_cash):
        raise ValueError("free_cash cannot be NaN")

    # Check order
    if order.size_type < 0 or order.size_type >= len(SizeType):
        raise ValueError("order.size_type is invalid")
    if order.direction < 0 or order.direction >= len(Direction):
        raise ValueError("order.direction is invalid")
    if order.direction == Direction.LongOnly and position < 0:
        raise ValueError("position is negative but order.direction is Direction.LongOnly")
    if order.direction == Direction.ShortOnly and position > 0:
        raise ValueError("position is positive but order.direction is Direction.ShortOnly")
    if not np.isfinite(order.price) or order.price < 0:
        raise ValueError("order.price must be finite and greater than or equal to 0")
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
        if np.isnan(value):
            return exec_state, order_not_filled_nb(OrderStatus.Ignored, StatusInfo.ValueNaN)
        if value <= 0:
            return exec_state, order_not_filled_nb(OrderStatus.Rejected, StatusInfo.ValueZeroNeg)

        order_size *= value
        order_size_type = SizeType.TargetValue

    if order_size_type == SizeType.Value or order_size_type == SizeType.TargetValue:
        # Target value
        if np.isinf(val_price) or val_price <= 0:
            raise ValueError("val_price_now must be finite and greater than 0")
        if np.isnan(val_price):
            return exec_state, order_not_filled_nb(OrderStatus.Ignored, StatusInfo.ValPriceNaN)

        order_size /= val_price
        if order_size_type == SizeType.Value:
            order_size_type = SizeType.Amount
        else:
            order_size_type = SizeType.TargetAmount

    if order_size_type == SizeType.TargetAmount:
        # Target amount
        order_size -= position
        order_size_type = SizeType.Amount

    if order_size_type == SizeType.Amount:
        if order.direction == Direction.ShortOnly or order.direction == Direction.All:
            if order_size < 0 and np.isinf(order_size):
                # Infinite negative size has a special meaning: 100% to short
                order_size = -1.
                order_size_type = SizeType.Percent

    percent = np.nan
    if order_size_type == SizeType.Percent:
        # Percentage of resources
        percent = abs(order_size)
        order_size = np.sign(order_size) * np.inf
        order_size_type = SizeType.Amount

    if order_size > 0:
        new_exec_state, order_result = buy_nb(
            exec_state,
            order_size,
            order.price,
            direction=order.direction,
            fees=order.fees,
            fixed_fees=order.fixed_fees,
            slippage=order.slippage,
            min_size=order.min_size,
            max_size=order.max_size,
            allow_partial=order.allow_partial,
            percent=percent
        )
    else:
        new_exec_state, order_result = sell_nb(
            exec_state,
            -order_size,
            order.price,
            direction=order.direction,
            fees=order.fees,
            fixed_fees=order.fixed_fees,
            slippage=order.slippage,
            min_size=order.min_size,
            max_size=order.max_size,
            lock_cash=order.lock_cash,
            allow_partial=order.allow_partial,
            percent=percent
        )

    if order.reject_prob > 0:
        if np.random.uniform(0, 1) < order.reject_prob:
            return exec_state, order_not_filled_nb(OrderStatus.Rejected, StatusInfo.RandomEvent)

    return new_exec_state, order_result


@njit(cache=True)
def fill_log_record_nb(record: tp.Record,
                       record_id: int,
                       i: int,
                       col: int,
                       group: int,
                       cash: float,
                       position: float,
                       debt: float,
                       free_cash: float,
                       val_price: float,
                       value: float,
                       order: Order,
                       new_cash: float,
                       new_position: float,
                       new_debt: float,
                       new_free_cash: float,
                       new_val_price: float,
                       new_value: float,
                       order_result: OrderResult,
                       order_id: int) -> None:
    """Fill a log record."""

    record['id'] = record_id
    record['idx'] = i
    record['col'] = col
    record['group'] = group
    record['cash'] = cash
    record['position'] = position
    record['debt'] = debt
    record['free_cash'] = free_cash
    record['val_price'] = val_price
    record['value'] = value
    record['size'] = order.size
    record['price'] = order.price
    record['size_type'] = order.size_type
    record['direction'] = order.direction
    record['fees'] = order.fees
    record['fixed_fees'] = order.fixed_fees
    record['slippage'] = order.slippage
    record['min_size'] = order.min_size
    record['max_size'] = order.max_size
    record['reject_prob'] = order.reject_prob
    record['lock_cash'] = order.lock_cash
    record['allow_partial'] = order.allow_partial
    record['raise_reject'] = order.raise_reject
    record['log'] = order.log
    record['new_cash'] = new_cash
    record['new_position'] = new_position
    record['new_debt'] = new_debt
    record['new_free_cash'] = new_free_cash
    record['new_val_price'] = new_val_price
    record['new_value'] = new_value
    record['res_size'] = order_result.size
    record['res_price'] = order_result.price
    record['res_fees'] = order_result.fees
    record['res_side'] = order_result.side
    record['res_status'] = order_result.status
    record['res_status_info'] = order_result.status_info
    record['order_id'] = order_id


@njit(cache=True)
def fill_order_record_nb(record: tp.Record,
                         record_id: int,
                         i: int,
                         col: int,
                         order_result: OrderResult) -> None:
    """Fill an order record."""

    record['id'] = record_id
    record['idx'] = i
    record['col'] = col
    record['size'] = order_result.size
    record['price'] = order_result.price
    record['fees'] = order_result.fees
    record['side'] = order_result.side


@njit(cache=True)
def raise_rejected_order_nb(order_result: OrderResult) -> None:
    """Raise an `vectorbt.portfolio.enums.RejectedOrderError`."""

    if order_result.status_info == StatusInfo.SizeNaN:
        raise RejectedOrderError("Size is NaN")
    if order_result.status_info == StatusInfo.PriceNaN:
        raise RejectedOrderError("Price is NaN")
    if order_result.status_info == StatusInfo.ValPriceNaN:
        raise RejectedOrderError("Asset valuation price is NaN")
    if order_result.status_info == StatusInfo.ValueNaN:
        raise RejectedOrderError("Asset/group value is NaN")
    if order_result.status_info == StatusInfo.ValueZeroNeg:
        raise RejectedOrderError("Asset/group value is zero or negative")
    if order_result.status_info == StatusInfo.SizeZero:
        raise RejectedOrderError("Size is zero")
    if order_result.status_info == StatusInfo.NoCashShort:
        raise RejectedOrderError("Not enough cash to short")
    if order_result.status_info == StatusInfo.NoCashLong:
        raise RejectedOrderError("Not enough cash to long")
    if order_result.status_info == StatusInfo.NoOpenPosition:
        raise RejectedOrderError("No open position to reduce/close")
    if order_result.status_info == StatusInfo.MaxSizeExceeded:
        raise RejectedOrderError("Size is greater than maximum allowed")
    if order_result.status_info == StatusInfo.RandomEvent:
        raise RejectedOrderError("Random event happened")
    if order_result.status_info == StatusInfo.CantCoverFees:
        raise RejectedOrderError("Not enough cash to cover fees")
    if order_result.status_info == StatusInfo.MinSizeNotReached:
        raise RejectedOrderError("Final size is less than minimum allowed")
    if order_result.status_info == StatusInfo.PartialFill:
        raise RejectedOrderError("Final size is less than requested")
    raise RejectedOrderError


@njit(cache=True)
def process_order_nb(i: int,
                     col: int,
                     group: int,
                     state: ProcessOrderState,
                     update_value: bool,
                     order: Order,
                     order_records: tp.RecordArray,
                     log_records: tp.RecordArray) -> tp.Tuple[OrderResult, ProcessOrderState]:
    """Process an order by executing it, saving relevant information to the logs, and returning a new state."""

    # Execute the order
    exec_state, order_result = execute_order_nb(state, order)

    # Raise if order rejected
    is_rejected = order_result.status == OrderStatus.Rejected
    if is_rejected and order.raise_reject:
        raise_rejected_order_nb(order_result)

    # Update valuation price and value
    is_filled = order_result.status == OrderStatus.Filled
    if is_filled and update_value:
        new_val_price, new_value = update_value_nb(
            state.cash,
            exec_state.cash,
            state.position,
            exec_state.position,
            state.val_price,
            order_result.price,
            state.value
        )
    else:
        new_val_price = state.val_price
        new_value = state.value

    new_oidx = state.oidx
    if is_filled:
        # Fill order record
        if state.oidx > len(order_records) - 1:
            raise IndexError("order_records index out of range. Set a higher max_orders.")
        fill_order_record_nb(
            order_records[state.oidx],
            state.oidx,
            i,
            col,
            order_result
        )
        new_oidx += 1

    new_lidx = state.lidx
    if order.log:
        # Fill log record
        if state.lidx > len(log_records) - 1:
            raise IndexError("log_records index out of range. Set a higher max_logs.")
        fill_log_record_nb(
            log_records[state.lidx],
            state.lidx,
            i,
            col,
            group,
            state.cash,
            state.position,
            state.debt,
            state.free_cash,
            state.val_price,
            state.value,
            order,
            exec_state.cash,
            exec_state.position,
            exec_state.debt,
            exec_state.free_cash,
            new_val_price,
            new_value,
            order_result,
            state.oidx if is_filled else -1
        )
        new_lidx += 1

    # Create new state
    new_state = ProcessOrderState(
        cash=exec_state.cash,
        position=exec_state.position,
        debt=exec_state.debt,
        free_cash=exec_state.free_cash,
        val_price=new_val_price,
        value=new_value,
        oidx=new_oidx,
        lidx=new_lidx
    )

    return order_result, new_state


@njit(cache=True)
def create_order_nb(size: float,
                    price: float,
                    size_type: int = SizeType.Amount,
                    direction: int = Direction.All,
                    fees: float = 0.,
                    fixed_fees: float = 0.,
                    slippage: float = 0.,
                    min_size: float = 0.,
                    max_size: float = np.inf,
                    reject_prob: float = 0.,
                    lock_cash: bool = False,
                    allow_partial: bool = True,
                    raise_reject: bool = False,
                    log: bool = False) -> Order:
    """Create an order with some defaults."""

    return Order(
        size=float(size),
        price=float(price),
        size_type=size_type,
        direction=direction,
        fees=float(fees),
        fixed_fees=float(fixed_fees),
        slippage=float(slippage),
        min_size=float(min_size),
        max_size=float(max_size),
        reject_prob=float(reject_prob),
        lock_cash=lock_cash,
        allow_partial=allow_partial,
        raise_reject=raise_reject,
        log=log
    )


@njit(cache=True)
def order_nothing() -> Order:
    """Convenience function to order nothing."""
    return NoOrder


@njit(cache=True)
def check_group_lens(group_lens: tp.Array1d, n_cols: int) -> None:
    """Check `group_lens`."""
    if np.sum(group_lens) != n_cols:
        raise ValueError("group_lens has incorrect total number of columns")


@njit(cache=True)
def check_group_init_cash(group_lens: tp.Array1d, n_cols: int, init_cash: tp.Array1d, cash_sharing: bool) -> None:
    """Check `init_cash`."""
    if cash_sharing:
        if len(init_cash) != len(group_lens):
            raise ValueError("If cash sharing is enabled, init_cash must match the number of groups")
    else:
        if len(init_cash) != n_cols:
            raise ValueError("If cash sharing is disabled, init_cash must match the number of columns")


@njit(cache=True)
def is_grouped_nb(group_lens: tp.Array1d) -> bool:
    """Check if columm,ns are grouped, that is, more than one column per group."""
    return np.any(group_lens > 1)


@njit(cache=True)
def shuffle_call_seq_nb(call_seq: tp.Array2d, group_lens: tp.Array1d) -> None:
    """Shuffle the call sequence array."""
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        for i in range(call_seq.shape[0]):
            np.random.shuffle(call_seq[i, from_col:to_col])
        from_col = to_col


@njit(cache=True)
def build_call_seq_nb(target_shape: tp.Shape,
                      group_lens: tp.Array1d,
                      call_seq_type: int = CallSeqType.Default) -> tp.Array2d:
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


def require_call_seq(call_seq: tp.Array2d) -> tp.Array2d:
    """Force the call sequence array to pass our requirements."""
    return np.require(call_seq, dtype=np.int_, requirements=['A', 'O', 'W', 'F'])


def build_call_seq(target_shape: tp.Shape,
                   group_lens: tp.Array1d,
                   call_seq_type: int = CallSeqType.Default) -> tp.Array2d:
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
def empty_prep_func_nb(context: tp.NamedTuple, *args) -> tp.Args:
    """Preparation function that forwards received arguments down the stack."""
    return args


@njit(cache=True)
def empty_after_order_func_nb(context: tp.NamedTuple, *args) -> None:
    """After order callback that does nothing."""
    return None


@njit(cache=True)
def get_group_value_nb(from_col: int,
                       to_col: int,
                       cash_now: float,
                       last_position: tp.Array1d,
                       last_val_price: tp.Array1d) -> float:
    """Get group value."""
    group_value = cash_now
    group_len = to_col - from_col
    for k in range(group_len):
        col = from_col + k
        if last_position[col] != 0:
            group_value += last_position[col] * last_val_price[col]
    return group_value


@njit(cache=True)
def get_group_value_ctx_nb(seg_ctx: SegmentContext) -> float:
    """Get group value from context.

    Accepts `vectorbt.portfolio.enums.SegmentContext`.

    Best called once from `segment_prep_func_nb`.
    To set the valuation price, change `last_val_price` of the context in-place.

    !!! note
        Cash sharing must be enabled."""
    if not seg_ctx.cash_sharing:
        raise ValueError("Cash sharing must be enabled")
    return get_group_value_nb(
        seg_ctx.from_col,
        seg_ctx.to_col,
        seg_ctx.last_cash[seg_ctx.group],
        seg_ctx.last_position,
        seg_ctx.last_val_price
    )


@njit(cache=True)
def approx_order_value_nb(size: float,
                          size_type: int,
                          cash_now: float,
                          position_now: float,
                          free_cash_now: float,
                          val_price_now: float,
                          value_now: float,
                          direction: int) -> float:
    """Approximate value of an order."""
    if direction == Direction.ShortOnly:
        size *= -1
    asset_value_now = position_now * val_price_now
    if size_type == SizeType.Amount:
        return size * val_price_now
    if size_type == SizeType.Value:
        return size
    if size_type == SizeType.Percent:
        if size >= 0:
            return size * cash_now
        else:
            if direction == Direction.LongOnly:
                return size * asset_value_now
            return size * (2 * max(asset_value_now, 0) + max(free_cash_now, 0))
    if size_type == SizeType.TargetAmount:
        return size * val_price_now - asset_value_now
    if size_type == SizeType.TargetValue:
        return size - asset_value_now
    if size_type == SizeType.TargetPercent:
        return size * value_now - asset_value_now
    return np.nan


@njit(cache=True)
def sort_call_seq_nb(seg_ctx: SegmentContext,
                     size: float,
                     size_type: int,
                     direction: int,
                     order_value_out: tp.Array1d) -> None:
    """Sort call sequence based on order value dynamically, for example, to rebalance.

    Accepts `vectorbt.portfolio.enums.SegmentContext`.

    `size`, `size_type` and `direction` utilize flexible broadcasting.

    Arrays `size`, `size_type`, `direction` and `order_value_out` should match the number
    of columns in the group. Array `order_value_out` should be empty and will contain
    sorted order values after execution.

    Best called once from `segment_prep_func_nb`.

    !!! note
        Cash sharing must be enabled and `call_seq_now` should follow `CallSeqType.Default`."""
    if not seg_ctx.cash_sharing:
        raise ValueError("Cash sharing must be enabled")
    size_arr = np.asarray(size)
    size_type_arr = np.asarray(size_type)
    direction_arr = np.asarray(direction)

    group_value_now = get_group_value_ctx_nb(seg_ctx)
    group_len = seg_ctx.to_col - seg_ctx.from_col
    for k in range(group_len):
        if seg_ctx.call_seq_now[k] != k:
            raise ValueError("call_seq_now should follow CallSeqType.Default")
        col = seg_ctx.from_col + k
        if seg_ctx.cash_sharing:
            cash_now = seg_ctx.last_cash[seg_ctx.group]
        else:
            cash_now = seg_ctx.last_cash[col]
        if seg_ctx.cash_sharing:
            free_cash_now = seg_ctx.last_free_cash[seg_ctx.group]
        else:
            free_cash_now = seg_ctx.last_free_cash[col]
        order_value_out[k] = approx_order_value_nb(
            flex_select_auto_nb(k, 0, size_arr, False),
            flex_select_auto_nb(k, 0, size_type_arr, False),
            cash_now,
            seg_ctx.last_position[col],
            free_cash_now,
            seg_ctx.last_val_price[col],
            group_value_now,
            flex_select_auto_nb(k, 0, direction_arr, False)
        )
    # Sort by order value
    insert_argsort_nb(order_value_out, seg_ctx.call_seq_now)


@njit(cache=True)
def try_order_nb(order_ctx: OrderContext, order: Order) -> tp.Tuple[ExecuteOrderState, OrderResult]:
    """Execute an order without side effects."""
    state = ProcessOrderState(
        cash=order_ctx.cash_now,
        position=order_ctx.position_now,
        debt=order_ctx.debt_now,
        free_cash=order_ctx.free_cash_now,
        val_price=order_ctx.val_price_now,
        value=order_ctx.value_now
    )
    return execute_order_nb(state, order)


@njit(cache=True)
def update_value_nb(cash_before: float,
                    cash_now: float,
                    position_before: float,
                    position_now: float,
                    val_price_before: float,
                    price: float,
                    value_before: float) -> tp.Tuple[float, float]:
    """Update valuation price and value."""
    val_price_now = price
    cash_flow = cash_now - cash_before
    if position_before != 0:
        asset_value_before = position_before * val_price_before
    else:
        asset_value_before = 0.
    if position_now != 0:
        asset_value_now = position_now * val_price_now
    else:
        asset_value_now = 0.
    asset_value_diff = asset_value_now - asset_value_before
    value_now = value_before + cash_flow + asset_value_diff
    return val_price_now, value_now


@njit(cache=True)
def init_records_nb(target_shape: tp.Shape,
                    max_orders: tp.Optional[int] = None,
                    max_logs: int = 0) -> tp.Tuple[tp.RecordArray, tp.RecordArray]:
    """Initialize order and log records."""
    if max_orders is None:
        max_orders = target_shape[0] * target_shape[1]
    order_records = np.empty(max_orders, dtype=order_dt)
    if max_logs == 0:
        max_logs = 1
    log_records = np.empty(max_logs, dtype=log_dt)
    return order_records, log_records


PrepFuncT = tp.Callable[[SimulationContext, tp.VarArg()], tp.Args]
GroupPrepFuncT = tp.Callable[[GroupContext, tp.VarArg()], tp.Args]
RowPrepFuncT = tp.Callable[[RowContext, tp.VarArg()], tp.Args]
SegmentPrepFuncT = tp.Callable[[SegmentContext, tp.VarArg()], tp.Args]
OrderFuncT = tp.Callable[[OrderContext, tp.VarArg()], Order]
AfterOrderFuncT = tp.Callable[[AfterOrderContext, OrderResult, tp.VarArg()], None]


@njit
def simulate_nb(target_shape: tp.Shape,
                close: tp.Array2d,
                group_lens: tp.Array1d,
                init_cash: tp.Array1d,
                cash_sharing: bool,
                call_seq: tp.Array2d,
                active_mask: tp.Array2d,
                prep_func_nb: PrepFuncT,
                prep_args: tp.Args,
                group_prep_func_nb: GroupPrepFuncT,
                group_prep_args: tp.Args,
                segment_prep_func_nb: SegmentPrepFuncT,
                segment_prep_args: tp.Args,
                order_func_nb: OrderFuncT,
                order_args: tp.Args,
                after_order_func_nb: AfterOrderFuncT,
                after_order_args: tp.Args,
                update_value: bool = False,
                max_orders: tp.Optional[int] = None,
                max_logs: int = 0) -> tp.Tuple[tp.RecordArray, tp.RecordArray]:
    """Simulate a portfolio by generating and filling orders.

    Starting with initial cash `init_cash`, iterates over each group and column over shape `target_shape`,
    and for each data point, generates an order using `order_func_nb`. Tries then to fulfill that
    order. Updates then the current cash balance and position if successful.

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

            If `cash_sharing` is True, should have shape `(group_lens.shape[0],)`.
            Otherwise, should have shape `(target_shape[1],)`.
        cash_sharing (bool): Whether to share cash within the same group.
        call_seq (array_like of int): Default sequence of calls per row and group.

            Should have shape `target_shape` and each value indicate the index of a column in a group.

            !!! note
                To use `sort_call_seq_nb`, should be generated using `CallSeqType.Default`.
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
                passed as `value_now` to `order_func_nb`. Internally used for converting
                `SizeType.Value` to `SizeType.Amount` and `SizeType.TargetPercent` and
                `SizeType.TargetValue` to `SizeType.TargetAmount`.
        segment_prep_args (tuple): Packed arguments passed to `segment_prep_func_nb`.
        order_func_nb (callable): Order generation function.

            Used for either generating an order or skipping. Should accept the current order context
            `vectorbt.portfolio.enums.OrderContext`, unpacked tuple from `segment_prep_func_nb`, and
            `*order_args`. Should either return `vectorbt.portfolio.enums.Order`, or
            `vectorbt.portfolio.enums.NoOrder` to do nothing.

            !!! note
                If the returned order has been rejected, there is no way of issuing a new order.
                You should make sure that the order passes, for example, by using `try_order_nb`.
        order_args (tuple): Arguments passed to `order_func_nb`.
        after_order_func_nb (callable): Callback that is called after the order has been processed.

            Used for checking the order status and doing some post-processing. Should accept the
            current after order context `vectorbt.portfolio.enums.AfterOrderContext`,
            unpacked tuple from `segment_prep_func_nb`, and `*after_order_args`. Should return nothing.
        after_order_args (tuple): Arguments passed to `after_order_func_nb`.
        update_value (bool): Whether to update group value after each filled order.

            Otherwise, stays the same for all columns in the group (the value is calculated
            only once, before executing any order).

            The change is marginal and mostly driven by transaction costs and slippage.
        max_orders (int): Size of the order records array.
        max_logs (int): Size of the log records array.

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

    ## Call hierarchy

    Like most things in the vectorbt universe, simulation is also done by iterating over a (imaginary) frame.
    This frame consists of two dimensions: time (rows) and assets/features (columns).
    Each element of this frame is a potential order, which gets generated by calling an order function.

    The question is: how do we move across this frame to simulate trading? There are two movement patterns:
    row-major (as done by `simulate_nb`) and column-major order (as done by `simulate_row_wise_nb`).
    In each of these patterns, we are always moving from top to bottom (time axis) and from left to right
    (asset/feature axis); the only difference between them is across which axis we are moving faster:
    do we want to process orders column-by-column (thus assuming that columns are independent) or row-by-row?
    Choosing between them is mostly a matter of preference, but it also makes different data being
    available when generating an order.

    The frame is further divided into "blocks": columns, groups, rows, segments, and elements.
    For example, columns can be grouped into groups that may or may not share the same capital.
    Regardless of capital sharing, each collection of elements within a group and time step is called
    a segment, which simply defines a single context (such as shared capital) for one or multiple orders.
    Each segment can also define a custom sequence (so-called call sequence) in which orders are executed.

    You can imagine each of these blocks as a rectangle drawn over different parts of the frame,
    and having its own context and preparation function. The preparation function is a simple callback
    is called before entering the block, and can be provided by the user to, for example, prepare arrays
    or do some custom calculations. It can also return a tuple that is then unpacked and passed as arguments
    to preparation functions coming next in the call hierarchy.

    Let's demonstrate a frame with one group of two columns and one group of one column, and the
    following call sequence:

    ```plaintext
    array([[0, 1, 0],
           [1, 0, 0]])
    ```

    ![](/vectorbt/docs/img/simulate_nb.gif)

    And here is the context information available at each step:

    ![](/vectorbt/docs/img/context_info.png)

    ## Example

    Create a group of three assets together sharing 100$ and simulate an equal-weighted portfolio
    that rebalances every second tick, all without leaving Numba:

    ```python-repl
    >>> import numpy as np
    >>> import pandas as pd
    >>> from numba import njit
    >>> from vectorbt.generic.plotting import Scatter
    >>> from vectorbt.records.nb import col_map_nb
    >>> from vectorbt.portfolio.enums import SizeType, Direction
    >>> from vectorbt.portfolio.nb import (
    ...     create_order_nb,
    ...     simulate_nb,
    ...     build_call_seq,
    ...     sort_call_seq_nb,
    ...     asset_flow_nb,
    ...     assets_nb,
    ...     asset_value_nb
    ... )

    >>> @njit
    ... def prep_func_nb(c):  # do nothing
    ...     print('preparing simulation')
    ...     return ()

    >>> @njit
    ... def group_prep_func_nb(c):
    ...     '''Define empty arrays for each group.'''
    ...     # Try to create new arrays as rarely as possible
    ...     print('\\tpreparing group', c.group)
    ...     order_value_out = np.empty(c.group_len, dtype=np.float_)
    ...     return (order_value_out,)

    >>> @njit
    ... def segment_prep_func_nb(c, order_value_out):
    ...     '''Perform rebalancing at each segment.'''
    ...     print('\\t\\tpreparing segment', c.i, '(row)')
    ...     for col in range(c.from_col, c.to_col):
    ...         # Here we use order price for group valuation
    ...         c.last_val_price[col] = close[c.i, col]
    ...     # Reorder call sequence such that selling orders come first and buying last
    ...     size = 1 / c.group_len
    ...     size_type = SizeType.TargetPercent
    ...     direction = Direction.LongOnly  # long positions only
    ...     sort_call_seq_nb(c, size, size_type, direction, order_value_out)
    ...     return (size, size_type, direction)

    >>> @njit
    ... def order_func_nb(c, size, size_type, direction, fees, fixed_fees, slippage):
    ...     '''Place an order.'''
    ...     print('\\t\\t\\tprocessing order', c.call_idx, 'at column', c.col)
    ...     return create_order_nb(
    ...         size,
    ...         close[c.i, c.col],
    ...         size_type=size_type,
    ...         direction=direction,
    ...         fees=fees,
    ...         fixed_fees=fixed_fees,
    ...         slippage=slippage
    ...     )

    >>> @njit
    ... def after_order_func_nb(c, size, size_type, direction):
    ...     '''Check order status.'''
    ...     print('\\t\\t\\t\\torder status:', c.order_result.status)

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
    ...     order_func_nb, (fees, fixed_fees, slippage),
    ...     after_order_func_nb, ()
    ... )
    preparing simulation
        preparing group 0
            preparing segment 0 (row)
                processing order 0 at column 0
                    order status: 0
                processing order 1 at column 1
                    order status: 0
                processing order 2 at column 2
                    order status: 0
            preparing segment 2 (row)
                processing order 0 at column 1
                    order status: 0
                processing order 1 at column 2
                    order status: 0
                processing order 2 at column 0
                    order status: 0
            preparing segment 4 (row)
                processing order 0 at column 0
                    order status: 0
                processing order 1 at column 2
                    order status: 0
                processing order 2 at column 1
                    order status: 0

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
    >>> asset_flow = asset_flow_nb(target_shape, order_records, col_map, Direction.All)
    >>> assets = assets_nb(asset_flow)
    >>> asset_value = asset_value_nb(close, assets)
    >>> Scatter(data=asset_value).fig
    ```

    ![](/vectorbt/docs/img/simulate_nb.svg)

    Note that the last order in a group with cash sharing is always disadvantaged
    as it has a bit less funds than the previous orders due to costs, which are not
    included when valuating the group.
    """
    check_group_lens(group_lens, target_shape[1])
    check_group_init_cash(group_lens, target_shape[1], init_cash, cash_sharing)

    order_records, log_records = init_records_nb(target_shape, max_orders, max_logs)
    init_cash = init_cash.astype(np.float_)
    last_cash = init_cash.copy()
    last_position = np.full(target_shape[1], 0., dtype=np.float_)
    last_debt = np.full(target_shape[1], 0., dtype=np.float_)
    last_free_cash = init_cash.copy()
    last_val_price = np.full(target_shape[1], np.nan, dtype=np.float_)
    last_value = init_cash.copy()
    last_oidx = np.full(target_shape[1], -1, dtype=np.int_)
    last_lidx = np.full(target_shape[1], -1, dtype=np.int_)
    oidx = 0
    lidx = 0

    # Run a function to prepare the simulation
    sim_ctx = SimulationContext(
        target_shape=target_shape,
        close=close,
        group_lens=group_lens,
        init_cash=init_cash,
        cash_sharing=cash_sharing,
        call_seq=call_seq,
        active_mask=active_mask,
        update_value=update_value,
        order_records=order_records,
        log_records=log_records,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        last_oidx=last_oidx,
        last_lidx=last_lidx
    )
    prep_out = prep_func_nb(sim_ctx, *prep_args)

    from_col = 0
    for group in range(len(group_lens)):
        # Is this group active?
        if np.any(active_mask[:, group]):
            to_col = from_col + group_lens[group]
            group_len = to_col - from_col

            # Run a function to preprocess this entire group
            group_ctx = GroupContext(
                target_shape=target_shape,
                close=close,
                group_lens=group_lens,
                init_cash=init_cash,
                cash_sharing=cash_sharing,
                call_seq=call_seq,
                active_mask=active_mask,
                update_value=update_value,
                order_records=order_records,
                log_records=log_records,
                last_cash=last_cash,
                last_position=last_position,
                last_debt=last_debt,
                last_free_cash=last_free_cash,
                last_val_price=last_val_price,
                last_value=last_value,
                last_oidx=last_oidx,
                last_lidx=last_lidx,
                group=group,
                group_len=group_len,
                from_col=from_col,
                to_col=to_col
            )
            group_prep_out = group_prep_func_nb(group_ctx, *prep_out, *group_prep_args)

            for i in range(target_shape[0]):
                # Is this row segment active?
                if active_mask[i, group]:
                    # Update valuation price
                    if i > 0:
                        for col in range(from_col, to_col):
                            last_val_price[col] = close[i - 1, col]

                    # Run a function to preprocess this group within this row
                    call_seq_now = call_seq[i, from_col:to_col]
                    seg_ctx = SegmentContext(
                        target_shape=target_shape,
                        close=close,
                        group_lens=group_lens,
                        init_cash=init_cash,
                        cash_sharing=cash_sharing,
                        call_seq=call_seq,
                        active_mask=active_mask,
                        update_value=update_value,
                        order_records=order_records,
                        log_records=log_records,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        last_oidx=last_oidx,
                        last_lidx=last_lidx,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=call_seq_now
                    )
                    segment_prep_out = segment_prep_func_nb(seg_ctx, *group_prep_out, *segment_prep_args)

                    # Get current values per group
                    if cash_sharing:
                        cash_now = last_cash[group]
                        value_now = get_group_value_nb(
                            from_col,
                            to_col,
                            cash_now,
                            last_position,
                            last_val_price
                        )
                        last_value[group] = value_now
                        free_cash_now = last_free_cash[group]

                    for k in range(group_len):
                        col_i = call_seq_now[k]
                        if col_i >= group_len:
                            raise ValueError("Call index exceeds bounds of the group")
                        col = from_col + col_i

                        # Get current values per column
                        position_now = last_position[col]
                        debt_now = last_debt[col]
                        val_price_now = last_val_price[col]
                        if not cash_sharing:
                            cash_now = last_cash[col]
                            free_cash_now = last_free_cash[col]
                            value_now = cash_now
                            if position_now != 0:
                                value_now += position_now * val_price_now
                            last_value[col] = value_now

                        # Generate the next order
                        order_ctx = OrderContext(
                            target_shape=target_shape,
                            close=close,
                            group_lens=group_lens,
                            init_cash=init_cash,
                            cash_sharing=cash_sharing,
                            call_seq=call_seq,
                            active_mask=active_mask,
                            update_value=update_value,
                            order_records=order_records,
                            log_records=log_records,
                            last_cash=last_cash,
                            last_position=last_position,
                            last_debt=last_debt,
                            last_free_cash=last_free_cash,
                            last_val_price=last_val_price,
                            last_value=last_value,
                            last_oidx=last_oidx,
                            last_lidx=last_lidx,
                            group=group,
                            group_len=group_len,
                            from_col=from_col,
                            to_col=to_col,
                            i=i,
                            call_seq_now=call_seq_now,
                            col=col,
                            call_idx=k,
                            cash_now=cash_now,
                            position_now=position_now,
                            debt_now=debt_now,
                            free_cash_now=free_cash_now,
                            val_price_now=val_price_now,
                            value_now=value_now
                        )
                        order = order_func_nb(order_ctx, *segment_prep_out, *order_args)

                        # Process the order
                        state = ProcessOrderState(
                            cash=cash_now,
                            position=position_now,
                            debt=debt_now,
                            free_cash=free_cash_now,
                            val_price=val_price_now,
                            value=value_now,
                            oidx=oidx,
                            lidx=lidx
                        )

                        order_result, new_state = process_order_nb(
                            i, col, group,
                            state,
                            update_value,
                            order,
                            order_records,
                            log_records
                        )

                        # Update state
                        cash_now = new_state.cash
                        position_now = new_state.position
                        debt_now = new_state.debt
                        free_cash_now = new_state.free_cash
                        val_price_now = new_state.val_price
                        value_now = new_state.value
                        oidx = new_state.oidx
                        lidx = new_state.lidx

                        if cash_sharing:
                            last_cash[group] = cash_now
                        else:
                            last_cash[col] = cash_now
                        last_position[col] = position_now
                        last_debt[col] = debt_now
                        if cash_sharing:
                            last_free_cash[group] = free_cash_now
                        else:
                            last_free_cash[col] = free_cash_now
                        last_val_price[col] = val_price_now
                        if cash_sharing:
                            last_value[group] = value_now
                        else:
                            last_value[col] = value_now
                        if state.oidx != new_state.oidx:
                            last_oidx[col] = state.oidx
                        if state.lidx != new_state.lidx:
                            last_lidx[col] = state.lidx

                        # After order callback
                        after_order_ctx = AfterOrderContext(
                            target_shape=target_shape,
                            close=close,
                            group_lens=group_lens,
                            init_cash=init_cash,
                            cash_sharing=cash_sharing,
                            call_seq=call_seq,
                            active_mask=active_mask,
                            update_value=update_value,
                            order_records=order_records,
                            log_records=log_records,
                            last_cash=last_cash,
                            last_position=last_position,
                            last_debt=last_debt,
                            last_free_cash=last_free_cash,
                            last_val_price=last_val_price,
                            last_value=last_value,
                            last_oidx=last_oidx,
                            last_lidx=last_lidx,
                            group=group,
                            group_len=group_len,
                            from_col=from_col,
                            to_col=to_col,
                            i=i,
                            call_seq_now=call_seq_now,
                            col=col,
                            call_idx=k,
                            cash_before=state.cash,
                            position_before=state.position,
                            debt_before=state.debt,
                            free_cash_before=state.free_cash,
                            val_price_before=state.val_price,
                            value_before=state.value,
                            order_result=order_result,
                            cash_now=cash_now,
                            position_now=position_now,
                            debt_now=debt_now,
                            free_cash_now=free_cash_now,
                            val_price_now=val_price_now,
                            value_now=value_now
                        )
                        after_order_func_nb(after_order_ctx, *segment_prep_out, *after_order_args)

            from_col = to_col

    return order_records[:oidx], log_records[:lidx]


@njit
def simulate_row_wise_nb(target_shape: tp.Shape,
                         close: tp.Array2d,
                         group_lens: tp.Array1d,
                         init_cash: tp.Array1d,
                         cash_sharing: bool,
                         call_seq: tp.Array2d,
                         active_mask: tp.Array2d,
                         prep_func_nb: PrepFuncT,
                         prep_args: tp.Args,
                         row_prep_func_nb: RowPrepFuncT,
                         row_prep_args: tp.Args,
                         segment_prep_func_nb: SegmentPrepFuncT,
                         segment_prep_args: tp.Args,
                         order_func_nb: OrderFuncT,
                         order_args: tp.Args,
                         after_order_func_nb: AfterOrderFuncT,
                         after_order_args: tp.Args,
                         update_value: bool = False,
                         max_orders: tp.Optional[int] = None,
                         max_logs: int = 0) -> tp.Tuple[tp.RecordArray, tp.RecordArray]:
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

    ## Call hierarchy

    Let's illustrate the same example as in `simulate_nb` but adapted for this function:

    ![](/vectorbt/docs/img/simulate_row_wise_nb.gif)

    ## Example

    Running the same example as in `simulate_nb` but adapted for this function:
    ```python-repl
    preparing simulation
        preparing row 0
            preparing segment 0 (group)
                processing order 0 at column 0
                    order status: 0
                processing order 1 at column 1
                    order status: 0
                processing order 2 at column 2
                    order status: 0
        preparing row 2
            preparing segment 2 (group)
                processing order 0 at column 1
                    order status: 0
                processing order 1 at column 2
                    order status: 0
                processing order 2 at column 0
                    order status: 0
        preparing row 4
            preparing segment 4 (group)
                processing order 0 at column 0
                    order status: 0
                processing order 1 at column 2
                    order status: 0
                processing order 2 at column 1
                    order status: 0
    ```

    Note, however, that we cannot create NumPy arrays per group anymore as there is no
    `group_prep_func_nb`, so you would need to move this part to `prep_func_nb`,
    make arrays wider, and use only the part of the array that points to the current group.
    """
    check_group_lens(group_lens, target_shape[1])
    check_group_init_cash(group_lens, target_shape[1], init_cash, cash_sharing)

    order_records, log_records = init_records_nb(target_shape, max_orders, max_logs)
    init_cash = init_cash.astype(np.float_)
    last_cash = init_cash.copy()
    last_position = np.full(target_shape[1], 0., dtype=np.float_)
    last_debt = np.full(target_shape[1], 0., dtype=np.float_)
    last_free_cash = init_cash.copy()
    last_val_price = np.full(target_shape[1], np.nan, dtype=np.float_)
    last_value = init_cash.copy()
    last_oidx = np.full(target_shape[1], -1, dtype=np.int_)
    last_lidx = np.full(target_shape[1], -1, dtype=np.int_)
    oidx = 0
    lidx = 0

    # Run a function to prepare the simulation
    sim_ctx = SimulationContext(
        target_shape=target_shape,
        close=close,
        group_lens=group_lens,
        init_cash=init_cash,
        cash_sharing=cash_sharing,
        call_seq=call_seq,
        active_mask=active_mask,
        update_value=update_value,
        order_records=order_records,
        log_records=log_records,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        last_oidx=last_oidx,
        last_lidx=last_lidx
    )
    prep_out = prep_func_nb(sim_ctx, *prep_args)

    for i in range(target_shape[0]):
        # Is this row active?
        if np.any(active_mask[i, :]):
            # Update valuation price
            if i > 0:
                for col in range(target_shape[1]):
                    last_val_price[col] = close[i - 1, col]

            # Run a function to preprocess this entire row
            row_ctx = RowContext(
                target_shape=target_shape,
                close=close,
                group_lens=group_lens,
                init_cash=init_cash,
                cash_sharing=cash_sharing,
                call_seq=call_seq,
                active_mask=active_mask,
                update_value=update_value,
                order_records=order_records,
                log_records=log_records,
                last_cash=last_cash,
                last_position=last_position,
                last_debt=last_debt,
                last_free_cash=last_free_cash,
                last_val_price=last_val_price,
                last_value=last_value,
                last_oidx=last_oidx,
                last_lidx=last_lidx,
                i=i
            )
            row_prep_out = row_prep_func_nb(row_ctx, *prep_out, *row_prep_args)

            from_col = 0
            for group in range(len(group_lens)):
                # Is this group segment active?
                if active_mask[i, group]:
                    to_col = from_col + group_lens[group]
                    group_len = to_col - from_col

                    # Run a function to preprocess this row within this group
                    call_seq_now = call_seq[i, from_col:to_col]
                    seg_ctx = SegmentContext(
                        target_shape=target_shape,
                        close=close,
                        group_lens=group_lens,
                        init_cash=init_cash,
                        cash_sharing=cash_sharing,
                        call_seq=call_seq,
                        active_mask=active_mask,
                        update_value=update_value,
                        order_records=order_records,
                        log_records=log_records,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        last_oidx=last_oidx,
                        last_lidx=last_lidx,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=call_seq_now
                    )
                    segment_prep_out = segment_prep_func_nb(seg_ctx, *row_prep_out, *segment_prep_args)

                    # Get current values per group
                    if cash_sharing:
                        cash_now = last_cash[group]
                        value_now = get_group_value_nb(from_col, to_col, cash_now, last_position, last_val_price)
                        last_value[group] = value_now
                        free_cash_now = last_free_cash[group]

                    for k in range(group_len):
                        col_i = call_seq_now[k]
                        if col_i >= group_len:
                            raise ValueError("Call index exceeds bounds of the group")
                        col = from_col + col_i

                        # Get current values per column
                        position_now = last_position[col]
                        debt_now = last_debt[col]
                        val_price_now = last_val_price[col]
                        if not cash_sharing:
                            cash_now = last_cash[col]
                            free_cash_now = last_free_cash[col]
                            value_now = cash_now
                            if position_now != 0:
                                value_now += position_now * val_price_now
                            last_value[col] = value_now

                        # Generate the next order
                        order_ctx = OrderContext(
                            target_shape=target_shape,
                            close=close,
                            group_lens=group_lens,
                            init_cash=init_cash,
                            cash_sharing=cash_sharing,
                            call_seq=call_seq,
                            active_mask=active_mask,
                            update_value=update_value,
                            order_records=order_records,
                            log_records=log_records,
                            last_cash=last_cash,
                            last_position=last_position,
                            last_debt=last_debt,
                            last_free_cash=last_free_cash,
                            last_val_price=last_val_price,
                            last_value=last_value,
                            last_oidx=last_oidx,
                            last_lidx=last_lidx,
                            group=group,
                            group_len=group_len,
                            from_col=from_col,
                            to_col=to_col,
                            i=i,
                            call_seq_now=call_seq_now,
                            col=col,
                            call_idx=k,
                            cash_now=cash_now,
                            position_now=position_now,
                            debt_now=debt_now,
                            free_cash_now=free_cash_now,
                            val_price_now=val_price_now,
                            value_now=value_now
                        )
                        order = order_func_nb(order_ctx, *segment_prep_out, *order_args)

                        # Process the order
                        state = ProcessOrderState(
                            cash=cash_now,
                            position=position_now,
                            debt=debt_now,
                            free_cash=free_cash_now,
                            val_price=val_price_now,
                            value=value_now,
                            oidx=oidx,
                            lidx=lidx
                        )

                        order_result, new_state = process_order_nb(
                            i, col, group,
                            state,
                            update_value,
                            order,
                            order_records,
                            log_records
                        )

                        # Update state
                        cash_now = new_state.cash
                        position_now = new_state.position
                        debt_now = new_state.debt
                        free_cash_now = new_state.free_cash
                        val_price_now = new_state.val_price
                        value_now = new_state.value
                        oidx = new_state.oidx
                        lidx = new_state.lidx

                        if cash_sharing:
                            last_cash[group] = cash_now
                        else:
                            last_cash[col] = cash_now
                        last_position[col] = position_now
                        last_debt[col] = debt_now
                        if cash_sharing:
                            last_free_cash[group] = free_cash_now
                        else:
                            last_free_cash[col] = free_cash_now
                        last_val_price[col] = val_price_now
                        if cash_sharing:
                            last_value[group] = value_now
                        else:
                            last_value[col] = value_now
                        if state.oidx != new_state.oidx:
                            last_oidx[col] = state.oidx
                        if state.lidx != new_state.lidx:
                            last_lidx[col] = state.lidx

                        # After order callback
                        after_order_ctx = AfterOrderContext(
                            target_shape=target_shape,
                            close=close,
                            group_lens=group_lens,
                            init_cash=init_cash,
                            cash_sharing=cash_sharing,
                            call_seq=call_seq,
                            active_mask=active_mask,
                            update_value=update_value,
                            order_records=order_records,
                            log_records=log_records,
                            last_cash=last_cash,
                            last_position=last_position,
                            last_debt=last_debt,
                            last_free_cash=last_free_cash,
                            last_val_price=last_val_price,
                            last_value=last_value,
                            last_oidx=last_oidx,
                            last_lidx=last_lidx,
                            group=group,
                            group_len=group_len,
                            from_col=from_col,
                            to_col=to_col,
                            i=i,
                            call_seq_now=call_seq_now,
                            col=col,
                            call_idx=k,
                            cash_before=state.cash,
                            position_before=state.position,
                            debt_before=state.debt,
                            free_cash_before=state.free_cash,
                            val_price_before=state.val_price,
                            value_before=state.value,
                            order_result=order_result,
                            cash_now=cash_now,
                            position_now=position_now,
                            debt_now=debt_now,
                            free_cash_now=free_cash_now,
                            val_price_now=val_price_now,
                            value_now=value_now
                        )
                        after_order_func_nb(after_order_ctx, *segment_prep_out, *after_order_args)

                    from_col = to_col

    return order_records[:oidx], log_records[:lidx]


@njit(cache=True)
def simulate_from_orders_nb(target_shape: tp.Shape,
                            group_lens: tp.Array1d,
                            init_cash: tp.Array1d,
                            call_seq: tp.Array2d,
                            size: tp.ArrayLike,
                            price: tp.ArrayLike,
                            size_type: tp.ArrayLike = SizeType.Amount,
                            direction: tp.ArrayLike = Direction.All,
                            fees: tp.ArrayLike = 0.,
                            fixed_fees: tp.ArrayLike = 0.,
                            slippage: tp.ArrayLike = 0.,
                            min_size: tp.ArrayLike = 0.,
                            max_size: tp.ArrayLike = np.inf,
                            reject_prob: tp.ArrayLike = 0.,
                            lock_cash: tp.ArrayLike = False,
                            allow_partial: tp.ArrayLike = True,
                            raise_reject: tp.ArrayLike = False,
                            log: tp.ArrayLike = False,
                            val_price: tp.ArrayLike = np.nan,
                            auto_call_seq: bool = False,
                            update_value: bool = False,
                            max_orders: tp.Optional[int] = None,
                            max_logs: int = 0,
                            flex_2d: bool = True) -> tp.Tuple[tp.RecordArray, tp.RecordArray]:
    """Adaptation of `simulate_nb` for simulation based on orders.

    Utilizes flexible broadcasting.

    !!! note
        Should be only grouped if cash sharing is enabled.

        If `auto_call_seq` is True, make sure that `call_seq` follows `CallSeqType.Default`."""
    check_group_lens(group_lens, target_shape[1])
    cash_sharing = is_grouped_nb(group_lens)
    check_group_init_cash(group_lens, target_shape[1], init_cash, cash_sharing)

    order_records, log_records = init_records_nb(target_shape, max_orders, max_logs)
    init_cash = init_cash.astype(np.float_)
    last_position = np.full(target_shape[1], 0., dtype=np.float_)
    last_debt = np.full(target_shape[1], 0., dtype=np.float_)
    temp_order_value = np.empty(target_shape[1], dtype=np.float_)
    oidx = 0
    lidx = 0

    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_len = to_col - from_col
        cash_now = init_cash[group]
        free_cash_now = init_cash[group]

        for i in range(target_shape[0]):
            # Calculate group value and rearrange if cash sharing is enabled
            if cash_sharing:
                # Same as get_group_value_ctx_nb but with flexible indexing
                value_now = cash_now
                for k in range(group_len):
                    col = from_col + k
                    if last_position[col] != 0:
                        _val_price = flex_select_auto_nb(i, col, val_price, flex_2d)
                        value_now += last_position[col] * _val_price

                # Dynamically sort by order value -> selling comes first to release funds early
                if auto_call_seq:
                    # Same as sort_by_order_value_ctx_nb but with flexible indexing
                    for k in range(group_len):
                        col = from_col + k
                        temp_order_value[k] = approx_order_value_nb(
                            flex_select_auto_nb(i, col, size, flex_2d),
                            flex_select_auto_nb(i, col, size_type, flex_2d),
                            cash_now,
                            last_position[col],
                            free_cash_now,
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

                # Get current values per column
                position_now = last_position[col]
                debt_now = last_debt[col]
                val_price_now = flex_select_auto_nb(i, col, val_price, flex_2d)
                if not cash_sharing:
                    value_now = cash_now
                    if position_now != 0:
                        value_now += position_now * val_price_now

                # Generate the next order
                order = create_order_nb(
                    flex_select_auto_nb(i, col, size, flex_2d),
                    flex_select_auto_nb(i, col, price, flex_2d),
                    size_type=flex_select_auto_nb(i, col, size_type, flex_2d),
                    direction=flex_select_auto_nb(i, col, direction, flex_2d),
                    fees=flex_select_auto_nb(i, col, fees, flex_2d),
                    fixed_fees=flex_select_auto_nb(i, col, fixed_fees, flex_2d),
                    slippage=flex_select_auto_nb(i, col, slippage, flex_2d),
                    min_size=flex_select_auto_nb(i, col, min_size, flex_2d),
                    max_size=flex_select_auto_nb(i, col, max_size, flex_2d),
                    reject_prob=flex_select_auto_nb(i, col, reject_prob, flex_2d),
                    lock_cash=flex_select_auto_nb(i, col, lock_cash, flex_2d),
                    allow_partial=flex_select_auto_nb(i, col, allow_partial, flex_2d),
                    raise_reject=flex_select_auto_nb(i, col, raise_reject, flex_2d),
                    log=flex_select_auto_nb(i, col, log, flex_2d)
                )

                # Process the order
                state = ProcessOrderState(
                    cash=cash_now,
                    position=position_now,
                    debt=debt_now,
                    free_cash=free_cash_now,
                    val_price=val_price_now,
                    value=value_now,
                    oidx=oidx,
                    lidx=lidx
                )

                order_result, new_state = process_order_nb(
                    i, col, group,
                    state,
                    update_value,
                    order,
                    order_records,
                    log_records
                )

                # Update state
                cash_now = new_state.cash
                position_now = new_state.position
                debt_now = new_state.debt
                free_cash_now = new_state.free_cash
                val_price_now = new_state.val_price
                value_now = new_state.value
                oidx = new_state.oidx
                lidx = new_state.lidx

                # Now becomes last
                last_position[col] = position_now
                last_debt[col] = debt_now

        from_col = to_col

    return order_records[:oidx], log_records[:lidx]


@njit(cache=True)
def signal_to_size_nb(position_now: float,
                      is_entry: bool,
                      is_exit: bool,
                      size: float,
                      size_type: int,
                      direction: int,
                      accumulate: bool,
                      conflict_mode: int,
                      close_first: bool,
                      val_price_now: float) -> tp.Tuple[bool, bool, float, int]:
    """Get order size given signals."""
    if size_type != SizeType.Amount and size_type != SizeType.Value and size_type != SizeType.Percent:
        raise ValueError("Only SizeType.Amount, SizeType.Value, and SizeType.Percent are supported")
    order_size = 0.
    abs_position_now = abs(position_now)
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
            # Take the signal opposite to the position we are in
            if position_now == 0:
                # Cannot decide -> ignore
                is_entry = False
                is_exit = False
            else:
                if direction == Direction.All:
                    if position_now > 0:
                        is_entry = False
                    elif position_now < 0:
                        is_exit = False
                else:
                    is_entry = False
        else:
            is_entry = False
            is_exit = False

    if is_entry:
        if direction == Direction.All:
            # Behaves like Direction.LongOnly
            if accumulate:
                order_size = abs_size
            else:
                if position_now < 0:
                    # Reverse short position
                    if close_first:
                        order_size = abs_position_now
                        size_type = SizeType.Amount
                    else:
                        if size_type == SizeType.Percent:
                            raise ValueError("SizeType.Percent does not support position reversal under Direction.All")
                        if size_type == SizeType.Value:
                            order_size = abs_position_now + abs_size / val_price_now
                        else:
                            order_size = abs_position_now + abs_size
                elif position_now == 0:
                    # Open long position
                    order_size = abs_size
        elif direction == Direction.LongOnly:
            if position_now == 0 or accumulate:
                # Open or increase long position
                order_size = abs_size
        else:
            if position_now == 0 or accumulate:
                # Open or increase short position
                order_size = -abs_size

    elif is_exit:
        if direction == Direction.All:
            # Behaves like Direction.ShortOnly
            if accumulate:
                order_size = -abs_size
            else:
                if position_now > 0:
                    # Reverse long position
                    if close_first:
                        order_size = -abs_position_now
                        size_type = SizeType.Amount
                    else:
                        if size_type == SizeType.Percent:
                            raise ValueError("SizeType.Percent does not support position reversal under Direction.All")
                        if size_type == SizeType.Value:
                            order_size = -abs_position_now - abs_size / val_price_now
                        else:
                            order_size = -abs_position_now - abs_size
                elif position_now == 0:
                    # Open short position
                    order_size = -abs_size
        elif direction == Direction.ShortOnly:
            if position_now < 0:
                if accumulate:
                    # Reduce short position
                    order_size = abs_size
                else:
                    # Close short position
                    order_size = abs_position_now
                    size_type = SizeType.Amount
        else:
            if position_now > 0:
                if accumulate:
                    # Reduce long position
                    order_size = -abs_size
                else:
                    # Close long position
                    order_size = -abs_position_now
                    size_type = SizeType.Amount

    return is_entry, is_exit, order_size, size_type


@njit(cache=True)
def simulate_from_signals_nb(target_shape: tp.Shape,
                             group_lens: tp.Array1d,
                             init_cash: tp.Array1d,
                             call_seq: tp.Array2d,
                             entries: tp.ArrayLike,
                             exits: tp.ArrayLike,
                             size: tp.ArrayLike,
                             price: tp.ArrayLike,
                             size_type: tp.ArrayLike = SizeType.Amount,
                             direction: tp.ArrayLike = Direction.LongOnly,
                             fees: tp.ArrayLike = 0.,
                             fixed_fees: tp.ArrayLike = 0.,
                             slippage: tp.ArrayLike = 0.,
                             min_size: tp.ArrayLike = 0.,
                             max_size: tp.ArrayLike = np.inf,
                             reject_prob: tp.ArrayLike = 0.,
                             lock_cash: tp.ArrayLike = False,
                             allow_partial: tp.ArrayLike = True,
                             raise_reject: tp.ArrayLike = False,
                             log: tp.ArrayLike = False,
                             accumulate: tp.ArrayLike = False,
                             conflict_mode: tp.ArrayLike = ConflictMode.Ignore,
                             close_first: tp.ArrayLike = False,
                             val_price: tp.ArrayLike = np.nan,
                             auto_call_seq: bool = False,
                             update_value: bool = False,
                             max_orders: tp.Optional[int] = None,
                             max_logs: int = 0,
                             flex_2d: bool = True) -> tp.Tuple[tp.RecordArray, tp.RecordArray]:
    """Adaptation of `simulate_nb` for simulation based on entry and exit signals.

    Utilizes flexible broadcasting.

    !!! note
        Should be only grouped if cash sharing is enabled."""
    check_group_lens(group_lens, target_shape[1])
    cash_sharing = is_grouped_nb(group_lens)
    check_group_init_cash(group_lens, target_shape[1], init_cash, cash_sharing)

    order_records, log_records = init_records_nb(target_shape, max_orders, max_logs)
    init_cash = init_cash.astype(np.float_)
    last_position = np.full(target_shape[1], 0., dtype=np.float_)
    last_debt = np.full(target_shape[1], 0., dtype=np.float_)
    order_size = np.empty(target_shape[1], dtype=np.float_)
    order_size_type = np.empty(target_shape[1], dtype=np.float_)
    temp_order_value = np.empty(target_shape[1], dtype=np.float_)
    oidx = 0
    lidx = 0

    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_len = to_col - from_col
        cash_now = init_cash[group]
        free_cash_now = init_cash[group]

        for i in range(target_shape[0]):

            # Get size and value of each order
            for k in range(group_len):
                col = from_col + k  # order doesn't matter
                _is_entry, _is_exit, _order_size, _order_size_type = signal_to_size_nb(
                    last_position[col],
                    flex_select_auto_nb(i, col, entries, flex_2d),
                    flex_select_auto_nb(i, col, exits, flex_2d),
                    flex_select_auto_nb(i, col, size, flex_2d),
                    flex_select_auto_nb(i, col, size_type, flex_2d),
                    flex_select_auto_nb(i, col, direction, flex_2d),
                    flex_select_auto_nb(i, col, accumulate, flex_2d),
                    flex_select_auto_nb(i, col, conflict_mode, flex_2d),
                    flex_select_auto_nb(i, col, close_first, flex_2d),
                    flex_select_auto_nb(i, col, val_price, flex_2d)
                )  # already takes into account direction
                order_size[col] = _order_size
                order_size_type[col] = _order_size_type

                if cash_sharing:
                    if _order_size == 0:
                        temp_order_value[k] = 0.
                    else:
                        # Approximate order value
                        _val_price = flex_select_auto_nb(i, col, val_price, flex_2d)
                        _direction = flex_select_auto_nb(i, col, direction, flex_2d)
                        if _order_size_type == SizeType.Amount:
                            temp_order_value[k] = _order_size * _val_price
                        elif _order_size_type == SizeType.Value:
                            temp_order_value[k] = _order_size
                        else:  # SizeType.Percent
                            if _order_size >= 0:
                                temp_order_value[k] = _order_size * cash_now
                            else:
                                asset_value_now = last_position[col] * _val_price
                                if _direction == Direction.LongOnly:
                                    temp_order_value[k] = _order_size * asset_value_now
                                else:
                                    max_exposure = (2 * max(asset_value_now, 0) + max(free_cash_now, 0))
                                    temp_order_value[k] = _order_size * max_exposure

            if cash_sharing:
                # Dynamically sort by order value -> selling comes first to release funds early
                if auto_call_seq:
                    insert_argsort_nb(temp_order_value[:group_len], call_seq[i, from_col:to_col])

                # Same as get_group_value_ctx_nb but with flexible indexing
                value_now = cash_now
                for k in range(group_len):
                    col = from_col + k
                    if last_position[col] != 0:
                        _val_price = flex_select_auto_nb(i, col, val_price, flex_2d)
                        value_now += last_position[col] * _val_price

            for k in range(group_len):
                col = from_col + k
                if cash_sharing:
                    col_i = call_seq[i, col]
                    if col_i >= group_len:
                        raise ValueError("Call index exceeds bounds of the group")
                    col = from_col + col_i

                # Get current values per column
                position_now = last_position[col]
                debt_now = last_debt[col]
                val_price_now = flex_select_auto_nb(i, col, val_price, flex_2d)
                if not cash_sharing:
                    value_now = cash_now
                    if position_now != 0:
                        value_now += position_now * val_price_now

                # Generate the next order
                _order_size = order_size[col]  # already takes into account direction
                _order_size_type = order_size_type[col]
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
                        _order_size,
                        flex_select_auto_nb(i, col, price, flex_2d),
                        size_type=_order_size_type,
                        direction=_direction,
                        fees=flex_select_auto_nb(i, col, fees, flex_2d),
                        fixed_fees=flex_select_auto_nb(i, col, fixed_fees, flex_2d),
                        slippage=flex_select_auto_nb(i, col, slippage, flex_2d),
                        min_size=flex_select_auto_nb(i, col, min_size, flex_2d),
                        max_size=flex_select_auto_nb(i, col, max_size, flex_2d),
                        reject_prob=flex_select_auto_nb(i, col, reject_prob, flex_2d),
                        lock_cash=flex_select_auto_nb(i, col, lock_cash, flex_2d),
                        allow_partial=flex_select_auto_nb(i, col, allow_partial, flex_2d),
                        raise_reject=flex_select_auto_nb(i, col, raise_reject, flex_2d),
                        log=flex_select_auto_nb(i, col, log, flex_2d)
                    )

                    # Process the order
                    state = ProcessOrderState(
                        cash=cash_now,
                        position=position_now,
                        debt=debt_now,
                        free_cash=free_cash_now,
                        val_price=val_price_now,
                        value=value_now,
                        oidx=oidx,
                        lidx=lidx
                    )

                    order_result, new_state = process_order_nb(
                        i, col, group,
                        state,
                        update_value,
                        order,
                        order_records,
                        log_records
                    )

                    # Update state
                    cash_now = new_state.cash
                    position_now = new_state.position
                    debt_now = new_state.debt
                    free_cash_now = new_state.free_cash
                    val_price_now = new_state.val_price
                    value_now = new_state.value
                    oidx = new_state.oidx
                    lidx = new_state.lidx

                # Now becomes last
                last_position[col] = position_now
                last_debt[col] = debt_now

        from_col = to_col

    return order_records[:oidx], log_records[:lidx]


# ############# Trades ############# #


@njit(cache=True)
def trade_duration_map_nb(record: tp.Record) -> int:
    """`map_func_nb` that returns trade duration."""
    return record['exit_idx'] - record['entry_idx']


@njit(cache=True)
def get_trade_stats_nb(size: float,
                       entry_price: float,
                       entry_fees: float,
                       exit_price: float,
                       exit_fees: float,
                       direction: int) -> tp.Tuple[float, float]:
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
def fill_trade_record_nb(record: tp.Record,
                         col: int,
                         entry_idx: int,
                         entry_size_sum: float,
                         entry_gross_sum: float,
                         entry_fees_sum: float,
                         exit_idx: int,
                         exit_size: float,
                         exit_price: float,
                         exit_fees: float,
                         direction: int,
                         status: int,
                         position_id: int) -> None:
    """Fill trade record."""
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
def orders_to_trades_nb(close: tp.Array2d, order_records: tp.RecordArray, col_map: tp.ColMap) -> tp.RecordArray:
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
    ...     empty_prep_func_nb,
    ...     empty_after_order_func_nb,
    ...     orders_to_trades_nb
    ... )

    >>> @njit
    ... def order_func_nb(c, order_size, order_price):
    ...     return create_order_nb(
    ...         order_size[c.i, c.col],
    ...         order_price[c.i, c.col],
    ...         fees=0.01,
    ...         slippage=0.01
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
    ...     empty_prep_func_nb, (),
    ...     empty_prep_func_nb, (),
    ...     empty_prep_func_nb, (),
    ...     order_func_nb, (order_size, order_price),
    ...     empty_after_order_func_nb, ()
    ... )

    >>> col_map = col_map_nb(order_records['col'], target_shape[1])
    >>> trade_records = orders_to_trades_nb(close, order_records, col_map)
    >>> pd.DataFrame.from_records(trade_records)
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
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    records = np.empty(len(order_records), dtype=trade_dt)
    tidx = 0
    entry_size_sum = 0.
    entry_gross_sum = 0.
    entry_fees_sum = 0.
    position_id = -1

    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        entry_idx = -1
        direction = -1
        last_id = -1

        for i in range(col_len):
            oidx = col_idxs[col_start_idxs[col] + i]
            record = order_records[oidx]

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

                # Reset current vars for a new position
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
                    fill_trade_record_nb(
                        records[tidx],
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
                    records[tidx]['id'] = tidx
                    tidx += 1

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
                    fill_trade_record_nb(
                        records[tidx],
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
                    records[tidx]['id'] = tidx
                    tidx += 1

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
            fill_trade_record_nb(
                records[tidx],
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
            records[tidx]['id'] = tidx
            tidx += 1

    return records[:tidx]


# ############# Positions ############# #

@njit(cache=True)
def fill_position_record_nb(record: tp.Record, trade_records: tp.RecordArray) -> None:
    """Fill position record."""
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
def copy_trade_record_nb(position_record: tp.Record, trade_record: tp.Record) -> None:
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
def trades_to_positions_nb(trade_records: tp.RecordArray, col_map: tp.ColMap) -> tp.RecordArray:
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
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    records = np.empty(len(trade_records), dtype=position_dt)
    pidx = 0
    from_tidx = -1

    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        last_id = -1
        last_position_id = -1

        for i in range(col_len):
            tidx = col_idxs[col_start_idxs[col] + i]
            record = trade_records[tidx]

            if record['id'] < last_id:
                raise ValueError("id must come in ascending order per column")
            last_id = record['id']

            position_id = record['position_id']

            if position_id != last_position_id:
                if last_position_id != -1:
                    if tidx - from_tidx > 1:
                        fill_position_record_nb(records[pidx], trade_records[from_tidx:tidx])
                    else:
                        # Speed up
                        copy_trade_record_nb(records[pidx], trade_records[from_tidx])
                    records[pidx]['id'] = pidx
                    pidx += 1
                from_tidx = tidx
                last_position_id = position_id

        if tidx - from_tidx > 0:
            fill_position_record_nb(records[pidx], trade_records[from_tidx:tidx + 1])
        else:
            # Speed up
            copy_trade_record_nb(records[pidx], trade_records[from_tidx])
        records[pidx]['id'] = pidx
        pidx += 1

    return records[:pidx]


# ############# Assets ############# #


@njit(cache=True)
def get_long_size_nb(position_before: float, position_now: float) -> float:
    """Get long size."""
    if position_before <= 0 and position_now <= 0:
        return 0.
    if position_before >= 0 and position_now < 0:
        return -position_before
    if position_before < 0 and position_now >= 0:
        return position_now
    return add_nb(position_now, -position_before)


@njit(cache=True)
def get_short_size_nb(position_before: float, position_now: float) -> float:
    """Get short size."""
    if position_before >= 0 and position_now >= 0:
        return 0.
    if position_before >= 0 and position_now < 0:
        return -position_now
    if position_before < 0 and position_now >= 0:
        return position_before
    return add_nb(position_before, -position_now)


@njit(cache=True)
def asset_flow_nb(target_shape: tp.Shape,
                  order_records: tp.RecordArray,
                  col_map: tp.ColMap,
                  direction: int) -> tp.Array2d:
    """Get asset flow series per column.

    Returns the total transacted amount of assets at each time step."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(target_shape, 0., dtype=np.float_)

    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        last_id = -1
        position_now = 0.

        for i in range(col_len):
            oidx = col_idxs[col_start_idxs[col] + i]
            record = order_records[oidx]

            if record['id'] < last_id:
                raise ValueError("id must come in ascending order per column")
            last_id = record['id']

            i = record['idx']
            side = record['side']
            size = record['size']

            if side == OrderSide.Sell:
                size *= -1
            new_position_now = add_nb(position_now, size)
            if direction == Direction.LongOnly:
                asset_flow = get_long_size_nb(position_now, new_position_now)
            elif direction == Direction.ShortOnly:
                asset_flow = get_short_size_nb(position_now, new_position_now)
            else:
                asset_flow = size
            out[i, col] = add_nb(out[i, col], asset_flow)
            position_now = new_position_now
    return out


@njit(cache=True)
def assets_nb(asset_flow: tp.Array2d) -> tp.Array2d:
    """Get asset series per column.

    Returns the current position at each time step."""
    out = np.empty_like(asset_flow)
    for col in range(asset_flow.shape[1]):
        position_now = 0.
        for i in range(asset_flow.shape[0]):
            flow_value = asset_flow[i, col]
            position_now = add_nb(position_now, flow_value)
            out[i, col] = position_now
    return out


@njit(cache=True)
def i_group_any_reduce_nb(i: int, group: int, a: tp.Array1d) -> bool:
    """Boolean "any" reducer for grouped columns."""
    return np.any(a)


@njit
def position_mask_grouped_nb(position_mask: tp.Array2d, group_lens: tp.Array1d) -> tp.Array2d:
    """Get whether in position for each row and group."""
    return generic_nb.squeeze_grouped_nb(position_mask, group_lens, i_group_any_reduce_nb).astype(np.bool_)


@njit(cache=True)
def group_mean_reduce_nb(group: int, a: tp.Array1d) -> float:
    """Mean reducer for grouped columns."""
    return np.mean(a)


@njit
def position_coverage_grouped_nb(position_mask: tp.Array2d, group_lens: tp.Array1d) -> tp.Array2d:
    """Get coverage of position for each row and group."""
    return generic_nb.reduce_grouped_nb(position_mask, group_lens, group_mean_reduce_nb)


# ############# Cash ############# #


@njit(cache=True)
def get_free_cash_diff_nb(position_before: float,
                          position_now: float,
                          debt_now: float,
                          price: float,
                          fees: float) -> tp.Tuple[float, float]:
    """Get updated debt and free cash flow."""
    size = add_nb(position_now, -position_before)
    final_cash = -size * price - fees
    if is_close_nb(size, 0):
        new_debt = debt_now
        free_cash_diff = 0.
    elif size > 0:
        if position_before < 0:
            if position_now < 0:
                short_size = abs(size)
            else:
                short_size = abs(position_before)
            avg_entry_price = debt_now / abs(position_before)
            debt_diff = short_size * avg_entry_price
            new_debt = add_nb(debt_now, -debt_diff)
            free_cash_diff = add_nb(2 * debt_diff, final_cash)
        else:
            new_debt = debt_now
            free_cash_diff = final_cash
    else:
        if position_now < 0:
            if position_before < 0:
                short_size = abs(size)
            else:
                short_size = abs(position_now)
            short_value = short_size * price
            new_debt = debt_now + short_value
            free_cash_diff = add_nb(final_cash, -2 * short_value)
        else:
            new_debt = debt_now
            free_cash_diff = final_cash
    return new_debt, free_cash_diff


@njit(cache=True)
def cash_flow_nb(target_shape: tp.Shape,
                 order_records: tp.RecordArray,
                 col_map: tp.ColMap,
                 free: bool) -> tp.Array2d:
    """Get (free) cash flow series per column."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(target_shape, 0., dtype=np.float_)

    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        last_id = -1
        position_now = 0.
        debt_now = 0.

        for i in range(col_len):
            oidx = col_idxs[col_start_idxs[col] + i]
            record = order_records[oidx]

            if record['id'] < last_id:
                raise ValueError("id must come in ascending order per column")
            last_id = record['id']

            i = record['idx']
            side = record['side']
            size = record['size']
            price = record['price']
            fees = record['fees']

            if side == OrderSide.Sell:
                size *= -1
            new_position_now = add_nb(position_now, size)
            if free:
                debt_now, cash_flow = get_free_cash_diff_nb(
                    position_now,
                    new_position_now,
                    debt_now,
                    price,
                    fees
                )
            else:
                cash_flow = -size * price - fees
            out[i, col] = add_nb(out[i, col], cash_flow)
            position_now = new_position_now
    return out


@njit(cache=True)
def sum_grouped_nb(a: tp.Array2d, group_lens: tp.Array1d) -> tp.Array2d:
    """Squeeze each group of columns into a single column using sum operation."""
    check_group_lens(group_lens, a.shape[1])

    out = np.empty((a.shape[0], len(group_lens)), dtype=np.float_)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        out[:, group] = np.sum(a[:, from_col:to_col], axis=1)
        from_col = to_col
    return out


@njit(cache=True)
def cash_flow_grouped_nb(cash_flow: tp.Array2d, group_lens: tp.Array1d) -> tp.Array2d:
    """Get cash flow series per group."""
    return sum_grouped_nb(cash_flow, group_lens)


@njit(cache=True)
def init_cash_grouped_nb(init_cash: tp.Array1d, group_lens: tp.Array1d, cash_sharing: bool) -> tp.Array1d:
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
def init_cash_nb(init_cash: tp.Array1d, group_lens: tp.Array1d, cash_sharing: bool) -> tp.Array1d:
    """Get initial cash per column."""
    if not cash_sharing:
        return init_cash
    group_lens_cs = np.cumsum(group_lens)
    out = np.full(group_lens_cs[-1], np.nan, dtype=np.float_)
    out[group_lens_cs - group_lens] = init_cash
    out = generic_nb.ffill_1d_nb(out)
    return out


@njit(cache=True)
def cash_nb(cash_flow: tp.Array2d, init_cash: tp.Array1d) -> tp.Array2d:
    """Get cash series per column."""
    out = np.empty_like(cash_flow)
    for col in range(cash_flow.shape[1]):
        for i in range(cash_flow.shape[0]):
            cash_now = init_cash[col] if i == 0 else out[i - 1, col]
            out[i, col] = add_nb(cash_now, cash_flow[i, col])
    return out


@njit(cache=True)
def cash_in_sim_order_nb(cash_flow: tp.Array2d,
                         group_lens: tp.Array1d,
                         init_cash_grouped: tp.Array1d,
                         call_seq: tp.Array2d) -> tp.Array2d:
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
                cash_now = add_nb(cash_now, cash_flow[i, col])
                out[i, col] = cash_now
        from_col = to_col
    return out


@njit(cache=True)
def cash_grouped_nb(target_shape: tp.Shape,
                    cash_flow_grouped: tp.Array2d,
                    group_lens: tp.Array1d,
                    init_cash_grouped: tp.Array1d) -> tp.Array2d:
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
def asset_value_nb(close: tp.Array2d, assets: tp.Array2d) -> tp.Array2d:
    """Get asset value series per column."""
    return close * assets


@njit(cache=True)
def asset_value_grouped_nb(asset_value: tp.Array2d, group_lens: tp.Array1d) -> tp.Array2d:
    """Get asset value series per group."""
    return sum_grouped_nb(asset_value, group_lens)


@njit(cache=True)
def value_in_sim_order_nb(cash: tp.Array2d,
                          asset_value: tp.Array2d,
                          group_lens: tp.Array1d,
                          call_seq: tp.Array2d) -> tp.Array2d:
    """Get portfolio value series in simulation order."""
    check_group_lens(group_lens, cash.shape[1])

    out = np.empty_like(cash)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_len = to_col - from_col
        asset_value_now = 0.
        # Without correctly treating NaN values, after one NaN all will be NaN
        since_last_nan = group_len
        for j in range(cash.shape[0] * group_len):
            i = j // group_len
            col = from_col + call_seq[i, from_col + j % group_len]
            if j >= group_len:
                last_j = j - group_len
                last_i = last_j // group_len
                last_col = from_col + call_seq[last_i, from_col + last_j % group_len]
                if not np.isnan(asset_value[last_i, last_col]):
                    asset_value_now -= asset_value[last_i, last_col]
            if np.isnan(asset_value[i, col]):
                since_last_nan = 0
            else:
                asset_value_now += asset_value[i, col]
            if since_last_nan < group_len:
                out[i, col] = np.nan
            else:
                out[i, col] = cash[i, col] + asset_value_now
            since_last_nan += 1

        from_col = to_col
    return out


@njit(cache=True)
def value_nb(cash: tp.Array2d, asset_value: tp.Array2d) -> tp.Array2d:
    """Get portfolio value series per column/group."""
    return cash + asset_value


@njit(cache=True)
def total_profit_nb(target_shape: tp.Shape,
                    close: tp.Array2d,
                    order_records: tp.RecordArray,
                    col_map: tp.ColMap) -> tp.Array1d:
    """Get total profit per column.

    A much faster version than the one based on `value_nb`."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    assets = np.full(target_shape[1], 0., dtype=np.float_)
    cash = np.full(target_shape[1], 0., dtype=np.float_)
    zero_mask = np.full(target_shape[1], False, dtype=np.bool_)

    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            zero_mask[col] = True
            continue
        last_id = -1

        for i in range(col_len):
            oidx = col_idxs[col_start_idxs[col] + i]
            record = order_records[oidx]

            if record['id'] < last_id:
                raise ValueError("id must come in ascending order per column")
            last_id = record['id']

            # Fill assets
            if record['side'] == OrderSide.Buy:
                order_size = record['size']
                assets[col] = add_nb(assets[col], order_size)
            else:
                order_size = record['size']
                assets[col] = add_nb(assets[col], -order_size)

            # Fill cash balance
            if record['side'] == OrderSide.Buy:
                order_cash = record['size'] * record['price'] + record['fees']
                cash[col] = add_nb(cash[col], -order_cash)
            else:
                order_cash = record['size'] * record['price'] - record['fees']
                cash[col] = add_nb(cash[col], order_cash)

    total_profit = cash + assets * close[-1, :]
    total_profit[zero_mask] = 0.
    return total_profit


@njit(cache=True)
def total_profit_grouped_nb(total_profit: tp.Array1d, group_lens: tp.Array1d) -> tp.Array1d:
    """Get total profit per group."""
    check_group_lens(group_lens, total_profit.shape[0])

    out = np.empty(len(group_lens), dtype=np.float_)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        out[group] = np.sum(total_profit[from_col:to_col])
        from_col = to_col
    return out


@njit(cache=True)
def final_value_nb(total_profit: tp.Array1d, init_cash: tp.Array1d) -> tp.Array1d:
    """Get total profit per column/group."""
    return total_profit + init_cash


@njit(cache=True)
def total_return_nb(total_profit: tp.Array1d, init_cash: tp.Array1d) -> tp.Array1d:
    """Get total return per column/group."""
    return total_profit / init_cash


@njit(cache=True)
def get_return_nb(input_value: float, output_value: float) -> float:
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
def returns_nb(value: tp.Array2d, init_cash: tp.Array1d) -> tp.Array2d:
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
def returns_in_sim_order_nb(value_iso: tp.Array2d,
                            group_lens: tp.Array1d,
                            init_cash_grouped: tp.Array1d,
                            call_seq: tp.Array2d) -> tp.Array2d:
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
def active_returns_nb(cash_flow: tp.Array2d, asset_value: tp.Array2d) -> tp.Array2d:
    """Get active return series per column/group."""
    out = np.empty_like(cash_flow)
    for col in range(cash_flow.shape[1]):
        for i in range(cash_flow.shape[0]):
            input_value = 0. if i == 0 else asset_value[i - 1, col]
            output_value = asset_value[i, col] + cash_flow[i, col]
            out[i, col] = get_return_nb(input_value, output_value)
    return out


@njit(cache=True)
def market_value_nb(close: tp.Array2d, init_cash: tp.Array1d) -> tp.Array2d:
    """Get market value per column."""
    return close / close[0] * init_cash


@njit(cache=True)
def market_value_grouped_nb(close: tp.Array2d, group_lens: tp.Array1d, init_cash_grouped: tp.Array1d) -> tp.Array2d:
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
def total_market_return_nb(market_value: tp.Array2d) -> tp.Array1d:
    """Get total market return per column/group."""
    out = np.empty(market_value.shape[1], dtype=np.float_)
    for col in range(market_value.shape[1]):
        out[col] = get_return_nb(market_value[0, col], market_value[-1, col])
    return out


@njit(cache=True)
def gross_exposure_nb(asset_value: tp.Array2d, cash: tp.Array2d) -> tp.Array2d:
    """Get gross exposure per column/group."""
    out = np.empty(asset_value.shape, dtype=np.float_)
    for col in range(out.shape[1]):
        for i in range(out.shape[0]):
            denom = add_nb(asset_value[i, col], cash[i, col])
            if denom == 0:
                out[i, col] = 0.
            else:
                out[i, col] = asset_value[i, col] / denom
    return out
