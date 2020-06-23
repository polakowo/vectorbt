"""Numba-compiled 1-dim and 2-dim functions.

!!! note
    `vectorbt` treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0).

    All functions passed as argument must be Numba-compiled.

    Records must remain in the order they were created."""

import numpy as np
from numba import njit, f8, i8

from vectorbt.records.enums import (
    DrawdownStatus,
    drawdown_dt,
    OrderSide,
    EventStatus,
    trade_dt,
    position_dt
)

size_zero_err = "Found order with size equal or less than zero"
price_zero_err = "Found order with price equal or less than zero"
sell_greater_than_buy_err = "Size of sell operations exceeds that of buy operations"


@njit(cache=True)
def isclose_nb(a, b, rel_tol=1e-09, abs_tol=0.0):
    """Tell whether two values are approximately equal."""
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


# ############# Indexing ############# #

@njit(cache=True)
def index_record_cols_nb(records, n_cols):
    """Index columns of `records`.

    Creates a 2-dim array with first column being start indices (inclusive) and
    second column being end indices (exclusive)."""
    # Record start and end indices for each column
    # Instead of doing np.flatnonzero and masking, this is much faster
    col_index = np.full((n_cols, 2), -1, dtype=i8)
    last_col = -1
    for r in range(records.shape[0]):
        col = records['col'][r]
        if last_col != col:
            if last_col != -1:
                col_index[last_col, 1] = r
            col_index[col, 0] = r
            last_col = col
        if r == records.shape[0] - 1:
            col_index[col, 1] = r + 1
    return col_index


@njit(cache=True)
def select_record_cols_nb(records, col_index, new_cols):
    """Select columns of `records` given column indices `col_index`."""
    col_index = col_index[new_cols]
    new_n = np.sum(col_index[:, 1] - col_index[:, 0])
    result = np.empty(new_n, dtype=records.dtype)
    j = 0
    for c in range(new_cols.shape[0]):
        col_records = np.copy(records[col_index[c, 0]:col_index[c, 1]])
        col_records['col'][:] = c  # don't forget to assign new column indices
        result[j:j + col_records.shape[0]] = col_records
        j += col_records.shape[0]
    return result


# ############# Mapping to matrix ############# #


@njit
def map_records_to_matrix_nb(records, target_shape, default_val, map_func_nb, *args):
    """Map each record to a value and store it in a matrix.

    `map_func_nb` must accept a single record and `*args`, and return a single value."""
    result = np.full(target_shape, default_val, dtype=f8)
    for r in range(records.shape[0]):
        result[records['idx'][r], records['col'][r]] = map_func_nb(records[r], *args)
    return result


@njit(cache=True)
def convert_array_to_matrix(a, records, target_shape, default_val):
    """Convert a 1-dim array already mapped by the user."""

    result = np.full(target_shape, default_val, dtype=f8)
    for r in range(a.shape[0]):
        result[records['idx'][r], records['col'][r]] = a[r]
    return result


# ############# Reducing ############# #


@njit
def reduce_records_nb(records, n_cols, default_val, reduce_func_nb, *args):
    """Reduce records by column.

    Faster than `map_records_to_matrix_nb` and `vbt.timeseries.*` used together, and also
    requires less memory. But does not take advantage of caching.

    `reduce_func_nb` must accept an array of records and `*args`, and return a single value."""
    result = np.full(n_cols, default_val, dtype=f8)
    from_r = 0
    col = -1
    for r in range(records.shape[0]):
        record_col = records['col'][r]
        if record_col != col:
            if col != -1:
                # At the beginning of second column do reduce on the first
                result[col] = reduce_func_nb(records[from_r:r], *args)
            from_r = r
            col = record_col
        if r == len(records) - 1:
            result[col] = reduce_func_nb(records[from_r:r + 1], *args)
    return result


@njit
def map_reduce_records_nb(records, n_cols, default_val, map_func_nb, reduce_func_nb, *args):
    """Map each record to a value and reduce all values by column.

    `map_func_nb` must accept a single record and `*args`, and return a single value.
    `reduce_func_nb` must accept an array of values and `*args`, and also return a single value."""
    result = np.full(n_cols, default_val, dtype=f8)
    mapped = np.empty(records.shape[0], dtype=f8)
    from_r = 0
    col = -1
    for r in range(records.shape[0]):
        mapped[r] = map_func_nb(records[r], *args)
        record_col = records['col'][r]
        if record_col != col:
            if col != -1:
                # At the beginning of second column do reduce on the first
                result[col] = reduce_func_nb(mapped[from_r:r], *args)
            from_r = r
            col = record_col
        if r == len(records) - 1:
            result[col] = reduce_func_nb(mapped[from_r:r + 1], *args)
    return result


# Some basic stat functions be passed to reduce_array_nb
min_reduce_nb = njit(cache=True)(lambda x, *args: np.min(x))
max_reduce_nb = njit(cache=True)(lambda x, *args: np.max(x))
mean_reduce_nb = njit(cache=True)(lambda x, *args: np.mean(x))
sum_reduce_nb = njit(cache=True)(lambda x, *args: np.sum(x))
count_reduce_nb = njit(cache=True)(lambda x, *args: len(x))


# ############# Drawdowns ############# #

@njit(cache=True)
def drawdown_records_nb(ts):
    """Find drawdows and store their information as records to an array.

        Example:
            Find drawdowns in time series:
            ```python-repl
            >>> import numpy as np
            >>> import pandas as pd
            >>> from numba import njit
            >>> from vectorbt.records.nb import drawdown_records_nb

            >>> ts = np.asarray([
            ...     [1, 5, 1, 3],
            ...     [2, 4, 2, 2],
            ...     [3, 3, 3, 1],
            ...     [4, 2, 2, 2],
            ...     [5, 1, 1, 3]
            ... ])
            >>> records = drawdown_records_nb(ts)

            >>> print(pd.DataFrame.from_records(records))
               col  idx  start_idx  valley_idx  end_idx  status
            0    1    4          0           4        4       0
            1    2    4          2           4        4       0
            2    3    4          0           2        4       1
            ```"""
    result = np.empty(ts.shape[0] * ts.shape[1], dtype=drawdown_dt)
    j = 0

    for col in range(ts.shape[1]):
        drawdown_started = False
        peak_idx = np.nan
        valley_idx = np.nan
        peak_val = ts[0, col]
        valley_val = ts[0, col]
        store_drawdown = False
        status = -1

        for i in range(ts.shape[0]):
            cur_val = ts[i, col]

            if not np.isnan(cur_val):
                if np.isnan(peak_val) or cur_val >= peak_val:
                    # Value increased
                    if not drawdown_started:
                        # If not running, register new peak
                        peak_val = cur_val
                        peak_idx = i
                    else:
                        # If running, potential recovery
                        if cur_val >= peak_val:
                            drawdown_started = False
                            store_drawdown = True
                            status = DrawdownStatus.Recovered
                else:
                    # Value decreased
                    if not drawdown_started:
                        # If not running, start new drawdown
                        drawdown_started = True
                        valley_val = cur_val
                        valley_idx = i
                    else:
                        # If running, potential valley
                        if cur_val < valley_val:
                            valley_val = cur_val
                            valley_idx = i

                if i == ts.shape[0] - 1 and drawdown_started:
                    # If still running, mark for save
                    drawdown_started = False
                    store_drawdown = True
                    status = DrawdownStatus.Active

                if store_drawdown:
                    # Save drawdown to the records
                    result[j]['col'] = col
                    result[j]['idx'] = i
                    result[j]['start_idx'] = peak_idx
                    result[j]['valley_idx'] = valley_idx
                    result[j]['end_idx'] = i
                    result[j]['status'] = status
                    j += 1

                    # Reset running vars for a new drawdown
                    peak_idx = i
                    valley_idx = i
                    peak_val = cur_val
                    valley_val = cur_val
                    store_drawdown = False
                    status = -1

    return result[:j]


@njit(cache=True)
def dd_start_value_map_nb(record, ts, *args):
    """`map_func_nb` that returns start value of a drawdown."""
    return ts[record['start_idx'], record['col']]


@njit(cache=True)
def dd_valley_value_map_nb(record, ts, *args):
    """`map_func_nb` that returns valley value of a drawdown."""
    return ts[record['valley_idx'], record['col']]


@njit(cache=True)
def dd_end_value_map_nb(record, ts, *args):
    """`map_func_nb` that returns end value of a drawdown.

    This can be either recovery value or last value of an active drawdown."""
    return ts[record['end_idx'], record['col']]


@njit(cache=True)
def dd_drawdown_map_nb(record, ts, *args):
    """`map_func_nb` that returns drawdown value of a drawdown."""
    valley_val = dd_valley_value_map_nb(record, ts)
    start_val = dd_start_value_map_nb(record, ts)
    return (valley_val - start_val) / start_val


@njit(cache=True)
def dd_duration_map_nb(record, *args):
    """`map_func_nb` that returns total duration of a drawdown."""
    return record['end_idx'] - record['start_idx']


@njit(cache=True)
def dd_ptv_duration_map_nb(record, *args):
    """`map_func_nb` that returns duration of the peak-to-valley (PtV) phase."""
    return record['valley_idx'] - record['start_idx']


@njit(cache=True)
def dd_vtr_duration_map_nb(record, *args):
    """`map_func_nb` that returns duration of the valley-to-recovery (VtR) phase."""
    return record['end_idx'] - record['valley_idx']


@njit(cache=True)
def dd_vtr_duration_ratio_map_nb(record, *args):
    """`map_func_nb` that returns ratio of VtR duration to total duration."""
    return dd_vtr_duration_map_nb(record) / dd_duration_map_nb(record)


@njit(cache=True)
def dd_recovery_return_map_nb(record, ts, *args):
    """`map_func_nb` that returns recovery return of a drawdown."""
    end_val = dd_end_value_map_nb(record, ts)
    valley_val = dd_valley_value_map_nb(record, ts)
    return (end_val - valley_val) / valley_val


# ############# Orders ############# #


@njit(cache=True)
def order_size_map_nb(record, *args):
    """`map_func_nb` that returns order size."""
    return record['size']


@njit(cache=True)
def order_price_map_nb(record, *args):
    """`map_func_nb` that returns order price."""
    return record['price']


@njit(cache=True)
def order_fees_map_nb(record, *args):
    """`map_func_nb` that returns order fees."""
    return record['fees']


# ############# Events ############# #


@njit(cache=True)
def event_duration_map_nb(record, *args):
    """`map_func_nb` that returns event duration."""
    return record['close_idx'] - record['open_idx']


@njit(cache=True)
def event_pnl_map_nb(record, *args):
    """`map_func_nb` that returns event PnL."""
    return record['pnl']


@njit(cache=True)
def event_return_map_nb(record, *args):
    """`map_func_nb` that returns event return."""
    return record['return']


@njit(cache=True)
def event_sqn_reduce_nb(records, ddof):
    """`reduce_func_nb` that returns event SQN."""
    cnt = len(records)
    if cnt - ddof == 0:
        return np.nan
    std = np.std(records['pnl']) * np.sqrt(cnt / (cnt - ddof))
    return np.sqrt(cnt) * np.mean(records['pnl']) / std


# ############# Trades ############# #


@njit(cache=True)
def save_trade_nb(record, col, i, order_size, order_price, order_fees, position_start,
                  buy_size_sum, buy_gross_sum, buy_fees_sum, position_idx, status):
    """Save trade to the record."""

    # Opening price is the size-weighted average over all purchase prices
    avg_buy_price = buy_gross_sum / buy_size_sum

    # Opening fees are the size-weighted average over all purchase fees
    frac_buy_fees = order_size / buy_size_sum * buy_fees_sum

    # Calculate PnL and return
    buy_val = order_size * avg_buy_price + frac_buy_fees
    sell_val = order_size * order_price - order_fees
    pnl = sell_val - buy_val
    ret = (sell_val - buy_val) / buy_val

    # Save trade
    record['col'] = col
    record['idx'] = i
    record['size'] = order_size
    record['open_idx'] = position_start
    record['open_price'] = avg_buy_price
    record['open_fees'] = frac_buy_fees
    record['close_idx'] = i
    record['close_price'] = order_price
    record['close_fees'] = order_fees
    record['pnl'] = pnl
    record['return'] = ret
    record['status'] = status
    record['position_idx'] = position_idx


@njit(cache=True)
def trade_records_nb(price, order_records):
    """Find trades and store their information as records to an array.

    One position can have multiple trades. A trade in this regard is just a sell operation.
    Performance for this operation is calculated based on the size weighted average of
    previous buy operations in the same position.

    Example:
        Find trades in simulated orders:
        ```python-repl
        >>> import numpy as np
        >>> import pandas as pd
        >>> from numba import njit
        >>> from vectorbt.portfolio import Order
        >>> from vectorbt.portfolio.nb import simulate_nb
        >>> from vectorbt.records.nb import trade_records_nb

        >>> init_capital = np.full(1, 100)
        >>> order_price = price = np.arange(1, 6)[:, None]
        >>> order_size = np.asarray([1, -1, 1, -1, 1])[:, None]

        >>> @njit
        ... def order_func_nb(col, i, run_cash, run_shares):
        ...     return Order(order_size[i, col], order_price[i, col],
        ...          fees=0.01, slippage=0., fixed_fees=0.)

        >>> order_records, cash, shares = simulate_nb(
        ...     price.shape, init_capital, order_func_nb)
        >>> records = trade_records_nb(price, order_records)

        >>> print(pd.DataFrame.from_records(records))
           col  idx  size  open_idx  open_price  open_fees  close_idx  close_price  \
        0    0    1   1.0         0         1.0       0.01          1          2.0
        1    0    3   1.0         2         3.0       0.03          3          4.0
        2    0    4   1.0         4         5.0       0.05          4          5.0

           close_fees   pnl    return  status  position_idx
        0        0.02  0.97  0.960396       1             0
        1        0.04  0.93  0.306931       1             1
        2        0.00 -0.05 -0.009901       0             2
        ```"""
    result = np.empty(price.shape[0] * price.shape[1], dtype=trade_dt)
    position_idx = -1
    j = 0
    prev_col = -1
    buy_size_sum = 0.
    buy_gross_sum = 0.
    buy_fees_sum = 0.
    position_start = -1

    for r in range(order_records.shape[0]):
        col = int(order_records[r]['col'])
        i = int(order_records[r]['idx'])
        order_size = order_records[r]['size']
        order_price = order_records[r]['price']
        order_fees = order_records[r]['fees']
        order_side = order_records[r]['side']

        if order_size <= 0.:
            raise Exception(size_zero_err)
        if order_price <= 0.:
            raise Exception(price_zero_err)

        if col != prev_col:
            # Column has changed
            if prev_col != -1 and buy_size_sum > 0.:
                # If trade in previous column hasn't been closed, calculate its unrealized metrics
                save_trade_nb(
                    result[j],
                    prev_col,
                    price.shape[0] - 1,
                    buy_size_sum,
                    price[price.shape[0] - 1, prev_col],
                    0.,
                    position_start,
                    buy_size_sum,
                    buy_gross_sum,
                    buy_fees_sum,
                    position_idx,
                    EventStatus.Open
                )
                j += 1

            prev_col = col
            buy_size_sum = 0.
            buy_gross_sum = 0.
            buy_fees_sum = 0.

        if order_side == OrderSide.Buy:
            # Buy operation
            if buy_size_sum == 0.:
                position_start = i
                position_idx += 1

            # Position increased
            buy_size_sum += order_size
            buy_gross_sum += order_size * order_price
            buy_fees_sum += order_fees

        elif order_side == OrderSide.Sell:
            # Sell operation
            # Close the current trade
            save_trade_nb(
                result[j],
                col,
                i,
                order_size,
                order_price,
                order_fees,
                position_start,
                buy_size_sum,
                buy_gross_sum,
                buy_fees_sum,
                position_idx,
                EventStatus.Closed
            )
            j += 1

            # Position decreased, previous purchases have now less impact
            if isclose_nb(buy_size_sum, order_size):
                size_fraction = 0.  # numerical stability
            else:
                if order_size > buy_size_sum:
                    raise Exception(sell_greater_than_buy_err)
                size_fraction = (buy_size_sum - order_size) / buy_size_sum
            buy_size_sum *= size_fraction
            buy_gross_sum *= size_fraction
            buy_fees_sum *= size_fraction

        if r == order_records.shape[0] - 1 and buy_size_sum > 0.:
            # If the last trade hasn't been closed, calculate its unrealized metrics
            save_trade_nb(
                result[j],
                col,
                price.shape[0] - 1,
                buy_size_sum,
                price[price.shape[0] - 1, col],
                0.,
                position_start,
                buy_size_sum,
                buy_gross_sum,
                buy_fees_sum,
                position_idx,
                EventStatus.Open
            )
            j += 1
    return result[:j]


# ############# Positions ############# #


@njit(cache=True)
def save_position_nb(record, col, i, position_start, buy_size_sum, buy_gross_sum, buy_fees_sum,
                     sell_size_sum, sell_gross_sum, sell_fees_sum, status):
    """Save position to the record."""

    # Opening price is the size-weighted average over all buy prices
    avg_buy_price = buy_gross_sum / buy_size_sum

    # Closing price is the size-weighted average over all sell prices
    avg_sell_price = sell_gross_sum / sell_size_sum

    # Calculate PnL and return
    buy_val = buy_size_sum * avg_buy_price + buy_fees_sum
    sell_val = buy_size_sum * avg_sell_price - sell_fees_sum
    pnl = sell_val - buy_val
    ret = (sell_val - buy_val) / buy_val

    # Save trade
    record['col'] = col
    record['idx'] = i
    record['size'] = buy_size_sum
    record['open_idx'] = position_start
    record['open_price'] = avg_buy_price
    record['open_fees'] = buy_fees_sum
    record['close_idx'] = i
    record['close_price'] = avg_sell_price
    record['close_fees'] = sell_fees_sum
    record['pnl'] = pnl
    record['return'] = ret
    record['status'] = status


@njit(cache=True)
def position_records_nb(price, order_records):
    """Find positions and store their information as records to an array.

    Example:
        Find positions in simulated orders:
        ```python-repl
        >>> import numpy as np
        >>> import pandas as pd
        >>> from numba import njit
        >>> from vectorbt.portfolio import Order
        >>> from vectorbt.portfolio.nb import simulate_nb
        >>> from vectorbt.records.nb import position_records_nb

        >>> init_capital = np.full(1, 100)
        >>> order_price = price = np.arange(1, 6)[:, None]
        >>> order_size = np.asarray([1, -1, 1, -1, 1])[:, None]

        >>> @njit
        ... def order_func_nb(col, i, run_cash, run_shares):
        ...     return Order(order_size[i, col], order_price[i, col],
        ...          fees=0.01, slippage=0., fixed_fees=0.)

        >>> order_records, cash, shares = simulate_nb(
        ...     price.shape, init_capital, order_func_nb)
        >>> records = position_records_nb(price, order_records)

        >>> print(pd.DataFrame.from_records(records))
           col  idx  size  open_idx  open_price  open_fees  close_idx  close_price  \
        0    0    1   1.0         0         1.0       0.01          1          2.0
        1    0    3   1.0         2         3.0       0.03          3          4.0
        2    0    4   1.0         4         5.0       0.05          4          5.0

           close_fees   pnl    return  status
        0        0.02  0.97  0.960396       1
        1        0.04  0.93  0.306931       1
        2        0.00 -0.05 -0.009901       0
        ```"""
    result = np.empty(price.shape[0] * price.shape[1], dtype=position_dt)
    j = 0
    prev_col = -1
    buy_size_sum = 0.
    buy_gross_sum = 0.
    buy_fees_sum = 0.
    sell_size_sum = 0.
    sell_gross_sum = 0.
    sell_fees_sum = 0.
    position_start = -1

    for r in range(order_records.shape[0]):
        col = int(order_records[r]['col'])
        i = int(order_records[r]['idx'])
        order_size = order_records[r]['size']
        order_price = order_records[r]['price']
        order_fees = order_records[r]['fees']
        order_side = order_records[r]['side']

        if order_size <= 0.:
            raise Exception(size_zero_err)
        if order_price <= 0.:
            raise Exception(price_zero_err)

        if col != prev_col:
            # Column has changed
            if prev_col != -1 and buy_size_sum > sell_size_sum:
                # If position in previous column hasn't been closed, calculate its unrealized metrics
                sell_gross_sum += (buy_size_sum - sell_size_sum) * price[price.shape[0] - 1, col]
                sell_size_sum = buy_size_sum
                # NOTE: We have no information about fees here, so we don't add them
                save_position_nb(
                    result[j],
                    prev_col,
                    price.shape[0] - 1,
                    position_start,
                    buy_size_sum,
                    buy_gross_sum,
                    buy_fees_sum,
                    sell_size_sum,
                    sell_gross_sum,
                    sell_fees_sum,
                    EventStatus.Open
                )
                j += 1

            prev_col = col
            buy_size_sum = 0.
            buy_gross_sum = 0.
            buy_fees_sum = 0.
            sell_size_sum = 0.
            sell_gross_sum = 0.
            sell_fees_sum = 0.

        if order_side == OrderSide.Buy:
            # Buy operation
            if buy_size_sum == 0.:
                position_start = i

            # Position increased
            buy_size_sum += order_size
            buy_gross_sum += order_size * order_price
            buy_fees_sum += order_fees

        elif order_side == OrderSide.Sell:
            # Sell operation
            # Position decreased
            sell_size_sum += order_size
            sell_gross_sum += order_size * order_price
            sell_fees_sum += order_fees
            if isclose_nb(buy_size_sum, sell_size_sum):
                sell_size_sum = buy_size_sum  # numerical stability
            else:
                if sell_size_sum > buy_size_sum:
                    raise Exception(sell_greater_than_buy_err)

        if buy_size_sum == sell_size_sum:
            # Close the current position
            save_position_nb(
                result[j],
                col,
                i,
                position_start,
                buy_size_sum,
                buy_gross_sum,
                buy_fees_sum,
                sell_size_sum,
                sell_gross_sum,
                sell_fees_sum,
                EventStatus.Closed
            )
            j += 1

            # Reset running vars for a new position
            buy_size_sum = 0.
            buy_gross_sum = 0.
            buy_fees_sum = 0.
            sell_size_sum = 0.
            sell_gross_sum = 0.
            sell_fees_sum = 0.

        if r == order_records.shape[0] - 1 and buy_size_sum > sell_size_sum:
            # If the last position hasn't been closed, calculate its unrealized metrics
            sell_gross_sum += (buy_size_sum - sell_size_sum) * price[price.shape[0] - 1, col]
            sell_size_sum = buy_size_sum
            # NOTE: We have no information about fees here, so we don't add them
            save_position_nb(
                result[j],
                col,
                price.shape[0] - 1,
                position_start,
                buy_size_sum,
                buy_gross_sum,
                buy_fees_sum,
                sell_size_sum,
                sell_gross_sum,
                sell_fees_sum,
                EventStatus.Open
            )
            j += 1

    return result[:j]
