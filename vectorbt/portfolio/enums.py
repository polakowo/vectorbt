"""Named tuples and enumerated types."""

import numpy as np
from collections import namedtuple
import json

__all__ = [
    'SimulationContext',
    'GroupContext',
    'RowContext',
    'SegmentContext',
    'OrderContext',
    'InitCashMode',
    'CallSeqType',
    'SizeType',
    'ConflictMode',
    'Order',
    'NoOrder',
    'OrderStatus',
    'OrderSide',
    'status_info_desc',
    'StatusInfo',
    'OrderResult',
    'RejectedOrderError',
    'Direction',
    'order_dt',
    'TradeDirection',
    'TradeStatus',
    'trade_dt',
    'position_dt',
    'log_dt',
    'TradeType',
    'BenchmarkSize'
]

__pdoc__ = {}

# We use namedtuple for enums and classes to be able to use them in Numba

# ############# Portfolio ############# #

SimulationContext = namedtuple('SimulationContext', [
    'target_shape',
    'close',
    'group_lens',
    'init_cash',
    'cash_sharing',
    'call_seq',
    'active_mask',
    'order_records',
    'log_records',
    'last_cash',
    'last_shares',
    'last_val_price'
])

__pdoc__['SimulationContext'] = "A named tuple representing context of the simulation."
__pdoc__['SimulationContext.target_shape'] = """Target shape.

A tuple with exactly two elements: the number of rows and columns.
"""
__pdoc__['SimulationContext.close'] = """Reference price, such as close.

Has shape `target_shape`.
"""
__pdoc__['SimulationContext.group_lens'] = """Column count per group.

Even if columns are not grouped, `group_lens` contains ones - one column per group.
"""
__pdoc__['SimulationContext.init_cash'] = """Initial capital per column, or per group if cash sharing is enabled.

If `cash_sharing` is True, has shape `(target_shape[0], group_lens.shape[0])`. 
Otherwise, has shape `target_shape`.
"""
__pdoc__['SimulationContext.cash_sharing'] = """Whether cash sharing is enabled."""
__pdoc__['SimulationContext.call_seq'] = """Default sequence of calls per segment.

Controls the sequence in which `order_func_nb` is executed within a segment. 

Has shape `target_shape` and each value must exist in the range `[0, group_len)`.

!!! note
    To change the call sequence dynamically, better change `call_seq_now` in-place.
"""
__pdoc__['SimulationContext.active_mask'] = """Mask of whether a particular segment should be executed.

A segment is simply a sequence of `order_func_nb` calls under the same group and row.

Has shape `(target_shape[0], group_lens.shape[0])`.
"""
__pdoc__['SimulationContext.order_records'] = "Order records filled up to this point."
__pdoc__['SimulationContext.log_records'] = "Log records filled up to this point."
__pdoc__['SimulationContext.last_cash'] = """Last cash per column, or per group if cash sharing is enabled.

Has the same shape as `init_cash`.
"""
__pdoc__['SimulationContext.last_shares'] = """Last shares per column.

Has shape `(target_shape[1],)`.
"""
__pdoc__['SimulationContext.last_val_price'] = """Last size valuation price.

Used to calculate `value_now`. Can be changed in-place before group valuation.

Has shape `(target_shape[1],)`.
"""

GroupContext = namedtuple('GroupContext', [
    *SimulationContext._fields,
    'group',
    'group_len',
    'from_col',
    'to_col'
])

__pdoc__['GroupContext'] = "A named tuple representing context of the group."
for field in SimulationContext._fields:
    __pdoc__[f'GroupContext.{field}'] = f"See `SimulationContext.{field}`."
__pdoc__['GroupContext.group'] = """Index of the group.

Has range `[0, group_lens.shape[0])`.
"""
__pdoc__['GroupContext.group_len'] = """Number of columns in the group.

Scalar value. Same as `group_lens[group]`.
"""
__pdoc__['GroupContext.from_col'] = """Index of the first column in the group.

Has range `[0, target_shape[1])`.
"""
__pdoc__['GroupContext.to_col'] = """Index of the first column in the next group.

Has range `[1, target_shape[1] + 1)`. 

If columns are not grouped, equals `from_col + 1`.
"""

RowContext = namedtuple('RowContext', [
    *SimulationContext._fields,
    'i'
])
__pdoc__['RowContext'] = "A named tuple representing context of the row."
for field in SimulationContext._fields:
    __pdoc__[f'RowContext.{field}'] = f"See `SimulationContext.{field}`."
__pdoc__['RowContext.i'] = """Current row (time axis).

Has range `[0, target_shape[0])`.
"""

SegmentContext = namedtuple('SegmentContext', [
    *SimulationContext._fields,
    'i',
    'group',
    'group_len',
    'from_col',
    'to_col',
    'call_seq_now'
])
__pdoc__['SegmentContext'] = "A named tuple representing context of the segment."
for field in SimulationContext._fields:
    __pdoc__[f'SegmentContext.{field}'] = f"See `SimulationContext.{field}`."
__pdoc__['SegmentContext.i'] = "See `RowContext.i`."
__pdoc__['SegmentContext.group'] = "See `GroupContext.group`."
__pdoc__['SegmentContext.group_len'] = "See `GroupContext.group_len`."
__pdoc__['SegmentContext.from_col'] = "See `GroupContext.from_col`."
__pdoc__['SegmentContext.to_col'] = "See `GroupContext.to_col`."
__pdoc__['SegmentContext.call_seq_now'] = """Current sequence of calls.

Has shape `(group_len,)`. 
"""

OrderContext = namedtuple('OrderContext', [
    *SegmentContext._fields,
    'col',
    'call_idx',
    'cash_now',
    'shares_now',
    'val_price_now',
    'value_now'
])
__pdoc__['OrderContext'] = "A named tuple representing context of the order."
for field in SegmentContext._fields:
    __pdoc__[f'OrderContext.{field}'] = f"See `SegmentContext.{field}`."
__pdoc__['OrderContext.col'] = """Current column (feature axis).

Has range `[0, target_shape[1])` and is always within `[from_col, to_col)`.
"""
__pdoc__['OrderContext.call_idx'] = """Index of the call in `call_seq_now`.

Has range `[0, group_len)`.
"""
__pdoc__['OrderContext.cash_now'] = """Current cash available.

Scalar value. Per group if cash sharing is enabled, otherwise per column.
"""
__pdoc__['OrderContext.shares_now'] = """Current shares available.

Scalar value. Always per column.
"""
__pdoc__['OrderContext.val_price_now'] = """Current size valuation price.

Scalar value. Always per column.
"""
__pdoc__['OrderContext.value_now'] = """Current value.

Scalar value. Per group if cash sharing is enabled, otherwise per column.

Current value is calculated using `last_val_price`.
"""

InitCashMode = namedtuple('InitCashMode', [
    'Auto',
    'AutoAlign'
])(*range(2))
"""_"""

__pdoc__['InitCashMode'] = f"""Initial cash mode.

```plaintext
{json.dumps(dict(zip(InitCashMode._fields, InitCashMode)), indent=2, default=str)}
```

Attributes:
    Auto: Optimal initial cash for each column.
    AutoAlign: Optimal initial cash aligned across all columns.
"""

CallSeqType = namedtuple('CallSeqType', [
    'Default',
    'Reversed',
    'Random',
    'Auto'
])(*range(4))
"""_"""

__pdoc__['CallSeqType'] = f"""Call sequence type.

```plaintext
{json.dumps(dict(zip(CallSeqType._fields, CallSeqType)), indent=2, default=str)}
```

Attributes:
    Default: Place calls from left to right.
    Reversed: Place calls from right to left.
    Random: Place calls randomly.
    Auto: Place calls dynamically based on order value.
"""

SizeType = namedtuple('SizeType', [
    'Shares',
    'TargetShares',
    'TargetValue',
    'TargetPercent'
])(*range(4))
"""_"""

__pdoc__['SizeType'] = f"""Size type.

```plaintext
{json.dumps(dict(zip(SizeType._fields, SizeType)), indent=2, default=str)}
```

Attributes:
    Shares: Number of shares.
    TargetShares: Number of shares to hold after transaction.
    TargetValue: Total value of holdings to hold after transaction.
    TargetPercent: Percentage of total value to hold after transaction.
"""

ConflictMode = namedtuple('ConflictMode', [
    'Ignore',
    'Entry',
    'Exit',
    'Opposite'
])(*range(4))
"""_"""

__pdoc__['ConflictMode'] = f"""Conflict mode.

```plaintext
{json.dumps(dict(zip(ConflictMode._fields, ConflictMode)), indent=2, default=str)}
```

What should happen if both entry and exit signals occur simultaneously?

Attributes:
    Ignore: Ignore both signals.
    Entry: Use entry signal.
    Exit: Use exit signal.
    Opposite: Use opposite signal. Takes effect only when in position.
"""

Order = namedtuple('Order', [
    'size',
    'size_type',
    'direction',
    'price',
    'fees',
    'fixed_fees',
    'slippage',
    'min_size',
    'max_size',
    'reject_prob',
    'close_first',
    'allow_partial',
    'raise_reject',
    'log'
])

__pdoc__['Order'] = "A named tuple representing an order."
__pdoc__['Order.size'] = "Size in shares. Final size will depend upon your funds."
__pdoc__['Order.size_type'] = "See `SizeType`."
__pdoc__['Order.direction'] = "See `Direction`."
__pdoc__['Order.price'] = "Price per share. Final price will depend upon slippage."
__pdoc__['Order.fees'] = "Fees in percentage of the order value."
__pdoc__['Order.fixed_fees'] = "Fixed amount of fees to pay for this order."
__pdoc__['Order.slippage'] = "Slippage in percentage of `price`."
__pdoc__['Order.min_size'] = "Minimum size. Lower than that will be rejected."
__pdoc__['Order.max_size'] = "Maximum size. Higher than that will be cut."
__pdoc__['Order.reject_prob'] = "Probability of rejecting this order."
__pdoc__['Order.close_first'] = """Whether reversal should close the position first. 

Requires second order to open opposite position."""
__pdoc__['Order.allow_partial'] = "Whether to allow partial fill."
__pdoc__['Order.raise_reject'] = "Whether to raise exception if order has been rejected."
__pdoc__['Order.log'] = "Whether to log this order by filling a log record."

NoOrder = Order(np.nan, -1, -1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, False, False)
"""_"""

__pdoc__['NoOrder'] = "Order that will not be processed."

OrderStatus = namedtuple('OrderStatus', [
    'Filled',
    'Ignored',
    'Rejected'
])(*range(3))
"""_"""

__pdoc__['OrderStatus'] = f"""Order status.

```plaintext
{json.dumps(dict(zip(OrderStatus._fields, OrderStatus)), indent=2, default=str)}
```

Attributes:
    Filled: Order filled.
    Ignored: Order ignored.
    Rejected: Order rejected.
"""

OrderSide = namedtuple('OrderSide', [
    'Buy',
    'Sell'
])(*range(2))
"""_"""

__pdoc__['OrderSide'] = f"""Order side.

```plaintext
{json.dumps(dict(zip(OrderSide._fields, OrderSide)), indent=2, default=str)}
```
"""

StatusInfo = namedtuple('StatusInfo', [
    'SizeNaN',
    'PriceNaN',
    'ValPriceNaN',
    'ValueNaN',
    'ValueZeroNeg',
    'SizeZero',
    'NoCashShort',
    'NoCashLong',
    'NoOpenPosition',
    'MaxSizeExceeded',
    'RandomEvent',
    'CantCoverFees',
    'MinSizeNotReached',
    'PartialFill'
])(*range(14))
"""_"""

__pdoc__['StatusInfo'] = f"""Order status information.

```plaintext
{json.dumps(dict(zip(StatusInfo._fields, StatusInfo)), indent=2, default=str)}
```
"""

status_info_desc = [
    "Size is NaN",
    "Price is NaN",
    "Asset valuation price is NaN",
    "Asset/group value is NaN",
    "Asset/group value is zero or negative",
    "Size is zero",
    "Not enough cash to short",
    "Not enough cash to long",
    "No open position to reduce/close",
    "Size is greater than maximum allowed",
    "Random event happened",
    "Not enough cash to cover fees",
    "Final size is less than minimum allowed",
    "Final size is less than requested"
]
"""_"""

__pdoc__['status_info_desc'] = f"""Order status description.

```plaintext
{json.dumps(status_info_desc, indent=2, default=str)}
```
"""

OrderResult = namedtuple('OrderResult', [
    'size',
    'price',
    'fees',
    'side',
    'status',
    'status_info'
])

__pdoc__['OrderResult'] = "A named tuple representing an order result."
__pdoc__['OrderResult.size'] = "Filled size in shares."
__pdoc__['OrderResult.price'] = "Filled price per share, adjusted with slippage."
__pdoc__['OrderResult.fees'] = "Total fees paid for this order."
__pdoc__['OrderResult.side'] = "See `vectorbt.portfolio.enums.OrderSide`."
__pdoc__['OrderResult.status'] = "See `OrderStatus`."
__pdoc__['OrderResult.status_info'] = "See `vectorbt.portfolio.enums.StatusInfo`."


class RejectedOrderError(Exception):
    """Rejected order error."""
    pass


Direction = namedtuple('Direction', [
    'LongOnly',
    'ShortOnly',
    'All'
])(*range(3))
"""_"""

__pdoc__['Direction'] = f"""Position direction.

```plaintext
{json.dumps(dict(zip(Direction._fields, Direction)), indent=2, default=str)}
```

Attributes:
    LongOnly: Only long positions.
    ShortOnly: Only short positions.
    All: Both long and short positions.
"""

# ############# Records ############# #

order_dt = np.dtype([
    ('id', np.int_),
    ('idx', np.int_),
    ('col', np.int_),
    ('size', np.float_),
    ('price', np.float_),
    ('fees', np.float_),
    ('side', np.int_),
], align=True)
"""_"""

__pdoc__['order_dt'] = f"""`np.dtype` of order records.

```plaintext
{json.dumps(dict(zip(
    dict(order_dt.fields).keys(),
    list(map(lambda x: str(x[0]), dict(order_dt.fields).values()))
)), indent=2, default=str)}
```
"""

TradeDirection = namedtuple('TradeDirection', [
    'Long',
    'Short'
])(*range(2))
"""_"""

__pdoc__['TradeDirection'] = f"""Event direction.

```plaintext
{json.dumps(dict(zip(TradeDirection._fields, TradeDirection)), indent=2, default=str)}
```
"""

TradeStatus = namedtuple('TradeStatus', [
    'Open',
    'Closed'
])(*range(2))
"""_"""

__pdoc__['TradeStatus'] = f"""Event status.

```plaintext
{json.dumps(dict(zip(TradeStatus._fields, TradeStatus)), indent=2, default=str)}
```
"""

_trade_fields = [
    ('id', np.int_),
    ('col', np.int_),
    ('size', np.float_),
    ('entry_idx', np.int_),
    ('entry_price', np.float_),
    ('entry_fees', np.float_),
    ('exit_idx', np.int_),
    ('exit_price', np.float_),
    ('exit_fees', np.float_),
    ('pnl', np.float_),
    ('return', np.float_),
    ('direction', np.int_),
    ('status', np.int_),
    ('position_id', np.int_)
]

trade_dt = np.dtype(_trade_fields, align=True)
"""_"""

__pdoc__['trade_dt'] = f"""`np.dtype` of trade records.

```plaintext
{json.dumps(dict(zip(
    dict(trade_dt.fields).keys(),
    list(map(lambda x: str(x[0]), dict(trade_dt.fields).values()))
)), indent=2, default=str)}
```
"""

_position_fields = _trade_fields[:-1]

position_dt = np.dtype(_position_fields, align=True)
"""_"""

__pdoc__['position_dt'] = f"""`np.dtype` of position records.

```plaintext
{json.dumps(dict(zip(
    dict(position_dt.fields).keys(),
    list(map(lambda x: str(x[0]), dict(position_dt.fields).values()))
)), indent=2, default=str)}
```
"""

_log_fields = [
    ('id', np.int_),
    ('idx', np.int_),
    ('col', np.int_),
    ('group', np.int_),
    ('cash_now', np.float_),
    ('shares_now', np.float_),
    ('val_price_now', np.float_),
    ('value_now', np.float_),
    ('size', np.float_),
    ('size_type', np.int_),
    ('direction', np.int_),
    ('price', np.float_),
    ('fees', np.float_),
    ('fixed_fees', np.float_),
    ('slippage', np.float_),
    ('min_size', np.float_),
    ('max_size', np.float_),
    ('reject_prob', np.float_),
    ('close_first', np.bool_),
    ('allow_partial', np.bool_),
    ('raise_reject', np.bool_),
    ('log', np.bool_),
    ('new_cash', np.float_),
    ('new_shares', np.float_),
    ('res_size', np.float_),
    ('res_price', np.float_),
    ('res_fees', np.float_),
    ('res_side', np.int_),
    ('res_status', np.int_),
    ('res_status_info', np.int_),
    ('order_id', np.int_)
]

log_dt = np.dtype(_log_fields, align=True)
"""_"""

__pdoc__['log_dt'] = f"""`np.dtype` of log records.

```plaintext
{json.dumps(dict(zip(
    dict(log_dt.fields).keys(),
    list(map(lambda x: str(x[0]), dict(log_dt.fields).values()))
)), indent=2, default=str)}
```
"""

TradeType = namedtuple('TradeType', [
    'Trade',
    'Position'
])(*range(2))
"""_"""

__pdoc__['TradeType'] = f"""Trade type.

```plaintext
{json.dumps(dict(zip(TradeType._fields, TradeType)), indent=2, default=str)}
```
"""

BenchmarkSize = namedtuple('BenchmarkSize', [
    'InfLong',
    'InfShort',
    'Auto'
])(*range(3))
"""_"""

__pdoc__['BenchmarkSize'] = f"""Benchmark size.

```plaintext
{json.dumps(dict(zip(BenchmarkSize._fields, BenchmarkSize)), indent=2, default=str)}
```

Attributes:
    InfLong: Long shares for all initial cash.
    InfShort: Short shares for all initial cash.
    Auto: Determine the number of shares automatically.
"""
