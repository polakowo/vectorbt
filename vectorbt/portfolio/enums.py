"""Named tuples and enumerated types.

Defines enums and other schemas for `vectorbt.portfolio`."""

import numpy as np
import json

from vectorbt import typing as tp

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

class SimulationContext(tp.NamedTuple):
    target_shape: tp.Shape
    close: tp.Array2d
    group_lens: tp.Array1d
    init_cash: tp.Array1d
    cash_sharing: bool
    call_seq: tp.Array2d
    active_mask: tp.Array2d
    order_records: tp.RecordArray
    log_records: tp.RecordArray
    last_cash: tp.Array1d
    last_shares: tp.Array1d
    last_val_price: tp.Array1d
    last_lidx: tp.Array1d
    last_ridx: tp.Array1d


__pdoc__['SimulationContext'] = """A named tuple representing context of the simulation.

Contains general information available to all other contexts."""
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

If `cash_sharing` is True, has shape `(group_lens.shape[0],)`. 
Otherwise, has shape `(target_shape[1],)`.
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
__pdoc__['SimulationContext.last_ridx'] = """Index of the last order record.

Has shape `(target_shape[1],)`.
"""
__pdoc__['SimulationContext.last_lidx'] = """Index of the last log record.

Has shape `(target_shape[1],)`.
"""


class GroupContext(tp.NamedTuple):
    target_shape: tp.Shape
    close: tp.Array2d
    group_lens: tp.Array1d
    init_cash: tp.Array1d
    cash_sharing: bool
    call_seq: tp.Array2d
    active_mask: tp.Array2d
    order_records: tp.RecordArray
    log_records: tp.RecordArray
    last_cash: tp.Array1d
    last_shares: tp.Array1d
    last_val_price: tp.Array1d
    last_lidx: tp.Array1d
    last_ridx: tp.Array1d
    group: int
    group_len: int
    from_col: int
    to_col: int


__pdoc__['GroupContext'] = "A named tuple representing context of the group."
for field in GroupContext._fields:
    if field in SimulationContext._fields:
        __pdoc__['GroupContext.' + field] = f"See `SimulationContext.{field}`."
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


class RowContext(tp.NamedTuple):
    target_shape: tp.Shape
    close: tp.Array2d
    group_lens: tp.Array1d
    init_cash: tp.Array1d
    cash_sharing: bool
    call_seq: tp.Array2d
    active_mask: tp.Array2d
    order_records: tp.RecordArray
    log_records: tp.RecordArray
    last_cash: tp.Array1d
    last_shares: tp.Array1d
    last_val_price: tp.Array1d
    last_lidx: tp.Array1d
    last_ridx: tp.Array1d
    i: int


__pdoc__['RowContext'] = "A named tuple representing context of the row."
for field in RowContext._fields:
    if field in SimulationContext._fields:
        __pdoc__['RowContext.' + field] = f"See `SimulationContext.{field}`."
__pdoc__['RowContext.i'] = """Current row (time axis).

Has range `[0, target_shape[0])`.
"""


class SegmentContext(tp.NamedTuple):
    target_shape: tp.Shape
    close: tp.Array2d
    group_lens: tp.Array1d
    init_cash: tp.Array1d
    cash_sharing: bool
    call_seq: tp.Array2d
    active_mask: tp.Array2d
    order_records: tp.RecordArray
    log_records: tp.RecordArray
    last_cash: tp.Array1d
    last_shares: tp.Array1d
    last_val_price: tp.Array1d
    last_lidx: tp.Array1d
    last_ridx: tp.Array1d
    group: int
    group_len: int
    from_col: int
    to_col: int
    i: int
    call_seq_now: tp.Array1d


__pdoc__['SegmentContext'] = "A named tuple representing context of the segment."
for field in SegmentContext._fields:
    if field in SimulationContext._fields:
        __pdoc__['SegmentContext.' + field] = f"See `SimulationContext.{field}`."
    elif field in GroupContext._fields:
        __pdoc__['SegmentContext.' + field] = f"See `GroupContext.{field}`."
    elif field in RowContext._fields:
        __pdoc__['SegmentContext.' + field] = f"See `RowContext.{field}`."
__pdoc__['SegmentContext.call_seq_now'] = """Current sequence of calls.

Has shape `(group_len,)`. 
"""


class OrderContext(tp.NamedTuple):
    target_shape: tp.Shape
    close: tp.Array2d
    group_lens: tp.Array1d
    init_cash: tp.Array1d
    cash_sharing: bool
    call_seq: tp.Array2d
    active_mask: tp.Array2d
    order_records: tp.RecordArray
    log_records: tp.RecordArray
    last_cash: tp.Array1d
    last_shares: tp.Array1d
    last_val_price: tp.Array1d
    last_lidx: tp.Array1d
    last_ridx: tp.Array1d
    group: int
    group_len: int
    from_col: int
    to_col: int
    i: int
    call_seq_now: tp.Array1d
    col: int
    call_idx: int
    cash_now: float
    shares_now: float
    val_price_now: float
    value_now: float


__pdoc__['OrderContext'] = "A named tuple representing context of the order."
for field in SegmentContext._fields:
    if field in SimulationContext._fields:
        __pdoc__['OrderContext.' + field] = f"See `SimulationContext.{field}`."
    elif field in GroupContext._fields:
        __pdoc__['OrderContext.' + field] = f"See `GroupContext.{field}`."
    elif field in RowContext._fields:
        __pdoc__['OrderContext.' + field] = f"See `RowContext.{field}`."
    elif field in SegmentContext._fields:
        __pdoc__['OrderContext.' + field] = f"See `SegmentContext.{field}`."
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


class InitCashModeT(tp.NamedTuple):
    Auto: int
    AutoAlign: int


InitCashMode = InitCashModeT(*range(2))
"""_"""

__pdoc__['InitCashMode'] = f"""Initial cash mode.

```plaintext
{json.dumps(dict(zip(InitCashMode._fields, InitCashMode)), indent=2, default=str)}
```

Attributes:
    Auto: Optimal initial cash for each column.
    AutoAlign: Optimal initial cash aligned across all columns.
"""


class CallSeqTypeT(tp.NamedTuple):
    Default: int
    Reversed: int
    Random: int
    Auto: int


CallSeqType = CallSeqTypeT(*range(4))
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


class SizeTypeT(tp.NamedTuple):
    Shares: int
    TargetShares: int
    TargetValue: int
    TargetPercent: int
    Percent: int


SizeType = SizeTypeT(*range(5))
"""_"""

__pdoc__['SizeType'] = f"""Size type.

```plaintext
{json.dumps(dict(zip(SizeType._fields, SizeType)), indent=2, default=str)}
```

Attributes:
    Shares: Number of shares.
    TargetShares: Target number of shares.
    TargetValue: Target total value of holdings.
    TargetPercent: Target percentage of total value.
    Percent: Percentage of available cash (if buy) or shares (if sell).
"""


class ConflictModeT(tp.NamedTuple):
    Ignore: int
    Entry: int
    Exit: int
    Opposite: int


ConflictMode = ConflictModeT(*range(4))
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


class Order(tp.NamedTuple):
    size: float
    size_type: int
    direction: int
    price: float
    fees: float
    fixed_fees: float
    slippage: float
    min_size: float
    max_size: float
    reject_prob: float
    allow_partial: bool
    raise_reject: bool
    log: bool


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
__pdoc__['Order.allow_partial'] = "Whether to allow partial fill."
__pdoc__['Order.raise_reject'] = "Whether to raise exception if order has been rejected."
__pdoc__['Order.log'] = "Whether to log this order by filling a log record."

NoOrder = Order(np.nan, -1, -1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, False, False)
"""_"""

__pdoc__['NoOrder'] = "Order that will not be processed."


class OrderStatusT(tp.NamedTuple):
    Filled: int
    Ignored: int
    Rejected: int


OrderStatus = OrderStatusT(*range(3))
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


class OrderSideT(tp.NamedTuple):
    Buy: int
    Sell: int


OrderSide = OrderSideT(*range(2))
"""_"""

__pdoc__['OrderSide'] = f"""Order side.

```plaintext
{json.dumps(dict(zip(OrderSide._fields, OrderSide)), indent=2, default=str)}
```
"""


class StatusInfoT(tp.NamedTuple):
    SizeNaN: int
    PriceNaN: int
    ValPriceNaN: int
    ValueNaN: int
    ValueZeroNeg: int
    SizeZero: int
    NoCashShort: int
    NoCashLong: int
    NoOpenPosition: int
    MaxSizeExceeded: int
    RandomEvent: int
    CantCoverFees: int
    MinSizeNotReached: int
    PartialFill: int


StatusInfo = StatusInfoT(*range(14))
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


class OrderResult(tp.NamedTuple):
    size: float
    price: float
    fees: float
    side: int
    status: int
    status_info: int


__pdoc__['OrderResult'] = "A named tuple representing an order result."
__pdoc__['OrderResult.size'] = "Filled size in shares."
__pdoc__['OrderResult.price'] = "Filled price per share, adjusted with slippage."
__pdoc__['OrderResult.fees'] = "Total fees paid for this order."
__pdoc__['OrderResult.side'] = "See `OrderSide`."
__pdoc__['OrderResult.status'] = "See `OrderStatus`."
__pdoc__['OrderResult.status_info'] = "See `StatusInfo`."


class RejectedOrderError(Exception):
    """Rejected order error."""
    pass


class DirectionT(tp.NamedTuple):
    LongOnly: int
    ShortOnly: int
    All: int


Direction = DirectionT(*range(3))
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


class TradeDirectionT(tp.NamedTuple):
    Long: int
    Short: int


TradeDirection = TradeDirectionT(*range(2))
"""_"""

__pdoc__['TradeDirection'] = f"""Event direction.

```plaintext
{json.dumps(dict(zip(TradeDirection._fields, TradeDirection)), indent=2, default=str)}
```
"""


class TradeStatusT(tp.NamedTuple):
    Open: int
    Closed: int


TradeStatus = TradeStatusT(*range(2))
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


class TradeTypeT(tp.NamedTuple):
    Trade: int
    Position: int


TradeType = TradeTypeT(*range(2))
"""_"""

__pdoc__['TradeType'] = f"""Trade type.

```plaintext
{json.dumps(dict(zip(TradeType._fields, TradeType)), indent=2, default=str)}
```
"""


class BenchmarkSizeT(tp.NamedTuple):
    InfLong: int
    InfShort: int
    Auto: int


BenchmarkSize = BenchmarkSizeT(*range(3))
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
