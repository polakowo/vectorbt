"""Named tuples and enumerated types.

Defines enums and other schemas for `vectorbt.portfolio`."""

import numpy as np

from vectorbt import _typing as tp
from vectorbt.utils.docs import to_doc

__all__ = [
    'SimulationContext',
    'GroupContext',
    'RowContext',
    'SegmentContext',
    'OrderContext',
    'AfterOrderContext',
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
    'ProcessOrderState',
    'ExecuteOrderState',
    'Direction',
    'order_dt',
    'TradeDirection',
    'TradeStatus',
    'trade_dt',
    'position_dt',
    'log_dt',
    'TradeType'
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
    update_value: bool
    order_records: tp.RecordArray
    log_records: tp.RecordArray
    last_cash: tp.Array1d
    last_shares: tp.Array1d
    last_debt: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_oidx: tp.Array1d
    last_lidx: tp.Array1d


__pdoc__['SimulationContext'] = """A named tuple representing the context of a simulation.

Contains general information available to all other contexts.

Passed to `prep_func_nb`."""
__pdoc__['SimulationContext.target_shape'] = """Target shape of the simulation.

A tuple with exactly two elements: the number of rows and columns.

## Example

One day of minute data for three assets would yield a `target_shape` of `(1440, 3)`,
where the first axis are rows (minutes) and the second axis are columns (assets).
"""
__pdoc__['SimulationContext.close'] = """Reference price, such as close price.

Reference price should be the last price known in each row.

Has shape `target_shape`.
"""
__pdoc__['SimulationContext.group_lens'] = """Number of columns per each group.

Even if columns are not grouped, `group_lens` contains ones - one column per group.

## Example

In pairs trading, `group_lens` would be `np.array([2])`, while three independent
columns would require `group_lens` of `np.array([1, 1, 1])`.
"""
__pdoc__['SimulationContext.init_cash'] = """Initial capital per column or group with cash sharing.

If `cash_sharing`, has shape `(group_lens.shape[0],)`, otherwise has shape `(target_shape[1],)`.

## Example

Consider a group of two columns sharing $100 and an another column with $200.
The `init_cash` would then be `np.array([100, 200])`. Without cash sharing, 
the `init_cash` would be `np.array([100, 100, 200])`.
"""
__pdoc__['SimulationContext.cash_sharing'] = "Whether cash sharing is enabled."
__pdoc__['SimulationContext.call_seq'] = """Default sequence of calls per segment.

Controls the sequence in which `order_func_nb` is executed within each segment.

Has shape `target_shape` and each value must exist in the range `[0, group_len)`.

!!! note
    To change the call sequence dynamically, better change `call_seq_now` in-place.
    
## Example

The default call sequence for three data points and two groups with three columns each:

```python
np.array([
    [0, 1, 2, 0, 1, 2],
    [0, 1, 2, 0, 1, 2],
    [0, 1, 2, 0, 1, 2]
])
```
"""
__pdoc__['SimulationContext.active_mask'] = """Mask of whether a particular segment is executed.

A segment is simply a sequence of `order_func_nb` calls within a group and row.
If a segment is not active, `segment_prep_func_nb`, `order_prep_func_nb` and `after_order_prep_func_nb`
are not executed for this group and row.

You can change this mask in-place to dynamically disable future segments.

Has shape `(target_shape[0], group_lens.shape[0])`.

## Example

Consider two groups with two columns each and the following activity mask:

```python
array([[ True, False],
       [False,  True]])
```

Only the first group is executed in the first row and only the second group is executed
in the second row.
"""
__pdoc__['SimulationContext.update_value'] = "Whether to update group value after each filled order."
__pdoc__['SimulationContext.order_records'] = """Order records.

It's a 1-dimensional array with records of type `order_dt`.

The array is initialized with empty records first (they contain random data), and then 
gradually filled with order data. The number of initialized records depends upon `max_orders`, 
but usually it's `target_shape[0] * target_shape[1]`, meaning there is maximal one order record per element.
`max_orders` can be chosen lower if not every `order_func_nb` leads to a filled order, to save memory.

You can use `last_oidx` to get the index of the last filled order of each column.

## Example

Before filling, each order record looks like this:

```python
array([(-8070450532247928832, -8070450532247928832, 4, 0., 0., 0., 5764616306889786413)]
```

After filling, it becomes like this:

```python
array([(0, 0, 1, 50., 1., 0., 1)]
```
"""
__pdoc__['SimulationContext.log_records'] = """Log records.

Similar to `SimulationContext.order_records` but of type `log_dt` and index `last_lidx`."""
__pdoc__['SimulationContext.last_cash'] = """Last cash per column or group with cash sharing.

Has the same shape as `init_cash`.

In `order_func_nb` and `after_order_func_nb`, has the same value as `cash_now`.

Gets updated right after `order_func_nb`.
"""
__pdoc__['SimulationContext.last_shares'] = """Last shares per column.

Has shape `(target_shape[1],)`.

In `order_func_nb` and `after_order_func_nb`, has the same value as `shares_now`.

Gets updated right after `order_func_nb`.
"""
__pdoc__['SimulationContext.last_debt'] = """Last debt from shorting per column.

Debt is the total value from shorting that hasn't been covered yet. Used to update `free_cash_now`.

Has shape `(target_shape[1],)`. 

Gets updated right after `order_func_nb`.
"""
__pdoc__['SimulationContext.last_free_cash'] = """Last free cash per column or group with cash sharing.

Free cash never goes above the initial level, because an operation always costs money.

Has shape `(target_shape[1],)`. 

Gets updated right after `order_func_nb`.
"""
__pdoc__['SimulationContext.last_val_price'] = """Last valuation price per column.

Has shape `(target_shape[1],)`.

Enables `SizeType.TargetValue` and `SizeType.TargetPercent`.

Gets multiplied by the number of shares to get the current value of the column.
The value of each column in a group with cash sharing is summed to get the value of the entire group.

Defaults to the previous `close` right before `segment_prep_func_nb`.
You can use `segment_prep_func_nb` to override `last_val_price` in-place.
The valuation then happens right after `segment_prep_func_nb`.
If `update_value`, gets also updated right after `order_func_nb`.

!!! note
    Since the previous `close` is NaN in the first row, the first `last_val_price` is also NaN.

## Example

Consider 10 shares in column 1 and 20 shares in column 2. The previous close of them is
$40 and $50 respectively, which is also the default valuation price in the current row,
available as `last_val_price` in `segment_prep_func_nb`. If both columns are in the same group 
with cash sharing, the group is valued at $1400 before any `order_func_nb` is called, and can 
be later accessed via `OrderContext.value_now`.

"""
__pdoc__['SimulationContext.last_value'] = """Last value per column or group with cash sharing.

Has the same shape as `init_cash`.

Gets updated using `last_val_price` right after `segment_prep_func_nb`.
If `update_value`, gets also updated right after `order_func_nb`.
"""
__pdoc__['SimulationContext.last_oidx'] = """Index of the last order record of each column.

Points to `order_records` and has shape `(target_shape[1],)`.

## Example

`last_oidx` of `np.array([1, 100, -1])` means the last filled order is `order_records[1]` for the
first column, `order_records[100]` for the second column, and no orders have been filled yet
for the third column.
"""
__pdoc__['SimulationContext.last_lidx'] = """Index of the last log record of each column.

Similar to `last_oidx` but for log records.
"""


class GroupContext(tp.NamedTuple):
    target_shape: tp.Shape
    close: tp.Array2d
    group_lens: tp.Array1d
    init_cash: tp.Array1d
    cash_sharing: bool
    call_seq: tp.Array2d
    active_mask: tp.Array2d
    update_value: bool
    order_records: tp.RecordArray
    log_records: tp.RecordArray
    last_cash: tp.Array1d
    last_shares: tp.Array1d
    last_debt: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_oidx: tp.Array1d
    last_lidx: tp.Array1d
    group: int
    group_len: int
    from_col: int
    to_col: int


__pdoc__['GroupContext'] = """A named tuple representing the context of a group.

A group is a set of nearby columns that are somehow related (for example, by sharing the same capital).
In each row, the columns under the same group are bound to the same segment.

Contains all fields from `SimulationContext` plus fields describing the current group.

Passed to `group_prep_func_nb`.

## Example

Consider a group of three columns, a group of two columns, and one more column:

| group | group_len | from_col | to_col |
| ----- | --------- | -------- | ------ |
| 0     | 3         | 0        | 3      |
| 1     | 2         | 3        | 5      |
| 2     | 1         | 5        | 6      |
"""
for field in GroupContext._fields:
    if field in SimulationContext._fields:
        __pdoc__['GroupContext.' + field] = f"See `SimulationContext.{field}`."
__pdoc__['GroupContext.group'] = """Index of the current group.

Has range `[0, group_lens.shape[0])`.
"""
__pdoc__['GroupContext.group_len'] = """Number of columns in the current group.

Scalar value. Same as `group_lens[group]`.
"""
__pdoc__['GroupContext.from_col'] = """Index of the first column in the current group.

Has range `[0, target_shape[1])`.
"""
__pdoc__['GroupContext.to_col'] = """Index of the last column in the current group plus one.

Has range `[1, target_shape[1] + 1)`. 

If columns are not grouped, equals to `from_col + 1`.

!!! warning
    In the last group, `to_col` points at a column that doesn't exist.
"""


class RowContext(tp.NamedTuple):
    target_shape: tp.Shape
    close: tp.Array2d
    group_lens: tp.Array1d
    init_cash: tp.Array1d
    cash_sharing: bool
    call_seq: tp.Array2d
    active_mask: tp.Array2d
    update_value: bool
    order_records: tp.RecordArray
    log_records: tp.RecordArray
    last_cash: tp.Array1d
    last_shares: tp.Array1d
    last_debt: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_oidx: tp.Array1d
    last_lidx: tp.Array1d
    i: int


__pdoc__['RowContext'] = """A named tuple representing the context of a row.

A row is a time step in which segments are executed.

Contains all fields from `SimulationContext` plus fields describing the current row.

Passed to `row_prep_func_nb`.
"""
for field in RowContext._fields:
    if field in SimulationContext._fields:
        __pdoc__['RowContext.' + field] = f"See `SimulationContext.{field}`."
__pdoc__['RowContext.i'] = """Index of the current row.

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
    update_value: bool
    order_records: tp.RecordArray
    log_records: tp.RecordArray
    last_cash: tp.Array1d
    last_shares: tp.Array1d
    last_debt: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_oidx: tp.Array1d
    last_lidx: tp.Array1d
    group: int
    group_len: int
    from_col: int
    to_col: int
    i: int
    call_seq_now: tp.Array1d


__pdoc__['SegmentContext'] = """A named tuple representing the context of a segment.

A segment is an intersection between groups and rows. It's an entity that defines
how and in which order elements within the same group and row are processed.

Contains all fields from `SimulationContext`, `GroupContext`, and `RowContext`, plus fields 
describing the current segment.

Passed to `segment_prep_func_nb`.
"""
for field in SegmentContext._fields:
    if field in SimulationContext._fields:
        __pdoc__['SegmentContext.' + field] = f"See `SimulationContext.{field}`."
    elif field in GroupContext._fields:
        __pdoc__['SegmentContext.' + field] = f"See `GroupContext.{field}`."
    elif field in RowContext._fields:
        __pdoc__['SegmentContext.' + field] = f"See `RowContext.{field}`."
__pdoc__['SegmentContext.call_seq_now'] = """Sequence of calls within the current segment.

Has shape `(group_len,)`. 

Each value in this sequence should indicate the position of column in the group to
call next. Processing goes always from left to right.

You can use `segment_prep_func_nb` to override `call_seq_now`.
    
## Example

`[2, 0, 1]` would first call column 2, then 0, and finally 1.
"""


class OrderContext(tp.NamedTuple):
    target_shape: tp.Shape
    close: tp.Array2d
    group_lens: tp.Array1d
    init_cash: tp.Array1d
    cash_sharing: bool
    call_seq: tp.Array2d
    active_mask: tp.Array2d
    update_value: bool
    order_records: tp.RecordArray
    log_records: tp.RecordArray
    last_cash: tp.Array1d
    last_shares: tp.Array1d
    last_debt: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_oidx: tp.Array1d
    last_lidx: tp.Array1d
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
    debt_now: float
    free_cash_now: float
    val_price_now: float
    value_now: float


__pdoc__['OrderContext'] = """A named tuple representing the context of an order.

Contains all fields from `SegmentContext` plus fields describing the current state.

Passed to `order_func_nb`.
"""
for field in OrderContext._fields:
    if field in SimulationContext._fields:
        __pdoc__['OrderContext.' + field] = f"See `SimulationContext.{field}`."
    elif field in GroupContext._fields:
        __pdoc__['OrderContext.' + field] = f"See `GroupContext.{field}`."
    elif field in RowContext._fields:
        __pdoc__['OrderContext.' + field] = f"See `RowContext.{field}`."
    elif field in SegmentContext._fields:
        __pdoc__['OrderContext.' + field] = f"See `SegmentContext.{field}`."
__pdoc__['OrderContext.col'] = """Current column.

Has range `[0, target_shape[1])` and is always within `[from_col, to_col)`.
"""
__pdoc__['OrderContext.call_idx'] = """Index of the current call in `call_seq_now`.

Has range `[0, group_len)`.
"""
__pdoc__['OrderContext.cash_now'] = """Cash in the current column or group with cash sharing.

Scalar value. Has the same value as `last_cash` for the current column/group.
"""
__pdoc__['OrderContext.shares_now'] = """Shares in the current column.

Scalar value. Has the same value as `last_shares` for the current column.
"""
__pdoc__['OrderContext.debt_now'] = """Debt from shorting in the current column.

Scalar value. Has the same value as `last_debt` for the current column.
"""
__pdoc__['OrderContext.free_cash_now'] = """Free cash in the current column or group with cash sharing.

Scalar value. Has the same value as `last_free_cash` for the current column/group.
"""
__pdoc__['OrderContext.val_price_now'] = """Valuation price in the current column.

Scalar value. Has the same value as `last_val_price` for the current column.
"""
__pdoc__['OrderContext.value_now'] = """Value in the current column or group with cash sharing.

Scalar value. Has the same value as `last_value` for the current column/group.
"""


class AfterOrderContext(tp.NamedTuple):
    target_shape: tp.Shape
    close: tp.Array2d
    group_lens: tp.Array1d
    init_cash: tp.Array1d
    cash_sharing: bool
    call_seq: tp.Array2d
    active_mask: tp.Array2d
    update_value: bool
    order_records: tp.RecordArray
    log_records: tp.RecordArray
    last_cash: tp.Array1d
    last_shares: tp.Array1d
    last_debt: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    last_oidx: tp.Array1d
    last_lidx: tp.Array1d
    group: int
    group_len: int
    from_col: int
    to_col: int
    i: int
    call_seq_now: tp.Array1d
    col: int
    call_idx: int
    cash_before: float
    shares_before: float
    debt_before: float
    free_cash_before: float
    val_price_before: float
    value_before: float
    order_result: "OrderResult"
    cash_now: float
    shares_now: float
    debt_now: float
    free_cash_now: float
    val_price_now: float
    value_now: float


__pdoc__['AfterOrderContext'] = """A named tuple representing the context of an order.

Contains all fields from `OrderContext` plus fields describing the order result and the previous state.

Passed to `after_order_func_nb`.
"""
for field in AfterOrderContext._fields:
    if field in SimulationContext._fields:
        __pdoc__['AfterOrderContext.' + field] = f"See `SimulationContext.{field}`."
    elif field in GroupContext._fields:
        __pdoc__['AfterOrderContext.' + field] = f"See `GroupContext.{field}`."
    elif field in RowContext._fields:
        __pdoc__['AfterOrderContext.' + field] = f"See `RowContext.{field}`."
    elif field in SegmentContext._fields:
        __pdoc__['AfterOrderContext.' + field] = f"See `SegmentContext.{field}`."
    elif field in OrderContext._fields:
        __pdoc__['AfterOrderContext.' + field] = f"See `OrderContext.{field}`."
__pdoc__['AfterOrderContext.cash_before'] = "`OrderContext.cash_now` before execution."
__pdoc__['AfterOrderContext.shares_before'] = "`OrderContext.shares_now` before execution."
__pdoc__['AfterOrderContext.debt_before'] = "`OrderContext.debt_now` before execution."
__pdoc__['AfterOrderContext.free_cash_before'] = "`OrderContext.free_cash_now` before execution."
__pdoc__['AfterOrderContext.val_price_before'] = "`OrderContext.val_price_now` before execution."
__pdoc__['AfterOrderContext.value_before'] = "`OrderContext.value_now` before execution."
__pdoc__['AfterOrderContext.order_result'] = """Order result of type `OrderResult`.

Can be used to check whether the order has been filled, ignored, or rejected.
"""
__pdoc__['AfterOrderContext.cash_now'] = "`OrderContext.cash_now` after execution."
__pdoc__['AfterOrderContext.shares_now'] = "`OrderContext.shares_now` after execution."
__pdoc__['AfterOrderContext.val_price_now'] = """`OrderContext.val_price_now` after execution.

If `update_value`, gets replaced with the fill price, as it becomes the most recently known price.
Otherwise, stays the same.
"""
__pdoc__['AfterOrderContext.value_now'] = """`OrderContext.value_now` after execution.

If `update_value`, gets updated with the new cash and value of the column. Otherwise, stays the same.
"""


class InitCashModeT(tp.NamedTuple):
    Auto: int
    AutoAlign: int


InitCashMode = InitCashModeT(*range(2))
"""_"""

__pdoc__['InitCashMode'] = f"""Initial cash mode.

```json
{to_doc(InitCashMode)}
```

Attributes:
    Auto: Initial cash is infinite within simulation, and then set to the total cash spent.
    AutoAlign: Initial cash is set to the total cash spent across all columns.
"""


class CallSeqTypeT(tp.NamedTuple):
    Default: int
    Reversed: int
    Random: int
    Auto: int


CallSeqType = CallSeqTypeT(*range(4))
"""_"""

__pdoc__['CallSeqType'] = f"""Call sequence type.

```json
{to_doc(CallSeqType)}
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

```json
{to_doc(SizeType)}
```

Attributes:
    Shares: Number of shares to buy or sell.
    TargetShares: Target number of shares.
    
        Uses `shares_now` to get the current number of shares.
        Gets converted into `SizeType.Shares`.
    TargetValue: Target holding value. 
    
        Uses `val_price_now` to get the current holding value. 
        Gets converted into `SizeType.TargetShares`.
    TargetPercent: Target percentage of total value. 
    
        Uses `value_now` to get the current total value.
        Gets converted into `SizeType.TargetValue`.
    Percent: Percentage of available resources in either direction to use.
    
        When buying, it's the percentage of `cash_now`. 
        When selling, it's the percentage of `shares_now`.
        When short selling, it's the percentage of the remaining free cash.
        When selling and short selling (reversing position), it's the percentage of both.
"""


class ConflictModeT(tp.NamedTuple):
    Ignore: int
    Entry: int
    Exit: int
    Opposite: int


ConflictMode = ConflictModeT(*range(4))
"""_"""

__pdoc__['ConflictMode'] = f"""Conflict mode.

```json
{to_doc(ConflictMode)}
```

What should happen if both entry and exit signals occur simultaneously?

Attributes:
    Ignore: Ignore both signals.
    Entry: Execute entry signal.
    Exit: Execute exit signal.
    Opposite: Execute opposite signal. Takes effect only when in position.
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
    lock_cash: bool
    allow_partial: bool
    raise_reject: bool
    log: bool


__pdoc__['Order'] = "A named tuple representing an order."
__pdoc__['Order.size'] = "Size in shares."
__pdoc__['Order.size_type'] = "See `SizeType`."
__pdoc__['Order.direction'] = "See `Direction`."
__pdoc__['Order.price'] = "Price per share. Final price will depend upon slippage."
__pdoc__['Order.fees'] = "Fees in percentage of the order value."
__pdoc__['Order.fixed_fees'] = "Fixed amount of fees to pay for this order."
__pdoc__['Order.slippage'] = "Slippage in percentage of `price`."
__pdoc__['Order.min_size'] = "Minimum size in both directions. Lower than that will be rejected."
__pdoc__['Order.max_size'] = "Maximum size in both directions. Higher than that will be partly filled."
__pdoc__['Order.reject_prob'] = "Probability of rejecting this order to simulate a random rejection event."
__pdoc__['Order.lock_cash'] = "Whether to lock cash when shorting. Keeps free cash from turning negative."
__pdoc__['Order.allow_partial'] = "Whether to allow partial fill."
__pdoc__['Order.raise_reject'] = "Whether to raise exception if order has been rejected."
__pdoc__['Order.log'] = "Whether to log this order by filling a log record. Remember to increase `max_logs`."

NoOrder = Order(
    np.nan,
    -1,
    -1,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    False,
    False,
    False,
    False
)
"""_"""

__pdoc__['NoOrder'] = "Order that should not be processed."


class OrderStatusT(tp.NamedTuple):
    Filled: int
    Ignored: int
    Rejected: int


OrderStatus = OrderStatusT(*range(3))
"""_"""

__pdoc__['OrderStatus'] = f"""Order status.

```json
{to_doc(OrderStatus)}
```

Attributes:
    Filled: Order has been filled.
    Ignored: Order has been ignored.
    Rejected: Order has been rejected.
"""


class OrderSideT(tp.NamedTuple):
    Buy: int
    Sell: int


OrderSide = OrderSideT(*range(2))
"""_"""

__pdoc__['OrderSide'] = f"""Order side.

```json
{to_doc(OrderSide)}
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

```json
{to_doc(StatusInfo)}
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

```json
{to_doc(status_info_desc)}
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


class ProcessOrderState(tp.NamedTuple):
    cash: float
    shares: float
    debt: float
    free_cash: float
    val_price: float
    value: float
    oidx: int
    lidx: int


__pdoc__['ProcessOrderState'] = "State before or after order processing."
__pdoc__['ProcessOrderState.cash'] = "Cash in the current column or group with cash sharing."
__pdoc__['ProcessOrderState.shares'] = "Shares in the current column."
__pdoc__['ProcessOrderState.debt'] = "Debt from shorting in the current column."
__pdoc__['ProcessOrderState.free_cash'] = "Free cash in the current column or group with cash sharing."
__pdoc__['ProcessOrderState.val_price'] = "Valuation price in the current column."
__pdoc__['ProcessOrderState.value'] = "Value in the current column or group with cash sharing."
__pdoc__['ProcessOrderState.oidx'] = "Index of order record."
__pdoc__['ProcessOrderState.lidx'] = "Index of log record."


class ExecuteOrderState(tp.NamedTuple):
    cash: float
    shares: float
    debt: float
    free_cash: float


__pdoc__['ExecuteOrderState'] = "State after order execution."
__pdoc__['ExecuteOrderState.cash'] = "See `ProcessOrderState.cash`."
__pdoc__['ExecuteOrderState.shares'] = "See `ProcessOrderState.shares`."
__pdoc__['ExecuteOrderState.debt'] = "See `ProcessOrderState.debt`."
__pdoc__['ExecuteOrderState.free_cash'] = "See `ProcessOrderState.free_cash`."


class DirectionT(tp.NamedTuple):
    LongOnly: int
    ShortOnly: int
    All: int


Direction = DirectionT(*range(3))
"""_"""

__pdoc__['Direction'] = f"""Position direction.

```json
{to_doc(Direction)}
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

```json
{to_doc(order_dt)}
```
"""


class TradeDirectionT(tp.NamedTuple):
    Long: int
    Short: int


TradeDirection = TradeDirectionT(*range(2))
"""_"""

__pdoc__['TradeDirection'] = f"""Event direction.

```json
{to_doc(TradeDirection)}
```
"""


class TradeStatusT(tp.NamedTuple):
    Open: int
    Closed: int


TradeStatus = TradeStatusT(*range(2))
"""_"""

__pdoc__['TradeStatus'] = f"""Event status.

```json
{to_doc(TradeStatus)}
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

```json
{to_doc(trade_dt)}
```
"""

_position_fields = _trade_fields[:-1]

position_dt = np.dtype(_position_fields, align=True)
"""_"""

__pdoc__['position_dt'] = f"""`np.dtype` of position records.

```json
{to_doc(position_dt)}
```
"""

_log_fields = [
    ('id', np.int_),
    ('idx', np.int_),
    ('col', np.int_),
    ('group', np.int_),
    ('cash', np.float_),
    ('shares', np.float_),
    ('debt', np.float_),
    ('free_cash', np.float_),
    ('val_price', np.float_),
    ('value', np.float_),
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
    ('lock_cash', np.bool_),
    ('allow_partial', np.bool_),
    ('raise_reject', np.bool_),
    ('log', np.bool_),
    ('new_cash', np.float_),
    ('new_shares', np.float_),
    ('new_debt', np.float_),
    ('new_free_cash', np.float_),
    ('new_val_price', np.float_),
    ('new_value', np.float_),
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

```json
{to_doc(log_dt)}
```
"""


class TradeTypeT(tp.NamedTuple):
    Trade: int
    Position: int


TradeType = TradeTypeT(*range(2))
"""_"""

__pdoc__['TradeType'] = f"""Trade type.

```json
{to_doc(TradeType)}
```
"""
