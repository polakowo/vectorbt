"""Named tuples and enumerated types.

Defines enums and other schemas for `vectorbt.portfolio`."""

import numpy as np

from vectorbt import _typing as tp
from vectorbt.utils.docs import to_doc

__all__ = [
    'RejectedOrderError',
    'InitCashMode',
    'CallSeqType',
    'ConflictMode',
    'SizeType',
    'Direction',
    'OrderStatus',
    'OrderSide',
    'StatusInfo',
    'TradeDirection',
    'TradeStatus',
    'TradeType',
    'ProcessOrderState',
    'ExecuteOrderState',
    'SimulationContext',
    'GroupContext',
    'RowContext',
    'SegmentContext',
    'OrderContext',
    'AfterOrderContext',
    'Order',
    'NoOrder',
    'OrderResult',
    'order_dt',
    'trade_dt',
    'position_dt',
    'log_dt'
]

__pdoc__ = {}


# ############# Errors ############# #


class RejectedOrderError(Exception):
    """Rejected order error."""
    pass


# ############# Enums ############# #


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


class SizeTypeT(tp.NamedTuple):
    Amount: int
    Value: int
    Percent: int
    TargetAmount: int
    TargetValue: int
    TargetPercent: int


SizeType = SizeTypeT(*range(6))
"""_"""

__pdoc__['SizeType'] = f"""Size type.

```json
{to_doc(SizeType)}
```

Attributes:
    Amount: Amount of assets to trade.
    Value: Asset value to trade.
    
        Gets converted into `SizeType.Amount` using `val_price_now`.
    Percent: Percentage of available resources to use in either direction (not to be confused with 
        the percentage of position value!)
    
        * When buying, it's the percentage of `cash_now`. 
        * When selling, it's the percentage of `position_now`.
        * When short selling, it's the percentage of `free_cash_now`.
        * When selling and short selling (i.e. reversing position), it's the percentage of 
        `position_now` and `free_cash_now`.
        
        !!! note
            Takes into account fees and slippage to find the limit.
            In reality, slippage and fees are not known beforehand.
    TargetAmount: Target amount of assets to hold (= target position).
    
        Uses `position_now` to get the current position.
        Gets converted into `SizeType.Amount`.
    TargetValue: Target asset value. 

        Uses `val_price_now` to get the current asset value. 
        Gets converted into `SizeType.TargetAmount`.
    TargetPercent: Target percentage of total value. 

        Uses `value_now` to get the current total value.
        Gets converted into `SizeType.TargetValue`.
"""


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


# ############# Named tuples ############# #


class ProcessOrderState(tp.NamedTuple):
    cash: float
    position: float
    debt: float
    free_cash: float
    val_price: float
    value: float
    oidx: int
    lidx: int


__pdoc__['ProcessOrderState'] = "State before or after order processing."
__pdoc__['ProcessOrderState.cash'] = "Cash in the current column or group with cash sharing."
__pdoc__['ProcessOrderState.position'] = "Position in the current column."
__pdoc__['ProcessOrderState.debt'] = "Debt from shorting in the current column."
__pdoc__['ProcessOrderState.free_cash'] = "Free cash in the current column or group with cash sharing."
__pdoc__['ProcessOrderState.val_price'] = "Valuation price in the current column."
__pdoc__['ProcessOrderState.value'] = "Value in the current column or group with cash sharing."
__pdoc__['ProcessOrderState.oidx'] = "Index of order record."
__pdoc__['ProcessOrderState.lidx'] = "Index of log record."


class ExecuteOrderState(tp.NamedTuple):
    cash: float
    position: float
    debt: float
    free_cash: float


__pdoc__['ExecuteOrderState'] = "State after order execution."
__pdoc__['ExecuteOrderState.cash'] = "See `ProcessOrderState.cash`."
__pdoc__['ExecuteOrderState.position'] = "See `ProcessOrderState.position`."
__pdoc__['ExecuteOrderState.debt'] = "See `ProcessOrderState.debt`."
__pdoc__['ExecuteOrderState.free_cash'] = "See `ProcessOrderState.free_cash`."


class SimulationContext(tp.NamedTuple):
    target_shape: tp.Shape
    close: tp.Array2d
    group_lens: tp.Array1d
    init_cash: tp.Array1d
    cash_sharing: bool
    call_seq: tp.Array2d
    active_mask: tp.Array2d
    ffill_val_price: bool
    update_value: bool
    order_records: tp.RecordArray
    log_records: tp.RecordArray
    last_cash: tp.Array1d
    last_position: tp.Array1d
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
__pdoc__['SimulationContext.close'] = """Last asset price at each time step.

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

Consider three columns, each having $100 of starting capital. If we built one group of two columns
with cash sharing and one (imaginary) group with the last column, the `init_cash` would be 
`np.array([200, 100])`. Without cash sharing, the `init_cash` would be `np.array([100, 100, 100])`.
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
__pdoc__['SimulationContext.ffill_val_price'] = """Whether to track valuation price only if it's known.

Otherwise, unknown `close` will lead to NaN in valuation price at the next timestamp."""
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
__pdoc__['SimulationContext.last_position'] = """Last position per column.

Has shape `(target_shape[1],)`.

In `order_func_nb` and `after_order_func_nb`, has the same value as `position_now`.

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

Gets multiplied by the current position to get the value of the column.
The value of each column in a group with cash sharing is summed to get the value of the entire group.

Defaults to the previous `close` right before `segment_prep_func_nb`, but only if it's not NaN.
For example, close of `[1, 2, np.nan, np.nan, 5]` yields valuation price of `[1, 2, 2, 2, 5]`.

You can use `segment_prep_func_nb` to override `last_val_price` in-place.
You are not allowed to use `-np.inf` or `np.inf`.
The valuation then happens right after `segment_prep_func_nb`.
If `update_value`, gets also updated right after `order_func_nb`.

!!! note
    Since the previous `close` is NaN in the first row, the first `last_val_price` is also NaN.

## Example

Consider 10 units in column 1 and 20 units in column 2. The previous close of them is
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
    ffill_val_price: bool
    update_value: bool
    order_records: tp.RecordArray
    log_records: tp.RecordArray
    last_cash: tp.Array1d
    last_position: tp.Array1d
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
    ffill_val_price: bool
    update_value: bool
    order_records: tp.RecordArray
    log_records: tp.RecordArray
    last_cash: tp.Array1d
    last_position: tp.Array1d
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
    ffill_val_price: bool
    update_value: bool
    order_records: tp.RecordArray
    log_records: tp.RecordArray
    last_cash: tp.Array1d
    last_position: tp.Array1d
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
    ffill_val_price: bool
    update_value: bool
    order_records: tp.RecordArray
    log_records: tp.RecordArray
    last_cash: tp.Array1d
    last_position: tp.Array1d
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
    position_now: float
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
__pdoc__['OrderContext.position_now'] = """Position in the current column.

Scalar value. Has the same value as `last_position` for the current column.
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
    ffill_val_price: bool
    update_value: bool
    order_records: tp.RecordArray
    log_records: tp.RecordArray
    last_cash: tp.Array1d
    last_position: tp.Array1d
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
    position_before: float
    debt_before: float
    free_cash_before: float
    val_price_before: float
    value_before: float
    order_result: "OrderResult"
    cash_now: float
    position_now: float
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
__pdoc__['AfterOrderContext.position_before'] = "`OrderContext.position_now` before execution."
__pdoc__['AfterOrderContext.debt_before'] = "`OrderContext.debt_now` before execution."
__pdoc__['AfterOrderContext.free_cash_before'] = "`OrderContext.free_cash_now` before execution."
__pdoc__['AfterOrderContext.val_price_before'] = "`OrderContext.val_price_now` before execution."
__pdoc__['AfterOrderContext.value_before'] = "`OrderContext.value_now` before execution."
__pdoc__['AfterOrderContext.order_result'] = """Order result of type `OrderResult`.

Can be used to check whether the order has been filled, ignored, or rejected.
"""
__pdoc__['AfterOrderContext.cash_now'] = "`OrderContext.cash_now` after execution."
__pdoc__['AfterOrderContext.position_now'] = "`OrderContext.position_now` after execution."
__pdoc__['AfterOrderContext.val_price_now'] = """`OrderContext.val_price_now` after execution.

If `update_value`, gets replaced with the fill price, as it becomes the most recently known price.
Otherwise, stays the same.
"""
__pdoc__['AfterOrderContext.value_now'] = """`OrderContext.value_now` after execution.

If `update_value`, gets updated with the new cash and value of the column. Otherwise, stays the same.
"""


class Order(tp.NamedTuple):
    size: float = np.inf
    price: float = np.inf
    size_type: int = SizeType.Amount
    direction: int = Direction.All
    fees: float = 0.0
    fixed_fees: float = 0.0
    slippage: float = 0.0
    min_size: float = 0.0
    max_size: float = np.inf
    reject_prob: float = 0.0
    lock_cash: bool = False
    allow_partial: bool = True
    raise_reject: bool = False
    log: bool = False


__pdoc__['Order'] = """A named tuple representing an order.

!!! note
    Currently, Numba has issues with using defaults when filling named tuples. 
    Use `vectorbt.portfolio.nb.order_nb` to create an order."""
__pdoc__['Order.size'] = """Size in units.

Behavior depends upon `Order.size_type` and `Order.direction`.

For any fixed size:

* Set to any number to buy/sell some fixed amount or value.
    Longs are limited by the current cash balance, while shorts are only limited if `Order.lock_cash`.
* Set to `np.inf` to buy for all cash, or `-np.inf` to sell for all free cash.
    If `Order.direction` is not `Direction.All`, `-np.inf` will close the position.
* Set to `np.nan` or 0 to skip.

For any target size:

* Set to any number to buy/sell an amount relative to the current position or value.
* Set to 0 to close the current position.
* Set to `np.nan` to skip.
"""
__pdoc__['Order.price'] = """Price per unit. 

Final price will depend upon slippage.

* If `-np.inf`, replaced by the previous close (~ the current open).
* If `np.inf`, replaced by the current close.

!!! note
    Make sure to use timestamps that come between (and ideally not including) the current open and close."""
__pdoc__['Order.size_type'] = "See `SizeType`."
__pdoc__['Order.direction'] = "See `Direction`."
__pdoc__['Order.fees'] = """Fees in percentage of the order value. 

Note that 0.01 = 1%."""
__pdoc__['Order.fixed_fees'] = "Fixed amount of fees to pay for this order."
__pdoc__['Order.slippage'] = """Slippage in percentage of `Order.price`. 

Note that 0.01 = 1%."""
__pdoc__['Order.min_size'] = """Minimum size in both directions. 

Lower than that will be rejected."""
__pdoc__['Order.max_size'] = """Maximum size in both directions. 

Higher than that will be partly filled."""
__pdoc__['Order.reject_prob'] = """Probability of rejecting this order to simulate a random rejection event.

Not everything goes smoothly in real life. Use random rejections to test your order management for robustness."""
__pdoc__['Order.lock_cash'] = """Whether to lock cash when shorting. 

Keeps free cash from turning negative."""
__pdoc__['Order.allow_partial'] = """Whether to allow partial fill.

Otherwise, the order gets rejected.

Does not apply when `Order.size` is `np.inf`."""
__pdoc__['Order.raise_reject'] = """Whether to raise exception if order has been rejected.

Terminates the simulation."""
__pdoc__['Order.log'] = """Whether to log this order by filling a log record. 

Remember to increase `max_logs`."""

NoOrder = Order(
    np.nan,
    np.nan,
    -1,
    -1,
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


class OrderResult(tp.NamedTuple):
    size: float
    price: float
    fees: float
    side: int
    status: int
    status_info: int


__pdoc__['OrderResult'] = "A named tuple representing an order result."
__pdoc__['OrderResult.size'] = "Filled size."
__pdoc__['OrderResult.price'] = "Filled price per unit, adjusted with slippage."
__pdoc__['OrderResult.fees'] = "Total fees paid for this order."
__pdoc__['OrderResult.side'] = "See `OrderSide`."
__pdoc__['OrderResult.status'] = "See `OrderStatus`."
__pdoc__['OrderResult.status_info'] = "See `StatusInfo`."

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
    ('position', np.float_),
    ('debt', np.float_),
    ('free_cash', np.float_),
    ('val_price', np.float_),
    ('value', np.float_),
    ('size', np.float_),
    ('price', np.float_),
    ('size_type', np.int_),
    ('direction', np.int_),
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
    ('new_position', np.float_),
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
