"""Named tuples and enumerated types."""

import numpy as np
from collections import namedtuple
import json

__pdoc__ = {}

# We use namedtuple for enums and classes to be able to use them in Numba

# ############# SimulationContext ############# #

SimulationContext = namedtuple('SimulationContext', [
    'target_shape',
    'close',
    'group_counts',
    'init_cash',
    'cash_sharing',
    'call_seq',
    'active_mask',
    'min_size',
    'order_records',
    'record_mask',
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
__pdoc__['SimulationContext.group_counts'] = """Column count per group.

Even if columns are not grouped, `group_counts` contains ones - one column per group.
"""
__pdoc__['SimulationContext.init_cash'] = """Initial capital per column, or per group if cash sharing is enabled.

If `cash_sharing` is True, has shape `(target_shape[0], group_counts.shape[0])`. 
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

Has shape `(target_shape[0], group_counts.shape[0])`.
"""
__pdoc__['SimulationContext.min_size'] = """Minimum size for an order to be accepted.

Has shape `(target_shape[1],)`.
"""
__pdoc__['SimulationContext.order_records'] = """Order records.

Order records are not filled incrementally from left to right, but based on their position 
in the matrix. To get index of a record, use `col * target_shape[0] + i`.

!!! warning
    Initially, all records are empty. Accessing empty records will not raise any errors or warnings, 
    but provide you with arbitrary data (see [np.empty](https://numpy.org/doc/stable/reference/generated/numpy.empty.html)).
    To check if a record is set, use `record_mask`.
"""
__pdoc__['SimulationContext.record_mask'] = """Mask of records that have been set.

Has the same length as `order_records`.
"""
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

# ############# GroupContext ############# #

GroupContext = namedtuple('GroupContext', [
    *SimulationContext._fields,
    'group',
    'group_len',
    'from_col',
    'to_col',
    'num_records'
])

__pdoc__['GroupContext'] = "A named tuple representing context of the group."
for field in SimulationContext._fields:
    __pdoc__[f'GroupContext.{field}'] = f"See `SimulationContext.{field}`."
__pdoc__['GroupContext.group'] = """Index of the group.

Has range `[0, group_counts.shape[0])`.
"""
__pdoc__['GroupContext.group_len'] = """Number of columns in the group.

Scalar value. Same as `group_counts[group]`.
"""
__pdoc__['GroupContext.from_col'] = """Index of the first column in the group.

Has range `[0, target_shape[1])`.
"""
__pdoc__['GroupContext.to_col'] = """Index of the first column in the next group.

Has range `[1, target_shape[1] + 1)`. 

If columns are not grouped, equals `from_col + 1`.
"""
__pdoc__['GroupContext.num_records'] = "Number of records filled up to this point."

# ############# RowContext ############# #

RowContext = namedtuple('RowContext', [
    *SimulationContext._fields,
    'i',
    'num_records'
])
__pdoc__['RowContext'] = "A named tuple representing context of the row."
for field in SimulationContext._fields:
    __pdoc__[f'RowContext.{field}'] = f"See `SimulationContext.{field}`."
__pdoc__['RowContext.i'] = """Current row (time axis).

Has range `[0, target_shape[0])`.
"""
__pdoc__['RowContext.num_records'] = "See `GroupContext.num_records`."

# ############# SegmentContext ############# #

SegmentContext = namedtuple('SegmentContext', [
    *SimulationContext._fields,
    'i',
    'group',
    'group_len',
    'from_col',
    'to_col',
    'num_records',
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
__pdoc__['SegmentContext.num_records'] = "See `GroupContext.num_records`."
__pdoc__['SegmentContext.call_seq_now'] = """Current sequence of calls.

Has shape `(group_len,)`. 
"""

# ############# OrderContext ############# #

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

# ############# InitCashMode ############# #

InitCashMode = namedtuple('InitCashMode', [
    'Auto',
    'AutoAlign'
])(*range(2))
"""_"""

__pdoc__['InitCashMode'] = f"""Initial cash mode.

```plaintext
{json.dumps(dict(zip(InitCashMode._fields, InitCashMode)), indent=2)}
```

Attributes:
    Auto: Optimal initial cash for each column.
    AutoAlign: Optimal initial cash aligned across all columns.
"""

# ############# CallSeqType ############# #

CallSeqType = namedtuple('CallSeqType', [
    'Default',
    'Reversed',
    'Random',
    'Auto'
])(*range(4))
"""_"""

__pdoc__['CallSeqType'] = f"""Call sequence type.

```plaintext
{json.dumps(dict(zip(CallSeqType._fields, CallSeqType)), indent=2)}
```

Attributes:
    Default: Place calls from left to right.
    Reversed: Place calls from right to left.
    Random: Place calls randomly.
    Auto: Place calls dynamically based on order value.
"""

# ############# SizeType ############# #

SizeType = namedtuple('SizeType', [
    'Shares',
    'TargetShares',
    'TargetValue',
    'TargetPercent'
])(*range(4))
"""_"""

__pdoc__['SizeType'] = f"""Size type.

```plaintext
{json.dumps(dict(zip(SizeType._fields, SizeType)), indent=2)}
```

Attributes:
    Shares: Number of shares.
    TargetShares: Number of shares to hold after transaction.
    TargetValue: Total value of holdings to hold after transaction.
    TargetPercent: Percentage of total value to hold after transaction.
"""

# ############# ConflictMode ############# #

ConflictMode = namedtuple('ConflictMode', [
    'Ignore',
    'Entry',
    'Exit',
    'Opposite'
])(*range(4))
"""_"""

__pdoc__['ConflictMode'] = f"""Conflict mode.

```plaintext
{json.dumps(dict(zip(ConflictMode._fields, ConflictMode)), indent=2)}
```

What should happen if both entry and exit signals occur simultaneously?

Attributes:
    Ignore: Ignore both signals.
    Entry: Use entry signal.
    Exit: Use exit signal.
    Opposite: Use opposite signal. Takes effect only when in position.
"""

# ############# Order ############# #

Order = namedtuple('Order', [
    'size',
    'size_type',
    'price',
    'fees',
    'fixed_fees',
    'slippage',
    'min_size',
    'max_size',
    'reject_prob',
    'allow_partial',
    'raise_by_reject'
])

__pdoc__['Order'] = "A named tuple representing an order."
__pdoc__['Order.size'] = "Size in shares. Filled size will depend upon your funds."
__pdoc__['Order.size_type'] = "See `SizeType`."
__pdoc__['Order.price'] = "Price per share. Filled price will depend upon slippage."
__pdoc__['Order.fees'] = "Fees in percentage of the order value."
__pdoc__['Order.fixed_fees'] = "Fixed amount of fees to pay for this order."
__pdoc__['Order.slippage'] = "Slippage in percentage of `price`."
__pdoc__['Order.min_size'] = "Minimum size. Lower than that will be rejected."
__pdoc__['Order.max_size'] = "Maximum size. Higher than that will be cut."
__pdoc__['Order.reject_prob'] = "Probability of rejecting this order."
__pdoc__['Order.allow_partial'] = "Whether to allow partial fill."
__pdoc__['Order.raise_by_reject'] = "Whether to raise exception if order has been rejected."

NoOrder = Order(np.nan, -1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, False)
"""_"""

__pdoc__['NoOrder'] = "Order that will not be processed."

# ############# OrderStatus ############# #

OrderStatus = namedtuple('OrderStatus', [
    'Filled',
    'Ignored',
    'Rejected'
])(*range(3))
"""_"""

__pdoc__['OrderStatus'] = f"""Order status.

```plaintext
{json.dumps(dict(zip(OrderStatus._fields, OrderStatus)), indent=2)}
```

Attributes:
    Filled: Order filled.
    Ignored: Order ignored.
    Rejected: Order rejected.
"""

# ############# OrderResult ############# #

OrderResult = namedtuple('OrderResult', [
    'size',
    'price',
    'fees',
    'side',
    'status'
])

__pdoc__['OrderResult'] = "A named tuple representing an order result."
__pdoc__['OrderResult.size'] = "Filled size in shares."
__pdoc__['OrderResult.price'] = "Filled price per share, adjusted with slippage."
__pdoc__['OrderResult.fees'] = "Total fees paid for this order."
__pdoc__['OrderResult.side'] = "See `vectorbt.records.enums.OrderSide`."
__pdoc__['OrderResult.status'] = "See `OrderStatus`."

IgnoredOrder = OrderResult(np.nan, np.nan, np.nan, -1, OrderStatus.Ignored)
RejectedOrder = OrderResult(np.nan, np.nan, np.nan, -1, OrderStatus.Rejected)


class RejectedOrderError(Exception):
    """Rejected order error."""
    pass


# ############# SignalType ############# #

SignalType = namedtuple('SignalType', [
    'Long',
    'Short',
    'LongShort'
])(*range(3))
"""_"""

__pdoc__['SignalType'] = f"""Signal type.

```plaintext
{json.dumps(dict(zip(SignalType._fields, SignalType)), indent=2)}
```

Attributes:
    Long: Entry signal to go long, exit signal to close the long position.
    Short: Entry signal to go short, exit signal to close the short position.
    LongShort: Entry signal to go long, exit signal to go short.
"""

