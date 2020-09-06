"""Named tuples and enumerated types."""

import numpy as np
from collections import namedtuple
import json

__pdoc__ = {}

# We use namedtuple for enums and classes to be able to use them in Numba

CallSeqType = namedtuple('CallSeqType', [
    'Default',
    'Reversed',
    'Random'
])(*range(3))
"""_"""

__pdoc__['CallSeqType'] = f"""Call sequence type.

```plaintext
{json.dumps(dict(zip(CallSeqType._fields, CallSeqType)), indent=2)}
```

Attributes:
    Default: Place calls from left to right.
    Reversed: Place calls from right to left.
    Random: Place calls randomly.
"""

OrderContext = namedtuple('OrderContext', [
    'i',
    'col',
    'group',
    'from_col',
    'to_col',
    'target_shape',
    'group_counts',
    'call_seq',
    'call_seq_now',
    'call_idx',
    'active_mask',
    'init_cash',
    'cash_sharing',
    'order_records',
    'record_mask',
    'num_records',
    'run_cash',
    'run_shares',
    'cash_now',
    'shares_now'
])

__pdoc__['OrderContext'] = """A named tuple representing the current order context."""
__pdoc__['OrderContext.i'] = """Current index (time axis).

Has range `[0, target_shape[0])`.
"""
__pdoc__['OrderContext.col'] = """Current column (feature axis).

Has range `[0, target_shape[1])` and is always within `[from_col, to_col)`.
"""
__pdoc__['OrderContext.group'] = """Current group index.

Has range `[0, group_counts.shape[0])`.
"""
__pdoc__['OrderContext.from_col'] = """Index of the first column in the current group.

Has range `[0, target_shape[1])`.
"""
__pdoc__['OrderContext.to_col'] = """Index of the first column in the next group.

Has range `[1, target_shape[1] + 1)`. 

If columns are not grouped, equals `from_col + 1`.
"""
__pdoc__['OrderContext.target_shape'] = """Target shape.

A tuple with exactly two elements: the number of steps and columns.
"""
__pdoc__['OrderContext.group_counts'] = """Column count per group.

Even if columns are not grouped, `group_counts` contains ones - one column per group.
"""
__pdoc__['OrderContext.call_seq'] = """Default sequence of calls per row and group.

Controls the sequence in which `order_func_nb` is executed within a group of multiple columns. 
By default, the sequence goes from left to right. Only effective if grouping and cash sharing are enabled.

Has shape `target_shape` and each value must exist in the range `[0, to_col - from_col)`.

!!! note
    If the call sequence for this row has been changed dynamically, it will be effective but may not 
    be visible in the `call_seq` until the entire row is processed. For this, use `call_seq_now`.
"""
__pdoc__['OrderContext.call_seq_now'] = """Current sequence of calls.

Has shape `(to_col - from_col,)`. 
"""
__pdoc__['OrderContext.call_idx'] = """Index of the current call in `call_seq_now`.

Has range `[0, to_col - from_col)`.
"""
__pdoc__['OrderContext.active_mask'] = """Mask of whether a particular row should be executed, per group.

Has shape `(target_shape[0], group_counts.shape[0])`.
"""
__pdoc__['OrderContext.init_cash'] = """Initial capital per column, or per group if cash sharing is enabled.

If `cash_sharing` is `True`, has shape `(target_shape[0], group_counts.shape[0])`. 
Otherwise, has shape `target_shape`.
"""
__pdoc__['OrderContext.cash_sharing'] = """Whether cash sharing is enabled."""
__pdoc__['OrderContext.order_records'] = """Order records, set and empty yet.

Order records are not filled incrementally from left to right, but based on their position 
in the matrix. To get index of a record, use `col * target_shape[0] + i`.

!!! note
    Records that are not set have arbitrary data. To check if a record is set, use `record_mask`.
"""
__pdoc__['OrderContext.record_mask'] = """Mask of set order records.

Contains a value for each record indicating whether it's set. 

Has the same length as `order_records`.
"""
__pdoc__['OrderContext.num_records'] = """Number of records filled up to this point."""
__pdoc__['OrderContext.run_cash'] = """Running cash per column, or per group if cash sharing is enabled.

Has the same shape as `init_cash`.
"""
__pdoc__['OrderContext.run_shares'] = """Running shares per column.

Has shape `target_shape[1]`.
"""
__pdoc__['OrderContext.cash_now'] = """Currently available cash.

Scalar value. Per group if cash sharing is enabled, otherwise per column.
"""
__pdoc__['OrderContext.shares_now'] = """Currently available shares.

Scalar value. Always per column.
"""

GroupRowContext = namedtuple('GroupRowContext', [
    'i',
    'group',
    'from_col',
    'to_col',
    'target_shape',
    'group_counts',
    'call_seq',
    'active_mask',
    'init_cash',
    'cash_sharing',
    'order_records',
    'record_mask',
    'num_records',
    'run_cash',
    'run_shares'
])

__pdoc__['GroupRowContext'] = "A named tuple representing the current row context."
__pdoc__['GroupRowContext.i'] = __pdoc__['OrderContext.i']
__pdoc__['GroupRowContext.group'] = __pdoc__['OrderContext.group']
__pdoc__['GroupRowContext.from_col'] = __pdoc__['OrderContext.from_col']
__pdoc__['GroupRowContext.to_col'] = __pdoc__['OrderContext.to_col']
__pdoc__['GroupRowContext.target_shape'] = __pdoc__['OrderContext.target_shape']
__pdoc__['GroupRowContext.group_counts'] = __pdoc__['OrderContext.group_counts']
__pdoc__['GroupRowContext.call_seq'] = __pdoc__['OrderContext.call_seq']
__pdoc__['GroupRowContext.active_mask'] = __pdoc__['OrderContext.active_mask']
__pdoc__['GroupRowContext.init_cash'] = __pdoc__['OrderContext.init_cash']
__pdoc__['GroupRowContext.cash_sharing'] = __pdoc__['OrderContext.cash_sharing']
__pdoc__['GroupRowContext.order_records'] = __pdoc__['OrderContext.order_records']
__pdoc__['GroupRowContext.record_mask'] = __pdoc__['OrderContext.record_mask']
__pdoc__['GroupRowContext.num_records'] = __pdoc__['OrderContext.num_records']
__pdoc__['GroupRowContext.run_cash'] = __pdoc__['OrderContext.run_cash']
__pdoc__['GroupRowContext.run_shares'] = __pdoc__['OrderContext.run_shares']

RowContext = namedtuple('RowContext', [
    'i',
    'target_shape',
    'group_counts',
    'call_seq',
    'active_mask',
    'init_cash',
    'cash_sharing',
    'order_records',
    'record_mask',
    'num_records',
    'run_cash',
    'run_shares'
])

__pdoc__['RowContext'] = "A named tuple representing the current row context."
__pdoc__['RowContext.i'] = __pdoc__['OrderContext.i']
__pdoc__['RowContext.target_shape'] = __pdoc__['OrderContext.target_shape']
__pdoc__['RowContext.group_counts'] = __pdoc__['OrderContext.group_counts']
__pdoc__['RowContext.call_seq'] = __pdoc__['OrderContext.call_seq']
__pdoc__['RowContext.active_mask'] = __pdoc__['OrderContext.active_mask']
__pdoc__['RowContext.init_cash'] = __pdoc__['OrderContext.init_cash']
__pdoc__['RowContext.cash_sharing'] = __pdoc__['OrderContext.cash_sharing']
__pdoc__['RowContext.order_records'] = __pdoc__['OrderContext.order_records']
__pdoc__['RowContext.record_mask'] = __pdoc__['OrderContext.record_mask']
__pdoc__['RowContext.num_records'] = __pdoc__['OrderContext.num_records']
__pdoc__['RowContext.run_cash'] = __pdoc__['OrderContext.run_cash']
__pdoc__['RowContext.run_shares'] = __pdoc__['OrderContext.run_shares']

SizeType = namedtuple('SizeType', [
    'Shares',
    'TargetShares',
    'Cash',
    'TargetCash',
    'TargetValue',
    'TargetPercent'
])(*range(6))
"""_"""

__pdoc__['SizeType'] = f"""Size type.

```plaintext
{json.dumps(dict(zip(SizeType._fields, SizeType)), indent=2)}
```

Attributes:
    Shares: Amount of shares to buy/sell.
    TargetShares: Target amount of shares to hold after transaction.
    Cash: Amount of cash to spend for transaction.
    TargetCash: Target amount of cash to hold after transaction.
    TargetValue: Target value of holdings.
    
        Holding value is calculated by multiplying shares by requested price of transaction.

    TargetPercent: Target percentage of total value.
    
        Total value is calculated by summing up available cash and holding value.

        !!! note
            Does not take into account holding value of other columns (even when grouped), since
            their value cannot be moved until the columns are processed.
"""

AccumulateExitMode = namedtuple('AccumulateExitMode', [
    'Close',
    'Reduce'
])(*range(2))
"""_"""

__pdoc__['AccumulateExitMode'] = f"""Accumulation exit mode.

```plaintext
{json.dumps(dict(zip(AccumulateExitMode._fields, AccumulateExitMode)), indent=2)}
```

What should happen if exit signal occurs and accumulation is turned on?

Attributes:
    Close: Close the position by selling all.
    Reduce: Reduce the position by selling size.
"""

ConflictMode = namedtuple('ConflictMode', [
    'Ignore',
    'Exit',
    'ExitAndEntry'
])(*range(3))
"""_"""

__pdoc__['ConflictMode'] = f"""Conflict mode.

```plaintext
{json.dumps(dict(zip(ConflictMode._fields, ConflictMode)), indent=2)}
```

What should happen if both entry and exit signals occur simultaneously?

Attributes:
    Ignore: Ignore both signals.
    Exit: Ignore entry signal.
    ExitAndEntry: Imitate exit and entry by using entry size as target.
"""

Order = namedtuple('Order', [
    'size',
    'size_type',
    'price',
    'fees',
    'fixed_fees',
    'slippage'
])

__pdoc__['Order'] = "A named tuple representing an order."
__pdoc__['Order.size'] = "Size in shares. Filled size will depend upon your funds."
__pdoc__['Order.size_type'] = "See `SizeType`."
__pdoc__['Order.price'] = "Price per share. Filled price will depend upon slippage."
__pdoc__['Order.fees'] = "Fees in percentage of the order value."
__pdoc__['Order.fixed_fees'] = "Fixed amount of fees to pay for this order."
__pdoc__['Order.slippage'] = "Slippage in percentage of `price`."

NoOrder = Order(np.nan, -1, np.nan, np.nan, np.nan, np.nan)
"""_"""

__pdoc__['NoOrder'] = "Order that will not be processed."

OrderStatus = namedtuple('OrderStatus', [
    'Filled',
    'Rejected'
])(*range(2))
"""_"""

__pdoc__['OrderStatus'] = f"""Order status.

```plaintext
{json.dumps(dict(zip(OrderStatus._fields, OrderStatus)), indent=2)}
```

Attributes:
    Filled: Order filled.
    Rejected: Order rejected.
"""

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
__pdoc__['OrderResult.status'] = "See `vectorbt.records.enums.OrderStatus`."

RejectedOrder = OrderResult(np.nan, np.nan, np.nan, -1, OrderStatus.Rejected)
