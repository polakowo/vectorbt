"""Named tuples and enumerated types."""

from collections import namedtuple
import json

__pdoc__ = {}

# We use namedtuple for enums and classes to be able to use them in Numba

# ############# Named tuples ############# #

OrderContext = namedtuple('OrderContext', [
    'col',
    'i',
    'target_shape',
    'init_capital',
    'order_records',
    'cash',
    'shares',
    'run_cash',
    'run_shares'
])

__pdoc__['OrderContext'] = "A named tuple representing the current order context."
__pdoc__['OrderContext.col'] = "Current column."
__pdoc__['OrderContext.i'] = "Current index."
__pdoc__['OrderContext.target_shape'] = "Target shape."
__pdoc__['OrderContext.init_capital'] = "Initial capital."
__pdoc__['OrderContext.order_records'] = "Order records filled up to this time point. Other elements are empty."
__pdoc__['OrderContext.cash'] = "Cash filled up to this time point. Other elements are empty."
__pdoc__['OrderContext.shares'] = "Shares filled up to this time point. Other elements are empty."
__pdoc__['OrderContext.run_cash'] = "Current cash holdings."
__pdoc__['OrderContext.run_shares'] = "Current shares holdings."

RowContext = namedtuple('RowContext', [
    'i',
    'target_shape',
    'init_capital',
    'order_records',
    'cash',
    'shares'
])

__pdoc__['RowContext'] = "A named tuple representing the current row context."
__pdoc__['RowContext.i'] = __pdoc__['OrderContext.i']
__pdoc__['RowContext.target_shape'] = __pdoc__['OrderContext.target_shape']
__pdoc__['RowContext.init_capital'] = __pdoc__['OrderContext.init_capital']
__pdoc__['RowContext.order_records'] = __pdoc__['OrderContext.order_records']
__pdoc__['RowContext.cash'] = __pdoc__['OrderContext.cash']
__pdoc__['RowContext.shares'] = __pdoc__['OrderContext.shares']

SizeType = namedtuple('SizeType', [
    'Size',
    'Value',
    'Percent',
    'TargetSize',
    'TargetValue',
    'TargetPercent'
])(*range(6))
"""_"""

__pdoc__['SizeType'] = f"""Size type.

```plaintext
{json.dumps(dict(zip(SizeType._fields, SizeType)), indent=2)}
```
"""

Order = namedtuple('Order', [
    'size',
    'price',
    'fees',
    'fixed_fees',
    'slippage'
])

__pdoc__['Order'] = "A named tuple representing an order."
__pdoc__['Order.size'] = "Size in shares. Filled size will depend upon your funds."
__pdoc__['Order.price'] = "Price per share. Filled price will depend upon slippage."
__pdoc__['Order.fees'] = "Fees in percentage of the order value."
__pdoc__['Order.fixed_fees'] = "Fixed amount of fees to pay for this order."
__pdoc__['Order.slippage'] = "Slippage in percentage of `price`."

FilledOrder = namedtuple('FilledOrder', [
    'size',
    'price',
    'fees',
    'side'
])

__pdoc__['FilledOrder'] = "A named tuple representing a filled order."
__pdoc__['FilledOrder.size'] = "Filled size in shares."
__pdoc__['FilledOrder.price'] = "Filled price per share, adjusted with slippage."
__pdoc__['FilledOrder.side'] = "See `vectorbt.records.enums.OrderSide`."
