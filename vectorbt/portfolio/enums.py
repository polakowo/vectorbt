"""Named tuples and enumerated types."""

from collections import namedtuple
import json

__pdoc__ = {}

# We use namedtuple for enums and classes to be able to use them in Numba

# ############# Named tuples ############# #

Order = namedtuple('Order', [
    'size',
    'price',
    'fees',
    'fixed_fees',
    'slippage'
], defaults=[0., 0., 0.])

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
__pdoc__['FilledOrder.side'] = "See `OrderSide`."

# ############# Enums ############# #

OrderSide = namedtuple('OrderSide', [
    'Buy',
    'Sell'
])(*range(2))
"""_"""

__pdoc__['OrderSide'] = f"""An enum representing the side of an order.

```plaintext
{json.dumps(dict(zip(OrderSide._fields, OrderSide)), indent=2)}
```
"""

PositionStatus = namedtuple('PositionStatus', [
    'Open',
    'Closed'
])(*range(2))
"""_"""

__pdoc__['PositionStatus'] = f"""An enum representing the status of a position.

```plaintext
{json.dumps(dict(zip(PositionStatus._fields, PositionStatus)), indent=2)}
```
"""

# ############# Record field enums ############# #

OrderRecord = namedtuple('OrderRecord', [
    'Column',
    'Index',
    'Size',
    'Price',
    'Fees',
    'Side'
])(*range(6))
"""_"""

__pdoc__['OrderRecord'] = f"""An enum representing fields of an order record.

```plaintext
{json.dumps(dict(zip(OrderRecord._fields, OrderRecord)), indent=2)}
```
"""

EventRecord = namedtuple('EventRecord', [
    'Column',
    'Size',
    'OpenAt',
    'OpenPrice',
    'OpenFees',
    'CloseAt',
    'ClosePrice',
    'CloseFees',
    'PnL',
    'Return'
])(*range(10))
"""_"""

__pdoc__['EventRecord'] = f"""An enum representing fields of an event record. 

An event can be anything that fits the following schema (e.g. trade and position):

```plaintext
{json.dumps(dict(zip(EventRecord._fields, EventRecord)), indent=2)}
```
"""

TradeRecord = namedtuple('TradeRecord', [
    *EventRecord._fields,
    'Position'
])(*range(11))
"""_"""

__pdoc__['TradeRecord'] = f"""An enum representing fields of a trade record. Follows `EventRecord`.

```plaintext
{json.dumps(dict(zip(TradeRecord._fields, TradeRecord)), indent=2)}
```
"""

PositionRecord = namedtuple('PositionRecord', [
    *EventRecord._fields,
    'Status'
])(*range(11))
"""_"""

__pdoc__['PositionRecord'] = f"""An enum representing fields of a position record. Follows `EventRecord`.

```plaintext
{json.dumps(dict(zip(PositionRecord._fields, PositionRecord)), indent=2)}
```
"""
