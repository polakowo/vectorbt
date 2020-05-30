"""Enumerated types."""

from collections import namedtuple

__pdoc__ = {}

# We use namedtuple for enums and classes to be able to use them in Numba

# ############# Classes ############# #

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

# ############# Constants ############# #

OrderSide = namedtuple('OrderSide', [
    'Buy', 
    'Sell'
])(*range(2))
"""An enum representing the side of an order."""

OrderRecord = namedtuple('OrderRecord', [
    'Column',
    'Index',
    'Size',
    'Price',
    'Fees',
    'Side'
])(*range(6))
"""An enum representing an order record."""

PositionStatus = namedtuple('PositionStatus', [
    'Open', 
    'Closed'
])(*range(2))
"""An enum representing the status of a position."""

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
"""An enum representing an event such as trade or position."""

TradeRecord = namedtuple('TradeRecord', [
    *EventRecord._fields,
    'Position'
])(*range(11))
"""An enum representing a trade. Follows `EventRecord`."""

PositionRecord = namedtuple('PositionRecord', [
    *EventRecord._fields,
    'Status'
])(*range(11))
"""An enum representing a position. Follows `EventRecord`."""