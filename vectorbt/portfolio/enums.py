"""Enumerated types."""

from collections import namedtuple

__pdoc__ = {}

Order = namedtuple('Order', [
    'size',
    'price',
    'fees',
    'fixed_fees',
    'slippage'
], defaults=[0., 0., 0.])

__pdoc__['Order'] = "A named tuple representing an order."
__pdoc__['Order.size'] = "Size of the trade. Final size will depend upon your funds."
__pdoc__['Order.price'] = "Price per share. Final price will depend upon slippage."
__pdoc__['Order.fees'] = "Fees in percentage of the trade value."
__pdoc__['Order.fixed_fees'] = "Fixed amount of fees to pay for this trade."
__pdoc__['Order.slippage'] = "Slippage in percentage of `price`."

Event = namedtuple('Event', [
    'col',
    'size',
    'open_i',
    'open_price',
    'open_fees',
    'close_i',
    'close_price',
    'close_fees'
])

__pdoc__['Event'] = "A named tuple representing an event, such as trade or position."
__pdoc__['Event.col'] = "Column index."
__pdoc__['Event.size'] = "Size in shares."
__pdoc__['Event.open_i'] = "Opening index."
__pdoc__['Event.open_price'] = "Opening price."
__pdoc__['Event.open_fees'] = "Opening fees."
__pdoc__['Event.close_i'] = "Closing index."
__pdoc__['Event.close_price'] = "Closing price."
__pdoc__['Event.close_fees'] = "Closing fees."

Trade = namedtuple('Trade', [*Event._fields, 'type', 'position_idx'])

__pdoc__['Trade'] = "A named tuple representing a trade."
__pdoc__['Trade.col'] = __pdoc__['Event.col']
__pdoc__['Trade.size'] = __pdoc__['Event.size']
__pdoc__['Trade.open_i'] = __pdoc__['Event.open_i']
__pdoc__['Trade.open_price'] = __pdoc__['Event.open_price']
__pdoc__['Trade.open_fees'] = __pdoc__['Event.open_fees']
__pdoc__['Trade.close_i'] = __pdoc__['Event.close_i']
__pdoc__['Trade.close_price'] = __pdoc__['Event.close_price']
__pdoc__['Trade.close_fees'] = __pdoc__['Event.close_fees']
__pdoc__['Trade.type'] = "See `TradeType`."
__pdoc__['Trade.position_idx'] = "Position index."

Position = namedtuple('Position', [*Event._fields, 'status'])

__pdoc__['Position'] = "A named tuple representing a position."
__pdoc__['Position.col'] = __pdoc__['Event.col']
__pdoc__['Position.size'] = __pdoc__['Event.size']
__pdoc__['Position.open_i'] = __pdoc__['Event.open_i']
__pdoc__['Position.open_price'] = __pdoc__['Event.open_price']
__pdoc__['Position.open_fees'] = __pdoc__['Event.open_fees']
__pdoc__['Position.close_i'] = __pdoc__['Event.close_i']
__pdoc__['Position.close_price'] = __pdoc__['Event.close_price']
__pdoc__['Position.close_fees'] = __pdoc__['Event.close_fees']
__pdoc__['Position.status'] = "See `PositionStatus`."

# Create enums using named tuples to be accepted as globals by Numba

TradeType = namedtuple('TradeType', [
    'Buy', 
    'Sell'
])(*range(2))
"""Buy or Sell."""

PositionStatus = namedtuple('PositionStatus', [
    'Open', 
    'Closed'
])(*range(2))
"""Open or Closed."""

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

__pdoc__['EventRecord'] = "A named tuple representing an event, such as trade or position."
__pdoc__['EventRecord.Column'] = "Column index."
__pdoc__['EventRecord.Size'] = "Size in shares."
__pdoc__['EventRecord.OpenAt'] = "Opening index."
__pdoc__['EventRecord.OpenPrice'] = "Opening price."
__pdoc__['EventRecord.OpenFees'] = "Opening fees."
__pdoc__['EventRecord.CloseAt'] = "Closing index."
__pdoc__['EventRecord.ClosePrice'] = "Closing price."
__pdoc__['EventRecord.CloseFees'] = "Closing fees."
__pdoc__['EventRecord.PnL'] = "Total P&L."
__pdoc__['EventRecord.Return'] = "Total return."

TradeRecord = namedtuple('TradeRecord', [
    *EventRecord._fields,
    'Type',
    'Position'
])(*range(12))

PositionRecord = namedtuple('PositionRecord', [
    *EventRecord._fields,
    'Status'
])(*range(11))