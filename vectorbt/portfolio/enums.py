"""Named tuples and enumerated types."""

from collections import namedtuple

__pdoc__ = {}

# We use namedtuple for enums and classes to be able to use them in Numba

# ############# Named tuples ############# #

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
