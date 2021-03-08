"""Modules for working with records.

Records are the second form of data representation in vectorbt. They allow storing sparse event data
such as drawdowns, orders, trades, and positions, without converting them back to the matrix form and
occupying the user's memory."""

from vectorbt.records.mapped_array import MappedArray
from vectorbt.records.base import Records

__all__ = [
    'MappedArray',
    'Records'
]

__pdoc__ = {k: False for k in __all__}
