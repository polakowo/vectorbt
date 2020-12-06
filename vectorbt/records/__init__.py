"""Modules for working with records.

Records are the second form of data representation in vectorbt. They allow storing sparse event data
such as drawdowns, orders, trades, and positions, without converting them back to the matrix form and
occupying the user's memory.

## Records class

Class `vectorbt.records.base.Records` wraps the actual records array (such as trades) and
exposes methods for mapping it to some array of values (such as P&L of each trade).

## MappedArray class

Class `vectorbt.records.mapped_array.MappedArray` exposes methods for reducing, converting,
and plotting arrays mapped by `vectorbt.records.base.Records` class.

## ColumnMapper class

Class `vectorbt.records.col_mapper.ColumnMapper` is used by `vectorbt.records.base.Records` and
`vectorbt.records.mapped_array.MappedArray` classes to make use of column and group metadata.

## Numba-compiled functions

Module `vectorbt.records.nb` provides an arsenal of Numba-compiled functions for records and mapped arrays.
These only accept NumPy arrays and other Numba-compatible types.
"""

from vectorbt.records.mapped_array import MappedArray
from vectorbt.records.base import Records

__all__ = [
    'MappedArray',
    'Records'
]

__pdoc__ = {k: False for k in __all__}
