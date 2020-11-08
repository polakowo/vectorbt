"""Modules for working with records.

Records are the second form of data representation in vectorbt. They allow storing sparse event data
such as drawdowns, orders, trades, and positions, without converting them back to the matrix form and
occupying the user's memory.

## Records class

The `vectorbt.records.base.Records` class wraps the actual records array (such as trades) and
exposes methods for mapping it to some array of values (such as P&L of each trade).

## MappedArray class

The `vectorbt.records.mapped_array.MappedArray` class exposes methods for reducing, converting,
or plotting arrays mapped by the `vectorbt.records.base.Records` class.

## Numba-compiled functions

`vectorbt.records.nb` provides an arsenal of Numba-compiled functions that are used for generating,
mapping, and reducing records. These only accept NumPy arrays and other Numba-compatible types.
"""

from vectorbt.records.base import Records
from vectorbt.records.mapped_array import MappedArray
