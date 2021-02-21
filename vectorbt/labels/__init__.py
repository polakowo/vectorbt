"""Modules for building and running look-ahead indicators and label generators.

## Generators

Module `vectorbt.labels.generators` provides a collection of look-ahead indicators and label generators.

You can access all the indicators either by `vbt.*` or `vbt.labels.*`.

## Numba-compiled functions

Module `vectorbt.labels.nb` provides an arsenal of Numba-compiled functions that are used by indicator
classes. These only accept NumPy arrays and other Numba-compatible types.

## Enums

Module `vectorbt.labels.enums` defines enums and other schemas for `vectorbt.labels`.
"""

from vectorbt.labels.enums import *
from vectorbt.labels.generators import (
    FMEAN,
    FSTD,
    FMIN,
    FMAX,
    FIXLB,
    MEANLB,
    LEXLB,
    TRENDLB,
    BOLB
)

__all__ = [
    'FMEAN',
    'FSTD',
    'FMIN',
    'FMAX',
    'FIXLB',
    'MEANLB',
    'LEXLB',
    'TRENDLB',
    'BOLB'
]

__pdoc__ = {k: False for k in __all__}
