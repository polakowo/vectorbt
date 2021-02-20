"""Modules for building and running look-ahead indicators and labelers.

## Basic

Module `vectorbt.labels.basic` provides a collection of basic look-ahead indicators and labelers.

You can access all the indicators either by `vbt.*` or `vbt.labels.*`.

## Numba-compiled functions

Module `vectorbt.labels.nb` provides an arsenal of Numba-compiled functions that are used by indicator
classes. These only accept NumPy arrays and other Numba-compatible types.

## Enums

Module `vectorbt.labels.enums` defines enums and other schemas for `vectorbt.labels`.
"""

from vectorbt.labels.basic import (
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
