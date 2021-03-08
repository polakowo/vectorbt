"""Modules for building and running look-ahead indicators and label generators."""

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
