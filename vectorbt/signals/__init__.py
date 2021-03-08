"""Modules for working with signals, such as entry and exit signals."""

from vectorbt.signals.enums import *
from vectorbt.signals.factory import SignalFactory
from vectorbt.signals.generators import (
    RAND,
    RPROB,
    RPROBEX,
    IRPROBEX,
    STEX,
    ISTEX,
    OHLCSTEX,
    IOHLCSTEX
)

__all__ = [
    'SignalFactory',
    'RAND',
    'RPROB',
    'RPROBEX',
    'IRPROBEX',
    'STEX',
    'ISTEX',
    'OHLCSTEX',
    'IOHLCSTEX'
]

__pdoc__ = {k: False for k in __all__}
