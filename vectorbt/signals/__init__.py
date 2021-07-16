"""Modules for working with signals, such as entry and exit signals."""

from vectorbt.signals.enums import *
from vectorbt.signals.factory import SignalFactory
from vectorbt.signals.generators import (
    RAND,
    RANDX,
    RANDNX,
    RPROB,
    RPROBX,
    RPROBCX,
    RPROBNX,
    STX,
    STCX,
    OHLCSTX,
    OHLCSTCX
)

__all__ = [
    'SignalFactory',
    'RAND',
    'RANDX',
    'RANDNX',
    'RPROB',
    'RPROBX',
    'RPROBCX',
    'RPROBNX',
    'STX',
    'STCX',
    'OHLCSTX',
    'OHLCSTCX'
]

__pdoc__ = {k: False for k in __all__}
