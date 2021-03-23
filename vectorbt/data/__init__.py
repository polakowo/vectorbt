"""Modules for working with data sources."""

from vectorbt.data.base import symbol_dict, Data
from vectorbt.data.updater import DataUpdater
from vectorbt.data.custom import SyntheticData, GBMData, YFData, BinanceData, CCXTData

__all__ = [
    'symbol_dict',
    'Data',
    'DataUpdater',
    'SyntheticData',
    'GBMData',
    'YFData',
    'BinanceData',
    'CCXTData'
]

__pdoc__ = {k: False for k in __all__}
