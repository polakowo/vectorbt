"""Modules for working with data sources."""

from vectorbt.data.base import symbol_dict, Data
from vectorbt.data.custom import SyntheticData, GBMData, YFData, BinanceData
from vectorbt.data.updater import DataUpdater

__all__ = [
    'symbol_dict',
    'Data',
    'SyntheticData',
    'GBMData',
    'YFData',
    'BinanceData',
    'DataUpdater'
]

__pdoc__ = {k: False for k in __all__}
