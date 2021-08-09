"""Modules for working with any time series.

In contrast to the `vectorbt.base` sub-package, focuses on the data itself."""

from vectorbt.generic.enums import *
from vectorbt.generic.ranges import Ranges
from vectorbt.generic.drawdowns import Drawdowns
from vectorbt.generic.splitters import RangeSplitter, RollingSplitter, ExpandingSplitter

__all__ = [
    'Ranges',
    'Drawdowns',
    'RangeSplitter',
    'RollingSplitter',
    'ExpandingSplitter'
]

__pdoc__ = {k: False for k in __all__}
