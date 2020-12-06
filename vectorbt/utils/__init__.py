"""Modules with utilities that are used throughout vectorbt."""

from vectorbt.utils.config import Config, Configured
from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.utils.widgets import CustomFigure, CustomFigureWidget

__all__ = [
    'Config',
    'Configured',
    'cached_property',
    'cached_method',
    'CustomFigure',
    'CustomFigureWidget'
]

__pdoc__ = {k: False for k in __all__}
