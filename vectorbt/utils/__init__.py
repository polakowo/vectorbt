"""Modules with utilities that are used throughout vectorbt."""

from vectorbt.utils.config import atomic_dict, merge_dicts, Config, Configured, AtomicConfig
from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.utils.widgets import Figure, FigureWidget, make_subplots
from vectorbt.utils.random import set_seed
from vectorbt.utils.image import save_animation
from vectorbt.utils.datetime import ScheduleManager

__all__ = [
    'atomic_dict',
    'merge_dicts',
    'Config',
    'Configured',
    'AtomicConfig',
    'cached_property',
    'cached_method',
    'Figure',
    'FigureWidget',
    'make_subplots',
    'set_seed',
    'save_animation',
    'ScheduleManager'
]

__pdoc__ = {k: False for k in __all__}
