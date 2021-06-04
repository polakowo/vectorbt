"""Modules with utilities that are used throughout vectorbt."""

from vectorbt.utils.config import atomic_dict, merge_dicts, Config, Configured, AtomicConfig
from vectorbt.utils.template import Rep, Sub
from vectorbt.utils.decorators import CacheCondition, cached_property, cached_method
from vectorbt.utils.figure import Figure, FigureWidget, make_figure, make_subplots
from vectorbt.utils.random import set_seed
from vectorbt.utils.image import save_animation
from vectorbt.utils.schedule import AsyncJob, AsyncScheduler, CancelledError, ScheduleManager

__all__ = [
    'atomic_dict',
    'merge_dicts',
    'Config',
    'Configured',
    'AtomicConfig',
    'Rep',
    'Sub',
    'CacheCondition',
    'cached_property',
    'cached_method',
    'Figure',
    'FigureWidget',
    'make_figure',
    'make_subplots',
    'set_seed',
    'save_animation',
    'AsyncJob',
    'AsyncScheduler',
    'CancelledError',
    'ScheduleManager'
]

__pdoc__ = {k: False for k in __all__}
