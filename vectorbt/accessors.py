import pandas as pd
from pandas.core.accessor import _register_accessor, DirNamesMixin
from vectorbt.utils import Base_DFAccessor, Base_SRAccessor


@pd.api.extensions.register_dataframe_accessor("vbt")
class Vbt_DFAccessor(DirNamesMixin, Base_DFAccessor):
    def __init__(self, obj):
        self._obj = obj


@pd.api.extensions.register_series_accessor("vbt")
class Vbt_SRAccessor(DirNamesMixin, Base_SRAccessor):
    def __init__(self, obj):
        self._obj = obj


def register_dataframe_accessor(name):
    return _register_accessor(name, Vbt_DFAccessor)


def register_series_accessor(name):
    return _register_accessor(name, Vbt_SRAccessor)
