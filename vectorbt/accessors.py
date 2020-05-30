"""Root pandas accessors."""

import pandas as pd
from pandas.core.accessor import _register_accessor, DirNamesMixin
from vectorbt.utils.accessors import Base_DFAccessor, Base_SRAccessor


@pd.api.extensions.register_dataframe_accessor("vbt")
class Vbt_DFAccessor(DirNamesMixin, Base_DFAccessor):
    """The main `vectorbt` accessor for `pandas.DataFrame`."""
    # By subclassing DirNamesMixin, we can build accessors on top of each other
    def __init__(self, obj):
        self._obj = obj

        DirNamesMixin.__init__(self)
        Base_DFAccessor.__init__(self, self)


@pd.api.extensions.register_series_accessor("vbt")
class Vbt_SRAccessor(DirNamesMixin, Base_SRAccessor):
    """The main `vectorbt` accessor for `pandas.Series`."""
    def __init__(self, obj):
        self._obj = obj
        
        DirNamesMixin.__init__(self)
        Base_SRAccessor.__init__(self, self)


def register_dataframe_accessor(name):
    """Decorator to register a custom `pandas.DataFrame` accessor on top of the `vbt` accessor."""
    return _register_accessor(name, Vbt_DFAccessor)


def register_series_accessor(name):
    """Decorator to register a custom `pandas.Series` accessor on top of the `vbt` accessor."""
    return _register_accessor(name, Vbt_SRAccessor)
