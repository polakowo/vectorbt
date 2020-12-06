"""Root pandas accessors.

An accessor adds additional “namespace” to pandas objects.

The `vectorbt.root_accessors` registers a custom `vbt` accessor on top of each `pd.Series`
and `pd.DataFrame` object. It is the main entry point for all other accessors:

```plaintext
vbt.base.accessors.Base_SR/DFAccessor           -> pd.Series/DataFrame.vbt.*
vbt.generic.accessors.Generic_SR/DFAccessor     -> pd.Series/DataFrame.vbt.*
vbt.signals.accessors.Signals_SR/DFAccessor     -> pd.Series/DataFrame.vbt.signals.*
vbt.returns.accessors.Returns_SR/DFAccessor     -> pd.Series/DataFrame.vbt.returns.*
vbt.ohlcv.accessors.OHLCV_DFAccessor            -> pd.DataFrame.vbt.ohlcv.*
```

Additionally, some accessors subclass other accessors building the following inheritance hiearchy:

```plaintext
vbt.base.accessors.Base_SR/DFAccessor
    -> vbt.generic.accessors.Generic_SR/DFAccessor
        -> vbt.signals.accessors.Signals_SR/DFAccessor
        -> vbt.returns.accessors.Returns_SR/DFAccessor
        -> vbt.ohlcv.accessors.OHLCV_DFAccessor
```

So, for example, the method `pd.Series.vbt.to_2d_array` is also available as
`pd.Series.vbt.returns.to_2d_array`."""

import pandas as pd
from pandas.core.accessor import _register_accessor, DirNamesMixin

from vectorbt.generic.accessors import Generic_SRAccessor, Generic_DFAccessor


# By subclassing DirNamesMixin, we can build accessors on top of each other
@pd.api.extensions.register_series_accessor("vbt")
class Vbt_SRAccessor(DirNamesMixin, Generic_SRAccessor):
    """The main vectorbt accessor for `pd.Series`."""

    def __init__(self, obj, **kwargs):
        self._obj = obj

        DirNamesMixin.__init__(self)
        Generic_SRAccessor.__init__(self, obj, **kwargs)


@pd.api.extensions.register_dataframe_accessor("vbt")
class Vbt_DFAccessor(DirNamesMixin, Generic_DFAccessor):
    """The main vectorbt accessor for `pd.DataFrame`."""

    def __init__(self, obj, **kwargs):
        self._obj = obj

        DirNamesMixin.__init__(self)
        Generic_DFAccessor.__init__(self, obj, **kwargs)


def register_dataframe_accessor(name):
    """Decorator to register a custom `pd.DataFrame` accessor on top of the `vbt` accessor."""
    return _register_accessor(name, Vbt_DFAccessor)


def register_series_accessor(name):
    """Decorator to register a custom `pd.Series` accessor on top of the `vbt` accessor."""
    return _register_accessor(name, Vbt_SRAccessor)
