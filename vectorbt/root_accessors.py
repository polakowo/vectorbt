"""Root pandas accessors.

An accessor adds additional “namespace” to pandas objects.

The `vectorbt.root_accessors` registers a custom `vbt` accessor on top of each `pd.Series`
and `pd.DataFrame` object. It is the main entry point for all other accessors:

```plaintext
vbt.base.accessors.BaseSR/DFAccessor           -> pd.Series/DataFrame.vbt.*
vbt.generic.accessors.GenericSR/DFAccessor     -> pd.Series/DataFrame.vbt.*
vbt.signals.accessors.SignalsSR/DFAccessor     -> pd.Series/DataFrame.vbt.signals.*
vbt.returns.accessors.ReturnsSR/DFAccessor     -> pd.Series/DataFrame.vbt.returns.*
vbt.ohlcv.accessors.OHLCVDFAccessor            -> pd.DataFrame.vbt.ohlcv.*
vbt.px_accessors.PXAccessor                    -> pd.DataFrame.vbt.px.*
```

Additionally, some accessors subclass other accessors building the following inheritance hiearchy:

```plaintext
vbt.base.accessors.BaseSR/DFAccessor
    -> vbt.generic.accessors.GenericSR/DFAccessor
        -> vbt.signals.accessors.SignalsSR/DFAccessor
        -> vbt.returns.accessors.ReturnsSR/DFAccessor
        -> vbt.ohlcv.accessors.OHLCVDFAccessor
    -> vbt.px_accessors.PXSR/DFAccessor
```

So, for example, the method `pd.Series.vbt.to_2d_array` is also available as
`pd.Series.vbt.returns.to_2d_array`."""

import pandas as pd
from pandas.core.accessor import _register_accessor, DirNamesMixin

from vectorbt import _typing as tp
from vectorbt.generic.accessors import GenericSRAccessor, GenericDFAccessor


# By subclassing DirNamesMixin, we can build accessors on top of each other
@pd.api.extensions.register_series_accessor("vbt")
class Vbt_SRAccessor(DirNamesMixin, GenericSRAccessor):
    """The main vectorbt accessor for `pd.Series`."""

    def __init__(self, obj: tp.Series, **kwargs) -> None:
        self._obj = obj

        DirNamesMixin.__init__(self)
        GenericSRAccessor.__init__(self, obj, **kwargs)


@pd.api.extensions.register_dataframe_accessor("vbt")
class Vbt_DFAccessor(DirNamesMixin, GenericDFAccessor):
    """The main vectorbt accessor for `pd.DataFrame`."""

    def __init__(self, obj: tp.Frame, **kwargs) -> None:
        self._obj = obj

        DirNamesMixin.__init__(self)
        GenericDFAccessor.__init__(self, obj, **kwargs)


def register_dataframe_accessor(name: str) -> tp.Callable:
    """Decorator to register a custom `pd.DataFrame` accessor on top of the `vbt` accessor."""
    return _register_accessor(name, Vbt_DFAccessor)


def register_series_accessor(name: str) -> tp.Callable:
    """Decorator to register a custom `pd.Series` accessor on top of the `vbt` accessor."""
    return _register_accessor(name, Vbt_SRAccessor)
