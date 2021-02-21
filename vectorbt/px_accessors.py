"""Plotly Express pandas accessors."""

import pandas as pd
from inspect import getmembers, isfunction
import plotly.express as px

from vectorbt.root_accessors import register_dataframe_accessor, register_series_accessor
from vectorbt.utils import checks
from vectorbt.utils.widgets import FigureWidget
from vectorbt.utils.config import merge_dicts
from vectorbt.base.accessors import BaseAccessor, BaseDFAccessor, BaseSRAccessor
from vectorbt.generic.plotting import clean_labels


def add_px_methods(cls):
    """Class decorator to add Plotly Express methods that accept a DataFrame as first argument."""

    for px_func_name, px_func in getmembers(px, isfunction):
        if checks.method_accepts_argument(px_func, 'data_frame') or px_func_name == 'imshow':
            def plot_func(self, *args, px_func_name=px_func_name, px_func=px_func, **kwargs):
                from vectorbt.settings import layout

                layout_kwargs = dict(
                    template=kwargs.pop('template', layout['template']),
                    width=kwargs.pop('width', layout['width']),
                    height=kwargs.pop('height', layout['height'])
                )
                # Fix category_orders
                if 'color' in kwargs:
                    if isinstance(kwargs['color'], str):
                        if isinstance(self._obj, pd.DataFrame):
                            if kwargs['color'] in self._obj.columns:
                                category_orders = dict()
                                category_orders[kwargs['color']] = sorted(self._obj[kwargs['color']].unique())
                                kwargs = merge_dicts(dict(category_orders=category_orders), kwargs)

                # Fix Series name
                obj = self._obj.copy(deep=False)
                if isinstance(obj, pd.Series):
                    if obj.name is not None:
                        obj = obj.rename(str(obj.name))
                else:
                    obj.columns = clean_labels(obj.columns)
                obj.index = clean_labels(obj.index)

                if px_func_name == 'imshow':
                    return FigureWidget(px_func(
                        obj.vbt.to_2d_array(), *args, **layout_kwargs, **kwargs
                    ), layout=layout_kwargs)
                return FigureWidget(px_func(
                    obj, *args, **layout_kwargs, **kwargs
                ), layout=layout_kwargs)

            setattr(cls, px_func_name, plot_func)
    return cls


@add_px_methods
class PXAccessor(BaseAccessor):
    """Accessor for running Plotly Express functions.

    Accessible through `pd.Series.vbt.px` and `pd.DataFrame.vbt.px`.

    ## Example

    ```python-repl
    >>> import pandas as pd
    >>> import vectorbt as vbt

    >>> vbt.settings.set_theme('seaborn')

    >>> pd.Series([1, 2, 3]).vbt.px.bar()
    ```

    ![](/vectorbt/docs/img/px_bar.png)
    """

    def __init__(self, obj, **kwargs):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        BaseAccessor.__init__(self, obj, **kwargs)


@register_series_accessor('px')
class PXSRAccessor(PXAccessor, BaseSRAccessor):
    """Accessor for running Plotly Express functions. For Series only.

    Accessible through `pd.Series.vbt.px`."""

    def __init__(self, obj, **kwargs):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        BaseSRAccessor.__init__(self, obj, **kwargs)
        PXAccessor.__init__(self, obj, **kwargs)


@register_dataframe_accessor('px')
class PXDFAccessor(PXAccessor, BaseDFAccessor):
    """Accessor for running Plotly Express functions. For DataFrames only.

    Accessible through `pd.DataFrame.vbt.px`."""

    def __init__(self, obj, **kwargs):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        BaseDFAccessor.__init__(self, obj, **kwargs)
        PXAccessor.__init__(self, obj, **kwargs)
