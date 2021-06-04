"""Signal generators built with `vectorbt.signals.factory.SignalFactory`."""

import numpy as np
import plotly.graph_objects as go

from vectorbt import _typing as tp
from vectorbt.utils.config import Config
from vectorbt.utils.figure import make_figure
from vectorbt.indicators.configs import flex_col_param_config, flex_elem_param_config
from vectorbt.signals.enums import StopType
from vectorbt.signals.factory import SignalFactory
from vectorbt.signals.nb import (
    rand_enex_apply_nb,
    rand_by_prob_choice_nb,
    stop_choice_nb,
    ohlc_stop_choice_nb
)

# ############# Random signals ############# #


RAND = SignalFactory(
    class_name='RAND',
    module_name=__name__,
    short_name='rand',
    param_names=['n']
).from_apply_func(  # apply_func since function is (almost) vectorized
    rand_enex_apply_nb,
    require_input_shape=True,
    param_settings=dict(
        n=flex_col_param_config
    ),
    kwargs_to_args=['entry_wait', 'exit_wait'],
    entry_wait=1,
    exit_wait=1,
    seed=None
)


class _RAND(RAND):
    """Random entry and exit signal generator based on the number of signals.

    Generates `entries` and `exits` based on `vectorbt.signals.nb.rand_enex_apply_nb`.

    !!! hint
        Parameter `n` can be either a single value (per frame) or a NumPy array (per column).
        To generate multiple combinations, pass it as a list.

    ## Example

    Test three different `n` values:
    ```python-repl
    >>> import vectorbt as vbt

    >>> rand = vbt.RAND.run(
    ...     input_shape=(6,),
    ...     n=[1, 2, 3],
    ...     seed=42)

    >>> rand.entries
    rand_n      1      2      3
    0        True   True   True
    1       False  False  False
    2       False   True   True
    3       False  False  False
    4       False  False   True
    5       False  False  False

    >>> rand.exits
    rand_n      1      2      3
    0       False  False  False
    1        True   True   True
    2       False  False  False
    3       False   True   True
    4       False  False  False
    5       False  False   True
    ```

    `n` can also be set per column:
    ```python-repl
    >>> import numpy as np

    >>> rand = vbt.RAND.run(
    ...     input_shape=(8, 2),
    ...     n=[np.array([1, 2]), 3],
    ...     seed=42)

    >>> rand.entries
    rand_n      1      2             3
                0      1      0      1
    0       False   True   True   True
    1        True  False  False  False
    2       False  False  False  False
    3       False  False   True  False
    4       False   True  False   True
    5       False  False   True  False
    6       False  False  False   True
    7       False  False  False  False

    >>> rand.exits
    rand_n      1      2             3
                0      1      0      1
    0       False  False  False  False
    1       False  False   True  False
    2       False  False  False   True
    3       False   True  False  False
    4       False  False   True  False
    5        True  False  False   True
    6       False  False   True  False
    7       False   True  False   True
    ```
    """
    pass


setattr(RAND, '__doc__', _RAND.__doc__)

RPROB = SignalFactory(
    class_name='RPROB',
    module_name=__name__,
    short_name='rprob',
    param_names=['entry_prob', 'exit_prob']
).from_choice_func(
    require_input_shape=True,
    entry_choice_func=rand_by_prob_choice_nb,
    entry_settings=dict(
        pass_params=['entry_prob'],
        pass_kwargs=['first', 'temp_idx_arr', 'flex_2d']
    ),
    exit_choice_func=rand_by_prob_choice_nb,
    exit_settings=dict(
        pass_params=['exit_prob'],
        pass_kwargs=['first', 'temp_idx_arr', 'flex_2d']
    ),
    pass_flex_2d=True,
    param_settings=dict(
        entry_prob=flex_elem_param_config,
        exit_prob=flex_elem_param_config
    ),
    seed=None
)


class _RPROB(RPROB):
    """Random entry and exit signal generator based on probabilities.

    Generates `entries` and `exits` based on `vectorbt.signals.nb.rand_by_prob_choice_nb`.

    !!! hint
        All parameters can be either a single value (per frame) or a NumPy array (per row, column,
        or element). To generate multiple combinations, pass them as lists.

    ## Example

    Test all probability combinations:
    ```python-repl
    >>> import vectorbt as vbt

    >>> rprob = vbt.RPROB.run(
    ...     input_shape=(5,),
    ...     entry_prob=[0.5, 1.],
    ...     exit_prob=[0.5, 1.],
    ...     param_product=True,
    ...     seed=42)

    >>> rprob.entries
    rprob_entry_prob           0.5           1.0
    rprob_exit_prob     0.5    1.0    0.5    1.0
    0                  True   True   True   True
    1                 False  False  False  False
    2                 False  False  False   True
    3                 False  False  False  False
    4                 False  False   True   True

    >>> rprob.exits
    rprob_entry_prob           0.5           1.0
    rprob_exit_prob     0.5    1.0    0.5    1.0
    0                 False  False  False  False
    1                 False   True  False   True
    2                 False  False  False  False
    3                 False  False   True   True
    4                  True  False  False  False
    ```

    `entry_prob` and `exit_prob` can also be set per row, column, or element:
    ```python-repl
    >>> import numpy as np

    >>> entry_prob1 = np.asarray([1., 0., 1., 0., 1.])
    >>> entry_prob2 = np.asarray([0., 1., 0., 1., 0.])
    >>> rprob = vbt.RPROB.run(
    ...     input_shape=(5,),
    ...     entry_prob=[entry_prob1, entry_prob2],
    ...     exit_prob=1.,
    ...     seed=42)

    >>> rprob.entries
    rprob_entry_prob array_0 array_1
    rprob_exit_prob      1.0     1.0
    0                   True   False
    1                  False    True
    2                   True   False
    3                  False    True
    4                   True   False

    >>> rprob.exits
    rprob_entry_prob array_0 array_1
    rprob_exit_prob      1.0     1.0
    0                  False   False
    1                   True   False
    2                  False    True
    3                   True   False
    4                  False    True
    ```
    """
    pass


setattr(RPROB, '__doc__', _RPROB.__doc__)

rprobex_config = Config(
    dict(
        class_name='RPROBEX',
        module_name=__name__,
        short_name='rprobex',
        param_names=['prob'],
        exit_only=True,
        iteratively=False
    )
)
"""Factory config for `RPROBEX`."""

rprobex_func_config = Config(
    dict(
        exit_choice_func=rand_by_prob_choice_nb,
        exit_settings=dict(
            pass_params=['prob'],
            pass_kwargs=['first', 'temp_idx_arr', 'flex_2d']
        ),
        pass_flex_2d=True,
        param_settings=dict(
            prob=flex_elem_param_config
        ),
        seed=None
    )
)
"""Exit function config for `RPROBEX`."""

RPROBEX = SignalFactory(
    **rprobex_config
).from_choice_func(
    **rprobex_func_config
)


class _RPROBEX(RPROBEX):
    """Random exit signal generator based on probabilities.

    Generates `exits` based on `entries` and `vectorbt.signals.nb.rand_by_prob_choice_nb`.

    See `RPROB` for notes on parameters."""
    pass


setattr(RPROBEX, '__doc__', _RPROBEX.__doc__)

IRPROBEX = SignalFactory(
    **rprobex_config.merge_with(
        dict(
            class_name='IRPROBEX',
            short_name='irprobex',
            iteratively=True
        )
    )
).from_choice_func(
    **rprobex_func_config
)


class _IRPROBEX(IRPROBEX):
    """Random exit signal generator based on probabilities.

    Iteratively generates `new_entries` and `exits` based on `entries` and
    `vectorbt.signals.nb.rand_by_prob_choice_nb`.

    See `RPROB` for notes on parameters."""
    pass


setattr(IRPROBEX, '__doc__', _IRPROBEX.__doc__)

# ############# Stop signals ############# #

stex_config = Config(
    dict(
        class_name='STEX',
        module_name=__name__,
        short_name='stex',
        input_names=['ts'],
        param_names=['stop', 'trailing'],
        exit_only=True,
        iteratively=False
    )
)
"""Factory config for `STEX`."""

stex_func_config = Config(
    dict(
        exit_choice_func=stop_choice_nb,
        exit_settings=dict(
            pass_inputs=['ts'],
            pass_params=['stop', 'trailing'],
            pass_kwargs=['wait', 'first', 'temp_idx_arr', 'flex_2d']
        ),
        pass_flex_2d=True,
        param_settings=dict(
            stop=flex_elem_param_config,
            trailing=flex_elem_param_config
        ),
        trailing=False
    )
)
"""Exit function config for `STEX`."""

STEX = SignalFactory(
    **stex_config
).from_choice_func(
    **stex_func_config
)


class _STEX(STEX):
    """Exit signal generator based on stop values.

    Generates `exits` based on `entries` and `vectorbt.signals.nb.stop_choice_nb`.

    !!! hint
        All parameters can be either a single value (per frame) or a NumPy array (per row, column,
        or element). To generate multiple combinations, pass them as lists."""
    pass


setattr(STEX, '__doc__', _STEX.__doc__)

ISTEX = SignalFactory(
    **stex_config.merge_with(
        dict(
            class_name='ISTEX',
            short_name='istex',
            iteratively=True
        )
    )
).from_choice_func(
    **stex_func_config
)


class _ISTEX(ISTEX):
    """Exit signal generator based on stop values.

    Iteratively generates `new_entries` and `exits` based on `entries` and
    `vectorbt.signals.nb.stop_choice_nb`.

    See `STEX` for notes on parameters."""
    pass


setattr(ISTEX, '__doc__', _ISTEX.__doc__)

# ############# OHLC stop signals ############# #

ohlcstex_config = Config(
    dict(
        class_name='OHLCSTEX',
        module_name=__name__,
        short_name='ohlcstex',
        input_names=['open', 'high', 'low', 'close'],
        in_output_names=['hit_price', 'stop_type'],
        param_names=['sl_stop', 'ts_stop', 'tp_stop'],
        attr_settings=dict(
            stop_type=dict(dtype=StopType)  # creates rand_type_readable
        ),
        exit_only=True,
        iteratively=False
    )
)
"""Factory config for `OHLCSTEX`."""

ohlcstex_func_config = Config(
    dict(
        exit_choice_func=ohlc_stop_choice_nb,
        exit_settings=dict(
            pass_inputs=['open', 'high', 'low', 'close'],  # do not pass entries
            pass_in_outputs=['hit_price', 'stop_type'],
            pass_params=['sl_stop', 'ts_stop', 'tp_stop'],
            pass_kwargs=[('is_open_safe', True), 'wait', 'first', 'temp_idx_arr', 'flex_2d'],
        ),
        pass_flex_2d=True,
        in_output_settings=dict(
            hit_price=dict(
                dtype=np.float_
            ),
            stop_type=dict(
                dtype=np.int_
            )
        ),
        param_settings=dict(
            sl_stop=flex_elem_param_config,
            ts_stop=flex_elem_param_config,
            tp_stop=flex_elem_param_config
        ),
        sl_stop=np.nan,
        ts_stop=np.nan,
        tp_stop=np.nan,
        hit_price=np.nan,
        stop_type=-1
    )
)
"""Exit function config for `OHLCSTEX`."""

OHLCSTEX = SignalFactory(
    **ohlcstex_config
).from_choice_func(
    **ohlcstex_func_config
)


def _bind_ohlcstex_plot(base_cls: type, entries_attr: str) -> tp.Callable:  # pragma: no cover

    base_cls_plot = base_cls.plot

    def plot(self,
             plot_type: tp.Union[None, str, tp.BaseTraceType] = None,
             ohlc_kwargs: tp.KwargsLike = None,
             entry_trace_kwargs: tp.KwargsLike = None,
             exit_trace_kwargs: tp.KwargsLike = None,
             add_trace_kwargs: tp.KwargsLike = None,
             fig: tp.Optional[tp.BaseFigure] = None,
             _base_cls_plot: tp.Callable = base_cls_plot,
             **layout_kwargs) -> tp.BaseFigure:  # pragma: no cover
        from vectorbt._settings import settings
        ohlcv_cfg = settings['ohlcv']
        plotting_cfg = settings['plotting']

        if self.wrapper.ndim > 1:
            raise TypeError("Select a column first. Use indexing.")

        if ohlc_kwargs is None:
            ohlc_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        if fig is None:
            fig = make_figure()
            fig.update_layout(
                showlegend=True,
                xaxis_rangeslider_visible=False,
                xaxis_showgrid=True,
                yaxis_showgrid=True
            )
        fig.update_layout(**layout_kwargs)

        if plot_type is None:
            plot_type = ohlcv_cfg['plot_type']
        if isinstance(plot_type, str):
            if plot_type.lower() == 'ohlc':
                plot_type = 'OHLC'
                plot_obj = go.Ohlc
            elif plot_type.lower() == 'candlestick':
                plot_type = 'Candlestick'
                plot_obj = go.Candlestick
            else:
                raise ValueError("Plot type can be either 'OHLC' or 'Candlestick'")
        else:
            plot_obj = plot_type
        ohlc = plot_obj(
            x=self.wrapper.index,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            name=plot_type,
            increasing=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['increasing']
                )
            ),
            decreasing=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['decreasing']
                )
            )
        )
        ohlc.update(**ohlc_kwargs)
        fig.add_trace(ohlc, **add_trace_kwargs)

        # Plot entry and exit markers
        _base_cls_plot(
            self,
            entry_y=self.open,
            exit_y=self.hit_price,
            exit_types=self.stop_type_readable,
            entry_trace_kwargs=entry_trace_kwargs,
            exit_trace_kwargs=exit_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig
        )
        return fig

    plot.__doc__ = """Plot OHLC, `{0}.{1}` and `{0}.exits`.
    
    Args:
        plot_type: Either 'OHLC', 'Candlestick' or Plotly trace.
        ohlc_kwargs (dict): Keyword arguments passed to `plot_type`.
        entry_trace_kwargs (dict): Keyword arguments passed to \
        `vectorbt.signals.accessors.SignalsSRAccessor.plot_as_entry_markers` for `{0}.{1}`.
        exit_trace_kwargs (dict): Keyword arguments passed to \
        `vectorbt.signals.accessors.SignalsSRAccessor.plot_as_exit_markers` for `{0}.exits`.
        fig (Figure or FigureWidget): Figure to add traces to.
        **layout_kwargs: Keyword arguments for layout.""".format(base_cls.__name__, entries_attr)

    if entries_attr == 'entries':
        plot.__doc__ += """
    ## Example
        
    ```python-repl
    >>> ohlcstex.iloc[:, 0].plot()
    ```
    
    ![](/vectorbt/docs/img/ohlcstex.svg)
    """
    return plot


class _OHLCSTEX(OHLCSTEX):
    """Advanced exit signal generator based on stop values.

    Generates `exits` based on `entries` and `vectorbt.signals.nb.ohlc_stop_choice_nb`.

    !!! hint
        All parameters can be either a single value (per frame) or a NumPy array (per row, column,
        or element). To generate multiple combinations, pass them as lists.

    ## Example

    Test each stop type individually:
    ```python-repl
    >>> import vectorbt as vbt
    >>> import pandas as pd

    >>> entries = pd.Series([True, False, False, False, False])
    >>> price = pd.DataFrame({
    ...     'open': [10, 11, 12, 11, 10],
    ...     'high': [11, 12, 13, 12, 11],
    ...     'low': [9, 10, 11, 10, 9],
    ...     'close': [10, 11, 12, 11, 10]
    ... })
    >>> ohlcstex = vbt.OHLCSTEX.run(
    ...     entries, price['open'], price['high'], price['low'], price['close'],
    ...     sl_stop=[0.1, 0., 0.], ts_stop=[0., 0.1, 0.], tp_stop=[0., 0., 0.1])

    >>> ohlcstex.entries
    ohlcstex_sl_stop    0.1    0.0    0.0
    ohlcstex_ts_stop    0.0    0.1    0.0
    ohlcstex_tp_stop    0.0    0.0    0.1
    0                  True   True   True
    1                 False  False  False
    2                 False  False  False
    3                 False  False  False
    4                 False  False  False

    >>> ohlcstex.exits
    ohlcstex_sl_stop    0.1    0.0    0.0
    ohlcstex_ts_stop    0.0    0.1    0.0
    ohlcstex_tp_stop    0.0    0.0    0.1
    0                 False  False  False
    1                 False  False   True
    2                 False  False  False
    3                 False   True  False
    4                  True  False  False

    >>> ohlcstex.hit_price
    ohlcstex_sl_stop  0.1   0.0   0.0
    ohlcstex_ts_stop  0.0   0.1   0.0
    ohlcstex_tp_stop  0.0   0.0   0.1
    0                 NaN   NaN   NaN
    1                 NaN   NaN  11.0
    2                 NaN   NaN   NaN
    3                 NaN  11.7   NaN
    4                 9.0   NaN   NaN

    >>> ohlcstex.stop_type_readable
    ohlcstex_sl_stop       0.1        0.0         0.0
    ohlcstex_ts_stop       0.0        0.1         0.0
    ohlcstex_tp_stop       0.0        0.0         0.1
    0
    1                                      TakeProfit
    2
    3                           TrailStop
    4                 StopLoss
    ```
    """

    plot = _bind_ohlcstex_plot(OHLCSTEX, 'entries')


setattr(OHLCSTEX, '__doc__', _OHLCSTEX.__doc__)
setattr(OHLCSTEX, 'plot', _OHLCSTEX.plot)

IOHLCSTEX = SignalFactory(
    **ohlcstex_config.merge_with(
        dict(
            class_name='IOHLCSTEX',
            short_name='iohlcstex',
            iteratively=True
        )
    )
).from_choice_func(
    **ohlcstex_func_config
)


class _IOHLCSTEX(IOHLCSTEX):
    """Advanced exit signal generator based on stop values.

    Iteratively generates `new_entries` and `exits` based on `entries` and
    `vectorbt.signals.nb.ohlc_stop_choice_nb`.

    See `OHLCSTEX` for notes on parameters."""

    plot = _bind_ohlcstex_plot(IOHLCSTEX, 'new_entries')


setattr(IOHLCSTEX, '__doc__', _IOHLCSTEX.__doc__)
setattr(IOHLCSTEX, 'plot', _IOHLCSTEX.plot)
