"""A moving average (MA) is a widely used indicator in technical analysis that helps smooth out 
price action by filtering out the “noise” from random short-term price fluctuations. 

See [Moving Average (MA)](https://www.investopedia.com/terms/m/movingaverage.asp).

Use `MA.from_params` or `MA.from_combinations` methods to run the indicator."""

import numpy as np
from numba import njit
from numba.types import UniTuple, f8, i8, b1, DictType
from vectorbt.timeseries import ewm_mean_nb, rolling_mean_nb
from vectorbt.indicators.indicator_factory import IndicatorFactory
from vectorbt.utils import *

__all__ = ['MA']


@njit(DictType(UniTuple(i8, 2), f8[:, :])(f8[:, :], i8[:], b1[:]), cache=True)
def ma_caching_nb(ts, windows, ewms):
    cache_dict = dict()
    for i in range(windows.shape[0]):
        if (windows[i], int(ewms[i])) not in cache_dict:
            if ewms[i]:
                ma = ewm_mean_nb(ts, windows[i])
            else:
                ma = rolling_mean_nb(ts, windows[i])
            cache_dict[(windows[i], int(ewms[i]))] = ma
    return cache_dict


@njit(f8[:, :](f8[:, :], i8, b1, DictType(UniTuple(i8, 2), f8[:, :])), cache=True)
def ma_apply_func_nb(ts, window, ewm, cache_dict):
    return cache_dict[(window, int(ewm))]


MA = IndicatorFactory(
    ts_names=['ts'],
    param_names=['window', 'ewm'],
    output_names=['ma'],
    name='ma'
).from_apply_func(ma_apply_func_nb, caching_func=ma_caching_nb)


class MA(MA):
    @classmethod
    def from_params(cls, ts, window, ewm=False, **kwargs):
        """Calculate moving average `MA.ma` from time series `ts` and parameters `window` and `ewm`.

        Args:
            ts (pandas_like): Time series (such as price).
            window (int or array_like): Size of the moving window. Can be one or more values.
            ewm (bool or array_like): If True, uses exponential moving average, otherwise 
                simple moving average. Can be one or more values. Defaults to False.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.indicator_factory.from_params_pipeline.`
        Returns:
            MA
        Examples:
            ```python-repl
            >>> ma = vbt.MA.from_params(price['Close'], [10, 20], ewm=[False, True])

            >>> print(ma.ma)
            ma_window          10            20
            ma_ewm          False          True
            Date                               
            2019-02-28        NaN           NaN
            2019-03-01        NaN           NaN
            2019-03-02        NaN           NaN
            ...               ...           ...
            2019-08-29  10155.972  10330.457140
            2019-08-30  10039.466  10260.715507
            2019-08-31   9988.727  10200.710220

            [185 rows x 2 columns]
            ```
        """
        return super().from_params(ts, window, ewm, **kwargs)

    @classmethod
    def from_combinations(cls, ts, windows, r, ewm=False, names=None, **kwargs):
        """Create multiple `MA` combinations according to `itertools.combinations`.

        Args:
            ts (pandas_like): Time series (such as price).
            windows (array_like of int): Size of the moving window. Must be multiple.
            r (int): The number of `MA` instances to combine.
            ewm (bool or array_like of bool): If True, uses exponential moving average, otherwise 
                uses simple moving average. Can be one or more values. Defaults to False.
            names (list of str, optional): A list of names for each `MA` instance.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.indicator_factory.from_params_pipeline.`
        Returns:
            tuple of MA
        Examples:
            ```python-repl
            >>> fast_ma, slow_ma = vbt.MA.from_combinations(price['Close'], 
            ...     [10, 20, 30], 2, ewm=[False, False, True], names=['fast', 'slow'])

            >>> print(fast_ma.ma)
            fast_window         10         10          20
            fast_ewm         False      False       False
            Date                                         
            2019-02-28         NaN        NaN         NaN
            2019-03-01         NaN        NaN         NaN
            2019-03-02         NaN        NaN         NaN
            ...                ...        ...         ...
            2019-08-29   10155.972  10155.972  10447.3480
            2019-08-30   10039.466  10039.466  10359.5555
            2019-08-31    9988.727   9988.727  10264.9095

            [185 rows x 3 columns]

            >>> print(slow_ma.ma)
            slow_window          20            30            30
            slow_ewm          False          True          True
            Date                                               
            2019-02-28          NaN           NaN           NaN
            2019-03-01          NaN           NaN           NaN
            2019-03-02          NaN           NaN           NaN
            ...                 ...           ...           ...
            2019-08-29   10447.3480  10423.585970  10423.585970
            2019-08-30   10359.5555  10370.333077  10370.333077
            2019-08-31   10264.9095  10322.612024  10322.612024

            [185 rows x 3 columns]

            ```

            The naive way without caching is the follows:
            ```py
            window_combs = itertools.combinations([10, 20, 30], 2)
            ewm_combs = itertools.combinations([False, False, True], 2)
            fast_windows, slow_windows = np.asarray(list(window_combs)).transpose()
            fast_ewms, slow_ewms = np.asarray(list(ewm_combs)).transpose()

            fast_ma = vbt.MA.from_params(price['Close'], 
            ...     fast_windows, fast_ewms, name='fast')
            slow_ma = vbt.MA.from_params(price['Close'], 
            ...     slow_windows, slow_ewms, name='slow')
            ```

            Having this, you can now compare these `MA` instances:
            ```python-repl
            >>> entry_signals = fast_ma.ma_above(slow_ma, crossover=True)
            >>> exit_signals = fast_ma.ma_below(slow_ma, crossover=True)

            >>> print(entry_signals)
            fast_window     10     10     20
            fast_ewm     False  False  False
            slow_window     20     30     30
            slow_ewm     False  True    True
            Date                            
            2019-02-28   False  False  False
            2019-03-01   False  False  False
            2019-03-02   False  False  False
            ...            ...    ...    ...
            2019-08-29   False  False  False
            2019-08-30   False  False  False
            2019-08-31   False  False  False

            [185 rows x 3 columns]
            ```

            Notice how `MA.ma_above` method created a new column hierarchy for you. You can now use
            it for indexing as follows:

            ```py
            fig = price['Close'].vbt.timeseries.plot(name='Price')
            fig = entry_signals[(10, False, 20, False)]\\
                .vbt.signals.plot_markers(price['Close'], signal_type='entry', fig=fig)
            fig = exit_signals[(10, False, 20, False)]\\
                .vbt.signals.plot_markers(price['Close'], signal_type='exit', fig=fig)

            fig.show()
            ```
            ![](img/MA_from_combinations.png)
        """

        if names is None:
            names = ['ma' + str(i+1) for i in range(r)]
        windows, ewm = broadcast(windows, ewm, writeable=True)
        cache_dict = cls.from_params(ts, windows, ewm=ewm, return_cache=True, **kwargs)
        param_lists = zip(*itertools.combinations(zip(windows, ewm), r))
        mas = []
        for i, param_list in enumerate(param_lists):
            i_windows, i_ewm = zip(*param_list)
            mas.append(cls.from_params(ts, i_windows, ewm=i_ewm, cache=cache_dict, name=names[i], **kwargs))
        return tuple(mas)

    def plot(self,
             ts_trace_kwargs={},
             ma_trace_kwargs={},
             fig=None,
             **layout_kwargs):
        """Plot `MA.ma` against `MA.ts`.

        Args:
            ts_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of `MA.ts`.
            ma_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of `MA.ma`.
            fig (plotly.graph_objects.Figure, optional): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Returns:
            vectorbt.widgets.FigureWidget
        Examples:
            ```py
            ma[(10, False)].plot()
            ```

            ![](img/MA.png)"""
        check_type(self.ts, pd.Series)
        check_type(self.ma, pd.Series)

        ts_trace_kwargs = {**dict(
            name=f'Price ({self.name})'
        ), **ts_trace_kwargs}
        ma_trace_kwargs = {**dict(
            name=f'MA ({self.name})'
        ), **ma_trace_kwargs}

        fig = self.ts.vbt.timeseries.plot(trace_kwargs=ts_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.ma.vbt.timeseries.plot(trace_kwargs=ma_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_pdoc(MA)
