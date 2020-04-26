"""On-balance volume (OBV) is a technical trading momentum indicator that uses volume flow to predict 
changes in stock price.

See [On-Balance Volume (OBV)](https://www.investopedia.com/terms/o/onbalancevolume.asp).

Use `OBV.from_params` methods to run the indicator."""

import numpy as np
from numba import njit
from numba.types import f8
from vectorbt.indicators.indicator_factory import IndicatorFactory
from vectorbt.utils import *

__all__ = ['OBV']


@njit(f8[:, :](f8[:, :], f8[:, :]))
def obv_custom_func_nb(close_ts, volume_ts):
    obv = np.full_like(close_ts, np.nan)
    for col in range(close_ts.shape[1]):
        cumsum = 0
        for i in range(1, close_ts.shape[0]):
            if np.isnan(close_ts[i, col]) or np.isnan(close_ts[i-1, col]) or np.isnan(volume_ts[i, col]):
                continue
            if close_ts[i, col] > close_ts[i-1, col]:
                cumsum += volume_ts[i, col]
            elif close_ts[i, col] < close_ts[i-1, col]:
                cumsum += -volume_ts[i, col]
            obv[i, col] = cumsum
    return obv


def obv_custom_func(close_ts, volume_ts):
    return obv_custom_func_nb(close_ts.vbt.to_2d_array(), volume_ts.vbt.to_2d_array())


OBV = IndicatorFactory(
    ts_names=['close_ts', 'volume_ts'],
    param_names=[],
    output_names=['obv'],
    name='obv'
).from_custom_func(obv_custom_func)


class OBV(OBV):
    @classmethod
    def from_params(cls, close_ts, volume_ts):
        """Calculate on-balance volume `OBV.obv` from time series `close_ts` and `volume_ts`, and no parameters.

        Args:
            close_ts (pandas_like): The last closing price.
            volume_ts (pandas_like): The volume.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.indicator_factory.from_params_pipeline.`
        Returns:
            OBV
        Examples:
            ```python-repl
            >>> obv = vbt.OBV.from_params(price['Close'], price['Volume'])

            >>> print(obv.obv)
            Date
            2019-02-28             NaN
            2019-03-01    7.661248e+09
            2019-03-02    1.524003e+10
            2019-03-03    7.986476e+09
            2019-03-04   -1.042700e+09
                                   ...     
            2019-08-27    5.613088e+11
            2019-08-28    5.437050e+11
            2019-08-29    5.266592e+11
            2019-08-30    5.402544e+11
            2019-08-31    5.517092e+11
            Name: (Close, Volume), Length: 185, dtype: float64
            ```
        """
        return super().from_params(close_ts, volume_ts)

    def plot(self,
             obv_trace_kwargs={},
             fig=None,
             **layout_kwargs):
        """Plot `OBV.obv`.

        Args:
            obv_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of `OBV.obv`.
            fig (plotly.graph_objects.Figure, optional): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Returns:
            vectorbt.widgets.FigureWidget
        Examples:
            ```py
            obv.plot()
            ```

            ![](img/OBV.png)"""
        check_type(self.obv, pd.Series)

        obv_trace_kwargs = {**dict(
            name=f'OBV ({self.name})'
        ), **obv_trace_kwargs}

        fig = self.obv.vbt.timeseries.plot(trace_kwargs=obv_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_pdoc(OBV)
