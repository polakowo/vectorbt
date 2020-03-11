import numpy as np
import plotly.graph_objects as go
from vectorbt.utils import *
import matplotlib.pyplot as plt

__all__ = ['Gauge', 'Bar', 'Scatter', 'Histogram', 'Heatmap']


class FigureWidget(go.FigureWidget):
    """Subclass of the graph_objects.FigureWidget class with default params."""

    def __init__(self):
        super().__init__()
        # Good default params
        self.update_layout(
            autosize=False,
            width=700,
            height=300,
            margin=go.layout.Margin(
                b=30,
                t=30
            ),
            hovermode='closest'
        )
        # You can then update them using update_layout

    def show_png(self):
        width = self.layout.width
        height = self.layout.height
        self.show(renderer="png", width=width, height=height)


class UpdatableFigureWidget(FigureWidget):
    """Subclass of the FigureWidget class with abstract update method."""

    def update_data(self, *args, **kwargs):
        raise NotImplementedError

# ############# Gauge ############# #


def rgb_from_cmap(cmap_name, value, vrange):
    """Map vrange to colormap and get RGB of the value from that range."""
    if vrange[0] == vrange[1]:
        norm_value = 0.5
    else:
        norm_value = (value - vrange[0]) / (vrange[1] - vrange[0])
    cmap = plt.get_cmap(cmap_name)
    return "rgb(%d,%d,%d)" % tuple(np.round(np.asarray(cmap(norm_value))[:3] * 255))


class Gauge(UpdatableFigureWidget):
    """Accepts a single value and draws an indicator."""

    def __init__(self,
                 value=None,
                 label=None,
                 cmap_name='Spectral',
                 indicator_kwargs={},
                 **layout_kwargs):
        self._label = label
        self._vrange = None
        self._cmap_name = cmap_name
        self._indicator_kwargs = indicator_kwargs

        super().__init__()
        self.update_layout(width=500, height=300)
        self.update_layout(**layout_kwargs)

        if value is not None:
            self.update_data(value)

    def update_data(self, value):
        # Checks and preprocessing
        # NOTE: If called by Plotly event handler and in case of error, this won't visible in a notebook cell, but in logs!
        check_type(value, (int, float, complex))

        if self._vrange is None:
            self._vrange = value, value
        else:
            self._vrange = min(self._vrange[0], value), max(self._vrange[1], value)
        with self.batch_update():
            if len(self.data) == 0:
                indicator = go.Indicator(
                    domain=dict(x=[0, 1], y=[0, 1]),
                    mode="gauge+number+delta",
                    title=dict(text=self._label)
                )
                indicator.update(**self._indicator_kwargs)
                self.add_trace(indicator)
            indicator = self.data[0]
            if self._vrange is not None:
                indicator.gauge.axis.range = self._vrange
                if self._cmap_name is not None:
                    indicator.gauge.bar.color = rgb_from_cmap(self._cmap_name, value, self._vrange)
            indicator.delta.reference = indicator.value
            indicator.value = value

# ############# Bar ############# #


class Bar(UpdatableFigureWidget):
    """Draw a barplot for each value in a series."""

    def __init__(self,
                 sr=None,
                 bar_kwargs={},
                 **layout_kwargs):
        self._bar_kwargs = bar_kwargs

        super().__init__()
        self.update_layout(**layout_kwargs)

        if sr is not None:
            self.update_data(sr)

    def update_data(self, sr):
        # Checks and preprocessing
        check_type(sr, pd.Series)

        with self.batch_update():
            if len(self.data) == 0:
                self.data = []
                bar = go.Bar(
                    marker=dict(colorscale='Spectral')
                )
                bar.update(**self._bar_kwargs)
                self.add_trace(bar)
            bar = self.data[0]
            bar.x = sr.index
            bar.y = sr.values
            if bar.marker.colorscale is not None:
                bar.marker.color = sr.values


@pd.api.extensions.register_series_accessor("bar")
class Bar_Accessor():
    def __init__(self, obj):
        self._obj = obj

    def __call__(self, **kwargs):
        return Bar(sr=self._obj, **kwargs)

# ############# Scatter ############# #


class Scatter(UpdatableFigureWidget):
    """Draws a scatterplot for each column in a dataframe."""

    def __init__(self,
                 df=None,
                 scatter_kwargs={},
                 **layout_kwargs):
        self._scatter_kwargs = scatter_kwargs

        super().__init__()
        self.update_layout(showlegend=True)
        self.update_layout(**layout_kwargs)

        if df is not None:
            self.update_data(df)

    def update_data(self, df):
        # Checks and preprocessing
        check_type(df, pd.DataFrame)

        with self.batch_update():
            if len(self.data) != len(df.columns):
                self.data = []
                for i, column in enumerate(df.columns):
                    scatter = go.Scatter(name=column)
                    scatter.update(
                        **(self._scatter_kwargs[i] if isinstance(self._scatter_kwargs, list) else self._scatter_kwargs))
                    self.add_trace(scatter)
            for i, scatter in enumerate(self.data):
                scatter.x = df.index
                scatter.y = df.iloc[:, i].values


@pd.api.extensions.register_dataframe_accessor("scatter")
class Scatter_Accessor():
    def __init__(self, obj):
        self._obj = obj

    def __call__(self, **kwargs):
        return Scatter(df=self._obj, **kwargs)

# ############# Histogram ############# #


class Histogram(UpdatableFigureWidget):
    """Draws a histogram for each column in a dataframe."""

    def __init__(self,
                 df=None,
                 horizontal=False,
                 histogram_kwargs={},
                 **layout_kwargs):
        self._horizontal = horizontal
        self._histogram_kwargs = histogram_kwargs

        super().__init__()
        self.update_layout(showlegend=True, barmode='overlay')
        self.update_layout(**layout_kwargs)
        if df is not None:
            self.update_data(df)

    def update_data(self, df):
        # Checks and preprocessing
        check_type(df, pd.DataFrame)

        with self.batch_update():
            if len(self.data) != len(df.columns):
                self.data = []
                for column in df.columns:
                    histogram = go.Histogram(
                        opacity=0.75 if len(df.columns) > 1 else 1
                    )
                    histogram.update(
                        **(self._histogram_kwargs[i] if isinstance(self._histogram_kwargs, list) else self._histogram_kwargs))
                    self.add_trace(histogram)
            for i, hist in enumerate(self.data):
                hist.name = df.columns[i]
                if self._horizontal:
                    hist.x = None
                    hist.y = df.iloc[:, i].values
                else:
                    hist.x = df.iloc[:, i].values
                    hist.y = None


@pd.api.extensions.register_dataframe_accessor("histogram")
class Histogram_Accessor():
    def __init__(self, obj):
        self._obj = obj

    def __call__(self, **kwargs):
        return Histogram(df=self._obj, **kwargs)


# ############# Heatmap ############# #

class Heatmap(UpdatableFigureWidget):
    """Draw a heatmap of a dataframe."""

    def __init__(self,
                 df=None,
                 heatmap_kwargs={},
                 **layout_kwargs):
        self._heatmap_kwargs = heatmap_kwargs

        super().__init__()
        self.update_layout(**layout_kwargs)
        if df is not None:
            self.update_data(df)

    def update_data(self, df):
        # Checks and preprocessing
        check_type(df, pd.DataFrame)

        with self.batch_update():
            if len(self.data) != len(df.columns):
                self.data = []
                heatmap = go.Heatmap(
                    hoverongaps=False,
                    colorscale='Plasma'
                )
                heatmap.update(**self._heatmap_kwargs)
                self.add_trace(heatmap)
            heatmap = self.data[0]
            heatmap.x = df.columns
            heatmap.y = df.index
            heatmap.z = df.values


@pd.api.extensions.register_dataframe_accessor("heatmap")
class Heatmap_Accessor():
    def __init__(self, obj):
        self._obj = obj

    def __call__(self, **kwargs):
        return Heatmap(df=self._obj, **kwargs)
