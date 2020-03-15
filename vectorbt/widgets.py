import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from vectorbt.utils import *
from vectorbt.accessors import *

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
                 data=None,
                 bar_kwargs={},
                 **layout_kwargs):
        self._bar_kwargs = bar_kwargs

        super().__init__()
        self.update_layout(**layout_kwargs)

        if data is not None:
            self.update_data(data)

    def update_data(self, data):
        # Checks and preprocessing
        check_type(data, (pd.Series, pd.DataFrame))
        df = to_2d(data)

        with self.batch_update():
            if len(self.data) != len(df.columns):
                # Create new traces
                self.data = []
                for i, column in enumerate(df.columns):
                    bar = go.Bar(
                        name=column
                    )
                    bar.update(**self._bar_kwargs)
                    self.add_trace(bar)

            for i, bar in enumerate(self.data):
                # Update traces
                bar.x = df.index
                bar.y = df.iloc[:, i].values
                bar.name = df.columns[i]
                if bar.marker.colorscale is not None:
                    bar.marker.color = df.iloc[:, i].values


@register_dataframe_accessor('bar')
@register_series_accessor('bar')
class Bar_Accessor():
    def __init__(self, obj):
        self._obj = obj._obj  # access pandas object

    def __call__(self, **kwargs):
        return Bar(data=self._obj, **kwargs)

# ############# Scatter ############# #


class Scatter(UpdatableFigureWidget):
    """Draws a scatterplot for each column in a dataframe."""

    def __init__(self,
                 data=None,
                 scatter_kwargs={},
                 **layout_kwargs):
        self._scatter_kwargs = scatter_kwargs

        super().__init__()
        self.update_layout(showlegend=True)
        self.update_layout(**layout_kwargs)

        if data is not None:
            self.update_data(data)

    def update_data(self, data):
        # Checks and preprocessing
        check_type(data, (pd.Series, pd.DataFrame))
        df = to_2d(data)

        with self.batch_update():
            if len(self.data) != len(df.columns):
                # Create new traces
                self.data = []
                for i, column in enumerate(df.columns):
                    scatter = go.Scatter(
                        name=column
                    )
                    scatter.update(self._scatter_kwargs)
                    self.add_trace(scatter)
            
            for i, scatter in enumerate(self.data):
                # Update traces
                scatter.x = df.index
                scatter.y = df.iloc[:, i].values
                scatter.name = df.columns[i]


@register_dataframe_accessor('scatter')
@register_series_accessor('scatter')
class Scatter_Accessor():
    def __init__(self, obj):
        self._obj = obj._obj  # access pandas object

    def __call__(self, **kwargs):
        return Scatter(data=self._obj, **kwargs)

# ############# Histogram ############# #


class Histogram(UpdatableFigureWidget):
    """Draws a histogram for each column in a dataframe."""

    def __init__(self,
                 data=None,
                 horizontal=False,
                 histogram_kwargs={},
                 **layout_kwargs):
        self._horizontal = horizontal
        self._histogram_kwargs = histogram_kwargs

        super().__init__()
        self.update_layout(showlegend=True, barmode='overlay')
        self.update_layout(**layout_kwargs)
        if data is not None:
            self.update_data(data)

    def update_data(self, data):
        # Checks and preprocessing
        check_type(data, (pd.Series, pd.DataFrame))
        df = to_2d(data)

        with self.batch_update():
            if len(self.data) != len(df.columns):
                # Create new traces
                self.data = []
                for i, column in enumerate(df.columns):
                    histogram = go.Histogram(
                        name=column,
                        opacity=0.75 if len(df.columns) > 1 else 1
                    )
                    histogram.update(self._histogram_kwargs)
                    self.add_trace(histogram)

            for i, histogram in enumerate(self.data):
                # Update traces
                histogram.name = df.columns[i]
                if self._horizontal:
                    histogram.x = None
                    histogram.y = df.iloc[:, i].values
                else:
                    histogram.x = df.iloc[:, i].values
                    histogram.y = None


@register_dataframe_accessor('histogram')
@register_series_accessor('histogram')
class Histogram_Accessor():
    def __init__(self, obj):
        self._obj = obj._obj  # access pandas object

    def __call__(self, **kwargs):
        return Histogram(data=self._obj, **kwargs)


# ############# Heatmap ############# #

class Heatmap(UpdatableFigureWidget):
    """Draw a heatmap of a dataframe."""

    def __init__(self,
                 data=None,
                 horizontal=False,
                 heatmap_kwargs={},
                 **layout_kwargs):
        self._horizontal = horizontal
        self._heatmap_kwargs = heatmap_kwargs

        super().__init__()
        self.update_layout(**layout_kwargs)
        if data is not None:
            self.update_data(data)

    def update_data(self, data):
        # Checks and preprocessing
        check_type(data, (pd.Series, pd.DataFrame))
        df = to_2d(data)

        with self.batch_update():
            if len(self.data) == 0:
                # Create new traces
                self.data = []
                heatmap = go.Heatmap(
                    hoverongaps=False,
                    colorscale='Plasma'
                )
                heatmap.update(**self._heatmap_kwargs)
                self.add_trace(heatmap)

            # Update traces
            heatmap = self.data[0]
            if self._horizontal:
                heatmap.y = df.columns
                heatmap.x = df.index
                heatmap.z = df.values.transpose()
            else:
                heatmap.x = df.columns
                heatmap.y = df.index
                heatmap.z = df.values
            


@register_dataframe_accessor('heatmap')
@register_series_accessor('heatmap')
class Heatmap_Accessor():
    def __init__(self, obj):
        self._obj = obj._obj  # access pandas object

    def __call__(self, **kwargs):
        return Heatmap(data=self._obj, **kwargs)
