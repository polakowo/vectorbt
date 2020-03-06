import numpy as np
import plotly.graph_objects as go
from vectorbt.decorators import *
import matplotlib.pyplot as plt

__all__ = ['Gauge', 'Bar', 'Scatter', 'Histogram', 'Heatmap']


def rgb_from_cmap(cmap_name, value, vrange):
    """Map vrange to colormap and get RGB of the value from that range."""
    if vrange[0] == vrange[1]:
        norm_value = 0.5
    else:
        norm_value = (value - vrange[0]) / (vrange[1] - vrange[0])
    cmap = plt.get_cmap(cmap_name)
    return "rgb(%d,%d,%d)" % tuple(np.round(np.asarray(cmap(norm_value))[:3] * 255))


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


class Gauge(UpdatableFigureWidget):
    """Accepts a single value and draws an indicator."""

    def __init__(self,
                 data=None,
                 label=None,
                 cmap_name='Spectral',
                 indicator_kwargs={},
                 **layout_kwargs):

        super().__init__()
        self.update_layout(width=500, height=300)
        self.update_layout(**layout_kwargs)
        self._vrange = None
        self._cmap_name = cmap_name
        indicator = go.Indicator(
            domain=dict(x=[0, 1], y=[0, 1]),
            mode="gauge+number+delta",
            title=dict(text=label)
        )
        indicator.update(**indicator_kwargs)
        self.add_trace(indicator)
        if data is not None:
            self.update_data(data)

    @has_type('data', (int, float, complex))
    # NOTE: If called by Plotly event handler and in case of error, this won't visible in a notebook cell, but in logs!
    def update_data(self, data):
        if self._vrange is None:
            self._vrange = data, data
        else:
            self._vrange = min(self._vrange[0], data), max(self._vrange[1], data)
        with self.batch_update():
            indicator = self.data[0]
            if self._vrange is not None:
                indicator.gauge.axis.range = self._vrange
                if self._cmap_name is not None:
                    indicator.gauge.bar.color = rgb_from_cmap(self._cmap_name, data, self._vrange)
            indicator.delta.reference = indicator.value
            indicator.value = data


class Bar(UpdatableFigureWidget):
    """Accepts a list of values and draws a bar for each."""

    def __init__(self,
                 data=None,
                 x_labels=None,
                 colorscale='Spectral',
                 bar_kwargs={},
                 **layout_kwargs):

        super().__init__()
        self.update_layout(**layout_kwargs)
        self._x_labels = x_labels
        bar = go.Bar(
            x=x_labels,
            marker=dict(colorscale=colorscale)
        )
        bar.update(**bar_kwargs)
        self.add_trace(bar)
        if data is not None:
            self.update_data(data)

    @to_1d('data')
    @have_same_shape('data', 'self._x_labels')
    def update_data(self, data):
        data = np.asarray(data)

        with self.batch_update():
            bar = self.data[0]
            bar.y = data
            if bar.marker.colorscale is not None:
                bar.marker.color = data


class Scatter(UpdatableFigureWidget):
    """Accepts a matrix and draws a scatter for each array along 1st axis."""

    @required('data_labels')
    def __init__(self,
                 data=None,
                 data_labels=None,
                 x_labels=None,
                 scatter_kwargs={},
                 **layout_kwargs):

        super().__init__()
        self.update_layout(showlegend=True)
        self.update_layout(**layout_kwargs)
        self._data_labels = data_labels
        self._x_labels = x_labels
        for i, data_label in enumerate(data_labels):
            scatter = go.Scatter(
                x=x_labels,
                name=data_label
            )
            scatter.update(**(scatter_kwargs[i] if isinstance(scatter_kwargs, list) else scatter_kwargs))
            self.add_trace(scatter)
        if data is not None:
            self.update_data(data)

    @to_2d('data', expand_axis=0)
    @have_same_shape('data', 'self._data_labels', along_axis=(0, 0))
    @have_same_shape('data', 'self._x_labels', along_axis=(1, 0))
    def update_data(self, data):
        data = np.asarray(data)

        with self.batch_update():
            for i, scatter in enumerate(self.data):
                scatter.y = data[i]


class Histogram(UpdatableFigureWidget):
    """Accepts a matrix and draws a histogram for each array along 1st axis."""

    @required('data_labels')
    def __init__(self,
                 data=None,
                 data_labels=None,
                 horizontal=False,
                 histogram_kwargs={},
                 **layout_kwargs):

        super().__init__()
        self.update_layout(showlegend=True, barmode='overlay')
        self.update_layout(**layout_kwargs)
        self._data_labels = data_labels
        self._horizontal = horizontal
        for data_label in data_labels:
            histogram = go.Histogram(
                name=data_label,
                opacity=0.75 if len(data_labels) > 1 else 1
            )
            histogram.update(**(histogram_kwargs[i] if isinstance(histogram_kwargs, list) else histogram_kwargs))
            self.add_trace(histogram)
        if data is not None:
            self.update_data(data)

    @to_2d('data', expand_axis=0)
    @have_same_shape('data', 'self._data_labels', along_axis=0)
    def update_data(self, data):
        data = np.asarray(data)

        with self.batch_update():
            for i, hist in enumerate(self.data):
                if self._horizontal:
                    hist.y = data[i]
                else:
                    hist.x = data[i]


class Heatmap(UpdatableFigureWidget):
    """Accepts a matrix and draws a heatmap of it."""

    def __init__(self,
                 data=None,
                 x_labels=None,
                 y_labels=None,
                 colorscale='Plasma',
                 heatmap_kwargs={},
                 **layout_kwargs):

        super().__init__()
        self.update_layout(**layout_kwargs)
        self._x_labels = x_labels
        self._y_labels = y_labels
        heatmap = go.Heatmap(
            x=x_labels,
            y=y_labels,
            hoverongaps=False,
            colorscale=colorscale
        )
        heatmap.update(**heatmap_kwargs)
        self.add_trace(heatmap)
        if data is not None:
            self.update_data(data)

    @to_2d('data', expand_axis=0)
    @have_same_shape('data', 'self._x_labels', along_axis=(1, 0))
    @have_same_shape('data', 'self._y_labels', along_axis=(0, 0))
    def update_data(self, data):
        data = np.asarray(data)

        with self.batch_update():
            heatmap = self.data[0]
            heatmap.z = data
