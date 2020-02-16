import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import ipywidgets as widgets
from vectorbt.utils.decorators import *


class UpdatableFigureWidget(go.FigureWidget):
    """Extend the graph_objects.FigureWidget class with update method."""

    def show_png(self):
        width = self.layout.width
        height = self.layout.height
        self.show(renderer="png", width=width, height=height)

    def update_data(self, *args, **kwargs):
        raise NotImplementedError


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
                 label,
                 data=None,
                 cmap_name='Spectral',
                 figsize=(500, 300),
                 layout_kwargs={},
                 indicator_kwargs={}):

        super().__init__()
        self._vrange = None
        self._cmap_name = cmap_name
        indicator = go.Indicator(
            domain=dict(x=[0, 1], y=[0, 1]),
            mode="gauge+number+delta",
            title=dict(text=label)
        )
        indicator.update(**indicator_kwargs)
        self.add_trace(indicator)
        self.update_layout(
            autosize=False,
            width=figsize[0],
            height=figsize[1],
            margin=go.layout.Margin(
                b=30,
                t=30
            )
        )
        self.update_layout(**layout_kwargs)
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
                 x_labels,
                 data=None,
                 colorscale='Spectral',
                 figsize=(800, 300),
                 layout_kwargs={},
                 bar_kwargs={}):

        super().__init__()
        self._x_labels = x_labels
        bar = go.Bar(
            x=x_labels,
            marker=dict(colorscale=colorscale)
        )
        bar.update(**bar_kwargs)
        self.add_trace(bar)
        self.update_layout(
            autosize=False,
            width=figsize[0],
            height=figsize[1],
            margin=go.layout.Margin(
                b=30,
                t=30
            ),
            xaxis_type='category',
            hovermode='closest'
        )
        self.update_layout(**layout_kwargs)
        if data is not None:
            self.update_data(data)

    @to_1d('data')
    @has_dtype('data', np.number)
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

    def __init__(self,
                 trace_names,
                 x_labels,
                 data=None,
                 figsize=(800, 300),
                 layout_kwargs={},
                 scatter_kwargs={}):

        super().__init__()
        self._trace_names = trace_names
        self._x_labels = x_labels
        for trace_name in trace_names:
            scatter = go.Scatter(
                x=x_labels,
                name=trace_name
            )
            scatter.update(**scatter_kwargs)
            self.add_trace(scatter)
        self.update_layout(
            autosize=False,
            width=figsize[0],
            height=figsize[1],
            margin=go.layout.Margin(
                b=30,
                t=30
            ),
            showlegend=True,
            hovermode='closest'
        )
        self.update_layout(**layout_kwargs)
        if data is not None:
            self.update_data(data)

    @to_2d('data', expand_axis=0)
    @has_dtype('data', np.number)
    @have_same_shape('data', 'self._trace_names', along_axis=(0, 0))
    @have_same_shape('data', 'self._x_labels', along_axis=(1, 0))
    def update_data(self, data):
        data = np.asarray(data)

        with self.batch_update():
            for i, scatter in enumerate(self.data):
                scatter.y = data[i]

            # Set y-axis range, mainly for fill='tozeroy'
            vmin, vmax = np.min(data), np.max(data)
            space = 0.05 * (vmax - vmin)
            self.update_yaxes(range=(vmin - space, vmax + space))


class Histogram(UpdatableFigureWidget):
    """Accepts a matrix and draws a histogram for each array along 1st axis."""

    def __init__(self,
                 trace_names,
                 data=None,
                 barmode=None,
                 figsize=(800, 300),
                 layout_kwargs={},
                 histogram_kwargs={}):

        super().__init__()
        self._trace_names = trace_names
        for trace_name in trace_names:
            histogram = go.Histogram(
                name=trace_name,
                opacity=0.75 if len(trace_names) > 1 else 1
            )
            histogram.update(**histogram_kwargs)
            self.add_trace(histogram)
        self.update_layout(
            autosize=False,
            width=figsize[0],
            height=figsize[1],
            margin=go.layout.Margin(
                b=30,
                t=30
            ),
            showlegend=True,
            hovermode='closest',
            barmode='overlay'
        )
        self.update_layout(**layout_kwargs)
        if data is not None:
            self.update_data(data)

    @to_2d('data', expand_axis=0)
    @has_dtype('data', np.number)
    @have_same_shape('data', 'self._trace_names', along_axis=0)
    def update_data(self, data):
        data = np.asarray(data)

        with self.batch_update():
            for i, hist in enumerate(self.data):
                hist.x = data[i]


class Heatmap(UpdatableFigureWidget):
    """Accepts a matrix and draws a heatmap of it."""

    def __init__(self,
                 x_labels,
                 y_labels,
                 data=None,
                 colorscale='Plasma',
                 figsize=(800, 300),
                 layout_kwargs={},
                 heatmap_kwargs={}):

        super().__init__()
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
        self.update_layout(
            autosize=False,
            width=figsize[0],
            height=figsize[1],
            margin=go.layout.Margin(
                b=30,
                t=30
            ),
            hovermode='closest'
        )
        self.update_layout(**layout_kwargs)
        if data is not None:
            self.update_data(data)

    @to_2d('data', expand_axis=0)
    @has_dtype('data', np.number)
    @have_same_shape('data', 'self._x_labels', along_axis=(1, 0))
    @have_same_shape('data', 'self._y_labels', along_axis=(0, 0))
    def update_data(self, data):
        data = np.asarray(data)

        with self.batch_update():
            heatmap = self.data[0]
            heatmap.z = data
