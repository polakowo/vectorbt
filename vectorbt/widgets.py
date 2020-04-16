import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from vectorbt.utils import *
from vectorbt.accessors import *

__all__ = ['Gauge', 'Bar', 'Scatter', 'Histogram', 'Heatmap']

# You can change this from code using vbt.widgets.layout_defaults[key] = value
layout_defaults = Config(
    autosize=False,
    width=700,
    height=300,
    margin=dict(
        b=30,
        t=30
    ),
    hovermode='closest',
    colorway=[
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
)


class FigureWidget(go.FigureWidget):
    """Subclass of the graph_objects.FigureWidget class with default params."""

    def __init__(self):
        super().__init__()
        # Good default params
        self.update_layout(**layout_defaults)
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


def rgb_from_cmap(cmap_name, value, value_range):
    """Map value_range to colormap and get RGB of the value from that range."""
    if value_range[0] == value_range[1]:
        norm_value = 0.5
    else:
        norm_value = (value - value_range[0]) / (value_range[1] - value_range[0])
    cmap = plt.get_cmap(cmap_name)
    return "rgb(%d,%d,%d)" % tuple(np.round(np.asarray(cmap(norm_value))[:3] * 255))


class Gauge(UpdatableFigureWidget):
    """Accepts a single value and draws an indicator."""

    def __init__(self, value=None, label=None, value_range=None, cmap_name='Spectral', trace_kwargs={}, **layout_kwargs):
        self._value_range = value_range
        self._cmap_name = cmap_name

        super().__init__()
        self.update_layout(width=500, height=300)
        self.update_layout(**layout_kwargs)

        # Add traces
        indicator = go.Indicator(
            domain=dict(x=[0, 1], y=[0, 1]),
            mode="gauge+number+delta",
            title=dict(text=label)
        )
        indicator.update(**trace_kwargs)
        self.add_trace(indicator)

        if value is not None:
            self.update_data(value)

    def update_data(self, value):
        # NOTE: If called by Plotly event handler and in case of error, this won't visible in a notebook cell, but in logs!
        check_type(value, (int, float, complex))

        # Update value range
        if self._value_range is None:
            self._value_range = value, value
        else:
            self._value_range = min(self._value_range[0], value), max(self._value_range[1], value)

        # Update traces
        with self.batch_update():
            indicator = self.data[0]
            if self._value_range is not None:
                indicator.gauge.axis.range = self._value_range
                if self._cmap_name is not None:
                    indicator.gauge.bar.color = rgb_from_cmap(self._cmap_name, value, self._value_range)
            indicator.delta.reference = indicator.value
            indicator.value = value

# ############# Bar ############# #


class Bar(UpdatableFigureWidget):
    """Draw a barplot for each value in a series."""

    def __init__(self, x_labels, trace_names, data=None, trace_kwargs={}, **layout_kwargs):
        self._x_labels = x_labels
        self._trace_names = trace_names

        super().__init__()
        if len(trace_names) > 1:
            self.update_layout(showlegend=True)
        self.update_layout(**layout_kwargs)

        # Add traces
        for i, trace_name in enumerate(trace_names):
            bar = go.Bar(
                x=x_labels,
                name=trace_name
            )
            bar.update(**(trace_kwargs[i] if isinstance(trace_kwargs, (list, tuple)) else trace_kwargs))
            self.add_trace(bar)

        if data is not None:
            self.update_data(data)

    def update_data(self, data):
        if not is_array(data):
            data = np.asarray(data)
        data = to_2d(data)
        check_same_shape(data, self._x_labels, along_axis=(0, 0))
        check_same_shape(data, self._trace_names, along_axis=(1, 0))

        # Update traces
        with self.batch_update():
            for i, bar in enumerate(self.data):
                bar.y = data[:, i]
                if bar.marker.colorscale is not None:
                    bar.marker.color = data[:, i]


@register_dataframe_accessor('Bar')
@register_series_accessor('Bar')
class Bar_Accessor():
    def __init__(self, obj):
        self._obj = obj._obj  # access pandas object

    def __call__(self, x_labels=None, trace_names=None, **kwargs):
        if x_labels is None:
            x_labels = self._obj.index
        if trace_names is None:
            trace_names = to_2d(self._obj).columns
        return Bar(x_labels, trace_names, data=self._obj.values, **kwargs)

# ############# Scatter ############# #


class Scatter(UpdatableFigureWidget):
    """Draws a scatterplot for each column in a dataframe."""

    def __init__(self, x_labels, trace_names, data=None, trace_kwargs={}, **layout_kwargs):
        self._x_labels = x_labels
        self._trace_names = trace_names

        super().__init__()
        if len(trace_names) > 1:
            self.update_layout(showlegend=True)
        self.update_layout(**layout_kwargs)

        # Add traces
        for i, trace_name in enumerate(trace_names):
            scatter = go.Scatter(
                x=x_labels,
                name=trace_name
            )
            scatter.update(**(trace_kwargs[i] if isinstance(trace_kwargs, (list, tuple)) else trace_kwargs))
            self.add_trace(scatter)

        if data is not None:
            self.update_data(data)

    def update_data(self, data):
        if not is_array(data):
            data = np.asarray(data)
        data = to_2d(data)
        check_same_shape(data, self._x_labels, along_axis=(0, 0))
        check_same_shape(data, self._trace_names, along_axis=(1, 0))

        # Update traces
        with self.batch_update():
            for i, scatter in enumerate(self.data):
                scatter.y = data[:, i]


@register_dataframe_accessor('Scatter')
@register_series_accessor('Scatter')
class Scatter_Accessor():
    def __init__(self, obj):
        self._obj = obj._obj  # access pandas object

    def __call__(self, x_labels=None, trace_names=None, **kwargs):
        if x_labels is None:
            x_labels = self._obj.index
        if trace_names is None:
            trace_names = to_2d(self._obj).columns
        return Scatter(x_labels, trace_names, data=self._obj.values, **kwargs)

# ############# Histogram ############# #


class Histogram(UpdatableFigureWidget):
    """Draws a histogram for each column in a dataframe."""

    def __init__(self, trace_names, data=None, horizontal=False, trace_kwargs={}, **layout_kwargs):
        self._trace_names = trace_names
        self._horizontal = horizontal

        super().__init__()
        if len(trace_names) > 1:
            self.update_layout(showlegend=True)
        self.update_layout(barmode='overlay')
        self.update_layout(**layout_kwargs)

        # Add traces
        for i, trace_name in enumerate(trace_names):
            histogram = go.Histogram(
                name=trace_name,
                opacity=0.75 if len(trace_names) > 1 else 1
            )
            histogram.update(**(trace_kwargs[i] if isinstance(trace_kwargs, (list, tuple)) else trace_kwargs))
            self.add_trace(histogram)

        if data is not None:
            self.update_data(data)

    def update_data(self, data):
        if not is_array(data):
            data = np.asarray(data)
        data = to_2d(data)
        check_same_shape(data, self._trace_names, along_axis=(1, 0))

        # Update traces
        with self.batch_update():
            for i, histogram in enumerate(self.data):
                if self._horizontal:
                    histogram.x = None
                    histogram.y = data[:, i]
                else:
                    histogram.x = data[:, i]
                    histogram.y = None


@register_dataframe_accessor('Histogram')
@register_series_accessor('Histogram')
class Histogram_Accessor():
    def __init__(self, obj):
        self._obj = obj._obj  # access pandas object

    def __call__(self, trace_names=None, **kwargs):
        if trace_names is None:
            trace_names = to_2d(self._obj).columns
        return Histogram(trace_names, data=self._obj.values, **kwargs)


# ############# Heatmap ############# #

class Heatmap(UpdatableFigureWidget):
    """Draw a heatmap of a dataframe."""

    def __init__(self, x_labels, y_labels, data=None, horizontal=False, trace_kwargs={}, **layout_kwargs):
        self._x_labels = x_labels
        self._y_labels = y_labels
        self._horizontal = horizontal

        super().__init__()
        self.update_layout(**layout_kwargs)

        # Add traces
        heatmap = go.Heatmap(
            hoverongaps=False,
            colorscale='Plasma'
        )
        if self._horizontal:
            heatmap.y = x_labels
            heatmap.x = y_labels
        else:
            heatmap.x = x_labels
            heatmap.y = y_labels
        heatmap.update(**trace_kwargs)
        self.add_trace(heatmap)

        if data is not None:
            self.update_data(data)

    def update_data(self, data):
        if not is_array(data):
            data = np.asarray(data)
        data = to_2d(data)
        check_same_shape(data, self._x_labels, along_axis=(1, 0))
        check_same_shape(data, self._y_labels, along_axis=(0, 0))

        # Update traces
        with self.batch_update():
            heatmap = self.data[0]
            if self._horizontal:
                heatmap.z = data.transpose()
            else:
                heatmap.z = data


@register_dataframe_accessor('Heatmap')
@register_series_accessor('Heatmap')
class Heatmap_Accessor():
    def __init__(self, obj):
        self._obj = obj._obj  # access pandas object

    def __call__(self, x_labels=None, y_labels=None, **kwargs):
        if x_labels is None:
            x_labels = to_2d(self._obj).columns
        if y_labels is None:
            y_labels = self._obj.index
        return Heatmap(x_labels, y_labels, data=self._obj.values, **kwargs)
