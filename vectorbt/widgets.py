"""This module contains a collection of Plotly widgets that can be easily created and updated.

Each widget is based upon [plotly.graph_objects.FigureWidget](https://plotly.com/python-api-reference/generated/plotly.graph_objects.html?highlight=figurewidget#plotly.graph_objects.FigureWidget),
which is then extended by `vectorbt.widgets.FigureWidget` and `vectorbt.widgets.UpdatableFigureWidget`.

## Default layout

Use `vectorbt.widgets.layout_defaults` dictionary to change the default layout.

For example, to change the default width:
```py
import vectorbt as vbt

vbt.widgets.layout_defaults['width'] = 800
```"""

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from vectorbt.utils import *
from vectorbt.accessors import *
from collections import namedtuple

__all__ = ['FigureWidget', 'UpdatableFigureWidget', 'Indicator', 'Bar', 'Scatter', 'Histogram', 'Heatmap']

# You can change this from code using vbt.widgets.layout_defaults[key] = value
layout_defaults = Config(
    frozen=False,
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
    def __init__(self):
        """Subclass of the [`plotly.graph_objects.FigureWidget`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.html?highlight=figurewidget#plotly.graph_objects.FigureWidget) class initialized 
        with default parameters from `vectorbt.widgets.layout_defaults`."""
        super().__init__()
        # Good default params
        self.update_layout(**layout_defaults)
        # You can then update them using update_layout

    def show_png(self):
        """Display the widget in PNG format."""
        width = self.layout.width
        height = self.layout.height
        self.show(renderer="png", width=width, height=height)


class UpdatableFigureWidget(FigureWidget):
    def __init__(self):
        """Subclass of the `vectorbt.widgets.FigureWidget` class with an abstract update method."""
        super().__init__()

    def update_data(self, *args, **kwargs):
        """Abstract method for updating the widget with new data."""
        raise NotImplementedError

# ############# Indicator ############# #


def rgb_from_cmap(cmap_name, value, value_range):
    """Map `value_range` to colormap and get RGB of the value from that range."""
    if value_range[0] == value_range[1]:
        norm_value = 0.5
    else:
        norm_value = (value - value_range[0]) / (value_range[1] - value_range[0])
    cmap = plt.get_cmap(cmap_name)
    return "rgb(%d,%d,%d)" % tuple(np.round(np.asarray(cmap(norm_value))[:3] * 255))


class Indicator(UpdatableFigureWidget):
    def __init__(self, value=None, label=None, value_range=None, cmap_name='Spectral', trace_kwargs={}, **layout_kwargs):
        """Create an updatable indicator plot.

        Args:
            value (int or float, optional): The value to be displayed.
            label (str, optional): The label to be displayed.
            value_range (list or tuple of 2 values, optional): The value range of the gauge.
            cmap_name (str, optional): A matplotlib-compatible colormap name, see the [list of available colormaps](https://matplotlib.org/tutorials/colors/colormaps.html).
            trace_kwargs (dict, optional): Keyword arguments passed to the [`plotly.graph_objects.Indicator`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Indicator.html).
            **layout_kwargs: Keyword arguments for layout.
        Examples:
            ```py
            vbt.Indicator(value=2, value_range=(1, 3), label='My Indicator')
            ```
            ![](img/Indicator.png)
            """

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
        """Update the data of the plot efficiently.

        Args:
            value (int or float): The value to be displayed.
        """
        # NOTE: If called by Plotly event handler and in case of error, this won't visible in a notebook cell, but in logs!
        check_type(value, (int, float))

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

    def __init__(self, x_labels, trace_names=None, data=None, trace_kwargs={}, **layout_kwargs):
        """Create an updatable bar plot.

        Args:
            x_labels (list of str): X-axis labels, corresponding to index in pandas.
            trace_names (str or list of str, optional): Trace names, corresponding to columns in pandas.
            data (array_like, optional): Data in any format that can be converted to NumPy.
            trace_kwargs (dict or list of dict, optional): Keyword arguments passed to each [`plotly.graph_objects.Bar`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Bar.html).
            **layout_kwargs: Keyword arguments for layout.
        Examples:
            One trace:
            ```py
            vbt.Bar(['x', 'y'], trace_names='a', data=[1, 2])

            # Or the same directly on pandas
            pd.Series([1, 2], index=['x', 'y'], name='a').vbt.Bar()
            ```
            ![](img/Bar.png)

            Multiple traces:
            ```py
            vbt.Bar(['x', 'y'], trace_names=['a', 'b'], data=[[1, 2], [3, 4]])

            # Or the same directly on pandas
            pd.DataFrame({'a': [1, 3], 'b': [2, 4]}, index=['x', 'y']).vbt.Bar()
            ```
            ![](img/Bar_mult.png)
            """
        if isinstance(trace_names, str) or trace_names is None:
            trace_names = [trace_names]
        self._x_labels = x_labels
        self._trace_names = trace_names

        super().__init__()
        if len(trace_names) > 1 or trace_names[0] is not None:
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
        """Update the data of the plot efficiently.

        Args:
            data (array_like): Data in any format that can be converted to NumPy.

                Must be of shape (`x_labels`, `trace_names`).
        Examples:
            ```py
            fig = pd.Series([1, 2], index=['x', 'y'], name='a').vbt.Bar()
            fig.update_data([2, 1])
            fig.show()
            ```
            ![](img/Bar_updated.png)
        """
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
            if is_frame(self._obj) or (is_series(self._obj) and self._obj.name is not None):
                trace_names = to_2d(self._obj).columns
        return Bar(x_labels, trace_names=trace_names, data=self._obj.values, **kwargs)

# ############# Scatter ############# #


class Scatter(UpdatableFigureWidget):
    def __init__(self, x_labels, trace_names=None, data=None, trace_kwargs={}, **layout_kwargs):
        """Create an updatable scatter plot.

        Args:
            x_labels (list of str): X-axis labels, corresponding to index in pandas.
            trace_names (str or list of str, optional): Trace names, corresponding to columns in pandas.
            data (array_like, optional): Data in any format that can be converted to NumPy.
            trace_kwargs (dict or list of dict, optional): Keyword arguments passed to each [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html).
            **layout_kwargs: Keyword arguments for layout.
        Examples:
            ```py
            vbt.Scatter(['x', 'y'], trace_names=['a', 'b'], data=[[1, 2], [3, 4]])

            # Or the same directly on pandas
            pd.DataFrame({'a': [1, 3], 'b': [2, 4]}, index=['x', 'y']).vbt.Scatter()
            ```
            ![](img/Scatter.png)
            """

        if isinstance(trace_names, str) or trace_names is None:
            trace_names = [trace_names]
        self._x_labels = x_labels
        self._trace_names = trace_names

        super().__init__()
        if len(trace_names) > 1 or trace_names[0] is not None:
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
        """Update the data of the plot efficiently.

        Args:
            data (array_like): Data in any format that can be converted to NumPy.

                Must be of shape (`x_labels`, `trace_names`).
        """
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
            if is_frame(self._obj) or (is_series(self._obj) and self._obj.name is not None):
                trace_names = to_2d(self._obj).columns
        return Scatter(x_labels, trace_names=trace_names, data=self._obj.values, **kwargs)

# ############# Histogram ############# #


class Histogram(UpdatableFigureWidget):
    def __init__(self, trace_names=None, data=None, horizontal=False, trace_kwargs={}, **layout_kwargs):
        """Create an updatable histogram plot.

        Args:
            trace_names (str or list of str, optional): Trace names, corresponding to columns in pandas.
            data (array_like, optional): Data in any format that can be converted to NumPy.
            horizontal (bool): Plot horizontally. Defaults to False.
            trace_kwargs (dict or list of dict, optional): Keyword arguments passed to each [`plotly.graph_objects.Histogram`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Histogram.html)
            **layout_kwargs: Keyword arguments for layout
        Examples:
            ```py
            vbt.Histogram(trace_names=['a', 'b'], data=[[1, 2], [3, 4], [2, 1]])

            # Or the same directly on pandas
            pd.DataFrame({'a': [1, 3, 2], 'b': [2, 4, 1]}).vbt.Histogram()
            ```
            ![](img/Histogram.png)
            """

        if isinstance(trace_names, str) or trace_names is None:
            trace_names = [trace_names]
        self._trace_names = trace_names
        self._horizontal = horizontal

        super().__init__()
        if len(trace_names) > 1 or trace_names[0] is not None:
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
        """Update the data of the plot efficiently.

        Args:
            data (array_like): Data in any format that can be converted to NumPy.

                Must be of shape (any, `trace_names`).
        """
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
            if is_frame(self._obj) or (is_series(self._obj) and self._obj.name is not None):
                trace_names = to_2d(self._obj).columns
        return Histogram(trace_names=trace_names, data=self._obj.values, **kwargs)


# ############# Heatmap ############# #

class Heatmap(UpdatableFigureWidget):
    def __init__(self, x_labels, y_labels, data=None, horizontal=False, trace_kwargs={}, **layout_kwargs):
        """Create an updatable heatmap plot.

        Args:
            x_labels (list of str): X-axis labels, corresponding to columns in pandas.
            y_labels (list of str): Y-axis labels, corresponding to index in pandas.
            data (array_like, optional): Data in any format that can be converted to NumPy.
            horizontal (bool): Plot horizontally. Defaults to False.
            trace_kwargs (dict or list of dict, optional): Keyword arguments passed to each [`plotly.graph_objects.Heatmap`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Heatmap.html).
            **layout_kwargs: Keyword arguments for layout.
        Examples:
            ```py
            vbt.Heatmap(['a', 'b'], ['x', 'y'], data=[[1, 2], [3, 4]])

            # Or the same directly on pandas
            pd.DataFrame({'a': [1, 3], 'b': [2, 4]}, index=['x', 'y']).vbt.Heatmap()
            ```
            ![](img/Heatmap.png)
            """

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
        """Update the data of the plot efficiently.

        Args:
            data (array_like): Data in any format that can be converted to NumPy.

                Must be of shape (`y_labels`, `x_labels`).
        """
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
