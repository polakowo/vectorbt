"""Base plotting functions.

!!! warning
    In case of errors, it won't be visible in the notebook cell, but in the logs."""

import numpy as np
import plotly.graph_objects as go
import math

from vectorbt.utils import checks
from vectorbt.utils.widgets import CustomFigureWidget
from vectorbt.utils.array import renormalize
from vectorbt.base import reshape_fns
from collections.abc import Iterable


# ############# Indicator ############# #

def rgb_from_cmap(cmap_name, value, value_range):
    """Map `value_range` to colormap with name `cmap_name` and get RGB of the `value` from that range."""
    import matplotlib.pyplot as plt

    if value_range[0] == value_range[1]:
        norm_value = 0.5
    else:
        norm_value = (value - value_range[0]) / (value_range[1] - value_range[0])
    cmap = plt.get_cmap(cmap_name)
    return "rgb(%d,%d,%d)" % tuple(np.round(np.asarray(cmap(norm_value))[:3] * 255))


def create_indicator(value=None, label=None, value_range=None, cmap_name='Spectral', trace_kwargs=None,
                     return_trace_idx=False, row=None, col=None, fig=None, **layout_kwargs):
    """Create an indicator plot.

    Args:
        value (int or float): The value to be displayed.
        label (str): The label to be displayed.
        value_range (list or tuple of 2 values): The value range of the gauge.
        cmap_name (str): A matplotlib-compatible colormap name.

            See the [list of available colormaps](https://matplotlib.org/tutorials/colors/colormaps.html).
        trace_kwargs (dict): Keyword arguments passed to the `plotly.graph_objects.Indicator`.
        return_trace_idx (bool): Whether to return trace index for `update_indicator_data`.
        row (int): Row position.
        col (int): Column position.
        fig (plotly.graph_objects.Figure): Figure to add traces to.
        **layout_kwargs: Keyword arguments for layout.

    ## Example

    ```python-repl
    >>> import vectorbt as vbt

    >>> vbt.plotting.create_indicator(
    ...     value=2,
    ...     value_range=(1, 3),
    ...     label='My Indicator'
    ... )
    ```
    ![](/vectorbt/docs/img/create_indicator.png)
    """
    from vectorbt.settings import layout

    if trace_kwargs is None:
        trace_kwargs = {}
    if fig is None:
        fig = CustomFigureWidget()
        if 'width' in layout:
            # Calculate nice width and height
            fig.update_layout(
                width=layout['width'] * 0.7,
                height=layout['width'] * 0.5,
                margin=dict(t=80)
            )
    fig.update_layout(**layout_kwargs)
    indicator = go.Indicator(
        domain=dict(x=[0, 1], y=[0, 1]),
        mode="gauge+number+delta",
        title=dict(text=label)
    )
    indicator.update(**trace_kwargs)
    fig.add_trace(indicator, row=row, col=col)
    trace_idx = len(fig.data) - 1
    if value is not None:
        update_indicator_data(fig, value, value_range=value_range, cmap_name=cmap_name, trace_idx=trace_idx)
    if return_trace_idx:
        return fig, trace_idx
    return fig


def update_indicator_data(fig, value, value_range=None, cmap_name='Spectral', trace_idx=None):
    """Update the indicator data.

    For keyword arguments, see `create_indicator`.
    Optionally, specify the index of the trace `trace_idx` to update."""
    if trace_idx is None:
        if len(fig.data) > 1:
            raise ValueError("Figure contains more traces than data. Must pass trace_idx.")
        trace_idx = 0

    # Update value range
    if value_range is None:
        value_range = value, value
    else:
        value_range = min(value_range[0], value), max(value_range[1], value)

    # Update traces
    with fig.batch_update():
        indicator = fig.data[trace_idx]
        if indicator.type != 'indicator':
            raise ValueError(f'Trace at index {trace_idx} is not a indicator')
        if value_range is not None:
            indicator.gauge.axis.range = value_range
            if cmap_name is not None:
                indicator.gauge.bar.color = rgb_from_cmap(cmap_name, value, value_range)
        indicator.delta.reference = indicator.value
        indicator.value = value
    return value_range


# ############# Bar ############# #


def create_bar(data=None, trace_names=None, x_labels=None, trace_kwargs=None,
               return_trace_idxs=False, row=None, col=None, fig=None, **layout_kwargs):
    """Create a bar plot.

    Args:
        data (array_like): Data in any format that can be converted to NumPy.

            Must be of shape (`x_labels`, `trace_names`).
        trace_names (str or list of str): Trace names, corresponding to columns in pandas.
        x_labels (array_like): X-axis labels, corresponding to index in pandas.
        trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Bar`.
        return_trace_idxs (bool): Whether to return trace indices for `update_bar_data`.
        row (int): Row position.
        col (int): Column position.
        fig (plotly.graph_objects.Figure): Figure to add traces to.
        **layout_kwargs: Keyword arguments for layout.

    ## Example

    ```python-repl
    >>> import vectorbt as vbt

    >>> vbt.plotting.create_bar(
    ...     data=[[1, 2], [3, 4]],
    ...     trace_names=['a', 'b'],
    ...     x_labels=['x', 'y']
    ... )
    ```
    ![](/vectorbt/docs/img/create_bar.png)
    """
    if trace_kwargs is None:
        trace_kwargs = {}
    if data is None:
        if trace_names is None:
            raise ValueError("At least trace_names must be passed")
    if trace_names is None:
        data = reshape_fns.to_2d(np.array(data))
        trace_names = [None] * data.shape[1]
    if isinstance(trace_names, str):
        trace_names = [trace_names]
    if fig is None:
        fig = CustomFigureWidget()
    fig.update_layout(**layout_kwargs)
    for i, trace_name in enumerate(trace_names):
        if trace_name is not None:
            trace_name = str(trace_name)
        bar = go.Bar(
            x=x_labels,
            name=trace_name,
            showlegend=trace_name is not None
        )
        bar.update(**(trace_kwargs[i] if isinstance(trace_kwargs, (list, tuple)) else trace_kwargs))
        fig.add_trace(bar, row=row, col=col)

    trace_idxs = list(range(len(fig.data) - len(trace_names), len(fig.data)))
    if data is not None:
        update_bar_data(fig, data, trace_idx=trace_idxs)
    if return_trace_idxs:
        return fig, trace_idxs
    return fig


def update_bar_data(fig, data, trace_idx=None):
    """Update the bar data.

    For keyword arguments, see `create_bar`.
    Optionally, specify one or multiple trace indices `trace_idx` to update.

    ## Example

    ```python-repl
    >>> vbt.plotting.update_bar_data(fig, [[2, 1], [4, 3]])
    ```
    ![](/vectorbt/docs/img/update_bar_data.png)
    """
    data = reshape_fns.to_2d(np.array(data))
    if trace_idx is None:
        if data.shape[1] < len(fig.data):
            raise ValueError("Figure contains more traces than data. Must pass trace_idx.")
        trace_idx = range(len(fig.data))
    if not isinstance(trace_idx, Iterable):
        trace_idx = [trace_idx]
    if data.shape[1] > len(trace_idx):
        raise ValueError("Data contains more traces than trace_idx")

    with fig.batch_update():
        for i, _trace_idx in enumerate(trace_idx):
            bar = fig.data[_trace_idx]
            if bar.type != 'bar':
                raise ValueError(f'Trace at index {_trace_idx} is not a bar')
            bar.y = data[:, i]
            if bar.marker.colorscale is not None:
                bar.marker.color = data[:, i]


# ############# Scatter ############# #


def create_scatter(data=None, trace_names=None, x_labels=None, trace_kwargs=None,
                   return_trace_idxs=False, row=None, col=None, fig=None, **layout_kwargs):
    """Create a scatter plot.

    Args:
        data (array_like): Data in any format that can be converted to NumPy.

            Must be of shape (`x_labels`, `trace_names`).
        trace_names (str or list of str): Trace names, corresponding to columns in pandas.
        x_labels (array_like): X-axis labels, corresponding to index in pandas.
        trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter`.
        return_trace_idxs (bool): Whether to return trace indices for `update_scatter_data`.
        row (int): Row position.
        col (int): Column position.
        fig (plotly.graph_objects.Figure): Figure to add traces to.
        **layout_kwargs: Keyword arguments for layout.

    ## Example

    ```python-repl
    >>> import vectorbt as vbt

    >>> vbt.plotting.create_scatter(
    ...     data=[[1, 2], [3, 4]],
    ...     trace_names=['a', 'b'],
    ...     x_labels=['x', 'y']
    ... )
    ```
    ![](/vectorbt/docs/img/create_scatter.png)
    """
    if trace_kwargs is None:
        trace_kwargs = {}
    if data is None:
        if trace_names is None:
            raise ValueError("At least trace_names must be passed")
    if trace_names is None:
        data = reshape_fns.to_2d(data)
        trace_names = [None] * data.shape[1]
    if isinstance(trace_names, str):
        trace_names = [trace_names]
    if fig is None:
        fig = CustomFigureWidget()
    fig.update_layout(**layout_kwargs)
    for i, trace_name in enumerate(trace_names):
        if trace_name is not None:
            trace_name = str(trace_name)
        scatter = go.Scatter(
            x=x_labels,
            name=trace_name,
            showlegend=trace_name is not None
        )
        scatter.update(**(trace_kwargs[i] if isinstance(trace_kwargs, (list, tuple)) else trace_kwargs))
        fig.add_trace(scatter, row=row, col=col)

    trace_idxs = list(range(len(fig.data) - len(trace_names), len(fig.data)))
    if data is not None:
        update_scatter_data(fig, data, trace_idx=trace_idxs)
    if return_trace_idxs:
        return fig, trace_idxs
    return fig


def update_scatter_data(fig, data, trace_idx=None):
    """Update the scatter data.

    For keyword arguments, see `create_scatter`.
    Optionally, specify one or multiple trace indices `trace_idx` to update."""
    data = reshape_fns.to_2d(np.array(data))
    if trace_idx is None:
        if data.shape[1] < len(fig.data):
            raise ValueError("Figure contains more traces than data. Must pass trace_idx.")
        trace_idx = range(len(fig.data))
    if not isinstance(trace_idx, Iterable):
        trace_idx = [trace_idx]
    if data.shape[1] > len(trace_idx):
        raise ValueError("Data contains more traces than trace_idx")

    with fig.batch_update():
        for i, _trace_idx in enumerate(trace_idx):
            scatter = fig.data[_trace_idx]
            if scatter.type != 'scatter':
                raise ValueError(f'Trace at index {_trace_idx} is not a scatter')
            scatter.y = data[:, i]


# ############# Histogram ############# #


def create_hist(data=None, trace_names=None, horizontal=False, remove_nan=True, trace_kwargs=None,
                return_trace_idxs=False, row=None, col=None, fig=None, **layout_kwargs):
    """Create a histogram plot.

    Args:
        data (array_like): Data in any format that can be converted to NumPy.

            Must be of shape (any, `trace_names`).
        trace_names (str or list of str): Trace names, corresponding to columns in pandas.
        horizontal (bool): Plot horizontally.
        remove_nan (bool): Whether to remove NaN values.
        trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Histogram`.
        return_trace_idxs (bool): Whether to return trace indices for `update_hist_data`.
        row (int): Row position.
        col (int): Column position.
        fig (plotly.graph_objects.Figure): Figure to add traces to.
        **layout_kwargs: Keyword arguments for layout.

    ## Example

    ```python-repl
    >>> import vectorbt as vbt

    >>> vbt.plotting.create_hist(
    ...     data=[[1, 2], [3, 4], [2, 1]],
    ...     trace_names=['a', 'b']
    ... )
    ```
    ![](/vectorbt/docs/img/create_hist.png)
    """
    if trace_kwargs is None:
        trace_kwargs = {}
    if data is None:
        if trace_names is None:
            raise ValueError("At least trace_names must be passed")
    if trace_names is None:
        data = reshape_fns.to_2d(data)
        trace_names = [None] * data.shape[1]
    if isinstance(trace_names, str):
        trace_names = [trace_names]
    if fig is None:
        fig = CustomFigureWidget()
        fig.update_layout(barmode='overlay')
    fig.update_layout(**layout_kwargs)
    for i, trace_name in enumerate(trace_names):
        if trace_name is not None:
            trace_name = str(trace_name)
        hist = go.Histogram(
            opacity=0.75 if len(trace_names) > 1 else 1,
            name=trace_name,
            showlegend=trace_name is not None
        )
        hist.update(**(trace_kwargs[i] if isinstance(trace_kwargs, (list, tuple)) else trace_kwargs))
        fig.add_trace(hist, row=row, col=col)

    trace_idxs = list(range(len(fig.data) - len(trace_names), len(fig.data)))
    if data is not None:
        update_hist_data(fig, data, horizontal=horizontal, trace_idx=trace_idxs, remove_nan=remove_nan)
    if return_trace_idxs:
        return fig, trace_idxs
    return fig


def update_hist_data(fig, data, horizontal=False, trace_idx=None, remove_nan=True):
    """Update the histogram data.

    For keyword arguments, see `create_hist`.
    Optionally, specify one or multiple trace indices `trace_idx` to update."""
    data = reshape_fns.to_2d(np.array(data))
    if trace_idx is None:
        if data.shape[1] < len(fig.data):
            raise ValueError("Figure contains more traces than data. Must pass trace_idx.")
        trace_idx = range(len(fig.data))
    if not isinstance(trace_idx, Iterable):
        trace_idx = [trace_idx]
    if data.shape[1] > len(trace_idx):
        raise ValueError("Data contains more traces than trace_idx")

    with fig.batch_update():
        for i, _trace_idx in enumerate(trace_idx):
            hist = fig.data[_trace_idx]
            if hist.type != 'histogram':
                raise ValueError(f'Trace at index {_trace_idx} is not a histogram')
            d = data[:, i]
            if remove_nan:
                d = d[~np.isnan(d)]
            if horizontal:
                hist.x = None
                hist.y = d
            else:
                hist.x = d
                hist.y = None


# ############# Box ############# #

def create_box(data=None, trace_names=None, horizontal=False, remove_nan=True, trace_kwargs=None,
               return_trace_idxs=False, row=None, col=None, fig=None, **layout_kwargs):
    """Create a box plot.

    Args:
        data (array_like): Data in any format that can be converted to NumPy.

            Must be of shape (any, `trace_names`).
        trace_names (str or list of str): Trace names, corresponding to columns in pandas.
        horizontal (bool): Plot horizontally.
        remove_nan (bool): Whether to remove NaN values.
        trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Box`.
        return_trace_idxs (bool): Whether to return trace indices for `update_box_data`.
        row (int): Row position.
        col (int): Column position.
        fig (plotly.graph_objects.Figure): Figure to add traces to.
        **layout_kwargs: Keyword arguments for layout.

    ## Example

    ```python-repl
    >>> import vectorbt as vbt

    >>> vbt.plotting.create_box(
    ...     data=[[1, 2], [3, 4], [2, 1]],
    ...     trace_names=['a', 'b']
    ... )
    ```
    ![](/vectorbt/docs/img/create_box.png)
    """
    if trace_kwargs is None:
        trace_kwargs = {}
    if data is None:
        if trace_names is None:
            raise ValueError("At least trace_names must be passed")
    if trace_names is None:
        data = reshape_fns.to_2d(data)
        trace_names = [None] * data.shape[1]
    if isinstance(trace_names, str):
        trace_names = [trace_names]
    if fig is None:
        fig = CustomFigureWidget()
        fig.update_layout(barmode='overlay')
    fig.update_layout(**layout_kwargs)
    for i, trace_name in enumerate(trace_names):
        if trace_name is not None:
            trace_name = str(trace_name)
        box = go.Box(
            name=trace_name,
            showlegend=trace_name is not None
        )
        box.update(**(trace_kwargs[i] if isinstance(trace_kwargs, (list, tuple)) else trace_kwargs))
        fig.add_trace(box, row=row, col=col)

    trace_idxs = list(range(len(fig.data) - len(trace_names), len(fig.data)))
    if data is not None:
        update_box_data(fig, data, horizontal=horizontal, trace_idx=trace_idxs, remove_nan=remove_nan)
    if return_trace_idxs:
        return fig, trace_idxs
    return fig


def update_box_data(fig, data, horizontal=False, trace_idx=None, remove_nan=True):
    """Update the box data.

    For keyword arguments, see `create_box`.
    Optionally, specify one or multiple trace indices `trace_idx` to update."""
    data = reshape_fns.to_2d(np.array(data))
    if trace_idx is None:
        if data.shape[1] < len(fig.data):
            raise ValueError("Figure contains more traces than data. Must pass trace_idx.")
        trace_idx = range(len(fig.data))
    if not isinstance(trace_idx, Iterable):
        trace_idx = [trace_idx]
    if data.shape[1] > len(trace_idx):
        raise ValueError("Data contains more traces than trace_idx")

    with fig.batch_update():
        for i, _trace_idx in enumerate(trace_idx):
            box = fig.data[_trace_idx]
            if box.type != 'box':
                raise ValueError(f'Trace at index {_trace_idx} is not a box')
            d = data[:, i]
            if remove_nan:
                d = d[~np.isnan(d)]
            if horizontal:
                box.x = d
                box.y = None
            else:
                box.x = None
                box.y = d


# ############# Heatmap ############# #

def create_heatmap(data=None, x_labels=None, y_labels=None, horizontal=False, trace_kwargs=None,
                   return_trace_idx=False, row=None, col=None, fig=None, **layout_kwargs):
    """Create a heatmap plot.

    Args:
        data (array_like): Data in any format that can be converted to NumPy.

            Must be of shape (`y_labels`, `x_labels`).
        x_labels (array_like): X-axis labels, corresponding to columns in pandas.
        y_labels (array_like): Y-axis labels, corresponding to index in pandas.
        horizontal (bool): Plot horizontally.
        trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Heatmap`.
        return_trace_idx (bool): Whether to return trace index for `update_heatmap_data`.
        row (int): Row position.
        col (int): Column position.
        fig (plotly.graph_objects.Figure): Figure to add traces to.
        **layout_kwargs: Keyword arguments for layout.

    ## Example

    ```python-repl
    >>> import vectorbt as vbt

    >>> vbt.plotting.create_heatmap(
    ...     data=[[1, 2], [3, 4]],
    ...     x_labels=['a', 'b'],
    ...     y_labels=['x', 'y']
    ... )
    ```
    ![](/vectorbt/docs/img/create_heatmap.png)
    """
    from vectorbt.settings import layout

    if trace_kwargs is None:
        trace_kwargs = {}
    if data is None:
        if x_labels is None or y_labels is None:
            raise ValueError("At least x_labels and y_labels must be passed")
    else:
        data = reshape_fns.to_2d(np.array(data))
    if horizontal:
        y_labels, y_labels = y_labels, x_labels
        if data is not None:
            data = data.transpose()
        horizontal = False
    if fig is None:
        fig = CustomFigureWidget()
        if 'width' in layout:
            # Calculate nice width and height
            max_width = layout['width']
            if data is not None:
                x_len = data.shape[1]
                y_len = data.shape[0]
            else:
                x_len = len(x_labels)
                y_len = len(y_labels)
            width = math.ceil(renormalize(
                x_len / (x_len + y_len),
                [0, 1],
                [0.3 * max_width, max_width]
            ))
            width = min(width + 150, max_width)  # account for colorbar
            height = math.ceil(renormalize(
                y_len / (x_len + y_len),
                [0, 1],
                [0.3 * max_width, max_width]
            ))
            height = min(height, max_width * 0.7)  # limit height
            fig.update_layout(
                width=width,
                height=height
            )

    fig.update_layout(**layout_kwargs)
    heatmap = go.Heatmap(
        hoverongaps=False,
        colorscale='Plasma',
        x=x_labels,
        y=y_labels
    )
    heatmap.update(**trace_kwargs)
    fig.add_trace(heatmap, row=row, col=col)
    trace_idx = len(fig.data) - 1
    if data is not None:
        update_heatmap_data(fig, data, horizontal=horizontal, trace_idx=trace_idx)
    if return_trace_idx:
        return fig, trace_idx
    return fig


def update_heatmap_data(fig, data, horizontal=False, trace_idx=None):
    """Update the heatmap data.

    For keyword arguments, see `create_heatmap`.
    Optionally, specify the index of the trace `trace_idx` to update."""
    data = reshape_fns.to_2d(np.array(data))
    if trace_idx is None:
        if len(fig.data) > 1:
            raise ValueError("Figure contains more traces than data. Must pass trace_idx.")
        trace_idx = 0

    with fig.batch_update():
        heatmap = fig.data[trace_idx]
        if heatmap.type != 'heatmap':
            raise ValueError(f'Trace at index {trace_idx} is not a heatmap')
        if horizontal:
            heatmap.z = data.transpose()
        else:
            heatmap.z = data


# ############# Volume ############# #

def create_volume(data=None, x_labels=None, y_labels=None, z_labels=False, trace_kwargs=None,
                  return_trace_idx=False, row=None, col=None, scene='scene', fig=None, **layout_kwargs):
    """Create a volume plot.

    Args:
        data (array_like): Data in any format that can be converted to NumPy.

            Must be a 3-dim array.
        x_labels (array_like): X-axis labels.
        y_labels (array_like): Y-axis labels.
        z_labels (array_like): Z-axis labels.
        trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Volume`.
        return_trace_idx (bool): Whether to return trace index for `update_volume_data`.
        row (int): Row position.
        col (int): Column position.
        scene (str): Reference to the 3D scene.
        fig (plotly.graph_objects.Figure): Figure to add traces to.
        **layout_kwargs: Keyword arguments for layout.

    !!! note
        Figure widgets have currently problems displaying NaNs.
        Use `.show()` method for rendering.

    ## Example

    ```python-repl
    >>> import vectorbt as vbt
    >>> import numpy as np

    >>> vbt.plotting.create_volume(
    ...     data=np.random.randint(1, 10, size=(3, 3, 3)),
    ...     x_labels=['a', 'b', 'c'],
    ...     y_labels=['d', 'e', 'f'],
    ...     z_labels=['g', 'h', 'i']
    ... )
    ```

    ![](/vectorbt/docs/img/create_volume.png)
    """
    from vectorbt.settings import layout

    if trace_kwargs is None:
        trace_kwargs = {}
    if data is None:
        raise ValueError("Data must be passed")
    data = np.asarray(data)
    checks.assert_ndim(data, 3)
    if x_labels is None:
        x_labels = np.arange(data.shape[0])
    if y_labels is None:
        y_labels = np.arange(data.shape[1])
    if z_labels is None:
        z_labels = np.arange(data.shape[2])
    x_labels = np.asarray(x_labels)
    y_labels = np.asarray(y_labels)
    z_labels = np.asarray(z_labels)

    if fig is None:
        fig = CustomFigureWidget()
        if 'width' in layout:
            # Calculate nice width and height
            fig.update_layout(
                width=layout['width'],
                height=0.7 * layout['width']
            )

    # Non-numeric data types are not supported by go.Volume, so use ticktext
    # Note: Currently plotly displays the entire tick array, in future versions it will be more sensible
    more_layout = dict()
    if not np.issubdtype(x_labels.dtype, np.number):
        x_ticktext = x_labels
        x_labels = np.arange(data.shape[0])
        more_layout[scene] = dict(xaxis=dict(ticktext=x_ticktext, tickvals=x_labels, tickmode='array'))
    if not np.issubdtype(y_labels.dtype, np.number):
        y_ticktext = y_labels
        y_labels = np.arange(data.shape[1])
        more_layout[scene] = dict(yaxis=dict(ticktext=y_ticktext, tickvals=y_labels, tickmode='array'))
    if not np.issubdtype(z_labels.dtype, np.number):
        z_ticktext = z_labels
        z_labels = np.arange(data.shape[2])
        more_layout[scene] = dict(zaxis=dict(ticktext=z_ticktext, tickvals=z_labels, tickmode='array'))
    fig.update_layout(**more_layout)
    fig.update_layout(**layout_kwargs)

    # Arrays must have the same length as the flattened data array
    x = np.repeat(x_labels, len(y_labels) * len(z_labels))
    y = np.tile(np.repeat(y_labels, len(z_labels)), len(x_labels))
    z = np.tile(z_labels, len(x_labels) * len(y_labels))

    volume = go.Volume(
        x=x,
        y=y,
        z=z,
        value=data.flatten(),
        opacity=0.2,
        surface_count=15,  # keep low for big data
        colorscale='Plasma'
    )
    volume.update(**trace_kwargs)
    fig.add_trace(volume, row=row, col=col)
    trace_idx = len(fig.data) - 1
    if return_trace_idx:
        return fig, trace_idx
    return fig


def update_volume_data(fig, data, trace_idx=None):
    """Update the volume data.

    For keyword arguments, see `create_volume`.
    Optionally, specify the index of the trace `trace_idx` to update."""
    data = np.asarray(data).flatten()
    if trace_idx is None:
        if len(fig.data) > 1:
            raise ValueError("Figure contains more traces than data. Must pass trace_idx.")
        trace_idx = 0

    with fig.batch_update():
        volume = fig.data[trace_idx]
        if volume.type != 'volume':
            raise ValueError(f'Trace at index {trace_idx} is not a volume')
        volume.value = data

