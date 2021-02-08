"""Base plotting functions.

!!! warning
    In case of errors, it won't be visible in the notebook cell, but in the logs."""

import numpy as np
import plotly.graph_objects as go
import math

from vectorbt.utils import checks
from vectorbt.utils.widgets import FigureWidget
from vectorbt.utils.array import renormalize
from vectorbt.utils.colors import rgb_from_cmap
from vectorbt.utils.config import Configured
from vectorbt.base import reshape_fns


class TraceUpdater:
    def __init__(self, fig, traces):
        """Base trace updating class."""
        self._fig = fig
        self._traces = traces

    @property
    def fig(self):
        """Figure."""
        return self._fig

    @property
    def traces(self):
        """Traces to update."""
        return self._traces

    def update(self, *agrs, **kwargs):
        """Update the trace data."""
        raise NotImplementedError


class Indicator(Configured, TraceUpdater):
    def __init__(self, value=None, label=None, value_range=None, cmap_name='Spectral',
                 trace_kwargs=None, add_trace_kwargs=None, fig=None, **layout_kwargs):
        """Create an indicator plot.

        Args:
            value (int or float): The value to be displayed.
            label (str): The label to be displayed.
            value_range (list or tuple of 2 values): The value range of the gauge.
            cmap_name (str): A matplotlib-compatible colormap name.

                See the [list of available colormaps](https://matplotlib.org/tutorials/colors/colormaps.html).
            trace_kwargs (dict): Keyword arguments passed to the `plotly.graph_objects.Indicator`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> import vectorbt as vbt

        >>> indicator = vbt.plotting.Indicator(
        ...     value=2,
        ...     value_range=(1, 3),
        ...     label='My Indicator'
        ... )
        >>> indicator.fig
        ```
        ![](/vectorbt/docs/img/Indicator.png)
        """
        Configured.__init__(
            self,
            value=value,
            label=label,
            value_range=value_range,
            cmap_name=cmap_name,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs
        )

        from vectorbt.settings import layout

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        if fig is None:
            fig = FigureWidget()
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
        fig.add_trace(indicator, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, [fig.data[-1]])
        self.value_range = value_range
        self.cmap_name = cmap_name

        if value is not None:
            self.update(value)

    def update(self, value):
        """Update the trace data."""
        if self.value_range is None:
            self.value_range = value, value
        else:
            self.value_range = min(self.value_range[0], value), max(self.value_range[1], value)

        with self.fig.batch_update():
            if self.value_range is not None:
                self.traces[0].gauge.axis.range = self.value_range
                if self.cmap_name is not None:
                    self.traces[0].gauge.bar.color = rgb_from_cmap(self.cmap_name, value, self.value_range)
            self.traces[0].delta.reference = self.traces[0].value
            self.traces[0].value = value


class Bar(Configured, TraceUpdater):
    def __init__(self, data=None, trace_names=None, x_labels=None, trace_kwargs=None,
                 add_trace_kwargs=None, fig=None, **layout_kwargs):
        """Create a bar plot.

        Args:
            data (array_like): Data in any format that can be converted to NumPy.

                Must be of shape (`x_labels`, `trace_names`).
            trace_names (str or list of str): Trace names, corresponding to columns in pandas.
            x_labels (array_like): X-axis labels, corresponding to index in pandas.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Bar`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> import vectorbt as vbt

        >>> bar = vbt.plotting.Bar(
        ...     data=[[1, 2], [3, 4]],
        ...     trace_names=['a', 'b'],
        ...     x_labels=['x', 'y']
        ... )
        >>> bar.fig
        ```
        ![](/vectorbt/docs/img/Bar.png)
        """
        Configured.__init__(
            self,
            data=data,
            trace_names=trace_names,
            x_labels=x_labels,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs
        )

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if data is None:
            if trace_names is None:
                raise ValueError("At least trace_names must be passed")
        if trace_names is None:
            data = reshape_fns.to_2d(np.array(data))
            trace_names = [None] * data.shape[1]
        if isinstance(trace_names, str):
            trace_names = [trace_names]

        if fig is None:
            fig = FigureWidget()
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
            fig.add_trace(bar, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, fig.data[-len(trace_names):])

        if data is not None:
            self.update(data)

    def update(self, data):
        """Update the trace data.

        ## Example

        ```python-repl
        >>> bar.update([[2, 1], [4, 3]])
        ```
        ![](/vectorbt/docs/img/update_bar_data.png)
        """
        data = reshape_fns.to_2d(np.array(data))
        with self.fig.batch_update():
            for i, bar in enumerate(self.traces):
                bar.y = data[:, i]
                if bar.marker.colorscale is not None:
                    bar.marker.color = data[:, i]


class Scatter(Configured, TraceUpdater):
    def __init__(self, data=None, trace_names=None, x_labels=None, trace_kwargs=None,
                 add_trace_kwargs=None, fig=None, **layout_kwargs):
        """Create a scatter plot.

        Args:
            data (array_like): Data in any format that can be converted to NumPy.

                Must be of shape (`x_labels`, `trace_names`).
            trace_names (str or list of str): Trace names, corresponding to columns in pandas.
            x_labels (array_like): X-axis labels, corresponding to index in pandas.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> import vectorbt as vbt

        >>> scatter = vbt.plotting.Scatter(
        ...     data=[[1, 2], [3, 4]],
        ...     trace_names=['a', 'b'],
        ...     x_labels=['x', 'y']
        ... )
        >>> scatter.fig
        ```
        ![](/vectorbt/docs/img/Scatter.png)
        """
        Configured.__init__(
            self,
            data=data,
            trace_names=trace_names,
            x_labels=x_labels,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs
        )

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if data is None:
            if trace_names is None:
                raise ValueError("At least trace_names must be passed")
        if trace_names is None:
            data = reshape_fns.to_2d(data)
            trace_names = [None] * data.shape[1]
        if isinstance(trace_names, str):
            trace_names = [trace_names]

        if fig is None:
            fig = FigureWidget()
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
            fig.add_trace(scatter, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, fig.data[-len(trace_names):])

        if data is not None:
            self.update(data)

    def update(self, data):
        """Update the trace data."""
        data = reshape_fns.to_2d(np.array(data))

        with self.fig.batch_update():
            for i, trace in enumerate(self.traces):
                trace.y = data[:, i]


class Histogram(Configured, TraceUpdater):
    def __init__(self, data=None, trace_names=None, horizontal=False, remove_nan=True,
                 from_quantile=None, to_quantile=None, trace_kwargs=None, add_trace_kwargs=None,
                 fig=None, **layout_kwargs):
        """Create a histogram plot.

        Args:
            data (array_like): Data in any format that can be converted to NumPy.

                Must be of shape (any, `trace_names`).
            trace_names (str or list of str): Trace names, corresponding to columns in pandas.
            horizontal (bool): Plot horizontally.
            remove_nan (bool): Whether to remove NaN values.
            from_quantile (float): Filter out data points before this quantile.

                Should be in range `[0, 1]`.
            to_quantile (float): Filter out data points after this quantile.

                Should be in range `[0, 1]`.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Histogram`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> import vectorbt as vbt

        >>> hist = vbt.plotting.Histogram(
        ...     data=[[1, 2], [3, 4], [2, 1]],
        ...     trace_names=['a', 'b']
        ... )
        >>> hist.fig
        ```
        ![](/vectorbt/docs/img/Histogram.png)
        """
        Configured.__init__(
            self,
            data=data,
            trace_names=trace_names,
            horizontal=horizontal,
            remove_nan=remove_nan,
            from_quantile=from_quantile,
            to_quantile=to_quantile,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs
        )

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if data is None:
            if trace_names is None:
                raise ValueError("At least trace_names must be passed")
        if trace_names is None:
            data = reshape_fns.to_2d(data)
            trace_names = [None] * data.shape[1]
        if isinstance(trace_names, str):
            trace_names = [trace_names]

        if fig is None:
            fig = FigureWidget()
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
            fig.add_trace(hist, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, fig.data[-len(trace_names):])
        self.horizontal = horizontal
        self.remove_nan = remove_nan
        self.from_quantile = from_quantile
        self.to_quantile = to_quantile

        if data is not None:
            self.update(data)

    def update(self, data):
        """Update the trace data."""
        data = reshape_fns.to_2d(np.array(data))

        with self.fig.batch_update():
            for i, trace in enumerate(self.traces):
                d = data[:, i]
                if self.remove_nan:
                    d = d[~np.isnan(d)]
                mask = np.full(d.shape, True)
                if self.from_quantile is not None:
                    mask &= d >= np.quantile(d, self.from_quantile)
                if self.to_quantile is not None:
                    mask &= d <= np.quantile(d, self.to_quantile)
                d = d[mask]
                if self.horizontal:
                    trace.x = None
                    trace.y = d
                else:
                    trace.x = d
                    trace.y = None


class Box(Configured, TraceUpdater):
    def __init__(self, data=None, trace_names=None, horizontal=False, remove_nan=True,
                 from_quantile=None, to_quantile=None, trace_kwargs=None, add_trace_kwargs=None,
                 fig=None, **layout_kwargs):
        """Create a box plot.

        For keyword arguments, see `Histogram`.

        ## Example

        ```python-repl
        >>> import vectorbt as vbt

        >>> box = vbt.plotting.Box(
        ...     data=[[1, 2], [3, 4], [2, 1]],
        ...     trace_names=['a', 'b']
        ... )
        >>> box.fig
        ```
        ![](/vectorbt/docs/img/Box.png)
        """
        Configured.__init__(
            self,
            data=data,
            trace_names=trace_names,
            horizontal=horizontal,
            remove_nan=remove_nan,
            from_quantile=from_quantile,
            to_quantile=to_quantile,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs
        )

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if data is None:
            if trace_names is None:
                raise ValueError("At least trace_names must be passed")
        if trace_names is None:
            data = reshape_fns.to_2d(data)
            trace_names = [None] * data.shape[1]
        if isinstance(trace_names, str):
            trace_names = [trace_names]

        if fig is None:
            fig = FigureWidget()
        fig.update_layout(**layout_kwargs)

        for i, trace_name in enumerate(trace_names):
            if trace_name is not None:
                trace_name = str(trace_name)
            box = go.Box(
                name=trace_name,
                showlegend=trace_name is not None
            )
            box.update(**(trace_kwargs[i] if isinstance(trace_kwargs, (list, tuple)) else trace_kwargs))
            fig.add_trace(box, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, fig.data[-len(trace_names):])
        self.horizontal = horizontal
        self.remove_nan = remove_nan
        self.from_quantile = from_quantile
        self.to_quantile = to_quantile

        if data is not None:
            self.update(data)

    def update(self, data):
        """Update the trace data."""
        data = reshape_fns.to_2d(np.array(data))

        with self.fig.batch_update():
            for i, trace in enumerate(self.traces):
                d = data[:, i]
                if self.remove_nan:
                    d = d[~np.isnan(d)]
                mask = np.full(d.shape, True)
                if self.from_quantile is not None:
                    mask &= d >= np.quantile(d, self.from_quantile)
                if self.to_quantile is not None:
                    mask &= d <= np.quantile(d, self.to_quantile)
                d = d[mask]
                if self.horizontal:
                    trace.x = d
                    trace.y = None
                else:
                    trace.x = None
                    trace.y = d


class Heatmap(Configured, TraceUpdater):
    def __init__(self, data=None, x_labels=None, y_labels=None, horizontal=False,
                 trace_kwargs=None, add_trace_kwargs=None, fig=None, **layout_kwargs):
        """Create a heatmap plot.

        Args:
            data (array_like): Data in any format that can be converted to NumPy.

                Must be of shape (`y_labels`, `x_labels`).
            x_labels (array_like): X-axis labels, corresponding to columns in pandas.
            y_labels (array_like): Y-axis labels, corresponding to index in pandas.
            horizontal (bool): Plot horizontally.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Heatmap`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> import vectorbt as vbt

        >>> heatmap = vbt.plotting.Heatmap(
        ...     data=[[1, 2], [3, 4]],
        ...     x_labels=['a', 'b'],
        ...     y_labels=['x', 'y']
        ... )
        >>> heatmap.fig
        ```
        ![](/vectorbt/docs/img/Heatmap.png)
        """
        Configured.__init__(
            self,
            data=data,
            x_labels=x_labels,
            y_labels=y_labels,
            horizontal=horizontal,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs
        )

        from vectorbt.settings import layout

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if data is None:
            if x_labels is None or y_labels is None:
                raise ValueError("At least x_labels and y_labels must be passed")
        else:
            data = reshape_fns.to_2d(np.array(data))
        if horizontal:
            x_labels, y_labels = y_labels, x_labels
            if data is not None:
                data = data.transpose()
            horizontal = False

        if fig is None:
            fig = FigureWidget()
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
        fig.add_trace(heatmap, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, [fig.data[-1]])
        self.horizontal = horizontal

        if data is not None:
            self.update(data)

    def update(self, data):
        """Update the trace data."""
        data = reshape_fns.to_2d(np.array(data))

        with self.fig.batch_update():
            if self.horizontal:
                self.traces[0].z = data.transpose()
            else:
                self.traces[0].z = data


class Volume(Configured, TraceUpdater):
    def __init__(self, data=None, x_labels=None, y_labels=None, z_labels=False, trace_kwargs=None,
                 add_trace_kwargs=None, scene_name='scene', fig=None, **layout_kwargs):
        """Create a volume plot.

        Args:
            data (array_like): Data in any format that can be converted to NumPy.

                Must be a 3-dim array.
            x_labels (array_like): X-axis labels.
            y_labels (array_like): Y-axis labels.
            z_labels (array_like): Z-axis labels.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Volume`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            scene_name (str): Reference to the 3D scene.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        !!! note
            Figure widgets have currently problems displaying NaNs.
            Use `.show()` method for rendering.

        ## Example

        ```python-repl
        >>> import vectorbt as vbt
        >>> import numpy as np

        >>> volume = vbt.plotting.Volume(
        ...     data=np.random.randint(1, 10, size=(3, 3, 3)),
        ...     x_labels=['a', 'b', 'c'],
        ...     y_labels=['d', 'e', 'f'],
        ...     z_labels=['g', 'h', 'i']
        ... )
        >>> volume.fig
        ```

        ![](/vectorbt/docs/img/Volume.png)
        """
        Configured.__init__(
            self,
            data=data,
            x_labels=x_labels,
            y_labels=y_labels,
            z_labels=z_labels,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            scene_name=scene_name,
            fig=fig,
            **layout_kwargs
        )

        from vectorbt.settings import layout

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if data is None:
            if x_labels is None or y_labels is None or z_labels is None:
                raise ValueError("At least x_labels, y_labels and z_labels must be passed")
            x_len = len(x_labels)
            y_len = len(y_labels)
            z_len = len(z_labels)
        else:
            checks.assert_ndim(data, 3)
            data = np.asarray(data)
            x_len, y_len, z_len = data.shape
        if x_labels is None:
            x_labels = np.arange(x_len)
        if y_labels is None:
            y_labels = np.arange(y_len)
        if z_labels is None:
            z_labels = np.arange(z_len)
        x_labels = np.asarray(x_labels)
        y_labels = np.asarray(y_labels)
        z_labels = np.asarray(z_labels)

        if fig is None:
            fig = FigureWidget()
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
            x_labels = np.arange(x_len)
            more_layout[scene_name] = dict(xaxis=dict(ticktext=x_ticktext, tickvals=x_labels, tickmode='array'))
        if not np.issubdtype(y_labels.dtype, np.number):
            y_ticktext = y_labels
            y_labels = np.arange(y_len)
            more_layout[scene_name] = dict(yaxis=dict(ticktext=y_ticktext, tickvals=y_labels, tickmode='array'))
        if not np.issubdtype(z_labels.dtype, np.number):
            z_ticktext = z_labels
            z_labels = np.arange(z_len)
            more_layout[scene_name] = dict(zaxis=dict(ticktext=z_ticktext, tickvals=z_labels, tickmode='array'))
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
            opacity=0.2,
            surface_count=15,  # keep low for big data
            colorscale='Plasma'
        )
        volume.update(**trace_kwargs)
        fig.add_trace(volume, **add_trace_kwargs)

        TraceUpdater.__init__(self, fig, [fig.data[-1]])

        if data is not None:
            self.update(data)

    def update(self, data):
        """Update the trace data."""
        data = np.asarray(data).flatten()

        with self.fig.batch_update():
            self.traces[0].value = data
