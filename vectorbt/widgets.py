import numpy as np
import plotly.graph_objects as go
from vectorbt.decorators import *


class FigureWidget(go.FigureWidget):
    """Subclass of the graph_objects.FigureWidget class with default params."""
    def __init__(self):
        super().__init__()
        # Good default params
        self.update_layout(
            autosize=False,
            width=800,
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

            # Set y-axis range, mainly for fill='tozeroy'
            vmin, vmax = np.min(data), np.max(data)
            space = 0.05 * (vmax - vmin)
            self.update_yaxes(range=(vmin - space, vmax + space))


class Histogram(UpdatableFigureWidget):
    """Accepts a matrix and draws a histogram for each array along 1st axis."""

    @required('data_labels')
    def __init__(self,
                 data=None,
                 data_labels=None,
                 histogram_kwargs={}, 
                 **layout_kwargs):

        super().__init__()
        self.update_layout(showlegend=True, barmode='overlay')
        self.update_layout(**layout_kwargs)
        self._data_labels = data_labels
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
