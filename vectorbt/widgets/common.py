"""Common functions and classes for working with widgets."""

import plotly.graph_objects as go

from vectorbt.utils.common import Config

# You can change this from code
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