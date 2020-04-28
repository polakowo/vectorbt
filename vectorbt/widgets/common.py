"""Common functions and classes for working with widgets."""

import plotly.graph_objects as go

from vectorbt import defaults

class FigureWidget(go.FigureWidget):
    def __init__(self):
        """Subclass of the [`plotly.graph_objects.FigureWidget`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.html?highlight=figurewidget#plotly.graph_objects.FigureWidget) class initialized 
        with default parameters from `vectorbt.defaults.layout`."""
        super().__init__()
        # Good default params
        self.update_layout(**defaults.layout)
        # You can then update them using update_layout

    def show_png(self):
        """Display the widget in PNG format."""
        width = self.layout.width
        height = self.layout.height
        self.show(renderer="png", width=width, height=height)


class UpdatableFigureWidget(FigureWidget):
    def __init__(self):
        """Subclass of the `vectorbt.widgets.common.FigureWidget` class with an abstract update method."""
        super().__init__()

    def update_data(self, *args, **kwargs):
        """Abstract method for updating the widget with new data."""
        raise NotImplementedError
