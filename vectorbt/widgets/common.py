"""Common functions and classes."""

import plotly.graph_objects as go

from vectorbt import defaults

class DefaultFigureWidget(go.FigureWidget):
    def __init__(self, *args, **kwargs):
        """Subclass of the `plotly.graph_objects.FigureWidget` class initialized 
        with default parameters from `vectorbt.defaults.layout`."""
        layout = kwargs.pop('layout', {})
        super().__init__(*args, **kwargs)
        # Good default params
        self.update_layout(**defaults.layout)
        self.update_layout(**layout)
        # You can then update them using update_layout

    def show_png(self):
        """Display the widget in PNG format."""
        width = self.layout.width
        height = self.layout.height
        self.show(renderer="png", width=width, height=height)