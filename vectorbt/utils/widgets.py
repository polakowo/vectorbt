"""Utilities for displaying widgets."""

import plotly.graph_objects as go

from vectorbt import defaults


class CustomFigure(go.Figure):
    """Subclass of the `plotly.graph_objects.Figure` class initialized
    with default parameters from `vectorbt.defaults.layout`."""

    def __init__(self, *args, **kwargs):
        layout = kwargs.pop('layout', {})
        super().__init__(*args, **kwargs)
        self.update_layout(**defaults.layout)
        self.update_layout(**layout)

    def show_png(self):
        """Display the widget in PNG format."""
        self.show(renderer="png", width=self.layout.width, height=self.layout.height)


class CustomFigureWidget(go.FigureWidget):
    """Subclass of the `plotly.graph_objects.FigureWidget` class initialized
        with default parameters from `vectorbt.defaults.layout`."""

    def __init__(self, *args, **kwargs):
        layout = kwargs.pop('layout', {})
        super().__init__(*args, **kwargs)
        self.update_layout(**defaults.layout)
        self.update_layout(**layout)

    def show_png(self):
        """Display the widget in PNG format."""
        self.show(renderer="png", width=self.layout.width, height=self.layout.height)
