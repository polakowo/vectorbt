"""Utilities for displaying widgets."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots as _make_subplots

from vectorbt.utils.config import merge_dicts


class Figure(go.Figure):
    def __init__(self, *args, **kwargs):
        """Subclass of the `plotly.graph_objects.Figure` class initialized
        with default parameters from `vectorbt.settings.layout`."""
        from vectorbt import settings

        layout = kwargs.pop('layout', {})
        super().__init__(*args, **kwargs)
        self.update_layout(**merge_dicts(settings.layout, layout))

    def show_png(self):
        """Display the figure in PNG format."""
        self.show(renderer="png", width=self.layout.width, height=self.layout.height)


class FigureWidget(go.FigureWidget):
    def __init__(self, *args, **kwargs):
        """Subclass of the `plotly.graph_objects.FigureWidget` class initialized
        with default parameters from `vectorbt.settings.layout`."""
        from vectorbt import settings

        layout = kwargs.pop('layout', {})
        super().__init__(*args, **kwargs)
        self.update_layout(**merge_dicts(settings.layout, layout))

    def show_png(self):
        """Display the widget in PNG format."""
        self.show(renderer="png", width=self.layout.width, height=self.layout.height)


def make_subplots(*args, **kwargs):
    """Makes subplots and passes them to `FigureWidget`."""
    return FigureWidget(_make_subplots(*args, **kwargs))
