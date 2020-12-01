"""Utilities for displaying widgets."""

import plotly.graph_objects as go


class CustomFigure(go.Figure):
    def __init__(self, *args, **kwargs):
        """Subclass of the `plotly.graph_objects.Figure` class initialized
        with default parameters from `vectorbt.defaults.layout`."""
        from vectorbt import defaults

        layout = kwargs.pop('layout', {})
        super().__init__(*args, **kwargs)
        self.update_layout(**defaults.layout)
        self.update_layout(colorway=list(defaults.color_schema.values()))
        self.update_layout(**layout)

    def show_png(self):
        """Display the widget in PNG format."""
        self.show(renderer="png", width=self.layout.width, height=self.layout.height)


class CustomFigureWidget(go.FigureWidget):
    def __init__(self, *args, **kwargs):
        """Subclass of the `plotly.graph_objects.FigureWidget` class initialized
        with default parameters from `vectorbt.defaults.layout`."""
        from vectorbt import defaults

        layout = kwargs.pop('layout', {})
        super().__init__(*args, **kwargs)
        self.update_layout(**defaults.layout)
        self.update_layout(colorway=list(defaults.color_schema.values()))
        self.update_layout(**layout)

    def show_png(self):
        """Display the widget in PNG format."""
        self.show(renderer="png", width=self.layout.width, height=self.layout.height)
