"""Figure widgets enable interactive data visualization in Jupyter Notebook and JupyterLab environments.

Each widget is based upon [plotly.graph_objects.FigureWidget](https://plotly.com/python-api-reference/generated/plotly.graph_objects.html?highlight=figurewidget#plotly.graph_objects.FigureWidget).

For more details on using Plotly, see [Getting Started with Plotly in Python](https://plotly.com/python/getting-started/)."""

from vectorbt.widgets import widgets, common, accessors

from vectorbt.widgets.common import DefaultFigureWidget
from vectorbt.widgets.widgets import Indicator, Bar, Scatter, Histogram, Heatmap


