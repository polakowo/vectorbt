"""Package for creating and displaying widgets.

Figure widgets enable interactive data visualization in Jupyter Notebook and JupyterLab environments.

Each widget is based upon `plotly.graph_objects.FigureWidget`.
For more details on using Plotly, see [Getting Started with Plotly in Python](https://plotly.com/python/getting-started/)."""

from vectorbt.widgets import basic, common, accessors

from vectorbt.widgets.common import DefaultFigureWidget
from vectorbt.widgets.basic import Indicator, Bar, Scatter, Histogram, Heatmap


