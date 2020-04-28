"""Figure widgets enable interactive data visualization in Jupyter Notebook and JupyterLab environments.

Each widget is based upon [plotly.graph_objects.FigureWidget](https://plotly.com/python-api-reference/generated/plotly.graph_objects.html?highlight=figurewidget#plotly.graph_objects.FigureWidget)
and extended by `vectorbt.widgets.common.FigureWidget` and `vectorbt.widgets.common.UpdatableFigureWidget`
to be created and updated easily and efficiently.

For more details on using Plotly, see [Getting Started with Plotly in Python](https://plotly.com/python/getting-started/).

## Default layout

Use `vectorbt.defaults.layout` dictionary to change the default layout.

For example, to change the default width:
```py
import vectorbt as vbt

vbt.defaults.layout['width'] = 800
```"""

from vectorbt.widgets import widgets, common, accessors

from vectorbt.widgets.common import FigureWidget, UpdatableFigureWidget
from vectorbt.widgets.widgets import Indicator, Bar, Scatter, Histogram, Heatmap


