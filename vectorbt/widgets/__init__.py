"""Modules for creating and displaying widgets.

Figure widgets enable interactive data visualization in Jupyter Notebook and JupyterLab environments.

Each widget is based upon `plotly.graph_objects.FigureWidget`.
For more details on using Plotly, see [Getting Started with Plotly in Python](https://plotly.com/python/getting-started/).

## Basic widgets

`vectorbt.widgets.basic` provides basic widgets for visualizing data in an efficient and convenient way.

## Accessors

You can access all widgets as `pd.Series.vbt.*` and `pd.DataFrame.vbt.*`.

```python-repl
>>> # vectorbt.widgets.accessors.Histogram_Accessor
>>> pd.Series(np.random.normal(size=100000)).vbt.hist()
```

![](/vectorbt/docs/img/hist_normal.png)
"""

from vectorbt.widgets import accessors, basic, common

from vectorbt.widgets.common import DefaultFigureWidget
from vectorbt.widgets.basic import Indicator, Bar, Scatter, Histogram, Box, Heatmap
