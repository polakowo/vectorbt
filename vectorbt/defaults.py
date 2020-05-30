"""Global defaults."""

import json

from vectorbt.utils.config import Config

__pdoc__ = {}

# Layout
layout = Config(
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
"""_"""

__pdoc__['layout'] = f"""Default Plotly layout.

```plaintext
{json.dumps(layout, indent=2)}
```
"""

# Portfolio
portfolio = Config(
    init_capital=100.,
    fees=0.,
    fixed_fees=0.,
    slippage=0.,
    year_freq='1Y',
    risk_free=0.,
    required_return=0.,
    cutoff=0.05
)
"""_"""

__pdoc__['portfolio'] = f"""Default portfolio parameters.

```plaintext
{json.dumps(portfolio, indent=2)}
```
"""

# Broadcasting
broadcasting = Config(
    index_from='strict',
    columns_from='stack',
    ignore_single=True,
    drop_duplicates=True,
    keep='last'
)
"""_"""

__pdoc__['broadcasting'] = f"""Default broadcasting rules for index and columns.

```plaintext
{json.dumps(broadcasting, indent=2)}
```
"""

# Cache
caching = True
"""If `True`, will cache properties and methods decorated accordingly.

Disable for performance tests."""
