"""Global defaults.

For example, you can change default width and height of each plot:
```python-repl
>>> import vectorbt as vbt

>>> vbt.defaults.layout['width'] = 800
>>> vbt.defaults.layout['height'] = 400
```

Changes take effect immediately."""

import numpy as np
import json

from vectorbt.utils.config import Config

__pdoc__ = {}

# Color schema
color_schema = Config(
    blue="#1f77b4",
    orange="#ff7f0e",
    green="#2ca02c",
    red="#dc3912",
    purple="#9467bd",
    brown="#8c564b",
    pink="#e377c2",
    gray="#7f7f7f",
    yellow="#bcbd22",
    cyan="#17becf"
)
"""_"""

__pdoc__['color_schema'] = f"""Color schema.

```plaintext
{json.dumps(color_schema, indent=2)}
```
"""

# Contrast color schema
contrast_color_schema = Config(
    blue='#4285F4',
    orange='#FFAA00',
    green='#37B13F',
    red='#EA4335',
    gray='#E2E2E2'
)
"""_"""

__pdoc__['contrast_color_schema'] = f"""Neon color schema.

```plaintext
{json.dumps(contrast_color_schema, indent=2)}
```
"""

# Layout
layout = Config(
    frozen=False,  # you can change the keys
    autosize=False,
    width=700,
    height=300,
    margin=dict(
        b=30,
        t=30
    ),
    hovermode='closest',
    colorway=list(color_schema.values())
)
"""_"""

__pdoc__['layout'] = f"""Plotly layout.

```plaintext
{json.dumps(layout, indent=2)}
```
"""

# OHLCV
ohlcv = Config(
    column_names=dict(
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume'
    )
)
"""_"""

__pdoc__['ohlcv'] = f"""Parameters for OHLCV.

```plaintext
{json.dumps(ohlcv, indent=2)}
```
"""

# Array wrapper
array_wrapper = Config(
    column_only_select=False,
    group_select=True,
    freq=None
)
"""_"""

__pdoc__['array_wrapper'] = f"""Parameters for array wrapper.

```plaintext
{json.dumps(array_wrapper, indent=2)}
```
"""

# Broadcasting
broadcasting = Config(
    align_index=False,
    align_columns=True,
    index_from='strict',
    columns_from='stack',
    drop_duplicates=True,
    keep='last',
    drop_redundant=True
)
"""_"""

__pdoc__['broadcasting'] = f"""Broadcasting rules for index and columns.

```plaintext
{json.dumps(broadcasting, indent=2)}
```
"""

# Cache
caching = True
"""If True, will cache properties and methods decorated accordingly.

Disable for performance tests."""

# Returns
returns = Config(
    year_freq='365 days'
)
"""_"""

__pdoc__['returns'] = f"""Parameters for returns.

```plaintext
{json.dumps(returns, indent=2)}
```
"""

# Portfolio
portfolio = Config(
    call_seq='Default',
    init_cash=100.,
    size=np.inf,
    size_type='Shares',
    fees=0.,
    fixed_fees=0.,
    slippage=0.,
    reject_prob=0.,
    min_size=1e-8,
    max_size=np.inf,
    allow_partial=True,
    raise_by_reject=False,
    accumulation_mode='Disable',
    conflict_mode='Ignore',
    signal_type='Long',
    cash_sharing=False,
    row_wise=False,
    seed=None,
    freq=None,
    incl_unrealized=False
)
"""_"""

__pdoc__['portfolio'] = f"""Parameters for portfolio.

```plaintext
{json.dumps(portfolio, indent=2)}
```
"""

