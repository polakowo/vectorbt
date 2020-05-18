"""Global defaults."""

from vectorbt.utils.common import Config

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
"""Default Plotly layout."""

# Portfolio
portfolio = Config(
    init_capital=1.,
    fees=0.,
    slippage=0.,
    year_freq='1Y',
    risk_free=0.,
    required_return=0.,
    cutoff=0.05
)
"""Default portfolio parameters."""

# Broadcasting
broadcasting = Config(
    index_from='strict',
    columns_from='stack',
    ignore_single=True,
    drop_duplicates=True,
    keep='last'
)
"""Default broadcasting rules for index and columns.."""

# Cache
caching = True
"""If `True`, will cache properties and methods decorated accordingly.

Disable for performance tests."""
