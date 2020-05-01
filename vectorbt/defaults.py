"""Default parameters for various parts of `vectorbt`."""

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
"""Default Plotly layout.

Used by `vectorbt.widgets.common.DefaultFigureWidget`."""

# Portfolio
portfolio = Config(
    investment=1.,
    slippage=0.,
    commission=0.
)
"""Default portfolio parameters.

Used by `vectorbt.portfolio.portfolio.Portfolio`."""

# Broadcasting
broadcast = Config(
    index_from='strict',
    columns_from='stack',
    ignore_single=True,
    drop_duplicates=True,
    keep='last'
)
"""Default broadcasting rules for index and columns.

Used by `vectorbt.utils.reshape_fns.broadcast_index`."""
