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

# Portfolio
portfolio = Config(
    investment=1.,
    slippage=0.,
    commission=0.
)

# Broadcasting
broadcast = Config(
    index_from='strict',
    columns_from='stack',
    ignore_single=True,
    drop_duplicates=True,
    keep='last'
)
