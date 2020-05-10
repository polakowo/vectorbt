"""Default parameters for various parts of `vectorbt`."""


class Config(dict):
    """A simple dict with (optionally) frozen keys."""

    def __init__(self, *args, frozen=True, **kwargs):
        self.frozen = frozen
        self.update(*args, **kwargs)
        self.default_config = dict(self)
        for key, value in dict.items(self):
            if isinstance(value, dict):
                dict.__setitem__(self, key, Config(value, frozen=frozen))

    def __setitem__(self, key, val):
        if self.frozen and key not in self:
            raise KeyError(f"Key {key} is not a valid parameter")
        dict.__setitem__(self, key, val)

    def reset(self):
        """Reset dictionary to the one passed at instantiation."""
        self.update(self.default_config)


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
    init_capital=1.,
    fees=0.,
    slippage=0.
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

# Cache
cached_property = True
"""If `True`, will cache properties decorated with `@cached_property`.

Used by `vectorbt.utils.common.cached_property`.

Disable for performance tests."""
