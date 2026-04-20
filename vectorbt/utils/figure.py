# Copyright (c) 2017-2026 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Utilities for constructing and displaying figures."""

import plotly.graph_objects as go
from plotly.graph_objects import Figure as _Figure, FigureWidget as _FigureWidget
from plotly.subplots import make_subplots as _make_subplots

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from vectorbt import _typing as tp
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.plotting_protocol import Capability, FigureProtocol

_ALL_CAPABILITIES = (
    Capability.TIME_SERIES | Capability.OHLC | Capability.LINE | Capability.AREA
    | Capability.HISTOGRAM | Capability.BAR | Capability.MARKERS | Capability.HLINE
    | Capability.ZONE | Capability.GAUGE | Capability.HEATMAP | Capability.BOX
    | Capability.SCATTER_XY | Capability.VOLUME_3D
)


def get_domain(ref: str, fig: tp.BaseFigure) -> tp.Tuple[int, int]:
    """Get domain of a coordinate axis."""
    axis = ref[0] + "axis" + ref[1:]
    if axis in fig.layout:
        if "domain" in fig.layout[axis]:
            if fig.layout[axis]["domain"] is not None:
                return fig.layout[axis]["domain"]
    return 0, 1


def _validate_row_col(row: tp.Optional[int], col: tp.Optional[int]) -> None:
    """Require `row` and `col` to be both None or both non-None.

    Matches Plotly's own `add_trace` contract and keeps every `plot_*`
    method consistent — either none of the subplot coordinates, or both.
    """
    if (row is None) != (col is None):
        raise ValueError("row and col must be specified together")


def _resolve_subplot_axes(fig: tp.BaseFigure,
                          row: tp.Optional[int],
                          col: tp.Optional[int]) -> tp.Tuple[str, str]:
    """Resolve (row, col) to (xref, yref) strings, defaulting to 'x'/'y'.

    Falls back to the primary axes only on plain (non-subplot) figures, where
    `get_subplot` raises `TypeError`. Out-of-range coordinates on a real subplot
    figure raise `ValueError` and are allowed to propagate so bad inputs surface
    cleanly rather than silently drawing on the wrong axes.

    Raises `ValueError` when the target cell is empty (``specs=None``) in a
    sparse layout, and `TypeError` when the cell holds a non-Cartesian subplot
    (e.g. polar, ternary) that has no x/y axis pair.
    """
    _validate_row_col(row, col)
    if row is None and col is None:
        return 'x', 'y'
    try:
        subplot = fig.get_subplot(row, col)
    except TypeError:
        # Plain (non-subplot) figure — fall back to primary axes.
        return 'x', 'y'
    if subplot is None:
        raise ValueError(
            f"No subplot at row={row}, col={col}. The cell may be empty "
            f"(specs=None) in a sparse subplot layout."
        )
    if not hasattr(subplot, 'xaxis') or not hasattr(subplot, 'yaxis'):
        raise TypeError(
            f"Subplot at row={row}, col={col} is not a Cartesian (XY) cell "
            f"(got {type(subplot).__name__}). plot_hline and plot_zone "
            f"require Cartesian subplots."
        )
    xref = subplot.xaxis.plotly_name.replace('axis', '')
    yref = subplot.yaxis.plotly_name.replace('axis', '')
    return xref, yref


def _add_trace_kwargs(row: tp.Optional[int],
                      col: tp.Optional[int]) -> tp.Kwargs:
    """Build add_trace keyword arguments for optional row/col positioning."""
    _validate_row_col(row, col)
    kwargs: tp.Kwargs = {}
    if row is not None:
        kwargs['row'] = row
    if col is not None:
        kwargs['col'] = col
    return kwargs


class PlotlyFigureProtocolMixin:
    capabilities: tp.ClassVar[Capability] = _ALL_CAPABILITIES
    renderer_name: tp.ClassVar[str] = 'plotly'

    @property
    def native(self) -> tp.Any:
        """Return the underlying native figure object."""
        return self

    def show(self, *args, **kwargs) -> None:
        """Display the figure in PNG format."""
        raise NotImplementedError

    def show_png(self, **kwargs) -> None:
        """Display the figure in PNG format."""
        self.show(renderer="png", **kwargs)

    def show_svg(self, **kwargs) -> None:
        """Display the figure in SVG format."""
        self.show(renderer="svg", **kwargs)

    def plot_line(self,
                  x: tp.ArrayLike,
                  y: tp.ArrayLike,
                  *,
                  name: tp.Optional[str] = None,
                  color: tp.Optional[str] = None,
                  width: tp.Optional[float] = None,
                  dash: tp.Optional[str] = None,
                  opacity: tp.Optional[float] = None,
                  showlegend: tp.Optional[bool] = None,
                  row: tp.Optional[int] = None,
                  col: tp.Optional[int] = None) -> Self:
        """Plot a line trace via the renderer-agnostic protocol."""
        trace_kwargs: tp.Kwargs = dict(x=x, y=y, mode='lines')
        if name is not None:
            trace_kwargs['name'] = name
        line: tp.Kwargs = {}
        if color is not None:
            line['color'] = color
        if width is not None:
            line['width'] = width
        if dash is not None:
            line['dash'] = dash
        if line:
            trace_kwargs['line'] = line
        if opacity is not None:
            trace_kwargs['opacity'] = opacity
        if showlegend is not None:
            trace_kwargs['showlegend'] = showlegend
        self.add_trace(go.Scatter(**trace_kwargs), **_add_trace_kwargs(row, col))
        return self

    def plot_markers(self,
                     x: tp.ArrayLike,
                     y: tp.ArrayLike,
                     *,
                     name: tp.Optional[str] = None,
                     color: tp.Optional[str] = None,
                     size: tp.Optional[float] = None,
                     symbol: tp.Optional[str] = None,
                     line_color: tp.Optional[str] = None,
                     line_width: tp.Optional[float] = None,
                     opacity: tp.Optional[float] = None,
                     showlegend: tp.Optional[bool] = None,
                     hover_text: tp.Union[str, tp.Sequence[str], None] = None,
                     row: tp.Optional[int] = None,
                     col: tp.Optional[int] = None) -> Self:
        """Plot a marker trace via the renderer-agnostic protocol."""
        trace_kwargs: tp.Kwargs = dict(x=x, y=y, mode='markers')
        if name is not None:
            trace_kwargs['name'] = name
        marker: tp.Kwargs = {}
        if color is not None:
            marker['color'] = color
        if size is not None:
            marker['size'] = size
        if symbol is not None:
            marker['symbol'] = symbol
        marker_line: tp.Kwargs = {}
        if line_color is not None:
            marker_line['color'] = line_color
        if line_width is not None:
            marker_line['width'] = line_width
        if marker_line:
            marker['line'] = marker_line
        if marker:
            trace_kwargs['marker'] = marker
        if opacity is not None:
            trace_kwargs['opacity'] = opacity
        if showlegend is not None:
            trace_kwargs['showlegend'] = showlegend
        if hover_text is not None:
            if isinstance(hover_text, str):
                n_points = len(x) if hasattr(x, '__len__') else None
                customdata = [hover_text] * n_points if n_points is not None else [hover_text]
            else:
                customdata = list(hover_text)
                n_points = len(x) if hasattr(x, '__len__') else None
                if n_points is not None and len(customdata) != n_points:
                    raise ValueError(
                        f"hover_text sequence length ({len(customdata)}) must match "
                        f"the number of data points ({n_points})"
                    )
            trace_kwargs['customdata'] = customdata
            trace_kwargs['hovertemplate'] = '%{customdata}<extra></extra>'
        self.add_trace(go.Scatter(**trace_kwargs), **_add_trace_kwargs(row, col))
        return self

    def plot_area(self,
                  x: tp.ArrayLike,
                  y: tp.ArrayLike,
                  *,
                  name: tp.Optional[str] = None,
                  color: tp.Optional[str] = None,
                  fillcolor: tp.Optional[str] = None,
                  showlegend: tp.Optional[bool] = None,
                  row: tp.Optional[int] = None,
                  col: tp.Optional[int] = None) -> Self:
        """Plot a filled-area trace via the renderer-agnostic protocol."""
        trace_kwargs: tp.Kwargs = dict(x=x, y=y, mode='lines', fill='tozeroy')
        if name is not None:
            trace_kwargs['name'] = name
        if color is not None:
            trace_kwargs['line'] = dict(color=color)
        if fillcolor is not None:
            trace_kwargs['fillcolor'] = fillcolor
        if showlegend is not None:
            trace_kwargs['showlegend'] = showlegend
        self.add_trace(go.Scatter(**trace_kwargs), **_add_trace_kwargs(row, col))
        return self

    def plot_ohlc(self,
                  x: tp.ArrayLike,
                  open: tp.ArrayLike,
                  high: tp.ArrayLike,
                  low: tp.ArrayLike,
                  close: tp.ArrayLike,
                  *,
                  name: tp.Optional[str] = None,
                  increasing_color: tp.Optional[str] = None,
                  decreasing_color: tp.Optional[str] = None,
                  style: str = 'candlestick',
                  row: tp.Optional[int] = None,
                  col: tp.Optional[int] = None) -> Self:
        """Plot an OHLC chart via the renderer-agnostic protocol.

        `style='candlestick'` (default) emits `go.Candlestick`;
        `style='bars'` emits `go.Ohlc`. Both accept the same
        `increasing=dict(line=dict(color=...))` / `decreasing=dict(line=dict(color=...))`
        color threading.
        """
        if style == 'candlestick':
            trace_cls = go.Candlestick
        elif style == 'bars':
            trace_cls = go.Ohlc
        else:
            raise ValueError(
                f"plot_ohlc style must be 'candlestick' or 'bars', got {style!r}"
            )
        trace_kwargs: tp.Kwargs = dict(x=x, open=open, high=high, low=low, close=close)
        if name is not None:
            trace_kwargs['name'] = name
        if increasing_color is not None:
            trace_kwargs['increasing'] = dict(line=dict(color=increasing_color))
        if decreasing_color is not None:
            trace_kwargs['decreasing'] = dict(line=dict(color=decreasing_color))
        self.add_trace(trace_cls(**trace_kwargs), **_add_trace_kwargs(row, col))
        return self

    def plot_histogram(self,
                       x: tp.ArrayLike,
                       *,
                       name: tp.Optional[str] = None,
                       opacity: tp.Optional[float] = None,
                       showlegend: tp.Optional[bool] = None,
                       row: tp.Optional[int] = None,
                       col: tp.Optional[int] = None) -> Self:
        """Plot a histogram via the renderer-agnostic protocol."""
        trace_kwargs: tp.Kwargs = dict(x=x)
        if name is not None:
            trace_kwargs['name'] = name
        if opacity is not None:
            trace_kwargs['opacity'] = opacity
        if showlegend is not None:
            trace_kwargs['showlegend'] = showlegend
        self.add_trace(go.Histogram(**trace_kwargs), **_add_trace_kwargs(row, col))
        return self

    def plot_bars(self,
                  x: tp.ArrayLike,
                  y: tp.ArrayLike,
                  *,
                  name: tp.Optional[str] = None,
                  color: tp.Union[str, tp.ArrayLike, None] = None,
                  line_width: tp.Optional[float] = None,
                  opacity: tp.Optional[float] = None,
                  showlegend: tp.Optional[bool] = None,
                  row: tp.Optional[int] = None,
                  col: tp.Optional[int] = None) -> Self:
        """Plot a bar trace via the renderer-agnostic protocol.

        `color` may be a scalar or a sequence (one color per bar). Plotly's
        `go.Bar` natively dispatches on both. `line_width=0` suppresses the
        bar border (matching the volume-under-OHLCV convention).
        """
        trace_kwargs: tp.Kwargs = dict(x=x, y=y)
        if name is not None:
            trace_kwargs['name'] = name
        marker: tp.Kwargs = {}
        if color is not None:
            marker['color'] = color
        if line_width is not None:
            marker['line'] = dict(width=line_width)
        if marker:
            trace_kwargs['marker'] = marker
        if opacity is not None:
            trace_kwargs['opacity'] = opacity
        if showlegend is not None:
            trace_kwargs['showlegend'] = showlegend
        self.add_trace(go.Bar(**trace_kwargs), **_add_trace_kwargs(row, col))
        return self

    def plot_hline(self,
                   y: float,
                   *,
                   color: tp.Optional[str] = None,
                   dash: tp.Optional[str] = None,
                   width: tp.Optional[float] = None,
                   row: tp.Optional[int] = None,
                   col: tp.Optional[int] = None) -> Self:
        """Plot a horizontal line via the renderer-agnostic protocol.

        Matches the existing `fig.add_shape(type='line', xref='paper', ...)`
        pattern used throughout vectorbt (21 call sites), so downstream migration
        in issues #5/#6 is a byte-identical refactor.
        """
        xref, yref = _resolve_subplot_axes(self, row, col)
        x_domain = get_domain(xref, self)
        line: tp.Kwargs = {}
        if color is not None:
            line['color'] = color
        if dash is not None:
            line['dash'] = dash
        if width is not None:
            line['width'] = width
        shape_kwargs: tp.Kwargs = dict(
            type='line',
            xref='paper',
            yref=yref,
            x0=x_domain[0],
            x1=x_domain[1],
            y0=y,
            y1=y,
        )
        if line:
            shape_kwargs['line'] = line
        self.add_shape(**shape_kwargs)
        return self

    def plot_zone(self,
                  y0: float,
                  y1: float,
                  *,
                  color: tp.Optional[str] = None,
                  opacity: tp.Optional[float] = None,
                  row: tp.Optional[int] = None,
                  col: tp.Optional[int] = None) -> Self:
        """Plot a horizontal zone (filled rectangle spanning the full x-axis)."""
        xref, yref = _resolve_subplot_axes(self, row, col)
        x_domain = get_domain(xref, self)
        shape_kwargs: tp.Kwargs = dict(
            type='rect',
            xref='paper',
            yref=yref,
            x0=x_domain[0],
            x1=x_domain[1],
            y0=y0,
            y1=y1,
            layer='below',
            line_width=0,
        )
        if color is not None:
            shape_kwargs['fillcolor'] = color
        if opacity is not None:
            shape_kwargs['opacity'] = opacity
        self.add_shape(**shape_kwargs)
        return self


class Figure(_Figure, PlotlyFigureProtocolMixin):
    """Figure."""

    def __init__(self, *args, **kwargs) -> None:
        """Extends `plotly.graph_objects.Figure`."""
        from vectorbt._settings import settings

        plotting_cfg = settings["plotting"]

        layout = kwargs.pop("layout", {})
        super().__init__(*args, **kwargs)
        self.update_layout(**merge_dicts(plotting_cfg["layout"], layout))

    def show(self, *args, **kwargs) -> None:
        """Show the figure."""
        from vectorbt._settings import settings

        plotting_cfg = settings["plotting"]

        fig_kwargs = dict(width=self.layout.width, height=self.layout.height)
        show_kwargs = merge_dicts(fig_kwargs, plotting_cfg["show_kwargs"], kwargs)
        _Figure.show(self, *args, **show_kwargs)


class FigureWidget(_FigureWidget, PlotlyFigureProtocolMixin):
    """Figure widget."""

    def __init__(self, *args, **kwargs) -> None:
        """Extends `plotly.graph_objects.FigureWidget`."""
        from vectorbt._settings import settings

        plotting_cfg = settings["plotting"]

        layout = kwargs.pop("layout", {})
        super().__init__(*args, **kwargs)
        self.update_layout(**merge_dicts(plotting_cfg["layout"], layout))

    def show(self, *args, **kwargs) -> None:
        """Show the figure."""
        from vectorbt._settings import settings

        plotting_cfg = settings["plotting"]

        fig_kwargs = dict(width=self.layout.width, height=self.layout.height)
        show_kwargs = merge_dicts(fig_kwargs, plotting_cfg["show_kwargs"], kwargs)
        _Figure.show(self, *args, **show_kwargs)


def make_figure(*args, **kwargs) -> tp.BaseFigure:
    """Make new figure.

    Returns either `Figure` or `FigureWidget`, depending on `use_widgets`
    defined under `plotting` in `vectorbt._settings.settings`."""
    from vectorbt._settings import settings

    plotting_cfg = settings["plotting"]

    if plotting_cfg["use_widgets"]:
        return FigureWidget(*args, **kwargs)
    return Figure(*args, **kwargs)


def make_subplots(*args, **kwargs) -> tp.BaseFigure:
    """Makes subplots and passes them to `FigureWidget`."""
    return make_figure(_make_subplots(*args, **kwargs))


# Subplot-only kwargs accepted by `plotly.subplots.make_subplots`. Used by the
# router below to detect when the caller wants real subplot semantics even
# without passing rows/cols explicitly. `figure` is included so that
# `create_figure(figure=fig)` routes to `make_subplots(figure=fig)` for
# Plotly's documented populate-existing semantics, rather than falling
# through to `make_figure`'s positional-figure path.
_SUBPLOT_ONLY_KWARGS = frozenset({
    'shared_xaxes', 'shared_yaxes', 'start_cell', 'print_grid',
    'horizontal_spacing', 'vertical_spacing', 'subplot_titles',
    'column_widths', 'row_heights', 'specs', 'insets',
    'column_titles', 'row_titles', 'x_title', 'y_title', 'figure',
})

RendererFactory = tp.Callable[..., FigureProtocol]
_RENDERER_REGISTRY: tp.Dict[str, RendererFactory] = {}


def register_renderer(name: str,
                      factory: RendererFactory,
                      *,
                      override: bool = False) -> None:
    """Register a plotting renderer factory under `name`.

    Not thread-safe; call during application startup.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("renderer name must be a non-empty string")
    if not callable(factory):
        raise TypeError("factory must be callable")
    if name in _RENDERER_REGISTRY and not override:
        raise ValueError(
            f"renderer {name!r} already registered; pass override=True to replace"
        )
    _RENDERER_REGISTRY[name] = factory


def get_renderer(name: str) -> RendererFactory:
    """Look up a registered plotting renderer factory by name."""
    try:
        return _RENDERER_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown plotting renderer {name!r}; "
            f"registered: {sorted(_RENDERER_REGISTRY)}"
        )


def list_renderers() -> tp.List[str]:
    """Return the sorted list of registered plotting renderer names."""
    return sorted(_RENDERER_REGISTRY)


def _plotly_factory(*,
                    rows: tp.Optional[int] = None,
                    cols: tp.Optional[int] = None,
                    **kwargs) -> FigureProtocol:
    """Built-in Plotly renderer factory.

    With no `rows`/`cols` and no subplot-only kwargs, delegates to
    `make_figure()` for byte-identical output. Any explicit `rows`/`cols`
    (even `rows=1, cols=1`) or any subplot-only kwarg routes to
    `make_subplots()` so the resulting figure has real subplot metadata.
    """
    wants_subplots = (
        rows is not None
        or cols is not None
        or bool(_SUBPLOT_ONLY_KWARGS & kwargs.keys())
    )
    if wants_subplots:
        return make_subplots(
            rows=1 if rows is None else rows,
            cols=1 if cols is None else cols,
            **kwargs,
        )
    return make_figure(**kwargs)


def create_figure(*,
                  renderer: tp.Optional[str] = None,
                  rows: tp.Optional[int] = None,
                  cols: tp.Optional[int] = None,
                  **kwargs) -> FigureProtocol:
    """Create a figure via the registered renderer factory.

    Keyword-only. `renderer=None` reads
    `settings['plotting']['default_renderer']`.

    Note: vectorbt's `renderer=` selects the *plotting library* (e.g.
    `'plotly'`, `'lightweight_charts'`) that produces the figure. This is
    distinct from Plotly's own `fig.show(renderer=...)` kwarg, which selects
    an *output format* (`'png'`, `'svg'`, `'browser'`, ...) when displaying
    a Plotly figure.

    With no `rows`/`cols` and no subplot-only kwargs, the built-in `'plotly'`
    renderer delegates to `make_figure()` — byte-identical output. Passing
    explicit `rows`/`cols` (even `rows=1, cols=1`) or any subplot-only kwarg
    (`specs`, `shared_xaxes`, ...) routes to `make_subplots()` so the
    resulting figure has real subplot metadata and `get_subplot(...)` works.

    To wrap an already-constructed `plotly.graph_objects.Figure`, use
    `make_figure(fig)` directly — `create_figure` is keyword-only and has
    no positional seat for that case.

    Note: only code paths that go through `create_figure` honor
    `settings['plotting']['default_renderer']`. Existing plot methods built
    on `make_figure` / `make_subplots` (accessors, indicators, the
    `vbt.plotting.Gauge`/`Scatter`/... helpers, etc.) are unaffected and
    will be migrated in follow-up issues.
    """
    from vectorbt._settings import settings
    if renderer is None:
        renderer = settings['plotting']['default_renderer']
    factory = get_renderer(renderer)
    return factory(rows=rows, cols=cols, **kwargs)


# ############# Renderer resolution helpers ############# #


def resolve_renderer_for_fig(fig: tp.Any) -> tp.Tuple[str, bool]:
    """Return `(renderer_name, is_plotly)` for an optional figure.

    When `fig is None`, resolves through
    `settings['plotting']['default_renderer']`. When `fig` is a Plotly
    `BaseFigure`, returns `('plotly', True)`. Otherwise looks up
    `type(fig).renderer_name` (falls back to the class name) — never hardcodes
    a specific non-Plotly renderer name at the registry layer, so future
    third-party renderers get their own name in error messages automatically.
    """
    from vectorbt._settings import settings
    if fig is None:
        renderer = settings['plotting']['default_renderer']
        return (renderer, renderer == 'plotly')
    if isinstance(fig, tp.BaseFigure):
        return ('plotly', True)
    return (getattr(type(fig), 'renderer_name', type(fig).__name__), False)


def resolve_renderer(
    fig: tp.Any = None,
    renderer: tp.Optional[str] = None,
) -> tp.Tuple[str, bool]:
    """Return `(renderer_name, is_plotly)` from an optional figure and/or renderer override.

    Precedence:
    1. If *fig* is not None, detect from the figure object. If *renderer*
       is also provided and conflicts, raise `ValueError`.
    2. If *fig* is None and *renderer* is not None, use *renderer* directly.
    3. If both are None, fall back to
       ``settings['plotting']['default_renderer']``.
    """
    if fig is not None:
        fig_renderer, fig_is_plotly = resolve_renderer_for_fig(fig)
        if renderer is not None and renderer != fig_renderer:
            raise ValueError(
                f"renderer={renderer!r} conflicts with the supplied fig, which "
                f"is a {fig_renderer!r} renderer figure. Either omit renderer= "
                f"or pass a fig created with renderer={renderer!r}."
            )
        return (fig_renderer, fig_is_plotly)
    if renderer is not None:
        return (renderer, renderer == 'plotly')
    # Both None — fall back to global default.
    return resolve_renderer_for_fig(None)


def assert_plotly_only_kwargs(is_plotly_renderer: bool,
                              resolved_renderer: str,
                              forcing_kwargs: tp.List[str],
                              *,
                              method_name: str) -> None:
    """Raise `NotImplementedError` if `forcing_kwargs` are used on a non-Plotly renderer.

    No-op if `forcing_kwargs` is empty or `is_plotly_renderer` is True. The
    error message mirrors the #5 inline pattern at
    `vectorbt/ohlcv_accessors.py:389-398` so users see one consistent voice.
    """
    if forcing_kwargs and not is_plotly_renderer:
        raise NotImplementedError(
            f"{method_name} cannot be called with {forcing_kwargs} on the "
            f"{resolved_renderer!r} renderer. These parameters are legacy "
            f"Plotly-specific escape hatches. Prefer first-class protocol "
            f"kwargs for portable customization, or operate on a Plotly "
            f"figure directly if you need Plotly-specific styling. To "
            f"resolve: drop the kwarg, switch to the 'plotly' renderer, or "
            f"construct a Plotly figure yourself and pass it via `fig=`."
        )


def assert_plotly_only_method(is_plotly_renderer: bool,
                              resolved_renderer: str,
                              *,
                              method_name: str,
                              reason: str) -> None:
    """Raise `NotImplementedError` if a permanently-Plotly-only method is called on non-Plotly.

    Used for methods like `plot_against` and `overlay_with_heatmap` whose
    core behavior has no cross-renderer equivalent (not a specific kwarg, the
    whole method).
    """
    if not is_plotly_renderer:
        raise NotImplementedError(
            f"{method_name} is permanently Plotly-only: {reason} "
            f"(current renderer: {resolved_renderer!r}). Switch to the "
            f"'plotly' renderer or construct a Plotly figure yourself and "
            f"pass it via `fig=`."
        )


def _extract_row_col_routing(add_trace_kwargs: tp.Any) -> tp.Kwargs:
    """Pull ``row``/``col`` out of *add_trace_kwargs* for forwarding to protocol methods."""
    if not add_trace_kwargs:
        return {}
    routing: tp.Kwargs = {}
    if 'row' in add_trace_kwargs:
        routing['row'] = add_trace_kwargs['row']
    if 'col' in add_trace_kwargs:
        routing['col'] = add_trace_kwargs['col']
    return routing


register_renderer('plotly', _plotly_factory)
