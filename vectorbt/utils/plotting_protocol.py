# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Backend-agnostic figure protocol and capability flags."""

import enum

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from vectorbt import _typing as tp


class Capability(enum.Flag):
    """Capability flags declaring which chart types a figure backend supports."""

    TIME_SERIES = enum.auto()
    OHLC = enum.auto()
    LINE = enum.auto()
    AREA = enum.auto()
    HISTOGRAM = enum.auto()
    BAR = enum.auto()
    MARKERS = enum.auto()
    HLINE = enum.auto()
    ZONE = enum.auto()
    GAUGE = enum.auto()
    HEATMAP = enum.auto()
    BOX = enum.auto()
    SCATTER_XY = enum.auto()
    VOLUME_3D = enum.auto()


@runtime_checkable
class FigureProtocol(Protocol):
    """Backend-agnostic figure interface."""

    capabilities: tp.ClassVar[Capability]
    backend_name: tp.ClassVar[str]

    @property
    def native(self) -> tp.Any:
        """Return the underlying native figure object."""

    def plot_line(
        self,
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
        col: tp.Optional[int] = None,
    ) -> Self:
        """Plot a line trace via the backend-agnostic protocol."""

    def plot_markers(
        self,
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
        col: tp.Optional[int] = None,
    ) -> Self:
        """Plot a marker trace via the backend-agnostic protocol.

        `line_color` and `line_width` style the marker border. Plotly maps
        them to `marker.line.color` / `marker.line.width`. LWC's
        `createSeriesMarkers` has no marker-border concept and accepts these
        parameters as no-ops for protocol parity (matching the
        `plot_bars.line_width` precedent).

        `hover_text` supplies per-marker annotation strings. A scalar string
        applies to every marker; a sequence must match the number of points.
        Plotly maps this to `customdata` + `hovertemplate='%{customdata}<extra></extra>'`
        so hovering a marker shows the string as a tooltip. LWC maps each
        string to the per-marker `tooltip.title` field consumed by litecharts'
        `marker_tooltips` plugin (rendered as a hover popup on the chart).
        """

    def plot_ohlc(
        self,
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
        col: tp.Optional[int] = None,
    ) -> Self:
        """Plot an OHLC chart via the backend-agnostic protocol.

        `style` selects the visualization:

        - `'candlestick'` (default): filled candle bodies with wicks. Plotly
          emits `go.Candlestick`; LWC emits `CandlestickSeries`.
        - `'bars'`: tick-style OHLC bars (thin vertical line per period with
          open/close ticks on either side). Plotly emits `go.Ohlc`; LWC emits
          `BarSeries`. Both backends support this natively.

        Unknown `style` values raise `ValueError`.
        """

    def plot_histogram(
        self,
        x: tp.ArrayLike,
        *,
        name: tp.Optional[str] = None,
        opacity: tp.Optional[float] = None,
        showlegend: tp.Optional[bool] = None,
        row: tp.Optional[int] = None,
        col: tp.Optional[int] = None,
    ) -> Self:
        """Plot a histogram via the backend-agnostic protocol."""

    def plot_bars(
        self,
        x: tp.ArrayLike,
        y: tp.ArrayLike,
        *,
        name: tp.Optional[str] = None,
        color: tp.Union[str, tp.ArrayLike, None] = None,
        line_width: tp.Optional[float] = None,
        opacity: tp.Optional[float] = None,
        showlegend: tp.Optional[bool] = None,
        row: tp.Optional[int] = None,
        col: tp.Optional[int] = None,
    ) -> Self:
        """Plot a bar trace via the backend-agnostic protocol.

        `color` accepts either a single color string or a sequence of per-bar
        colors (one per data point). Plotly's `go.Bar` natively dispatches on
        both shapes; LWC injects per-datapoint colors into the `HistogramSeries`
        data dicts.

        `line_width` controls the bar border width. Pass `0` to suppress the
        border entirely (the convention used by volume panes on OHLCV charts).
        LWC's `HistogramSeries` has no bar-border concept and accepts this
        parameter as a no-op for protocol parity.
        """

    def plot_hline(
        self,
        y: float,
        *,
        color: tp.Optional[str] = None,
        dash: tp.Optional[str] = None,
        width: tp.Optional[float] = None,
        row: tp.Optional[int] = None,
        col: tp.Optional[int] = None,
    ) -> Self:
        """Plot a horizontal line spanning the plot or a specific subplot."""

    def plot_zone(
        self,
        y0: float,
        y1: float,
        *,
        color: tp.Optional[str] = None,
        opacity: tp.Optional[float] = None,
        row: tp.Optional[int] = None,
        col: tp.Optional[int] = None,
    ) -> Self:
        """Plot a horizontal zone (filled rectangle spanning the full x-axis)."""

    def plot_area(
        self,
        x: tp.ArrayLike,
        y: tp.ArrayLike,
        *,
        name: tp.Optional[str] = None,
        color: tp.Optional[str] = None,
        fillcolor: tp.Optional[str] = None,
        showlegend: tp.Optional[bool] = None,
        row: tp.Optional[int] = None,
        col: tp.Optional[int] = None,
    ) -> Self:
        """Plot a filled area trace via the backend-agnostic protocol."""

    def show(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        """Display the figure."""

    def to_html(self, *args: tp.Any, **kwargs: tp.Any) -> str:
        """Render the figure as an HTML string."""
