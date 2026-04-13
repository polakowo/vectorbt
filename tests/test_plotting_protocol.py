"""Tests for the backend-agnostic figure protocol and its Plotly implementation.

Each protocol method is asserted to produce output byte-identical (modulo
auto-generated trace `uid`s on widgets) to the equivalent hand-written
Plotly construction, for both `Figure` and `FigureWidget`, with and without
subplot positioning via `row`/`col`.
"""

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import pytest
from plotly.basedatatypes import BaseFigure

import vectorbt as vbt
from vectorbt.utils.figure import (
    Figure,
    FigureWidget,
    get_domain,
    make_figure,
    make_subplots,
)
from vectorbt.utils.plotting_protocol import Capability, FigureProtocol


# ############# Fixtures ############# #

index_5 = pd.DatetimeIndex([datetime(2020, 1, d) for d in range(1, 6)])
x_arr = list(index_5)
y_arr = [1.0, 2.0, 3.0, 4.0, 5.0]
open_arr = [1.0, 2.0, 3.0, 2.5, 2.0]
high_arr = [1.5, 2.5, 3.5, 3.0, 2.5]
low_arr = [0.5, 1.5, 2.5, 2.0, 1.5]
close_arr = [1.2, 2.2, 3.2, 2.7, 2.1]


# ############# Helpers ############# #

def trace_dict(trace):
    """Serialize a trace, stripping auto-generated widget `uid`s."""
    d = trace.to_plotly_json()
    d.pop("uid", None)
    return d


def shape_dict(shape):
    """Serialize a shape for comparison."""
    return shape.to_plotly_json()


def assert_last_trace_matches(fig_under_test, expected_trace):
    """Last trace on the figure must equal `expected_trace`."""
    actual = trace_dict(fig_under_test.data[-1])
    expected = trace_dict(expected_trace)
    assert actual == expected, f"trace mismatch\nactual:   {actual}\nexpected: {expected}"


def assert_last_shape_matches(fig_under_test, expected_shape_kwargs):
    """Last shape on the figure must equal the shape produced by the expected kwargs."""
    reference = go.Figure()
    reference.add_shape(**expected_shape_kwargs)
    actual = shape_dict(fig_under_test.layout.shapes[-1])
    expected = shape_dict(reference.layout.shapes[-1])
    assert actual == expected, f"shape mismatch\nactual:   {actual}\nexpected: {expected}"


FIG_CLASSES = [Figure, FigureWidget]


@pytest.fixture(params=[True, False], ids=["FigureWidget", "Figure"])
def use_widgets_setting(request):
    """Parametrize over both Figure and FigureWidget for tests that go through make_subplots()."""
    saved = vbt.settings['plotting']['use_widgets']
    vbt.settings['plotting']['use_widgets'] = request.param
    yield request.param
    vbt.settings['plotting']['use_widgets'] = saved


# ############# Protocol conformance ############# #

class TestProtocolConformance:
    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_isinstance_figure_protocol(self, cls):
        """vbt figures structurally satisfy FigureProtocol."""
        fig = cls()
        assert isinstance(fig, FigureProtocol)

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_isinstance_base_figure(self, cls):
        """vbt figures remain BaseFigure subclasses (no wrapping)."""
        fig = cls()
        assert isinstance(fig, BaseFigure)

    def test_figure_still_go_figure(self):
        """Figure is still a go.Figure subclass."""
        assert isinstance(Figure(), go.Figure)

    def test_figurewidget_still_go_figurewidget(self):
        """FigureWidget is still a go.FigureWidget subclass (sibling of go.Figure)."""
        assert isinstance(FigureWidget(), go.FigureWidget)
        assert not isinstance(FigureWidget(), go.Figure)

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_native_returns_self(self, cls):
        """`.native` returns the figure itself — no wrapping."""
        fig = cls()
        assert fig.native is fig

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_to_html_returns_string(self, cls):
        """to_html() is declared on FigureProtocol and must return an HTML string."""
        fig = cls()
        fig.plot_line(x_arr, y_arr, name="smoke")
        html = fig.to_html()
        assert isinstance(html, str)
        assert "<div" in html.lower()

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_capabilities_has_all_flags(self, cls):
        """Plotly-backed figures declare support for every capability flag."""
        fig = cls()
        for flag in Capability:
            assert flag in fig.capabilities, f"missing capability: {flag.name}"

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_capabilities_is_class_level(self, cls):
        """`capabilities` is a class-level constant, not a per-instance attribute."""
        assert cls.capabilities is cls().capabilities


# ############# add_line ############# #

class TestPlotLine:
    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_minimal(self, cls):
        fig = cls()
        fig.plot_line(x_arr, y_arr)
        assert_last_trace_matches(fig, go.Scatter(x=x_arr, y=y_arr, mode="lines"))

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_with_name(self, cls):
        fig = cls()
        fig.plot_line(x_arr, y_arr, name="series")
        assert_last_trace_matches(
            fig, go.Scatter(x=x_arr, y=y_arr, mode="lines", name="series")
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_full_styling(self, cls):
        fig = cls()
        fig.plot_line(
            x_arr, y_arr,
            name="series",
            color="red",
            width=2.5,
            dash="dash",
            opacity=0.75,
            showlegend=True,
        )
        assert_last_trace_matches(
            fig,
            go.Scatter(
                x=x_arr, y=y_arr, mode="lines", name="series",
                line=dict(color="red", width=2.5, dash="dash"),
                opacity=0.75, showlegend=True,
            ),
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_none_params_omitted(self, cls):
        """Passing color=None must not emit an explicit `color: None` in the trace."""
        fig = cls()
        fig.plot_line(x_arr, y_arr, color=None, width=None, dash=None)
        dumped = trace_dict(fig.data[-1])
        assert "line" not in dumped


# ############# add_markers ############# #

class TestPlotMarkers:
    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_minimal(self, cls):
        fig = cls()
        fig.plot_markers(x_arr, y_arr)
        assert_last_trace_matches(fig, go.Scatter(x=x_arr, y=y_arr, mode="markers"))

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_full_styling(self, cls):
        fig = cls()
        fig.plot_markers(
            x_arr, y_arr,
            name="pts",
            color="blue",
            size=10,
            symbol="triangle-up",
            opacity=0.5,
            showlegend=False,
        )
        assert_last_trace_matches(
            fig,
            go.Scatter(
                x=x_arr, y=y_arr, mode="markers", name="pts",
                marker=dict(color="blue", size=10, symbol="triangle-up"),
                opacity=0.5, showlegend=False,
            ),
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_line_color_only(self, cls):
        fig = cls()
        fig.plot_markers(x_arr, y_arr, line_color="red")
        assert_last_trace_matches(
            fig,
            go.Scatter(
                x=x_arr, y=y_arr, mode="markers",
                marker=dict(line=dict(color="red")),
            ),
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_line_width_only(self, cls):
        fig = cls()
        fig.plot_markers(x_arr, y_arr, line_width=2)
        assert_last_trace_matches(
            fig,
            go.Scatter(
                x=x_arr, y=y_arr, mode="markers",
                marker=dict(line=dict(width=2)),
            ),
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_line_color_and_width_combined(self, cls):
        fig = cls()
        fig.plot_markers(
            x_arr, y_arr,
            color="blue",
            size=8,
            symbol="circle",
            line_color="darkblue",
            line_width=1,
        )
        assert_last_trace_matches(
            fig,
            go.Scatter(
                x=x_arr, y=y_arr, mode="markers",
                marker=dict(
                    color="blue",
                    size=8,
                    symbol="circle",
                    line=dict(color="darkblue", width=1),
                ),
            ),
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_hover_text_scalar_broadcasts(self, cls):
        fig = cls()
        fig.plot_markers(x_arr, y_arr, hover_text="ping")
        trace = fig.data[-1]
        assert list(trace.customdata) == ["ping"] * len(x_arr)
        assert trace.hovertemplate == "%{customdata}<extra></extra>"

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_hover_text_sequence_per_marker(self, cls):
        fig = cls()
        labels = [f"pt{i}" for i in range(len(x_arr))]
        fig.plot_markers(x_arr, y_arr, hover_text=labels)
        trace = fig.data[-1]
        assert list(trace.customdata) == labels
        assert trace.hovertemplate == "%{customdata}<extra></extra>"

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_hover_text_sequence_length_mismatch_raises(self, cls):
        """hover_text sequence whose length differs from x must raise ValueError."""
        fig = cls()
        with pytest.raises(ValueError, match="hover_text sequence length"):
            fig.plot_markers(x_arr, y_arr, hover_text=["a"])

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_hover_text_none_leaves_trace_unchanged(self, cls):
        fig = cls()
        fig.plot_markers(x_arr, y_arr)
        trace = fig.data[-1]
        assert trace.customdata is None
        assert trace.hovertemplate is None


# ############# add_ohlc ############# #

class TestPlotOhlc:
    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_minimal_produces_candlestick(self, cls):
        """add_ohlc must emit a Candlestick, not an Ohlc (matches vectorbt convention)."""
        fig = cls()
        fig.plot_ohlc(x_arr, open_arr, high_arr, low_arr, close_arr)
        assert fig.data[-1].type == "candlestick"
        assert_last_trace_matches(
            fig,
            go.Candlestick(
                x=x_arr, open=open_arr, high=high_arr, low=low_arr, close=close_arr
            ),
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_with_colors(self, cls):
        fig = cls()
        fig.plot_ohlc(
            x_arr, open_arr, high_arr, low_arr, close_arr,
            name="OHLC",
            increasing_color="green",
            decreasing_color="orange",
        )
        assert_last_trace_matches(
            fig,
            go.Candlestick(
                x=x_arr, open=open_arr, high=high_arr, low=low_arr, close=close_arr,
                name="OHLC",
                increasing=dict(line=dict(color="green")),
                decreasing=dict(line=dict(color="orange")),
            ),
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_style_default_is_candlestick(self, cls):
        """Omitting style= must be byte-identical to style='candlestick'."""
        fig_default = cls()
        fig_default.plot_ohlc(x_arr, open_arr, high_arr, low_arr, close_arr)
        fig_explicit = cls()
        fig_explicit.plot_ohlc(
            x_arr, open_arr, high_arr, low_arr, close_arr, style='candlestick'
        )
        assert trace_dict(fig_default.data[-1]) == trace_dict(fig_explicit.data[-1])

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_style_bars_minimal(self, cls):
        fig = cls()
        fig.plot_ohlc(
            x_arr, open_arr, high_arr, low_arr, close_arr, style='bars'
        )
        assert fig.data[-1].type == "ohlc"
        assert_last_trace_matches(
            fig,
            go.Ohlc(
                x=x_arr, open=open_arr, high=high_arr, low=low_arr, close=close_arr
            ),
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_style_bars_with_colors(self, cls):
        fig = cls()
        fig.plot_ohlc(
            x_arr, open_arr, high_arr, low_arr, close_arr,
            name="OHLC",
            increasing_color="green",
            decreasing_color="orange",
            style='bars',
        )
        assert_last_trace_matches(
            fig,
            go.Ohlc(
                x=x_arr, open=open_arr, high=high_arr, low=low_arr, close=close_arr,
                name="OHLC",
                increasing=dict(line=dict(color="green")),
                decreasing=dict(line=dict(color="orange")),
            ),
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_style_invalid_raises(self, cls):
        fig = cls()
        with pytest.raises(ValueError, match="style"):
            fig.plot_ohlc(
                x_arr, open_arr, high_arr, low_arr, close_arr, style='foo',
            )

    def test_style_bars_subplot_routing(self, use_widgets_setting):
        """style='bars' must route to row/col subplots the same way as candlestick."""
        fig = make_subplots(rows=2, cols=1)
        fig.plot_ohlc(
            x_arr, open_arr, high_arr, low_arr, close_arr, style='bars',
            row=2, col=1,
        )
        last = fig.data[-1]
        assert last.type == "ohlc"
        assert last.xaxis == 'x2'
        assert last.yaxis == 'y2'


# ############# add_histogram ############# #

class TestPlotHistogram:
    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_minimal(self, cls):
        fig = cls()
        fig.plot_histogram([1, 1, 2, 3, 3, 3])
        assert fig.data[-1].type == "histogram"
        assert_last_trace_matches(fig, go.Histogram(x=[1, 1, 2, 3, 3, 3]))

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_with_opacity_and_name(self, cls):
        fig = cls()
        fig.plot_histogram([1, 2, 3], name="h", opacity=0.6, showlegend=True)
        assert_last_trace_matches(
            fig,
            go.Histogram(x=[1, 2, 3], name="h", opacity=0.6, showlegend=True),
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_none_params_omitted(self, cls):
        """opacity=None and name=None must not leak into the trace dict."""
        fig = cls()
        fig.plot_histogram([1, 2, 3], name=None, opacity=None)
        dumped = trace_dict(fig.data[-1])
        assert "name" not in dumped
        assert "opacity" not in dumped

# ############# add_bars ############# #

class TestPlotBars:
    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_minimal(self, cls):
        fig = cls()
        fig.plot_bars(x_arr, y_arr)
        assert fig.data[-1].type == "bar"
        assert_last_trace_matches(fig, go.Bar(x=x_arr, y=y_arr))

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_with_name(self, cls):
        fig = cls()
        fig.plot_bars(x_arr, y_arr, name="series")
        assert_last_trace_matches(fig, go.Bar(x=x_arr, y=y_arr, name="series"))

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_full_styling(self, cls):
        fig = cls()
        fig.plot_bars(
            x_arr, y_arr,
            name="series",
            color="red",
            opacity=0.75,
            showlegend=True,
        )
        assert_last_trace_matches(
            fig,
            go.Bar(
                x=x_arr, y=y_arr, name="series",
                marker=dict(color="red"),
                opacity=0.75, showlegend=True,
            ),
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_none_params_omitted(self, cls):
        """Passing color=None must not emit a `marker` block in the trace."""
        fig = cls()
        fig.plot_bars(x_arr, y_arr, color=None)
        dumped = trace_dict(fig.data[-1])
        assert "marker" not in dumped

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_color_array_minimal(self, cls):
        """color as a list should produce per-bar colors via marker.color."""
        fig = cls()
        colors = ["red", "green", "blue", "yellow", "purple"]
        fig.plot_bars(x_arr, y_arr, color=colors)
        assert_last_trace_matches(
            fig, go.Bar(x=x_arr, y=y_arr, marker=dict(color=colors)),
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_color_array_with_name(self, cls):
        fig = cls()
        colors = ["red", "green", "blue", "yellow", "purple"]
        fig.plot_bars(x_arr, y_arr, name="volume", color=colors)
        assert_last_trace_matches(
            fig,
            go.Bar(x=x_arr, y=y_arr, name="volume", marker=dict(color=colors)),
        )

    def test_color_array_subplot_routing(self, use_widgets_setting):
        """Per-bar colors must still route cleanly to row/col subplots."""
        fig = make_subplots(rows=2, cols=1)
        colors = ["red", "green", "blue", "yellow", "purple"]
        fig.plot_bars(x_arr, y_arr, color=colors, row=2, col=1)
        last = fig.data[-1]
        assert last.type == "bar"
        assert last.xaxis == 'x2'
        assert last.yaxis == 'y2'
        assert list(last.marker.color) == colors

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_line_width_zero(self, cls):
        """line_width=0 should emit marker.line.width=0 for border suppression."""
        fig = cls()
        fig.plot_bars(x_arr, y_arr, line_width=0)
        assert_last_trace_matches(
            fig, go.Bar(x=x_arr, y=y_arr, marker=dict(line=dict(width=0))),
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_line_width_with_color(self, cls):
        """line_width and color must both flow into the same marker dict."""
        fig = cls()
        colors = ["red", "green", "blue", "yellow", "purple"]
        fig.plot_bars(x_arr, y_arr, color=colors, line_width=0)
        assert_last_trace_matches(
            fig,
            go.Bar(
                x=x_arr, y=y_arr,
                marker=dict(color=colors, line=dict(width=0)),
            ),
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_line_width_none_omitted(self, cls):
        """line_width=None must not leak into the trace dict."""
        fig = cls()
        fig.plot_bars(x_arr, y_arr, line_width=None)
        dumped = trace_dict(fig.data[-1])
        assert "marker" not in dumped


# ############# add_area ############# #

class TestPlotArea:
    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_minimal(self, cls):
        fig = cls()
        fig.plot_area(x_arr, y_arr)
        assert_last_trace_matches(
            fig, go.Scatter(x=x_arr, y=y_arr, mode="lines", fill="tozeroy")
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_with_colors(self, cls):
        fig = cls()
        fig.plot_area(
            x_arr, y_arr,
            name="fill",
            color="purple",
            fillcolor="rgba(128,0,128,0.3)",
            showlegend=False,
        )
        assert_last_trace_matches(
            fig,
            go.Scatter(
                x=x_arr, y=y_arr, mode="lines", fill="tozeroy", name="fill",
                line=dict(color="purple"),
                fillcolor="rgba(128,0,128,0.3)",
                showlegend=False,
            ),
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_none_params_omitted(self, cls):
        """color=None must not emit a line key; fillcolor=None must not emit fillcolor."""
        fig = cls()
        fig.plot_area(x_arr, y_arr, color=None, fillcolor=None)
        dumped = trace_dict(fig.data[-1])
        assert "line" not in dumped
        assert "fillcolor" not in dumped


# ############# add_hline ############# #

class TestPlotHline:
    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_minimal_single_plot(self, cls):
        fig = cls()
        fig.plot_hline(y=5.0)
        assert len(fig.layout.shapes) == 1
        assert_last_shape_matches(
            fig,
            dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=5.0, y1=5.0),
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_with_styling(self, cls):
        fig = cls()
        fig.plot_hline(y=3.14, color="gray", dash="dash", width=1.5)
        assert_last_shape_matches(
            fig,
            dict(
                type="line",
                xref="paper",
                yref="y",
                x0=0, x1=1, y0=3.14, y1=3.14,
                line=dict(color="gray", dash="dash", width=1.5),
            ),
        )

    def test_subplot_domain_resolution(self, use_widgets_setting):
        """Critical regression: plot_hline in a subplot must use vectorbt's pattern.

        Pins the resolution chain: (row, col) -> get_subplot -> xaxis.plotly_name
        -> get_domain(xref, fig) -> add_shape(xref='paper', yref='yN', x0/x1).
        """
        fig = make_subplots(rows=2, cols=1)
        # Give row=2 real data so Plotly assigns x2/y2
        fig.plot_line(x_arr, y_arr, row=2, col=1)
        fig.plot_hline(y=7.0, row=2, col=1)

        shape = fig.layout.shapes[-1]
        assert shape.type == "line"
        assert shape.xref == "paper"
        assert shape.yref == "y2"
        expected_domain = get_domain("x2", fig)
        assert shape.x0 == expected_domain[0]
        assert shape.x1 == expected_domain[1]
        assert shape.y0 == 7.0
        assert shape.y1 == 7.0

    def test_subplot_row_1_uses_y1(self, use_widgets_setting):
        """Row 1 in a subplot figure still resolves to yref='y' (not 'y1')."""
        fig = make_subplots(rows=2, cols=1)
        fig.plot_line(x_arr, y_arr, row=1, col=1)
        fig.plot_hline(y=1.0, row=1, col=1)
        shape = fig.layout.shapes[-1]
        assert shape.yref == "y"


# ############# plot_zone ############# #

class TestPlotZone:
    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_minimal_single_plot(self, cls):
        fig = cls()
        fig.plot_zone(y0=1.0, y1=3.0)
        assert len(fig.layout.shapes) == 1
        assert_last_shape_matches(
            fig,
            dict(type="rect", xref="paper", yref="y",
                 x0=0, x1=1, y0=1.0, y1=3.0,
                 layer="below", line_width=0),
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_with_color_and_opacity(self, cls):
        fig = cls()
        fig.plot_zone(y0=2.0, y1=4.0, color="green", opacity=0.3)
        assert_last_shape_matches(
            fig,
            dict(type="rect", xref="paper", yref="y",
                 x0=0, x1=1, y0=2.0, y1=4.0,
                 layer="below", line_width=0,
                 fillcolor="green", opacity=0.3),
        )

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_none_params_omitted(self, cls):
        """color=None must not emit fillcolor; opacity=None must not emit opacity."""
        fig = cls()
        fig.plot_zone(y0=0.0, y1=1.0, color=None, opacity=None)
        dumped = shape_dict(fig.layout.shapes[-1])
        assert "fillcolor" not in dumped
        assert "opacity" not in dumped

    def test_subplot_domain_resolution(self, use_widgets_setting):
        """plot_zone in a subplot must resolve yref and x-domain correctly."""
        fig = make_subplots(rows=2, cols=1)
        fig.plot_line(x_arr, y_arr, row=2, col=1)
        fig.plot_zone(y0=1.0, y1=5.0, row=2, col=1)

        shape = fig.layout.shapes[-1]
        assert shape.type == "rect"
        assert shape.xref == "paper"
        assert shape.yref == "y2"
        expected_domain = get_domain("x2", fig)
        assert shape.x0 == expected_domain[0]
        assert shape.x1 == expected_domain[1]
        assert shape.y0 == 1.0
        assert shape.y1 == 5.0

    def test_subplot_row_1_uses_y(self, use_widgets_setting):
        """Row 1 in a subplot figure still resolves to yref='y' (not 'y1')."""
        fig = make_subplots(rows=2, cols=1)
        fig.plot_line(x_arr, y_arr, row=1, col=1)
        fig.plot_zone(y0=0.0, y1=2.0, row=1, col=1)
        shape = fig.layout.shapes[-1]
        assert shape.yref == "y"


# ############# Subplot positioning ############# #

class TestSubplotPositioning:
    """row/col on the trace-producing methods must route to correct subplot axes."""

    def test_add_line_row_col(self, use_widgets_setting):
        fig = make_subplots(rows=2, cols=1)
        fig.plot_line(x_arr, y_arr, row=2, col=1)
        assert fig.data[-1].xaxis == "x2"
        assert fig.data[-1].yaxis == "y2"

    def test_add_markers_row_col(self, use_widgets_setting):
        fig = make_subplots(rows=2, cols=1)
        fig.plot_markers(x_arr, y_arr, row=2, col=1)
        assert fig.data[-1].xaxis == "x2"
        assert fig.data[-1].yaxis == "y2"

    def test_add_area_row_col(self, use_widgets_setting):
        fig = make_subplots(rows=2, cols=1)
        fig.plot_area(x_arr, y_arr, row=2, col=1)
        assert fig.data[-1].xaxis == "x2"
        assert fig.data[-1].yaxis == "y2"

    def test_add_ohlc_row_col(self, use_widgets_setting):
        fig = make_subplots(rows=2, cols=1)
        fig.plot_ohlc(x_arr, open_arr, high_arr, low_arr, close_arr, row=2, col=1)
        assert fig.data[-1].xaxis == "x2"
        assert fig.data[-1].yaxis == "y2"

    def test_add_histogram_row_col(self, use_widgets_setting):
        fig = make_subplots(rows=2, cols=1)
        fig.plot_histogram(y_arr, row=2, col=1)
        assert fig.data[-1].xaxis == "x2"
        assert fig.data[-1].yaxis == "y2"

    def test_add_bars_row_col(self, use_widgets_setting):
        fig = make_subplots(rows=2, cols=1)
        fig.plot_bars(x_arr, y_arr, row=2, col=1)
        assert fig.data[-1].xaxis == "x2"
        assert fig.data[-1].yaxis == "y2"


class TestSubplotColumnRouting:
    """Exercises the col dimension of subplot routing, which the row=2, cols=1 tests do not cover.

    Uses a 1x2 layout targeting col=2 for traces and a 2x2 layout for shapes,
    so a regression that drops or ignores col would be caught.
    """

    def test_line_col_2(self, use_widgets_setting):
        fig = make_subplots(rows=1, cols=2)
        fig.plot_line(x_arr, y_arr, row=1, col=2)
        assert fig.data[-1].xaxis == "x2"
        assert fig.data[-1].yaxis == "y2"

    def test_markers_col_2(self, use_widgets_setting):
        fig = make_subplots(rows=1, cols=2)
        fig.plot_markers(x_arr, y_arr, row=1, col=2)
        assert fig.data[-1].xaxis == "x2"
        assert fig.data[-1].yaxis == "y2"

    def test_bars_col_2(self, use_widgets_setting):
        fig = make_subplots(rows=1, cols=2)
        fig.plot_bars(x_arr, y_arr, row=1, col=2)
        assert fig.data[-1].xaxis == "x2"
        assert fig.data[-1].yaxis == "y2"

    def test_hline_col_2_domain(self, use_widgets_setting):
        """plot_hline in col=2 of a 2x2 layout must resolve to correct yref and x-domain."""
        fig = make_subplots(rows=2, cols=2)
        fig.plot_line(x_arr, y_arr, row=1, col=2)
        fig.plot_hline(y=3.0, row=1, col=2)
        shape = fig.layout.shapes[-1]
        assert shape.yref == "y2"
        expected_domain = get_domain("x2", fig)
        assert shape.x0 == expected_domain[0]
        assert shape.x1 == expected_domain[1]

    def test_zone_col_2_domain(self, use_widgets_setting):
        """plot_zone in col=2 of a 2x2 layout must resolve to correct yref and x-domain."""
        fig = make_subplots(rows=2, cols=2)
        fig.plot_line(x_arr, y_arr, row=1, col=2)
        fig.plot_zone(y0=1.0, y1=5.0, row=1, col=2)
        shape = fig.layout.shapes[-1]
        assert shape.yref == "y2"
        expected_domain = get_domain("x2", fig)
        assert shape.x0 == expected_domain[0]
        assert shape.x1 == expected_domain[1]


# ############# Chainability ############# #

class TestChainability:
    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_all_methods_return_self(self, cls):
        fig = cls()
        assert fig.plot_line(x_arr, y_arr) is fig
        assert fig.plot_markers(x_arr, y_arr) is fig
        assert fig.plot_area(x_arr, y_arr) is fig
        assert fig.plot_ohlc(x_arr, open_arr, high_arr, low_arr, close_arr) is fig
        assert fig.plot_histogram(y_arr) is fig
        assert fig.plot_bars(x_arr, y_arr) is fig
        assert fig.plot_hline(y=3.0) is fig
        assert fig.plot_zone(y0=1.0, y1=2.0) is fig

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_fluent_chain(self, cls):
        """Multiple methods chain cleanly on one figure."""
        fig = cls()
        result = (
            fig.plot_line(x_arr, y_arr, name="a")
               .plot_markers(x_arr, y_arr, name="b")
               .plot_bars(x_arr, y_arr, name="c")
               .plot_hline(y=4.0)
               .plot_zone(y0=1.0, y1=2.0)
        )
        assert result is fig
        assert len(fig.data) == 3
        assert len(fig.layout.shapes) == 2


# ############# make_figure factory ############# #

class TestMakeFigureBackwardCompat:
    """The make_figure factory still returns the extended classes."""

    def test_returns_extended_class(self):
        fig = make_figure()
        assert isinstance(fig, (Figure, FigureWidget))
        assert isinstance(fig, FigureProtocol)

    def test_make_subplots_returns_extended_class(self):
        fig = make_subplots(rows=2, cols=1)
        assert isinstance(fig, (Figure, FigureWidget))
        assert isinstance(fig, FigureProtocol)
        assert fig.native is fig


# ############# Plotly drop-in compatibility ############# #

class TestPlotlyCompatibilityPreserved:
    """Plotly's native methods must still work on vbt figures with their full kwargs.

    Because the abstract protocol uses `plot_*` names rather than shadowing
    Plotly's `add_*` family, every Plotly method remains reachable with its
    original signature. This is the contract that lets vbt figures stay
    drop-in replacements for `go.Figure` / `go.FigureWidget`.
    """

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_add_hline_with_plotly_kwargs(self, cls):
        """Plotly's `add_hline` with `annotation_text` and `line_color` still works.

        Compared against a plain go.Figure reference so the test survives
        upstream changes to Plotly's internal serialization details.
        """
        fig = cls()
        fig.add_hline(y=5, annotation_text="target", line_color="red")
        ref = go.Figure()
        ref.add_hline(y=5, annotation_text="target", line_color="red")
        assert shape_dict(fig.layout.shapes[-1]) == shape_dict(ref.layout.shapes[-1])
        assert len(fig.layout.annotations) == len(ref.layout.annotations)
        assert fig.layout.annotations[-1].text == ref.layout.annotations[-1].text

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_add_ohlc_still_produces_ohlc_trace(self, cls):
        """Plotly's `add_ohlc` still emits an `ohlc` trace, not a candlestick."""
        fig = cls()
        fig.add_ohlc(x=x_arr, open=open_arr, high=high_arr, low=low_arr, close=close_arr)
        assert fig.data[-1].type == "ohlc"

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_add_histogram_accepts_nbinsx(self, cls):
        """Plotly's `add_histogram(nbinsx=...)` still accepts Plotly-only kwargs."""
        fig = cls()
        fig.add_histogram(x=[1, 2, 3, 4, 5], nbinsx=3)
        assert fig.data[-1].type == "histogram"
        assert fig.data[-1].nbinsx == 3

    @pytest.mark.parametrize("cls", FIG_CLASSES)
    def test_add_trace_scatter_still_works(self, cls):
        """`fig.add_trace(go.Scatter(...))` still works unchanged."""
        fig = cls()
        fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4], mode="lines", name="s"))
        assert len(fig.data) == 1
        assert fig.data[-1].name == "s"

    def test_plot_hline_and_add_hline_produce_different_shapes(self):
        """Same-named concept, different implementations — proof that our methods coexist."""
        fig = Figure()
        fig.plot_hline(y=5)            # vbt: xref='paper'
        fig.add_hline(y=5)             # plotly native: xref='x domain'
        assert fig.layout.shapes[0].xref == "paper"
        assert fig.layout.shapes[1].xref == "x domain"


# ############# Input validation ############# #

class TestInvalidSubplotCoordinates:
    """Out-of-range row/col on a real subplot figure must propagate as ValueError.

    Finding 2 from the Codex review: the earlier implementation silently caught
    ValueError and fell back to the primary axes, which produced wrong charts
    instead of errors. Only `TypeError` (plain, non-subplot figure) should be
    caught as a fallback.
    """

    def test_plot_hline_out_of_range_row_raises(self):
        fig = make_subplots(rows=2, cols=1)
        with pytest.raises(ValueError):
            fig.plot_hline(y=5, row=5, col=1)

    def test_plot_hline_out_of_range_col_raises(self):
        fig = make_subplots(rows=2, cols=1)
        with pytest.raises(ValueError):
            fig.plot_hline(y=5, row=1, col=5)

    def test_plot_hline_on_plain_figure_uses_primary_axes(self):
        """Plain (non-subplot) figures should still accept row/col without raising,
        falling back to primary axes. This is the `TypeError` branch."""
        fig = Figure()
        fig.plot_hline(y=5, row=1, col=1)
        assert fig.layout.shapes[-1].yref == "y"

    def test_plot_hline_on_polar_cell_raises(self):
        """Non-Cartesian subplot cells must raise rather than crash on missing xaxis."""
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]])
        with pytest.raises(TypeError, match="not a Cartesian"):
            fig.plot_hline(y=1, row=1, col=1)

    def test_plot_zone_on_polar_cell_raises(self):
        """plot_zone must also reject non-Cartesian cells."""
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]])
        with pytest.raises(TypeError, match="not a Cartesian"):
            fig.plot_zone(y0=0, y1=1, row=1, col=1)

    def test_plot_hline_on_sparse_none_cell_raises(self):
        """Empty cells in sparse layouts must raise, not silently misroute."""
        fig = make_subplots(rows=1, cols=2, specs=[[None, {}]])
        with pytest.raises(ValueError, match="No subplot"):
            fig.plot_hline(y=1, row=1, col=1)

    def test_plot_zone_on_sparse_none_cell_raises(self):
        """plot_zone must also reject empty cells in sparse layouts."""
        fig = make_subplots(rows=1, cols=2, specs=[[None, {}]])
        with pytest.raises(ValueError, match="No subplot"):
            fig.plot_zone(y0=0, y1=1, row=1, col=1)


class TestPartialRowColValidation:
    """Every `plot_*` method must reject partial (row xor col) subplot coordinates.

    Before the fix, `plot_hline(row=2)` silently defaulted `col` to 1 while the
    trace-producing methods raised. That inconsistency meant the same protocol
    signature behaved differently depending on which method you called. Now the
    shared `_validate_row_col` helper enforces "both or neither" everywhere.
    """

    def _call(self, method_name, fig, **extra):
        fn = getattr(fig, method_name)
        if method_name == "plot_hline":
            return fn(y=5, **extra)
        if method_name == "plot_zone":
            return fn(y0=0, y1=1, **extra)
        if method_name == "plot_ohlc":
            return fn(x_arr, open_arr, high_arr, low_arr, close_arr, **extra)
        if method_name == "plot_histogram":
            return fn(y_arr, **extra)
        return fn(x_arr, y_arr, **extra)

    @pytest.mark.parametrize(
        "method",
        ["plot_line", "plot_markers", "plot_area", "plot_ohlc", "plot_histogram", "plot_bars", "plot_hline", "plot_zone"],
    )
    def test_row_without_col_raises(self, method):
        fig = make_subplots(rows=2, cols=1)
        with pytest.raises(ValueError, match="row and col"):
            self._call(method, fig, row=2)

    @pytest.mark.parametrize(
        "method",
        ["plot_line", "plot_markers", "plot_area", "plot_ohlc", "plot_histogram", "plot_bars", "plot_hline", "plot_zone"],
    )
    def test_col_without_row_raises(self, method):
        fig = make_subplots(rows=2, cols=1)
        with pytest.raises(ValueError, match="row and col"):
            self._call(method, fig, col=1)
