"""Regression tests for vectorbt plotting behavior.

These tests are intended to ensure that current plotting behavior is preserved as much as possible
as issue #815 progresses and the plotting abstractions are refactored.

Organized in tiers:
  Tier 1: Figure factory (make_figure, make_subplots)
  Tier 2: Core plot methods (accessors, OHLCV, Orders, Trades, Drawdowns, Ranges)
  Tier 3: Trace wrapper classes (Scatter, Bar, Histogram, Heatmap)
  Tier 4: Portfolio.plots() orchestrator
  Tier 5: Portfolio plot_* smoke tests
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from plotly.basedatatypes import BaseFigure

import vectorbt as vbt
from vectorbt.generic.plotting import Scatter, Bar, Histogram, Heatmap, Gauge, Box, Volume, TraceUpdater
from vectorbt.utils.figure import make_figure, make_subplots, Figure, FigureWidget


# ############# Fixtures ############# #

index_5 = pd.DatetimeIndex([datetime(2020, 1, d) for d in range(1, 6)])
index_8 = pd.DatetimeIndex([datetime(2020, 1, d) for d in range(1, 9)])

# Generic accessor tests
sr = pd.Series([1., 2., 3., 4., 5.], index=index_5, name='TestSR')
sr2 = pd.Series([3., 2., 1., 2., 3.], index=index_5, name='Other')
sr_above = pd.Series([5., 5., 5., 5., 5.], index=index_5, name='Above')
df = pd.DataFrame({'a': [1., 2., 3., 4., 5.], 'b': [5., 4., 3., 2., 1.]}, index=index_5)

# OHLCV tests
ohlcv_df = pd.DataFrame({
    'Open': [2., 3., 4., 3.5, 2.5],
    'High': [3., 4., 4.5, 4., 3.],
    'Low': [1.5, 2.5, 3.5, 2.5, 1.5],
    'Close': [2.5, 3.5, 4., 3., 2.],
    'Volume': [100., 110., 120., 90., 80.]
}, index=index_5)

# Drawdown tests — produces 1 recovered + 1 active drawdown
dd_ts = pd.Series([10., 12., 8., 10., 15., 11., 7., 9.], index=index_8, name='Price')

# Range tests — produces 2 closed + 1 open range
range_mask = pd.Series(
    [True, True, False, True, True, True, False, True],
    index=index_8, name='Mask'
)

# Portfolio fixtures — constructed in setup_module() after caching is disabled
pf_price = pd.Series([10., 11., 12., 10., 13.], index=index_5)
_group_by = pd.Index(['first', 'first', 'second'], name='group')
pf_simple = None
pf_multi = None
pf_grouped = None
pf_shared = None

# Indicator fixtures (need longer series for window-based indicators)
# Generated with np.random.seed(42): cumsum(randn(20))+100 for close,
# abs(randn(20)) for high/low deltas. Hardcoded to avoid global RNG mutation.
index_20 = pd.DatetimeIndex([datetime(2020, 1, d) for d in range(1, 21)])
_ind_close_vals = np.array([
    100.496714, 100.35845, 101.006138, 102.529168, 102.295015,
    102.060878, 103.640091, 104.407525, 103.938051, 104.480611,
    104.017193, 103.551464, 103.793426, 101.880146, 100.155228,
    99.59294, 98.580109, 98.894357, 97.986332, 96.574029,
])
_ind_high_delta = np.array([
    1.465649, 0.225776, 0.067528, 1.424748, 0.544383,
    0.110923, 1.150994, 0.375698, 0.600639, 0.291694,
    0.601707, 1.852278, 0.013497, 1.057711, 0.822545,
    1.220844, 0.208864, 1.95967, 1.328186, 0.196861,
])
_ind_low_delta = np.array([
    0.738467, 0.171368, 0.115648, 0.301104, 1.478522,
    0.719844, 0.460639, 1.057122, 0.343618, 1.76304,
    0.324084, 0.385082, 0.676922, 0.611676, 1.031,
    0.93128, 0.839218, 0.309212, 0.331263, 0.975545,
])
ind_close = pd.Series(_ind_close_vals, index=index_20, name='Close')
ind_high = pd.Series(_ind_close_vals + _ind_high_delta, index=index_20, name='High')
ind_low = pd.Series(_ind_close_vals - _ind_low_delta, index=index_20, name='Low')

# Signal fixtures
sig_entries = pd.Series(
    [True, False, False, True, False, False, True, False, False, False,
     True, False, False, True, False, False, True, False, False, False],
    index=index_20,
)
sig_exits = pd.Series(
    [False, False, True, False, False, True, False, False, True, False,
     False, False, True, False, False, True, False, False, True, False],
    index=index_20,
)

# 2D data for Heatmap tests
heatmap_data = np.array([[1., 2., 3.], [4., 5., 6.]])


# ############# Helpers ############# #

def traces_by_name(fig):
    """Return {name: [trace, ...]} mapping. Unnamed traces go under None key."""
    result = {}
    for t in fig.data:
        result.setdefault(t.name, []).append(t)
    return result


def named_traces(fig):
    """Return {name: trace} for first trace of each non-None name."""
    result = {}
    for t in fig.data:
        if t.name is not None and t.name not in result:
            result[t.name] = t
    return result


# ############# Setup ############# #

def setup_module():
    vbt.settings.caching.enabled = False
    vbt.settings.caching.whitelist = []
    vbt.settings.caching.blacklist = []

    global pf_simple, pf_multi, pf_grouped, pf_shared
    _order_size = pd.Series([1., 0., -1., 1., -1.])
    pf_simple = vbt.Portfolio.from_orders(
        pf_price, _order_size,
        size_type='amount', direction='both',
        fees=0.01, init_cash=1000., freq='1D'
    )
    pf_multi = vbt.Portfolio.from_orders(
        pf_price.vbt.tile(3, keys=['a', 'b', 'c']),
        _order_size, size_type='amount', direction='both',
        fees=0.01, init_cash=1000., freq='1D'
    )
    pf_grouped = pf_multi.regroup(_group_by)
    pf_shared = vbt.Portfolio.from_orders(
        pf_price.vbt.tile(3, keys=['a', 'b', 'c']),
        _order_size, size_type='amount', direction='both',
        fees=0.01, init_cash=[2000., 1000.], freq='1D',
        group_by=_group_by, cash_sharing=True
    )


def teardown_module():
    vbt.settings.reset()


# ############# Tier 1: Figure Factory ############# #

class TestFigureFactory:
    def test_make_figure_default_returns_widget(self):
        """Default use_widgets=True should return FigureWidget."""
        fig = make_figure()
        assert isinstance(fig, FigureWidget)
        assert isinstance(fig, BaseFigure)

    def test_make_figure_returns_figure_when_widgets_disabled(self):
        """Toggling use_widgets=False should return plain Figure."""
        old_val = vbt.settings.plotting['use_widgets']
        try:
            vbt.settings.plotting['use_widgets'] = False
            fig = make_figure()
            assert isinstance(fig, Figure)
            assert not isinstance(fig, FigureWidget)
            assert isinstance(fig, BaseFigure)
        finally:
            vbt.settings.plotting['use_widgets'] = old_val

    def test_make_figure_applies_default_layout(self):
        """Default layout settings should be applied to the figure."""
        defaults = vbt.settings.plotting['layout']
        fig = make_figure()
        assert fig.layout.width == defaults['width']
        assert fig.layout.height == defaults['height']
        assert fig.layout.margin.t == defaults['margin']['t']
        assert fig.layout.margin.b == defaults['margin']['b']
        assert fig.layout.margin.l == defaults['margin']['l']
        assert fig.layout.margin.r == defaults['margin']['r']

    def test_make_figure_custom_layout_merges(self):
        """Custom layout kwargs are recursively merged, not replaced."""
        defaults = vbt.settings.plotting['layout']
        fig = make_figure(layout=dict(margin=dict(t=50)))
        assert fig.layout.margin.t == 50
        # Other margin values should be preserved from defaults
        assert fig.layout.margin.b == defaults['margin']['b']

    def test_make_subplots_returns_basefigure(self):
        """make_subplots should return a BaseFigure instance."""
        fig = make_subplots(rows=2, cols=1)
        assert isinstance(fig, BaseFigure)


# ############# Tier 2: Core Plot Methods ############# #

class TestGenericAccessorPlot:
    def test_generic_sr_plot(self):
        """Series.vbt.plot() should produce 1 Scatter trace with matching data."""
        fig = sr.vbt.plot()
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) == 1
        trace = fig.data[0]
        assert isinstance(trace, go.Scatter)
        assert trace.name == 'TestSR'
        np.testing.assert_array_equal(trace.y, sr.values)

    def test_generic_df_plot(self):
        """DataFrame.vbt.plot() should produce 1 Scatter trace per column."""
        fig = df.vbt.plot()
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) == 2
        names = {t.name for t in fig.data}
        assert names == {'a', 'b'}
        for t in fig.data:
            assert isinstance(t, go.Scatter)
            np.testing.assert_array_equal(t.y, df[t.name].values)

    def test_generic_plot_return_fig_false(self):
        """return_fig=False should return Scatter wrapper, not BaseFigure."""
        result = sr.vbt.plot(return_fig=False)
        assert isinstance(result, Scatter)
        assert isinstance(result, TraceUpdater)
        assert isinstance(result.fig, BaseFigure)
        assert len(result.traces) == 1

    def test_generic_plot_adds_to_existing_fig(self):
        """Passing fig= should add traces to existing figure, not create new one."""
        existing_fig = make_figure()
        existing_fig.add_trace(go.Scatter(x=index_5, y=[100] * 5, name='Pre-existing'))
        result_fig = sr.vbt.plot(fig=existing_fig)
        assert result_fig is existing_fig
        assert len(result_fig.data) == 2
        assert result_fig.data[0].name == 'Pre-existing'
        assert result_fig.data[1].name == 'TestSR'


class TestGenericAccessorConvenience:
    """Tests for lineplot, scatterplot, barplot, histplot, boxplot convenience methods."""

    def test_lineplot(self):
        """lineplot() should produce Scatter with mode='lines' and matching data."""
        fig = sr.vbt.lineplot()
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Scatter)
        assert fig.data[0].mode == 'lines'
        np.testing.assert_array_equal(fig.data[0].y, sr.values)

    def test_scatterplot(self):
        """scatterplot() should produce Scatter with mode='markers' and matching data."""
        fig = sr.vbt.scatterplot()
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Scatter)
        assert fig.data[0].mode == 'markers'
        np.testing.assert_array_equal(fig.data[0].y, sr.values)

    def test_barplot(self):
        """barplot() should produce Bar traces with matching data."""
        fig = df.vbt.barplot()
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) == 2
        for t in fig.data:
            assert isinstance(t, go.Bar)
            np.testing.assert_array_equal(t.y, df[t.name].values)
        assert {t.name for t in fig.data} == {'a', 'b'}

    def test_histplot(self):
        """histplot() should produce Histogram traces with matching data."""
        fig = df.vbt.histplot()
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) == 2
        for t in fig.data:
            assert isinstance(t, go.Histogram)
            np.testing.assert_array_equal(t.x, df[t.name].values)
        assert {t.name for t in fig.data} == {'a', 'b'}

    def test_boxplot(self):
        """boxplot() should produce Box traces."""
        fig = df.vbt.boxplot()
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) == 2
        for t in fig.data:
            assert isinstance(t, go.Box)
        assert {t.name for t in fig.data} == {'a', 'b'}


class TestGenericPlotAgainst:
    def test_plot_against_both_regions(self):
        """plot_against with both pos and neg regions should have fill + main traces."""
        fig = sr.vbt.plot_against(sr2)
        assert isinstance(fig, BaseFigure)

        nt = named_traces(fig)
        assert 'TestSR' in nt
        assert 'Other' in nt
        assert isinstance(nt['TestSR'], go.Scatter)
        assert isinstance(nt['Other'], go.Scatter)

        # Should have unnamed fill traces
        unnamed = [t for t in fig.data if t.name is None]
        assert len(unnamed) > 0

        # Total: 4 unnamed (2 pos fill + 2 neg fill) + 2 named
        assert len(fig.data) == 6

    def test_plot_against_only_positive(self):
        """When self > other everywhere, only positive fill traces should exist."""
        fig = sr_above.vbt.plot_against(sr2)
        assert isinstance(fig, BaseFigure)

        nt = named_traces(fig)
        assert 'Above' in nt
        assert 'Other' in nt

        # 2 pos fill (hidden baseline + fill) + 2 named (Above + Other), no neg fill
        assert len(fig.data) == 4


class TestOHLCVPlot:
    def test_ohlcv_plot_ohlc_no_volume(self):
        """OHLC plot without volume should have 1 Ohlc trace."""
        fig = ohlcv_df.vbt.ohlcv.plot(show_volume=False)
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) == 1
        trace = fig.data[0]
        assert isinstance(trace, go.Ohlc)
        np.testing.assert_array_equal(trace.open, ohlcv_df['Open'].values)
        np.testing.assert_array_equal(trace.high, ohlcv_df['High'].values)
        np.testing.assert_array_equal(trace.low, ohlcv_df['Low'].values)
        np.testing.assert_array_equal(trace.close, ohlcv_df['Close'].values)

    def test_ohlcv_plot_candlestick_with_volume(self):
        """Candlestick plot with volume should have Candlestick + Bar traces."""
        fig = ohlcv_df.vbt.ohlcv.plot(plot_type='Candlestick', show_volume=True)
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) == 2
        assert isinstance(fig.data[0], go.Candlestick)
        np.testing.assert_array_equal(fig.data[0].open, ohlcv_df['Open'].values)
        np.testing.assert_array_equal(fig.data[0].high, ohlcv_df['High'].values)
        np.testing.assert_array_equal(fig.data[0].low, ohlcv_df['Low'].values)
        np.testing.assert_array_equal(fig.data[0].close, ohlcv_df['Close'].values)
        assert isinstance(fig.data[1], go.Bar)
        assert fig.data[1].name == 'Volume'
        np.testing.assert_array_equal(fig.data[1].y, ohlcv_df['Volume'].values)

    def test_ohlcv_plot_default_type(self):
        """Default plot type should be OHLC (from settings)."""
        fig = ohlcv_df.vbt.ohlcv.plot()
        assert isinstance(fig, BaseFigure)
        # Default includes volume, so 2 traces
        assert len(fig.data) == 2
        assert isinstance(fig.data[0], go.Ohlc)

    def test_ohlcv_volume_subplot_geometry(self):
        """OHLCV with volume should create 2-row subplot with correct structure."""
        fig = ohlcv_df.vbt.ohlcv.plot(show_volume=True)

        # OHLC in row 1 (yaxis), volume in row 2 (yaxis2)
        assert fig.data[0].yaxis == 'y'
        assert fig.data[1].yaxis == 'y2'

        # Price subplot should be larger than volume subplot
        price_height = fig.layout.yaxis.domain[1] - fig.layout.yaxis.domain[0]
        volume_height = fig.layout.yaxis2.domain[1] - fig.layout.yaxis2.domain[0]
        assert price_height > volume_height

        # Subplots should be adjacent (no gap)
        assert fig.layout.yaxis2.domain[1] == pytest.approx(fig.layout.yaxis.domain[0], abs=1e-6)


class TestOrdersPlot:
    def test_orders_plot_with_close(self):
        """Orders.plot() should show Close line + Buy/Sell markers."""
        fig = pf_simple.orders.plot()
        assert isinstance(fig, BaseFigure)

        nt = named_traces(fig)
        assert 'Close' in nt
        assert isinstance(nt['Close'], go.Scatter)
        np.testing.assert_array_equal(nt['Close'].y, pf_price.values)

        assert 'Buy' in nt
        assert isinstance(nt['Buy'], go.Scatter)
        assert nt['Buy'].marker.symbol == 'triangle-up'

        assert 'Sell' in nt
        assert isinstance(nt['Sell'], go.Scatter)
        assert nt['Sell'].marker.symbol == 'triangle-down'

        assert len(fig.data) == 3

    def test_orders_plot_marker_data(self):
        """Buy/Sell marker positions should match order record indices."""
        fig = pf_simple.orders.plot()
        nt = named_traces(fig)

        # Buy orders at idx 0 (2020-01-01) and idx 3 (2020-01-04)
        buy_x = list(nt['Buy'].x)
        assert len(buy_x) == 2
        assert buy_x[0] == index_5[0]
        assert buy_x[1] == index_5[3]
        np.testing.assert_array_equal(nt['Buy'].y, [10., 10.])

        # Sell orders at idx 2 (2020-01-03) and idx 4 (2020-01-05)
        sell_x = list(nt['Sell'].x)
        assert len(sell_x) == 2
        assert sell_x[0] == index_5[2]
        assert sell_x[1] == index_5[4]
        np.testing.assert_array_equal(nt['Sell'].y, [12., 13.])

    def test_orders_plot_adds_to_existing_fig(self):
        """Passing fig= should add order traces to existing figure."""
        existing_fig = make_figure()
        existing_fig.add_trace(go.Scatter(x=index_5, y=[100] * 5, name='Pre-existing'))
        result_fig = pf_simple.orders.plot(fig=existing_fig)
        assert result_fig is existing_fig
        assert len(result_fig.data) == 4  # Pre-existing + Close + Buy + Sell
        assert result_fig.data[0].name == 'Pre-existing'


class TestTradesPlot:
    def test_trades_plot_with_zones(self):
        """Trades.plot() should show Close, Entry, Exit markers and profit zones."""
        fig = pf_simple.trades.plot()
        assert isinstance(fig, BaseFigure)

        nt = named_traces(fig)
        assert 'Close' in nt
        assert 'Entry' in nt

        # pf_simple has 2 profitable closed trades, no losses, no active
        assert 'Exit - Profit' in nt
        assert 'Exit - Loss' not in nt
        assert 'Exit' not in nt  # no neutral trades
        assert 'Active' not in nt

        assert len(fig.data) == 3  # Close + Entry + Exit - Profit

        # 2 profit zone shapes (one per profitable trade)
        assert len(fig.layout.shapes) == 2
        for shape in fig.layout.shapes:
            assert shape.type == 'rect'

    def test_trades_plot_no_zones(self):
        """plot_zones=False should produce same traces but no shapes."""
        fig = pf_simple.trades.plot(plot_zones=False)
        assert isinstance(fig, BaseFigure)

        nt = named_traces(fig)
        assert 'Entry' in nt
        assert 'Exit - Profit' in nt
        assert len(fig.data) == 3  # Same trace count as with zones
        assert len(fig.layout.shapes) == 0

    def test_trades_plot_entry_exit_positions(self):
        """Entry/Exit trace positions should match trade records."""
        fig = pf_simple.trades.plot()
        nt = named_traces(fig)

        # Entry at idx 0 and idx 3, prices 10.0 and 10.0
        entry_x = list(nt['Entry'].x)
        assert entry_x[0] == index_5[0]
        assert entry_x[1] == index_5[3]
        np.testing.assert_array_equal(nt['Entry'].y, [10., 10.])

        # Exit - Profit at idx 2 and idx 4, prices 12.0 and 13.0
        exit_x = list(nt['Exit - Profit'].x)
        assert exit_x[0] == index_5[2]
        assert exit_x[1] == index_5[4]
        np.testing.assert_array_equal(nt['Exit - Profit'].y, [12., 13.])

    def test_trades_plot_adds_to_existing_fig(self):
        """Passing fig= should add trade traces to existing figure."""
        existing_fig = make_figure()
        existing_fig.add_trace(go.Scatter(x=index_5, y=[100] * 5, name='Pre-existing'))
        result_fig = pf_simple.trades.plot(fig=existing_fig)
        assert result_fig is existing_fig
        assert len(result_fig.data) == 4  # Pre-existing + Close + Entry + Exit - Profit
        assert result_fig.data[0].name == 'Pre-existing'


class TestTradesPlotPnl:
    def test_trades_plot_pnl_traces(self):
        """plot_pnl should show profit markers and zeroline shape."""
        fig = pf_simple.trades.plot_pnl()
        assert isinstance(fig, BaseFigure)

        nt = named_traces(fig)
        # Both trades are profitable, so only Closed - Profit
        assert 'Closed - Profit' in nt
        assert 'Closed - Loss' not in nt
        assert 'Open' not in nt

        assert len(fig.data) == 1

        # 1 shape: horizontal line at y=0
        assert len(fig.layout.shapes) == 1
        assert fig.layout.shapes[0].type == 'line'
        assert fig.layout.shapes[0].y0 == 0
        assert fig.layout.shapes[0].y1 == 0

    def test_trades_plot_pnl_pct_scale(self):
        """pct_scale=True should use returns, False should use raw PnL."""
        fig_pct = pf_simple.trades.plot_pnl(pct_scale=True)
        fig_abs = pf_simple.trades.plot_pnl(pct_scale=False)

        # pct_scale=True: y-values are fractional returns (PnL / entry_price)
        np.testing.assert_array_almost_equal(fig_pct.data[0].y, [0.178, 0.277])
        # pct_scale=False: y-values are absolute PnL
        np.testing.assert_array_almost_equal(fig_abs.data[0].y, [1.78, 2.77])


class TestDrawdownsPlot:
    def test_drawdowns_plot_with_zones(self):
        """Drawdowns.plot() should show TS, Peak, Valley, Recovery, Active traces + zones."""
        dd = dd_ts.vbt.get_drawdowns()
        fig = dd.plot(top_n=None)
        assert isinstance(fig, BaseFigure)

        nt = named_traces(fig)
        # Time series trace inherits Series name
        assert 'Price' in nt
        assert 'Peak' in nt
        # 1 recovered drawdown produces Valley and Recovery/Peak
        assert 'Valley' in nt
        assert 'Recovery/Peak' in nt
        # 1 active drawdown
        assert 'Active' in nt

        assert len(fig.data) == 5

        # Verify marker positions match drawdown structure
        # Recovered DD: peak at Jan 2 (12.0), valley at Jan 3 (8.0), recovery at Jan 5 (15.0)
        assert list(nt['Peak'].x) == [index_8[1]]
        np.testing.assert_array_equal(nt['Peak'].y, [12.0])
        assert list(nt['Valley'].x) == [index_8[2]]
        np.testing.assert_array_equal(nt['Valley'].y, [8.0])
        assert list(nt['Recovery/Peak'].x) == [index_8[4]]
        np.testing.assert_array_equal(nt['Recovery/Peak'].y, [15.0])
        # Active DD: last point at Jan 8 (9.0)
        assert list(nt['Active'].x) == [index_8[7]]
        np.testing.assert_array_equal(nt['Active'].y, [9.0])

        # 3 shapes: 1 decline zone + 1 recovery zone + 1 active zone
        assert len(fig.layout.shapes) == 3
        for shape in fig.layout.shapes:
            assert shape.type == 'rect'

    def test_drawdowns_top_n_filtering(self):
        """top_n=1 should show only the largest drawdown, reducing traces and shapes."""
        dd = dd_ts.vbt.get_drawdowns()
        fig = dd.plot(top_n=1)
        assert isinstance(fig, BaseFigure)

        nt = named_traces(fig)
        # Largest DD is the active one (peak 15 → 7, depth=8)
        assert 'Price' in nt
        assert 'Peak' in nt
        assert 'Active' in nt
        # Recovered DD (depth=4) is excluded
        assert 'Valley' not in nt
        assert 'Recovery/Peak' not in nt

        # Peak at Jan 5 (15.0), Active at Jan 8 (9.0)
        assert list(nt['Peak'].x) == [index_8[4]]
        np.testing.assert_array_equal(nt['Peak'].y, [15.0])
        assert list(nt['Active'].x) == [index_8[7]]
        np.testing.assert_array_equal(nt['Active'].y, [9.0])

        # Only 1 shape for the active drawdown zone
        assert len(fig.layout.shapes) == 1
        assert fig.layout.shapes[0].type == 'rect'

    def test_drawdowns_plot_no_zones(self):
        """plot_zones=False should produce same traces but no shapes."""
        dd = dd_ts.vbt.get_drawdowns()
        fig = dd.plot(top_n=None, plot_zones=False)
        assert isinstance(fig, BaseFigure)

        nt = named_traces(fig)
        assert 'Peak' in nt
        assert 'Valley' in nt
        assert 'Recovery/Peak' in nt
        assert 'Active' in nt

        assert len(fig.data) == 5
        assert len(fig.layout.shapes) == 0


class TestRangesPlot:
    def test_ranges_plot_with_zones(self):
        """Ranges.plot() should show TS, Start, Closed, Open traces + zones."""
        rng = range_mask.vbt.get_ranges()
        fig = rng.plot(top_n=None)
        assert isinstance(fig, BaseFigure)

        nt = named_traces(fig)
        assert 'Mask' in nt  # TS trace inherits Series name
        assert 'Start' in nt
        assert 'Closed' in nt
        assert 'Open' in nt

        assert len(fig.data) == 4

        # Verify marker positions: 3 ranges start at Jan 1, Jan 4, Jan 8
        assert list(nt['Start'].x) == [index_8[0], index_8[3], index_8[7]]
        # 2 closed ranges end at Jan 3 and Jan 7
        assert list(nt['Closed'].x) == [index_8[2], index_8[6]]
        # 1 open range: last point at Jan 8
        assert list(nt['Open'].x) == [index_8[7]]

        # 3 shapes: 2 closed zones + 1 open zone
        assert len(fig.layout.shapes) == 3
        for shape in fig.layout.shapes:
            assert shape.type == 'rect'

    def test_ranges_plot_no_zones(self):
        """plot_zones=False should produce same traces but no shapes."""
        rng = range_mask.vbt.get_ranges()
        fig = rng.plot(top_n=None, plot_zones=False)
        assert isinstance(fig, BaseFigure)

        nt = named_traces(fig)
        assert 'Start' in nt
        assert 'Closed' in nt
        assert 'Open' in nt

        assert len(fig.data) == 4
        assert len(fig.layout.shapes) == 0


# ############# Tier 3: Trace Wrapper Classes ############# #

class TestScatterWrapper:
    def test_scatter_creation(self):
        """Scatter wrapper should create 1 go.Scatter per column."""
        scatter = Scatter(
            data=df.values,
            trace_names=df.columns.tolist(),
            x_labels=df.index
        )
        assert isinstance(scatter, TraceUpdater)
        assert len(scatter.traces) == 2
        for trace in scatter.traces:
            assert isinstance(trace, go.Scatter)

        names = {t.name for t in scatter.traces}
        assert names == {'a', 'b'}

        for t in scatter.traces:
            np.testing.assert_array_equal(t.y, df[t.name].values)
            np.testing.assert_array_equal(t.x, df.index)

    def test_scatter_update(self):
        """Scatter.update() should change y values on all traces."""
        scatter = Scatter(
            data=df.values,
            trace_names=df.columns.tolist(),
            x_labels=df.index
        )
        new_data = df.values * 2
        scatter.update(new_data)
        for i, t in enumerate(scatter.traces):
            np.testing.assert_array_equal(t.y, new_data[:, i])

    def test_scatter_subplot_positioning(self):
        """add_trace_kwargs with row/col should position traces in correct subplot."""
        subfig = make_subplots(rows=2, cols=1)
        Scatter(
            data=df.values,
            trace_names=df.columns.tolist(),
            x_labels=df.index,
            add_trace_kwargs=dict(row=2, col=1),
            fig=subfig,
        )
        # Traces added to row 2 should be on xaxis2/yaxis2
        for t in subfig.data:
            assert t.xaxis == 'x2'
            assert t.yaxis == 'y2'


class TestBarWrapper:
    def test_bar_creation(self):
        """Bar wrapper should create 1 go.Bar per column."""
        bar = Bar(
            data=df.values,
            trace_names=df.columns.tolist(),
            x_labels=df.index
        )
        assert len(bar.traces) == 2
        for trace in bar.traces:
            assert isinstance(trace, go.Bar)
        assert {t.name for t in bar.traces} == {'a', 'b'}

        for t in bar.traces:
            np.testing.assert_array_equal(t.y, df[t.name].values)
            np.testing.assert_array_equal(t.x, df.index)

    def test_bar_update(self):
        """Bar.update() should change y values."""
        bar = Bar(
            data=df.values,
            trace_names=df.columns.tolist(),
            x_labels=df.index
        )
        new_data = df.values * 3
        bar.update(new_data)
        for i, t in enumerate(bar.traces):
            np.testing.assert_array_equal(t.y, new_data[:, i])

    def test_bar_subplot_positioning(self):
        """add_trace_kwargs with row/col should position Bar traces in correct subplot."""
        subfig = make_subplots(rows=2, cols=1)
        Bar(
            data=df.values,
            trace_names=df.columns.tolist(),
            x_labels=df.index,
            add_trace_kwargs=dict(row=2, col=1),
            fig=subfig,
        )
        for t in subfig.data:
            assert t.xaxis == 'x2'
            assert t.yaxis == 'y2'


class TestHistogramWrapper:
    def test_histogram_creation(self):
        """Histogram wrapper should create 1 go.Histogram per column with matching data."""
        hist = Histogram(
            data=df.values,
            trace_names=df.columns.tolist()
        )
        assert len(hist.traces) == 2
        for i, trace in enumerate(hist.traces):
            assert isinstance(trace, go.Histogram)
            np.testing.assert_array_equal(trace.x, df.values[:, i])

    def test_histogram_update(self):
        """Histogram.update() should change trace data."""
        hist = Histogram(
            data=df.values,
            trace_names=df.columns.tolist()
        )
        new_data = df.values * 2
        hist.update(new_data)
        # After update, x values should reflect new data
        for i, t in enumerate(hist.traces):
            np.testing.assert_array_equal(t.x, new_data[:, i])

    def test_histogram_horizontal(self):
        """horizontal=True should put data on y-axis instead of x-axis."""
        hist_h = Histogram(
            data=df.values,
            trace_names=df.columns.tolist(),
            horizontal=True,
        )
        for i, t in enumerate(hist_h.traces):
            assert t.x is None
            np.testing.assert_array_equal(t.y, df.values[:, i])

        # Compare with vertical (default)
        hist_v = Histogram(
            data=df.values,
            trace_names=df.columns.tolist(),
        )
        for i, t in enumerate(hist_v.traces):
            np.testing.assert_array_equal(t.x, df.values[:, i])
            assert t.y is None


class TestHeatmapWrapper:
    def test_heatmap_creation(self):
        """Heatmap wrapper should create 1 go.Heatmap trace."""
        x_labels = ['x1', 'x2', 'x3']
        y_labels = ['y1', 'y2']
        hm = Heatmap(
            data=heatmap_data,
            x_labels=x_labels,
            y_labels=y_labels
        )
        assert len(hm.traces) == 1
        trace = hm.traces[0]
        assert isinstance(trace, go.Heatmap)
        np.testing.assert_array_equal(trace.z, heatmap_data)
        assert list(trace.x) == x_labels
        assert list(trace.y) == y_labels

    def test_heatmap_update(self):
        """Heatmap.update() should change z values."""
        hm = Heatmap(
            data=heatmap_data,
            x_labels=['x1', 'x2', 'x3'],
            y_labels=['y1', 'y2']
        )
        new_data = heatmap_data * 10
        hm.update(new_data)
        np.testing.assert_array_equal(hm.traces[0].z, new_data)


class TestGaugeWrapper:
    def test_gauge_creation(self):
        """Gauge wrapper should create 1 go.Indicator trace."""
        gauge = Gauge(value=0.5, value_range=(0, 1))
        assert isinstance(gauge, TraceUpdater)
        assert len(gauge.traces) == 1
        assert isinstance(gauge.traces[0], go.Indicator)
        assert gauge.traces[0].value == 0.5


class TestBoxWrapper:
    def test_box_creation(self):
        """Box wrapper should create 1 go.Box per column."""
        box = Box(data=df.values, trace_names=df.columns.tolist())
        assert isinstance(box, TraceUpdater)
        assert len(box.traces) == 2
        for trace in box.traces:
            assert isinstance(trace, go.Box)
        names = {t.name for t in box.traces}
        assert names == {'a', 'b'}


class TestVolumeWrapper:
    def test_volume_creation(self):
        """Volume wrapper should create 1 go.Volume trace."""
        vol_data = np.ones((3, 3, 3)) * 0.5
        volume = Volume(
            data=vol_data,
            x_labels=np.arange(3),
            y_labels=np.arange(3),
            z_labels=np.arange(3),
        )
        assert isinstance(volume, TraceUpdater)
        assert len(volume.traces) == 1
        assert isinstance(volume.traces[0], go.Volume)


# ############# Tier 4: Portfolio.plots() Orchestrator ############# #

class TestPortfolioPlots:
    def test_plots_returns_basefigure(self):
        """Portfolio.plot() with column selection should return BaseFigure."""
        fig = pf_multi.plot(column='a')
        assert isinstance(fig, BaseFigure)

    def test_plots_subplots_all(self):
        """subplots='all' should produce all subplots with expected structure."""
        fig = pf_multi.plot(column='a', subplots='all')
        assert isinstance(fig, BaseFigure)
        # Lower bounds: 14 subplots produce ~49 traces (named + unnamed fill helpers)
        # and ~15 shapes (hlines + zones). Using >= to tolerate internal helper changes.
        assert len(fig.data) >= 40
        assert len(fig.layout.shapes) >= 10

        nt = named_traces(fig)
        expected_subset = {
            'Buy', 'Sell', 'Close', 'Entry', 'Exit - Profit',
            'Closed - Profit', 'Assets', 'Cash', 'Asset Value',
            'Value', 'Benchmark', 'Peak', 'Valley', 'Recovery/Peak',
            'Drawdown', 'Exposure',
        }
        assert expected_subset.issubset(set(nt.keys()))

    def test_plots_single_subplot_filter(self):
        """subplots=['orders'] should only produce order-related traces."""
        fig = pf_multi.plot(column='a', subplots=['orders'])
        assert isinstance(fig, BaseFigure)

        nt = named_traces(fig)
        # Orders subplot produces Close + Buy + Sell
        expected_names = {'Close', 'Buy', 'Sell'}
        assert set(nt.keys()) == expected_names

    def test_plots_tag_filter(self):
        """tags='orders' should filter to order-tagged subplots only."""
        fig = pf_multi.plot(column='a', tags='orders')
        assert isinstance(fig, BaseFigure)

        nt = named_traces(fig)
        expected_names = {'Close', 'Buy', 'Sell'}
        assert set(nt.keys()) == expected_names

    def test_plots_grouped_warns_for_not_grouped_subplots(self):
        """Grouped portfolio should warn for subplots that don't support grouping."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fig = pf_grouped.plot(column='first', subplots='all')

        # Subplots with check_is_not_grouped should each produce a warning
        not_grouped_warnings = [
            wi for wi in w
            if 'does not support grouped data' in str(wi.message)
        ]
        assert len(not_grouped_warnings) >= 5
        # Verify the known subplots are warned about
        warned_names = {str(wi.message) for wi in not_grouped_warnings}
        for name in ('orders', 'trades', 'trade_pnl', 'asset_flow', 'assets'):
            assert any(name in msg for msg in warned_names)

        # Remaining subplots should still render
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) > 0

    def test_plots_raises_without_column(self):
        """Plotting multi-column portfolio without column selection should raise."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(TypeError, match='Only one column is allowed'):
                pf_multi.plot(subplots='all')
            with pytest.raises(TypeError, match='Only one group is allowed'):
                pf_grouped.plot(subplots='all')

    def test_plots_shared_portfolio(self):
        """Shared portfolio with group_by=False should produce same structure as ungrouped."""
        fig = pf_shared.plot(column='a', subplots='all', group_by=False)
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) >= 40
        assert len(fig.layout.shapes) >= 10

        nt = named_traces(fig)
        expected_subset = {
            'Buy', 'Sell', 'Close', 'Entry', 'Exit - Profit',
            'Closed - Profit', 'Assets', 'Cash', 'Asset Value',
            'Value', 'Benchmark', 'Peak', 'Valley', 'Recovery/Peak',
            'Drawdown', 'Exposure',
        }
        assert expected_subset.issubset(set(nt.keys()))

    def test_plots_multi_subplot_combination(self):
        """Requesting multiple specific subplots should produce traces from each."""
        fig = pf_multi.plot(column='a', subplots=['orders', 'trades'])
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) == 6
        assert len(fig.layout.shapes) == 2

        nt = traces_by_name(fig)
        # Orders subplot contributes Close + Buy + Sell
        assert 'Buy' in nt
        assert 'Sell' in nt
        # Trades subplot contributes Close + Entry + Exit - Profit
        assert 'Entry' in nt
        assert 'Exit - Profit' in nt
        # Both have 'Close' traces (one each)
        assert len(nt['Close']) == 2

    def test_plots_multi_subplot_positioning(self):
        """Multiple subplots should place traces in separate rows."""
        fig = pf_multi.plot(column='a', subplots=['orders', 'trades'])

        # Orders traces should be in row 1 (yaxis 'y')
        order_traces = [t for t in fig.data if t.name in ('Buy', 'Sell')]
        for t in order_traces:
            assert t.yaxis == 'y'

        # Trades traces should be in row 2 (yaxis 'y2')
        trade_traces = [t for t in fig.data if t.name in ('Entry', 'Exit - Profit')]
        for t in trade_traces:
            assert t.yaxis == 'y2'


# ############# Tier 5: Portfolio plot_* Smoke Tests ############# #

class TestPortfolioPlotSmoke:
    """Regression tests for each Portfolio plot_* method.

    Asserts return type, trace count, shape count, and key named traces
    to catch regressions from refactors.
    """

    def _assert_fig(self, fig, min_traces, min_shapes, expected_names):
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) >= min_traces
        assert len(fig.layout.shapes) >= min_shapes
        nt = named_traces(fig)
        assert set(expected_names).issubset(set(nt.keys()))

    def test_smoke_plot_orders(self):
        self._assert_fig(
            pf_multi.plot_orders(column='a'),
            min_traces=3, min_shapes=0, expected_names={'Close', 'Buy', 'Sell'},
        )

    def test_smoke_plot_trades(self):
        self._assert_fig(
            pf_multi.plot_trades(column='a'),
            min_traces=3, min_shapes=2, expected_names={'Close', 'Entry', 'Exit - Profit'},
        )

    def test_smoke_plot_trade_pnl(self):
        self._assert_fig(
            pf_multi.plot_trade_pnl(column='a'),
            min_traces=1, min_shapes=1, expected_names={'Closed - Profit'},
        )

    def test_smoke_plot_positions(self):
        self._assert_fig(
            pf_multi.plot_positions(column='a'),
            min_traces=3, min_shapes=2, expected_names={'Close', 'Entry', 'Exit - Profit'},
        )

    def test_smoke_plot_position_pnl(self):
        self._assert_fig(
            pf_multi.plot_position_pnl(column='a'),
            min_traces=1, min_shapes=1, expected_names={'Closed - Profit'},
        )

    def test_smoke_plot_asset_flow(self):
        self._assert_fig(
            pf_multi.plot_asset_flow(column='a'),
            min_traces=1, min_shapes=1, expected_names={'Assets'},
        )

    def test_smoke_plot_cash_flow(self):
        self._assert_fig(
            pf_multi.plot_cash_flow(column='a'),
            min_traces=1, min_shapes=1, expected_names={'Cash'},
        )
        self._assert_fig(
            pf_grouped.plot_cash_flow(column='first'),
            min_traces=1, min_shapes=1, expected_names={'Cash'},
        )

    def test_smoke_plot_assets(self):
        self._assert_fig(
            pf_multi.plot_assets(column='a'),
            min_traces=4, min_shapes=1, expected_names={'Assets'},
        )

    def test_smoke_plot_cash(self):
        self._assert_fig(
            pf_multi.plot_cash(column='a'),
            min_traces=6, min_shapes=1, expected_names={'Cash'},
        )
        self._assert_fig(
            pf_grouped.plot_cash(column='first'),
            min_traces=6, min_shapes=1, expected_names={'Cash'},
        )

    def test_smoke_plot_asset_value(self):
        self._assert_fig(
            pf_multi.plot_asset_value(column='a'),
            min_traces=4, min_shapes=1, expected_names={'Asset Value'},
        )
        self._assert_fig(
            pf_grouped.plot_asset_value(column='first'),
            min_traces=4, min_shapes=1, expected_names={'Asset Value'},
        )

    def test_smoke_plot_value(self):
        self._assert_fig(
            pf_multi.plot_value(column='a'),
            min_traces=6, min_shapes=1, expected_names={'Value'},
        )
        self._assert_fig(
            pf_grouped.plot_value(column='first'),
            min_traces=6, min_shapes=1, expected_names={'Value'},
        )

    def test_smoke_plot_cum_returns(self):
        self._assert_fig(
            pf_multi.plot_cum_returns(column='a'),
            min_traces=7, min_shapes=1, expected_names={'Value', 'Benchmark'},
        )
        self._assert_fig(
            pf_grouped.plot_cum_returns(column='first'),
            min_traces=7, min_shapes=1, expected_names={'Value', 'Benchmark'},
        )

    def test_cum_returns_benchmark_trace(self):
        """Benchmark trace should be a Scatter with data matching cumulative benchmark returns."""
        fig = pf_multi.plot_cum_returns(column='a')
        nt = named_traces(fig)
        assert 'Benchmark' in nt
        assert isinstance(nt['Benchmark'], go.Scatter)
        # Benchmark should have same length as portfolio value
        assert len(nt['Benchmark'].y) == len(index_5)
        assert 'Value' in nt
        assert isinstance(nt['Value'], go.Scatter)
        assert len(nt['Value'].y) == len(index_5)

    def test_smoke_plot_drawdowns(self):
        self._assert_fig(
            pf_multi.plot_drawdowns(column='a'),
            min_traces=4, min_shapes=2,
            expected_names={'Value', 'Peak', 'Valley', 'Recovery/Peak'},
        )
        self._assert_fig(
            pf_grouped.plot_drawdowns(column='first'),
            min_traces=4, min_shapes=2,
            expected_names={'Value', 'Peak', 'Valley', 'Recovery/Peak'},
        )

    def test_smoke_plot_underwater(self):
        self._assert_fig(
            pf_multi.plot_underwater(column='a'),
            min_traces=1, min_shapes=1, expected_names={'Drawdown'},
        )
        self._assert_fig(
            pf_grouped.plot_underwater(column='first'),
            min_traces=1, min_shapes=1, expected_names={'Drawdown'},
        )

    def test_smoke_plot_gross_exposure(self):
        self._assert_fig(
            pf_multi.plot_gross_exposure(column='a'),
            min_traces=4, min_shapes=1, expected_names={'Exposure'},
        )
        self._assert_fig(
            pf_grouped.plot_gross_exposure(column='first'),
            min_traces=4, min_shapes=1, expected_names={'Exposure'},
        )

    def test_smoke_plot_net_exposure(self):
        self._assert_fig(
            pf_multi.plot_net_exposure(column='a'),
            min_traces=4, min_shapes=1, expected_names={'Exposure'},
        )
        self._assert_fig(
            pf_grouped.plot_net_exposure(column='first'),
            min_traces=4, min_shapes=1, expected_names={'Exposure'},
        )


# ############# Tier 6: Indicator Plot Methods ############# #

class TestIndicatorPlots:
    """Regression tests for built-in indicator plot methods.

    These create subplots, add traces, and add shapes — all directly
    affected by the migration to protocol methods.
    """

    def test_ma_plot(self):
        """MA.plot() should show Close line + MA line."""
        ma = vbt.MA.run(ind_close, window=5)
        fig = ma.plot()
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) == 2
        nt = named_traces(fig)
        assert 'Close' in nt
        assert 'MA' in nt
        for t in fig.data:
            assert isinstance(t, go.Scatter)

    def test_rsi_plot(self):
        """RSI.plot() should show RSI line + band shape."""
        rsi = vbt.RSI.run(ind_close, window=14)
        fig = rsi.plot()
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) == 1
        assert fig.data[0].name == 'RSI'
        assert isinstance(fig.data[0], go.Scatter)
        # RSI adds a rectangular band shape (overbought/oversold zone)
        assert len(fig.layout.shapes) == 1
        assert fig.layout.shapes[0].type == 'rect'

    def test_bbands_plot(self):
        """BBANDS.plot() should show Close + 3 band lines."""
        bb = vbt.BBANDS.run(ind_close, window=5)
        fig = bb.plot()
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) == 4
        nt = named_traces(fig)
        assert 'Close' in nt
        assert 'Upper Band' in nt
        assert 'Middle Band' in nt
        assert 'Lower Band' in nt
        for t in fig.data:
            assert isinstance(t, go.Scatter)

    def test_macd_plot(self):
        """MACD.plot() should show MACD + Signal lines + Histogram bar."""
        macd = vbt.MACD.run(ind_close, fast_window=5, slow_window=10, signal_window=3)
        fig = macd.plot()
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) == 3
        nt = named_traces(fig)
        assert 'MACD' in nt
        assert 'Signal' in nt
        assert 'Histogram' in nt
        assert isinstance(nt['MACD'], go.Scatter)
        assert isinstance(nt['Signal'], go.Scatter)
        assert isinstance(nt['Histogram'], go.Bar)

    def test_stoch_plot(self):
        """STOCH.plot() should show %K + %D lines + band shape."""
        stoch = vbt.STOCH.run(ind_high, ind_low, ind_close, k_window=5, d_window=3)
        fig = stoch.plot()
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) == 2
        nt = named_traces(fig)
        assert '%K' in nt
        assert '%D' in nt
        for t in fig.data:
            assert isinstance(t, go.Scatter)
        assert len(fig.layout.shapes) == 1
        assert fig.layout.shapes[0].type == 'rect'

    def test_atr_plot(self):
        """ATR.plot() should show TR + ATR lines."""
        atr = vbt.ATR.run(ind_high, ind_low, ind_close, window=5)
        fig = atr.plot()
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) == 2
        nt = named_traces(fig)
        assert 'TR' in nt
        assert 'ATR' in nt
        for t in fig.data:
            assert isinstance(t, go.Scatter)

    def test_obv_plot(self):
        """OBV.plot() should show OBV line."""
        volume = ind_close * np.linspace(0.5, 1.5, 20)
        obv = vbt.OBV.run(ind_close, volume)
        fig = obv.plot()
        assert isinstance(fig, BaseFigure)
        assert len(fig.data) == 1
        assert fig.data[0].name == 'OBV'
        assert isinstance(fig.data[0], go.Scatter)


# ############# Tier 7: Signals Accessor Plots ############# #

class TestSignalsPlot:
    """Regression tests for signals accessor plot methods.

    These overlay markers on existing figures via fig= parameter —
    directly affected by the records plot migration.
    """

    def test_plot_as_entry_markers(self):
        """plot_as_entry_markers should add triangle-up markers to existing figure."""
        fig = ind_close.vbt.plot()
        sig_entries.vbt.signals.plot_as_entry_markers(ind_close, fig=fig)
        assert isinstance(fig, BaseFigure)

        nt = named_traces(fig)
        assert 'Entry' in nt
        assert isinstance(nt['Entry'], go.Scatter)
        assert nt['Entry'].marker.symbol == 'triangle-up'
        # Entry markers should be at positions where sig_entries is True
        assert len(nt['Entry'].x) == sig_entries.sum()

    def test_plot_as_exit_markers(self):
        """plot_as_exit_markers should add triangle-down markers to existing figure."""
        fig = ind_close.vbt.plot()
        sig_exits.vbt.signals.plot_as_exit_markers(ind_close, fig=fig)
        assert isinstance(fig, BaseFigure)

        nt = named_traces(fig)
        assert 'Exit' in nt
        assert isinstance(nt['Exit'], go.Scatter)
        assert nt['Exit'].marker.symbol == 'triangle-down'
        assert len(nt['Exit'].x) == sig_exits.sum()

    def test_signals_compose_on_existing_fig(self):
        """Entry + Exit markers should compose correctly on same figure."""
        fig = ind_close.vbt.plot()
        initial_count = len(fig.data)
        sig_entries.vbt.signals.plot_as_entry_markers(ind_close, fig=fig)
        sig_exits.vbt.signals.plot_as_exit_markers(ind_close, fig=fig)
        # Should have original trace + Entry + Exit
        assert len(fig.data) == initial_count + 2
        names = {t.name for t in fig.data}
        assert {'Close', 'Entry', 'Exit'}.issubset(names)
