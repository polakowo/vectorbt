"""Tests for the plotting backend registry, `create_figure` factory,
and the `default_backend` setting.

These tests cover:
- registry round-trip, validation, and override semantics
- the `_plotly_factory` router (no-args -> make_figure;
  explicit rows/cols or subplot-only kwargs -> make_subplots)
- runtime mutation of `settings['plotting']['default_backend']`
- public API exposure on `vbt.plotting.*` and top-level `vbt.*`
"""

import pytest

import vectorbt as vbt
from vectorbt.utils import figure as _figure_mod
from vectorbt.utils.figure import (
    Figure,
    FigureWidget,
    assert_plotly_only_kwargs,
    assert_plotly_only_method,
    create_figure,
    get_backend,
    list_backends,
    make_figure,
    make_subplots,
    register_backend,
    resolve_backend,
    resolve_backend_for_fig,
)
from vectorbt.utils.plotting_protocol import Capability, FigureProtocol


class _StubNonPlotlyFig:
    """Minimal non-Plotly figure for testing resolution helpers."""
    backend_name = 'stub'


def _strip_uid(node):
    if isinstance(node, dict):
        return {k: _strip_uid(v) for k, v in node.items() if k != 'uid'}
    if isinstance(node, list):
        return [_strip_uid(x) for x in node]
    return node


@pytest.fixture(autouse=True)
def _restore_backend_state():
    saved_registry = dict(_figure_mod._BACKEND_REGISTRY)
    saved_default = vbt.settings['plotting']['default_backend']
    saved_use_widgets = vbt.settings['plotting']['use_widgets']
    try:
        yield
    finally:
        _figure_mod._BACKEND_REGISTRY.clear()
        _figure_mod._BACKEND_REGISTRY.update(saved_registry)
        vbt.settings['plotting']['default_backend'] = saved_default
        vbt.settings['plotting']['use_widgets'] = saved_use_widgets


class TestRegistryDefaults:
    def test_default_backend_is_plotly(self):
        assert 'plotly' in list_backends()
        assert vbt.settings['plotting']['default_backend'] == 'plotly'


class TestBackendNameAttribute:
    def test_plotly_backend_name(self):
        from vectorbt.utils.figure import PlotlyFigureProtocolMixin
        assert PlotlyFigureProtocolMixin.backend_name == 'plotly'
        fig = make_figure()
        assert type(fig).backend_name == 'plotly'


class TestResolveBackendForFig:
    def test_none_defaults_to_plotly(self):
        vbt.settings['plotting']['default_backend'] = 'plotly'
        backend, is_plotly = resolve_backend_for_fig(None)
        assert backend == 'plotly'
        assert is_plotly is True

    def test_none_defaults_to_custom(self):
        vbt.settings['plotting']['default_backend'] = 'other'
        backend, is_plotly = resolve_backend_for_fig(None)
        assert backend == 'other'
        assert is_plotly is False

    def test_plotly_figure_instance(self):
        fig = make_figure()
        backend, is_plotly = resolve_backend_for_fig(fig)
        assert backend == 'plotly'
        assert is_plotly is True

    def test_plotly_subplots_instance(self):
        fig = make_subplots(rows=2, cols=1)
        backend, is_plotly = resolve_backend_for_fig(fig)
        assert backend == 'plotly'
        assert is_plotly is True

    def test_stub_figure_instance(self):
        fig = _StubNonPlotlyFig()
        backend, is_plotly = resolve_backend_for_fig(fig)
        assert backend == 'stub'
        assert is_plotly is False

    def test_third_party_backend_uses_class_attribute(self):
        class FakeFig:
            backend_name = 'bokeh'
        backend, is_plotly = resolve_backend_for_fig(FakeFig())
        assert backend == 'bokeh'
        assert is_plotly is False

    def test_third_party_backend_falls_back_to_class_name(self):
        class UnnamedFig:
            pass
        backend, is_plotly = resolve_backend_for_fig(UnnamedFig())
        assert backend == 'UnnamedFig'
        assert is_plotly is False


class TestAssertPlotlyOnlyKwargs:
    def test_noop_when_forcing_empty(self):
        # No raise even on non-Plotly backend.
        assert_plotly_only_kwargs(False, 'other', [], method_name='Foo.plot')

    def test_noop_when_plotly(self):
        # No raise even with forcing kwargs.
        assert_plotly_only_kwargs(True, 'plotly', ['trace_kwargs'], method_name='Foo.plot')

    def test_raises_on_non_plotly_with_forcing(self):
        with pytest.raises(NotImplementedError) as exc:
            assert_plotly_only_kwargs(
                False, 'other', ['trace_kwargs', 'layout_kwargs'],
                method_name='GenericAccessor.plot',
            )
        msg = str(exc.value)
        assert 'GenericAccessor.plot' in msg
        assert "'other'" in msg
        assert 'trace_kwargs' in msg
        assert 'layout_kwargs' in msg
        assert 'legacy Plotly-specific escape hatches' in msg

    def test_raises_mentions_third_party_backend_name(self):
        with pytest.raises(NotImplementedError) as exc:
            assert_plotly_only_kwargs(
                False, 'bokeh', ['trace_kwargs'],
                method_name='X.plot',
            )
        assert "'bokeh'" in str(exc.value)


class TestAssertPlotlyOnlyMethod:
    def test_noop_when_plotly(self):
        assert_plotly_only_method(
            True, 'plotly',
            method_name='Foo.bar', reason='anything',
        )

    def test_raises_on_non_plotly(self):
        with pytest.raises(NotImplementedError) as exc:
            assert_plotly_only_method(
                False, 'other',
                method_name='GenericSRAccessor.plot_against',
                reason='its fill-between-lines semantics have no portable equivalent.',
            )
        msg = str(exc.value)
        assert 'permanently Plotly-only' in msg
        assert 'plot_against' in msg
        assert 'fill-between-lines' in msg
        assert "'other'" in msg


class TestCreateFigureRouting:
    @pytest.mark.parametrize('use_widgets', [True, False])
    def test_create_figure_no_args_matches_make_figure(self, use_widgets):
        vbt.settings['plotting']['use_widgets'] = use_widgets
        a = create_figure()
        b = make_figure()
        expected_cls = FigureWidget if use_widgets else Figure
        assert isinstance(a, expected_cls)
        assert isinstance(b, expected_cls)
        if not use_widgets:
            assert not isinstance(a, FigureWidget)
            assert not isinstance(b, FigureWidget)
        assert _strip_uid(a.to_plotly_json()) == _strip_uid(b.to_plotly_json())

    def test_create_figure_rows_1_cols_1_routes_to_subplots(self):
        fig = create_figure(rows=1, cols=1)
        assert 'xaxis' in fig.layout
        assert 'yaxis' in fig.layout
        assert fig.get_subplot(1, 1) is not None
        ref = make_subplots(rows=1, cols=1)
        assert _strip_uid(fig.to_plotly_json()) == _strip_uid(ref.to_plotly_json())

    def test_create_figure_rows_cols_matches_make_subplots(self):
        fig = create_figure(rows=2, cols=1, shared_xaxes=True)
        ref = make_subplots(rows=2, cols=1, shared_xaxes=True)
        assert _strip_uid(fig.to_plotly_json()) == _strip_uid(ref.to_plotly_json())

    def test_create_figure_specs_routes_to_make_subplots(self):
        fig = create_figure(specs=[[{'secondary_y': True}]])
        assert 'yaxis2' in fig.layout

    def test_create_figure_rows_only_fills_cols_to_1(self):
        """rows=2 without cols must route to make_subplots(rows=2, cols=1)."""
        fig = create_figure(rows=2)
        ref = make_subplots(rows=2, cols=1)
        assert _strip_uid(fig.to_plotly_json()) == _strip_uid(ref.to_plotly_json())

    def test_create_figure_cols_only_fills_rows_to_1(self):
        """cols=2 without rows must route to make_subplots(rows=1, cols=2)."""
        fig = create_figure(cols=2)
        ref = make_subplots(rows=1, cols=2)
        assert _strip_uid(fig.to_plotly_json()) == _strip_uid(ref.to_plotly_json())

    def test_create_figure_figure_kwarg_routes_to_subplots(self):
        """figure= is in _SUBPLOT_ONLY_KWARGS, so create_figure(figure=fig) must
        route through make_subplots with Plotly's populate-existing semantics:
        pre-existing traces and layout on the base figure are preserved."""
        import plotly.graph_objects as go
        base = make_figure()
        base.add_trace(go.Scatter(x=[1, 2], y=[3, 4], name="existing"))
        base.update_layout(title_text="preserved")
        fig = create_figure(figure=base)
        # Subplot metadata was injected.
        assert fig.get_subplot(1, 1) is not None
        # Pre-existing trace survived.
        assert len(fig.data) == 1
        assert fig.data[0].name == "existing"
        # Pre-existing layout survived.
        assert fig.layout.title.text == "preserved"

    def test_create_figure_rejects_positional_args(self):
        with pytest.raises(TypeError):
            create_figure(object())  # type: ignore[misc]


class TestRegistryAPI:
    def test_register_backend_roundtrip(self):
        called = []
        sentinel = object()

        def _dummy(*, rows, cols, **kw):
            called.append((rows, cols, kw))
            return sentinel

        register_backend('dummy', _dummy)
        assert 'dummy' in list_backends()
        assert create_figure(backend='dummy') is sentinel
        assert called == [(None, None, {})]

    def test_register_backend_forwards_kwargs(self):
        """create_figure must forward rows, cols, and extra kwargs to the factory."""
        called = []

        def _dummy(*, rows, cols, **kw):
            called.append((rows, cols, kw))
            return 'dummy-figure'

        register_backend('dummy', _dummy)
        create_figure(backend='dummy', rows=3, cols=2, shared_xaxes=True)
        assert called == [(3, 2, {'shared_xaxes': True})]

    def test_register_backend_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            register_backend('plotly', lambda **kw: None)

    def test_register_backend_override_allowed(self):
        def replacement(**kw):
            return 'replaced'
        register_backend('plotly', replacement, override=True)
        assert get_backend('plotly') is replacement

    def test_unknown_backend_raises_keyerror(self):
        with pytest.raises(KeyError, match="nope"):
            create_figure(backend='nope')

    def test_unregistered_default_backend_raises_keyerror(self):
        """When the default_backend setting points to an unknown name,
        create_figure() without explicit backend= must raise KeyError."""
        vbt.settings['plotting']['default_backend'] = 'nonexistent'
        with pytest.raises(KeyError, match="nonexistent"):
            create_figure()

    def test_settings_override_default_backend(self):
        called = []

        def _dummy(*, rows, cols, **kw):
            called.append((rows, cols, kw))
            return 'dummy-figure'

        register_backend('dummy', _dummy)
        vbt.settings['plotting']['default_backend'] = 'dummy'
        result = create_figure()
        assert result == 'dummy-figure'
        assert called == [(None, None, {})]

    def test_register_backend_validation(self):
        with pytest.raises(ValueError):
            register_backend('', lambda **kw: None)
        with pytest.raises(TypeError):
            register_backend('bad', 'not-callable')


class TestPublicAPIExposure:
    def test_create_figure_exposed_via_vbt_plotting(self):
        assert vbt.plotting.create_figure is create_figure
        assert vbt.plotting.register_backend is register_backend
        assert vbt.plotting.get_backend is get_backend
        assert vbt.plotting.list_backends is list_backends
        assert vbt.plotting.FigureProtocol is FigureProtocol
        assert vbt.plotting.Capability is Capability
        fig = vbt.plotting.create_figure()
        assert isinstance(fig, (Figure, FigureWidget))

    def test_create_figure_exposed_at_top_level(self):
        assert vbt.create_figure is create_figure
        assert vbt.register_backend is register_backend
        assert vbt.get_backend is get_backend
        assert vbt.list_backends is list_backends
        a = vbt.create_figure()
        b = vbt.make_figure()
        assert isinstance(a, (Figure, FigureWidget))
        assert isinstance(b, (Figure, FigureWidget))
        assert _strip_uid(a.to_plotly_json()) == _strip_uid(b.to_plotly_json())

    def test_vbt_plotting_existing_classes_unaffected(self):
        assert hasattr(vbt.plotting, 'Gauge')
        assert hasattr(vbt.plotting, 'Scatter')
        assert hasattr(vbt.plotting, 'Heatmap')
        assert hasattr(vbt.plotting, 'Histogram')
        assert hasattr(vbt.plotting, 'Bar')
        assert hasattr(vbt.plotting, 'Box')
        assert hasattr(vbt.plotting, 'Volume')


class TestResolveBackend:
    """Tests for the resolve_backend() helper."""

    def test_both_none_uses_default_plotly(self):
        vbt.settings['plotting']['default_backend'] = 'plotly'
        backend, is_plotly = resolve_backend(None, None)
        assert backend == 'plotly'
        assert is_plotly is True

    def test_both_none_uses_default_custom(self):
        vbt.settings['plotting']['default_backend'] = 'other'
        backend, is_plotly = resolve_backend(None, None)
        assert backend == 'other'
        assert is_plotly is False

    def test_backend_override_alone(self):
        backend, is_plotly = resolve_backend(None, 'other')
        assert backend == 'other'
        assert is_plotly is False

    def test_backend_override_plotly(self):
        vbt.settings['plotting']['default_backend'] = 'other'
        backend, is_plotly = resolve_backend(None, 'plotly')
        assert backend == 'plotly'
        assert is_plotly is True

    def test_fig_alone_plotly(self):
        fig = make_figure()
        backend, is_plotly = resolve_backend(fig, None)
        assert backend == 'plotly'
        assert is_plotly is True

    def test_fig_alone_stub(self):
        fig = _StubNonPlotlyFig()
        backend, is_plotly = resolve_backend(fig, None)
        assert backend == 'stub'
        assert is_plotly is False

    def test_fig_and_matching_backend(self):
        fig = make_figure()
        backend, is_plotly = resolve_backend(fig, 'plotly')
        assert backend == 'plotly'
        assert is_plotly is True

    def test_fig_and_conflicting_backend_raises(self):
        fig = make_figure()
        with pytest.raises(ValueError, match="conflicts"):
            resolve_backend(fig, 'other')

    def test_stub_fig_and_conflicting_backend_raises(self):
        fig = _StubNonPlotlyFig()
        with pytest.raises(ValueError, match="conflicts"):
            resolve_backend(fig, 'plotly')
