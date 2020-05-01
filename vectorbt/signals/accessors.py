import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt.accessors import register_dataframe_accessor, register_series_accessor
from vectorbt.utils import checks, reshape_fns, index_fns
from vectorbt.utils.common import add_safe_nb_methods, cached_property
from vectorbt.utils.accessors import Base_DFAccessor, Base_SRAccessor
from vectorbt.signals import nb
from vectorbt.widgets import DefaultFigureWidget


@add_safe_nb_methods(
    nb.shuffle_nb,
    nb.fshift_nb)
class Signals_Accessor():
    dtype = np.bool

    @classmethod
    def _validate(cls, obj):
        if cls.dtype is not None:
            checks.assert_dtype(obj, cls.dtype)

    def random_exits(self, n, every_nth=1, seed=None):
        return self.wrap_array(nb.random_exits_nb(self.to_2d_array(), reshape_fns.to_1d(n), every_nth, seed))

    def random_exits_by_func(self, choice_func_nb, *args, seed=None):
        return self.wrap_array(nb.random_exits_by_func_nb(self.to_2d_array(), choice_func_nb, seed, *args))

    def exits(self, exit_mask_nb, *args, only_first=True):
        return self.wrap_array(nb.exits_nb(self.to_2d_array(), exit_mask_nb, only_first, *args))

    def stop_loss_exits(self, ts, stops, relative=True, only_first=True, trailing=False, as_columns=None, broadcast_kwargs={}):
        entries = self._obj
        checks.assert_type(ts, (pd.Series, pd.DataFrame))
        ts.vbt.timeseries.validate()

        entries, ts = reshape_fns.broadcast(entries, ts, **broadcast_kwargs, writeable=True)
        stops = reshape_fns.broadcast_to_array_of(stops, entries.vbt.to_2d_array())
        stops = stops.astype(np.float64)

        exits = nb.stop_loss_exits_nb(
            entries.vbt.to_2d_array(),
            ts.vbt.to_2d_array(),
            stops, relative, only_first, trailing)

        # Build column hierarchy
        if as_columns is not None:
            param_columns = as_columns
        else:
            name = 'trail_stop' if trailing else 'stop_loss'
            param_columns = index_fns.from_values(stops, name=name)
        columns = index_fns.combine(param_columns, reshape_fns.to_2d(entries).columns)
        return entries.vbt.wrap_array(exits, columns=columns)

    def AND(self, *others):
        # you can also do A & B, but then you need to have the same index and columns
        return self.combine_with_multiple(others, combine_func=np.logical_and)

    def OR(self, *others):
        return self.combine_with_multiple(others, combine_func=np.logical_or)

    def XOR(self, *others):
        return self.combine_with_multiple(others, combine_func=np.logical_xor)

    @cached_property
    def num_signals(self):
        return self._obj.sum(axis=0)

    @cached_property
    def avg_distance(self):
        sr = pd.Series(nb.avg_distance_nb(self.to_2d_array()), index=reshape_fns.to_2d(self._obj).columns)
        if isinstance(self._obj, pd.Series):
            return sr.iloc[0]
        return sr

    def avg_distance_to(self, other, **kwargs):
        return self.map_reduce_between(other=other, map_func_nb=nb.diff_map_nb, reduce_func_nb=nb.avg_reduce_nb, **kwargs)

    def map_reduce_between(self, *args, other=None, map_func_nb=None, reduce_func_nb=None, broadcast_kwargs={}):
        checks.assert_not_none(map_func_nb)
        checks.assert_not_none(reduce_func_nb)
        if other is None:
            result = nb.map_reduce_between_one_nb(self.to_2d_array(), map_func_nb, reduce_func_nb, *args)
            if isinstance(self._obj, pd.Series):
                return result[0]
            return pd.Series(result, index=reshape_fns.to_2d(self._obj).columns)
        else:
            obj, other = reshape_fns.broadcast(self._obj, other, **broadcast_kwargs)
            other.vbt.signals.validate()
            result = nb.map_reduce_between_two_nb(
                self.to_2d_array(), other.vbt.to_2d_array(), map_func_nb, reduce_func_nb, *args)
            if isinstance(obj, pd.Series):
                return result[0]
            return pd.Series(result, index=reshape_fns.to_2d(obj).columns)

    def rank(self, reset_signals=None, after_false=False, allow_gaps=False, broadcast_kwargs={}):
        if reset_signals is not None:
            obj, reset_signals = reshape_fns.broadcast(self._obj, reset_signals, **broadcast_kwargs)
            reset_signals = reset_signals.vbt.to_2d_array()
        else:
            obj = self._obj
        ranked = nb.rank_nb(
            obj.vbt.to_2d_array(),
            reset_b=reset_signals,
            after_false=after_false,
            allow_gaps=allow_gaps)
        return obj.vbt.wrap_array(ranked)

    def first(self, **kwargs):
        return self.wrap_array(self.rank(**kwargs).values == 1)

    def nst(self, n, **kwargs):
        return self.wrap_array(self.rank(**kwargs).values == n)

    def from_nst(self, n, **kwargs):
        return self.wrap_array(self.rank(**kwargs).values >= n)


@register_dataframe_accessor('signals')
class Signals_DFAccessor(Signals_Accessor, Base_DFAccessor):

    @classmethod
    def empty(cls, *args, fill_value=False, **kwargs):
        return Base_DFAccessor.empty(*args, fill_value=fill_value, **kwargs)

    @classmethod
    def empty_like(cls, *args, fill_value=False, **kwargs):
        return Base_DFAccessor.empty_like(*args, fill_value=fill_value, **kwargs)

    @classmethod
    def random(cls, shape, n, every_nth=1, seed=None, multiple=False, name='random_n', **kwargs):
        return pd.DataFrame(nb.random_nb(shape, reshape_fns.to_1d(n), every_nth, seed), **kwargs)

    @classmethod
    def random_by_func(cls, shape, choice_func_nb, *args, seed=None, **kwargs):
        return pd.DataFrame(nb.random_by_func_nb(shape, choice_func_nb, seed, *args), **kwargs)

    @classmethod
    def entries_and_exits(cls, shape, entry_mask_nb, exit_mask_nb, *args, **kwargs):
        entries, exits = nb.entries_and_exits_nb(shape, entry_mask_nb, exit_mask_nb, *args)
        return pd.DataFrame(entries, **kwargs), pd.DataFrame(exits, **kwargs)

    def plot(self, trace_kwargs={}, fig=None, **layout_kwargs):
        for col in range(self._obj.shape[1]):
            fig = self._obj.iloc[:, col].vbt.signals.plot(
                trace_kwargs=trace_kwargs,
                fig=fig,
                **layout_kwargs
            )

        return fig


@register_series_accessor('signals')
class Signals_SRAccessor(Signals_Accessor, Base_SRAccessor):

    @classmethod
    def empty(cls, *args, fill_value=False, **kwargs):
        return Base_SRAccessor.empty(*args, fill_value=fill_value, **kwargs)

    @classmethod
    def empty_like(cls, *args, fill_value=False, **kwargs):
        return Base_SRAccessor.empty_like(*args, fill_value=fill_value, **kwargs)

    @classmethod
    def random(cls, size, n, every_nth=1, seed=None, **kwargs):
        return pd.Series(nb.random_nb((size, 1), reshape_fns.to_1d(n), every_nth, seed)[:, 0], **kwargs)

    @classmethod
    def random_by_func(cls, size, choice_func_nb, *args, seed=None, **kwargs):
        return pd.Series(nb.random_by_func_nb(size, choice_func_nb, seed, *args), **kwargs)

    @classmethod
    def entries_and_exits(cls, size, entry_mask_nb, exit_mask_nb, *args, **kwargs):
        entries, exits = nb.entries_and_exits_nb((size, 1), entry_mask_nb, exit_mask_nb, *args)
        return pd.Series(entries[:, 0], **kwargs), pd.Series(exits[:, 0], **kwargs)

    def plot(self, name=None, trace_kwargs={}, fig=None, **layout_kwargs):
        # Set up figure
        if fig is None:
            fig = DefaultFigureWidget()
            fig.update_layout(
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1],
                    ticktext=['false', 'true']
                )
            )
            fig.update_layout(**layout_kwargs)
        if name is None:
            name = self._obj.name
        if name is not None:
            fig.update_layout(showlegend=True)

        scatter = go.Scatter(
            x=self._obj.index,
            y=self._obj.values,
            mode='lines',
            name=str(name) if name is not None else None
        )
        scatter.update(**trace_kwargs)
        fig.add_trace(scatter)

        return fig

    def plot_markers(self, ts, name=None, signal_type=None, trace_kwargs={}, fig=None, **layout_kwargs):
        checks.assert_type(ts, pd.Series)
        ts.vbt.timeseries.validate()
        checks.assert_same_index(self._obj, ts)

        if fig is None:
            fig = DefaultFigureWidget()
            fig.update_layout(**layout_kwargs)

        # Plot markers
        scatter = go.Scatter(
            x=ts.index[self._obj],
            y=ts[self._obj],
            mode='markers',
            marker=dict(
                size=10
            )
        )
        if signal_type == 'entry':
            scatter.marker.symbol = 'triangle-up'
            scatter.marker.color = 'limegreen'
            scatter.name = 'Entry'
        if signal_type == 'exit':
            scatter.marker.symbol = 'triangle-down'
            scatter.marker.color = 'orangered'
            scatter.name = 'Exit'
        scatter.update(**trace_kwargs)
        fig.add_trace(scatter)
        return fig