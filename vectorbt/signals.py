from numba.types.containers import UniTuple
from numba import njit, f8, i8, b1, optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt.utils import *
from vectorbt.accessors import *
from vectorbt.widgets import FigureWidget
from vectorbt.timeseries import expanding_max_1d_nb, pct_change_1d_nb, ffill_1d_nb

__all__ = []

# ############# Random signal generation ############# #


@njit
def rand_choice_nb(arr, prob):
    """
    https://github.com/numba/numba/issues/2539
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


@njit(b1[:, :](b1[:, :], optional(i8)), cache=True)
def shuffle_nb(a, seed=None):
    """Shuffle along first axis."""
    if seed is not None:
        np.random.seed(seed)
    b = np.full_like(a, np.nan)
    for col in range(a.shape[1]):
        b[:, col] = np.random.permutation(a[:, col])
    return b


@njit
def random_by_func_nb(shape, choice_func_nb, seed, *args):
    """Generate random signals based on function."""
    if seed is not None:
        np.random.seed(seed)
    a = np.full(shape, False, dtype=b1)
    for col in range(a.shape[1]):
        idxs = choice_func_nb(col, np.arange(shape[0]), *args)
        a[idxs, col] = True
    return a


@njit(i8[:](i8, i8[:], i8[:], optional(i8)), cache=True)
def random_choice_nb(col, from_range, n_range, every_nth):
    # n_range is a range of n's to uniformly pick from
    return np.random.choice(from_range[::every_nth], size=np.random.choice(n_range), replace=False)


@njit(b1[:, :](UniTuple(i8, 2), i8[:], optional(i8), optional(i8)), cache=True)
def random_nb(shape, n_range, every_nth, seed):
    """Generate signals by:
    1) randomly picking the number of signals from n_range, and
    2) randomly picking signals based on this number.

    To fix the number of signals, use n_range with one element."""
    return random_by_func_nb(shape, random_choice_nb, seed, n_range, every_nth)


@njit
def random_exits_by_func_nb(entries, choice_func_nb, seed, *args):
    """Generate random exit signals based on function."""
    if seed is not None:
        np.random.seed(seed)
    a = np.full_like(entries, False)
    for col in range(entries.shape[1]):
        entry_idxs = np.flatnonzero(entries[:, col])
        for i in range(entry_idxs.shape[0]):
            prev_idx = entry_idxs[i]
            if i < entry_idxs.shape[0] - 1:
                next_idx = entry_idxs[i+1]
            else:
                next_idx = entries.shape[0]
            if prev_idx < entries.shape[0] - 1:
                from_range = np.arange(prev_idx+1, next_idx)
                if len(from_range) > 0:
                    idxs = choice_func_nb(col, from_range, *args)
                    a[idxs, col] = True
    return a


@njit(i8[:](i8, i8[:], i8[:], optional(i8)), cache=True)
def random_exit_choice_nb(col, from_range, n_range, every_nth):
    return np.random.choice(from_range[::every_nth], size=np.random.choice(n_range), replace=False)


@njit(b1[:, :](b1[:, :], i8[:], optional(i8), optional(i8)), cache=True)
def random_exits_nb(entries, n_range, every_nth, seed):
    """Randomly generate exit signals between entry signals."""
    return random_exits_by_func_nb(entries, random_exit_choice_nb, seed, n_range, every_nth)

# ############# Custom signal generation ############# #


@njit
def exits_nb(entries, exit_mask_nb, only_first, *args):
    """Generate entries based on exit_mask_nb."""
    # exit_mask_nb must return a boolean mask for a column specified by col.
    # You will have to write your exit_func to be compatible with numba!

    exits = np.full(entries.shape, False)

    for col in range(entries.shape[1]):
        entry_idxs = np.flatnonzero(entries[:, col])

        for i in range(entry_idxs.shape[0]):
            # prev_idx is the previous entry index, next_idx is the next entry index
            prev_idx = entry_idxs[i]
            if i < entry_idxs.shape[0] - 1:
                next_idx = entry_idxs[i+1]
            else:
                next_idx = entries.shape[0]

            # If entry is the last element, ignore
            if prev_idx < entries.shape[0] - 1:
                # Exit mask must return mask with exit signals for that column
                exit_mask = exit_mask_nb(entries[:, col], col, prev_idx, next_idx, *args)
                exit_idxs = np.where(exit_mask)[0]
                # Filter out signals before previous entry and after next entry
                idx_mask = (exit_idxs > prev_idx) & (exit_idxs < next_idx)
                if not idx_mask.any():
                    continue
                if only_first:
                    # consider only the first signal
                    exits[exit_idxs[idx_mask][0], col] = True
                else:
                    exits[exit_idxs[idx_mask], col] = True
    return exits


@njit
def entries_and_exits_nb(shape, entry_mask_nb, exit_mask_nb, *args):
    """Generate entries and exits based on entry_mask_nb and exit_mask_nb."""
    # entry_mask_nb and exit_mask_nb must return boolean masks for a column specified by col_idx.
    # You will have to write them to be compatible with numba!

    entries = np.full(shape, False)
    exits = np.full(shape, False)

    for col in range(shape[1]):
        prev_idx = -1
        i = 0
        while prev_idx < shape[0] - 1:
            if i % 2 == 0:
                # Cannot assign two functions to a var in numba
                mask = entry_mask_nb(exits[:, col], col, prev_idx, shape[0], *args)
                a = entries
            else:
                mask = exit_mask_nb(entries[:, col], col, prev_idx, shape[0], *args)
                a = exits
            signal_idxs = np.where(mask)[0]
            idx_mask = signal_idxs > prev_idx
            if not idx_mask.any():
                break
            prev_idx = signal_idxs[idx_mask][0]
            a[prev_idx, col] = True
            i += 1
    return entries, exits

# ############# Ranking ############# #


@njit(i8[:, :](b1[:, :], optional(b1[:, :]), b1, b1), cache=True)
def rank_nb(a, reset_b=None, after_false=False, allow_gaps=False):
    """Rank over each partition of true values.

    after_false: the partition must come after at least one false value.
    allow_gaps: ignore gaps between true values.
    b_ref: use true values in this array to reset ranks."""
    b = np.zeros(a.shape, dtype=i8)
    for col in range(a.shape[1]):
        false_seen = ~after_false
        inc = 0
        for i in range(a.shape[0]):
            if reset_b is not None:
                if reset_b[i, col]:
                    # Signal in b_ref resets rank
                    false_seen = ~after_false
                    inc = 0
            if a[i, col] and false_seen:
                inc += 1
                b[i, col] = inc
            else:
                false_seen = True
                if not allow_gaps:
                    inc = 0
    return b

# ############# Distance properties ############# #


@njit
def map_reduce_between_one_nb(a, map_func_nb, reduce_func_nb, *args):
    """Map and reduce pairwise between True values in one array."""
    b = np.full((a.shape[1],), np.nan)
    for col in range(a.shape[1]):
        a_idxs = np.flatnonzero(a[:, col])
        if a_idxs.shape[0] > 1:
            col_results = np.full(a_idxs.shape[0], np.nan)
            for i in range(1, a_idxs.shape[0]):
                a_prev = a_idxs[i-1]
                a_next = a_idxs[i]
                col_results[i] = map_func_nb(col, a_prev, a_next, *args)
            if len(col_results) > 0:
                b[col] = reduce_func_nb(col_results, *args)
    return b


diff_map_nb = njit(lambda col, a_prev, a_next: a_next - a_prev)
avg_reduce_nb = njit(lambda a: np.nanmean(a))


@njit(f8[:](b1[:, :]), cache=True)
def avg_distance_nb(a):
    """Average distance between True values in the same array."""
    return map_reduce_between_one_nb(a, diff_map_nb, avg_reduce_nb)


@njit
def map_reduce_between_two_nb(a, b, map_func_nb, reduce_func_nb, *args):
    """Map and reduce pairwise between True values in two arrays.

    Applies a mapper function on each pair (a_prev, a_prev <= b < a_next).
    Applies a reducer function on all mapper results in a column."""
    c = np.full((a.shape[1],), np.nan)
    for col in range(a.shape[1]):
        a_idxs = np.flatnonzero(a[:, col])
        if a_idxs.shape[0] > 0:
            b_idxs = np.flatnonzero(b[:, col])
            if b_idxs.shape[0] > 0:
                col_results = np.full(b_idxs.shape, np.nan)
                for i, b_i in enumerate(b_idxs):
                    valid_a_idxs = a_idxs[b_i >= a_idxs]
                    if len(valid_a_idxs) > 0:
                        a_i = valid_a_idxs[-1]  # last preceding a
                        col_results[i] = map_func_nb(col, a_i, b_i, *args)
                c[col] = reduce_func_nb(col_results, *args)
    return c


# ############# Boolean operations ############# #

# Boolean operations are natively supported by pandas
# You can, for example, perform Signals_1 & Signals_2 to get logical AND of both arrays
# NOTE: We don't implement backward operations to avoid look-ahead bias!


@njit(b1[:, :](b1[:, :], i8), cache=True)
def fshift_nb(a, n):
    b = np.full_like(a, False)
    b[n:, :] = a[:-n, :]
    return b


# ############# Stop-loss operations ############# #

@njit(b1[:](b1[:], i8, i8, i8, f8[:, :], f8[:, :], b1), cache=True)
def stop_loss_exit_mask_nb(entries, col, prev_idx, next_idx, ts, stop, relative):
    """Return the mask of values that are below the stop.

    A stop-loss is designed to limit an investor's loss on a security position. 
    Setting a stop-loss order for 10% below the price at which you bought the stock 
    will limit your loss to 10%."""
    ts = ts[:, col]
    # Stop is defined at the entry point
    stop = stop[prev_idx, col]
    if relative:
        stop = (1 - stop) * ts[prev_idx]
    return ts < stop


@njit(b1[:](b1[:], i8, i8, i8, f8[:, :], f8[:, :], b1), cache=True)
def trailing_stop_exit_mask_nb(entries, col, prev_idx, next_idx, ts, stop, relative):
    """Return the mask of values that are below the trailing stop."""
    exit_mask = np.empty(ts.shape[0], dtype=b1)
    ts = ts[prev_idx:next_idx, col]
    stop = stop[prev_idx:next_idx, col]
    # Propagate the maximum value from the entry using expanding max
    peak = expanding_max_1d_nb(ts)
    if np.min(stop) != np.max(stop):
        # Propagate the stop value of the last max
        raising_idxs = np.flatnonzero(pct_change_1d_nb(peak))
        stop_temp = np.full(ts.shape, np.nan)
        stop_temp[raising_idxs] = stop[raising_idxs]
        stop_temp = ffill_1d_nb(stop_temp)
        stop_temp[np.isnan(stop_temp)] = -np.inf
        stop = stop_temp
    if relative:
        stop = (1 - stop) * peak
    exit_mask[prev_idx:next_idx] = ts < stop
    return exit_mask


@njit(b1[:, :](i8, b1[:, :], f8[:, :], f8[:, :, :], b1, b1, b1), cache=True)
def apply_stop_loss_nb(i, entries, ts, stops, relative, only_first, trailing):
    if trailing:
        return exits_nb(entries, trailing_stop_exit_mask_nb, only_first, ts, stops[i, :, :], relative)
    else:
        return exits_nb(entries, stop_loss_exit_mask_nb, only_first, ts, stops[i, :, :], relative)


@njit(b1[:, :](b1[:, :], f8[:, :], f8[:, :, :], b1, b1, b1), cache=True)
def stop_loss_exits_nb(entries, ts, stops, relative, only_first, trailing):
    """Calculate exit signals based on stop loss strategy."""
    return apply_and_concat_one_nb(len(stops), apply_stop_loss_nb, entries, ts, stops, relative, only_first, trailing)

# ############# Custom accessors ############# #


@add_safe_nb_methods(
    shuffle_nb,
    fshift_nb)
class Signals_Accessor():
    dtype = np.bool

    @classmethod
    def _validate(cls, obj):
        if cls.dtype is not None:
            check_dtype(obj, cls.dtype)

    def random_exits(self, n, every_nth=1, seed=None):
        return self.wrap_array(random_exits_nb(self.to_2d_array(), to_1d(n), every_nth, seed))

    def random_exits_by_func(self, choice_func_nb, *args, seed=None):
        return self.wrap_array(random_exits_by_func_nb(self.to_2d_array(), choice_func_nb, seed, *args))

    def exits(self, exit_mask_nb, *args, only_first=True):
        return self.wrap_array(exits_nb(self.to_2d_array(), exit_mask_nb, only_first, *args))

    def stop_loss_exits(self, ts, stops, relative=True, only_first=True, trailing=False, as_columns=None, broadcast_kwargs={}):
        entries = self._obj
        check_type(ts, (pd.Series, pd.DataFrame))
        ts.vbt.timeseries.validate()

        entries, ts = broadcast(entries, ts, **broadcast_kwargs, writeable=True)
        stops = broadcast_to_array_of(stops, entries.vbt.to_2d_array())
        stops = stops.astype(np.float64)

        exits = stop_loss_exits_nb(
            entries.vbt.to_2d_array(),
            ts.vbt.to_2d_array(),
            stops, relative, only_first, trailing)

        # Build column hierarchy
        if as_columns is not None:
            param_columns = as_columns
        else:
            name = 'trail_stop' if trailing else 'stop_loss'
            param_columns = index_from_values(stops, name=name)
        columns = combine_indices(param_columns, to_2d(entries).columns)
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
        sr = pd.Series(avg_distance_nb(self.to_2d_array()), index=to_2d(self._obj).columns)
        if isinstance(self._obj, pd.Series):
            return sr.iloc[0]
        return sr

    def avg_distance_to(self, other, **kwargs):
        return self.map_reduce_between(other=other, map_func_nb=diff_map_nb, reduce_func_nb=avg_reduce_nb, **kwargs)

    def map_reduce_between(self, *args, other=None, map_func_nb=None, reduce_func_nb=None, broadcast_kwargs={}):
        check_not_none(map_func_nb)
        check_not_none(reduce_func_nb)
        if other is None:
            result = map_reduce_between_one_nb(self.to_2d_array(), map_func_nb, reduce_func_nb, *args)
            if isinstance(self._obj, pd.Series):
                return result[0]
            return pd.Series(result, index=to_2d(self._obj).columns)
        else:
            obj, other = broadcast(self._obj, other, **broadcast_kwargs)
            other.vbt.signals.validate()
            result = map_reduce_between_two_nb(
                self.to_2d_array(), other.vbt.to_2d_array(), map_func_nb, reduce_func_nb, *args)
            if isinstance(obj, pd.Series):
                return result[0]
            return pd.Series(result, index=to_2d(obj).columns)

    def rank(self, reset_signals=None, after_false=False, allow_gaps=False, broadcast_kwargs={}):
        if reset_signals is not None:
            obj, reset_signals = broadcast(self._obj, reset_signals, **broadcast_kwargs)
            reset_signals = reset_signals.vbt.to_2d_array()
        else:
            obj = self._obj
        ranked = rank_nb(
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
        return pd.DataFrame(random_nb(shape, to_1d(n), every_nth, seed), **kwargs)

    @classmethod
    def random_by_func(cls, shape, choice_func_nb, *args, seed=None, **kwargs):
        return pd.DataFrame(random_by_func_nb(shape, choice_func_nb, seed, *args), **kwargs)

    @classmethod
    def entries_and_exits(cls, shape, entry_mask_nb, exit_mask_nb, *args, **kwargs):
        entries, exits = entries_and_exits_nb(shape, entry_mask_nb, exit_mask_nb, *args)
        return pd.DataFrame(entries, **kwargs), pd.DataFrame(exits, **kwargs)

    def plot(self, scatter_kwargs={}, fig=None, **layout_kwargs):
        for col in range(self._obj.shape[1]):
            fig = self._obj.iloc[:, col].vbt.signals.plot(
                scatter_kwargs=scatter_kwargs,
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
        return pd.Series(random_nb((size, 1), to_1d(n), every_nth, seed)[:, 0], **kwargs)

    @classmethod
    def random_by_func(cls, size, choice_func_nb, *args, seed=None, **kwargs):
        return pd.Series(random_by_func_nb(size, choice_func_nb, seed, *args), **kwargs)

    @classmethod
    def entries_and_exits(cls, size, entry_mask_nb, exit_mask_nb, *args, **kwargs):
        entries, exits = entries_and_exits_nb((size, 1), entry_mask_nb, exit_mask_nb, *args)
        return pd.Series(entries[:, 0], **kwargs), pd.Series(exits[:, 0], **kwargs)

    def plot(self, name=None, scatter_kwargs={}, fig=None, **layout_kwargs):
        # Set up figure
        if fig is None:
            fig = FigureWidget()
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
        scatter.update(**scatter_kwargs)
        fig.add_trace(scatter)

        return fig
