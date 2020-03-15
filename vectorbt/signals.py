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


@njit(b1[:, :](b1[:, :], optional(i8)), cache=True)
def shuffle_nb(a, seed=None):
    """Shuffle along first axis."""
    if seed is not None:
        np.random.seed(seed)
    b = np.full_like(a, np.nan)
    for col in range(a.shape[1]):
        b[:, col] = np.random.permutation(a[:, col])
    return b


@njit(b1[:, :](UniTuple(i8, 2), i8, optional(i8), optional(i8)), cache=True)
def random_entries_nb(shape, n, every_nth, seed):
    """Randomly generate entry signals."""
    if seed is not None:
        np.random.seed(seed)
    if every_nth is None:
        every_nth = 1
    a = np.full(shape, False, dtype=b1)
    for col in range(a.shape[1]):
        idxs = np.random.choice(np.arange(shape[0])[::every_nth], size=n, replace=False)
        a[idxs, col] = True
    return a


@njit(b1[:, :](b1[:, :], optional(i8)), cache=True)
def random_exits_nb(entries, seed):
    """Randomly generate exit signals between entry signals."""
    if seed is not None:
        np.random.seed(seed)
    a = np.full_like(entries, False)
    for col in range(entries.shape[1]):
        prev_entry_idx = -1
        for i in range(entries.shape[0]):
            if entries[i, col] or i == a.shape[0]-1:
                if prev_entry_idx == -1:
                    prev_entry_idx = i
                    continue
                if i == a.shape[0]-1 and not entries[i, col]:
                    rand_range = np.arange(prev_entry_idx+1, i+1)
                else:
                    rand_range = np.arange(prev_entry_idx+1, i)
                if len(rand_range) > 0:
                    rand_idx = np.random.choice(rand_range)
                    a[rand_idx, col] = True
                prev_entry_idx = i
    return a

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


@njit(i8[:, :](b1[:, :], b1), cache=True)
def rank_true_nb(a, after_false=False):
    """Rank over each partition of true values.
    after_false: must come after at least one false."""
    b = np.zeros(a.shape, dtype=i8)
    for col in range(a.shape[1]):
        if after_false:
            inc = -1
        else:
            inc = 0
        for i in range(a.shape[0]):
            if a[i, col]:
                if not after_false or (after_false and inc != -1):
                    inc += 1
                    b[i, col] = inc
            else:
                inc = 0
    return b


@njit(i8[:, :](b1[:, :], b1), cache=True)
def rank_false_nb(a, after_true=False):
    """Rank over each partition of false values.
    after_true: must come after at least one true."""
    return rank_true_nb(~a, after_true)

# ############# Signal properties ############# #


@njit(f8[:](b1[:, :]), cache=True)
def avg_distance_nb(a):
    b = np.full((a.shape[1],), np.nan)
    for col in range(a.shape[1]):
        b[col] = np.mean(np.diff(np.flatnonzero(a[:, col])))
    return b


# ############# Boolean operations ############# #

# Boolean operations are natively supported by pandas
# You can, for example, perform Signals_1 & Signals_2 to get logical AND of both arrays
# NOTE: We don't implement backward operations to avoid look-ahead bias!


@njit(b1[:, :](b1[:, :], i8), cache=True)
def fshift_nb(a, n):
    b = np.full_like(a, False)
    b[n:, :] = a[:-n, :]
    return b


@njit(b1[:, :](b1[:, :], b1), cache=True)
def first_true_nb(a, after_false=False):
    """Select the first true value in each row of true values."""
    return rank_true_nb(a, after_false=after_false) == 1


@njit(b1[:, :](b1[:, :], b1), cache=True)
def first_false_nb(a, after_true=False):
    """Select the first false value in each row of false values."""
    return rank_false_nb(a, after_true=after_true) == 1


@njit(b1[:, :](b1[:, :], i8, b1), cache=True)
def nst_true_nb(a, n, after_false=False):
    """Select the nst true value in each row of true values."""
    return rank_true_nb(a, after_false=after_false) == n


@njit(b1[:, :](b1[:, :], i8, b1), cache=True)
def nst_false_nb(a, n, after_true=False):
    """Select the nst false value in each row of false values."""
    return rank_false_nb(a, after_true=after_true) == n


@njit(b1[:, :](b1[:, :], i8, b1), cache=True)
def from_nst_true_nb(a, n, after_false=False):
    """Select the nst true value and beyond in each row of true values."""
    return rank_true_nb(a, after_false=after_false) >= n


@njit(b1[:, :](b1[:, :], i8, b1), cache=True)
def from_nst_false_nb(a, n, after_true=False):
    """Select the nst false value and beyond in each row of false values."""
    return rank_false_nb(a, after_true=after_true) >= n


# ############# Stop-loss operations ############# #

@njit(b1[:](b1[:], i8, i8, i8, f8[:, :], f8[:, :], b1), cache=True)
def stoploss_exit_mask_nb(entries, col, prev_idx, next_idx, ts, stop, is_relative):
    """Index of the first event below the stop."""
    ts = ts[:, col]
    # Stop is defined at the entry point
    stop = stop[prev_idx, col]
    if is_relative:
        stop = (1 - stop) * ts[prev_idx]
    return ts < stop


@njit(b1[:, :](b1[:, :], f8[:, :], f8[:, :, :], b1, b1), cache=True)
def stoploss_exits_nb(entries, ts, stops, is_relative, only_first):
    """Calculate exit signals based on stop loss strategy.

    A stop-loss is designed to limit an investor's loss on a security position. 
    Setting a stop-loss order for 10% below the price at which you bought the stock 
    will limit your loss to 10%.

    An approach here significantly differs from the approach with rolling windows.
    If user wants to try out different rolling windows, he can pass them as a 1d array.
    Here, user must be able to try different stops not only for the `ts` itself,
    but also for each element in `ts`, since stops may vary with time.
    This requires the variable `stops` to be a 3d array (cube) out of 2d matrices of form of `ts`.
    For example, if you want to try stops 0.1 and 0.2, both must have the shape of `ts`,
    wrapped into an array, thus forming a cube (2, ts.shape[0], ts.shape[1])"""

    exits = np.empty((ts.shape[0], ts.shape[1] * stops.shape[0]), dtype=b1)
    for i in range(stops.shape[0]):
        exits[:, i*ts.shape[1]:(i+1)*ts.shape[1]] = exits_nb(
            entries, stoploss_exit_mask_nb, only_first, ts, stops[i, :, :], is_relative)
    return exits


@njit(b1[:](b1[:], i8, i8, i8, f8[:, :], f8[:, :], b1), cache=True)
def trailstop_exit_mask_nb(entries, col, prev_idx, next_idx, ts, stop, is_relative):
    """Index of the first event below the trailing stop."""
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
    if is_relative:
        stop = (1 - stop) * peak
    exit_mask[prev_idx:next_idx] = ts < stop
    return exit_mask


@njit(b1[:, :](b1[:, :], f8[:, :], f8[:, :, :], b1, b1), cache=True)
def trailstop_exits_nb(entries, ts, stops, is_relative, only_first):
    """Calculate exit signals based on trailing stop strategy.

    A Trailing Stop order is a stop order that can be set at a defined percentage 
    or amount away from the current market price. The main difference between a regular 
    stop loss and a trailing stop is that the trailing stop moves as the price moves."""
    exits = np.empty((ts.shape[0], ts.shape[1] * stops.shape[0]), dtype=b1)
    for i in range(stops.shape[0]):
        exits[:, i*ts.shape[1]:(i+1)*ts.shape[1]] = exits_nb(
            entries, trailstop_exit_mask_nb, only_first, ts, stops[i, :, :], is_relative)
    return exits

# ############# Custom accessors ############# #


@add_safe_nb_methods(
    shuffle_nb,
    rank_true_nb,
    rank_false_nb,
    fshift_nb,
    first_true_nb,
    first_false_nb,
    nst_true_nb,
    nst_false_nb,
    from_nst_true_nb,
    from_nst_false_nb)
class Signals_Accessor():
    dtype = np.bool

    @classmethod
    def _validate(cls, obj):
        if cls.dtype is not None:
            check_dtype(obj, cls.dtype)

    def random_exits(self, seed=None):
        return self.wrap_array(random_exits_nb(self.to_2d_array(), seed))

    def exits(self, exit_mask_nb, *args, only_first=True):
        return self.wrap_array(exits_nb(self.to_2d_array(), exit_mask_nb, only_first, *args))

    def stoploss_exits(self, ts, stops, is_relative=True, only_first=True):
        # Checks and preprocessing
        entries = self._obj
        check_type(ts, (pd.Series, pd.DataFrame))
        ts.vbt.timeseries.validate()
        check_same_index(entries, ts)

        entries, ts = broadcast(entries, ts)
        stops = broadcast_to_array_of(stops, entries.vbt.to_2d_array())

        exits = stoploss_exits_nb(
            entries.vbt.to_2d_array(),
            ts.vbt.to_2d_array(),
            stops, is_relative, only_first)

        # Build column hierarchy
        param_columns = pd.DataFrame.vbt.index_from_params(stops, name='stoploss')
        columns = entries.vbt.combine_columns(param_columns)
        return entries.vbt.wrap_array(exits, columns=columns)

    def trailstop_exits(self, ts, stops, is_relative=True, only_first=True):
        # Checks and preprocessing
        entries = self._obj
        check_type(ts, (pd.Series, pd.DataFrame))
        ts.vbt.timeseries.validate()
        check_same_index(entries, ts)

        entries, ts = broadcast(entries, ts)
        stops = broadcast_to_array_of(stops, entries.vbt.to_2d_array())

        exits = trailstop_exits_nb(
            entries.vbt.to_2d_array(),
            ts.vbt.to_2d_array(),
            stops, is_relative, only_first)

        # Build column hierarchy
        param_columns = pd.DataFrame.vbt.index_from_params(stops, name='stoploss')
        columns = entries.vbt.combine_columns(param_columns)
        return entries.vbt.wrap_array(exits, columns=columns)

    def AND(self, *others):
        # you can also do A & B, but then you need to have the same index and columns
        return self.broadcast_and_combine(*others, combine_func=np.logical_and)

    def OR(self, *others):
        return self.broadcast_and_combine(*others, combine_func=np.logical_or)

    def XOR(self, *others):
        return self.broadcast_and_combine(*others, combine_func=np.logical_xor)

    def NOT(self):
        return np.logical_not(self._obj)

    @cached_property
    def num_signals(self):
        return self._obj.sum(axis=0)

    @cached_property
    def avg_distance(self):
        sr = pd.Series(avg_distance_nb(self.to_2d_array()), index=to_2d(self._obj).columns)
        if isinstance(self._obj, pd.Series):
            return sr.iloc[0]
        return sr


@register_dataframe_accessor('signals')
class Signals_DFAccessor(Signals_Accessor, Base_DFAccessor):

    @classmethod
    def empty(cls, *args, fill_value=False, **kwargs):
        return Base_DFAccessor.empty(*args, fill_value=fill_value, **kwargs)

    @classmethod
    def empty_like(cls, *args, fill_value=False, **kwargs):
        return Base_DFAccessor.empty_like(*args, fill_value=fill_value, **kwargs)

    @classmethod
    def random_entries(cls, shape, n, every_nth=1, seed=None, **kwargs):
        return pd.DataFrame(random_entries_nb(shape, n, every_nth, seed), **kwargs)

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
    def random_entries(cls, size, n, every_nth=1, seed=None, **kwargs):
        return pd.Series(random_entries_nb((size, 1), n, every_nth, seed)[:, 0], **kwargs)

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
