from vectorbt.decorators import *
from vectorbt.widgets import FigureWidget
from vectorbt.timeseries import CustomBaseAccessor, expanding_max_1d_nb, pct_change_1d_nb, ffill_1d_nb
from numba.types.containers import UniTuple
from numba import njit, f8, i8, b1, optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go

__all__ = []


class Signals:
    pass

# ############# Random signal generation ############# #


@njit(b1[:, :](b1[:, :], optional(i8)), cache=True)
def shuffle_2d_nb(a, seed=None):
    """Shuffle along first axis."""
    if seed is not None:
        np.random.seed(seed)
    b = np.full_like(a, np.nan)
    for col in range(a.shape[1]):
        b[:, col] = np.random.permutation(a[:, col])
    return b


@njit(b1[:, :](UniTuple(i8, 2), i8, optional(i8), optional(i8)), cache=True)
def generate_random_entries_2d_nb(shape, n, every_nth, seed):
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
def generate_random_exits_2d_nb(entries, seed):
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
def generate_exits_2d_nb(entries, exit_func_nb, only_first, *args):
    """Generate entries based on exit_func_nb."""
    # exit_func_nb must return a boolean mask for a column specified by col.
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
                exit_mask = exit_func_nb(entries[:, col], col, prev_idx, next_idx, *args)
                exit_idxs = np.where(exit_mask)[0]
                # Filter out signals before previous entry and after next entry
                idx_mask = (exit_idxs > prev_idx) & (exit_idxs < next_idx)
                if not idx_mask.any():
                    continue
                exit_mask[:] = False
                if only_first:
                    # consider only the first signal
                    exit_mask[exit_idxs[idx_mask][0]] = True
                else:
                    exit_mask[exit_idxs[idx_mask]] = True
                exits[:, col] = exits[:, col] | exit_mask
    return exits


@njit
def generate_entries_and_exits_2d_nb(shape, entry_func_nb, exit_func_nb, *args):
    """Generate entries and exits based on entry_func_nb and exit_func_nb."""
    # entry_func_nb and exit_func_nb must return boolean masks for a column specified by col_idx.
    # You will have to write them to be compatible with numba!

    entries = np.full(shape, False)
    exits = np.full(shape, False)

    for col in range(shape[1]):
        prev_idx = -1
        i = 0
        while prev_idx < shape[0] - 1:
            if i % 2 == 0:
                # Cannot assign two functions to a var in numba
                mask = entry_func_nb(exits[:, col], col, prev_idx, shape[0], *args)
                a = entries
            else:
                mask = exit_func_nb(entries[:, col], col, prev_idx, shape[0], *args)
                a = exits
            if prev_idx != -1:
                mask[:prev_idx+1] = False
            if not mask.any():
                break
            prev_idx = np.where(mask)[0][0]
            mask[:] = False
            mask[prev_idx] = True  # consider only the first signal
            a[:, col] = a[:, col] | mask
            i += 1
    return entries, exits

# ############# Ranking ############# #


@njit(i8[:, :](b1[:, :], b1), cache=True)
def rank_true_2d_nb(a, after_false=False):
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
def rank_false_2d_nb(a, after_true=False):
    """Rank over each partition of false values.
    after_true: must come after at least one true."""
    return rank_true_2d_nb(~a, after_true)

# ############# Signal properties ############# #


@njit(f8[:](b1[:, :]), cache=True)
def avg_distance_2d_nb(a):
    b = np.full((a.shape[1],), np.nan)
    for col in range(a.shape[1]):
        b[col] = np.mean(np.diff(np.flatnonzero(a[:, col])))
    return b


# ############# Boolean operations ############# #

# Boolean operations are natively supported by pandas
# You can, for example, perform Signals_1 & Signals_2 to get logical AND of both arrays
# NOTE: We don't implement backward operations to avoid look-ahead bias!


@njit(b1[:, :](b1[:, :], i8), cache=True)
def fshift_2d_nb(a, n):
    b = np.full_like(a, False)
    b[n:, :] = a[:-n, :]
    return b


@njit(b1[:, :](b1[:, :], b1), cache=True)
def first_true_2d_nb(a, after_false=False):
    """Select the first true value in each row of true values."""
    return rank_true_2d_nb(a, after_false=after_false) == 1


@njit(b1[:, :](b1[:, :], b1), cache=True)
def first_false_2d_nb(a, after_true=False):
    """Select the first false value in each row of false values."""
    return rank_false_2d_nb(a, after_true=after_true) == 1


@njit(b1[:, :](b1[:, :], i8, b1), cache=True)
def nst_true_2d_nb(a, n, after_false=False):
    """Select the nst true value in each row of true values."""
    return rank_true_2d_nb(a, after_false=after_false) == n


@njit(b1[:, :](b1[:, :], i8, b1), cache=True)
def nst_false_2d_nb(a, n, after_true=False):
    """Select the nst false value in each row of false values."""
    return rank_false_2d_nb(a, after_true=after_true) == n


@njit(b1[:, :](b1[:, :], i8, b1), cache=True)
def from_nst_true_2d_nb(a, n, after_false=False):
    """Select the nst true value and beyond in each row of true values."""
    return rank_true_2d_nb(a, after_false=after_false) >= n


@njit(b1[:, :](b1[:, :], i8, b1), cache=True)
def from_nst_false_2d_nb(a, n, after_true=False):
    """Select the nst false value and beyond in each row of false values."""
    return rank_false_2d_nb(a, after_true=after_true) >= n


# ############# Stop-loss operations ############# #

@njit(b1[:](b1[:], i8, i8, i8, f8[:, :], f8[:, :], b1), cache=True)
def stoploss_exit_mask_2d_nb(entries, col, prev_idx, next_idx, ts, stop, is_relative):
    """Index of the first event below the stop."""
    ts = ts[:, col]
    # Stop is defined at the entry point
    stop = stop[prev_idx, col]
    if is_relative:
        stop = (1 - stop) * ts[prev_idx]
    return ts < stop


@njit(b1[:, :](b1[:, :], f8[:, :], f8[:, :, :], b1, b1), cache=True)
def stoploss_exits_2d_nb(entries, ts, stops, is_relative, only_first):
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
        exits[:, i*ts.shape[1]:(i+1)*ts.shape[1]] = generate_exits_2d_nb(
            entries, stoploss_exit_mask_2d_nb, only_first, ts, stops[i, :, :], is_relative)
    return exits


@njit(b1[:](b1[:], i8, i8, i8, f8[:, :], f8[:, :], b1), cache=True)
def trailstop_exit_mask_2d_nb(entries, col, prev_idx, next_idx, ts, stop, is_relative):
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
def trailstop_exits_2d_nb(entries, ts, stops, is_relative, only_first):
    """Calculate exit signals based on trailing stop strategy.

    A Trailing Stop order is a stop order that can be set at a defined percentage 
    or amount away from the current market price. The main difference between a regular 
    stop loss and a trailing stop is that the trailing stop moves as the price moves."""
    exits = np.empty((ts.shape[0], ts.shape[1] * stops.shape[0]), dtype=b1)
    for i in range(stops.shape[0]):
        exits[:, i*ts.shape[1]:(i+1)*ts.shape[1]] = generate_exits_2d_nb(
            entries, trailstop_exit_mask_2d_nb, only_first, ts, stops[i, :, :], is_relative)
    return exits

# ############# Custom pd.DataFrame accessor ############# #


@pd.api.extensions.register_dataframe_accessor("signals")
@add_safe_nb_methods(
    shuffle_2d_nb,
    rank_true_2d_nb,
    rank_false_2d_nb,
    fshift_2d_nb,
    first_true_2d_nb,
    first_false_2d_nb,
    nst_true_2d_nb,
    nst_false_2d_nb,
    from_nst_true_2d_nb,
    from_nst_false_2d_nb)
class CustomDFAccessor(CustomBaseAccessor):
    def __init__(self, obj):
        self._validate(obj)
        self._obj = obj

    @staticmethod
    def _validate(obj):
        if (obj.dtypes != np.bool_).any():
            raise ValueError("All columns must be boolean")

    @classmethod
    def generate_empty(cls, shape, **kwargs):
        return pd.DataFrame(np.full(shape, False), **kwargs)

    @classmethod
    def generate_random_entries(cls, shape, n, every_nth=1, seed=None, **kwargs):
        return pd.DataFrame(generate_random_entries_2d_nb(shape, n, every_nth, seed), **kwargs)

    def generate_random_exits(self, seed=None):
        return self._after_nb(generate_random_exits_2d_nb(self._before_nb(), seed))

    @classmethod
    def generate_entries_and_exits(cls, shape, entry_func_nb, exit_func_nb, *args, **kwargs):
        entries, exits = generate_entries_and_exits_2d_nb(shape, entry_func_nb, exit_func_nb, *args)
        return pd.DataFrame(entries, **kwargs), pd.DataFrame(exits, **kwargs)

    def generate_exits(self, exit_func_nb, *args, only_first=True):
        return self._after_nb(generate_exits_2d_nb(self._before_nb(), exit_func_nb, only_first, *args))

    @pass_pd_obj('entries')
    @has_type('entries', (pd.Series, pd.DataFrame))
    @has_type('ts', (pd.Series, pd.DataFrame))
    @has_type('stop_index', pd.Index)
    @to_2d('entries')
    @to_2d('ts')
    @broadcast('entries', 'ts')
    @broadcast_to_combs_of('stops', 'entries')
    def generate_stoploss_exits(self, entries, ts, stops, is_relative=True, only_first=True, stop_index=None):
        exits = stoploss_exits_2d_nb(entries.values, ts.values, stops, is_relative, only_first)
        if stops.shape[0] > 1:
            if stop_index is not None:
                upper_cols = stop_index
            else:
                upper_cols = pd.Index(np.arange(stops.shape[0]))
            lower_cols = entries.columns
            columns = self.vstack_columns(upper_cols, lower_cols)
            return self._after_nb(exits, columns=columns)
        else:
            return self._after_nb(exits)

    @pass_pd_obj('entries')
    @has_type('entries', (pd.Series, pd.DataFrame))
    @has_type('ts', (pd.Series, pd.DataFrame))
    @has_type('stop_index', pd.Index)
    @to_2d('entries')
    @to_2d('ts')
    @broadcast('entries', 'ts')
    @broadcast_to_combs_of('stops', 'entries')
    def generate_trailstop_exits(self, entries, ts, stops, is_relative=True, only_first=True, stop_index=None):
        exits = trailstop_exits_2d_nb(entries.values, ts.values, stops, is_relative, only_first)
        if stops.shape[0] > 1:
            if stop_index is not None:
                upper_cols = stop_index
            else:
                upper_cols = pd.Index(np.arange(stops.shape[0]))
            lower_cols = entries.columns
            columns = self.vstack_columns(upper_cols, lower_cols)
            return self._after_nb(exits, columns=columns)
        else:
            return self._after_nb(exits)

    @cached_property
    def n(self):
        """Number of signals."""
        return pd.Series(np.sum(self._obj.values, axis=0), index=self._obj.columns)

    @cached_property
    def avg_distance(self):
        """Average distance between signals."""
        return pd.Series(avg_distance_2d_nb(self._obj.values), index=self._obj.columns)

    def plot(self, scatter_kwargs={}, fig=None, **layout_kwargs):
        # Plot TimeSeries
        for col in range(self._obj.shape[1]):
            fig = self._obj.iloc[:, col].signals.plot(
                scatter_kwargs=scatter_kwargs[col] if isinstance(scatter_kwargs, list) else scatter_kwargs,
                fig=fig,
                **layout_kwargs
            )

        return fig


# ############# Custom pd.Series accessor ############# #


@pd.api.extensions.register_series_accessor("signals")
class CustomSRAccessor(CustomDFAccessor):
    def __init__(self, obj):
        self._validate(obj)
        self._obj = obj

    @staticmethod
    def _validate(obj):
        if obj.dtype != np.bool_:
            raise ValueError("Must be boolean")

    @classmethod
    def generate_empty(cls, size, **kwargs):
        return pd.Series(np.full(size, False), **kwargs)

    @classmethod
    def generate_random_entries(cls, size, n, every_nth=1, seed=None, **kwargs):
        return pd.Series(generate_random_entries_2d_nb((size, 1), n, every_nth, seed)[:, 0], **kwargs)

    @classmethod
    def generate_entries_and_exits(cls, size, entry_func_nb, exit_func_nb, *args, **kwargs):
        entries, exits = generate_entries_and_exits_2d_nb((size, 1), entry_func_nb, exit_func_nb, *args)
        return pd.Series(entries[:, 0], **kwargs), pd.Series(exits[:, 0], **kwargs)

    @cached_property
    def n(self):
        return np.sum(self._before_nb(force_1d=True))

    @cached_property
    def avg_distance(self):
        return avg_distance_2d_nb(self._before_nb())[0]

    def plot(self, scatter_kwargs={}, fig=None, **layout_kwargs):
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
        if self._obj.name is not None:
            fig.update_layout(showlegend=True)

        # Plot TimeSeries
        scatter = go.Scatter(
            x=self._obj.index,
            y=self._obj.values.astype(np.uint8),
            mode='lines',
            name=str(self._obj.name)
        )
        scatter.update(**scatter_kwargs)
        fig.add_trace(scatter)

        return fig
