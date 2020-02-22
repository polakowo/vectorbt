from vectorbt.decorators import *
from vectorbt.widgets import FigureWidget
from numba.types.containers import UniTuple
from numba import njit, f8, i8, b1, optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ############# Numba functions ############# #


@njit(b1[:, :](UniTuple(i8, 2), i8, optional(i8), optional(i8)), cache=True)  # 1.17 ms vs 56.4 ms for vectorized
def generate_random_entries_nb(shape, n, every_nth, seed):
    """Randomly generate entry signals."""
    if seed is not None:
        np.random.seed(seed)
    if every_nth is None:
        every_nth = 1
    a = np.full(shape, False, dtype=b1)
    for i in range(shape[1]):
        idxs = np.random.choice(np.arange(shape[0])[::every_nth], size=n, replace=False)
        a[idxs, i] = True
    return a


@njit(b1[:, :](b1[:, :], optional(i8)), cache=True)  # 3.01 ms vs 3.81 ms for vectorized
def generate_random_exits_nb(entries, seed):
    """Randomly generate exit signals between entry signals."""
    if seed is not None:
        np.random.seed(seed)
    a = np.full_like(entries, False)
    for j in range(entries.shape[1]):
        prev_entry_idx = -1
        for i in range(entries.shape[0]):
            if entries[i, j] or i == a.shape[0]-1:
                if prev_entry_idx == -1:
                    prev_entry_idx = i
                    continue
                if i == a.shape[0]-1 and not entries[i, j]:
                    rand_range = np.arange(prev_entry_idx+1, i+1)
                else:
                    rand_range = np.arange(prev_entry_idx+1, i)
                if len(rand_range) > 0:
                    rand_idx = np.random.choice(rand_range)
                    a[rand_idx, j] = True
                prev_entry_idx = i
    return a


@njit  # 5.49 ms vs 48.8 ms for entries.shape = (1000, 20) and number of entries = 200
# NOTE: no explicit types since args are not known before the runtime
def generate_exits_nb(entries, exit_func_nb, only_first, *args):
    """Generate entries based on exit_func_nb."""
    # exit_func_nb must return a boolean mask for a column specified by col_idx.
    # You will have to write your exit_func to be compatible with numba!

    exits = np.full(entries.shape, False)

    for col_idx in range(entries.shape[1]):
        entry_idxs = np.flatnonzero(entries[:, col_idx])

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
                exit_mask = exit_func_nb(entries, col_idx, prev_idx, next_idx, *args)
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
                exits[:, col_idx] = exits[:, col_idx] | exit_mask
    return exits


@njit  # 39.6 ms vs 274 ms for ts.shape = (1000, 20)
# NOTE: no explicit types since args are not known before the runtime
def generate_entries_and_exits_nb(shape, entry_func_nb, exit_func_nb, *args):
    """Generate entries and exits based on entry_func_nb and exit_func_nb."""
    # entry_func_nb and exit_func_nb must return boolean masks for a column specified by col_idx.
     # You will have to write them to be compatible with numba!

    entries = np.full(shape, False)
    exits = np.full(shape, False)

    for col_idx in range(shape[1]):
        prev_idx = -1
        i = 0
        while prev_idx < shape[0] - 1:
            if i % 2 == 0:
                # Cannot assign two functions to a var in numba
                mask = entry_func_nb(exits, col_idx, prev_idx, -1, *args)
                a = entries
            else:
                mask = exit_func_nb(entries, col_idx, prev_idx, -1, *args)
                a = exits
            if prev_idx != -1:
                mask[:prev_idx+1] = False
            if not mask.any():
                break
            prev_idx = np.where(mask)[0][0]
            mask[:] = False
            mask[prev_idx] = True  # consider only the first signal
            a[:, col_idx] = a[:, col_idx] | mask
            i += 1
    return entries, exits


@njit(i8[:, :](b1[:, :], b1), cache=True)
def rank_true_nb(a, after_false):
    """Rank over each partition of true values.
    
    after_false: must come after at least one false."""
    b = np.zeros(a.shape, dtype=i8)
    for j in range(a.shape[1]):
        if after_false:
            inc = -1
        else:
            inc = 0
        for i in range(a.shape[0]):
            if a[i, j]:
                if not after_false or (after_false and inc != -1):
                    inc += 1
                    b[i, j] = inc
            else:
                inc = 0
    return b


@njit(i8[:, :](b1[:, :], b1), cache=True)
def rank_false_nb(a, after_true):
    """Rank over each partition of false values.
    
    after_true: must come after at least one true."""
    return rank_true_nb(~a, after_true)


@njit(b1[:, :](b1[:, :], optional(i8)), cache=True)
def shuffle(a, seed=None):
    """Shuffle along first axis."""
    if seed is not None:
        np.random.seed(seed)
    b = np.full_like(a, np.nan)
    for i in range(a.shape[1]):
        b[:, i] = np.random.permutation(a[:, i])
    return b


@njit(f8[:](b1[:, :]), cache=True)
def avg_distance_nb(a):
    b = np.full((a.shape[1],), np.nan)
    for i in range(a.shape[1]):
        b[i] = np.mean(np.diff(np.flatnonzero(a[:, i])))
    return b


# ############# Boolean operations ############# #

# Boolean operations are natively supported by np.ndarray
# You can, for example, perform Signals_1 & Signals_2 to get logical AND of both arrays
# NOTE: We don't implement backward operations to avoid look-ahead bias!

@njit(b1[:, :](b1[:, :], i8, b1), cache=True)
def prepend_nb(a, n, fill_value):
    """Prepend n values to the array."""
    b = np.full((a.shape[0]+n, a.shape[1]), fill_value, dtype=a.dtype)
    b[-a.shape[0]:] = a
    return b


@njit(b1[:, :](b1[:, :], i8), cache=True)
def fshift_nb(a, n):
    """Shift forward by n."""
    a = prepend_nb(a, n, False)
    return a[:-n, :]


@njit(b1[:, :](b1[:, :], b1), cache=True)
def first_true_nb(a, after_false=False):
    """Select the first true value in each row of true values."""
    return rank_true_nb(a, after_false) == 1


@njit(b1[:, :](b1[:, :], b1), cache=True)
def first_false_nb(a, after_true=False):
    """Select the first false value in each row of false values."""
    return rank_false_nb(a, after_true) == 1


@njit(b1[:, :](b1[:, :], i8, b1), cache=True)
def nst_true_nb(a, n, after_false=False):
    """Select the nst true value in each row of true values."""
    return rank_true_nb(a, after_false) == n


@njit(b1[:, :](b1[:, :], i8, b1), cache=True)
def nst_false_nb(a, n, after_true=False):
    """Select the nst false value in each row of false values."""
    return rank_false_nb(a, after_true) == n


@njit(b1[:, :](b1[:, :], i8, b1), cache=True)
def from_nst_true_nb(a, n, after_false=False):
    """Select the nst true value and beyond in each row of true values."""
    return rank_true_nb(a, after_false) >= n


@njit(b1[:, :](b1[:, :], i8, b1), cache=True)
def from_nst_false_nb(a, n, after_true=False):
    """Select the nst false value and beyond in each row of false values."""
    return rank_false_nb(a, after_true) >= n

# ############# Main class ############# #

# Add numba functions as methods to the Signals class


@add_nb_methods(
    prepend_nb,
    fshift_nb,
    first_true_nb,
    first_false_nb,
    nst_true_nb,
    nst_false_nb,
    from_nst_true_nb,
    from_nst_false_nb,
    shuffle
)
class Signals(np.ndarray):
    """Signals class extends the np.ndarray class by implementing boolean operations."""

    @to_2d('input_array')
    @has_dtype('input_array', np.bool_)
    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)

    @classmethod
    def falses(cls, shape):
        return cls(np.full(shape, False, dtype=np.bool))

    @classmethod
    def falses_like(cls, a):
        return cls.falses(a.shape)

    @classmethod
    def generate_random_entries(cls, shape, n, every_nth=1, seed=None):
        """Generate entry signals randomly."""
        return cls(generate_random_entries_nb(shape, n, every_nth, seed))

    @classmethod
    def generate_entries_and_exits(cls, shape, entry_func_nb, exit_func_nb, *args):
        """Generate entry and exit signals one after another iteratively.
        Use this if your entries depend on previous exit signals, otherwise generate entries first."""
        entries, exits = generate_entries_and_exits_nb(shape, entry_func_nb, exit_func_nb, *args)
        return cls(entries), cls(exits)

    @to_2d('self')
    def generate_random_exits(self, seed=None):
        """Generate an exit signal after every entry signal randomly."""
        exits = generate_random_exits_nb(self, seed)
        return Signals(exits)

    @to_2d('self')
    def generate_exits(self, exit_func_nb, *args, only_first=True):
        """Generate an exit signal after every entry signal using exit_func."""
        exits = generate_exits_nb(self, exit_func_nb, only_first, *args)
        return Signals(exits)

    @property
    @to_2d('self')
    def n(self):
        """Number of signals."""
        return np.asarray(np.sum(self, axis=0))

    @property
    @to_2d('self')
    def avg_distance(self):
        """Average distance between signals."""
        return avg_distance_nb(self)

    @to_2d('self')
    def plot(self,
             column=None,
             index=None,
             label='Signals',
             scatter_kwargs={},
             fig=None, 
             **layout_kwargs):

        if column is None:
            if self.shape[1] == 1:
                column = 0
            else:
                raise ValueError("For an array with multiple columns, you must pass a column index")
        signals = self[:, column]
        if index is None:
            index = np.arange(signals.shape[0])
        if fig is None:
            fig = FigureWidget()
            fig.update_layout(showlegend=True)
            fig.update_layout(**layout_kwargs)
        # Plot Signals
        scatter = go.Scatter(x=index, y=signals, mode='lines', name=label)
        scatter.update(**scatter_kwargs)
        fig.add_trace(scatter)

        return fig
