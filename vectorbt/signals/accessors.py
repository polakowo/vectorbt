"""Custom pandas accessors.

!!! note
    The underlying Series/DataFrame must already be a signal series.

    Input arrays must be `np.bool`.
    
```python-repl
>>> import vectorbt as vbt
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> from datetime import datetime

>>> sig = pd.DataFrame({
...     'a': [True, False, False, False, False],
...     'b': [True, False, True, False, True],
...     'c': [True, True, True, False, False]
... }, index=pd.Index([
...     datetime(2020, 1, 1),
...     datetime(2020, 1, 2),
...     datetime(2020, 1, 3),
...     datetime(2020, 1, 4),
...     datetime(2020, 1, 5)
... ]))
>>> sig
                a      b      c
2020-01-01   True   True   True
2020-01-02  False  False   True
2020-01-03  False   True   True
2020-01-04  False  False  False
2020-01-05  False   True  False
```"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt.defaults import contrast_color_schema
from vectorbt.root_accessors import register_dataframe_accessor, register_series_accessor
from vectorbt.utils import checks
from vectorbt.utils.config import merge_kwargs
from vectorbt.utils.colors import adjust_lightness
from vectorbt.utils.decorators import cached_property
from vectorbt.base import reshape_fns, index_fns
from vectorbt.base.common import add_nb_methods
from vectorbt.generic.accessors import Generic_Accessor, Generic_SRAccessor, Generic_DFAccessor
from vectorbt.signals import nb
from vectorbt.utils.widgets import CustomFigureWidget


@add_nb_methods([
    nb.shuffle_nb,
    nb.fshift_nb,
], module_name='vectorbt.signals.nb')
class Signals_Accessor(Generic_Accessor):
    """Accessor on top of signal series. For both, Series and DataFrames.

    Accessible through `pd.Series.vbt.signals` and `pd.DataFrame.vbt.signals`."""

    def __init__(self, obj, freq=None):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        checks.assert_dtype(obj, np.bool)

        Generic_Accessor.__init__(self, obj, freq=freq)

    @classmethod
    def empty(cls, *args, fill_value=False, **kwargs):
        """`vectorbt.base.accessors.Base_Accessor.empty` with `fill_value=False`.

        Example:
            ```python-repl
            >>> pd.Series.vbt.signals.empty(5, index=sig.index, name=sig['a'].name)
            2020-01-01    False
            2020-01-02    False
            2020-01-03    False
            2020-01-04    False
            2020-01-05    False
            Name: a, dtype: bool
            >>> pd.DataFrame.vbt.signals.empty((5, 3), index=sig.index, columns=sig.columns)
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False  False  False
            2020-01-03  False  False  False
            2020-01-04  False  False  False
            2020-01-05  False  False  False
            ```"""
        return Generic_Accessor.empty(*args, fill_value=fill_value, **kwargs)

    @classmethod
    def empty_like(cls, *args, fill_value=False, **kwargs):
        """`vectorbt.base.accessors.Base_Accessor.empty_like` with `fill_value=False`.

        Example:
            ```python-repl
            >>> pd.Series.vbt.signals.empty_like(sig['a'])
            2020-01-01    False
            2020-01-02    False
            2020-01-03    False
            2020-01-04    False
            2020-01-05    False
            Name: a, dtype: bool
            >>> pd.DataFrame.vbt.signals.empty_like(sig)
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False  False  False
            2020-01-03  False  False  False
            2020-01-04  False  False  False
            2020-01-05  False  False  False
            ```"""
        return Generic_Accessor.empty_like(*args, fill_value=fill_value, **kwargs)

    # ############# Signal generation ############# #

    @classmethod
    def generate(cls, shape, choice_func_nb, *args, **kwargs):
        """See `vectorbt.signals.nb.generate_nb`.

        `**kwargs` will be passed to pandas constructor.

        Example:
            Generate random signals manually:
            ```python-repl
            >>> @njit
            ... def choice_func_nb(col, from_i, to_i):
            ...     return col + from_i

            >>> pd.DataFrame.vbt.signals.generate((5, 3),
            ...     choice_func_nb, index=sig.index, columns=sig.columns)
                            a      b      c
            2020-01-01   True  False  False
            2020-01-02  False   True  False
            2020-01-03  False  False   True
            2020-01-04  False  False  False
            2020-01-05  False  False  False
            ```"""
        checks.assert_numba_func(choice_func_nb)

        if not isinstance(shape, tuple):
            shape = (shape, 1)
        elif isinstance(shape, tuple) and len(shape) == 1:
            shape = (shape[0], 1)

        result = nb.generate_nb(shape, choice_func_nb, *args)

        if cls.is_series():
            if shape[1] > 1:
                raise ValueError("Use DataFrame accessor")
            return pd.Series(result[:, 0], **kwargs)
        return pd.DataFrame(result, **kwargs)

    @classmethod
    def generate_entries_and_exits(cls, shape, entry_choice_func_nb, exit_choice_func_nb,
                                   entry_args, exit_args, **kwargs):
        """See `vectorbt.signals.nb.generate_enex_nb`.

        `**kwargs` will be passed to pandas constructor.

        Example:
            Generate entry and exit signals one after another:
            ```python-repl
            >>> @njit
            ... def entry_choice_func_nb(col, from_i, to_i, wait1):
            ...     next_pos = col + from_i + wait1
            ...     if next_pos < to_i:
            ...          return np.array([next_pos])
            ...     return np.empty(0, dtype=np.int_)
            >>> @njit
            ... def exit_choice_func_nb(col, from_i, to_i, wait2):
            ...     next_pos = col + from_i + wait2
            ...     if next_pos < to_i:
            ...          return np.array([next_pos])
            ...     return np.empty(0, dtype=np.int_)

            >>> en, ex = pd.DataFrame.vbt.signals.generate_entries_and_exits(
            ...     (5, 3), entry_choice_func_nb, exit_choice_func_nb, (0,), (1,),
            ...     index=sig.index, columns=sig.columns)
            >>> en
                            a      b      c
            2020-01-01   True  False  False
            2020-01-02  False   True  False
            2020-01-03  False  False   True
            2020-01-04   True  False  False
            2020-01-05  False  False  False
            >>> ex
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False  False  False
            2020-01-03   True  False  False
            2020-01-04  False  False  False
            2020-01-05  False   True  False
            ```"""
        checks.assert_numba_func(entry_choice_func_nb)
        checks.assert_numba_func(exit_choice_func_nb)

        if not isinstance(shape, tuple):
            shape = (shape, 1)
        elif isinstance(shape, tuple) and len(shape) == 1:
            shape = (shape[0], 1)

        result1, result2 = nb.generate_enex_nb(
            shape,
            entry_choice_func_nb,
            exit_choice_func_nb,
            entry_args,
            exit_args
        )
        if cls.is_series():
            if shape[1] > 1:
                raise ValueError("Use DataFrame accessor")
            return pd.Series(result1[:, 0], **kwargs), pd.Series(result2[:, 0], **kwargs)
        return pd.DataFrame(result1, **kwargs), pd.DataFrame(result2, **kwargs)

    def generate_exits(self, exit_choice_func_nb, *args):
        """See `vectorbt.signals.nb.generate_ex_nb`.

        Example:
            Fill all space between signals in `sig`:
            ```python-repl
            >>> @njit
            ... def exit_choice_func_nb(col, from_i, to_i):
            ...     return np.arange(from_i, to_i)

            >>> sig.vbt.signals.generate_exits(exit_choice_func_nb)
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02   True   True  False
            2020-01-03   True  False  False
            2020-01-04   True   True   True
            2020-01-05   True  False   True
            ```"""
        checks.assert_numba_func(exit_choice_func_nb)

        return self.wrap(nb.generate_ex_nb(self.to_2d_array(), exit_choice_func_nb, *args))

    # ############# Random ############# #

    @classmethod
    def generate_random(cls, shape, n=None, prob=None, seed=None, **kwargs):
        """Generate signals randomly.

        If `n` is set, see `vectorbt.signals.nb.generate_rand_nb`.
        If `prob` is set, see `vectorbt.signals.nb.generate_rand_by_prob_nb`.

        `prob` must be either a single number or an array that will be broadcast to match `shape`.
        `**kwargs` will be passed to pandas constructor.

        Example:
            For each column, generate two signals randomly:
            ```python-repl
            >>> pd.DataFrame.vbt.signals.generate_random((5, 3), n=2,
            ...     seed=42, index=sig.index, columns=sig.columns)
                            a      b      c
            2020-01-01  False  False   True
            2020-01-02   True   True   True
            2020-01-03  False  False  False
            2020-01-04  False   True  False
            2020-01-05   True  False  False
            ```

            For each column and time step, pick a signal with 50% probability:
            ```python-repl
            >>> pd.DataFrame.vbt.signals.generate_random((5, 3), prob=0.5,
            ...     seed=42, index=sig.index, columns=sig.columns)
                            a      b      c
            2020-01-01   True   True   True
            2020-01-02  False   True  False
            2020-01-03  False  False  False
            2020-01-04  False  False   True
            2020-01-05   True  False   True
            ```"""
        if not isinstance(shape, tuple):
            shape = (shape, 1)
        elif isinstance(shape, tuple) and len(shape) == 1:
            shape = (shape[0], 1)

        if n is not None:
            result = nb.generate_rand_nb(shape, n, seed=seed)
        elif prob is not None:
            probs = np.broadcast_to(prob, shape)
            result = nb.generate_rand_by_prob_nb(shape, probs, seed=seed)
        else:
            raise ValueError("At least n or prob must be set")

        if cls.is_series():
            if shape[1] > 1:
                raise ValueError("Use DataFrame accessor")
            return pd.Series(result[:, 0], **kwargs)
        return pd.DataFrame(result, **kwargs)

    # ############# Exits ############# #

    @classmethod
    def generate_random_entries_and_exits(cls, shape, n=None, entry_prob=None, exit_prob=None, seed=None, **kwargs):
        """Generate entry and exit signals randomly and iteratively.

        If `n` is set, see `vectorbt.signals.nb.generate_rand_enex_nb`.
        If `prob` is set, see `vectorbt.signals.nb.generate_rand_enex_by_prob_nb`.

        `entry_prob` and `exit_prob` must be either a single number or an array that will be
        broadcast to match `shape`. `**kwargs` will be passed to pandas constructor.

        Example:
            For each column, generate two entries and exits randomly:
            ```python-repl
            >>> en, ex = pd.DataFrame.vbt.signals.generate_random_entries_and_exits(
            ...      (5, 3), n=2, seed=42, index=sig.index, columns=sig.columns)
            >>> en
                            a      b      c
            2020-01-01   True   True   True
            2020-01-02  False  False  False
            2020-01-03   True   True  False
            2020-01-04  False  False   True
            2020-01-05  False  False  False
            >>> ex
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02   True   True   True
            2020-01-03  False  False  False
            2020-01-04  False   True  False
            2020-01-05   True  False   True
            ```

            For each column and time step, pick entry with 50% probability and exit right after:
            ```python-repl
            >>> en, ex = pd.DataFrame.vbt.signals.generate_random_entries_and_exits(
            ...     (5, 3), entry_prob=0.5, exit_prob=1.,
            ...     seed=42, index=sig.index, columns=sig.columns)
            >>> en
                            a      b      c
            2020-01-01   True   True  False
            2020-01-02  False  False   True
            2020-01-03  False   True  False
            2020-01-04   True  False   True
            2020-01-05  False  False  False
            >>> ex
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02   True   True  False
            2020-01-03  False  False   True
            2020-01-04  False   True  False
            2020-01-05   True  False   True
            ```"""
        if not isinstance(shape, tuple):
            shape = (shape, 1)
        elif isinstance(shape, tuple) and len(shape) == 1:
            shape = (shape[0], 1)

        if n is not None:
            entries, exits = nb.generate_rand_enex_nb(shape, n, seed=seed)
        elif entry_prob is not None and exit_prob is not None:
            entry_prob = np.broadcast_to(entry_prob, shape)
            exit_prob = np.broadcast_to(exit_prob, shape)
            entries, exits = nb.generate_rand_enex_by_prob_nb(shape, entry_prob, exit_prob, seed=seed)
        else:
            raise ValueError("At least n, or entry_prob and exit_prob must be set")

        if cls.is_series():
            if shape[1] > 1:
                raise ValueError("Use DataFrame accessor")
            return pd.Series(entries[:, 0], **kwargs), pd.Series(exits[:, 0], **kwargs)
        return pd.DataFrame(entries, **kwargs), pd.DataFrame(exits, **kwargs)

    def generate_random_exits(self, prob=None, seed=None):
        """Generate exit signals randomly.

        If `prob` is `None`, see `vectorbt.signals.nb.generate_rand_ex_nb`.
        Otherwise, see `vectorbt.signals.nb.generate_rand_ex_by_prob_nb`.

        Example:
            After each entry in `sig`, generate exactly one exit:
            ```python-repl
            >>> sig.vbt.signals.generate_random_exits(seed=42)
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02  False   True  False
            2020-01-03   True  False  False
            2020-01-04  False   True  False
            2020-01-05  False  False   True
            ```

            After each entry in `sig` and at each time step, generate exit with 50% probability:
            ```python-repl
            >>> sig.vbt.signals.generate_random_exits(prob=0.5, seed=42)
                            a      b      c
            2020-01-01  False  False  False
            2020-01-02   True  False  False
            2020-01-03  False  False  False
            2020-01-04  False  False  False
            2020-01-05  False  False   True
            ```"""
        if prob is not None:
            obj, prob = reshape_fns.broadcast(self._obj, prob)
            prob = reshape_fns.to_2d(prob, raw=True)
            return obj.vbt.wrap(nb.generate_rand_ex_by_prob_nb(self.to_2d_array(), prob, seed=seed))
        return self.wrap(nb.generate_rand_ex_nb(self.to_2d_array(), seed=seed))

    def generate_stop_loss_exits(self, ts, stops, trailing=False, first=True, iteratively=False,
                                 keys=None, broadcast_kwargs={}):
        """Generate (trailing) stop loss exits.

        If `iteratively` is `True`, see `vectorbt.signals.nb.generate_sl_ex_iter_nb`.
        Otherwise, see `vectorbt.signals.nb.generate_sl_ex_nb`.

        Arguments will be broadcasted using `vectorbt.base.reshape_fns.broadcast`
        with `broadcast_kwargs`. Argument `stops` can be either a single number, an array of
        numbers, or a 3D array, where each matrix corresponds to a single configuration.
        Use `keys` as the outermost level.

        Example:
            For each entry in `sig`, set stop loss for 10% and 20% below the entry price:
            ```python-repl
            >>> ts = pd.Series([1, 2, 3, 2, 1])
            >>> sig.vbt.signals.generate_stop_loss_exits(ts, [0.1, 0.5])
            stop_loss                   0.1                  0.5
                            a      b      c      a      b      c
            2020-01-01  False  False  False  False  False  False
            2020-01-02  False  False  False  False  False  False
            2020-01-03  False  False  False  False  False  False
            2020-01-04  False   True   True  False  False  False
            2020-01-05  False  False  False  False  False   True
            >>> sig.vbt.signals.generate_stop_loss_exits(ts, [0.1, 0.5], trailing=True)
            trail_stop                  0.1                  0.5
                            a      b      c      a      b      c
            2020-01-01  False  False  False  False  False  False
            2020-01-02  False  False  False  False  False  False
            2020-01-03  False  False  False  False  False  False
            2020-01-04   True   True   True  False  False  False
            2020-01-05  False  False  False   True  False   True
            ```"""
        entries = self._obj
        checks.assert_type(ts, (pd.Series, pd.DataFrame))

        entries, ts = reshape_fns.broadcast(entries, ts, **broadcast_kwargs, writeable=True)
        stops = reshape_fns.broadcast_to_array_of(stops, entries.vbt.to_2d_array())

        # Build column hierarchy
        if keys is not None:
            param_columns = keys
        else:
            name = 'trail_stop' if trailing else 'stop_loss'
            param_columns = index_fns.index_from_values(stops, name=name)
        columns = index_fns.combine_indexes(param_columns, entries.vbt.columns)

        # Perform generation
        if iteratively:
            new_entries, exits = nb.generate_sl_ex_iter_nb(
                entries.vbt.to_2d_array(),
                ts.vbt.to_2d_array(),
                stops,
                trailing=trailing)
            return entries.vbt.wrap(new_entries, columns=columns), entries.vbt.wrap(exits, columns=columns)
        else:
            exits = nb.generate_sl_ex_nb(
                entries.vbt.to_2d_array(),
                ts.vbt.to_2d_array(),
                stops,
                trailing=trailing,
                first=first)
            return entries.vbt.wrap(exits, columns=columns)

    def generate_take_profit_exits(self, ts, stops, first=True, iteratively=False, keys=None, broadcast_kwargs={}):
        """Generate take profit exits.

        See `vectorbt.signals.nb.generate_tp_ex_iter_nb` if `iteratively` is `True`, otherwise see
        `vectorbt.signals.nb.generate_tp_ex_nb`.

        Arguments will be broadcasted using `vectorbt.base.reshape_fns.broadcast`
        with `broadcast_kwargs`. Argument `stops` can be either a single number, an array of
        numbers, or a 3D array, where each matrix corresponds to a single configuration.
        Use `keys` as the outermost level.

        Example:
            For each entry in `sig`, set take profit for 10% and 20% above the entry price:
            ```python-repl
            >>> ts = pd.Series([1, 2, 3, 4, 5])
            >>> sig.vbt.signals.generate_take_profit_exits(ts, [0.1, 0.5])
            take_profit                  0.1                  0.5
                             a      b      c      a      b      c
            2020-01-01   False  False  False  False  False  False
            2020-01-02    True   True  False   True   True  False
            2020-01-03   False  False  False  False  False  False
            2020-01-04   False   True   True  False  False  False
            2020-01-05   False  False  False  False  False   True
            ```"""
        entries = self._obj
        checks.assert_type(ts, (pd.Series, pd.DataFrame))

        entries, ts = reshape_fns.broadcast(entries, ts, **broadcast_kwargs, writeable=True)
        stops = reshape_fns.broadcast_to_array_of(stops, entries.vbt.to_2d_array())

        # Build column hierarchy
        if keys is not None:
            param_columns = keys
        else:
            param_columns = index_fns.index_from_values(stops, name='take_profit')
        columns = index_fns.combine_indexes(param_columns, entries.vbt.columns)

        # Perform generation
        if iteratively:
            new_entries, exits = nb.generate_tp_ex_iter_nb(
                entries.vbt.to_2d_array(),
                ts.vbt.to_2d_array(),
                stops)
            return entries.vbt.wrap(new_entries, columns=columns), entries.vbt.wrap(exits, columns=columns)
        else:
            exits = nb.generate_tp_ex_nb(
                entries.vbt.to_2d_array(),
                ts.vbt.to_2d_array(),
                stops,
                first=first)
            return entries.vbt.wrap(exits, columns=columns)

    # ############# Map and reduce ############# #

    def map_reduce_between(self, *args, other=None, map_func_nb=None, reduce_func_nb=None, broadcast_kwargs={}):
        """See `vectorbt.signals.nb.map_reduce_between_nb`.

        If `other` specified, see `vectorbt.signals.nb.map_reduce_between_two_nb`.

        Arguments will be broadcasted using `vectorbt.base.reshape_fns.broadcast`
        with `broadcast_kwargs`.

        Example:
            Get average distance between signals in `sig`:
            ```python-repl
            >>> distance_map_nb = njit(lambda col, from_i, to_i: to_i - from_i)
            >>> mean_reduce_nb = njit(lambda col, a: np.nanmean(a))

            >>> sig.vbt.signals.map_reduce_between(
            ...     map_func_nb=distance_map_nb,
            ...     reduce_func_nb=mean_reduce_nb)
            a    NaN
            b    2.0
            c    1.0
            dtype: float64
            ```"""
        checks.assert_not_none(map_func_nb)
        checks.assert_not_none(reduce_func_nb)
        checks.assert_numba_func(map_func_nb)
        checks.assert_numba_func(reduce_func_nb)

        if other is None:
            # One input array
            result = nb.map_reduce_between_nb(self.to_2d_array(), map_func_nb, reduce_func_nb, *args)
            if isinstance(self._obj, pd.Series):
                return result[0]
            return pd.Series(result, index=self.columns)
        else:
            # Two input arrays
            obj, other = reshape_fns.broadcast(self._obj, other, **broadcast_kwargs)
            checks.assert_dtype(other, np.bool_)
            result = nb.map_reduce_between_two_nb(
                self.to_2d_array(), other.vbt.to_2d_array(), map_func_nb, reduce_func_nb, *args)
            return self.wrap_reduced(result)

    def map_reduce_partitions(self, *args, map_func_nb=None, reduce_func_nb=None):
        """See `vectorbt.signals.nb.map_reduce_partitions_nb`.

        Example:
            Get average length of each partition in `sig`:
            ```python-repl
            >>> distance_map_nb = njit(lambda col, from_i, to_i: to_i - from_i)
            >>> mean_reduce_nb = njit(lambda col, a: np.nanmean(a))

            >>> sig.vbt.signals.map_reduce_partitions(
            ...     map_func_nb=distance_map_nb,
            ...     reduce_func_nb=mean_reduce_nb)
            a    1.0
            b    1.0
            c    3.0
            dtype: float64
            ```"""
        checks.assert_not_none(map_func_nb)
        checks.assert_not_none(reduce_func_nb)
        checks.assert_numba_func(map_func_nb)
        checks.assert_numba_func(reduce_func_nb)

        result = nb.map_reduce_partitions_nb(self.to_2d_array(), map_func_nb, reduce_func_nb, *args)
        return self.wrap_reduced(result)

    @cached_property
    def num_signals(self):
        """Sum up `True` values."""
        return self.sum()

    @cached_property
    def avg_distance(self):
        """Calculate the average distance between `True` values in `self`.

        See `Signals_Accessor.map_reduce_between`."""
        return self.map_reduce_between(
            other=None, map_func_nb=nb.distance_map_nb, reduce_func_nb=nb.mean_reduce_nb)

    def avg_distance_to(self, other, **kwargs):
        """Calculate the average distance between `True` values in `self` and `other`.

        See `Signals_Accessor.map_reduce_between`."""
        return self.map_reduce_between(
            other=other, map_func_nb=nb.distance_map_nb, reduce_func_nb=nb.mean_reduce_nb, **kwargs)

    # ############# Ranking ############# #

    def rank(self, reset_by=None, after_false=False, allow_gaps=False, broadcast_kwargs={}):
        """See `vectorbt.signals.nb.rank_nb`.

        Example:
            Rank each `True` value in each partition in `sig`:
            ```python-repl
            >>> sig.vbt.signals.rank()
                        a  b  c
            2020-01-01  1  1  1
            2020-01-02  0  0  2
            2020-01-03  0  1  3
            2020-01-04  0  0  0
            2020-01-05  0  1  0
            >>> sig.vbt.signals.rank(after_false=True)
                        a  b  c
            2020-01-01  0  0  0
            2020-01-02  0  0  0
            2020-01-03  0  1  0
            2020-01-04  0  0  0
            2020-01-05  0  1  0
            >>> sig.vbt.signals.rank(allow_gaps=True)
                        a  b  c
            2020-01-01  1  1  1
            2020-01-02  0  0  2
            2020-01-03  0  2  3
            2020-01-04  0  0  0
            2020-01-05  0  3  0
            >>> sig.vbt.signals.rank(reset_by=~sig, allow_gaps=True)
                        a  b  c
            2020-01-01  1  1  1
            2020-01-02  0  0  2
            2020-01-03  0  1  3
            2020-01-04  0  0  0
            2020-01-05  0  1  0
            ```"""
        if reset_by is not None:
            obj, reset_by = reshape_fns.broadcast(self._obj, reset_by, **broadcast_kwargs)
            reset_by = reset_by.vbt.to_2d_array()
        else:
            obj = self._obj
        ranked = nb.rank_nb(
            obj.vbt.to_2d_array(),
            reset_by=reset_by,
            after_false=after_false,
            allow_gaps=allow_gaps)
        return obj.vbt.wrap(ranked)

    def rank_partitions(self, reset_by=None, after_false=False, broadcast_kwargs={}):
        """See `vectorbt.signals.nb.rank_partitions_nb`.

        Example:
            Rank each partition of `True` values in `sig`:
            ```python-repl
            >>> sig.vbt.signals.rank_partitions()
                        a  b  c
            2020-01-01  1  1  1
            2020-01-02  0  0  1
            2020-01-03  0  2  1
            2020-01-04  0  0  0
            2020-01-05  0  3  0
            >>> sig.vbt.signals.rank_partitions(after_false=True)
                        a  b  c
            2020-01-01  0  0  0
            2020-01-02  0  0  0
            2020-01-03  0  1  0
            2020-01-04  0  0  0
            2020-01-05  0  2  0
            >>> sig.vbt.signals.rank_partitions(reset_by=sig)
                        a  b  c
            2020-01-01  1  1  1
            2020-01-02  0  0  1
            2020-01-03  0  1  1
            2020-01-04  0  0  0
            2020-01-05  0  1  0
            ```"""
        if reset_by is not None:
            obj, reset_by = reshape_fns.broadcast(self._obj, reset_by, **broadcast_kwargs)
            reset_by = reset_by.vbt.to_2d_array()
        else:
            obj = self._obj
        ranked = nb.rank_partitions_nb(
            obj.vbt.to_2d_array(),
            reset_by=reset_by,
            after_false=after_false)
        return obj.vbt.wrap(ranked)

    def first(self, **kwargs):
        """`vectorbt.signals.nb.rank_nb` == 1."""
        return self.wrap(self.rank(**kwargs).values == 1)

    def nst(self, n, **kwargs):
        """`vectorbt.signals.nb.rank_nb` == n."""
        return self.wrap(self.rank(**kwargs).values == n)

    def from_nst(self, n, **kwargs):
        """`vectorbt.signals.nb.rank_nb` >= n."""
        return self.wrap(self.rank(**kwargs).values >= n)

    # ############# Logical operations ############# #

    def AND(self, *others, **kwargs):
        """Combine with each in `*others` using logical AND.

        See `vectorbt.base.accessors.Base_Accessor.combine_with_multiple`.

        """
        return self.combine_with_multiple(others, combine_func=np.logical_and, **kwargs)

    def OR(self, *others, **kwargs):
        """Combine with each in `*others` using logical OR.

        See `vectorbt.base.accessors.Base_Accessor.combine_with_multiple`.

        Example:
            Perform two OR operations and concatenate them:
            ```python-repl
            >>> ts = pd.Series([1, 2, 3, 2, 1])
            >>> sig.vbt.signals.OR(ts > 1, ts > 2, concat=True, keys=['>1', '>2'])
                                        >1                   >2
                            a     b      c      a      b      c
            2020-01-01   True  True   True   True   True   True
            2020-01-02   True  True   True  False  False   True
            2020-01-03   True  True   True   True   True   True
            2020-01-04   True  True   True  False  False  False
            2020-01-05  False  True  False  False   True  False
            ```"""
        return self.combine_with_multiple(others, combine_func=np.logical_or, **kwargs)

    def XOR(self, *others, **kwargs):
        """Combine with each in `*others` using logical XOR.

        See `vectorbt.base.accessors.Base_Accessor.combine_with_multiple`."""
        return self.combine_with_multiple(others, combine_func=np.logical_xor, **kwargs)


@register_series_accessor('signals')
class Signals_SRAccessor(Signals_Accessor, Generic_SRAccessor):
    """Accessor on top of signal series. For Series only.

    Accessible through `pd.Series.vbt.signals`."""

    def __init__(self, obj, freq=None):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        Generic_SRAccessor.__init__(self, obj, freq=freq)
        Signals_Accessor.__init__(self, obj, freq=freq)

    def plot(self, name=None, trace_kwargs={}, fig=None, **layout_kwargs):  # pragma: no cover
        """Plot Series as a line.

        Args:
            name (str): Name of the signals.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```python-repl
            >>> signals['a'].vbt.signals.plot()
            ```

            ![](/vectorbt/docs/img/signals_sr_plot.png)"""
        # Set up figure
        if fig is None:
            fig = CustomFigureWidget()
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

        scatter = go.Scatter(
            x=self.index,
            y=self._obj.values,
            mode='lines',
            name=str(name),
            showlegend=name is not None
        )
        scatter.update(**trace_kwargs)
        fig.add_trace(scatter)

        return fig

    def plot_as_markers(self, ts, name=None, trace_kwargs={}, fig=None, **layout_kwargs):  # pragma: no cover
        """Plot Series as markers.

        Args:
            ts (pandas.Series): Time series to plot markers on.

                !!! note
                    Doesn't plot `ts`.

            name (str): Name of the signals.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Example:
            ```python-repl
            >>> ts = pd.Series([1, 2, 3, 2, 1], index=sig.index)
            >>> fig = ts.vbt.plot()
            >>> sig['b'].vbt.signals.plot_as_entry_markers(ts, fig=fig)
            >>> (~sig['b']).vbt.signals.plot_as_exit_markers(ts, fig=fig)
            ```

            ![](/vectorbt/docs/img/signals_plot_as_markers.png)"""
        checks.assert_type(ts, pd.Series)
        checks.assert_same_index(self._obj, ts)

        if fig is None:
            fig = CustomFigureWidget()
        fig.update_layout(**layout_kwargs)
        if name is None:
            name = self._obj.name

        # Plot markers
        scatter = go.Scatter(
            x=ts.index[self._obj],
            y=ts[self._obj],
            mode='markers',
            marker=dict(
                symbol='circle',
                color=contrast_color_schema['blue'],
                size=7,
                line=dict(
                    width=1,
                    color=adjust_lightness(contrast_color_schema['blue'])
                )
            ),
            name=str(name),
            showlegend=name is not None
        )
        scatter.update(**trace_kwargs)
        fig.add_trace(scatter)
        return fig

    def plot_as_entry_markers(self, *args, name='Entry', trace_kwargs={}, **kwargs):  # pragma: no cover
        """Plot signals as entry markers.
        
        See `Signals_SRAccessor.plot_as_markers`."""
        trace_kwargs = merge_kwargs(dict(
            marker=dict(
                symbol='circle',
                color=contrast_color_schema['green'],
                size=7,
                line=dict(
                    width=1,
                    color=adjust_lightness(contrast_color_schema['green'])
                )
            )
        ), trace_kwargs)
        return self.plot_as_markers(*args, name=name, trace_kwargs=trace_kwargs, **kwargs)

    def plot_as_exit_markers(self, *args, name='Exit', trace_kwargs={}, **kwargs):  # pragma: no cover
        """Plot signals as exit markers.
        
        See `Signals_SRAccessor.plot_as_markers`."""
        trace_kwargs = merge_kwargs(dict(
            marker=dict(
                symbol='circle',
                color=contrast_color_schema['orange'],
                size=7,
                line=dict(
                    width=1,
                    color=adjust_lightness(contrast_color_schema['orange'])
                )
            )
        ), trace_kwargs)
        return self.plot_as_markers(*args, name=name, trace_kwargs=trace_kwargs, **kwargs)


@register_dataframe_accessor('signals')
class Signals_DFAccessor(Signals_Accessor, Generic_DFAccessor):
    """Accessor on top of signal series. For DataFrames only.

    Accessible through `pd.DataFrame.vbt.signals`."""

    def __init__(self, obj, freq=None):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        Generic_DFAccessor.__init__(self, obj, freq=freq)
        Signals_Accessor.__init__(self, obj, freq=freq)

    def plot(self, trace_kwargs={}, fig=None, **layout_kwargs):  # pragma: no cover
        """Plot each column in DataFrame as a line.

        Args:
            trace_kwargs (dict or list of dict): Keyword arguments passed to each `plotly.graph_objects.Scatter`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Example:
            ```python-repl
            >>> signals[['a', 'c']].vbt.signals.plot()
            ```

            ![](/vectorbt/docs/img/signals_signals_plot.png)"""
        for col in range(self._obj.shape[1]):
            fig = self._obj.iloc[:, col].vbt.signals.plot(
                trace_kwargs=trace_kwargs,
                fig=fig,
                **layout_kwargs
            )

        return fig
