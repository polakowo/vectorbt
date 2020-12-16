"""Custom pandas accessors.

!!! note
    The underlying Series/DataFrame should already be a signal series.

    Input arrays should be `np.bool`.

    Accessors do not utilize caching.
    
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

from vectorbt.root_accessors import register_dataframe_accessor, register_series_accessor
from vectorbt.utils import checks
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.colors import adjust_lightness
from vectorbt.utils.widgets import CustomFigureWidget
from vectorbt.base import reshape_fns
from vectorbt.base.class_helpers import add_nb_methods
from vectorbt.generic.accessors import Generic_Accessor, Generic_SRAccessor, Generic_DFAccessor
from vectorbt.signals import nb


@add_nb_methods([
    nb.fshift_nb,
], module_name='vectorbt.signals.nb')
class Signals_Accessor(Generic_Accessor):
    """Accessor on top of signal series. For both, Series and DataFrames.

    Accessible through `pd.Series.vbt.signals` and `pd.DataFrame.vbt.signals`."""

    def __init__(self, obj, **kwargs):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        checks.assert_dtype(obj, np.bool)

        Generic_Accessor.__init__(self, obj, **kwargs)

    @classmethod
    def empty(cls, *args, fill_value=False, **kwargs):
        """`vectorbt.base.accessors.Base_Accessor.empty` with `fill_value=False`.

        ## Example

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
        ```
        """
        return Generic_Accessor.empty(*args, fill_value=fill_value, dtype=bool, **kwargs)

    @classmethod
    def empty_like(cls, *args, fill_value=False, **kwargs):
        """`vectorbt.base.accessors.Base_Accessor.empty_like` with `fill_value=False`.

        ## Example

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
        ```
        """
        return Generic_Accessor.empty_like(*args, fill_value=fill_value, dtype=bool, **kwargs)

    # ############# Signal generation ############# #

    @classmethod
    def generate(cls, shape, choice_func_nb, *args, **kwargs):
        """See `vectorbt.signals.nb.generate_nb`.

        `**kwargs` will be passed to pandas constructor.

        ## Example

        Generate random signals manually:
        ```python-repl
        >>> @njit
        ... def choice_func_nb(from_i, to_i, col):
        ...     return col + from_i

        >>> pd.DataFrame.vbt.signals.generate((5, 3),
        ...     choice_func_nb, index=sig.index, columns=sig.columns)
                        a      b      c
        2020-01-01   True  False  False
        2020-01-02  False   True  False
        2020-01-03  False  False   True
        2020-01-04  False  False  False
        2020-01-05  False  False  False
        ```
        """
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
    def generate_both(cls, shape, entry_choice_func_nb, exit_choice_func_nb,
                      entry_args=None, exit_args=None, entry_wait=1, exit_wait=1, **kwargs):
        """See `vectorbt.signals.nb.generate_enex_nb`.

        `**kwargs` will be passed to pandas constructor.

        ## Example

        Generate entry and exit signals one after another. Each column increment
        the number of ticks to wait before placing the exit signal.
        ```python-repl
        >>> @njit
        ... def entry_choice_func_nb(from_i, to_i, col, temp_idx_arr):
        ...     temp_idx_arr[0] = from_i
        ...     return temp_idx_arr[:1]  # array with one signal

        >>> @njit
        ... def exit_choice_func_nb(from_i, to_i, col, temp_idx_arr):
        ...     wait = col
        ...     temp_idx_arr[0] = from_i + wait
        ...     if temp_idx_arr[0] < to_i:
        ...         return temp_idx_arr[:1]  # array with one signal
        ...     return temp_idx_arr[:0]  # empty array

        >>> temp_idx_arr = np.empty((1,), dtype=np.int_)  # reuse memory
        >>> en, ex = pd.DataFrame.vbt.signals.generate_both(
        ...     (5, 3), entry_choice_func_nb, exit_choice_func_nb,
        ...     entry_args=(temp_idx_arr,), exit_args=(temp_idx_arr,),
        ...     index=sig.index, columns=sig.columns)
        >>> en
                        a      b      c
        2020-01-01   True   True   True
        2020-01-02  False  False  False
        2020-01-03   True  False  False
        2020-01-04  False   True  False
        2020-01-05   True  False   True
        >>> ex
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02   True  False  False
        2020-01-03  False   True  False
        2020-01-04   True  False   True
        2020-01-05  False  False  False
        ```
        """
        checks.assert_numba_func(entry_choice_func_nb)
        checks.assert_numba_func(exit_choice_func_nb)
        if entry_args is None:
            entry_args = ()
        if exit_args is None:
            exit_args = ()

        if not isinstance(shape, tuple):
            shape = (shape, 1)
        elif isinstance(shape, tuple) and len(shape) == 1:
            shape = (shape[0], 1)

        result1, result2 = nb.generate_enex_nb(
            shape,
            entry_wait, exit_wait,
            entry_choice_func_nb, entry_args,
            exit_choice_func_nb, exit_args
        )
        if cls.is_series():
            if shape[1] > 1:
                raise ValueError("Use DataFrame accessor")
            return pd.Series(result1[:, 0], **kwargs), pd.Series(result2[:, 0], **kwargs)
        return pd.DataFrame(result1, **kwargs), pd.DataFrame(result2, **kwargs)

    def generate_exits(self, exit_choice_func_nb, *args, wait=1):
        """See `vectorbt.signals.nb.generate_ex_nb`.

        ## Example

        Fill all space after signals in `sig`:
        ```python-repl
        >>> @njit
        ... def exit_choice_func_nb(from_i, to_i, col, temp_range):
        ...     return temp_range[from_i:to_i]

        >>> temp_range = np.arange(sig.shape[0])  # reuse memory
        >>> sig.vbt.signals.generate_exits(exit_choice_func_nb, temp_range)
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02   True   True  False
        2020-01-03   True  False  False
        2020-01-04   True   True   True
        2020-01-05   True  False   True
        ```
        """
        checks.assert_numba_func(exit_choice_func_nb)

        return self.wrapper.wrap(nb.generate_ex_nb(self.to_2d_array(), wait, exit_choice_func_nb, *args))

    # ############# Random ############# #

    @classmethod
    def generate_random(cls, shape, n=None, prob=None, seed=None, **kwargs):
        """Generate signals randomly.

        If `n` is set, see `vectorbt.signals.nb.generate_rand_nb`.
        If `prob` is set, see `vectorbt.signals.nb.generate_rand_by_prob_nb`.

        `n` should be either a scalar or an array that will be broadcast to the number of columns.
        `prob` should be either a single number or an array that will be broadcast to match `shape`.
        `**kwargs` will be passed to pandas constructor.

        ## Example

        For each column, generate a variable number of signals:
        ```python-repl
        >>> pd.DataFrame.vbt.signals.generate_random((5, 3), n=[0, 1, 2],
        ...     seed=42, index=sig.index, columns=sig.columns)
                        a      b      c
        2020-01-01  False  False   True
        2020-01-02  False  False   True
        2020-01-03  False  False  False
        2020-01-04  False   True  False
        2020-01-05  False  False  False
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
        ```
        """
        flex_2d = True
        if not isinstance(shape, tuple):
            flex_2d = False
            shape = (shape, 1)
        elif isinstance(shape, tuple) and len(shape) == 1:
            flex_2d = False
            shape = (shape[0], 1)

        if n is not None:
            n = np.broadcast_to(n, shape[1])
            result = nb.generate_rand_nb(shape, n, seed=seed)
        elif prob is not None:
            prob = np.broadcast_to(prob, shape)
            result = nb.generate_rand_by_prob_nb(shape, prob, flex_2d, seed=seed)
        else:
            raise ValueError("At least n or prob should be set")

        if cls.is_series():
            if shape[1] > 1:
                raise ValueError("Use DataFrame accessor")
            return pd.Series(result[:, 0], **kwargs)
        return pd.DataFrame(result, **kwargs)

    # ############# Exits ############# #

    @classmethod
    def generate_random_both(cls, shape, n=None, entry_prob=None, exit_prob=None, seed=None,
                             entry_wait=1, exit_wait=1, **kwargs):
        """Generate entry and exit signals randomly and iteratively.

        If `n` is set, see `vectorbt.signals.nb.generate_rand_enex_nb`.
        If `entry_prob` and `exit_prob` are set, see `vectorbt.signals.nb.generate_rand_enex_by_prob_nb`.

        For arguments, see `Signals_Accessor.generate_random`.

        ## Example

        For each column, generate two entries and exits randomly:
        ```python-repl
        >>> en, ex = pd.DataFrame.vbt.signals.generate_random_both(
        ...     (5, 3), n=2, seed=42, index=sig.index, columns=sig.columns)
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
        >>> en, ex = pd.DataFrame.vbt.signals.generate_random_both(
        ...     (5, 3), entry_prob=0.5, exit_prob=1.,
        ...     seed=42, index=sig.index, columns=sig.columns)
        >>> en
                        a      b      c
        2020-01-01   True   True   True
        2020-01-02  False  False  False
        2020-01-03  False  False  False
        2020-01-04  False  False   True
        2020-01-05   True  False  False
        >>> ex
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02   True   True  False
        2020-01-03  False  False   True
        2020-01-04  False   True  False
        2020-01-05   True  False   True
        ```
        """
        flex_2d = True
        if not isinstance(shape, tuple):
            flex_2d = False
            shape = (shape, 1)
        elif isinstance(shape, tuple) and len(shape) == 1:
            flex_2d = False
            shape = (shape[0], 1)

        if n is not None:
            n = np.broadcast_to(n, shape[1])
            entries, exits = nb.generate_rand_enex_nb(shape, n, entry_wait, exit_wait, seed=seed)
        elif entry_prob is not None and exit_prob is not None:
            entry_prob = np.broadcast_to(entry_prob, shape)
            exit_prob = np.broadcast_to(exit_prob, shape)
            entries, exits = nb.generate_rand_enex_by_prob_nb(
                shape, entry_prob, exit_prob, entry_wait, exit_wait, flex_2d, seed=seed)
        else:
            raise ValueError("At least n, or entry_prob and exit_prob should be set")

        if cls.is_series():
            if shape[1] > 1:
                raise ValueError("Use DataFrame accessor")
            return pd.Series(entries[:, 0], **kwargs), pd.Series(exits[:, 0], **kwargs)
        return pd.DataFrame(entries, **kwargs), pd.DataFrame(exits, **kwargs)

    def generate_random_exits(self, prob=None, seed=None, wait=1):
        """Generate exit signals randomly.

        If `prob` is None, see `vectorbt.signals.nb.generate_rand_ex_nb`.
        Otherwise, see `vectorbt.signals.nb.generate_rand_ex_by_prob_nb`.

        ## Example

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
        ```
        """
        if prob is not None:
            obj, prob = reshape_fns.broadcast(self._obj, prob, keep_raw=[False, True])
            return obj.vbt.wrapper.wrap(nb.generate_rand_ex_by_prob_nb(
                obj.vbt.to_2d_array(), prob, wait, obj.ndim == 2, seed=seed))
        return self.wrapper.wrap(nb.generate_rand_ex_nb(self.to_2d_array(), wait, seed=seed))

    def generate_stop_exits(self, ts, stop, trailing=False, entry_wait=1, exit_wait=1,
                            first=True, iteratively=False, broadcast_kwargs=None):
        """Generate exits based on when `ts` hits the stop.

        If `iteratively` is True, see `vectorbt.signals.nb.generate_stop_ex_iter_nb`.
        Otherwise, see `vectorbt.signals.nb.generate_stop_ex_nb`.

        Arguments `entries`, `ts` and `stop` will be broadcast using
        `vectorbt.base.reshape_fns.broadcast` with `broadcast_kwargs`.

        For arguments, see `vectorbt.signals.nb.stop_choice_nb`.

        ## Example

        ```python-repl
        >>> ts = pd.Series([1, 2, 3, 2, 1])

        >>> # stop loss
        >>> sig.vbt.signals.generate_stop_exits(ts, -0.1)
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02  False  False  False
        2020-01-03  False  False  False
        2020-01-04  False   True   True
        2020-01-05  False  False  False

        >>> # trailing stop loss
        >>> sig.vbt.signals.generate_stop_exits(ts, -0.1, trailing=True)
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02  False  False  False
        2020-01-03  False  False  False
        2020-01-04   True   True   True
        2020-01-05  False  False  False
        ```
        """
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        entries = self._obj

        keep_raw = (False, True, True)
        broadcast_kwargs = merge_dicts(dict(require_kwargs=dict(requirements='W')), broadcast_kwargs)
        entries, ts, stop = reshape_fns.broadcast(entries, ts, stop, **broadcast_kwargs, keep_raw=keep_raw)

        # Perform generation
        if iteratively:
            new_entries, exits = nb.generate_stop_ex_iter_nb(
                entries.vbt.to_2d_array(), ts, stop, trailing, entry_wait, exit_wait, entries.ndim == 2)
            return entries.vbt.wrapper.wrap(new_entries), entries.vbt.wrapper.wrap(exits)
        else:
            exits = nb.generate_stop_ex_nb(
                entries.vbt.to_2d_array(), ts, stop, trailing, exit_wait, first, entries.ndim == 2)
            return entries.vbt.wrapper.wrap(exits)

    def generate_adv_stop_exits(self, open, high=None, low=None, close=None, is_open_safe=True,
                                out_dict=None, sl_stop=0., ts_stop=0., tp_stop=0., entry_wait=1,
                                exit_wait=1, first=True, iteratively=False, broadcast_kwargs=None):
        """Generate exits based on when price hits (trailing) stop loss or take profit.

        If any of `high`, `low` or `close` is None, it will be set to `open`.

        Use `out_dict` as a dict to pass `hit_price` and `stop_type` arrays. You can also
        set `out_dict` to {} to produce these arrays automatically and still have access to them.

        If `iteratively` is True, see `vectorbt.signals.nb.generate_adv_stop_ex_iter_nb`.
        Otherwise, see `vectorbt.signals.nb.generate_adv_stop_ex_nb`.

        All array-like arguments including stops and `out_dict` will be broadcast using
        `vectorbt.base.reshape_fns.broadcast` with `broadcast_kwargs`.

        For arguments, see `vectorbt.signals.nb.adv_stop_choice_nb`.

        ## Example

        ```python-repl
        >>> from vectorbt.signals.enums import StopType

        >>> price = pd.DataFrame({
        ...     'open': [10, 11, 12, 11, 10],
        ...     'high': [11, 12, 13, 12, 11],
        ...     'low': [9, 10, 11, 10, 9],
        ...     'close': [10, 11, 12, 11, 10]
        ... })
        >>> out_dict = {}
        >>> exits = sig.vbt.signals.generate_adv_stop_exits(
        ...     price['open'], price['high'], price['low'], price['close'],
        ...     out_dict=out_dict, sl_stop=0.2, ts_stop=0.2,
        ...     tp_stop=0.2
        ... )
        >>> out_dict['hit_price'][~exits] = np.nan
        >>> out_dict['stop_type'][~exits] = -1

        >>> exits
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02   True   True  False
        2020-01-03  False  False  False
        2020-01-04  False  False  False
        2020-01-05  False  False   True

        >>> out_dict['hit_price']
                       a     b    c
        2020-01-01   NaN   NaN  NaN
        2020-01-02  12.0  12.0  NaN
        2020-01-03   NaN   NaN  NaN
        2020-01-04   NaN   NaN  NaN
        2020-01-05   NaN   NaN  9.6

        >>> out_dict['stop_type'].applymap(
        ...     lambda x: StopType._fields[x] if x in StopType else '')
                             a           b         c
        2020-01-01
        2020-01-02  TakeProfit  TakeProfit
        2020-01-03
        2020-01-04
        2020-01-05                          StopLoss
        ```
        """
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        entries = self._obj
        if high is None:
            high = open
        if low is None:
            low = open
        if close is None:
            close = open
        if out_dict is None:
            out_dict = {}
        hit_price_out = out_dict.get('hit_price', None)
        stop_type_out = out_dict.get('stop_type', None)
        out_args = ()
        if hit_price_out is not None:
            out_args += (hit_price_out,)
        if stop_type_out is not None:
            out_args += (stop_type_out,)

        keep_raw = (False, True, True, True, True, True, True, True) + (False,) * len(out_args)
        broadcast_kwargs = merge_dicts(dict(require_kwargs=dict(requirements='W')), broadcast_kwargs)
        entries, open, high, low, close, sl_stop, ts_stop, tp_stop, *out_args = reshape_fns.broadcast(
            entries, open, high, low, close, sl_stop, ts_stop, tp_stop, *out_args,
            **broadcast_kwargs, keep_raw=keep_raw)
        if hit_price_out is None:
            hit_price_out = np.empty_like(entries, dtype=np.float_)
        else:
            hit_price_out = out_args[0]
            if checks.is_pandas(hit_price_out):
                hit_price_out = hit_price_out.vbt.to_2d_array()
            out_args = out_args[1:]
        if stop_type_out is None:
            stop_type_out = np.empty_like(entries, dtype=np.int_)
        else:
            stop_type_out = out_args[0]
            if checks.is_pandas(stop_type_out):
                stop_type_out = stop_type_out.vbt.to_2d_array()

        # Perform generation
        if iteratively:
            new_entries, exits = nb.generate_adv_stop_ex_iter_nb(
                entries.vbt.to_2d_array(), open, high, low, close, hit_price_out,
                stop_type_out, sl_stop, ts_stop, tp_stop, is_open_safe, entry_wait,
                exit_wait, first, entries.ndim == 2)
            out_dict['hit_price'] = entries.vbt.wrapper.wrap(hit_price_out)
            out_dict['stop_type'] = entries.vbt.wrapper.wrap(stop_type_out)
            return entries.vbt.wrapper.wrap(new_entries), entries.vbt.wrapper.wrap(exits)
        else:
            exits = nb.generate_adv_stop_ex_nb(
                entries.vbt.to_2d_array(), open, high, low, close, hit_price_out,
                stop_type_out, sl_stop, ts_stop, tp_stop, is_open_safe, exit_wait,
                first, entries.ndim == 2)
            out_dict['hit_price'] = entries.vbt.wrapper.wrap(hit_price_out)
            out_dict['stop_type'] = entries.vbt.wrapper.wrap(stop_type_out)
            return entries.vbt.wrapper.wrap(exits)

    # ############# Map and reduce ############# #

    def map_reduce_between(self, other=None, map_func_nb=None, map_args=None,
                           reduce_func_nb=None, reduce_args=None, broadcast_kwargs=None):
        """See `vectorbt.signals.nb.map_reduce_between_nb`.

        If `other` specified, see `vectorbt.signals.nb.map_reduce_between_two_nb`.
        Both will be broadcast using `vectorbt.base.reshape_fns.broadcast`
        with `broadcast_kwargs`.

        Note that `map_args` and `reduce_args` won't be broadcast.

        ## Example

        Get average distance between signals in `sig`:
        ```python-repl
        >>> distance_map_nb = njit(lambda from_i, to_i, col: to_i - from_i)
        >>> mean_reduce_nb = njit(lambda col, a: np.nanmean(a))

        >>> sig.vbt.signals.map_reduce_between(
        ...     map_func_nb=distance_map_nb,
        ...     reduce_func_nb=mean_reduce_nb)
        a    NaN
        b    2.0
        c    1.0
        dtype: float64
        ```
        """
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        checks.assert_not_none(map_func_nb)
        checks.assert_not_none(reduce_func_nb)
        checks.assert_numba_func(map_func_nb)
        checks.assert_numba_func(reduce_func_nb)
        if map_args is None:
            map_args = ()
        if reduce_args is None:
            reduce_args = ()

        if other is None:
            # One input array
            result = nb.map_reduce_between_nb(
                self.to_2d_array(),
                map_func_nb, map_args,
                reduce_func_nb, reduce_args
            )
            if isinstance(self._obj, pd.Series):
                return result[0]
            return pd.Series(result, index=self.wrapper.columns)
        else:
            # Two input arrays
            obj, other = reshape_fns.broadcast(self._obj, other, **broadcast_kwargs)
            checks.assert_dtype(other, np.bool)
            result = nb.map_reduce_between_two_nb(
                obj.vbt.to_2d_array(),
                other.vbt.to_2d_array(),
                map_func_nb, map_args,
                reduce_func_nb, reduce_args
            )
            return obj.vbt.wrapper.wrap_reduced(result)

    def map_reduce_partitions(self, map_func_nb=None, map_args=None,
                              reduce_func_nb=None, reduce_args=None):
        """See `vectorbt.signals.nb.map_reduce_partitions_nb`.

        ## Example

        Get average length of each partition in `sig`:
        ```python-repl
        >>> distance_map_nb = njit(lambda from_i, to_i, col: to_i - from_i)
        >>> mean_reduce_nb = njit(lambda col, a: np.nanmean(a))

        >>> sig.vbt.signals.map_reduce_partitions(
        ...     map_func_nb=distance_map_nb,
        ...     reduce_func_nb=mean_reduce_nb)
        a    1.0
        b    1.0
        c    3.0
        dtype: float64
        ```
        """
        checks.assert_not_none(map_func_nb)
        checks.assert_not_none(reduce_func_nb)
        checks.assert_numba_func(map_func_nb)
        checks.assert_numba_func(reduce_func_nb)
        if map_args is None:
            map_args = ()
        if reduce_args is None:
            reduce_args = ()

        result = nb.map_reduce_partitions_nb(
            self.to_2d_array(),
            map_func_nb, map_args,
            reduce_func_nb, reduce_args
        )
        return self.wrapper.wrap_reduced(result)

    def num_signals(self):
        """Sum up True values."""
        return self.sum()

    def avg_distance(self, to=None, **kwargs):
        """Calculate the average distance between True values in `self` and optionally `to`.

        See `Signals_Accessor.map_reduce_between`."""
        return self.map_reduce_between(
            other=to,
            map_func_nb=nb.distance_map_nb,
            reduce_func_nb=nb.mean_reduce_nb,
            **kwargs
        )

    # ############# Ranking ############# #

    def rank(self, reset_by=None, after_false=False, allow_gaps=False, broadcast_kwargs=None):
        """See `vectorbt.signals.nb.rank_nb`.

        ## Example

        Rank each True value in each partition in `sig`:
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
        ```
        """
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
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
        return obj.vbt.wrapper.wrap(ranked)

    def rank_partitions(self, reset_by=None, after_false=False, broadcast_kwargs=None):
        """See `vectorbt.signals.nb.rank_partitions_nb`.

        ## Example

        Rank each partition of True values in `sig`:
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
        ```
        """
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if reset_by is not None:
            obj, reset_by = reshape_fns.broadcast(self._obj, reset_by, **broadcast_kwargs)
            reset_by = reset_by.vbt.to_2d_array()
        else:
            obj = self._obj
        ranked = nb.rank_partitions_nb(
            obj.vbt.to_2d_array(),
            reset_by=reset_by,
            after_false=after_false)
        return obj.vbt.wrapper.wrap(ranked)

    def first(self, **kwargs):
        """`vectorbt.signals.nb.rank_nb` == 1."""
        return self.wrapper.wrap(self.rank(**kwargs).values == 1)

    def nst(self, n, **kwargs):
        """`vectorbt.signals.nb.rank_nb` == n."""
        return self.wrapper.wrap(self.rank(**kwargs).values == n)

    def from_nst(self, n, **kwargs):
        """`vectorbt.signals.nb.rank_nb` >= n."""
        return self.wrapper.wrap(self.rank(**kwargs).values >= n)

    # ############# Logical operations ############# #

    def AND(self, *others, **kwargs):
        """Combine with each in `*others` using logical AND.

        See `vectorbt.base.accessors.Base_Accessor.combine_with_multiple`.

        """
        return self.combine_with_multiple(others, combine_func=np.logical_and, **kwargs)

    def OR(self, *others, **kwargs):
        """Combine with each in `*others` using logical OR.

        See `vectorbt.base.accessors.Base_Accessor.combine_with_multiple`.

        ## Example

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
        ```
        """
        return self.combine_with_multiple(others, combine_func=np.logical_or, **kwargs)

    def XOR(self, *others, **kwargs):
        """Combine with each in `*others` using logical XOR.

        See `vectorbt.base.accessors.Base_Accessor.combine_with_multiple`."""
        return self.combine_with_multiple(others, combine_func=np.logical_xor, **kwargs)


@register_series_accessor('signals')
class Signals_SRAccessor(Signals_Accessor, Generic_SRAccessor):
    """Accessor on top of signal series. For Series only.

    Accessible through `pd.Series.vbt.signals`."""

    def __init__(self, obj, **kwargs):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        Generic_SRAccessor.__init__(self, obj, **kwargs)
        Signals_Accessor.__init__(self, obj, **kwargs)

    def plot(self, name=None, trace_kwargs=None, row=None, col=None, yref='y',
             fig=None, **layout_kwargs):  # pragma: no cover
        """Plot Series as a line.

        Args:
            name (str): Name of the signals.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter`.
            row (int): Row position.
            col (int): Column position.
            yref (str): Y coordinate axis.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> sig['a'].vbt.signals.plot()
        ```

        ![](/vectorbt/docs/img/signals_sr_plot.png)
        """
        if trace_kwargs is None:
            trace_kwargs = {}
        # Set up figure
        if fig is None:
            fig = CustomFigureWidget()
        default_layout = dict()
        default_layout['yaxis' + yref[1:]] = dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['false', 'true']
        )
        fig.update_layout(**default_layout)
        fig.update_layout(**layout_kwargs)
        if name is None:
            if 'name' in trace_kwargs:
                name = trace_kwargs.pop('name')
            else:
                name = self._obj.name
        if name is not None:
            name = str(name)

        scatter = go.Scatter(
            x=self.wrapper.index,
            y=self._obj.values,
            mode='lines',
            name=name,
            showlegend=name is not None
        )
        scatter.update(**trace_kwargs)
        fig.add_trace(scatter, row=row, col=col)

        return fig

    def plot_as_markers(self, y=None, name=None, trace_kwargs=None, row=None, col=None,
                        fig=None, **layout_kwargs):  # pragma: no cover
        """Plot Series as markers.

        Args:
            y (array_like): Y-axis values to plot markers on.

                !!! note
                    Doesn't plot `y`.

            name (str): Name of the signals.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter`.
            row (int): Row position.
            col (int): Column position.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> ts = pd.Series([1, 2, 3, 2, 1], index=sig.index)
        >>> fig = ts.vbt.plot()
        >>> sig['b'].vbt.signals.plot_as_entry_markers(y=ts, fig=fig)
        >>> (~sig['b']).vbt.signals.plot_as_exit_markers(y=ts, fig=fig)
        ```

        ![](/vectorbt/docs/img/signals_plot_as_markers.png)
        """
        from vectorbt.settings import contrast_color_schema

        if trace_kwargs is None:
            trace_kwargs = {}

        if fig is None:
            fig = CustomFigureWidget()
        fig.update_layout(**layout_kwargs)
        if name is None:
            if 'name' in trace_kwargs:
                name = trace_kwargs.pop('name')
            else:
                name = self._obj.name
        if name is not None:
            name = str(name)

        # Plot markers
        _y = 1 if y is None else y
        scatter = go.Scatter(
            x=self.wrapper.index,
            y=np.where(self._obj, _y, np.nan),
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
            name=name,
            showlegend=name is not None
        )
        scatter.update(**trace_kwargs)
        fig.add_trace(scatter, row=row, col=col)
        return fig

    def plot_as_entry_markers(self, *args, name='Entry', trace_kwargs=None, **kwargs):  # pragma: no cover
        """Plot signals as entry markers.

        See `Signals_SRAccessor.plot_as_markers`."""
        from vectorbt.settings import contrast_color_schema

        if trace_kwargs is None:
            trace_kwargs = {}
        trace_kwargs = merge_dicts(dict(
            marker=dict(
                symbol='triangle-up',
                color=contrast_color_schema['green'],
                size=8,
                line=dict(
                    width=1,
                    color=adjust_lightness(contrast_color_schema['green'])
                )
            )
        ), trace_kwargs)
        return self.plot_as_markers(*args, name=name, trace_kwargs=trace_kwargs, **kwargs)

    def plot_as_exit_markers(self, *args, name='Exit', trace_kwargs=None, **kwargs):  # pragma: no cover
        """Plot signals as exit markers.

        See `Signals_SRAccessor.plot_as_markers`."""
        from vectorbt.settings import contrast_color_schema

        if trace_kwargs is None:
            trace_kwargs = {}
        trace_kwargs = merge_dicts(dict(
            marker=dict(
                symbol='triangle-down',
                color=contrast_color_schema['red'],
                size=8,
                line=dict(
                    width=1,
                    color=adjust_lightness(contrast_color_schema['red'])
                )
            )
        ), trace_kwargs)
        return self.plot_as_markers(*args, name=name, trace_kwargs=trace_kwargs, **kwargs)


@register_dataframe_accessor('signals')
class Signals_DFAccessor(Signals_Accessor, Generic_DFAccessor):
    """Accessor on top of signal series. For DataFrames only.

    Accessible through `pd.DataFrame.vbt.signals`."""

    def __init__(self, obj, **kwargs):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        Generic_DFAccessor.__init__(self, obj, **kwargs)
        Signals_Accessor.__init__(self, obj, **kwargs)

    def plot(self, trace_kwargs=None, fig=None, **kwargs):  # pragma: no cover
        """Plot each column in DataFrame as a line.

        Args:
            trace_kwargs (dict or list of dict): Keyword arguments passed to each `plotly.graph_objects.Scatter`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **kwargs: Keyword arguments passed to `Signals_SRAccessor.plot`.

        ## Example

        ```python-repl
        >>> sig[['a', 'c']].vbt.signals.plot()
        ```

        ![](/vectorbt/docs/img/signals_df_plot.png)
        """
        if trace_kwargs is None:
            trace_kwargs = {}
        for col in range(self._obj.shape[1]):
            fig = self._obj.iloc[:, col].vbt.signals.plot(
                trace_kwargs=trace_kwargs,
                fig=fig,
                **kwargs
            )

        return fig
