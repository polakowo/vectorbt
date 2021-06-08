"""Custom pandas accessors.

Methods can be accessed as follows:

* `SignalsSRAccessor` -> `pd.Series.vbt.signals.*`
* `SignalsDFAccessor` -> `pd.DataFrame.vbt.signals.*`

```python-repl
>>> import pandas as pd
>>> import vectorbt as vbt

>>> # vectorbt.signals.accessors.SignalsAccessor.rank
>>> pd.Series([False, True, True, True, False]).vbt.signals.rank()
0    0
1    1
2    2
3    3
4    0
dtype: int64
```

The accessors extend `vectorbt.generic.accessors`.

!!! note
    The underlying Series/DataFrame should already be a signal series.

    Input arrays should be `np.bool_`.

Run for the examples below:
    
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

from vectorbt import _typing as tp
from vectorbt.root_accessors import register_dataframe_accessor, register_series_accessor
from vectorbt.utils import checks
from vectorbt.utils.decorators import class_or_instancemethod
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.colors import adjust_lightness
from vectorbt.base import reshape_fns
from vectorbt.base.class_helpers import add_nb_methods
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.generic.accessors import GenericAccessor, GenericSRAccessor, GenericDFAccessor
from vectorbt.generic import plotting
from vectorbt.signals import nb

MaybeSeriesFrameTupleT = tp.Union[tp.SeriesFrame, tp.Tuple[tp.SeriesFrame, tp.SeriesFrame]]


@add_nb_methods([
    (nb.fshift_nb, False),
], module_name='vectorbt.signals.nb')
class SignalsAccessor(GenericAccessor):
    """Accessor on top of signal series. For both, Series and DataFrames.

    Accessible through `pd.Series.vbt.signals` and `pd.DataFrame.vbt.signals`."""

    def __init__(self, obj: tp.SeriesFrame, **kwargs) -> None:
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        checks.assert_dtype(obj, np.bool_)

        GenericAccessor.__init__(self, obj, **kwargs)

    @classmethod
    def empty(cls, *args, fill_value: bool = False, **kwargs) -> tp.SeriesFrame:
        """`vectorbt.base.accessors.BaseAccessor.empty` with `fill_value=False`.

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
        return GenericAccessor.empty(*args, fill_value=fill_value, dtype=bool, **kwargs)

    @classmethod
    def empty_like(cls, *args, fill_value: bool = False, **kwargs) -> tp.SeriesFrame:
        """`vectorbt.base.accessors.BaseAccessor.empty_like` with `fill_value=False`.

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
        return GenericAccessor.empty_like(*args, fill_value=fill_value, dtype=bool, **kwargs)

    # ############# Generation ############# #

    @classmethod
    def generate(cls, shape: tp.RelaxedShape, choice_func_nb: tp.SignalChoiceFunc, *args, **kwargs) -> tp.SeriesFrame:
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
    def generate_both(cls, shape: tp.RelaxedShape, entry_choice_func_nb: tp.SignalChoiceFunc,
                      exit_choice_func_nb: tp.SignalChoiceFunc, entry_args: tp.Optional[tp.Args] = None,
                      exit_args: tp.Optional[tp.Args] = None, entry_wait: int = 1, exit_wait: int = 1,
                      **kwargs) -> tp.Tuple[tp.SeriesFrame, tp.SeriesFrame]:
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

    def generate_exits(self, exit_choice_func_nb: tp.SignalChoiceFunc, *args, wait: int = 1,
                       wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
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

        exits = nb.generate_ex_nb(self.to_2d_array(), wait, exit_choice_func_nb, *args)
        return self.wrapper.wrap(exits, **merge_dicts({}, wrap_kwargs))

    # ############# Filtering ############# #

    @class_or_instancemethod
    def clean(self_or_cls, *args, entry_first: bool = True, broadcast_kwargs: tp.KwargsLike = None,
              wrap_kwargs: tp.KwargsLike = None) -> MaybeSeriesFrameTupleT:
        """Clean signals.

        If one array passed, see `SignalsAccessor.first`.
        If two arrays passed, entries and exits, see `vectorbt.signals.nb.clean_enex_nb`."""
        if not isinstance(self_or_cls, type):
            args = (self_or_cls._obj, *args)
        if len(args) == 1:
            obj = args[0]
            if not isinstance(obj, (pd.Series, pd.DataFrame)):
                wrapper = ArrayWrapper.from_shape(np.asarray(obj).shape)
                obj = wrapper.wrap(obj)
            return obj.vbt.signals.first(wrap_kwargs=wrap_kwargs)
        elif len(args) == 2:
            if broadcast_kwargs is None:
                broadcast_kwargs = {}
            entries, exits = reshape_fns.broadcast(*args, **broadcast_kwargs)
            entries_out, exits_out = nb.clean_enex_nb(
                entries.vbt.to_2d_array(),
                exits.vbt.to_2d_array(),
                entry_first
            )
            return (
                entries.vbt.wrapper.wrap(entries_out, **merge_dicts({}, wrap_kwargs)),
                exits.vbt.wrapper.wrap(exits_out, **merge_dicts({}, wrap_kwargs))
            )
        else:
            raise ValueError("Either one or two arrays must be passed")

    # ############# Random ############# #

    @classmethod
    def generate_random(cls, shape: tp.RelaxedShape, n: tp.Optional[tp.ArrayLike] = None,
                        prob: tp.Optional[tp.ArrayLike] = None, seed: tp.Optional[int] = None,
                        **kwargs) -> tp.SeriesFrame:
        """Generate signals randomly.

        If `n` is set, see `vectorbt.signals.nb.generate_rand_nb`.
        If `prob` is set, see `vectorbt.signals.nb.generate_rand_by_prob_nb`.

        `n` should be either a scalar or an array that will broadcast to the number of columns.
        `prob` should be either a single number or an array that will broadcast to match `shape`.
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

        if n is not None and prob is not None:
            raise ValueError("Either n or prob should be set")
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
    def generate_random_both(cls, shape: tp.RelaxedShape, n: tp.Optional[tp.ArrayLike] = None,
                             entry_prob: tp.Optional[tp.ArrayLike] = None, exit_prob: tp.Optional[tp.ArrayLike] = None,
                             seed: tp.Optional[int] = None, entry_wait: int = 1, exit_wait: int = 1,
                             **kwargs) -> tp.Tuple[tp.SeriesFrame, tp.SeriesFrame]:
        """Generate entry and exit signals randomly and iteratively.

        If `n` is set, see `vectorbt.signals.nb.generate_rand_enex_nb`.
        If `entry_prob` and `exit_prob` are set, see `vectorbt.signals.nb.generate_rand_enex_by_prob_nb`.

        For arguments, see `SignalsAccessor.generate_random`.

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

    def generate_random_exits(self, prob: tp.Optional[tp.ArrayLike] = None, seed: tp.Optional[int] = None,
                              wait: int = 1, wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
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
            exits = nb.generate_rand_ex_by_prob_nb(obj.vbt.to_2d_array(), prob, wait, obj.ndim == 2, seed=seed)
            return obj.vbt.wrapper.wrap(exits, **merge_dicts({}, wrap_kwargs))
        exits = nb.generate_rand_ex_nb(self.to_2d_array(), wait, seed=seed)
        return self.wrapper.wrap(exits, **merge_dicts({}, wrap_kwargs))

    def generate_stop_exits(self,
                            ts: tp.ArrayLike,
                            stop: tp.ArrayLike,
                            trailing: tp.ArrayLike = False,
                            entry_wait: int = 1,
                            exit_wait: int = 1,
                            first: bool = True,
                            iteratively: bool = False,
                            broadcast_kwargs: tp.KwargsLike = None,
                            wrap_kwargs: tp.KwargsLike = None) -> MaybeSeriesFrameTupleT:
        """Generate exits based on when `ts` hits the stop.

        For arguments, see `vectorbt.signals.nb.stop_choice_nb`.
        If `iteratively` is True, see `vectorbt.signals.nb.generate_stop_ex_iter_nb`.
        Otherwise, see `vectorbt.signals.nb.generate_stop_ex_nb`.

        Arguments `entries`, `ts` and `stop` will broadcast using `vectorbt.base.reshape_fns.broadcast`
        with `broadcast_kwargs`.

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

        keep_raw = (False, True, True, True)
        broadcast_kwargs = merge_dicts(dict(require_kwargs=dict(requirements='W')), broadcast_kwargs)
        entries, ts, stop, trailing = reshape_fns.broadcast(
            entries, ts, stop, trailing, **broadcast_kwargs, keep_raw=keep_raw)

        # Perform generation
        if iteratively:
            new_entries, exits = nb.generate_stop_ex_iter_nb(
                entries.vbt.to_2d_array(), ts, stop, trailing, entry_wait, exit_wait, entries.ndim == 2)
            return entries.vbt.wrapper.wrap(new_entries, **merge_dicts({}, wrap_kwargs)), \
                   entries.vbt.wrapper.wrap(exits, **merge_dicts({}, wrap_kwargs))
        else:
            exits = nb.generate_stop_ex_nb(
                entries.vbt.to_2d_array(), ts, stop, trailing, exit_wait, first, entries.ndim == 2)
            return entries.vbt.wrapper.wrap(exits, **merge_dicts({}, wrap_kwargs))

    def generate_ohlc_stop_exits(self,
                                 open: tp.ArrayLike,
                                 high: tp.Optional[tp.ArrayLike] = None,
                                 low: tp.Optional[tp.ArrayLike] = None,
                                 close: tp.Optional[tp.ArrayLike] = None,
                                 is_open_safe: bool = True,
                                 out_dict: tp.Optional[tp.Dict[str, tp.ArrayLike]] = None,
                                 sl_stop: tp.Optional[tp.ArrayLike] = np.nan,
                                 ts_stop: tp.Optional[tp.ArrayLike] = np.nan,
                                 tp_stop: tp.Optional[tp.ArrayLike] = np.nan,
                                 entry_wait: int = 1,
                                 exit_wait: int = 1,
                                 first: bool = True,
                                 iteratively: bool = False,
                                 broadcast_kwargs: tp.KwargsLike = None,
                                 wrap_kwargs: tp.KwargsLike = None) -> MaybeSeriesFrameTupleT:
        """Generate exits based on when the price hits (trailing) stop loss or take profit.

        If any of `high`, `low` or `close` is None, it will be set to `open`.

        Use `out_dict` as a dict to pass `hit_price` and `stop_type` arrays. You can also
        set `out_dict` to {} to produce these arrays automatically and still have access to them.

        For arguments, see `vectorbt.signals.nb.ohlc_stop_choice_nb`.
        If `iteratively` is True, see `vectorbt.signals.nb.generate_ohlc_stop_ex_iter_nb`.
        Otherwise, see `vectorbt.signals.nb.generate_ohlc_stop_ex_nb`.

        All array-like arguments including stops and `out_dict` will broadcast using
        `vectorbt.base.reshape_fns.broadcast` with `broadcast_kwargs`.

        For arguments, see `vectorbt.signals.nb.ohlc_stop_choice_nb`.

        !!! note
            `open` isn't necessarily open price, but can be any entry price (even previous close).
            Stop price is calculated based solely upon `open`.

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
        >>> exits = sig.vbt.signals.generate_ohlc_stop_exits(
        ...     price['open'], price['high'], price['low'], price['close'],
        ...     out_dict=out_dict, sl_stop=0.2, ts_stop=0.2, tp_stop=0.2)
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

        >>> out_dict['stop_type'].vbt.map_enum(StopType)
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
            out_dict_passed = False
            out_dict = {}
        else:
            out_dict_passed = True
        hit_price_out = out_dict.get('hit_price', np.nan if out_dict_passed else None)
        stop_type_out = out_dict.get('stop_type', -1 if out_dict_passed else None)
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
            out_args = out_args[1:]
        if stop_type_out is None:
            stop_type_out = np.empty_like(entries, dtype=np.int_)
        else:
            stop_type_out = out_args[0]
        hit_price_out = reshape_fns.to_2d(hit_price_out, raw=True)
        stop_type_out = reshape_fns.to_2d(stop_type_out, raw=True)

        # Perform generation
        if iteratively:
            new_entries, exits = nb.generate_ohlc_stop_ex_iter_nb(
                entries.vbt.to_2d_array(), open, high, low, close, hit_price_out,
                stop_type_out, sl_stop, ts_stop, tp_stop, is_open_safe, entry_wait,
                exit_wait, first, entries.ndim == 2)
            out_dict['hit_price'] = entries.vbt.wrapper.wrap(hit_price_out, **merge_dicts({}, wrap_kwargs))
            out_dict['stop_type'] = entries.vbt.wrapper.wrap(stop_type_out, **merge_dicts({}, wrap_kwargs))
            return entries.vbt.wrapper.wrap(new_entries, **merge_dicts({}, wrap_kwargs)), \
                   entries.vbt.wrapper.wrap(exits, **merge_dicts({}, wrap_kwargs))
        else:
            exits = nb.generate_ohlc_stop_ex_nb(
                entries.vbt.to_2d_array(), open, high, low, close, hit_price_out,
                stop_type_out, sl_stop, ts_stop, tp_stop, is_open_safe, exit_wait,
                first, entries.ndim == 2)
            out_dict['hit_price'] = entries.vbt.wrapper.wrap(hit_price_out, **merge_dicts({}, wrap_kwargs))
            out_dict['stop_type'] = entries.vbt.wrapper.wrap(stop_type_out, **merge_dicts({}, wrap_kwargs))
            return entries.vbt.wrapper.wrap(exits, **merge_dicts({}, wrap_kwargs))

    # ############# Map and reduce ############# #

    def map_reduce_between(self,
                           other: tp.Optional[tp.ArrayLike] = None,
                           map_func_nb: tp.Optional[tp.SignalMapFunc] = None,
                           map_args: tp.Optional[tp.Args] = None,
                           reduce_func_nb: tp.Optional[tp.SignalReduceFunc] = None,
                           reduce_args: tp.Optional[tp.Args] = None,
                           broadcast_kwargs: tp.KwargsLike = None,
                           wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """See `vectorbt.signals.nb.map_reduce_between_nb`.

        If `other` specified, see `vectorbt.signals.nb.map_reduce_between_two_nb`.
        Both will broadcast using `vectorbt.base.reshape_fns.broadcast`
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

        wrap_kwargs = merge_dicts(dict(name_or_index='map_reduce_between'), wrap_kwargs)
        if other is None:
            # One input array
            result = nb.map_reduce_between_nb(
                self.to_2d_array(),
                map_func_nb, map_args,
                reduce_func_nb, reduce_args
            )
            return self.wrapper.wrap_reduced(result, **wrap_kwargs)
        else:
            # Two input arrays
            obj, other = reshape_fns.broadcast(self._obj, other, **broadcast_kwargs)
            checks.assert_dtype(other, np.bool_)
            result = nb.map_reduce_between_two_nb(
                obj.vbt.to_2d_array(),
                other.vbt.to_2d_array(),
                map_func_nb, map_args,
                reduce_func_nb, reduce_args
            )
            return obj.vbt.wrapper.wrap_reduced(result, **wrap_kwargs)

    def map_reduce_partitions(self,
                              map_func_nb: tp.Optional[tp.SignalMapFunc] = None,
                              map_args: tp.Optional[tp.Args] = None,
                              reduce_func_nb: tp.Optional[tp.SignalReduceFunc] = None,
                              reduce_args: tp.Optional[tp.Args] = None,
                              wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
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
        wrap_kwargs = merge_dicts(dict(name_or_index='map_reduce_partitions'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def num_signals(self, **kwargs) -> tp.MaybeSeries:
        """Sum up True values."""
        kwargs = merge_dicts(dict(wrap_kwargs=dict(name_or_index='num_signals')), kwargs)
        return self.sum(**kwargs)

    def avg_distance(self, to=None, **kwargs) -> tp.MaybeSeries:
        """Calculate the average distance between True values in `self` and optionally `to`.

        See `SignalsAccessor.map_reduce_between`."""
        kwargs = merge_dicts(dict(wrap_kwargs=dict(name_or_index='avg_distance')), kwargs)
        return self.map_reduce_between(
            other=to, map_func_nb=nb.distance_map_nb,
            reduce_func_nb=nb.mean_reduce_nb, **kwargs)

    # ############# Ranking ############# #

    def rank(self, reset_by: tp.Optional[tp.ArrayLike] = None, after_false: bool = False,
             allow_gaps: bool = False, broadcast_kwargs: tp.KwargsLike = None,
             wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
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
        return obj.vbt.wrapper.wrap(ranked, **merge_dicts({}, wrap_kwargs))

    def rank_partitions(self, reset_by: tp.Optional[tp.ArrayLike] = None, after_false: bool = False,
                        broadcast_kwargs: tp.KwargsLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
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
        return obj.vbt.wrapper.wrap(ranked, **merge_dicts({}, wrap_kwargs))

    def first(self, wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.SeriesFrame:
        """`vectorbt.signals.nb.rank_nb` == 1."""
        return self.wrapper.wrap(self.rank(**kwargs).values == 1, **merge_dicts({}, wrap_kwargs))

    def nst(self, n, wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.SeriesFrame:
        """`vectorbt.signals.nb.rank_nb` == n."""
        return self.wrapper.wrap(self.rank(**kwargs).values == n, **merge_dicts({}, wrap_kwargs))

    def from_nst(self, n, wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.SeriesFrame:
        """`vectorbt.signals.nb.rank_nb` >= n."""
        return self.wrapper.wrap(self.rank(**kwargs).values >= n, **merge_dicts({}, wrap_kwargs))

    # ############# Logical operations ############# #

    def AND(self, other: tp.ArrayLike, **kwargs) -> tp.SeriesFrame:
        """Combine with `other` using logical AND.

        See `vectorbt.base.accessors.BaseAccessor.combine`.

        """
        return self.combine(other, combine_func=np.logical_and, **kwargs)

    def OR(self, other: tp.ArrayLike, **kwargs) -> tp.SeriesFrame:
        """Combine with `other` using logical OR.

        See `vectorbt.base.accessors.BaseAccessor.combine`.

        ## Example

        Perform two OR operations and concatenate them:
        ```python-repl
        >>> ts = pd.Series([1, 2, 3, 2, 1])
        >>> sig.vbt.signals.OR([ts > 1, ts > 2], concat=True, keys=['>1', '>2'])
                                    >1                   >2
                        a     b      c      a      b      c
        2020-01-01   True  True   True   True   True   True
        2020-01-02   True  True   True  False  False   True
        2020-01-03   True  True   True   True   True   True
        2020-01-04   True  True   True  False  False  False
        2020-01-05  False  True  False  False   True  False
        ```
        """
        return self.combine(other, combine_func=np.logical_or, **kwargs)

    def XOR(self, other: tp.ArrayLike, **kwargs) -> tp.SeriesFrame:
        """Combine with `other` using logical XOR.

        See `vectorbt.base.accessors.BaseAccessor.combine`."""
        return self.combine(other, combine_func=np.logical_xor, **kwargs)

    def plot(self, yref: str = 'y', **kwargs) -> tp.Union[tp.BaseFigure, plotting.Scatter]:  # pragma: no cover
        """Plot signals.

        Args:
            yref (str): Y coordinate axis.
            **kwargs: Keyword arguments passed to `vectorbt.generic.accessors.GenericAccessor.lineplot`.

        ## Example

        ```python-repl
        >>> sig[['a', 'c']].vbt.signals.plot()
        ```

        ![](/vectorbt/docs/img/signals_df_plot.svg)
        """
        default_layout = dict()
        default_layout['yaxis' + yref[1:]] = dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['false', 'true']
        )
        return self._obj.vbt.lineplot(**merge_dicts(default_layout, kwargs))


@register_series_accessor('signals')
class SignalsSRAccessor(SignalsAccessor, GenericSRAccessor):
    """Accessor on top of signal series. For Series only.

    Accessible through `pd.Series.vbt.signals`."""

    def __init__(self, obj: tp.Series, **kwargs) -> None:
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        GenericSRAccessor.__init__(self, obj, **kwargs)
        SignalsAccessor.__init__(self, obj, **kwargs)

    def plot_as_markers(self, y: tp.Optional[tp.ArrayLike] = None,
                        **kwargs) -> tp.Union[tp.BaseFigure, plotting.Scatter]:  # pragma: no cover
        """Plot Series as markers.

        Args:
            y (array_like): Y-axis values to plot markers on.
            **kwargs: Keyword arguments passed to `vectorbt.generic.accessors.GenericAccessor.scatterplot`.

        ## Example

        ```python-repl
        >>> ts = pd.Series([1, 2, 3, 2, 1], index=sig.index)
        >>> fig = ts.vbt.lineplot()
        >>> sig['b'].vbt.signals.plot_as_entry_markers(y=ts, fig=fig)
        >>> (~sig['b']).vbt.signals.plot_as_exit_markers(y=ts, fig=fig)
        ```

        ![](/vectorbt/docs/img/signals_plot_as_markers.svg)
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        if y is None:
            y = pd.Series.vbt.empty_like(self._obj, 1)
        else:
            y = reshape_fns.to_pd_array(y)

        return y[self._obj].vbt.scatterplot(**merge_dicts(dict(
            trace_kwargs=dict(
                marker=dict(
                    symbol='circle',
                    color=plotting_cfg['contrast_color_schema']['blue'],
                    size=7,
                    line=dict(
                        width=1,
                        color=adjust_lightness(plotting_cfg['contrast_color_schema']['blue'])
                    )
                )
            )
        ), kwargs))

    def plot_as_entry_markers(self, y: tp.Optional[tp.ArrayLike] = None,
                              **kwargs) -> tp.Union[tp.BaseFigure, plotting.Scatter]:  # pragma: no cover
        """Plot signals as entry markers.

        See `SignalsSRAccessor.plot_as_markers`."""
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        return self.plot_as_markers(y=y, **merge_dicts(dict(
            trace_kwargs=dict(
                marker=dict(
                    symbol='triangle-up',
                    color=plotting_cfg['contrast_color_schema']['green'],
                    size=8,
                    line=dict(
                        width=1,
                        color=adjust_lightness(plotting_cfg['contrast_color_schema']['green'])
                    )
                ),
                name='Entry'
            )
        ), kwargs))

    def plot_as_exit_markers(self, y: tp.Optional[tp.ArrayLike] = None,
                             **kwargs) -> tp.Union[tp.BaseFigure, plotting.Scatter]:  # pragma: no cover
        """Plot signals as exit markers.

        See `SignalsSRAccessor.plot_as_markers`."""
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        return self.plot_as_markers(y=y, **merge_dicts(dict(
            trace_kwargs=dict(
                marker=dict(
                    symbol='triangle-down',
                    color=plotting_cfg['contrast_color_schema']['red'],
                    size=8,
                    line=dict(
                        width=1,
                        color=adjust_lightness(plotting_cfg['contrast_color_schema']['red'])
                    )
                ),
                name='Exit'
            )
        ), kwargs))


@register_dataframe_accessor('signals')
class SignalsDFAccessor(SignalsAccessor, GenericDFAccessor):
    """Accessor on top of signal series. For DataFrames only.

    Accessible through `pd.DataFrame.vbt.signals`."""

    def __init__(self, obj: tp.Frame, **kwargs) -> None:
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        GenericDFAccessor.__init__(self, obj, **kwargs)
        SignalsAccessor.__init__(self, obj, **kwargs)
