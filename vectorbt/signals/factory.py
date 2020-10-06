"""A factory for building new signal generators with ease."""

import numpy as np
from numba import njit
from numba.typed import List

from vectorbt.utils import checks
from vectorbt.base import combine_fns
from vectorbt.indicators.factory import IndicatorFactory
from vectorbt.signals.nb import generate_ex_nb, generate_enex_nb, first_choice_nb


class SignalFactory(IndicatorFactory):
    """A factory for building signal generators.

    Extends `vectorbt.indicators.factory.IndicatorFactory` with choice functions."""

    def __init__(self,
                 *args,
                 class_name='CustomSignals',
                 input_names=None,
                 exit_only=False,
                 iteratively=False,
                 **kwargs):
        if input_names is None:
            input_names = []
        if exit_only:
            if len(input_names) > 0:
                if input_names[0] != 'entries':
                    input_names = ['entries'] + input_names
            else:
                input_names = ['entries']
            output_names = ['exits']
            if iteratively:
                output_names = ['new_entries'] + output_names
        else:
            output_names = ['entries', 'exits']
        IndicatorFactory.__init__(
            self, *args,
            class_name=class_name,
            input_names=input_names,
            output_names=output_names,
            **kwargs
        )
        self.exit_only = exit_only
        self.iteratively = iteratively

    def from_choice_func(
            self,
            entry_choice_func=None,
            exit_choice_func=None,
            cache_func=None,
            entry_settings=None,
            exit_settings=None,
            cache_settings=None,
            **kwargs):
        """Build signal generator class around entry and exit choice functions.

        Each choice function takes broadcasted time series, broadcasted parameter arrays, 
        and other arguments, and returns an array of indices corresponding to signals. 
        See `vectorbt.signals.nb.generate_nb`.

        Args:
            entry_choice_func (callable): `choice_func_nb` that returns indices of entries.
            exit_choice_func (callable): `choice_func_nb` that returns indices of exits.
            cache_func (callable): A caching function to preprocess data beforehand.
                All returned objects will be passed as last arguments to choice functions.
            entry_settings (dict): Settings dictionary for `entry_choice_func`.
            exit_settings (dict): Settings dictionary for `exit_choice_func`.
            cache_settings (dict): Settings dictionary for `cache_func`.
            **kwargs: Keyword arguments passed to `IndicatorFactory.from_custom_func`.

        !!! note
            Choice functions should be Numba-compiled.

        Which arguments to pass to each function can be defined by its settings:

        Attributes:
            input_names (list of str): Input names to pass to the choice function.
                Defaults to []. Order matters. Each name must be in `input_names`.
            param_names (list of str): Parameter names to pass to the choice function.
                Defaults to []. Order matters. Each name must be in `param_names`.
            pass_first (bool): Whether to pass `first` argument to the choice function.
                Defaults to False, while `first` itself defaults to True.
                When `first` is True, function should stop once first signal is found.
            pass_temp_int (bool): Whether to pass `temp_int` argument to the choice function.
                Defaults to False, while `temp_int` is automatically generated prior to passing
                and has length `input.shape[0]`.
            pass_is_2d (bool): Whether to pass `is_2d` argument to the choice function.
                Defaults to False, while `is_2d` defaults to True since passed time series
                are always 2-dimensional. Can be used for flexible indexing.
            pass_cache (bool): Whether to pass cache from `cache_func` to the choice function.
                Defaults to False. Cache is passed unpacked.

        Also, `first`, `temp_int` and `is_2d` are passed in order defined above and can be
        overridden by the user using `entry_kwargs` and `exit_kwargs`.

        When using `run` and `run_combs` methods, you can also pass the following arguments:

        Args:
            *args: Can be used instead of `exit_kwargs` when `exit_only` is True.
            input_shape (tuple): Input shape if no input time series passed.
            entry_args (tuple): Arguments passed to the entry choice function after parameters.
            exit_args (tuple): Arguments passed to the exit choice function after parameters.
            cache_args (tuple): Arguments passed to the caching function after parameters.
            entry_kwargs (tuple): Arguments to override values from `entry_settings`.
            exit_kwargs (tuple): Arguments to override values from `exit_settings`.
            cache_kwargs (tuple): Arguments to override values from `cache_settings`.

        Example:
            Given an entry, finds the next trailing stop loss OR take profit exit. Then takes
            the next entry right after, and repeats. Returns exits along with new entries.

            ```python-repl
            >>> import pandas as pd
            >>> import numpy as np
            >>> from numba import njit
            >>> from vectorbt.signals.factory import SignalFactory
            >>> from vectorbt.signals.nb import stop_choice_nb
            >>> from vectorbt.signals.enums import StopPosition

            >>> @njit
            ... def custom_choice_func_nb(col, from_i, to_i, ts, tsl_stop, tp_stop, temp_int, is_2d):
            ...     # Find first index of trailing stop loss (TSL)
            ...     tsl_exit_idxs = stop_choice_nb(col, from_i, to_i,
            ...         ts, tsl_stop, StopPosition.ExpMax, True, temp_int, is_2d)
            ...     if len(tsl_exit_idxs) > 0:
            ...         # No need to go beyond first TSL signal
            ...         to_i = tsl_exit_idxs[0]
            ...     # Find first index of take profit (TP)
            ...     tp_exit_idxs = stop_choice_nb(col, from_i, to_i,
            ...         ts, tp_stop, StopPosition.Entry, True, temp_int, is_2d)
            ...     if len(tp_exit_idxs) > 0:  # first index is TP
            ...         return tp_exit_idxs[:1]
            ...     if len(tsl_exit_idxs) > 0:  # first index is TSL
            ...         return tsl_exit_idxs[:1]
            ...     return temp_int[:0]  # no signals

            >>> # Build signal generator
            >>> MySignals = SignalFactory(
            ...     input_names=['ts'],
            ...     param_names=['tsl_stop', 'tp_stop'],
            ...     exit_only=True,
            ...     iteratively=True
            ... ).from_choice_func(
            ...     exit_choice_func=custom_choice_func_nb,
            ...     exit_settings=dict(
            ...         input_names=['ts'],
            ...         param_names=['tsl_stop', 'tp_stop'],
            ...         pass_temp_int=True,
            ...         pass_is_2d=True,
            ...     )
            ... )

            >>> # Run signal generator
            >>> entries = pd.Series([True, True, True, True, True])
            >>> ts = pd.Series([10., 11., 12., 11., 10.])
            >>> my_sig = MySignals.run(entries, ts, [-0.1, -0.2], [0.1, 0.2])

            >>> my_sig.entries  # input entries
            custom_tsl_stop  -0.1  -0.2
            custom_tp_stop    0.1   0.2
            0                True  True
            1                True  True
            2                True  True
            3                True  True
            4                True  True
            >>> my_sig.new_entries  # output entries
            custom_tsl_stop   -0.1   -0.2
            custom_tp_stop     0.1    0.2
            0                 True   True
            1                False  False
            2                 True  False
            3                False   True
            4                False  False
            >>> my_sig.exits  # output exits
            custom_tsl_stop   -0.1   -0.2
            custom_tp_stop     0.1    0.2
            0                False  False
            1                 True  False
            2                False   True
            3                False  False
            4                 True  False
            ```
        """

        exit_only = self.exit_only
        iteratively = self.iteratively
        input_names = self.input_names
        param_names = self.param_names

        checks.assert_not_none(exit_choice_func)
        checks.assert_numba_func(exit_choice_func)
        if exit_only:
            if entry_choice_func is not None:
                raise ValueError("entry_choice_func cannot be set when exit_only=True")
            if entry_settings is not None:
                raise ValueError("entry_settings cannot be set when exit_only=True")
            if iteratively:
                entry_choice_func = first_choice_nb
                entry_settings = dict(
                    input_names=['entries']
                )
        else:
            checks.assert_not_none(entry_choice_func)
            checks.assert_numba_func(entry_choice_func)

        if entry_settings is None:
            entry_settings = {}
        if exit_settings is None:
            exit_settings = {}
        if cache_settings is None:
            cache_settings = {}

        # Get input names for each function
        def _get_func_input_names(func_settings):
            func_input_names = func_settings.get('input_names', None)
            if func_input_names is None:
                return []
            else:
                for name in func_input_names:
                    checks.assert_in(name, input_names)
            return func_input_names

        entry_input_names = _get_func_input_names(entry_settings)
        exit_input_names = _get_func_input_names(exit_settings)
        cache_input_names = _get_func_input_names(cache_settings)

        # Get param names for each function
        def _get_func_param_names(func_settings):
            func_param_names = func_settings.get('param_names', None)
            if func_param_names is None:
                return []
            else:
                for name in func_param_names:
                    checks.assert_in(name, param_names)
            return func_param_names

        entry_param_names = _get_func_param_names(entry_settings)
        exit_param_names = _get_func_param_names(exit_settings)
        cache_param_names = _get_func_param_names(cache_settings)

        if exit_only and not iteratively:
            @njit
            def apply_nb(i, entries, exit_input_list, exit_param_tuples, exit_args):
                return generate_ex_nb(entries, exit_choice_func, *exit_input_list, *exit_param_tuples[i], *exit_args)
        else:
            @njit
            def apply_nb(i, shape, entry_input_list, exit_input_list, entry_param_tuples,
                         exit_param_tuples, entry_args, exit_args):
                return generate_enex_nb(
                    shape,
                    entry_choice_func, (*entry_input_list, *entry_param_tuples[i], *entry_args),
                    exit_choice_func, (*exit_input_list, *exit_param_tuples[i], *exit_args)
                )

        def custom_func(input_list, param_list, *args, input_shape=None, entry_args=None, exit_args=None,
                        cache_args=None, entry_kwargs=None, exit_kwargs=None, cache_kwargs=None,
                        return_cache=False, use_cache=None):
            if entry_args is None:
                entry_args = ()
            if exit_args is None:
                exit_args = ()
            if cache_args is None:
                cache_args = ()
            if entry_kwargs is None:
                entry_kwargs = {}
            if exit_kwargs is None:
                exit_kwargs = {}
            if cache_kwargs is None:
                cache_kwargs = {}

            if len(input_list) == 0:
                if input_shape is None:
                    raise ValueError("Pass input_shape if no input time series passed")
            else:
                input_shape = input_list[0].shape
            if exit_only:
                if len(exit_args) > 0:
                    raise ValueError("Use *args instead of exit_args when exit_only=True")
                exit_args = args
            else:
                if len(args) > 0:
                    raise ValueError("*args can be only used when exit_only=True")

            # Distribute arguments across functions
            entry_input_list = ()
            exit_input_list = ()
            cache_input_list = ()
            entry_param_list = ()
            exit_param_list = ()
            cache_param_list = ()

            for input_name in entry_input_names:
                entry_input_list += (input_list[input_names.index(input_name)],)
            for input_name in exit_input_names:
                exit_input_list += (input_list[input_names.index(input_name)],)
            for input_name in cache_input_names:
                cache_input_list += (input_list[input_names.index(input_name)],)
            for param_name in entry_param_names:
                entry_param_list += (param_list[param_names.index(param_name)],)
            for param_name in exit_param_names:
                exit_param_list += (param_list[param_names.index(param_name)],)
            for param_name in cache_param_names:
                cache_param_list += (param_list[param_names.index(param_name)],)

            n_params = len(param_list[0]) if len(param_list) > 0 else 1
            entry_param_tuples = tuple(zip(*entry_param_list))
            if len(entry_param_tuples) == 0:
                entry_param_tuples = ((),) * n_params
            exit_param_tuples = tuple(zip(*exit_param_list))
            if len(exit_param_tuples) == 0:
                exit_param_tuples = ((),) * n_params

            def _build_more_args(func_settings, func_kwargs):
                more_args = ()
                if func_settings.get('pass_first', False):
                    first = func_kwargs.get('first', True)
                    more_args += (first,)
                if func_settings.get('pass_temp_int', False):
                    temp_int = func_kwargs.get('temp_int', None)
                    if temp_int is None:
                        temp_int = np.empty((input_shape[0],), dtype=np.int_)
                    more_args += (temp_int,)
                if func_settings.get('pass_is_2d', False):
                    is_2d = func_kwargs.get('is_2d', True)
                    if is_2d is None:
                        is_2d = len(input_shape) == 2
                    more_args += (is_2d,)
                return more_args

            entry_more_args = _build_more_args(entry_settings, entry_kwargs)
            exit_more_args = _build_more_args(exit_settings, exit_kwargs)
            cache_more_args = _build_more_args(cache_settings, cache_kwargs)

            # Caching
            cache = use_cache
            if cache is None and cache_func is not None:
                cache = cache_func(*cache_input_list, *cache_param_list, *cache_args, *cache_more_args)
            if return_cache:
                return cache
            if cache is None:
                cache = ()
            if not isinstance(cache, (tuple, list, List)):
                cache = (cache,)

            entry_cache = ()
            exit_cache = ()
            if entry_settings.get('pass_cache', False):
                entry_cache = cache
            if exit_settings.get('pass_cache', False):
                exit_cache = cache

            # Apply and concatenate
            if exit_only and not iteratively:
                return combine_fns.apply_and_concat_one_nb(
                    n_params,
                    apply_nb,
                    input_list[0],
                    exit_input_list,
                    exit_param_tuples,
                    exit_args + exit_more_args + exit_cache
                )
            else:
                return combine_fns.apply_and_concat_multiple_nb(
                    n_params,
                    apply_nb,
                    input_shape,
                    entry_input_list,
                    exit_input_list,
                    entry_param_tuples,
                    exit_param_tuples,
                    entry_args + entry_more_args + entry_cache,
                    exit_args + exit_more_args + exit_cache
                )

        return self.from_custom_func(custom_func, pass_lists=True, **kwargs)
