"""A factory for building new signal generators with ease."""

import numpy as np
from numba import njit
from numba.typed import List

from vectorbt.utils import checks
from vectorbt.utils.config import merge_dicts
from vectorbt.base import combine_fns
from vectorbt.indicators.factory import IndicatorFactory
from vectorbt.signals.nb import generate_ex_nb, generate_enex_nb, first_choice_nb


class SignalFactory(IndicatorFactory):
    """A factory for building signal generators.

    Extends `vectorbt.indicators.factory.IndicatorFactory` with choice functions.

    Generates a fixed number of outputs (see arguments). If you need to generate other outputs,
    use in-place outputs (via `in_output_names`).

    Args:
        exit_only (bool): Whether to generate exit signals only.

            If True, uses `entries` as input and `exits` as output.
            Otherwise, uses `entries` and `exits` as outputs.
        iteratively (bool): Whether to use entries to iteratively generate new entries and exits.

            If True, uses `entries` as input and `new_entries` and `exits` as outputs.
        **kwargs: Keyword arguments passed to `vectorbt.indicators.factory.IndicatorFactory`.
    ```"""

    def __init__(self,
                 *args,
                 class_name='CustomSignals',
                 input_names=None,
                 attr_settings=None,
                 exit_only=False,
                 iteratively=False,
                 **kwargs):
        if input_names is None:
            input_names = []
        if attr_settings is None:
            attr_settings = {}
        if iteratively:
            exit_only = True
        if exit_only:
            if len(input_names) > 0:
                if input_names[0] != 'entries':
                    input_names = ['entries'] + input_names
            else:
                input_names = ['entries']
            output_names = ['exits']
            if iteratively:
                output_names = ['new_entries'] + output_names
                attr_settings['new_entries'] = dict(dtype=np.bool)
        else:
            output_names = ['entries', 'exits']
        attr_settings['entries'] = dict(dtype=np.bool)
        attr_settings['exits'] = dict(dtype=np.bool)
        IndicatorFactory.__init__(
            self, *args,
            class_name=class_name,
            input_names=input_names,
            output_names=output_names,
            attr_settings=attr_settings,
            **kwargs
        )
        self.exit_only = exit_only
        self.iteratively = iteratively

        def plot(_self,
                 entry_y=None,
                 exit_y=None,
                 entry_types=None,
                 exit_types=None,
                 entry_trace_kwargs=None,
                 exit_trace_kwargs=None,
                 fig=None,
                 **kwargs):  # pragma: no cover
            if _self.wrapper.ndim > 1:
                raise TypeError("Select a column first. Use indexing.")

            if entry_trace_kwargs is None:
                entry_trace_kwargs = {}
            if exit_trace_kwargs is None:
                exit_trace_kwargs = {}
            if entry_types is not None:
                entry_trace_kwargs = merge_dicts(dict(
                    customdata=entry_types,
                    hovertemplate="(%{x}, %{y})<br>Type: %{customdata}"
                ), entry_trace_kwargs)
            if exit_types is not None:
                exit_trace_kwargs = merge_dicts(dict(
                    customdata=exit_types,
                    hovertemplate="(%{x}, %{y})<br>Type: %{customdata}"
                ), exit_trace_kwargs)
            if exit_only and iteratively:
                entries = _self.new_entries
            else:
                entries = _self.entries
            exits = _self.exits
            fig = entries.vbt.signals.plot_as_entry_markers(
                y=entry_y, trace_kwargs=entry_trace_kwargs, fig=fig, **kwargs)
            fig = exits.vbt.signals.plot_as_exit_markers(
                y=exit_y, trace_kwargs=exit_trace_kwargs, fig=fig, **kwargs)

            return fig

        plot.__doc__ = """Plot `{0}.{1}` and `{0}.exits`.

        Args:
            entry_y (array_like): Y-axis values to plot entry markers on.
            exit_y (array_like): Y-axis values to plot exit markers on.
            entry_types (array_like): Entry types in string format.
            exit_types (array_like): Exit types in string format.
            entry_trace_kwargs (dict): Keyword arguments passed to \
            `vectorbt.signals.accessors.Signals_SRAccessor.plot_as_entry_markers` for `{0}.{1}`.
            exit_trace_kwargs (dict): Keyword arguments passed to \
            `vectorbt.signals.accessors.Signals_SRAccessor.plot_as_exit_markers` for `{0}.exits`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **kwargs: Keyword arguments passed to `vectorbt.signals.accessors.Signals_SRAccessor.plot_as_markers`.
        """.format(
            class_name, 'new_entries' if exit_only and iteratively else 'entries'
        )

        setattr(self.CustomIndicator, 'plot', plot)

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

        A choice function is simply a function that returns indices of signals.
        There are two types of it: entry choice function and exit choice function.
        Each choice function takes broadcast time series, broadcast in-place output time series,
        broadcast parameter arrays, and other arguments, and returns an array of indices
        corresponding to chosen signals. See `vectorbt.signals.nb.generate_nb`.

        If `exit_only` is True, calls `vectorbt.signals.nb.generate_ex_nb`.
        If `exit_only` is False or `iteratively` is True, calls `vectorbt.signals.nb.generate_enex_nb`.

        Args:
            entry_choice_func (callable): `choice_func_nb` that returns indices of entries.

                If `exit_only` is True, automatically set to `vectorbt.signals.nb.first_choice_nb`.
            exit_choice_func (callable): `choice_func_nb` that returns indices of exits.
            cache_func (callable): A caching function to preprocess data beforehand.

                All returned objects will be passed as last arguments to choice functions.
            entry_settings (dict): Settings dict for `entry_choice_func`.
            exit_settings (dict): Settings dict for `exit_choice_func`.
            cache_settings (dict): Settings dict for `cache_func`.
            **kwargs: Keyword arguments passed to `IndicatorFactory.from_custom_func`.

        !!! note
            Choice functions should be Numba-compiled.

            Which inputs, parameters and arguments to pass to each function should be
            explicitly indicated in the function's settings dict. By default, nothing is passed.

        Settings dict of each function can have the following keys:

        Attributes:
            pass_inputs (list of str): Input names to pass to the choice function.

                Defaults to []. Order matters. Each name must be in `input_names`.
            pass_in_outputs (list of str): In-place output names to pass to the choice function.

                Defaults to []. Order matters. Each name must be in `in_output_names`.
            pass_params (list of str): Parameter names to pass to the choice function.

                Defaults to []. Order matters. Each name must be in `param_names`.
            pass_kwargs (list of str or list of tuple): Keyword arguments from `kwargs` dict to
                pass as positional arguments to the choice function.

                Defaults to []. Order matters.

                If any element is a tuple, should contain the name and the default value.
                If any element is a string, the default value is None.

                Built-in keys include:

                * `input_shape`: Input shape if no input time series passed.
                    Default is provided by the pipeline if `forward_input_shape` is True.
                * `wait`: Number of ticks to wait before placing signals.
                    Default is 1.
                * `first`: Whether to stop as soon as the first exit signal is found.
                    Default is True.
                * `temp_idx_arr`: Empty integer array used to temporarily store indices.
                    Default is an automatically generated array of shape `input_shape[0]`.

                    You can also pass `temp_idx_arr1`, `temp_idx_arr2`, etc. to generate multiple.
                * `flex_2d`: See `vectorbt.base.reshape_fns.flex_choose_i_and_col_nb`.
                    Default is provided by the pipeline if `forward_flex_2d` is True.
            pass_cache (bool): Whether to pass cache from `cache_func` to the choice function.

                Defaults to False. Cache is passed unpacked.

        The following arguments can be passed to `run` and `run_combs` methods:

        Args:
            *args: Should be used instead of `exit_args` when `exit_only` is True.
            entry_args (tuple): Arguments passed to the entry choice function.
            exit_args (tuple): Arguments passed to the exit choice function.
            cache_args (tuple): Arguments passed to the cache function.
            entry_kwargs (tuple): Settings for the entry choice function. Also contains arguments
                passed as positional if in `pass_kwargs`.
            exit_kwargs (tuple): Settings for the exit choice function. Also contains arguments
                passed as positional if in `pass_kwargs`.
            cache_kwargs (tuple): Settings for the cache function. Also contains arguments
                passed as positional if in `pass_kwargs`.
            return_cache (bool): Whether to return only cache.
            use_cache (any): Cache to use.
            **kwargs: Should be used instead of `exit_kwargs` when `exit_only` is True.

        For more arguments, see `vectorbt.indicators.factory.run_pipeline`.

        ## Example

        Take the first entry and place an exit after waiting `n` ticks. Find the next entry and repeat.
        Test three different `n` values.

        ```python-repl
        >>> from numba import njit
        >>> from vectorbt.signals.factory import SignalFactory

        >>> @njit
        ... def wait_choice_nb(from_i, to_i, col, n, temp_idx_arr):
        ...     temp_idx_arr[0] = from_i + n  # index of next exit
        ...     if temp_idx_arr[0] < to_i:
        ...         return temp_idx_arr[:1]
        ...     return temp_idx_arr[:0]  # must return array anyway

        >>> # Build signal generator
        >>> MySignals = SignalFactory(
        ...     param_names=['n'],
        ...     iteratively=True
        ... ).from_choice_func(
        ...     exit_choice_func=wait_choice_nb,
        ...     exit_settings=dict(
        ...         pass_params=['n'],
        ...         pass_kwargs=['temp_idx_arr']  # built-in kwarg
        ...     )
        ... )

        >>> # Run signal generator
        >>> entries = [True, True, True, True, True]
        >>> my_sig = MySignals.run(entries, [0, 1, 2])

        >>> my_sig.entries  # input entries
        custom_n     0     1     2
        0         True  True  True
        1         True  True  True
        2         True  True  True
        3         True  True  True
        4         True  True  True

        >>> my_sig.new_entries  # output entries
        custom_n      0      1      2
        0          True   True   True
        1         False  False  False
        2          True  False  False
        3         False   True  False
        4          True  False   True

        >>> my_sig.exits  # output exits
        custom_n      0      1      2
        0         False  False  False
        1          True  False  False
        2         False   True  False
        3          True  False   True
        4         False  False  False
        ```

        To combine multiple iterative signals, you would need to create a choice function
        that does that. Here is an example of combining two random generators using "OR" rule:

        ```python-repl
        >>> from numba import njit
        >>> from collections import namedtuple
        >>> from vectorbt.signals.factory import SignalFactory
        >>> from vectorbt.signals.nb import rand_by_prob_choice_nb
        >>> from vectorbt.signals.basic import flex_elem_param_config

        >>> # Enum to distinguish random generators
        >>> RandType = namedtuple('RandType', ['R1', 'R2'])(0, 1)

        >>> # Define exit choice function
        >>> @njit
        ... def rand_exit_choice_nb(from_i, to_i, col, rand_type_out, prob1,
        ...                         prob2, temp_idx_arr1, temp_idx_arr2, flex_2d):
        ...     idxs1 = rand_by_prob_choice_nb(
        ...         from_i, to_i, col, prob1, True, temp_idx_arr1, flex_2d)
        ...     if len(idxs1) > 0:
        ...         to_i = idxs1[0]  # no need to go beyond first signal
        ...     idxs2 = rand_by_prob_choice_nb(
        ...         from_i, to_i, col, prob2, True, temp_idx_arr2, flex_2d)
        ...     if len(idxs2) > 0:
        ...         rand_type_out[idxs2[0], col] = RandType.R2
        ...         return idxs2
        ...     if len(idxs1) > 0:
        ...         rand_type_out[idxs1[0], col] = RandType.R1
        ...         return idxs1
        ...     return temp_idx_arr1[:0]

        >>> # Build signal generator
        >>> MySignals = SignalFactory(
        ...     in_output_names=['rand_type'],
        ...     in_output_settings=dict(
        ...         rand_type=dict(
        ...             dtype=int,
        ...             default=-1
        ...         )
        ...     ),
        ...     param_names=['prob1', 'prob2'],
        ...     param_settings=dict(
        ...         prob1=flex_elem_param_config,  # param per frame/row/col/element
        ...         prob2=flex_elem_param_config
        ...     ),
        ...     attr_settings=dict(
        ...         rand_type=dict(dtype=RandType)  # creates rand_type_readable
        ...     ),
        ...     iteratively=True
        ... ).from_choice_func(
        ...     exit_choice_func=rand_exit_choice_nb,
        ...     exit_settings=dict(
        ...         pass_in_outputs=['rand_type'],
        ...         pass_params=['prob1', 'prob2'],
        ...         pass_kwargs=['temp_idx_arr1', 'temp_idx_arr2', 'flex_2d']
        ...     ),
        ...     forward_flex_2d=True
        ... )

        >>> # Run signal generator
        >>> entries = [True, True, True, True, True]
        >>> my_sig = MySignals.run(entries, [0., 1.], [0., 1.], param_product=True)

        >>> my_sig.new_entries
        custom_prob1           0.0           1.0
        custom_prob2    0.0    1.0    0.0    1.0
        0              True   True   True   True
        1             False  False  False  False
        2             False   True   True   True
        3             False  False  False  False
        4             False   True   True   True

        >>> my_sig.exits
        custom_prob1           0.0           1.0
        custom_prob2    0.0    1.0    0.0    1.0
        0             False  False  False  False
        1             False   True   True   True
        2             False  False  False  False
        3             False   True   True   True
        4             False  False  False  False

        >>> my_sig.rand_type_readable
        custom_prob1     0.0     1.0
        custom_prob2 0.0 1.0 0.0 1.0
        0
        1                 R2  R1  R1
        2
        3                 R2  R1  R1
        4
        ```
        """

        exit_only = self.exit_only
        iteratively = self.iteratively
        input_names = self.input_names
        param_names = self.param_names
        in_output_names = self.in_output_names

        checks.assert_not_none(exit_choice_func)
        checks.assert_numba_func(exit_choice_func)
        if exit_only:
            if iteratively:
                if entry_choice_func is None:
                    entry_choice_func = first_choice_nb
                if entry_settings is None:
                    entry_settings = {}
                entry_settings = merge_dicts(dict(
                    pass_inputs=['entries']
                ), entry_settings)
        else:
            checks.assert_not_none(entry_choice_func)
            checks.assert_numba_func(entry_choice_func)

        if entry_settings is None:
            entry_settings = {}
        if exit_settings is None:
            exit_settings = {}
        if cache_settings is None:
            cache_settings = {}

        def _check_settings(func_settings):
            for k in func_settings:
                if k not in (
                        'pass_inputs',
                        'pass_in_outputs',
                        'pass_params',
                        'pass_kwargs',
                        'pass_cache'
                ):
                    raise ValueError(f"Can't find key {k} in function settings")

        _check_settings(entry_settings)
        _check_settings(exit_settings)
        _check_settings(cache_settings)

        # Get input names for each function
        def _get_func_names(func_settings, setting, all_names):
            func_input_names = func_settings.get(setting, None)
            if func_input_names is None:
                return []
            else:
                for name in func_input_names:
                    checks.assert_in(name, all_names)
            return func_input_names

        entry_input_names = _get_func_names(entry_settings, 'pass_inputs', input_names)
        exit_input_names = _get_func_names(exit_settings, 'pass_inputs', input_names)
        cache_input_names = _get_func_names(cache_settings, 'pass_inputs', input_names)

        entry_in_output_names = _get_func_names(entry_settings, 'pass_in_outputs', in_output_names)
        exit_in_output_names = _get_func_names(exit_settings, 'pass_in_outputs', in_output_names)
        cache_in_output_names = _get_func_names(cache_settings, 'pass_in_outputs', in_output_names)

        entry_param_names = _get_func_names(entry_settings, 'pass_params', param_names)
        exit_param_names = _get_func_names(exit_settings, 'pass_params', param_names)
        cache_param_names = _get_func_names(cache_settings, 'pass_params', param_names)

        if exit_only and not iteratively:
            @njit
            def apply_nb(i, entries, exit_wait, exit_input_list, exit_in_output_tuples,
                         exit_param_tuples, exit_args):
                return generate_ex_nb(
                    entries,
                    exit_wait,
                    exit_choice_func,
                    *exit_input_list,
                    *exit_in_output_tuples[i],
                    *exit_param_tuples[i],
                    *exit_args
                )
        else:
            @njit
            def apply_nb(i, shape, entry_wait, exit_wait, entry_input_list, exit_input_list,
                         entry_in_output_tuples, exit_in_output_tuples, entry_param_tuples,
                         exit_param_tuples, entry_args, exit_args):
                return generate_enex_nb(
                    shape,
                    entry_wait, exit_wait,
                    entry_choice_func, (
                        *entry_input_list,
                        *entry_in_output_tuples[i],
                        *entry_param_tuples[i],
                        *entry_args
                    ),
                    exit_choice_func, (
                        *exit_input_list,
                        *exit_in_output_tuples[i],
                        *exit_param_tuples[i],
                        *exit_args
                    )
                )

        def custom_func(input_list, in_output_list, param_list, *args, input_shape=None, flex_2d=None,
                        entry_args=None, exit_args=None, cache_args=None, entry_kwargs=None,
                        exit_kwargs=None, cache_kwargs=None, return_cache=False, use_cache=None, **_kwargs):
            # Get arguments
            if len(input_list) == 0:
                if input_shape is None:
                    raise ValueError("Pass input_shape if no input time series passed")
            else:
                input_shape = input_list[0].shape

            if entry_args is None:
                entry_args = ()
            if exit_args is None:
                exit_args = ()
            if cache_args is None:
                cache_args = ()
            if exit_only:
                if len(exit_args) > 0:
                    raise ValueError("Use *args instead of exit_args when exit_only=True")
                exit_args = args
            else:
                if len(args) > 0:
                    raise ValueError("*args can be only used when exit_only=True")

            if entry_kwargs is None:
                entry_kwargs = {}
            if exit_kwargs is None:
                exit_kwargs = {}
            if cache_kwargs is None:
                cache_kwargs = {}
            if exit_only:
                if len(exit_kwargs) > 0:
                    raise ValueError("Use **kwargs instead of exit_kwargs when exit_only=True")
                exit_kwargs = _kwargs
            else:
                if len(_kwargs) > 0:
                    raise ValueError("**kwargs can be only used when exit_only=True")

            kwargs_defaults = dict(
                input_shape=input_shape,
                wait=1,
                first=True,
                flex_2d=flex_2d,
            )
            entry_kwargs = merge_dicts(kwargs_defaults, entry_kwargs)
            exit_kwargs = merge_dicts(kwargs_defaults, exit_kwargs)
            cache_kwargs = merge_dicts(kwargs_defaults, cache_kwargs)
            entry_wait = entry_kwargs['wait']
            exit_wait = exit_kwargs['wait']

            # Distribute arguments across functions
            entry_input_list = ()
            exit_input_list = ()
            cache_input_list = ()
            for input_name in entry_input_names:
                entry_input_list += (input_list[input_names.index(input_name)],)
            for input_name in exit_input_names:
                exit_input_list += (input_list[input_names.index(input_name)],)
            for input_name in cache_input_names:
                cache_input_list += (input_list[input_names.index(input_name)],)

            entry_in_output_list = ()
            exit_in_output_list = ()
            cache_in_output_list = ()
            for in_output_name in entry_in_output_names:
                entry_in_output_list += (in_output_list[in_output_names.index(in_output_name)],)
            for in_output_name in exit_in_output_names:
                exit_in_output_list += (in_output_list[in_output_names.index(in_output_name)],)
            for in_output_name in cache_in_output_names:
                cache_in_output_list += (in_output_list[in_output_names.index(in_output_name)],)

            entry_param_list = ()
            exit_param_list = ()
            cache_param_list = ()
            for param_name in entry_param_names:
                entry_param_list += (param_list[param_names.index(param_name)],)
            for param_name in exit_param_names:
                exit_param_list += (param_list[param_names.index(param_name)],)
            for param_name in cache_param_names:
                cache_param_list += (param_list[param_names.index(param_name)],)

            n_params = len(param_list[0]) if len(param_list) > 0 else 1
            entry_in_output_tuples = tuple(zip(*entry_in_output_list))
            if len(entry_in_output_tuples) == 0:
                entry_in_output_tuples = ((),) * n_params
            exit_in_output_tuples = tuple(zip(*exit_in_output_list))
            if len(exit_in_output_tuples) == 0:
                exit_in_output_tuples = ((),) * n_params
            entry_param_tuples = tuple(zip(*entry_param_list))
            if len(entry_param_tuples) == 0:
                entry_param_tuples = ((),) * n_params
            exit_param_tuples = tuple(zip(*exit_param_list))
            if len(exit_param_tuples) == 0:
                exit_param_tuples = ((),) * n_params

            def _build_more_args(func_settings, func_kwargs):
                pass_kwargs = func_settings.get('pass_kwargs', [])
                more_args = ()
                for key in pass_kwargs:
                    value = None
                    if isinstance(key, tuple):
                        key, value = key
                    else:
                        if key.startswith('temp_idx_arr'):
                            value = np.empty((input_shape[0],), dtype=np.int_)
                    value = func_kwargs.get(key, value)
                    more_args += (value,)
                return more_args

            entry_more_args = _build_more_args(entry_settings, entry_kwargs)
            exit_more_args = _build_more_args(exit_settings, exit_kwargs)
            cache_more_args = _build_more_args(cache_settings, cache_kwargs)

            # Caching
            cache = use_cache
            if cache is None and cache_func is not None:
                cache = cache_func(
                    *cache_input_list,
                    *cache_in_output_list,
                    *cache_param_list,
                    *cache_args,
                    *cache_more_args
                )
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
                    exit_wait,
                    exit_input_list,
                    exit_in_output_tuples,
                    exit_param_tuples,
                    exit_args + exit_more_args + exit_cache
                )
            else:
                return combine_fns.apply_and_concat_multiple_nb(
                    n_params,
                    apply_nb,
                    input_shape,
                    entry_wait,
                    exit_wait,
                    entry_input_list,
                    exit_input_list,
                    entry_in_output_tuples,
                    exit_in_output_tuples,
                    entry_param_tuples,
                    exit_param_tuples,
                    entry_args + entry_more_args + entry_cache,
                    exit_args + exit_more_args + exit_cache
                )

        return self.from_custom_func(custom_func, pass_lists=True, **kwargs)
