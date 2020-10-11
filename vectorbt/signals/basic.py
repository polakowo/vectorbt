"""Signal generators built with `vectorbt.signals.factory.SignalFactory`."""

import numpy as np

from vectorbt.utils.config import Config
from vectorbt.utils.docs import fix_class_for_docs
from vectorbt.signals.enums import StopType
from vectorbt.signals.factory import SignalFactory
from vectorbt.signals.nb import (
    rand_enex_apply_nb,
    rand_by_prob_choice_nb,
    stop_choice_nb,
    adv_stop_choice_nb
)

elem_param_config = Config(
    frozen=False,
    **dict(
        array_like=True,  # passing a NumPy array means passing one value, for multiple use list
        bc_to_input=True,  # broadcast to input
        broadcast_kwargs=dict(
            keep_raw=True  # don't materialize, keep shape for flexible indexing
        )
    )
)
"""Config for element-wise parameters."""

col_param_config = Config(
    frozen=False,
    **dict(
        array_like=True,
        bc_to_input=1,  # broadcast to axis 1 (columns)
        broadcast_kwargs=dict(
            keep_raw=True
        )
    )
)
"""Config for column-wise parameters."""

# ############# Random signals ############# #


RAND = SignalFactory(
    class_name='RAND',
    module_name=__name__,
    short_name='rand',
    param_names=['n'],
    param_settings=dict(
        n=col_param_config
    )
).from_apply_func(  # apply_func since function is (almost) vectorized
    rand_enex_apply_nb,
    pass_kwargs=[
        ('entry_wait', 1),
        ('exit_wait', 1)
    ]
)


class RAND(RAND):
    """Random entry and exit signal generator based on the number of signals.

    Generates `entries` and `exits` based on `vectorbt.signals.nb.rand_enex_apply_nb`.

    !!! note
        Parameter `n` can be either a single value (per frame) or a NumPy array (per column).
        To generate multiple combinations, pass it as a list.

    Example:
        Test three different `n` values:
        ```python-repl
        >>> import vectorbt as vbt

        >>> rand = vbt.RAND.run(
        ...     n=[1, 2, 3], input_shape=(6,), seed=42
        ... )
        >>> rand.entries
        rand_n      1      2      3
        0        True   True   True
        1       False  False  False
        2       False   True   True
        3       False  False  False
        4       False  False   True
        5       False  False  False
        >>> rand.exits
        rand_n      1      2      3
        0       False  False  False
        1        True   True   True
        2       False  False  False
        3       False   True   True
        4       False  False  False
        5       False  False   True
        ```

        `n` can also be set per column:
        ```python-repl
        >>> import numpy as np

        >>> rand = vbt.RAND.run(
        ...     n=[np.array([1, 2]), np.array([3, 4])],
        ...     input_shape=(8, 2), seed=42
        ... )
        >>> rand.entries
        rand_n         mix_0         mix_1
                    0      1      0      1
        0       False   True   True   True
        1        True  False  False  False
        2       False  False  False   True
        3       False  False   True  False
        4       False   True  False   True
        5       False  False   True  False
        6       False  False  False   True
        7       False  False  False  False
        >>> rand.exits
        rand_n         mix_0         mix_1
                    0      1      0      1
        0       False  False  False  False
        1       False  False   True   True
        2       False  False  False  False
        3       False   True  False   True
        4       False  False   True  False
        5        True  False  False   True
        6       False  False   True  False
        7       False   True  False   True
        ```"""
    pass


fix_class_for_docs(RAND)

RPROB = SignalFactory(
    class_name='RPROB',
    module_name=__name__,
    short_name='rprob',
    param_names=['entry_prob', 'exit_prob'],
    param_settings=dict(
        entry_prob=elem_param_config,
        exit_prob=elem_param_config
    )
).from_choice_func(
    entry_choice_func=rand_by_prob_choice_nb,
    entry_settings=dict(
        pass_params=['entry_prob'],
        pass_kwargs=['first', 'temp_int', 'flex_2d']
    ),
    exit_choice_func=rand_by_prob_choice_nb,
    exit_settings=dict(
        pass_params=['exit_prob'],
        pass_kwargs=['first', 'temp_int', 'flex_2d']
    ),
    forward_flex_2d=True
)


class RPROB(RPROB):
    """Random entry and exit signal generator based on probabilities.

    Generates `entries` and `exits` based on `vectorbt.signals.nb.rand_by_prob_choice_nb`.

    !!! note
        All parameters can be either a single value (per frame) or a NumPy array (per row, column,
        or element). To generate multiple combinations, pass them as lists.

    Example:
        Test all probability combinations:
        ```python-repl
        >>> import vectorbt as vbt

        >>> rprob = vbt.RPROB.run(
        ...     entry_prob=[0.5, 1.], exit_prob=[0.5, 1.],
        ...     input_shape=(5,), param_product=True, seed=42
        ... )

        >>> rprob.entries
        rprob_entry_prob           0.5           1.0
        rprob_exit_prob     0.5    1.0    0.5    1.0
        0                  True   True   True   True
        1                 False  False  False  False
        2                 False  False  False   True
        3                 False  False  False  False
        4                 False  False   True   True
        >>> rprob.exits
        rprob_entry_prob           0.5           1.0
        rprob_exit_prob     0.5    1.0    0.5    1.0
        0                 False  False  False  False
        1                 False   True  False   True
        2                 False  False  False  False
        3                 False  False   True   True
        4                  True  False  False  False
        ```

        `entry_prob` and `exit_prob` can also be set per row, column, or element:
        ```python-repl
        >>> import numpy as np

        >>> entry_prob1 = np.asarray([1., 0., 1., 0., 1.])
        >>> entry_prob2 = np.asarray([0., 1., 0., 1., 0.])
        >>> rprob = vbt.RPROB.run(
        ...     entry_prob=[entry_prob1, entry_prob2], exit_prob=1.,
        ...     input_shape=(5,), seed=42
        ... )

        >>> rprob.entries
        rprob_entry_prob  mix_0  mix_1
        rprob_exit_prob     1.0    1.0
        0                  True  False
        1                 False   True
        2                  True  False
        3                 False   True
        4                  True  False
        >>> rprob.exits
        rprob_entry_prob  mix_0  mix_1
        rprob_exit_prob     1.0    1.0
        0                 False  False
        1                  True  False
        2                 False   True
        3                  True  False
        4                 False   True
        ```"""
    pass


fix_class_for_docs(RPROB)

rprobex_config = Config(
    frozen=False,
    **dict(
        class_name='RPROBEX',
        module_name=__name__,
        short_name='rprobex',
        param_names=['prob'],
        param_settings=dict(
            prob=elem_param_config
        ),
        exit_only=True,
        iteratively=False
    )
)
"""Factory config for `RPROBEX`."""

rprobex_func_config = Config(
    frozen=False,
    **dict(
        exit_choice_func=rand_by_prob_choice_nb,
        exit_settings=dict(
            pass_params=['prob'],
            pass_kwargs=['first', 'temp_int', 'flex_2d']
        ),
        forward_flex_2d=True
    )
)
"""Exit function config for `RPROBEX`."""

RPROBEX = SignalFactory(
    **rprobex_config
).from_choice_func(
    **rprobex_func_config
)


class RPROBEX(RPROBEX):
    """Random exit signal generator based on probabilities.

    Generates `exits` based on `entries` and `vectorbt.signals.nb.rand_by_prob_choice_nb`.

    See `RPROB` for notes on parameters."""
    pass


fix_class_for_docs(RPROBEX)

IRPROBEX = SignalFactory(
    **rprobex_config.merge_with(
        dict(
            class_name='IRPROBEX',
            short_name='irprobex',
            iteratively=True
        )
    )
).from_choice_func(
    **rprobex_func_config
)


class IRPROBEX(IRPROBEX):
    """Random exit signal generator based on probabilities.

    Iteratively generates `new_entries` and `exits` based on `entries` and
    `vectorbt.signals.nb.rand_by_prob_choice_nb`.

    See `RPROB` for notes on parameters."""
    pass


fix_class_for_docs(IRPROBEX)

# ############# Stop signals ############# #

stex_config = Config(
    frozen=False,
    **dict(
        class_name='STEX',
        module_name=__name__,
        short_name='stex',
        input_names=['ts'],
        param_names=['stop', 'trailing'],
        param_settings=dict(
            stop=elem_param_config,
            trailing=elem_param_config
        ),
        exit_only=True,
        iteratively=False
    )
)
"""Factory config for `STEX`."""

stex_func_config = Config(
    frozen=False,
    **dict(
        exit_choice_func=stop_choice_nb,
        exit_settings=dict(
            pass_inputs=['ts'],
            pass_params=['stop', 'trailing'],
            pass_kwargs=['wait', 'first', 'temp_int', 'flex_2d']
        ),
        forward_flex_2d=True
    )
)
"""Exit function config for `STEX`."""

STEX = SignalFactory(
    **stex_config
).from_choice_func(
    **stex_func_config
)


class STEX(STEX):
    """Exit signal generator based on stop values.

    Generates `exits` based on `entries` and `vectorbt.signals.nb.stop_choice_nb`.

    !!! note
        All parameters can be either a single value (per frame) or a NumPy array (per row, column,
        or element). To generate multiple combinations, pass them as lists."""
    pass


fix_class_for_docs(STEX)

ISTEX = SignalFactory(
    **stex_config.merge_with(
        dict(
            class_name='ISTEX',
            short_name='istex',
            iteratively=True
        )
    )
).from_choice_func(
    **stex_func_config
)


class ISTEX(ISTEX):
    """Exit signal generator based on stop values.

    Iteratively generates `new_entries` and `exits` based on `entries` and
    `vectorbt.signals.nb.stop_choice_nb`.

    See `STEX` for notes on parameters."""
    pass


fix_class_for_docs(ISTEX)


# ############# Advanced stop signals ############# #

@property
def _hit_price_filled(self):
    """Hit price with empty values replaced by NaN."""
    hit_price = self.hit_price.copy()
    hit_price[~self.exits] = np.nan
    return hit_price


@property
def _stop_type_filled(self):
    """Stop type with empty values replaced by -1."""
    stop_type = self.stop_type.copy()
    stop_type[~self.exits] = -1
    return stop_type


@property
def _stop_type_readable(self):
    """Stop type in readable (string) format."""
    return self.stop_type_filled.applymap(
        lambda x: StopType._fields[x] if x in StopType else '')


advstex_config = Config(
    frozen=False,
    **dict(
        class_name='ADVSTEX',
        module_name=__name__,
        short_name='advstex',
        input_names=['open', 'high', 'low', 'close'],
        in_output_names=['hit_price', 'stop_type'],
        output_settings=dict(
            stop_type=dict(
                dtype=np.int_
            )
        ),
        param_names=['sl_stop', 'ts_stop', 'tp_stop'],
        param_settings=dict(stop=elem_param_config),
        custom_output_funcs={
            'hit_price_filled': _hit_price_filled,
            'stop_type_filled': _stop_type_filled,
            'stop_type_readable': _stop_type_readable
        },
        obj_settings=dict(
            hit_price=dict(create_methods=False),
            stop_type=dict(create_methods=False),
            stop_type_filled=dict(create_methods=False),
            stop_type_readable=dict(create_methods=False)
        ),
        exit_only=True,
        iteratively=False
    )
)
"""Factory config for `ADVSTEX`."""

advstex_func_config = Config(
    frozen=False,
    **dict(
        exit_choice_func=adv_stop_choice_nb,
        exit_settings=dict(
            pass_inputs=['open', 'high', 'low', 'close'],  # do not pass entries
            pass_in_outputs=['hit_price', 'stop_type'],
            pass_params=['sl_stop', 'ts_stop', 'tp_stop'],
            pass_kwargs=[('is_open_safe', True), 'wait', 'first', 'temp_int', 'flex_2d'],
        ),
        forward_flex_2d=True
    )
)
"""Exit function config for `ADVSTEX`."""

ADVSTEX = SignalFactory(
    **advstex_config
).from_choice_func(
    **advstex_func_config
)


class ADVSTEX(ADVSTEX):
    """Advanced exit signal generator based on stop values.

    Generates `exits` based on `entries` and `vectorbt.signals.nb.adv_stop_choice_nb`.

    !!! note
        All parameters can be either a single value (per frame) or a NumPy array (per row, column,
        or element). To generate multiple combinations, pass them as lists.

    Example:
        Test each stop type individually:
        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd

        >>> entries = pd.Series([True, False, False, False, False])
        >>> price = pd.DataFrame({
        ...     'open': [10, 11, 12, 11, 10],
        ...     'high': [11, 12, 13, 12, 11],
        ...     'low': [9, 10, 11, 10, 9],
        ...     'close': [10, 11, 12, 11, 10]
        ... })
        >>> advstex = vbt.ADVSTEX.run(
        ...     entries, price['open'], price['high'], price['low'], price['close'],
        ...     sl_stop=[0.1, 0., 0.], ts_stop=[0., 0.1, 0.], tp_stop=[0., 0., 0.1]
        ... )

        >>> advstex.entries
        advstex_sl_stop    0.1    0.0    0.0
        advstex_ts_stop    0.0    0.1    0.0
        advstex_tp_stop    0.0    0.0    0.1
        0                 True   True   True
        1                False  False  False
        2                False  False  False
        3                False  False  False
        4                False  False  False
        >>> advstex.exits
        advstex_sl_stop    0.1    0.0    0.0
        advstex_ts_stop    0.0    0.1    0.0
        advstex_tp_stop    0.0    0.0    0.1
        0                False  False  False
        1                False  False   True
        2                False  False  False
        3                False   True  False
        4                 True  False  False
        >>> advstex.hit_price_filled
        advstex_sl_stop  0.1   0.0   0.0
        advstex_ts_stop  0.0   0.1   0.0
        advstex_tp_stop  0.0   0.0   0.1
        0                NaN   NaN   NaN
        1                NaN   NaN  11.0
        2                NaN   NaN   NaN
        3                NaN  11.7   NaN
        4                9.0   NaN   NaN
        >>> advstex.stop_type_readable
        advstex_sl_stop       0.1        0.0         0.0
        advstex_ts_stop       0.0        0.1         0.0
        advstex_tp_stop       0.0        0.0         0.1
        0
        1                                     TakeProfit
        2
        3                          TrailStop
        4                StopLoss
        ```"""
    pass


fix_class_for_docs(ADVSTEX)

IADVSTEX = SignalFactory(
    **advstex_config.merge_with(
        dict(
            class_name='IADVSTEX',
            short_name='iadvstex',
            iteratively=True
        )
    )
).from_choice_func(
    **advstex_func_config
)


class IADVSTEX(IADVSTEX):
    """Advanced exit signal generator based on stop values.

    Iteratively generates `new_entries` and `exits` based on `entries` and
    `vectorbt.signals.nb.adv_stop_choice_nb`.

    See `ADVSTEX` for notes on parameters."""
    pass


fix_class_for_docs(IADVSTEX)
