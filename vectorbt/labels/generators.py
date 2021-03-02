"""Basic look-ahead indicators and label generators."""

from vectorbt.utils.docs import fix_class_for_docs
from vectorbt.indicators.factory import IndicatorFactory
from vectorbt.indicators.configs import flex_elem_param_config
from vectorbt.labels import nb
from vectorbt.labels.enums import TrendMode

# ############# Look-ahead indicators ############# #

FMEAN = IndicatorFactory(
    class_name='FMEAN',
    module_name=__name__,
    input_names=['close'],
    param_names=['window', 'ewm'],
    output_names=['fmean']
).from_apply_func(
    nb.future_mean_apply_nb,
    kwargs_to_args=['wait', 'adjust'],
    ewm=False,
    wait=1,
    adjust=False
)
"""Look-ahead indicator based on `future_mean_apply_nb`."""

FSTD = IndicatorFactory(
    class_name='FSTD',
    module_name=__name__,
    input_names=['close'],
    param_names=['window', 'ewm'],
    output_names=['fstd']
).from_apply_func(
    nb.future_std_apply_nb,
    kwargs_to_args=['wait', 'adjust', 'ddof'],
    ewm=False,
    wait=1,
    adjust=False,
    ddof=0
)
"""Look-ahead indicator based on `future_std_apply_nb`."""

FMIN = IndicatorFactory(
    class_name='FMIN',
    module_name=__name__,
    input_names=['close'],
    param_names=['window'],
    output_names=['fmin']
).from_apply_func(
    nb.future_min_apply_nb,
    kwargs_to_args=['wait'],
    wait=1
)
"""Look-ahead indicator based on `future_min_apply_nb`."""

FMAX = IndicatorFactory(
    class_name='FMAX',
    module_name=__name__,
    input_names=['close'],
    param_names=['window'],
    output_names=['fmax']
).from_apply_func(
    nb.future_max_apply_nb,
    kwargs_to_args=['wait'],
    wait=1
)
"""Look-ahead indicator based on `future_max_apply_nb`."""


# ############# Label generators ############# #


def _plot(self, **kwargs):  # pragma: no cover
    """Plot `close` and overlay it with the heatmap of `labels`."""
    if self.wrapper.ndim > 1:
        raise TypeError("Select a column first. Use indexing.")

    return self.close.rename('close').vbt.overlay_with_heatmap(self.labels.rename('labels'), **kwargs)


FIXLB = IndicatorFactory(
    class_name='FIXLB',
    module_name=__name__,
    input_names=['close'],
    param_names=['n'],
    output_names=['labels']
).from_apply_func(
    nb.fixed_labels_apply_nb
)


class FIXLB(FIXLB):
    """Label generator based on `fixed_labels_apply_nb`."""

    plot = _plot


fix_class_for_docs(FIXLB)

MEANLB = IndicatorFactory(
    class_name='MEANLB',
    module_name=__name__,
    input_names=['close'],
    param_names=['window', 'ewm'],
    output_names=['labels']
).from_apply_func(
    nb.mean_labels_apply_nb,
    kwargs_to_args=['wait', 'adjust'],
    ewm=False,
    wait=1,
    adjust=False
)


class MEANLB(MEANLB):
    """Label generator based on `mean_labels_apply_nb`."""

    plot = _plot


fix_class_for_docs(MEANLB)

LEXLB = IndicatorFactory(
    class_name='LEXLB',
    module_name=__name__,
    input_names=['close'],
    param_names=['pos_th', 'neg_th'],
    output_names=['labels']
).from_apply_func(
    nb.local_extrema_apply_nb,
    param_settings=dict(
        pos_th=flex_elem_param_config,
        neg_th=flex_elem_param_config
    ),
    pass_flex_2d=True
)


class LEXLB(LEXLB):
    """Label generator based on `local_extrema_apply_nb`."""

    plot = _plot


fix_class_for_docs(LEXLB)

TRENDLB = IndicatorFactory(
    class_name='TRENDLB',
    module_name=__name__,
    input_names=['close'],
    param_names=['pos_th', 'neg_th', 'mode'],
    output_names=['labels']
).from_apply_func(
    nb.trend_labels_apply_nb,
    param_settings=dict(
        pos_th=flex_elem_param_config,
        neg_th=flex_elem_param_config,
        mode=dict(dtype=TrendMode)
    ),
    pass_flex_2d=True,
    mode=TrendMode.Binary
)


class TRENDLB(TRENDLB):
    """Label generator based on `trend_labels_apply_nb`."""

    plot = _plot


fix_class_for_docs(TRENDLB)

BOLB = IndicatorFactory(
    class_name='BOLB',
    module_name=__name__,
    input_names=['close'],
    param_names=['window', 'pos_th', 'neg_th'],
    output_names=['labels']
).from_apply_func(
    nb.breakout_labels_nb,
    param_settings=dict(
        pos_th=flex_elem_param_config,
        neg_th=flex_elem_param_config
    ),
    pass_flex_2d=True,
    kwargs_to_args=['wait'],
    pos_th=0.,
    neg_th=0.,
    wait=1
)


class BOLB(BOLB):
    """Label generator based on `breakout_labels_nb`."""

    plot = _plot


fix_class_for_docs(BOLB)
