"""Mixin for building statistics out of performance metrics."""

import pandas as pd
import numpy as np
from collections import Counter
import warnings
import sys

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.config import Config, merge_dicts, get_func_arg_names
from vectorbt.utils.template import deep_substitute
from vectorbt.utils.tags import match_tags
from vectorbt.base.array_wrapper import Wrapping


class StatsBuilderMixin:
    """Mixin that implements `StatsBuilderMixin.stats`.

    Required to be a subclass of `vectorbt.base.array_wrapper.Wrapping`."""

    def __init__(self):
        checks.assert_type(self, Wrapping)

        # Copy writeable attrs
        self.metrics = self.__class__.metrics.copy()

    @property
    def writeable_attrs(self) -> tp.Set[str]:
        """Set of writeable attributes that will be saved/copied along with the config."""
        return {'metrics'}

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `StatsBuilderMixin.stats`.

        See `stats_builder` in `vectorbt._settings.settings`."""
        from vectorbt._settings import settings
        stats_builder_cfg = settings['stats_builder']

        return merge_dicts(
            stats_builder_cfg,
            dict(settings=dict(freq=self.wrapper.freq))
        )

    metrics: tp.ClassVar[Config] = Config(
        dict(
            start=dict(
                title='Start',
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None
            ),
            end=dict(
                title='End',
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None
            ),
            period=dict(
                title='Period',
                calc_func=lambda self:
                len(self.wrapper.index) * (self.wrapper.freq if self.wrapper.freq is not None else 1),
                agg_func=None
            )
        ),
        copy_kwargs=dict(copy_mode='deep')
    )
    """Metrics supported by `StatsBuilderMixin.stats`.

    !!! note
        It's safe to change this config - it's a (deep) copy of the class variable.
        
        But copying `StatsBuilderMixin` using `StatsBuilderMixin.copy` won't create a copy of the config."""

    def stats(self,
              metrics: tp.Optional[tp.MaybeIterable[tp.Union[str, tp.Tuple[str, tp.Kwargs]]]] = None,
              tags: tp.Optional[tp.MaybeIterable[str]] = None,
              column: tp.Optional[tp.Label] = None,
              group_by: tp.GroupByLike = None,
              agg_func: tp.Optional[tp.Callable] = np.mean,
              silence_warnings: tp.Optional[bool] = None,
              template_mapping: tp.Optional[tp.Mapping] = None,
              settings: tp.KwargsLike = None,
              filters: tp.KwargsLike = None,
              metric_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Compute various metrics on this object.

        Args:
            metrics (str, tuple, iterable, or dict): Metrics to calculate.

                Each element can be either:

                * a metric name (see keys in `StatsBuilderMixin.metrics`)
                * a tuple of a metric name and a settings dict as in `StatsBuilderMixin.metrics`.

                The settings dict can contain the following keys:

                * `title`: Title of the metric. Defaults to the name.
                * `tags`: Single or multiple tags to associate this metric with.
                    If any of these tags is in `tags`, keeps this metric.
                * `check_{filter}` and `inv_check_{filter}`: Whether to check this metric against a
                    filter defined in `filters`. True (or False for inverse) means to keep this metric.
                * `calc_func`: Calculation function for custom metrics.
                    Should return either a scalar for one column/group, pd.Series for multiple columns/groups,
                    or a dict of such for multiple sub-metrics.
                * `resolve_calc_func`: whether to resolve `calc_func`. If the function can be accessed
                    by traversing attributes of this object, you can specify the path to this function
                    as a string (see `vectorbt.utils.attr.deep_getattr` for the path format).
                    If `calc_func` is a function, arguments from merged metric settings are matched with
                    arguments in the signature (see below). If `resolve_calc_func` is False, `calc_func`
                    should accept (resolved) self and dictionary of merged metric settings.
                    Defaults to True.
                * `post_calc_func`: Function to post-process the result of `calc_func`.
                    Should accept (resolved) self, output of `calc_func`, and dictionary of merged metric settings,
                    and return whatever is acceptable to be returned by `calc_func`.
                * `pass_{arg}`: Whether to pass any optional argument (see below). Defaults to True if this argument
                    was found in the function's signature. Set to False to not pass.
                    If argument to be passed was not found, `pass_{arg}` is removed.
                * `resolve_{arg}`: Whether to resolve an argument that is meant to be an attribute of
                    this object (see `StatsBuilderMixin.resolve_attr`). Defaults to True if this argument
                    was found in the function's signature. Set to False to not resolve.
                * `template_mapping`: Mapping to replace templates in metric settings. Used across all settings.
                * Any other keyword argument overrides optional arguments (see below)
                    or is passed directly to `calc_func`.

                If `resolve_calc_func` is True, the calculation function may "request" any of the
                following optional arguments by accepting them or if `pass_{arg}` was found in the settings dict:

                * Each of `StatsBuilderMixin.self_aliases`: original object (ungrouped, with no column selected)
                * `group_by` - won't be passed if it was used in resolving the first attribute of `calc_func`
                    specified as a path, use `pass_group_by=True` to pass anyway
                * `column`
                * `metric_name`
                * `agg_func`
                * Any optional argument from `settings`
                * Any attribute of this object if it meant to be resolved (see `StatsBuilderMixin.resolve_attr`)

                Pass `metrics='all'` to calculate all supported metrics.
            tags (str or iterable): Tags to select.

                See `vectorbt.utils.tags.match_tags`.
            column (str): Name of the column/group.

                !!! hint
                    There are two ways to select a column: `obj['a'].stats()` and `obj.stats(column='a')`.
                    They both accomplish the same thing but in different ways: `obj['a'].stats()` computes
                    statistics of the column 'a' only, while `obj.stats(column='a')` computes statistics of
                    all columns first and only then selects the column 'a'. The first method is preferred
                    when you have a lot of data or caching is disabled. The second method is preferred when
                    most attributes have already been cached.
            group_by (any): Group or ungroup columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            agg_func (callable): Aggregation function to aggregate statistics across all columns.
                Defaults to mean.

                Should take `pd.Series` and return a const.

                Has only effect if `column` was specified or this object contains only one column of data.
                If `agg_func` has been overridden by a metric, it only takes effect if global `agg_func` is not None.
            silence_warnings (bool): Whether to silence all warnings.
            template_mapping (mapping): Global mapping to replace templates.

                Also applied on `filters`, `settings`, and `metric_kwargs`.

                Gets merged over `template_mapping` from `StatsBuilderMixin.stats_defaults`.
            filters (dict): Filters to apply.

                Each item consists of the filter name and settings dict.

                The settings dict can contain the following keys:

                * `filter_func`: Filter function that should accept resolved self and
                    merged settings for a metric, and return either True or False.
                * `warning_message`: Warning message to be shown when skipping a metric.
                    Can be a template that will be substituted using merged metric settings as mapping.
                    Defaults to None.
                * `inv_warning_message`: Same as `warning_message` but for inverse checks.

                Gets merged over `filters` from `StatsBuilderMixin.stats_defaults`.
            settings (dict): Global settings that override/extend optional arguments.

                Gets merged over `settings` from `StatsBuilderMixin.stats_defaults`.
            metric_kwargs (dict): Keyword arguments for each metric.

                They override any key defined in `settings` and metric settings.

                Gets merged over `metric_kwargs` from `StatsBuilderMixin.stats_defaults`.

        For template logic, see `vectorbt.utils.template`.

        For defaults, see `StatsBuilderMixin.stats_defaults`.

        !!! hint
            There are two types of arguments: optional and mandatory. Optional arguments
            are only passed if they are found in the function's signature. Mandatory arguments
            are passed regardless of this. Optional arguments can only be defined using `settings`
            (that is, globally), while mandatory arguments can be defined both using default metric
            settings and `{metric_name}_kwargs`. Overriding optional arguments using default metric settings
            or `{metric_name}_kwargs` won't turn them into mandatory. For this, pass `pass_{arg}=True`.

        !!! hint
            Make sure to resolve and then to re-use as many object attributes as possible to
            utilize built-in caching (even if global caching is disabled).
        """
        # Resolve defaults
        if silence_warnings is None:
            silence_warnings = self.stats_defaults['silence_warnings']
        template_mapping = merge_dicts(self.stats_defaults['template_mapping'], template_mapping)
        filters = merge_dicts(self.stats_defaults['filters'], filters)
        settings = merge_dicts(self.stats_defaults['settings'], settings)
        metric_kwargs = merge_dicts(self.stats_defaults['metric_kwargs'], metric_kwargs)

        # Replace templates globally
        if len(template_mapping) > 0:
            settings = deep_substitute(settings, mapping=template_mapping)
            filters = deep_substitute(filters, mapping=template_mapping)
            metric_kwargs = deep_substitute(metric_kwargs, mapping=template_mapping)

        # Resolve self
        reself = self.resolve_self(
            cond_kwargs=settings,
            impacts_caching=False,
            silence_warnings=silence_warnings
        )

        # Prepare metrics
        if metrics is None:
            metrics = reself.stats_defaults['metrics']
        if metrics == 'all':
            metrics = reself.metrics
        if isinstance(metrics, dict):
            metrics = list(metrics.items())
        if isinstance(metrics, (str, tuple)):
            metrics = [metrics]

        # Prepare tags
        if tags is None:
            tags = reself.stats_defaults['tags']
        if isinstance(tags, str) and tags == 'all':
            tags = None
        if isinstance(tags, (str, tuple)):
            tags = [tags]

        # Bring to the same shape
        new_metrics = []
        for i, metric in enumerate(metrics):
            if isinstance(metric, str):
                metric = (metric, reself.metrics[metric])
            if not isinstance(metric, tuple):
                raise TypeError(f"Metric at index {i} must be either a string or a tuple")
            new_metrics.append(metric)
        metrics = new_metrics

        # Handle duplicate names
        metric_counts = Counter(list(map(lambda x: x[0], metrics)))
        metric_i = {k: -1 for k in metric_counts.keys()}
        metrics_dct = {}
        for i, (metric_name, metric_settings) in enumerate(metrics):
            if metric_counts[metric_name] > 1:
                metric_i[metric_name] += 1
                metric_name = metric_name + '_' + str(metric_i[metric_name])
            metrics_dct[metric_name] = metric_settings

        # Check metric_kwargs
        missed_keys = set(metric_kwargs.keys()).difference(set(metrics_dct.keys()))
        if len(missed_keys) > 0:
            raise ValueError(f"Keys {missed_keys} in metric_kwargs could not be matched with any metric")

        # Merge settings
        opt_arg_names_dct = {}
        custom_arg_names_dct = {}
        resolved_self_dct = {}
        for metric_name, metric_settings in list(metrics_dct.items()):
            opt_settings = merge_dicts(
                {name: reself for name in reself.self_aliases},
                dict(
                    column=column,
                    group_by=group_by,
                    metric_name=metric_name,
                    agg_func=agg_func
                ),
                settings
            )
            metric_settings = metric_settings.copy()
            passed_settings = metric_kwargs.get(metric_name, {})
            merged_settings = merge_dicts(
                opt_settings,
                metric_settings,
                passed_settings
            )
            metric_template_mapping = merged_settings.pop('template_mapping', {})
            mapping = merge_dicts(merged_settings, template_mapping, metric_template_mapping)
            merged_settings = deep_substitute(merged_settings, mapping=mapping)

            # Filter by tag
            if tags is not None:
                in_tags = merged_settings.get('tags', None)
                if in_tags is None or not match_tags(tags, in_tags):
                    metrics_dct.pop(metric_name, None)
                    continue

            custom_arg_names = set(metric_settings.keys()).union(set(passed_settings.keys()))
            opt_arg_names = set(opt_settings.keys())
            custom_reself = reself.resolve_self(
                cond_kwargs=merged_settings,
                custom_arg_names=custom_arg_names,
                impacts_caching=True,
                silence_warnings=silence_warnings
            )

            metrics_dct[metric_name] = merged_settings
            custom_arg_names_dct[metric_name] = custom_arg_names
            opt_arg_names_dct[metric_name] = opt_arg_names
            resolved_self_dct[metric_name] = custom_reself

        # Filter metrics
        for filter_name, filter_settings in filters.items():
            for metric_name, metric_settings in list(metrics_dct.items()):
                custom_reself = resolved_self_dct[metric_name]
                filter_func = filter_settings['filter_func']
                warning_message = filter_settings.get('warning_message', None)
                inv_warning_message = filter_settings.get('inv_warning_message', None)
                to_check = metric_settings.get('check_' + filter_name, False)
                inv_to_check = metric_settings.get('inv_check_' + filter_name, False)

                if to_check or inv_to_check:
                    whether_true = filter_func(custom_reself, metric_settings)
                    to_remove = (to_check and not whether_true) or (inv_to_check and whether_true)
                    if to_remove:
                        if to_check and warning_message is not None and not silence_warnings:
                            warning_message = deep_substitute(warning_message, mapping=metric_settings)
                            warnings.warn(warning_message)
                        if inv_to_check and inv_warning_message is not None and not silence_warnings:
                            inv_warning_message = deep_substitute(inv_warning_message, mapping=metric_settings)
                            warnings.warn(inv_warning_message)

                        metrics_dct.pop(metric_name, None)
                        custom_arg_names_dct.pop(metric_name, None)
                        opt_arg_names_dct.pop(metric_name, None)
                        resolved_self_dct.pop(metric_name, None)

        # Compute stats
        arg_cache_dct = {}
        stats_dct = {}
        for i, (metric_name, metric_settings) in enumerate(metrics_dct.items()):
            try:
                final_kwargs = metric_settings.copy()
                opt_arg_names = opt_arg_names_dct[metric_name]
                custom_arg_names = custom_arg_names_dct[metric_name]
                custom_reself = resolved_self_dct[metric_name]

                # Clean up keys
                for k, v in list(final_kwargs.items()):
                    if k.startswith('check_') or k.startswith('inv_check_') or k in ('tags',):
                        final_kwargs.pop(k, None)

                # Get metric-specific values
                _column = final_kwargs.get('column')
                _group_by = final_kwargs.get('group_by')
                _agg_func = final_kwargs.get('agg_func')
                title = final_kwargs.pop('title', metric_name)
                calc_func = final_kwargs.pop('calc_func')
                resolve_calc_func = final_kwargs.pop('resolve_calc_func', True)
                post_calc_func = final_kwargs.pop('post_calc_func', None)
                use_caching = final_kwargs.pop('use_caching', True)

                # Resolve calc_func
                if resolve_calc_func:
                    if not callable(calc_func):
                        passed_kwargs_out = {}

                        def _getattr_func(obj: tp.Any,
                                          attr: str,
                                          args: tp.ArgsLike = None,
                                          kwargs: tp.KwargsLike = None,
                                          call_attr: bool = True,
                                          _custom_arg_names: tp.Set[str] = custom_arg_names,
                                          _arg_cache_dct: tp.Kwargs = arg_cache_dct,
                                          _final_kwargs: tp.Kwargs = final_kwargs) -> tp.Any:
                            if args is None:
                                args = ()
                            if kwargs is None:
                                kwargs = {}
                            if obj is custom_reself and _final_kwargs.pop('resolve_' + attr, True):
                                if call_attr:
                                    return custom_reself.resolve_attr(
                                        attr,
                                        args=args,
                                        cond_kwargs=_final_kwargs,
                                        kwargs=kwargs,
                                        custom_arg_names=_custom_arg_names,
                                        cache_dct=_arg_cache_dct,
                                        use_caching=use_caching,
                                        passed_kwargs_out=passed_kwargs_out
                                    )
                                return getattr(obj, attr)
                            out = getattr(obj, attr)
                            if callable(out) and call_attr:
                                return out(*args, **kwargs)
                            return out

                        calc_func = custom_reself.deep_getattr(
                            calc_func,
                            getattr_func=_getattr_func,
                            call_last_attr=False
                        )

                        if 'group_by' in passed_kwargs_out:
                            if 'pass_group_by' not in final_kwargs:
                                final_kwargs.pop('group_by', None)
                    if not callable(calc_func):
                        raise TypeError("calc_func must be callable")

                    # Resolve arguments
                    func_arg_names = get_func_arg_names(calc_func)
                    for k in func_arg_names:
                        if k not in final_kwargs:
                            if final_kwargs.pop('resolve_' + k, True):
                                try:
                                    arg_out = custom_reself.resolve_attr(
                                        k,
                                        cond_kwargs=final_kwargs,
                                        custom_arg_names=custom_arg_names,
                                        cache_dct=arg_cache_dct,
                                        use_caching=use_caching
                                    )
                                except AttributeError:
                                    continue
                                final_kwargs[k] = arg_out
                    for k in list(final_kwargs.keys()):
                        if k in opt_arg_names:
                            if 'pass_' + k in final_kwargs:
                                if not final_kwargs.get('pass_' + k):  # first priority
                                    final_kwargs.pop(k, None)
                            elif k not in func_arg_names:  # second priority
                                final_kwargs.pop(k, None)
                    for k in list(final_kwargs.keys()):
                        if k.startswith('pass_'):
                            final_kwargs.pop(k, None)  # cleanup

                    # Call calc_func
                    out = calc_func(**final_kwargs)
                else:
                    # Do not resolve calc_func
                    out = calc_func(custom_reself, metric_settings)

                # Call post_calc_func
                if post_calc_func is not None:
                    out = post_calc_func(custom_reself, out, metric_settings)

                # Post-process and store the metric
                if not isinstance(out, dict):
                    out = {None: out}
                for k, v in out.items():
                    if k is None:
                        t = title
                    elif title is None:
                        t = str(k)
                    else:
                        t = title + ': ' + str(k)
                    if checks.is_any_array(v) and not checks.is_series(v):
                        raise TypeError("calc_func must return either a scalar for one column/group, "
                                        "pd.Series for multiple columns/groups, or a dict of such. "
                                        f"Not {type(v)}.")
                    if checks.is_series(v):
                        if _column is not None:
                            v = custom_reself.select_one_from_obj(
                                v, custom_reself.wrapper.regroup(_group_by), column=_column)
                        elif _agg_func is not None and agg_func is not None:
                            v = _agg_func(v)
                        elif _agg_func is None and agg_func is not None:
                            if not silence_warnings:
                                warnings.warn(f"Metric '{metric_name}' returned multiple values "
                                              f"despite having no aggregation function", stacklevel=2)
                            continue
                    if t in stats_dct:
                        if not silence_warnings:
                            warnings.warn(f"Duplicate metric title '{t}'", stacklevel=2)
                    stats_dct[t] = v
            except Exception as e:
                warnings.warn(f"Metric '{metric_name}' raised an exception", stacklevel=2)
                raise e

        # Return the stats
        if reself.wrapper.get_ndim(group_by=group_by) == 1:
            return pd.Series(stats_dct, name=reself.wrapper.get_name(group_by=group_by))
        if column is not None:
            return pd.Series(stats_dct, name=column)
        if agg_func is not None:
            if not silence_warnings:
                warnings.warn(f"Object has multiple columns. Aggregating using {agg_func}.", stacklevel=2)
            return pd.Series(stats_dct, name='agg_func_' + agg_func.__name__)
        new_index = reself.wrapper.grouper.get_columns(group_by=group_by)
        stats_df = pd.DataFrame(stats_dct, index=new_index)
        return stats_df
