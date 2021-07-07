"""Class for building plots out of subplots."""

from collections import Counter
import warnings

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.config import Config, merge_dicts, get_func_arg_names
from vectorbt.utils.figure import make_subplots, get_domain
from vectorbt.utils.datetime import freq_to_timedelta
from vectorbt.utils.template import deep_substitute
from vectorbt.base.array_wrapper import Wrapping


class PlotBuilderMixin:
    """Class that implements `PlotBuilderMixin.plot`.

    This class has a similar structure to that of `vectorbt.generic.stats_builder.StatsBuilderMixin`."""

    def __init__(self):
        checks.assert_type(self, Wrapping)

        # Copy writeable attrs
        self.subplots = self.__class__.subplots.copy()

    @property
    def writeable_attrs(self) -> tp.Set[str]:
        """Set of writeable attributes that will be saved/copied along with the config."""
        return {'subplots'}

    @property
    def plot_defaults(self) -> tp.Kwargs:
        """Defaults for `PlotBuilderMixin.plot`.

        See `plot_builder` in `vectorbt._settings.settings`."""
        from vectorbt._settings import settings
        plot_builder_cfg = settings['plot_builder']

        return dict(
            subplots=plot_builder_cfg['subplots'],
            grouped_subplots=plot_builder_cfg['grouped_subplots'],
            show_titles=plot_builder_cfg['show_titles'],
            hide_id_labels=plot_builder_cfg['hide_id_labels'],
            group_id_labels=plot_builder_cfg['group_id_labels'],
            make_subplots_kwargs=plot_builder_cfg['make_subplots_kwargs'],
            silence_warnings=plot_builder_cfg['silence_warnings'],
            template_mapping=plot_builder_cfg['template_mapping'],
            global_settings=plot_builder_cfg['global_settings'],
            kwargs=plot_builder_cfg['kwargs']
        )

    @property
    def plot_res_settings(self) -> tp.Kwargs:
        """Resolution settings for `PlotBuilderMixin.plot`.

        See `plot_builder` in `vectorbt._settings.settings`."""
        from vectorbt._settings import settings
        plot_builder_cfg = settings['plot_builder']

        return dict(
            freq=self.wrapper.freq,
            use_caching=plot_builder_cfg['use_caching'],
            hline_shape_kwargs=plot_builder_cfg['hline_shape_kwargs']
        )

    subplots: tp.ClassVar[Config] = Config(
        dict(),
        copy_kwargs=dict(copy_mode='deep')
    )
    """Subplots supported by `PlotBuilderMixin.plot`.

    !!! note
        It's safe to change this config - it's a (deep) copy of the class variable.
                
        But copying `PlotBuilderMixin` using `PlotBuilderMixin.copy` won't create a copy of the config."""

    def plot(self,
             subplots: tp.Optional[tp.MaybeIterable[tp.Union[str, tp.Tuple[str, tp.Kwargs]]]] = None,
             column: tp.Optional[tp.Label] = None,
             group_by: tp.GroupByLike = None,
             show_titles: bool = None,
             hide_id_labels: bool = None,
             group_id_labels: bool = None,
             make_subplots_kwargs: tp.KwargsLike = None,
             silence_warnings: bool = None,
             template_mapping: tp.Optional[tp.Mapping] = None,
             global_settings: tp.DictLike = None,
             **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot various parts of this object.

        Args:
            subplots (str, tuple, iterable, or dict): List of subplots to plot.

                Each element can be either:

                * a subplot name (see keys in `PlotBuilderMixin.subplots`)
                * a tuple of a subplot name and a settings dict as in `PlotBuilderMixin.subplots`.

                Each settings dict can contain the following keys:

                * `title`: title of the subplot. Defaults to None.
                * `yaxis_title`: title of the y-axis. Defaults to `title`.
                * `xaxis_title`: title of the x-axis. Defaults to 'Date'.
                * `allow_grouped`: whether this subplot supports grouped data. Defaults to True.
                    Must be known beforehand and cannot be provided as a template.
                * `plot_func`: plotting function for custom subplots. If the function can be accessed
                    by traversing attributes of this object, you can pass the path to this function
                    as a string (see `vectorbt.utils.attr.deep_getattr` for the path format).
                    Should write the supplied figure in-place and can return anything (it won't be used).
                * `pass_{arg}`: whether to pass a resolution argument (see below). Defaults to True if
                    this argument was found in the function's signature. Set to False to not pass.
                * `glob_pass_{arg}`: whether to pass an argument from `global_settings`. Defaults to True if
                    this argument was found both in `global_settings` and the function's signature.
                    Set to False to not pass.
                * `resolve_{arg}`: whether to resolve an argument that is meant to be an attribute of
                    the object (see `PlotBuilderMixin.resolve_attr`). Defaults to True if this argument was found
                    in the function's signature. Set to False to not resolve.
                * `template_mapping`: mapping to replace templates in subplot settings and keyword arguments.
                    Used across all settings.
                * Any other keyword argument overrides resolution arguments or is passed directly to `plot_func`.

                A plotting function may accept any keyword argument, but it should accept the current figure via
                a `fig` keyword argument. It may also "request" any of the following resolution arguments by
                accepting them or if `pass_{arg}` was found in the settings dict:

                * Each of `PlotBuilderMixin.self_aliases`: original object (ungrouped, with no column selected)
                * `column`
                * `group_by`
                * `subplot_name`
                * `trace_names`: list with the subplot name
                * `add_trace_kwargs`
                * `xref`
                * `yref`
                * `xaxis`
                * `yaxis`
                * `x_domain`
                * `y_domain`
                * Any argument from `PlotBuilderMixin.plot_res_settings`
                * Any attribute of this object if it meant to be resolved (see `PlotBuilderMixin.resolve_attr`)

                Pass `subplots='all'` to plot all supported subplots.
            column (str): Name of the column/group to plot.

                Won't have effect on this object, but passed down to each plotting function.
            group_by (any): Group or ungroup columns. See `vectorbt.base.column_grouper.ColumnGrouper`.

                Won't have effect on this object, but passed down to each plotting function.
            show_titles (bool): Whether to show the title of each subplot.
            hide_id_labels (bool): Whether to hide identical legend labels.

                Two labels are identical if their name, marker style and line style match.
            group_id_labels (bool): Whether to group identical legend labels.
            make_subplots_kwargs (dict): Keyword arguments passed to `plotly.subplots.make_subplots`.
            silence_warnings (bool): Whether to silence all warnings.
            template_mapping (mapping): Global mapping to replace templates.

                Applied on `PlotBuilderMixin.plot_res_settings`, `global_settings`, and `kwargs`.
            global_settings (dict): Keyword arguments that override default settings for each subplot.
                Additionally, passes any argument that has the matching key in the signature of `plot_func`.
                Use `glob_pass_{arg}` to force or ignore passing an argument.
            **kwargs: Additional keyword arguments.

                Can contain keyword arguments for each subplot, specified as `{subplot_name}_kwargs`.
                Can also contain keyword arguments that override arguments from `PlotBuilderMixin.plot_res_settings`.
                Other keyword arguments are used to update the layout of the figure.

        For template logic, see `vectorbt.utils.template`.

        For defaults, see `PlotBuilderMixin.plot_defaults`.
        
        !!! hint
            Make sure to resolve and then to re-use as many object attributes as possible to
            utilize built-in caching (even if global caching is disabled).
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        # Resolve defaults
        plot_res_settings = self.plot_res_settings
        for k in list(kwargs.keys()):
            if k in plot_res_settings:
                plot_res_settings[k] = kwargs.pop(k)
        make_subplots_kwargs = merge_dicts(self.plot_defaults['make_subplots_kwargs'], make_subplots_kwargs)
        template_mapping = merge_dicts(self.plot_defaults['template_mapping'], template_mapping)
        global_settings = merge_dicts(self.plot_defaults['global_settings'], global_settings)
        kwargs = merge_dicts(self.plot_defaults['kwargs'], kwargs)

        # Check if grouped
        is_grouped = self.wrapper.grouper.is_grouped(group_by=group_by)

        # Replace templates globally
        if len(template_mapping) > 0:
            plot_res_settings = deep_substitute(plot_res_settings, mapping=template_mapping)
            global_settings = deep_substitute(global_settings, mapping=template_mapping)
            kwargs = deep_substitute(kwargs, mapping=template_mapping)

        # Prepare subplots
        if subplots is None:
            subplots = self.plot_defaults['subplots']
            if is_grouped:
                grouped_subplots = self.plot_defaults['grouped_subplots']
                if grouped_subplots is None:
                    grouped_subplots = subplots
                subplots = grouped_subplots
        if subplots == 'all':
            subplots = self.subplots
        if isinstance(subplots, dict):
            subplots = list(subplots.items())
        if isinstance(subplots, (str, tuple)):
            subplots = [subplots]
        # Bring to the same shape
        new_subplots = []
        for i, subplot in enumerate(subplots):
            if isinstance(subplot, str):
                subplot = (subplot, self.subplots[subplot])
            if not isinstance(subplot, tuple):
                raise TypeError(f"Subplot at index {i} must be either a string or a tuple")
            new_subplots.append(subplot)
        subplots = new_subplots
        # Handle duplicate names
        subplot_counts = Counter(list(map(lambda x: x[0], subplots)))
        subplot_i = {k: -1 for k in subplot_counts.keys()}
        subplots_dct = {}
        for i, (subplot_name, subplot_defaults) in enumerate(subplots):
            if subplot_counts[subplot_name] > 1:
                subplot_i[subplot_name] += 1
                subplot_name = subplot_name + '_' + str(subplot_i[subplot_name])
            subplots_dct[subplot_name] = subplot_defaults
        # Merge settings
        custom_arg_names_dct = {}
        for subplot_name, subplot_defaults in subplots_dct.items():
            passed_settings = kwargs.pop(f'{subplot_name}_kwargs', {})
            subplots_dct[subplot_name] = merge_dicts(
                subplot_defaults,
                global_settings,
                passed_settings
            )
            custom_arg_names_dct[subplot_name] = set(subplot_defaults.keys()).union(set(passed_settings.keys()))
        # Filter subplots
        if is_grouped:
            left_out_names = []
            for subplot_name in list(subplots_dct.keys()):
                if not subplots_dct[subplot_name].get('allow_grouped', True):
                    subplots_dct.pop(subplot_name, None)
                    custom_arg_names_dct.pop(subplot_name, None)
                    left_out_names.append(subplot_name)
            if len(left_out_names) > 0 and not silence_warnings:
                warnings.warn(f"Subplots {left_out_names} do not support grouped data", stacklevel=2)
        if len(subplots_dct) == 0:
            raise ValueError("There is no subplot to plot")

        # Set up figure
        rows = make_subplots_kwargs.pop('rows', len(subplots_dct))
        cols = make_subplots_kwargs.pop('cols', 1)
        specs = make_subplots_kwargs.pop('specs', [[{} for _ in range(cols)] for _ in range(rows)])
        row_col_tuples = []
        for row, row_spec in enumerate(specs):
            for col, col_spec in enumerate(row_spec):
                if col_spec is not None:
                    row_col_tuples.append((row + 1, col + 1))
        shared_xaxes = make_subplots_kwargs.pop('shared_xaxes', True)
        shared_yaxes = make_subplots_kwargs.pop('shared_yaxes', False)
        default_height = plotting_cfg['layout']['height']
        default_width = plotting_cfg['layout']['width'] + 50
        min_space = 10  # space between subplots with no axis sharing
        max_title_spacing = 30
        max_xaxis_spacing = 50
        max_yaxis_spacing = 100
        legend_height = 50
        if show_titles:
            title_spacing = max_title_spacing
        else:
            title_spacing = 0
        if not shared_xaxes and rows > 1:
            xaxis_spacing = max_xaxis_spacing
        else:
            xaxis_spacing = 0
        if not shared_yaxes and cols > 1:
            yaxis_spacing = max_yaxis_spacing
        else:
            yaxis_spacing = 0
        if 'height' in kwargs:
            height = kwargs.pop('height')
        else:
            height = default_height + title_spacing
            if rows > 1:
                height *= rows
                height += min_space * rows - min_space
                height += legend_height - legend_height * rows
                if shared_xaxes:
                    height += max_xaxis_spacing - max_xaxis_spacing * rows
        if 'width' in kwargs:
            width = kwargs.pop('width')
        else:
            width = default_width
            if cols > 1:
                width *= cols
                width += min_space * cols - min_space
                if shared_yaxes:
                    width += max_yaxis_spacing - max_yaxis_spacing * cols
        if height is not None:
            if 'vertical_spacing' in make_subplots_kwargs:
                vertical_spacing = make_subplots_kwargs.pop('vertical_spacing')
            else:
                vertical_spacing = min_space + title_spacing + xaxis_spacing
            if vertical_spacing is not None and vertical_spacing > 1:
                vertical_spacing /= height
            legend_y = 1 + (min_space + title_spacing) / height
        else:
            vertical_spacing = make_subplots_kwargs.pop('vertical_spacing', None)
            legend_y = 1.02
        if width is not None:
            if 'horizontal_spacing' in make_subplots_kwargs:
                horizontal_spacing = make_subplots_kwargs.pop('horizontal_spacing')
            else:
                horizontal_spacing = min_space + yaxis_spacing
            if horizontal_spacing is not None and horizontal_spacing > 1:
                horizontal_spacing /= width
        else:
            horizontal_spacing = make_subplots_kwargs.pop('horizontal_spacing', None)
        if show_titles:
            _subplot_titles = []
            for i in range(len(subplots_dct)):
                _subplot_titles.append('$title_' + str(i))
        else:
            _subplot_titles = None
        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=specs,
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes,
            subplot_titles=_subplot_titles,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing,
            **make_subplots_kwargs
        )
        kwargs = merge_dicts(dict(
            width=width,
            height=height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=legend_y,
                xanchor="right",
                x=1,
                traceorder='normal'
            )
        ), kwargs)
        fig.update_layout(**kwargs)  # final destination for kwargs

        # Show subplots
        arg_cache_dct = {}
        for i, (subplot_name, subplot_defaults) in enumerate(subplots_dct.items()):
            final_settings = subplot_defaults.copy()
            final_settings.pop('allow_grouped', None)

            # Compute figure artifacts
            row, col = row_col_tuples[i]
            xref = 'x' if i == 0 else 'x' + str(i + 1)
            yref = 'y' if i == 0 else 'y' + str(i + 1)
            xaxis = 'xaxis' + xref[1:]
            yaxis = 'yaxis' + yref[1:]
            x_domain = get_domain(xref, fig)
            y_domain = get_domain(yref, fig)

            # Replace templates
            res_settings = merge_dicts(
                {name: self for name in self.self_aliases},
                dict(
                    column=column,
                    group_by=group_by,
                    subplot_name=subplot_name,
                    trace_names=[subplot_name],
                    add_trace_kwargs=dict(row=row, col=col),
                    xref=xref,
                    yref=yref,
                    xaxis=xaxis,
                    yaxis=yaxis,
                    x_domain=x_domain,
                    y_domain=y_domain,
                    fig=fig
                ),
                plot_res_settings
            )
            res_arg_names = set(res_settings.keys())
            res_arg_names.remove('fig')
            final_settings = merge_dicts(res_settings, final_settings)
            subplot_template_mapping = final_settings.pop('template_mapping', {})
            mapping = merge_dicts(final_settings, template_mapping, subplot_template_mapping)
            final_settings = deep_substitute(final_settings, mapping=mapping)
            if final_settings['freq'] is not None:
                final_settings['freq'] = freq_to_timedelta(final_settings['freq'])
            for name in self.self_aliases:
                final_settings[name] = self.resolve_self(name, final_settings)

            # Pop values
            title = final_settings.pop('title', None)
            plot_func = final_settings.pop('plot_func', None)
            xaxis_title = final_settings.pop('xaxis_title', 'Date')
            yaxis_title = final_settings.pop('yaxis_title', title)

            # Prepare function and keyword arguments
            if plot_func is not None:
                # Prepare function and keyword arguments
                custom_arg_names = custom_arg_names_dct[subplot_name]
                if not callable(plot_func):
                    def _getattr_func(obj: tp.Any,
                                      attr: str,
                                      args: tp.ArgsLike = None,
                                      kwargs: tp.KwargsLike = None,
                                      call_attr: bool = True,
                                      _custom_arg_names: tp.Set[str] = custom_arg_names,
                                      _arg_cache_dct: tp.Kwargs = arg_cache_dct,
                                      _final_settings: tp.Kwargs = final_settings) -> tp.Any:
                        if args is None:
                            args = ()
                        if kwargs is None:
                            kwargs = {}
                        if obj is self and _final_settings.pop('resolve_' + attr, True):
                            if call_attr:
                                return self.resolve_attr(
                                    attr,
                                    args=args,
                                    cond_kwargs=_final_settings,
                                    kwargs=kwargs,
                                    custom_arg_names=_custom_arg_names,
                                    cache_dct=_arg_cache_dct
                                )
                            return getattr(obj, attr)
                        out = getattr(obj, attr)
                        if callable(out) and call_attr:
                            return out(*args, **kwargs)
                        return out

                    plot_func = self.getattr(plot_func, getattr_func=_getattr_func, call_last_attr=False)
                if not callable(plot_func):
                    raise TypeError("calc_func must be callable")

                func_arg_names = get_func_arg_names(plot_func)
                for k in func_arg_names:
                    if k not in final_settings:
                        if final_settings.pop('resolve_' + k, True):
                            try:
                                arg_out = self.resolve_attr(
                                    k,
                                    cond_kwargs=final_settings,
                                    custom_arg_names=custom_arg_names,
                                    cache_dct=arg_cache_dct
                                )
                            except AttributeError:
                                continue
                            final_settings[k] = arg_out

                for k in res_arg_names:
                    if 'pass_' + k in final_settings:
                        if not final_settings.pop('pass_' + k):  # first priority
                            final_settings.pop(k, None)
                    elif k not in func_arg_names:  # second priority
                        final_settings.pop(k, None)
                for k in list(final_settings.keys()):
                    if 'glob_pass_' + k in final_settings:
                        if k not in global_settings or not final_settings.pop('glob_pass_' + k, True):
                            final_settings.pop(k, None)  # global setting should not be utilized
                    else:
                        if k in global_settings and k not in custom_arg_names and k not in func_arg_names:
                            final_settings.pop(k, None)  # global setting not utilized
                for k in list(final_settings.keys()):
                    if k.startswith('glob_pass_'):
                        final_settings.pop(k, None)  # cleanup

                # Call plotting function
                plot_func(**final_settings)

            # Update global layout
            for annotation in fig.layout.annotations:
                if 'text' in annotation and annotation['text'] == '$title_' + str(i):
                    annotation['text'] = title
            fig.layout[xaxis]['title'] = xaxis_title
            fig.layout[yaxis]['title'] = yaxis_title

        # Remove duplicate legend labels
        found_ids = dict()
        unique_idx = 0
        for trace in fig.data:
            if 'name' in trace:
                name = trace['name']
            else:
                name = None
            if 'marker' in trace:
                marker = trace['marker']
            else:
                marker = {}
            if 'symbol' in marker:
                marker_symbol = marker['symbol']
            else:
                marker_symbol = None
            if 'color' in marker:
                marker_color = marker['color']
            else:
                marker_color = None
            if 'line' in trace:
                line = trace['line']
            else:
                line = {}
            if 'dash' in line:
                line_dash = line['dash']
            else:
                line_dash = None
            if 'color' in line:
                line_color = line['color']
            else:
                line_color = None

            id = (name, marker_symbol, marker_color, line_dash, line_color)
            if id in found_ids:
                if hide_id_labels:
                    trace['showlegend'] = False
                if group_id_labels:
                    trace['legendgroup'] = found_ids[id]
            else:
                if group_id_labels:
                    trace['legendgroup'] = unique_idx
                found_ids[id] = unique_idx
                unique_idx += 1

        # Remove all except the last title if sharing the same axis
        if shared_xaxes:
            i = 0
            for row in range(rows):
                for col in range(cols):
                    if specs[row][col] is not None:
                        xaxis = 'xaxis' if i == 0 else 'xaxis' + str(i + 1)
                        if row < rows - 1:
                            fig.layout[xaxis]['title'] = None
                        i += 1
        if shared_yaxes:
            i = 0
            for row in range(rows):
                for col in range(cols):
                    if specs[row][col] is not None:
                        yaxis = 'yaxis' if i == 0 else 'yaxis' + str(i + 1)
                        if col > 0:
                            fig.layout[yaxis]['title'] = None
                        i += 1

        return fig