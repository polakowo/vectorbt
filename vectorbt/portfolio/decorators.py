"""Class and function decorators."""

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.config import Config, get_func_arg_names
from vectorbt.utils.decorators import cached_method

WrapperFuncT = tp.Callable[[tp.Type[tp.T]], tp.Type[tp.T]]


def add_returns_acc_methods(config: Config) -> WrapperFuncT:
    """Class decorator to add returns accessor methods.

    `config` should contain target method names (keys) and dictionaries (values) with the following keys:

    * `source_name`: Name of the source method. Defaults to the target name.
    * `docstring`: Method docstring. Defaults to "See `vectorbt.returns.accessors.ReturnsAccessor.{source_name}`.".

    The class should be a subclass of `vectorbt.portfolio.base.Portfolio`.
    """

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        checks.assert_subclass(cls, "Portfolio")

        for target_name, settings in config.items():
            source_name = settings.get('source_name', target_name)
            docstring = settings.get('docstring', f"See `vectorbt.returns.accessors.ReturnsAccessor.{source_name}`.")

            def new_method(self,
                           *,
                           group_by: tp.GroupByLike = None,
                           freq: tp.Optional[tp.FrequencyLike] = None,
                           year_freq: tp.Optional[tp.FrequencyLike] = None,
                           use_asset_returns: bool = False,
                           _source_name: str = source_name,
                           **kwargs) -> tp.Any:
                returns_acc = self.returns_acc(
                    group_by=group_by,
                    freq=freq,
                    year_freq=year_freq,
                    use_asset_returns=use_asset_returns
                )
                source_method = getattr(returns_acc, _source_name)
                if 'benchmark_rets' in kwargs or 'benchmark_rets' in get_func_arg_names(source_method):
                    benchmark_rets = kwargs.pop('benchmark_rets', None)
                    if benchmark_rets is None:
                        benchmark_rets = self.benchmark_returns(group_by=group_by)
                    kwargs['benchmark_rets'] = benchmark_rets
                return source_method(**kwargs)

            new_method.__name__ = target_name
            new_method.__qualname__ = f"{cls.__name__}.{target_name}"
            new_method.__doc__ = docstring
            setattr(cls, target_name, cached_method(new_method))
        return cls

    return wrapper
