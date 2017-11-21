from vectorbt.optimizer.gridsearch import mapper


def from_func(func, params):
    """Apply KPI and pack into Series"""
    return dict(zip(params, mapper.map(func, params)))
