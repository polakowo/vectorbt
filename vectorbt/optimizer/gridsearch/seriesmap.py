from vectorbt.optimizer.gridsearch import mapper


def from_func(func, params):
    """Build a list of tuples out of parameter combinations and function outputs"""
    return list(zip(params, mapper.map(func, params)))
