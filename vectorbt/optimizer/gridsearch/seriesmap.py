def from_func(func, params):
    from vectorbt.optimizer.gridsearch import mapper
    """Build a list of tuples out of parameter combinations and function outputs"""
    return list(zip(params, mapper.map(func, params)))
