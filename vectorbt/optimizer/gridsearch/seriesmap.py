import numpy as np

def from_func(func, params):
    """Generate series keyed by parameters"""
    from vectorbt.optimizer.gridsearch import mapper
    """Build a list of tuples out of parameter combinations and function outputs"""
    return list(zip(params, mapper.map(func, params)))
