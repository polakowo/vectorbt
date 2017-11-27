import numpy as np

def from_func(func, params, multiprocess=False):
    """Build a list of tuples out of parameter combinations and function outputs"""
    from vectorbt.optimizer.gridsearch import mapper
    # CAUTION: Multiprocessing only slows down generation of large series
    return dict(zip(params, mapper.map(func, params, multiprocess=multiprocess)))
