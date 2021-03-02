"""Utilities for working with parameters."""

from numba.typed import List
import itertools

from vectorbt.utils import checks


def to_typed_list(lst):
    """Cast Python list to typed list.

    Direct construction is flawed in Numba 0.52.0.
    See https://github.com/numba/numba/issues/6651."""
    nb_lst = List()
    for elem in lst:
        nb_lst.append(elem)
    return nb_lst


def flatten_param_tuples(param_tuples):
    """Flattens a nested list of tuples using unzipping."""
    param_list = []
    unzipped_tuples = zip(*param_tuples)
    for i, unzipped in enumerate(unzipped_tuples):
        unzipped = list(unzipped)
        if isinstance(unzipped[0], tuple):
            param_list.extend(flatten_param_tuples(unzipped))
        else:
            param_list.append(unzipped)
    return param_list


def create_param_combs(op_tree, depth=0):
    """Create arbitrary parameter combinations from the operation tree `op_tree`.

    `op_tree` must be a tuple of tuples, each being an instruction to generate parameters.
    The first element of each tuple should a function that takes remaining elements as arguments.
    If one of the elements is a tuple, it will be unfolded in the same way.

    ## Example

    ```python-repl
    >>> import numpy as np
    >>> from itertools import combinations, product

    >>> create_param_combs((product, (combinations, [0, 1, 2, 3], 2), [4, 5]))
    [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2],
     [1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 3, 3],
     [4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5]]
    ```
    """
    checks.assert_type(op_tree, tuple)
    new_op_tree = (op_tree[0],)
    for elem in op_tree[1:]:
        if isinstance(elem, tuple):
            new_op_tree += (create_param_combs(elem, depth=depth + 1),)
        else:
            new_op_tree += (elem,)
    out = list(new_op_tree[0](*new_op_tree[1:]))
    if depth == 0:
        # do something
        return flatten_param_tuples(out)
    return out


def broadcast_params(param_list, to_n=None):
    """Broadcast parameters in `param_list`."""
    if to_n is None:
        to_n = max(list(map(len, param_list)))
    new_param_list = []
    for i in range(len(param_list)):
        params = param_list[i]
        if len(params) in [1, to_n]:
            if len(params) < to_n:
                new_param_list.append(list(params * to_n))
            else:
                new_param_list.append(list(params))
        else:
            raise ValueError(f"Parameters at index {i} have length {len(params)} that cannot be broadcast to {to_n}")
    return new_param_list


def create_param_product(param_list):
    """Make Cartesian product out of all params in `param_list`."""
    return list(map(list, zip(*list(itertools.product(*param_list)))))


class DefaultParam:
    """Class for wrapping default values."""

    def __repr__(self):
        return self.value.__repr__()

    def __init__(self, value):
        self.value = value
