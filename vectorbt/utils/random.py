"""Utilities for random number generation."""

import random
import numpy as np
from numba import njit


@njit(cache=True)
def set_seed_nb(seed):
    """Set seed in numba."""
    np.random.seed(seed)


def set_seed(seed):
    """Set seed."""
    random.seed(seed)
    np.random.seed(seed)
    set_seed_nb(seed)

