from functools import wraps

def expand_dims(func):
    @wraps(func)
    def wrapper_decorator(self, *args, **kwargs):
        """If 1D array passed, expand its dimensions and then collapse them back once returned."""
        if self.ndim == 1:
            obj = func(self[:, None], *args, **kwargs)[:, 0]
            obj.columns = None
            return obj
        else:
            return func(self, *args, **kwargs)
    return wrapper_decorator

def requires_1dim(func):
    @wraps(func)
    def wrapper_decorator(self, *args, **kwargs):
        """If 2D array passed, raise an exception."""
        if self.ndim == 1:
            return func(self, *args, **kwargs)
        else:
            raise ValueError("You must select a column (use select_column)")
    return wrapper_decorator