"""Common functions and classes."""


class ArrayWrapper():
    """Provides methods for wrapping NumPy arrays."""

    def __init__(self, ref_obj):
        self.ref_obj = ref_obj

    def wrap_array(self, a, **kwargs):
        """Wrap output array to the time series format of this portfolio."""
        return self.ref_obj.vbt.wrap_array(a, **kwargs)

    def wrap_reduced_array(self, a, **kwargs):
        """Wrap output array to the metric format of this portfolio."""
        return self.ref_obj.vbt.timeseries.wrap_reduced_array(a, **kwargs)
