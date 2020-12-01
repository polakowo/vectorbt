"""Utilities for configuration."""

import numpy as np
import pandas as pd


def merge_kwargs(*dicts):
    """Merge dictionaries `dicts`."""
    z = {}
    x, y = dicts[0], dicts[1]
    overlapping_keys = x.keys() & y.keys()
    for key in overlapping_keys:
        if isinstance(x[key], dict) and isinstance(y[key], dict):
            z[key] = merge_kwargs(x[key], y[key])
        else:
            z[key] = y[key]
    for key in x.keys() - overlapping_keys:
        z[key] = x[key]
    for key in y.keys() - overlapping_keys:
        z[key] = y[key]
    if len(dicts) > 2:
        return merge_kwargs(z, *dicts[2:])
    return z


class Config(dict):
    """A simple dict with (optionally) frozen keys."""

    def __init__(self, *args, frozen=True, read_only=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen = frozen
        self.read_only = read_only
        self.default_config = dict(self)
        for key, value in dict.items(self):
            if isinstance(value, dict):
                dict.__setitem__(self, key, Config(value, read_only=read_only, frozen=frozen))

    def __setitem__(self, key, val):
        if self.read_only:
            raise ValueError("Config is read-only")
        if self.frozen and key not in self:
            raise KeyError(f"Key {key} is not a valid parameter")
        dict.__setitem__(self, key, val)

    def merge_with(self, other):
        """Update with the elements and properties from the another config.

        Returns a new config."""
        read_only = other.read_only if isinstance(other, Config) else self.read_only
        frozen = other.frozen if isinstance(other, Config) else self.frozen
        return Config(read_only=read_only, frozen=frozen, **merge_kwargs(self, other))

    def reset(self):
        """Reset dictionary to the one passed at instantiation."""
        self.update(self.default_config)


class Configured:
    """Class with an initialization config."""

    def __init__(self, **config):
        self._config = Config(config, read_only=True)

    @property
    def config(self):
        """Initialization config (read-only)."""
        return self._config

    def copy(self, **new_config):
        """Create a new instance based on the config.

        !!! warning
            This "copy" operation won't return a copy of the instance but a new instance
            initialized with the same config. If the instance has writable attributes,
            their values won't be copied over."""
        return self.__class__(**self.config.merge_with(new_config))

    def __eq__(self, other):
        """Objects are equals if their configs are equal."""
        if type(self) != type(other):
            return False
        my_config = self.config
        other_config = other.config
        if my_config.keys() != other_config.keys():
            return False
        for k, v in my_config.items():
            other_v = other_config[k]
            if isinstance(v, pd.Series) or isinstance(other_v, pd.Series):
                try:
                    pd.testing.assert_series_equal(v, other_v)
                except:
                    return False
            elif isinstance(v, pd.DataFrame) or isinstance(other_v, pd.DataFrame):
                try:
                    pd.testing.assert_frame_equal(v, other_v)
                except:
                    return False
            elif isinstance(v, pd.Index) or isinstance(other_v, pd.Index):
                try:
                    pd.testing.assert_index_equal(v, other_v)
                except:
                    return False
            elif isinstance(v, np.ndarray) or isinstance(other_v, np.ndarray):
                if v.dtype.fields is not None and other_v.dtype.fields is not None:  # records
                    if v.dtype.fields != other_v.dtype.fields:
                        return False
                    for field in v.dtype.names:
                        try:
                            np.testing.assert_array_equal(v[field], other_v[field])
                        except:
                            return False
                else:
                    try:
                        np.testing.assert_array_equal(v, other_v)
                    except:
                        return False
            else:
                if v != other_v:
                    return False
        return True
