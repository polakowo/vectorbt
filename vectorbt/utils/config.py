"""Utilities for configuration."""


class Config(dict):
    """A simple dict with (optionally) frozen keys."""

    def __init__(self, *args, frozen=True, read_only=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen = frozen
        self.read_only = read_only
        self.default_config = dict(self)
        for key, value in dict.items(self):
            if isinstance(value, dict):
                dict.__setitem__(self, key, Config(value, frozen=frozen))

    def __setitem__(self, key, val):
        if self.read_only:
            raise ValueError("Config is read-only")
        if self.frozen and key not in self:
            raise KeyError(f"Key {key} is not a valid parameter")
        dict.__setitem__(self, key, val)

    def reset(self):
        """Reset dictionary to the one passed at instantiation."""
        self.update(self.default_config)


def merge_kwargs(x, y):
    """Merge dictionaries `x` and `y`.

    By conflicts, `y` wins."""
    z = {}
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
    return z


class Configured:
    """Class with an initialization config."""
    def __init__(self, **config):
        self._config = Config(config, read_only=True)

    def copy(self, **new_config):
        """Create a new instance based on the config.

        !!! warning
            This "copy" operation won't return a copy of the instance but a new instance
            initialized with the same config. If the instance has writable attributes,
            their values won't be copied over."""
        return self.__class__(**merge_kwargs(self._config, new_config))
