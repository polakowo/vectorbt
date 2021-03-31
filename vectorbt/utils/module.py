"""Utilities for modules."""

import inspect
import sys
import importlib
import pkgutil


def is_from_module(obj, module):
    """Return whether `obj` is from module `module`."""
    mod = inspect.getmodule(inspect.unwrap(obj))
    return mod is None or mod.__name__ == module.__name__


def list_module_keys(module_name, whitelist=None, blacklist=None):
    """List the names of all public functions and classes defined in the module `module_name`.

    Includes the names listed in `whitelist` and excludes the names listed in `blacklist`."""
    if whitelist is None:
        whitelist = []
    if blacklist is None:
        blacklist = []
    module = sys.modules[module_name]
    return [name for name, obj in inspect.getmembers(module)
            if (not name.startswith("_") and is_from_module(obj, module)
                and ((inspect.isroutine(obj) and callable(obj)) or inspect.isclass(obj))
                and name not in blacklist) or name in whitelist]


def import_submodules(package):
    """Import all submodules of a module, recursively, including subpackages."""
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        results[name] = importlib.import_module(name)
        if is_pkg:
            results.update(import_submodules(name))
    return results
