import inspect
import re

import pytest

from vectorbt import _engine


RUST_ENGINE_EXPR_RE = re.compile(r"engine\s*=\s*['\"]rust['\"]")
RUST_UNAVAILABLE_SKIP = pytest.mark.skip(reason="vectorbt-rust is not installed or version-compatible")


def pytest_collection_modifyitems(items):
    if _engine.is_rust_available():
        return

    for item in items:
        try:
            source = inspect.getsource(item.obj)
        except (OSError, TypeError):
            continue
        if RUST_ENGINE_EXPR_RE.search(source):
            item.add_marker(RUST_UNAVAILABLE_SKIP)
