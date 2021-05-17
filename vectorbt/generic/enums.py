"""Named tuples and enumerated types.

Defines enums and other schemas for `vectorbt.generic`."""

import numpy as np

from vectorbt import _typing as tp
from vectorbt.utils.docs import to_doc

__all__ = [
    'DrawdownStatus',
    'drawdown_dt'
]

__pdoc__ = {}


# ############# Records ############# #

class DrawdownStatusT(tp.NamedTuple):
    Active: int
    Recovered: int


DrawdownStatus = DrawdownStatusT(*range(2))
"""_"""

__pdoc__['DrawdownStatus'] = f"""Drawdown status.

```json
{to_doc(DrawdownStatus)}
```
"""

drawdown_dt = np.dtype([
    ('id', np.int_),
    ('col', np.int_),
    ('start_idx', np.int_),
    ('valley_idx', np.int_),
    ('end_idx', np.int_),
    ('status', np.int_),
], align=True)
"""_"""

__pdoc__['drawdown_dt'] = f"""`np.dtype` of drawdown records.

```json
{to_doc(drawdown_dt)}
```
"""
