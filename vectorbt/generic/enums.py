"""Named tuples and enumerated types.

Defines enums and other schemas for `vectorbt.generic`."""

import numpy as np
import json

from vectorbt import typing as tp

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

```plaintext
{json.dumps(dict(zip(DrawdownStatus._fields, DrawdownStatus)), indent=2, default=str)}
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

```plaintext
{json.dumps(dict(zip(
    dict(drawdown_dt.fields).keys(),
    list(map(lambda x: str(x[0]), dict(drawdown_dt.fields).values()))
)), indent=2, default=str)}
```
"""
