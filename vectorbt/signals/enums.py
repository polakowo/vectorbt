"""Named tuples and enumerated types.

Defines enums and other schemas for `vectorbt.signals`."""

import json

from vectorbt import typing as tp

__all__ = [
    'StopType'
]

__pdoc__ = {}


class StopTypeT(tp.NamedTuple):
    StopLoss: int
    TrailStop: int
    TakeProfit: int


StopType = StopTypeT(*range(3))
"""_"""

__pdoc__['StopType'] = f"""Stop type.

```plaintext
{json.dumps(dict(zip(StopType._fields, StopType)), indent=2, default=str)}
```
"""
