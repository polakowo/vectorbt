"""Named tuples and enumerated types.

Defines enums and other schemas for `vectorbt.signals`."""

import json

from vectorbt import _typing as tp

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

```json
{json.dumps(dict(zip(StopType._fields, StopType)), indent=4, default=str)}
```
"""
