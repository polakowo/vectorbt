"""Named tuples and enumerated types.

Defines enums and other schemas for `vectorbt.signals`."""

from vectorbt import _typing as tp
from vectorbt.utils.docs import to_doc

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
{to_doc(StopType)}
```
"""
