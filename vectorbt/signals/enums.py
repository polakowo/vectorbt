"""Named tuples and enumerated types."""

from collections import namedtuple
import json

__all__ = [
    'StopType'
]

__pdoc__ = {}

# We use namedtuple for enums and classes to be able to use them in Numba

StopType = namedtuple('StopType', [
    'StopLoss',
    'TrailStop',
    'TakeProfit'
])(*range(3))
"""_"""

__pdoc__['StopType'] = f"""Stop type.

```plaintext
{json.dumps(dict(zip(StopType._fields, StopType)), indent=2, default=str)}
```
"""
