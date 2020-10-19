"""Named tuples and enumerated types."""

from collections import namedtuple
import json

__pdoc__ = {}

# ############# StopType ############# #

StopType = namedtuple('StopType', [
    'StopLoss',
    'TrailStop',
    'TakeProfit'
])(*range(3))
"""_"""

__pdoc__['StopType'] = f"""Stop type.

```plaintext
{json.dumps(dict(zip(StopType._fields, StopType)), indent=2)}
```
"""
