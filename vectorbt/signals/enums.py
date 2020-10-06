"""Named tuples and enumerated types."""

from collections import namedtuple
import json

__pdoc__ = {}

# ############# StopPosition ############# #

StopPosition = namedtuple('StopPosition', [
    'Entry',
    'ExpMin',
    'ExpMax'
])(*range(3))
"""_"""

__pdoc__['StopPosition'] = f"""Stop position.

```plaintext
{json.dumps(dict(zip(StopPosition._fields, StopPosition)), indent=2)}
```
"""
