"""Named tuples and enumerated types."""

import numpy as np
from collections import namedtuple
import json

__all__ = [
    'DrawdownStatus',
    'drawdown_dt'
]

__pdoc__ = {}

# We use namedtuple for enums and classes to be able to use them in Numba

# ############# Records ############# #

DrawdownStatus = namedtuple('DrawdownStatus', [
    'Active',
    'Recovered'
])(*range(2))
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
