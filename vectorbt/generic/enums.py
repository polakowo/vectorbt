# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Named tuples and enumerated types.

Defines enums and other schemas for `vectorbt.generic`."""

import numpy as np

from vectorbt import _typing as tp
from vectorbt.utils.docs import to_doc

__all__ = [
    'RangeStatus',
    'DrawdownStatus',
    'drawdown_dt',
    'range_dt'
]

__pdoc__ = {}


# ############# Enums ############# #

class RangeStatusT(tp.NamedTuple):
    Open: int
    Closed: int


RangeStatus = RangeStatusT(*range(2))
"""_"""

__pdoc__['RangeStatus'] = f"""Range status.

```json
{to_doc(RangeStatus)}
```
"""


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

# ############# Records ############# #

range_dt = np.dtype([
    ('id', np.int_),
    ('col', np.int_),
    ('start_idx', np.int_),
    ('end_idx', np.int_),
    ('status', np.int_)
], align=True)
"""_"""

__pdoc__['range_dt'] = f"""`np.dtype` of range records.

```json
{to_doc(range_dt)}
```
"""

drawdown_dt = np.dtype([
    ('id', np.int_),
    ('col', np.int_),
    ('peak_idx', np.int_),
    ('start_idx', np.int_),
    ('valley_idx', np.int_),
    ('end_idx', np.int_),
    ('peak_val', np.float_),
    ('valley_val', np.float_),
    ('end_val', np.float_),
    ('status', np.int_),
], align=True)
"""_"""

__pdoc__['drawdown_dt'] = f"""`np.dtype` of drawdown records.

```json
{to_doc(drawdown_dt)}
```
"""
