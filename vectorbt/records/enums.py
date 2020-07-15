"""Named tuples and enumerated types."""

import numpy as np
from collections import namedtuple
import json

__pdoc__ = {}

# ############# Drawdown ############# #

DrawdownStatus = namedtuple('DrawdownStatus', [
    'Active',
    'Recovered'
])(*range(2))
"""_"""

__pdoc__['DrawdownStatus'] = f"""Drawdown status.

```plaintext
{json.dumps(dict(zip(DrawdownStatus._fields, DrawdownStatus)), indent=2)}
```
"""

drawdown_dt = np.dtype([
    ('col', np.int64),
    ('start_idx', np.int64),
    ('valley_idx', np.int64),
    ('end_idx', np.int64),
    ('status', np.int64),
], align=True)
"""_"""

__pdoc__['drawdown_dt'] = f"""`np.dtype` of drawdown records.

```plaintext
{json.dumps(dict(zip(
    dict(drawdown_dt.fields).keys(),
    list(map(lambda x: str(x[0]), dict(drawdown_dt.fields).values()))
)), indent=2)}
```
"""

# ############# Order ############# #

OrderSide = namedtuple('OrderSide', [
    'Buy',
    'Sell'
])(*range(2))
"""_"""

__pdoc__['OrderSide'] = f"""Order side.

```plaintext
{json.dumps(dict(zip(OrderSide._fields, OrderSide)), indent=2)}
```
"""

order_dt = np.dtype([
    ('col', np.int64),
    ('idx', np.int64),
    ('size', np.float64),
    ('price', np.float64),
    ('fees', np.float64),
    ('side', np.int64),
], align=True)
"""_"""

__pdoc__['order_dt'] = f"""`np.dtype` of order records.

```plaintext
{json.dumps(dict(zip(
    dict(order_dt.fields).keys(),
    list(map(lambda x: str(x[0]), dict(order_dt.fields).values()))
)), indent=2)}
```
"""

# ############# Event ############# #

EventStatus = namedtuple('EventStatus', [
    'Open',
    'Closed'
])(*range(2))
"""_"""

__pdoc__['EventStatus'] = f"""Event status.

```plaintext
{json.dumps(dict(zip(EventStatus._fields, EventStatus)), indent=2)}
```
"""

_event_fields = [
    ('col', np.int64),
    ('size', np.float64),
    ('open_idx', np.int64),
    ('open_price', np.float64),
    ('open_fees', np.float64),
    ('close_idx', np.int64),
    ('close_price', np.float64),
    ('close_fees', np.float64),
    ('pnl', np.float64),
    ('return', np.float64),
    ('status', np.int64)
]

event_dt = np.dtype(_event_fields, align=True)
"""_"""

__pdoc__['event_dt'] = f"""`np.dtype` of event records.

```plaintext
{json.dumps(dict(zip(
    dict(event_dt.fields).keys(),
    list(map(lambda x: str(x[0]), dict(event_dt.fields).values()))
)), indent=2)}
```
"""

# ############# Trade ############# #

trade_dt = np.dtype([
    *_event_fields,
    ('position_idx', np.int64)
], align=True)
"""_"""

__pdoc__['trade_dt'] = f"""`np.dtype` of trade records. Follows `event_dt`.

```plaintext
{json.dumps(dict(zip(
    dict(trade_dt.fields).keys(),
    list(map(lambda x: str(x[0]), dict(trade_dt.fields).values()))
)), indent=2)}
```
"""

# ############# Position ############# #

position_dt = np.dtype([
    *_event_fields
], align=True)
"""_"""

__pdoc__['position_dt'] = f"""`np.dtype` of position records. Follows `event_dt`.

```plaintext
{json.dumps(dict(zip(
    dict(position_dt.fields).keys(),
    list(map(lambda x: str(x[0]), dict(position_dt.fields).values()))
)), indent=2)}
```
"""
