"""Named tuples and enumerated types.

Defines enums and other schemas for `vectorbt.labels`."""

from collections import namedtuple
import json

__all__ = [
    'TrendMode'
]

__pdoc__ = {}

TrendMode = namedtuple('TrendMode', [
    'Binary',
    'BinaryCont',
    'BinaryContSat',
    'PctChange',
    'PctChangeNorm',
])(*range(5))
"""_"""

__pdoc__['TrendMode'] = f"""Trend mode.

```plaintext
{json.dumps(dict(zip(TrendMode._fields, TrendMode)), indent=2, default=str)}
```

Attributes:
    Binary: See `vectorbt.labels.nb.bn_trend_labels_nb`.
    BinaryCont: See `vectorbt.labels.nb.bn_cont_trend_labels_nb`.
    BinaryContSat: See `vectorbt.labels.nb.bn_cont_sat_trend_labels_nb`.
    PctChange: See `vectorbt.labels.nb.pct_trend_labels_nb`.
    PctChangeNorm: See `vectorbt.labels.nb.pct_trend_labels_nb` with `normalize` set to True.
"""
