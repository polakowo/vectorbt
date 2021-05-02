"""Named tuples and enumerated types.

Defines enums and other schemas for `vectorbt.labels`."""

import json

from vectorbt import typing as tp

__all__ = [
    'TrendMode'
]

__pdoc__ = {}


class TrendModeT(tp.NamedTuple):
    Binary: int
    BinaryCont: int
    BinaryContSat: int
    PctChange: int
    PctChangeNorm: int


TrendMode = TrendModeT(*range(5))
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
