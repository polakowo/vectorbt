#%%
"""Named tuples and enumerated types."""

from collections import namedtuple
from enum import IntEnum,Enum
from typing import Union, Type

import json

__all__ = [
    'StopType'
]

__pdoc__ = {}


class StopType(IntEnum):
    """_"""
    StopLoss = 0
    TrailStop = 1
    TakeProfit = 2

def _enum_to_json(enum:Union[Type[Enum],Type[IntEnum]]) -> str:
    """Render an (Int)Enum class to json

    Returns:
        str: A json string
    """
    return json.dumps({enum_option.name:enum_option.value for enum_option in StopType}, indent=2,default=str)


__pdoc__['StopType'] = f"""Stop type.

```plaintext
{_enum_to_json(StopType)}
```
"""
