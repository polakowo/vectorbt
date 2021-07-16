"""Named tuples and enumerated types.

Defines enums and other schemas for `vectorbt.signals`."""

from vectorbt import _typing as tp
from vectorbt.utils.docs import to_doc

__all__ = [
    'StopType',
    'FactoryMode'
]

__pdoc__ = {}


class StopTypeT(tp.NamedTuple):
    StopLoss: int = 0
    TrailStop: int = 1
    TakeProfit: int = 2


StopType = StopTypeT()
"""_"""

__pdoc__['StopType'] = f"""Stop type.

```json
{to_doc(StopType)}
```
"""


class FactoryModeT(tp.NamedTuple):
    Entries: int = 0
    Exits: int = 1
    Both: int = 2
    Chain: int = 3


FactoryMode = FactoryModeT()
"""_"""

__pdoc__['FactoryMode'] = f"""Factory mode.

```json
{to_doc(FactoryMode)}
```

Attributes:
    Entries: Generate entries only using `generate_func`.
    
        Takes no input signal arrays.
        Produces one output signal array - `entries`.
        
        Such generators often have no suffix.
    Exits: Generate exits only using `generate_ex_func`.
        
        Takes one input signal array - `entries`.
        Produces one output signal array - `exits`.
        
        Such generators often have suffix 'X'.
    Both: Generate both entries and exits using `generate_enex_func`.
            
        Takes no input signal arrays.
        Produces two output signal arrays - `entries` and `exits`.
        
        Such generators often have suffix 'NX'.
    Chain: Generate chain of entries and exits using `generate_enex_func`.
                
        Takes one input signal array - `entries`.
        Produces two output signal arrays - `new_entries` and `exits`.
        
        Such generators often have suffix 'CX'.
"""
