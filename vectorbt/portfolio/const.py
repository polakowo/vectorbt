"""Constants."""

class PositionType:
    OPEN = 0
    """Open position."""
    CLOSED = 1
    """Closed position."""


class OutputFormat:
    PERCENT = '%'
    """Output is a ratio that can be converted to percentage."""
    CURRENCY = '$'
    """Output is in currency units such as USD."""
    TIME = 'time'
    """Output is in time units such as days."""
    NOMINAL = 'nominal'
    """Output consists of nominal data."""
    NONE = ''
    """Output doesn't need any formatting."""
