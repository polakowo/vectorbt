# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Modules for messaging."""

__all__ = []
__blacklist__ = []

try:
    import telegram
    del telegram
except ImportError:
    __blacklist__.append('telegram')
else:
    try:
        from telegram import __version_info__
    except ImportError:
        __version_info__ = (0, 0)

    if __version_info__ >= (20, 0):
        __blacklist__.append('telegram')
    else:
        from vectorbt.messaging.telegram import TelegramBot

        __all__.append('TelegramBot')

__pdoc__ = {k: False for k in __all__}
