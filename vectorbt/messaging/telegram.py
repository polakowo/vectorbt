# Copyright (c) 2017-2026 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Messaging using `python-telegram-bot`."""

import asyncio
import inspect
import logging
import threading
from functools import partial, wraps

from telegram import __version__ as TG_VER

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0)

from vectorbt import _typing as tp
from vectorbt.utils.config import merge_dicts, get_func_kwargs, Configured
from vectorbt.utils.requests_ import text_to_giphy_url

logger = logging.getLogger(__name__)


if __version_info__ >= (20, 0):
    from telegram import Update
    from telegram.error import ChatMigrated, Forbidden
    from telegram.ext import (
        BaseHandler as Handler,
        CallbackContext,
        ApplicationBuilder,
        CommandHandler,
        ConversationHandler,
        MessageHandler,
        PicklePersistence,
        Defaults,
        filters,
    )

    def _message_type(message: object) -> tp.Optional[str]:
        if getattr(message, "text", None) is not None:
            return "text"
        for attr in (
            "animation",
            "audio",
            "contact",
            "document",
            "game",
            "photo",
            "poll",
            "sticker",
            "video",
            "video_note",
            "voice",
            "location",
            "venue",
            "dice",
        ):
            if getattr(message, attr, None) is not None:
                return attr
        return None

    async def _maybe_await(value: tp.Any) -> tp.Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def _run_callback(callback: tp.Callable, *args, **kwargs) -> tp.Any:
        if inspect.iscoroutinefunction(callback):
            return await callback(*args, **kwargs)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, partial(callback, *args, **kwargs))
        return await _maybe_await(result)

    class LogHandler(Handler):
        """Handler to log user updates."""

        def __init__(self) -> None:
            async def _callback(update: object, context: CallbackContext) -> None:
                return None

            Handler.__init__(self, _callback)

        def check_update(self, update: object) -> tp.Optional[tp.Union[bool, object]]:
            if isinstance(update, Update) and update.effective_message:
                message = update.effective_message
                message_type = _message_type(message)
                if message_type is not None:
                    if message_type == "text":
                        logger.info(f'{message.chat_id} - User: "%s"', message.text)
                    else:
                        logger.info(f"{message.chat_id} - User: %s", message_type)
                return False
            return None

    def send_action(action: str) -> tp.Callable:
        """Sends `action` while processing func command.

        Suitable only for bound callbacks taking arguments `self`, `update`, `context` and optionally other."""

        def decorator(func: tp.Callable) -> tp.Callable:
            @wraps(func)
            async def command_func(self, update: Update, context: CallbackContext, *args, **kwargs) -> tp.Callable:
                if update.effective_chat:
                    await _maybe_await(context.bot.send_chat_action(chat_id=update.effective_chat.id, action=action))
                return await _run_callback(func, self, update, context, *args, **kwargs)

            return command_func

        return decorator

    def self_decorator(self, func: tp.Callable) -> tp.Callable:
        """Pass bot object to func command."""

        async def command_func(update, context, *args, **kwargs):
            return await _run_callback(func, self, update, context, *args, **kwargs)

        return command_func

    class TelegramBot(Configured):
        """Blocking Telegram bot for `python-telegram-bot` 20 and later.

        `**kwargs` are passed to `telegram.ext.ApplicationBuilder` when they match a
        builder method and to `telegram.ext.Updater.start_polling` from `start`.
        """

        def __init__(self, giphy_kwargs: tp.KwargsLike = None, **kwargs) -> None:
            from vectorbt._settings import settings

            telegram_cfg = settings["messaging"]["telegram"]
            giphy_cfg = settings["messaging"]["giphy"]

            Configured.__init__(self, giphy_kwargs=giphy_kwargs, **kwargs)

            # Resolve kwargs
            giphy_kwargs = merge_dicts(giphy_cfg, giphy_kwargs)
            self.giphy_kwargs = giphy_kwargs
            init_kwargs = merge_dicts(telegram_cfg, kwargs)
            builder = ApplicationBuilder()

            token = init_kwargs.pop("token", None)
            if token is not None:
                builder.token(token)
            persistence = init_kwargs.pop("persistence", None)
            if isinstance(persistence, str):
                try:
                    persistence = PicklePersistence(filepath=persistence)
                except TypeError:
                    persistence = PicklePersistence(persistence)
            if persistence is not None:
                builder.persistence(persistence)
            defaults = init_kwargs.pop("defaults", None)
            if isinstance(defaults, dict):
                defaults = Defaults(**defaults)
            if defaults is not None:
                builder.defaults(defaults)

            for k, v in init_kwargs.items():
                builder_func = getattr(builder, k, None)
                if callable(builder_func):
                    builder_func(v)

            # Create the (persistent) Application and pass it your bot's token.
            logger.info("Initializing bot")
            self._application = builder.build()
            self._loop = None
            self._thread = None
            self._stop_future = None
            self._started_event = threading.Event()
            self._stopped_event = threading.Event()
            self._stopped_event.set()
            self._startup_error = None

            # Register handlers
            self.dispatcher.add_handler(self.log_handler)
            self.dispatcher.add_handler(CommandHandler("start", self._to_async_callback(self.start_callback)))
            self.dispatcher.add_handler(CommandHandler("help", self._to_async_callback(self.help_callback)))
            for handler in self.custom_handlers:
                self.dispatcher.add_handler(self._wrap_handler(handler))
            self.dispatcher.add_handler(MessageHandler(filters.StatusUpdate.MIGRATE, self._to_async_callback(
                self.chat_migration_callback
            )))
            self.dispatcher.add_handler(MessageHandler(filters.COMMAND, self._to_async_callback(self.unknown_callback)))
            self.dispatcher.add_error_handler(self._to_async_callback(self.error_callback))

            # Set up data
            if "chat_ids" not in self.dispatcher.bot_data:
                self.dispatcher.bot_data["chat_ids"] = []
            else:
                logger.info("Loaded chat ids %s", str(self.dispatcher.bot_data["chat_ids"]))

        @property
        def application(self):
            """Application."""
            return self._application

        @property
        def updater(self):
            """Updater."""
            return self.application.updater

        @property
        def dispatcher(self):
            """Dispatcher-like application."""
            return self.application

        @property
        def log_handler(self) -> LogHandler:
            """Log handler."""
            return LogHandler()

        @property
        def custom_handlers(self) -> tp.Iterable[Handler]:
            """Custom handlers to add.

            Override to add custom handlers. Order counts."""
            return ()

        @property
        def chat_ids(self) -> tp.List[int]:
            """Chat ids that ever interacted with this bot.

            A chat id is added upon receiving the "/start" command."""
            return self.dispatcher.bot_data.setdefault("chat_ids", [])

        def _to_async_callback(self, callback: tp.Callable) -> tp.Callable:
            async def callback_func(update, context, *args, **kwargs):
                return await _run_callback(callback, update, context, *args, **kwargs)

            return callback_func

        def _wrap_handler(self, handler: Handler) -> Handler:
            if hasattr(handler, "callback"):
                handler.callback = self._to_async_callback(handler.callback)
            if isinstance(handler, ConversationHandler):
                for entry_point in handler.entry_points:
                    self._wrap_handler(entry_point)
                for state_handlers in handler.states.values():
                    for state_handler in state_handlers:
                        self._wrap_handler(state_handler)
                for fallback in handler.fallbacks:
                    self._wrap_handler(fallback)
            return handler

        def _resolve_polling_kwargs(self, kwargs: dict) -> dict:
            from vectorbt._settings import settings

            telegram_cfg = settings["messaging"]["telegram"]
            ignored_keys = {"token", "use_context", "persistence", "defaults"}
            try:
                signature = inspect.signature(self.updater.start_polling)
                accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
                accepted_keys = set(get_func_kwargs(self.updater.start_polling))
            except (TypeError, ValueError):
                accepts_kwargs = True
                accepted_keys = set()

            polling_kwargs = dict()
            for k, v in telegram_cfg.items():
                if k in ignored_keys:
                    continue
                if accepts_kwargs or k in accepted_keys:
                    polling_kwargs[k] = v
            for k, v in kwargs.items():
                if accepts_kwargs or k in accepted_keys:
                    polling_kwargs[k] = v
            return polling_kwargs

        async def _polling_main(self, polling_kwargs: dict) -> None:
            initialized = False
            application_stopped = False
            updater_started = False
            if self._stop_future is None or self._stop_future.done():
                self._stop_future = asyncio.get_running_loop().create_future()
            try:
                await _maybe_await(self.application.initialize())
                initialized = True
                if self.application.post_init:
                    await _maybe_await(self.application.post_init(self.application))
                logger.info("Running bot %s", str((await _maybe_await(self.application.bot.get_me())).username))
                await _maybe_await(self.application.start())
                if self.updater is not None:
                    await _maybe_await(self.updater.start_polling(**polling_kwargs))
                    updater_started = True
                await _run_callback(self.started_callback)
                self._started_event.set()
                await self._stop_future
            except BaseException as e:
                self._startup_error = e
                self._started_event.set()
                raise
            finally:
                try:
                    if updater_started and self.updater is not None:
                        await _maybe_await(self.updater.stop())
                    if getattr(self.application, "running", False):
                        await _maybe_await(self.application.stop())
                        application_stopped = True
                    if application_stopped and self.application.post_stop:
                        await _maybe_await(self.application.post_stop(self.application))
                    if initialized:
                        await _maybe_await(self.application.shutdown())
                        if self.application.post_shutdown:
                            await _maybe_await(self.application.post_shutdown(self.application))
                finally:
                    self._stopped_event.set()

        def _run_loop(self, polling_kwargs: dict) -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            task = None
            asyncio.set_event_loop(loop)
            try:
                self._stop_future = loop.create_future()
                task = loop.create_task(self._polling_main(polling_kwargs))
                try:
                    loop.run_until_complete(task)
                except (KeyboardInterrupt, SystemExit):
                    self._request_stop()
                    if not task.done():
                        loop.run_until_complete(task)
            finally:
                try:
                    if task is not None and not task.done():
                        self._request_stop()
                        loop.run_until_complete(task)
                    loop.close()
                finally:
                    self._loop = None

        def _run_async(self, value: tp.Any) -> tp.Any:
            if not inspect.isawaitable(value):
                return value
            if self._loop is not None and self._loop.is_running():
                try:
                    running_loop = asyncio.get_running_loop()
                except RuntimeError:
                    running_loop = None
                if running_loop is self._loop:
                    raise RuntimeError("Cannot call blocking TelegramBot method from the bot event loop")
                return asyncio.run_coroutine_threadsafe(value, self._loop).result()
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(value)
            finally:
                loop.close()

        def start(self, in_background: bool = False, **kwargs) -> None:
            """Start the bot.

            `**kwargs` are passed to `telegram.ext.Updater.start_polling`
            and override settings under `messaging.telegram` in `vectorbt._settings.settings`."""
            if self._thread is not None and self._thread.is_alive():
                return

            polling_kwargs = self._resolve_polling_kwargs(kwargs)
            self._started_event.clear()
            self._stopped_event.clear()
            self._startup_error = None

            if in_background:
                self._thread = threading.Thread(target=self._run_loop, args=(polling_kwargs,), daemon=True)
                self._thread.start()
                self._started_event.wait()
                if self._startup_error is not None:
                    raise self._startup_error
                return

            try:
                self._run_loop(polling_kwargs)
            except KeyboardInterrupt:
                self.stop()

        def started_callback(self) -> None:
            """Callback once the bot has been started.

            Override to execute custom commands upon starting the bot."""
            self.send_message_to_all("I'm back online!")

        def send(self, kind: str, chat_id: int, *args, log_msg: tp.Optional[str] = None, **kwargs) -> None:
            """Send message of any kind to `chat_id`."""
            try:
                self._run_async(getattr(self.application.bot, "send_" + kind)(chat_id, *args, **kwargs))
                if log_msg is None:
                    log_msg = kind
                logger.info(f"{chat_id} - Bot: %s", log_msg)
            except ChatMigrated as e:
                # transfer data, if old data is still present
                new_id = e.new_chat_id
                if chat_id in self.chat_ids:
                    self.chat_ids.remove(chat_id)
                self.chat_ids.append(new_id)
                # Resend to new chat id
                self.send(kind, new_id, *args, log_msg=log_msg, **kwargs)
            except Forbidden:
                logger.info(f"{chat_id} - Unauthorized to send the %s", kind)

        def send_to_all(self, kind: str, *args, **kwargs) -> None:
            """Send message of any kind to all in `TelegramBot.chat_ids`."""
            for chat_id in self.chat_ids:
                self.send(kind, chat_id, *args, **kwargs)

        def send_message(self, chat_id: int, text: str, *args, **kwargs) -> None:
            """Send text message to `chat_id`."""
            log_msg = '"%s"' % text
            self.send("message", chat_id, text, *args, log_msg=log_msg, **kwargs)

        def send_message_to_all(self, text: str, *args, **kwargs) -> None:
            """Send text message to all in `TelegramBot.chat_ids`."""
            log_msg = '"%s"' % text
            self.send_to_all("message", text, *args, log_msg=log_msg, **kwargs)

        def send_giphy(self, chat_id: int, text: str, *args, giphy_kwargs: tp.KwargsLike = None, **kwargs) -> None:
            """Send GIPHY from text to `chat_id`."""
            if giphy_kwargs is None:
                giphy_kwargs = self.giphy_kwargs
            gif_url = text_to_giphy_url(text, **giphy_kwargs)
            log_msg = '"%s" as GIPHY %s' % (text, gif_url)
            self.send("animation", chat_id, gif_url, *args, log_msg=log_msg, **kwargs)

        def send_giphy_to_all(self, text: str, *args, giphy_kwargs: tp.KwargsLike = None, **kwargs) -> None:
            """Send GIPHY from text to all in `TelegramBot.chat_ids`."""
            if giphy_kwargs is None:
                giphy_kwargs = self.giphy_kwargs
            gif_url = text_to_giphy_url(text, **giphy_kwargs)
            log_msg = '"%s" as GIPHY %s' % (text, gif_url)
            self.send_to_all("animation", gif_url, *args, log_msg=log_msg, **kwargs)

        @property
        def start_message(self) -> str:
            """Message to be sent upon "/start" command.

            Override to define your own message."""
            return "Hello!"

        def start_callback(self, update: object, context: CallbackContext) -> None:
            """Start command callback."""
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                if chat_id not in self.chat_ids:
                    self.chat_ids.append(chat_id)
                self.send_message(chat_id, self.start_message)

        @property
        def help_message(self) -> str:
            """Message to be sent upon "/help" command.

            Override to define your own message."""
            return "Can't help you here, buddy."

        def help_callback(self, update: object, context: CallbackContext) -> None:
            """Help command callback."""
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                self.send_message(chat_id, self.help_message)

        def chat_migration_callback(self, update: object, context: CallbackContext) -> None:
            """Chat migration callback."""
            if isinstance(update, Update) and update.message:
                old_id = update.message.migrate_from_chat_id or update.message.chat_id
                new_id = update.message.migrate_to_chat_id or update.message.chat_id
                if old_id in self.chat_ids:
                    self.chat_ids.remove(old_id)
                self.chat_ids.append(new_id)
                logger.info(f"{old_id} - Chat migrated to {new_id}")

        def unknown_callback(self, update: object, context: CallbackContext) -> None:
            """Unknown command callback."""
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                logger.info(f'{chat_id} - Unknown command "{update.message}"')
                self.send_message(chat_id, "Sorry, I didn't understand that command.")

        def error_callback(self, update: object, context: CallbackContext, *args) -> None:
            """Error callback."""
            logger.error('Exception while handling an update "%s": ', update, exc_info=context.error)
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                self.send_message(chat_id, "Sorry, an error happened.")

        def stop(self) -> None:
            """Stop the bot."""
            logger.info("Stopping bot")
            self._request_stop()
            if self._thread is not None and self._thread.is_alive() and self._thread is not threading.current_thread():
                self._thread.join()

        def _request_stop(self) -> None:
            if self._stop_future is None:
                return

            def _stop() -> None:
                if not self._stop_future.done():
                    self._stop_future.set_result(None)

            if self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(_stop)
            else:
                _stop()

        @property
        def running(self) -> bool:
            """Whether the bot is running."""
            return bool(getattr(self.application, "running", False))

else:
    from telegram import Update
    from telegram.error import Unauthorized, ChatMigrated
    from telegram.ext import (
        Handler,
        CallbackContext,
        Updater,
        Dispatcher,
        CommandHandler,
        MessageHandler,
        Filters,
        PicklePersistence,
        Defaults,
    )
    from telegram.utils.helpers import effective_message_type

    class LogHandler(Handler):
        """Handler to log user updates."""

        def check_update(self, update: object) -> tp.Optional[tp.Union[bool, object]]:
            if isinstance(update, Update) and update.effective_message:
                message = update.effective_message
                message_type = effective_message_type(message)
                if message_type is not None:
                    if message_type == "text":
                        logger.info(f'{message.chat_id} - User: "%s"', message.text)
                    else:
                        logger.info(f"{message.chat_id} - User: %s", message_type)
                return False
            return None

    def send_action(action: str) -> tp.Callable:
        """Sends `action` while processing func command.

        Suitable only for bound callbacks taking arguments `self`, `update`, `context` and optionally other."""

        def decorator(func: tp.Callable) -> tp.Callable:
            @wraps(func)
            def command_func(self, update: Update, context: CallbackContext, *args, **kwargs) -> tp.Callable:
                if update.effective_chat:
                    context.bot.send_chat_action(chat_id=update.effective_chat.id, action=action)
                return func(self, update, context, *args, **kwargs)

            return command_func

        return decorator

    def self_decorator(self, func: tp.Callable) -> tp.Callable:
        """Pass bot object to func command."""

        def command_func(update, context, *args, **kwargs):
            return func(self, update, context, *args, **kwargs)

        return command_func

    class TelegramBot(Configured):
        """Telegram bot.

        See [Extensions - Your first Bot](https://github.com/python-telegram-bot/python-telegram-bot/wiki/Extensions-%E2%80%93-Your-first-Bot).

        `**kwargs` are passed to `telegram.ext.updater.Updater` and override
        settings under `messaging.telegram` in `vectorbt._settings.settings`.
        """

        def __init__(self, giphy_kwargs: tp.KwargsLike = None, **kwargs) -> None:
            from vectorbt._settings import settings

            telegram_cfg = settings["messaging"]["telegram"]
            giphy_cfg = settings["messaging"]["giphy"]

            Configured.__init__(self, giphy_kwargs=giphy_kwargs, **kwargs)

            # Resolve kwargs
            giphy_kwargs = merge_dicts(giphy_cfg, giphy_kwargs)
            self.giphy_kwargs = giphy_kwargs
            default_kwargs = dict()
            passed_kwargs = dict()
            for k in get_func_kwargs(Updater.__init__):
                if k in telegram_cfg:
                    default_kwargs[k] = telegram_cfg[k]
                if k in kwargs:
                    passed_kwargs[k] = kwargs.pop(k)
            updater_kwargs = merge_dicts(default_kwargs, passed_kwargs)
            persistence = updater_kwargs.pop("persistence", None)
            if isinstance(persistence, str):
                persistence = PicklePersistence(persistence)
            defaults = updater_kwargs.pop("defaults", None)
            if isinstance(defaults, dict):
                defaults = Defaults(**defaults)

            # Create the (persistent) Updater and pass it your bot's token.
            logger.info("Initializing bot")
            self._updater = Updater(persistence=persistence, defaults=defaults, **updater_kwargs)

            # Get the dispatcher to register handlers
            self._dispatcher = self.updater.dispatcher

            # Register handlers
            self.dispatcher.add_handler(self.log_handler)
            self.dispatcher.add_handler(CommandHandler("start", self.start_callback))
            self.dispatcher.add_handler(CommandHandler("help", self.help_callback))
            for handler in self.custom_handlers:
                self.dispatcher.add_handler(handler)
            self.dispatcher.add_handler(MessageHandler(Filters.status_update.migrate, self.chat_migration_callback))
            self.dispatcher.add_handler(MessageHandler(Filters.command, self.unknown_callback))
            self.dispatcher.add_error_handler(self_decorator(self, self.__class__.error_callback))

            # Set up data
            if "chat_ids" not in self.dispatcher.bot_data:
                self.dispatcher.bot_data["chat_ids"] = []
            else:
                logger.info("Loaded chat ids %s", str(self.dispatcher.bot_data["chat_ids"]))

        @property
        def updater(self) -> Updater:
            """Updater."""
            return self._updater

        @property
        def dispatcher(self) -> Dispatcher:
            """Dispatcher."""
            return self._dispatcher

        @property
        def log_handler(self) -> LogHandler:
            """Log handler."""
            return LogHandler(lambda update, context: None)

        @property
        def custom_handlers(self) -> tp.Iterable[Handler]:
            """Custom handlers to add.

            Override to add custom handlers. Order counts."""
            return ()

        @property
        def chat_ids(self) -> tp.List[int]:
            """Chat ids that ever interacted with this bot.

            A chat id is added upon receiving the "/start" command."""
            return self.dispatcher.bot_data["chat_ids"]

        def start(self, in_background: bool = False, **kwargs) -> None:
            """Start the bot.

            `**kwargs` are passed to `telegram.ext.updater.Updater.start_polling`
            and override settings under `messaging.telegram` in `vectorbt._settings.settings`."""
            from vectorbt._settings import settings

            telegram_cfg = settings["messaging"]["telegram"]

            # Resolve kwargs
            default_kwargs = dict()
            passed_kwargs = dict()
            for k in get_func_kwargs(self.updater.start_polling):
                if k in telegram_cfg:
                    default_kwargs[k] = telegram_cfg[k]
                if k in kwargs:
                    passed_kwargs[k] = kwargs.pop(k)
            polling_kwargs = merge_dicts(default_kwargs, passed_kwargs)

            # Start the Bot
            logger.info("Running bot %s", str(self.updater.bot.get_me().username))
            self.updater.start_polling(**polling_kwargs)
            self.started_callback()

            if not in_background:
                # Run the bot until you press Ctrl-C or the process receives SIGINT,
                # SIGTERM or SIGABRT. This should be used most of the time, since
                # start_polling() is non-blocking and will stop the bot gracefully.
                self.updater.idle()

        def started_callback(self) -> None:
            """Callback once the bot has been started.

            Override to execute custom commands upon starting the bot."""
            self.send_message_to_all("I'm back online!")

        def send(self, kind: str, chat_id: int, *args, log_msg: tp.Optional[str] = None, **kwargs) -> None:
            """Send message of any kind to `chat_id`."""
            try:
                getattr(self.updater.bot, "send_" + kind)(chat_id, *args, **kwargs)
                if log_msg is None:
                    log_msg = kind
                logger.info(f"{chat_id} - Bot: %s", log_msg)
            except ChatMigrated as e:
                # transfer data, if old data is still present
                new_id = e.new_chat_id
                if chat_id in self.chat_ids:
                    self.chat_ids.remove(chat_id)
                self.chat_ids.append(new_id)
                # Resend to new chat id
                self.send(kind, new_id, *args, log_msg=log_msg, **kwargs)
            except Unauthorized:
                logger.info(f"{chat_id} - Unauthorized to send the %s", kind)

        def send_to_all(self, kind: str, *args, **kwargs) -> None:
            """Send message of any kind to all in `TelegramBot.chat_ids`."""
            for chat_id in self.chat_ids:
                self.send(kind, chat_id, *args, **kwargs)

        def send_message(self, chat_id: int, text: str, *args, **kwargs) -> None:
            """Send text message to `chat_id`."""
            log_msg = '"%s"' % text
            self.send("message", chat_id, text, *args, log_msg=log_msg, **kwargs)

        def send_message_to_all(self, text: str, *args, **kwargs) -> None:
            """Send text message to all in `TelegramBot.chat_ids`."""
            log_msg = '"%s"' % text
            self.send_to_all("message", text, *args, log_msg=log_msg, **kwargs)

        def send_giphy(self, chat_id: int, text: str, *args, giphy_kwargs: tp.KwargsLike = None, **kwargs) -> None:
            """Send GIPHY from text to `chat_id`."""
            if giphy_kwargs is None:
                giphy_kwargs = self.giphy_kwargs
            gif_url = text_to_giphy_url(text, **giphy_kwargs)
            log_msg = '"%s" as GIPHY %s' % (text, gif_url)
            self.send("animation", chat_id, gif_url, *args, log_msg=log_msg, **kwargs)

        def send_giphy_to_all(self, text: str, *args, giphy_kwargs: tp.KwargsLike = None, **kwargs) -> None:
            """Send GIPHY from text to all in `TelegramBot.chat_ids`."""
            if giphy_kwargs is None:
                giphy_kwargs = self.giphy_kwargs
            gif_url = text_to_giphy_url(text, **giphy_kwargs)
            log_msg = '"%s" as GIPHY %s' % (text, gif_url)
            self.send_to_all("animation", gif_url, *args, log_msg=log_msg, **kwargs)

        @property
        def start_message(self) -> str:
            """Message to be sent upon "/start" command.

            Override to define your own message."""
            return "Hello!"

        def start_callback(self, update: object, context: CallbackContext) -> None:
            """Start command callback."""
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                if chat_id not in self.chat_ids:
                    self.chat_ids.append(chat_id)
                self.send_message(chat_id, self.start_message)

        @property
        def help_message(self) -> str:
            """Message to be sent upon "/help" command.

            Override to define your own message."""
            return "Can't help you here, buddy."

        def help_callback(self, update: object, context: CallbackContext) -> None:
            """Help command callback."""
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                self.send_message(chat_id, self.help_message)

        def chat_migration_callback(self, update: object, context: CallbackContext) -> None:
            """Chat migration callback."""
            if isinstance(update, Update) and update.message:
                old_id = update.message.migrate_from_chat_id or update.message.chat_id
                new_id = update.message.migrate_to_chat_id or update.message.chat_id
                if old_id in self.chat_ids:
                    self.chat_ids.remove(old_id)
                self.chat_ids.append(new_id)
                logger.info(f"{old_id} - Chat migrated to {new_id}")

        def unknown_callback(self, update: object, context: CallbackContext) -> None:
            """Unknown command callback."""
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                logger.info(f'{chat_id} - Unknown command "{update.message}"')
                self.send_message(chat_id, "Sorry, I didn't understand that command.")

        def error_callback(self, update: object, context: CallbackContext, *args) -> None:
            """Error callback."""
            logger.error('Exception while handling an update "%s": ', update, exc_info=context.error)
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                self.send_message(chat_id, "Sorry, an error happened.")

        def stop(self) -> None:
            """Stop the bot."""
            logger.info("Stopping bot")
            self.updater.stop()

        @property
        def running(self) -> bool:
            """Whether the bot is running."""
            return self.updater.running
