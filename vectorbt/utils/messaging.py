"""Utilities for messaging."""

import logging
from functools import wraps
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from urllib.parse import urlencode

from vectorbt.utils.config import merge_dicts, get_func_kwargs, Configured

logger = logging.getLogger(__name__)


def send_action(action):
    """Sends `action` while processing func command.

    Suitable only for bound callbacks taking arguments `self`, `update`, `context` and optionally other."""

    def decorator(func):
        @wraps(func)
        def command_func(self, update, context, *args, **kwargs):
            context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=action)
            return func(self, update, context, *args, **kwargs)

        return command_func

    return decorator


def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None):
    """Retry `retries` times if unsuccessful."""
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def text_to_giphy_url(text, api_key=None, weirdness=None):
    """Translate text to GIF.

    See https://engineering.giphy.com/contextually-aware-search-giphy-gets-work-specific/."""
    from vectorbt import settings

    if api_key is None:
        api_key = settings.messaging['giphy']['api_key']
    if weirdness is None:
        weirdness = settings.messaging['giphy']['weirdness']

    params = {
        'api_key': api_key,
        's': text,
        'weirdness': weirdness
    }
    url = "http://api.giphy.com/v1/gifs/translate?" + urlencode(params)
    response = requests_retry_session().get(url)
    return response.json()['data']['images']['fixed_height']['url']


def self_decorator(self, func):
    """Pass bot object to func command."""

    def command_func(update, context, *args, **kwargs):
        return func(self, update, context, *args, **kwargs)

    return command_func


try:
    from telegram.ext import Handler
    from telegram.utils.helpers import effective_message_type


    class LogHandler(Handler):
        """Handler to log user updates."""
        def check_update(self, update):
            message = update.effective_message
            message_type = effective_message_type(message)
            if message_type is not None:
                if message_type == 'text':
                    logger.info(f"{message.chat_id} - User: \"%s\"", message.text)
                else:
                    logger.info(f"{message.chat_id} - User: %s", message_type)
            return False
except ImportError:
    LogHandler = None


class TelegramBot(Configured):
    """Telegram bot.

    See [Extensions – Your first Bot](https://github.com/python-telegram-bot/python-telegram-bot/wiki/Extensions-%E2%80%93-Your-first-Bot).

    `**kwargs` are passed to `telegram.ext.updater.Updater` and override settings
    under `telegram` in `vectorbt.settings.messaging`.

    ## Example

    Let's extend `TelegramBot` to track cryptocurrency prices:

    ```python-repl
    >>> from telegram.ext import CommandHandler
    >>> import ccxt
    >>> import logging
    >>> import vectorbt as vbt

    >>> logging.basicConfig(level=logging.INFO)  # enable logging

    >>> class MyTelegramBot(vbt.TelegramBot):
    ...     @property
    ...     def custom_handlers(self):
    ...         return (CommandHandler('get', self.get),)
    ...
    ...     @property
    ...     def help_message(self):
    ...         return "Type /get [symbol] [exchange id (optional)] to get the latest price."
    ...
    ...     def get(self, update, context):
    ...         chat_id = update.effective_message.chat_id
    ...
    ...         if len(context.args) == 1:
    ...             symbol = context.args[0]
    ...             exchange = 'binance'
    ...         elif len(context.args) == 2:
    ...             symbol = context.args[0]
    ...             exchange = context.args[1]
    ...         else:
    ...             self.send_message(chat_id, "This command requires symbol and optionally exchange id.")
    ...             return
    ...         try:
    ...             ticker = getattr(ccxt, exchange)().fetchTicker(symbol)
    ...         except Exception as e:
    ...             self.send_message(chat_id, str(e))
    ...             return
    ...         self.send_message(chat_id, str(ticker['last']))

    >>> bot = MyTelegramBot(token='YOUR_TOKEN')
    >>> bot.start()
    INFO:vectorbt.utils.messaging:Initializing bot
    INFO:vectorbt.utils.messaging:Loaded chat ids [447924619]
    INFO:vectorbt.utils.messaging:Running bot vectorbt_bot
    INFO:apscheduler.scheduler:Scheduler started
    INFO:vectorbt.utils.messaging:447924619 - Bot: "I'm back online!"
    INFO:vectorbt.utils.messaging:447924619 - User: "/start"
    INFO:vectorbt.utils.messaging:447924619 - Bot: "Hello!"
    INFO:vectorbt.utils.messaging:447924619 - User: "/help"
    INFO:vectorbt.utils.messaging:447924619 - Bot: "Type /get [symbol] [exchange id (optional)] to get the latest price."
    INFO:vectorbt.utils.messaging:447924619 - User: "/get BTC/USDT"
    INFO:vectorbt.utils.messaging:447924619 - Bot: "55530.55"
    INFO:vectorbt.utils.messaging:447924619 - User: "/get BTC/USD bitmex"
    INFO:vectorbt.utils.messaging:447924619 - Bot: "55509.0"
    INFO:telegram.ext.updater:Received signal 2 (SIGINT), stopping...
    INFO:apscheduler.scheduler:Scheduler has been shut down
    ```"""

    def __init__(self, giphy_kwargs=None, **kwargs):
        from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, PicklePersistence, Defaults
        from vectorbt import settings

        Configured.__init__(
            self,
            giphy_kwargs=giphy_kwargs,
            **kwargs
        )

        # Resolve kwargs
        giphy_kwargs = merge_dicts(settings.messaging['giphy'], giphy_kwargs)
        self.giphy_kwargs = giphy_kwargs
        default_kwargs = dict()
        passed_kwargs = dict()
        for k in get_func_kwargs(Updater):
            if k in settings.messaging['telegram']:
                default_kwargs[k] = settings.messaging['telegram'][k]
            if k in kwargs:
                passed_kwargs[k] = kwargs.pop(k)
        updater_kwargs = merge_dicts(default_kwargs, passed_kwargs)
        persistence = updater_kwargs.pop('persistence', None)
        if isinstance(persistence, str):
            persistence = PicklePersistence(persistence)
        defaults = updater_kwargs.pop('defaults', None)
        if isinstance(defaults, dict):
            defaults = Defaults(**defaults)

        # Create the (persistent) Updater and pass it your bot's token.
        logger.info("Initializing bot")
        self._updater = Updater(persistence=persistence, defaults=defaults, **updater_kwargs)

        # Get the dispatcher to register handlers
        self._dispatcher = self.updater.dispatcher

        # Register handlers
        self.dispatcher.add_handler(self.log_handler)
        self.dispatcher.add_handler(CommandHandler('start', self.start_callback))
        self.dispatcher.add_handler(CommandHandler("help", self.help_callback))
        for handler in self.custom_handlers:
            self.dispatcher.add_handler(handler)
        self.dispatcher.add_handler(MessageHandler(Filters.status_update.migrate, self.chat_migration_callback))
        self.dispatcher.add_handler(MessageHandler(Filters.command, self.unknown_callback))
        self.dispatcher.add_error_handler(self_decorator(self, self.__class__.error_callback))

        # Set up data
        if 'chat_ids' not in self.dispatcher.bot_data:
            self.dispatcher.bot_data['chat_ids'] = []
        else:
            logger.info("Loaded chat ids %s", str(self.dispatcher.bot_data['chat_ids']))

    @property
    def updater(self):
        """Updater."""
        return self._updater

    @property
    def dispatcher(self):
        """Dispatcher."""
        return self._dispatcher

    @property
    def log_handler(self):
        """Log handler."""
        return LogHandler(lambda update, context: None)

    @property
    def custom_handlers(self):
        """Custom handlers to add.

        Override to add custom handlers. Order counts."""
        return ()

    @property
    def chat_ids(self):
        """Chat ids that ever interacted with this bot.

        A chat id is added upon receiving the "/start" command."""
        return self.dispatcher.bot_data['chat_ids']

    def start(self, in_background=False, **kwargs):
        """Start the bot.

        `**kwargs` are passed to `telegram.ext.updater.Updater.start_polling`
        and override settings under `telegram` in `vectorbt.settings.messaging`."""
        from vectorbt import settings

        # Resolve kwargs
        default_kwargs = dict()
        passed_kwargs = dict()
        for k in get_func_kwargs(self.updater.start_polling):
            if k in settings.messaging['telegram']:
                default_kwargs[k] = settings.messaging['telegram'][k]
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

    def started_callback(self):
        """Callback once the bot has been started.

        Override to execute custom commands upon starting the bot."""
        self.send_message_to_all("I'm back online!")

    def send(self, kind, chat_id, *args, log_msg=None, **kwargs):
        """Send message of any kind to `chat_id`."""
        from telegram.error import Unauthorized, ChatMigrated

        try:
            getattr(self.updater.bot, 'send_' + kind)(chat_id, *args, **kwargs)
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
        except Unauthorized as e:
            logger.info(f"{chat_id} - Unauthorized to send the %s", kind)

    def send_to_all(self, kind, *args, **kwargs):
        """Send message of any kind to all in `TelegramBot.chat_ids`."""
        for chat_id in self.chat_ids:
            self.send(kind, chat_id, *args, **kwargs)

    def send_message(self, chat_id, text, *args, **kwargs):
        """Send text message to `chat_id`."""
        log_msg = "\"%s\"" % text
        self.send('message', chat_id, text, *args, log_msg=log_msg, **kwargs)

    def send_message_to_all(self, text, *args, **kwargs):
        """Send text message to all in `TelegramBot.chat_ids`."""
        log_msg = "\"%s\"" % text
        self.send_to_all('message', text, *args, log_msg=log_msg, **kwargs)

    def send_giphy(self, chat_id, text, *args, giphy_kwargs=None, **kwargs):
        """Send GIPHY from text to `chat_id`."""
        if giphy_kwargs is None:
            giphy_kwargs = self.giphy_kwargs
        gif_url = text_to_giphy_url(text, **giphy_kwargs)
        log_msg = "\"%s\" as GIPHY %s" % (text, gif_url)
        self.send('animation', chat_id, gif_url, *args, log_msg=log_msg, **kwargs)

    def send_giphy_to_all(self, text, *args, giphy_kwargs=None, **kwargs):
        """Send GIPHY from text to all in `TelegramBot.chat_ids`."""
        if giphy_kwargs is None:
            giphy_kwargs = self.giphy_kwargs
        gif_url = text_to_giphy_url(text, **giphy_kwargs)
        log_msg = "\"%s\" as GIPHY %s" % (text, gif_url)
        self.send_to_all('animation', gif_url, *args, log_msg=log_msg, **kwargs)

    @property
    def start_message(self):
        """Message to be sent upon "/start" command.

        Override to define your own message."""
        return "Hello!"

    def start_callback(self, update, context):
        """Start command callback."""
        chat_id = update.effective_chat.id
        if chat_id not in self.chat_ids:
            self.chat_ids.append(chat_id)
        self.send_message(chat_id, self.start_message)

    @property
    def help_message(self):
        """Message to be sent upon "/help" command.

        Override to define your own message."""
        return "Can't help you here, buddy."

    def help_callback(self, update, context):
        """Help command callback."""
        chat_id = update.effective_chat.id
        self.send_message(chat_id, self.help_message)

    def chat_migration_callback(self, update, context):
        """Chat migration callback."""
        old_id = update.message.migrate_from_chat_id or update.message.chat_id
        new_id = update.message.migrate_to_chat_id or update.message.chat_id
        if old_id in self.chat_ids:
            self.chat_ids.remove(old_id)
        self.chat_ids.append(new_id)
        logger.info(f"{old_id} - Chat migrated to {new_id}")

    def unknown_callback(self, update, context):
        """Unknown command callback."""
        logger.info(f"{update.effective_message.chat_id} - Unknown command \"{update.message.text}\"")
        self.send_message(update.effective_chat.id, "Sorry, I didn't understand that command.")

    def error_callback(self, update, context, *args):
        """Error callback."""
        logger.error("Exception while handling an update \"%s\": ", update, exc_info=context.error)
        if update.effective_message:
            self.send_message(update.effective_chat.id, "Sorry, an error happened.")

    def stop(self):
        """Stop the bot."""
        logger.info("Stopping bot")
        self.updater.stop()

    @property
    def running(self):
        """Whether the bot is running."""
        return self.updater.running
