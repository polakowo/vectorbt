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


class TelegramBot(Configured):
    """Telegram bot."""

    def __init__(self, giphy_kwargs=None, **kwargs):
        from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, PicklePersistence, Defaults
        from vectorbt import settings

        Configured.__init__(
            self,
            giphy_kwargs=giphy_kwargs,
            **kwargs
        )

        # Resolve defaults
        giphy_kwargs = merge_dicts(settings.messaging['giphy'], giphy_kwargs)
        self.giphy_kwargs = giphy_kwargs

        # Create the (persistent) Updater and pass it your bot's token.
        logger.info("Initializing bot")
        updater_kwargs = dict()
        for k in get_func_kwargs(Updater):
            if k in kwargs:
                updater_kwargs[k] = kwargs.pop(k)
        updater_kwargs = merge_dicts(settings.messaging['telegram'], updater_kwargs)
        persistence = updater_kwargs.pop('persistence', None)
        if isinstance(persistence, str):
            persistence = PicklePersistence(persistence)
        defaults = updater_kwargs.pop('defaults', None)
        if isinstance(defaults, dict):
            defaults = Defaults(**defaults)
        self._updater = Updater(persistence=persistence, defaults=defaults, **updater_kwargs)

        # Get the dispatcher to register handlers
        self._dispatcher = self.updater.dispatcher

        # Set up data
        if 'chat_ids' not in self.dispatcher.bot_data:
            self.dispatcher.bot_data['chat_ids'] = []
        else:
            logger.info("Loaded chat ids %s", str(self.dispatcher.bot_data['chat_ids']))

        # Register handlers
        self.dispatcher.add_handler(CommandHandler('start', self.start_callback))
        self.dispatcher.add_handler(CommandHandler("help", self.help_callback))
        for handler in self.custom_handlers:
            self.dispatcher.add_handler(handler)
        self.dispatcher.add_handler(MessageHandler(Filters.status_update.migrate, self.chat_migration_callback))
        self.dispatcher.add_handler(MessageHandler(Filters.command, self.unknown_callback))
        self.dispatcher.add_error_handler(self_decorator(self, self.__class__.error_callback))

    @property
    def updater(self):
        """Updater."""
        return self._updater

    @property
    def dispatcher(self):
        """Dispatcher."""
        return self._dispatcher

    @property
    def custom_handlers(self):
        """Custom handlers to add.

        Override this property to add custom handlers. Order counts."""
        return ()

    @property
    def chat_ids(self):
        """Chat ids that interacted with this bot."""
        return self.dispatcher.bot_data['chat_ids']

    def start(self, in_background=False):
        """Start the bot."""
        logger.info("Running bot %s", str(self.updater.bot.get_me().username))

        # Start the Bot
        self.updater.start_polling()
        if not in_background:
            # Run the bot until you press Ctrl-C or the process receives SIGINT,
            # SIGTERM or SIGABRT. This should be used most of the time, since
            # start_polling() is non-blocking and will stop the bot gracefully.
            self.updater.idle()

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
        """Start message."""
        return "Hello!"

    def start_callback(self, update, context):
        """Start command callback."""
        logger.info(f"{update.effective_message.chat_id} - User: \"/start\"")
        chat_id = update.effective_chat.id
        if chat_id not in self.chat_ids:
            self.chat_ids.append(chat_id)
        self.send_message(chat_id, self.start_message)

    @property
    def help_message(self):
        """Help message."""
        return "Can't help you here, buddy."

    def help_callback(self, update, context):
        """Help command callback."""
        logger.info(f"{update.effective_message.chat_id} - User: \"/help\"")
        self.send_message(update.effective_message.chat_id, self.help_message)

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
