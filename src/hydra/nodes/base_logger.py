import logging
import logging.config
from logging import Logger


def get_base_logger() -> Logger:
    logging.config.fileConfig("logging.ini", disable_existing_loggers=False)
    logger = logging.getLogger(__name__)

    return logger
