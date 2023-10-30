import sys
import logging

from logging import Logger
from google.cloud import logging as gcp_logging


_LOGGING_INITIALIZED = False


def setup_logging():
    global _LOGGING_INITIALIZED

    if not _LOGGING_INITIALIZED:
        logging_client = gcp_logging.Client()
        logging_client.setup_logging()

        # Add a StreamHandler to also log to stdout
        root_logger = logging.getLogger()
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
        stdout_handler.setFormatter(formatter)
        root_logger.addHandler(stdout_handler)

        _LOGGING_INITIALIZED = True


def get_logger(name: str) -> Logger:
    logger = logging.getLogger(name)
    return logger


# Set up Google Cloud Logging on import
setup_logging()