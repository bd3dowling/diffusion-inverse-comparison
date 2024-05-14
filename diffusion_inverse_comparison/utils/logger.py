"""Functions related to logging."""

import logging
from logging import Logger


def get_logger() -> Logger:
    logger = logging.getLogger(name="DIPS")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(name)s] >> %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
