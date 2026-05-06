import logging
import os
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    """
    Simple logger factory. Writes to both console and a log file.
    Using the same name returns the same logger (standard Python behavior).
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already set up, don't add duplicate handlers

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d')}.log")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
