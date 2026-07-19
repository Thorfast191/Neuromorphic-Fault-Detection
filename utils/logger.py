"""
Centralized logging utilities.
"""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(
    name: str,
    log_dir: str = "experiments/logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create or retrieve a configured logger.

    Args:
        name: Logger name.
        log_dir: Directory where log files are stored.
        level: Logging level.

    Returns:
        Configured logger.
    """

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        Path(log_dir) / f"{name}.log"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger