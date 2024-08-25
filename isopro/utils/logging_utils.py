"""Logging utilities for the isopro package."""
import logging
import sys

def setup_logger(name, level=logging.INFO):
    """
    Set up a logger with the given name and level.

    Args:
        name (str): The name of the logger.
        level (int): The logging level (default: logging.INFO).

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger