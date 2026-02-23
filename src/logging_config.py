"""
Structured Logging Configuration

Provides a single setup_logging() function to configure logging
consistently across the application.
"""

import logging
import os


def setup_logging():
    """
    Configure logging using LOG_LEVEL env var (default: INFO).

    Format: %(asctime)s [%(levelname)s] %(name)s: %(message)s
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
