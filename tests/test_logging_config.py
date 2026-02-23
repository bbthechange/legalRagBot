"""Tests for src/logging_config.py"""

import logging
import os
from unittest.mock import patch

from src.logging_config import setup_logging


class TestSetupLogging:
    def test_does_not_crash(self):
        setup_logging()

    def test_respects_log_level_env(self):
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            setup_logging()
            assert logging.getLogger().level == logging.DEBUG

    def test_default_level_is_info(self):
        env = os.environ.copy()
        env.pop("LOG_LEVEL", None)
        with patch.dict(os.environ, env, clear=True):
            setup_logging()
            assert logging.getLogger().level == logging.INFO
