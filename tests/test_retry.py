"""Tests for src/retry.py"""

import logging
from unittest.mock import patch

import pytest

from src.retry import retry_with_backoff


class TestRetryWithBackoff:
    def test_succeeds_first_try(self):
        @retry_with_backoff(max_retries=3)
        def succeed():
            return "ok"

        assert succeed() == "ok"

    def test_fails_then_succeeds(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient error")
            return "ok"

        assert flaky() == "ok"
        assert call_count == 3

    def test_exhausts_retries(self):
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fail():
            raise RuntimeError("permanent error")

        with pytest.raises(RuntimeError, match="permanent error"):
            always_fail()

    def test_only_catches_specified_exceptions(self):
        @retry_with_backoff(max_retries=3, retryable_exceptions=(ValueError,))
        def raise_type_error():
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            raise_type_error()

    def test_delay_caps_at_max_delay(self):
        @retry_with_backoff(max_retries=5, base_delay=1.0, max_delay=3.0)
        def always_fail():
            raise ValueError("fail")

        with patch("src.retry.time.sleep") as mock_sleep:
            with pytest.raises(ValueError):
                always_fail()

            for call in mock_sleep.call_args_list:
                delay = call[0][0]
                assert delay <= 3.0

    def test_logging(self, caplog):
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def fail_once():
            if not hasattr(fail_once, "_called"):
                fail_once._called = True
                raise ValueError("retry me")
            return "ok"

        with caplog.at_level(logging.WARNING, logger="src.retry"):
            fail_once()

        assert "Retry" in caplog.text
