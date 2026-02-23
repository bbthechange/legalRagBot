"""
Retry Decorator — Exponential Backoff for Transient Failures

Configurable retry logic with exponential backoff, max delay cap,
and selective exception handling.
"""

import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple = (Exception,),
):
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (not counting the first call).
        base_delay: Initial delay in seconds between retries.
        max_delay: Maximum delay in seconds (caps exponential growth).
        retryable_exceptions: Tuple of exception types to catch and retry.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            "Retry %d/%d for %s after error: %s. Waiting %.1fs",
                            attempt + 1, max_retries, func.__name__, e, delay,
                        )
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator
