"""Retry with exponential backoff for network operations."""

import logging
import time

logger = logging.getLogger(__name__)


def retry_with_backoff(
    fn,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 16.0,
    exceptions: tuple = (Exception,),
):
    """Call fn() with exponential backoff on failure.

    Args:
        fn: Callable to execute
        max_retries: Maximum number of retry attempts (0 = no retries)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exceptions: Tuple of exception types to retry on

    Returns:
        Result of fn()

    Raises:
        The last exception if all retries are exhausted
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except exceptions as e:
            last_error = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"All {max_retries + 1} attempts failed: {e}"
                )
    raise last_error
