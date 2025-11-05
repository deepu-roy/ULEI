"""
Retry policy implementation for handling flaky evaluations.

Provides configurable retry logic with exponential backoff and jitter
to handle transient failures in LLM API calls and evaluation providers.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from ulei.utils.errors import ProviderError, RateLimitError

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Base for exponential backoff
    jitter: bool = True  # Add random jitter to delays

    # Exception types that should trigger retries
    retryable_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        RateLimitError,
        ProviderError,  # Provider-specific transient errors
    )

    # Exception types that should NOT trigger retries
    non_retryable_exceptions: tuple = (ValueError, TypeError, AttributeError, KeyError)

    def __post_init__(self):
        """Validate configuration."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.exponential_base <= 1:
            raise ValueError("exponential_base must be > 1")


class RetryManager:
    """Manages retry logic for evaluation operations."""

    def __init__(self, config: Optional[RetryConfig] = None):
        """Initialize retry manager.

        Args:
            config: Retry configuration (uses defaults if None)
        """
        self.config = config or RetryConfig()
        self.logger = logging.getLogger("ulei.retry")

    async def execute_with_retry(
        self,
        operation: Callable[[], Awaitable[Any]],
        context: str = "operation",
        custom_config: Optional[RetryConfig] = None,
    ) -> Any:
        """
        Execute an async operation with retry logic.

        Args:
            operation: Async callable to execute
            context: Description of operation for logging
            custom_config: Override default retry config for this operation

        Returns:
            Result of the operation

        Raises:
            The last exception if all retries are exhausted
        """
        config = custom_config or self.config
        last_exception = None

        for attempt in range(1, config.max_attempts + 1):
            try:
                start_time = time.time()
                result = await operation()
                execution_time = time.time() - start_time

                if attempt > 1:
                    self.logger.info(
                        f"✅ {context} succeeded on attempt {attempt}/{config.max_attempts} "
                        f"after {execution_time:.2f}s"
                    )

                return result

            except Exception as e:
                last_exception = e
                execution_time = time.time() - start_time

                # Check if this exception should trigger a retry
                if not self._should_retry(e, config):
                    self.logger.warning(
                        f"❌ {context} failed with non-retryable error: {type(e).__name__}: {e}"
                    )
                    raise e

                # Don't sleep after the last attempt
                if attempt == config.max_attempts:
                    self.logger.error(
                        f"❌ {context} failed after {config.max_attempts} attempts. "
                        f"Final error: {type(e).__name__}: {e}"
                    )
                    break

                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt, config)

                self.logger.warning(
                    f"⚠️  {context} failed on attempt {attempt}/{config.max_attempts} "
                    f"after {execution_time:.2f}s: {type(e).__name__}: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                await asyncio.sleep(delay)

        # All retries exhausted
        raise last_exception

    def execute_with_retry_sync(
        self,
        operation: Callable[[], Any],
        context: str = "operation",
        custom_config: Optional[RetryConfig] = None,
    ) -> Any:
        """
        Execute a synchronous operation with retry logic.

        Args:
            operation: Callable to execute
            context: Description of operation for logging
            custom_config: Override default retry config for this operation

        Returns:
            Result of the operation

        Raises:
            The last exception if all retries are exhausted
        """
        config = custom_config or self.config
        last_exception = None

        for attempt in range(1, config.max_attempts + 1):
            try:
                start_time = time.time()
                result = operation()
                execution_time = time.time() - start_time

                if attempt > 1:
                    self.logger.info(
                        f"✅ {context} succeeded on attempt {attempt}/{config.max_attempts} "
                        f"after {execution_time:.2f}s"
                    )

                return result

            except Exception as e:
                last_exception = e
                execution_time = time.time() - start_time

                # Check if this exception should trigger a retry
                if not self._should_retry(e, config):
                    self.logger.warning(
                        f"❌ {context} failed with non-retryable error: {type(e).__name__}: {e}"
                    )
                    raise e

                # Don't sleep after the last attempt
                if attempt == config.max_attempts:
                    self.logger.error(
                        f"❌ {context} failed after {config.max_attempts} attempts. "
                        f"Final error: {type(e).__name__}: {e}"
                    )
                    break

                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt, config)

                self.logger.warning(
                    f"⚠️  {context} failed on attempt {attempt}/{config.max_attempts} "
                    f"after {execution_time:.2f}s: {type(e).__name__}: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                time.sleep(delay)

        # All retries exhausted
        raise last_exception

    def _should_retry(self, exception: Exception, config: RetryConfig) -> bool:
        """
        Determine if an exception should trigger a retry.

        Args:
            exception: The exception that occurred
            config: Retry configuration

        Returns:
            True if operation should be retried
        """
        # Never retry non-retryable exceptions
        if isinstance(exception, config.non_retryable_exceptions):
            return False

        # Always retry retryable exceptions
        if isinstance(exception, config.retryable_exceptions):
            return True

        # For unknown exceptions, be conservative and retry
        # This can be adjusted based on experience
        return True

    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """
        Calculate delay before next retry attempt.

        Args:
            attempt: Current attempt number (1-based)
            config: Retry configuration

        Returns:
            Delay in seconds
        """
        # Exponential backoff: base_delay * (exponential_base ^ (attempt - 1))
        delay = config.base_delay * (config.exponential_base ** (attempt - 1))

        # Cap at maximum delay
        delay = min(delay, config.max_delay)

        # Add jitter if enabled (±25% of delay)
        if config.jitter:
            jitter_range = delay * 0.25
            jitter = random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay + jitter)

        return delay


# Decorator for easy retry functionality
def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[tuple] = None,
    context: Optional[str] = None,
):
    """
    Decorator to add retry functionality to async functions.

    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter
        retryable_exceptions: Exception types that should trigger retries
        context: Description for logging

    Returns:
        Decorated function with retry logic
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
                retryable_exceptions=retryable_exceptions or RetryConfig().retryable_exceptions,
            )

            retry_manager = RetryManager(config)
            operation_context = context or f"{func.__name__}"

            async def operation():
                return await func(*args, **kwargs)

            return await retry_manager.execute_with_retry(operation, operation_context)

        return wrapper

    return decorator


# Global retry manager instance
default_retry_manager = RetryManager()


# Convenience functions
async def retry_async(
    operation: Callable[[], Awaitable[Any]], context: str = "operation", **retry_kwargs
) -> Any:
    """
    Convenience function for retrying async operations.

    Args:
        operation: Async operation to retry
        context: Description for logging
        **retry_kwargs: Retry configuration parameters

    Returns:
        Result of the operation
    """
    config = RetryConfig(**retry_kwargs) if retry_kwargs else None
    return await default_retry_manager.execute_with_retry(operation, context, config)


def retry_sync(operation: Callable[[], Any], context: str = "operation", **retry_kwargs) -> Any:
    """
    Convenience function for retrying sync operations.

    Args:
        operation: Synchronous operation to retry
        context: Description for logging
        **retry_kwargs: Retry configuration parameters

    Returns:
        Result of the operation
    """
    config = RetryConfig(**retry_kwargs) if retry_kwargs else None
    return default_retry_manager.execute_with_retry_sync(operation, context, config)
