"""
Base adapter class with common functionality for provider implementations.
"""

import asyncio
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

from ulei.core.interfaces import BaseAdapter as CoreBaseAdapter
from ulei.core.schemas import DatasetItem, MetricResult
from ulei.utils.errors import EvaluationError

logger = logging.getLogger(__name__)


class BaseAdapter(CoreBaseAdapter):
    """Enhanced base adapter with common functionality."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize adapter with configuration and validation."""
        super().__init__(config)
        self._setup_logging()
        self._setup_caching()
        self._setup_rate_limiting()

    def _setup_logging(self) -> None:
        """Setup provider-specific logging."""
        try:
            provider = self.provider_name
        except Exception:
            # Fallback if provider_name not accessible yet
            provider = "unknown"
        self.logger = logging.getLogger(f"ulei.adapters.{provider}")

    def _setup_caching(self) -> None:
        """Setup response caching configuration."""
        self.cache_enabled = self.config.get("cache_enabled", True)
        self._cache: Dict[str, MetricResult] = {}

    def _setup_rate_limiting(self) -> None:
        """Setup rate limiting configuration."""
        self.rate_limit = self.config.get("rate_limit", None)
        self._last_request_time = 0.0
        self._request_count = 0

    async def evaluate_metric_with_retry(
        self,
        metric_name: str,
        item: DatasetItem,
        config: Dict[str, Any],
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> MetricResult:
        """Evaluate metric with retry logic and error handling.

        Args:
            metric_name: Name of the metric to evaluate
            item: Dataset item to evaluate
            config: Configuration for the evaluation
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier

        Returns:
            MetricResult with evaluation outcome
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Check cache first
                if self.cache_enabled:
                    cache_key = self._generate_cache_key(metric_name, item, config)
                    if cache_key in self._cache:
                        self.logger.debug(f"Cache hit for {metric_name} on item {item.id}")
                        return self._cache[cache_key]

                # Apply rate limiting
                await self._apply_rate_limiting()

                # Perform evaluation
                start_time = time.time()
                result = await self.evaluate_metric(metric_name, item, config)
                result.execution_time = time.time() - start_time

                # Cache successful result
                if self.cache_enabled and result.error is None:
                    self._cache[cache_key] = result  # pyright: ignore[reportPossiblyUnboundVariable]

                self.logger.debug(
                    f"Successfully evaluated {metric_name} for item {item.id} "
                    f"(attempt {attempt + 1})"
                )
                return result

            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"Attempt {attempt + 1} failed for {metric_name} on item {item.id}: {e}"
                )

                if attempt < max_retries:
                    wait_time = backoff_factor * (2**attempt)
                    self.logger.debug(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)

        # All retries failed
        error_msg = f"All {max_retries + 1} attempts failed. Last error: {last_error}"
        self.logger.error(error_msg)

        return self._create_error_result(
            metric_name=metric_name,
            item_id=item.id,
            error=error_msg,
            execution_time=time.time() - start_time if "start_time" in locals() else 0.0,  # pyright: ignore[reportPossiblyUnboundVariable]
        )

    def _generate_cache_key(
        self, metric_name: str, item: DatasetItem, config: Dict[str, Any]
    ) -> str:
        """Generate cache key for caching evaluation results.

        Args:
            metric_name: Name of the metric
            item: Dataset item
            config: Evaluation configuration

        Returns:
            Cache key string
        """
        # Create deterministic hash from item content and config
        content = {
            "provider": self.provider_name,
            "metric": metric_name,
            "input": item.input,
            "output": item.output,
            "reference": item.reference,
            "context": [
                ctx.model_dump() if hasattr(ctx, "model_dump") else ctx.dict()
                for ctx in (item.context or [])
            ]
            if item.context
            else None,
            "config": config,
        }

        # Sort keys for deterministic hashing
        import json

        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

    async def _apply_rate_limiting(self) -> None:
        """Apply rate limiting to API requests."""
        if self.rate_limit is None:
            return

        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < (1.0 / self.rate_limit):
            wait_time = (1.0 / self.rate_limit) - time_since_last
            self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

        self._last_request_time = time.time()
        self._request_count += 1

    def _create_error_result(
        self,
        metric_name: str,
        item_id: str,
        error: str,
        execution_time: float = 0.0,
        raw_response: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """Create a MetricResult for failed evaluations.

        Args:
            metric_name: Name of the metric
            item_id: ID of the dataset item
            error: Error message
            execution_time: Time taken for the evaluation
            raw_response: Optional raw response from provider

        Returns:
            MetricResult with error information
        """
        return self._create_result(
            metric_name=metric_name,
            item_id=item_id,
            score=None,
            error=error,
            execution_time=execution_time,
            raw_response=raw_response,
        )

    def _normalize_score(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize score to [0,1] range.

        Args:
            score: Raw score from provider
            min_val: Minimum possible value from provider
            max_val: Maximum possible value from provider

        Returns:
            Normalized score in [0,1] range
        """
        if max_val == min_val:
            return 0.0

        normalized = (score - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))

    def _validate_required_fields(self, item: DatasetItem, required_fields: List[str]) -> None:
        """Validate that dataset item has required fields for evaluation.

        Args:
            item: Dataset item to validate
            required_fields: List of required field paths (e.g., ['input.query', 'output.answer'])

        Raises:
            EvaluationError: If required fields are missing
        """
        missing_fields = []

        for field_path in required_fields:
            if not self._has_field(item, field_path):
                missing_fields.append(field_path)

        if missing_fields:
            raise EvaluationError(
                f"Missing required fields for {self.provider_name}: {missing_fields}"
            )

    def _has_field(self, item: DatasetItem, field_path: str) -> bool:
        """Check if dataset item has a specific field.

        Args:
            item: Dataset item to check
            field_path: Dot-separated field path (e.g., 'input.query')

        Returns:
            True if field exists and is not None
        """
        parts = field_path.split(".")
        current = item

        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False

            if current is None:
                return False

        return True

    def _get_field_value(self, item: DatasetItem, field_path: str, default: Any = None) -> Any:
        """Get value from dataset item using field path.

        Args:
            item: Dataset item
            field_path: Dot-separated field path
            default: Default value if field doesn't exist

        Returns:
            Field value or default
        """
        parts = field_path.split(".")
        current = item

        try:
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                elif isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current
        except (AttributeError, KeyError, TypeError):
            return default

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate provider configuration with common checks.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid
        """
        # Allow empty config for introspection (e.g., checking supported metrics)
        if not config:
            return True

        # Check for required configuration keys
        required_keys = getattr(self, "required_config_keys", [])
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required configuration key: {key}")
                return False

        # Validate timeout
        if "timeout" in config:
            timeout = config["timeout"]
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                logger.error(f"Invalid timeout value: {timeout}")
                return False

        return True

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache hit count and total size
        """
        return {
            "cache_size": len(self._cache),
            "request_count": self._request_count,
            "hit_rate": int(len(self._cache) / max(self._request_count, 1) * 100),
        }

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self._cache.clear()
        self.logger.debug("Cleared evaluation cache")
