"""
Ragas provider adapter for RAG evaluation metrics.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from ulei.adapters.base import BaseAdapter
from ulei.core.schemas import DatasetItem, MetricResult
from ulei.utils.errors import EvaluationError, MetricNotSupportedError, ProviderError

try:
    from datasets import Dataset  # type: ignore
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


class RagasAdapter(BaseAdapter):
    """Adapter for Ragas evaluation provider."""

    # Required configuration keys
    required_config_keys = ["api_key"]

    def __init__(self, config: Dict[str, Any]):
        """Initialize Ragas adapter.

        Args:
            config: Configuration dictionary with API key and other settings
        """
        if not RAGAS_AVAILABLE:
            raise ProviderError(
                "Ragas is not installed. Please install with: pip install ragas", provider="ragas"
            )

        super().__init__(config)
        self._setup_ragas()

    def _setup_ragas(self) -> None:
        """Setup Ragas configuration."""
        # Set up API key
        import os

        if "api_key" in self.config:
            os.environ["OPENAI_API_KEY"] = self.config["api_key"]

        # Metric mapping
        self._metric_map = {
            "faithfulness": faithfulness,  # pyright: ignore[reportPossiblyUnboundVariable]
            "answer_relevancy": answer_relevancy,  # pyright: ignore[reportPossiblyUnboundVariable]
            "context_precision": context_precision,  # pyright: ignore[reportPossiblyUnboundVariable]
            "context_recall": context_recall,  # pyright: ignore[reportPossiblyUnboundVariable]
        }

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "ragas"

    @property
    def supported_metrics(self) -> List[str]:
        """Return list of supported metrics."""
        return list(self._metric_map.keys())

    async def evaluate_metric(
        self, metric_name: str, item: DatasetItem, config: Dict[str, Any]
    ) -> MetricResult:
        """Evaluate a single metric for a dataset item.

        Args:
            metric_name: Name of the metric to evaluate
            item: Dataset item to evaluate
            config: Metric-specific configuration

        Returns:
            MetricResult with evaluation outcome
        """
        start_time = time.time()

        try:
            # Validate metric support
            if not self.supports_metric(metric_name):
                raise MetricNotSupportedError(
                    f"Metric '{metric_name}' not supported by Ragas",
                    provider=self.provider_name,
                    metric=metric_name,
                    supported_metrics=self.supported_metrics,
                )

            # Validate required fields based on metric
            self._validate_item_for_metric(item, metric_name)

            # Convert item to Ragas format
            dataset_dict = self._item_to_ragas_format(item, metric_name)

            # Create Hugging Face dataset
            dataset = Dataset.from_dict(dataset_dict)  # pyright: ignore[reportPossiblyUnboundVariable]

            # Get metric instance
            metric = self._metric_map[metric_name]

            # Run evaluation
            result = await self._run_ragas_evaluation(dataset, [metric], config)

            # Extract score for this metric
            score = result[metric_name].iloc[0] if metric_name in result else None

            execution_time = time.time() - start_time

            # Estimate cost (approximate)
            cost_estimate = self.estimate_cost(metric_name, item, config)

            return self._create_result(
                metric_name=metric_name,
                item_id=item.id,
                score=score,
                execution_time=execution_time,
                cost_estimate=cost_estimate,
                evidence={
                    "ragas_result": result[metric_name].iloc[0] if metric_name in result else None
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time

            if isinstance(e, (EvaluationError, MetricNotSupportedError)):
                raise

            error_msg = f"Ragas evaluation failed: {str(e)}"
            self.logger.error(error_msg)

            return self._create_error_result(
                metric_name=metric_name,
                item_id=item.id,
                error=error_msg,
                execution_time=execution_time,
            )

    def _validate_item_for_metric(self, item: DatasetItem, metric_name: str) -> None:
        """Validate that item has required fields for the metric.

        Args:
            item: Dataset item to validate
            metric_name: Metric being evaluated

        Raises:
            EvaluationError: If required fields are missing
        """
        # Common requirements
        required_fields = ["output.answer"]

        # Metric-specific requirements
        if metric_name in [
            "faithfulness",
            "context_precision",
            "context_recall",
        ]:
            required_fields.extend(["input.question", "context"])
        elif metric_name == "answer_relevancy":
            required_fields.append("input.question")

        self._validate_required_fields(item, required_fields)

        # Additional validation for context-based metrics
        if metric_name in [
            "faithfulness",
            "context_precision",
            "context_recall",
        ]:
            if not item.context or len(item.context) == 0:
                raise EvaluationError(
                    f"Metric '{metric_name}' requires context but none provided",
                    provider=self.provider_name,
                    metric=metric_name,
                    item_id=item.id,
                )

    def _item_to_ragas_format(self, item: DatasetItem, metric_name: str) -> Dict[str, List]:
        """Convert dataset item to Ragas format.

        Args:
            item: Dataset item to convert
            metric_name: Metric being evaluated

        Returns:
            Dictionary in Ragas dataset format
        """
        # Base format
        ragas_dict = {
            "question": [self._get_field_value(item, "input.question", "")],
            "answer": [self._get_field_value(item, "output.answer", "")],
        }

        # Add context if available and needed
        if item.context and metric_name in [
            "faithfulness",
            "context_precision",
            "context_recall",
        ]:
            contexts = [ctx.text for ctx in item.context]
            ragas_dict["contexts"] = [contexts]

        # Add ground truth if available
        ground_truth = (
            self._get_field_value(item, "reference.ground_truth")
            or self._get_field_value(item, "reference.answer")
            or self._get_field_value(item, "reference.expected")
        )

        if ground_truth and metric_name in ["context_recall", "context_precision"]:
            ragas_dict["ground_truth"] = [ground_truth]

        return ragas_dict

    async def _run_ragas_evaluation(
        self,
        dataset: Dataset,  # pyright: ignore[reportPossiblyUnboundVariable]
        metrics: List[Any],
        config: Dict[str, Any],
    ) -> Any:
        """Run Ragas evaluation asynchronously.

        Args:
            dataset: Hugging Face dataset
            metrics: List of Ragas metrics
            config: Evaluation configuration

        Returns:
            Ragas evaluation result
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        def _evaluate():
            return evaluate(dataset, metrics=metrics)  # pyright: ignore[reportPossiblyUnboundVariable]

        return await loop.run_in_executor(None, _evaluate)

    def estimate_cost(
        self, metric_name: str, item: DatasetItem, config: Dict[str, Any]
    ) -> Optional[float]:
        """Estimate cost for Ragas evaluation.

        Args:
            metric_name: Metric name
            item: Dataset item
            config: Configuration

        Returns:
            Estimated cost in USD
        """
        # Rough cost estimation based on OpenAI pricing
        # This is approximate and depends on the model used

        base_cost_per_request = 0.002  # Approximate cost for gpt-3.5-turbo

        # Adjust based on content length
        question_len = len(str(self._get_field_value(item, "input.question", "")))
        answer_len = len(str(self._get_field_value(item, "output.answer", "")))
        context_len = 0

        if item.context:
            context_len = sum(len(ctx.text) for ctx in item.context)

        total_chars = question_len + answer_len + context_len

        # Scale cost based on content length (very rough approximation)
        length_multiplier = max(1.0, total_chars / 1000)  # Base 1000 chars

        return base_cost_per_request * length_multiplier

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Ragas-specific configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid
        """
        # Check base configuration
        if not super().validate_config(config):
            return False

        # Check API key
        if "api_key" not in config:
            self.logger.error("Ragas requires 'api_key' in configuration")
            return False

        api_key = config["api_key"]
        if not isinstance(api_key, str) or not api_key.strip():
            self.logger.error("Ragas API key must be a non-empty string")
            return False

        # Validate model if specified
        if "model" in config:
            model = config["model"]
            supported_models = [
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-4",
                "gpt-4-32k",
                "gpt-4-turbo-preview",
            ]
            if model not in supported_models:
                self.logger.warning(f"Model '{model}' may not be supported by Ragas")

        return True


# Register adapter with global registry
from ulei.core.registry import get_registry, register_adapter

if RAGAS_AVAILABLE:
    register_adapter("ragas", RagasAdapter)

    # Register metric mappings
    registry = get_registry()
    registry.register_provider_metrics(
        "ragas",
        {
            "faithfulness": "faithfulness",
            "answer_relevancy": "answer_relevancy",
            "context_relevancy": "context_precision",
            "context_recall": "context_recall",
        },
    )
