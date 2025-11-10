"""
DeepEval provider adapter for LLM evaluation metrics.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from ulei.adapters.base import BaseAdapter
from ulei.core.schemas import DatasetItem, MetricResult
from ulei.utils.errors import EvaluationError, MetricNotSupportedError, ProviderError

logger = logging.getLogger(__name__)

try:
    from deepeval.metrics import (  # type: ignore
        AnswerRelevancyMetric,
        BiasMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        HallucinationMetric,
        PromptAlignmentMetric,
        RoleAdherenceMetric,
        SummarizationMetric,
        ToolCorrectnessMetric,
        ToxicityMetric,
    )
    from deepeval.test_case import LLMTestCase  # type: ignore

    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False


class DeepEvalAdapter(BaseAdapter):
    """Adapter for DeepEval evaluation provider."""

    # Required configuration keys
    required_config_keys = ["api_key"]

    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize DeepEval adapter.

        Args:
            config: Configuration dictionary with API key and other settings
        """
        if not DEEPEVAL_AVAILABLE:
            raise ProviderError(
                "DeepEval is not installed. Please install with: pip install deepeval",
                provider="deepeval",
            )

        # Use empty dict if no config provided (for introspection)
        super().__init__(config or {})
        self._setup_deepeval()

    def _setup_deepeval(self) -> None:
        """Setup DeepEval configuration."""
        # Set up API key
        import os

        if "api_key" in self.config:
            os.environ["OPENAI_API_KEY"] = self.config["api_key"]

        # Metric class mapping
        self._metric_classes = {
            # Core RAG metrics
            "faithfulness": FaithfulnessMetric,  # pyright: ignore[reportPossiblyUnboundVariable]
            "answer_relevancy": AnswerRelevancyMetric,  # pyright: ignore[reportPossiblyUnboundVariable]
            "contextual_precision": ContextualPrecisionMetric,  # pyright: ignore[reportPossiblyUnboundVariable]
            "contextual_recall": ContextualRecallMetric,  # pyright: ignore[reportPossiblyUnboundVariable]
            "contextual_relevancy": ContextualRelevancyMetric,  # pyright: ignore[reportPossiblyUnboundVariable]
            # Safety and quality metrics
            "toxicity": ToxicityMetric,  # pyright: ignore[reportPossiblyUnboundVariable]
            "bias": BiasMetric,  # pyright: ignore[reportPossiblyUnboundVariable]
            "hallucination": HallucinationMetric,  # pyright: ignore[reportPossiblyUnboundVariable]
            # Advanced task-specific metrics
            "summarization": SummarizationMetric,  # pyright: ignore[reportPossiblyUnboundVariable]
            "tool_correctness": ToolCorrectnessMetric,  # pyright: ignore[reportPossiblyUnboundVariable]
            "role_adherence": RoleAdherenceMetric,  # pyright: ignore[reportPossiblyUnboundVariable]
            "prompt_alignment": PromptAlignmentMetric,  # pyright: ignore[reportPossiblyUnboundVariable]
        }

        # Alternative metric name mappings
        self._metric_aliases = {
            "answer_relevance": "answer_relevancy",
            "context_precision": "contextual_precision",
            "context_recall": "contextual_recall",
            "context_relevancy": "contextual_relevancy",
        }

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "deepeval"

    @property
    def supported_metrics(self) -> List[str]:
        """Return list of supported metrics."""
        return list(self._metric_classes.keys()) + list(self._metric_aliases.keys())

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
            # Resolve metric name aliases
            resolved_metric = self._metric_aliases.get(metric_name, metric_name)

            # Validate metric support
            if not self.supports_metric(metric_name):
                raise MetricNotSupportedError(
                    f"Metric '{metric_name}' not supported by DeepEval",
                    provider=self.provider_name,
                    metric=metric_name,
                    supported_metrics=self.supported_metrics,
                )

            # Validate required fields
            self._validate_item_for_metric(item, resolved_metric)

            # Create test case
            test_case = self._create_test_case(item, resolved_metric)

            # Create and configure metric
            metric_instance = self._create_metric_instance(resolved_metric, config)

            # Run evaluation
            score = await self._run_deepeval_evaluation(metric_instance, test_case)

            execution_time = time.time() - start_time

            # Estimate cost
            cost_estimate = self.estimate_cost(metric_name, item, config)

            return self._create_result(
                metric_name=metric_name,
                item_id=item.id,
                score=score,
                execution_time=execution_time,
                cost_estimate=cost_estimate,
                evidence={
                    "deepeval_score": score,
                    "metric_type": resolved_metric,
                    "reason": getattr(metric_instance, "reason", None)
                    if hasattr(metric_instance, "reason")
                    else None,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time

            if isinstance(e, (EvaluationError, MetricNotSupportedError)):
                raise

            error_msg = f"DeepEval evaluation failed: {str(e)}"
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

        # Metric-specific requirements based on DeepEval documentation
        if metric_name in [
            "faithfulness",
            "contextual_precision",
            "contextual_recall",
            "contextual_relevancy",
            "hallucination",
        ]:
            # Context-based metrics need input query and context
            # Note: DeepEval uses 'query' but we accept both 'query' and 'question'
            if not (
                self._get_field_value(item, "input.query")
                or self._get_field_value(item, "input.question")
            ):
                required_fields.append("input.query")
            required_fields.append("context")
        elif metric_name == "answer_relevancy":
            # Answer relevancy needs query and answer
            if not (
                self._get_field_value(item, "input.query")
                or self._get_field_value(item, "input.question")
            ):
                required_fields.append("input.query")
        elif metric_name == "summarization":
            # Summarization needs original text (in context) and summary (in answer)
            if not (
                self._get_field_value(item, "input.query")
                or self._get_field_value(item, "input.question")
            ):
                required_fields.append("input.query")
            required_fields.append("context")
        elif metric_name in ["toxicity", "bias"]:
            # Safety metrics only need the answer
            pass
        elif metric_name == "tool_correctness":
            # Tool correctness needs expected tools in config
            if "available_tools" not in self.config:
                raise EvaluationError(
                    "Metric 'tool_correctness' requires 'available_tools' in configuration",
                    provider=self.provider_name,
                    metric=metric_name,
                    item_id=item.id,
                )
        elif metric_name == "role_adherence":
            # Role adherence needs the answer to check against role
            pass
        elif metric_name == "prompt_alignment":
            # Prompt alignment needs prompt instructions in config
            if "prompt_instructions" not in self.config:
                raise EvaluationError(
                    "Metric 'prompt_alignment' requires 'prompt_instructions' in configuration",
                    provider=self.provider_name,
                    metric=metric_name,
                    item_id=item.id,
                )

        self._validate_required_fields(item, required_fields)

        # Additional validation for context-based metrics
        if metric_name in [
            "faithfulness",
            "contextual_precision",
            "contextual_recall",
            "contextual_relevancy",
            "hallucination",
            "summarization",
        ]:
            if not item.context or len(item.context) == 0:
                raise EvaluationError(
                    f"Metric '{metric_name}' requires context but none provided",
                    provider=self.provider_name,
                    metric=metric_name,
                    item_id=item.id,
                )

    def _create_test_case(self, item: DatasetItem, metric_name: str) -> "LLMTestCase":
        """Create DeepEval test case from dataset item.

        Args:
            item: Dataset item
            metric_name: Metric name

        Returns:
            DeepEval LLMTestCase
        """
        # Extract required fields
        input_text = (
            self._get_field_value(item, "input.query")
            or self._get_field_value(item, "input.question")
            or self._get_field_value(item, "input.prompt", "")
        )

        actual_output = self._get_field_value(item, "output.answer") or self._get_field_value(
            item, "output.response", ""
        )

        # Create base test case
        test_case_kwargs = {"input": input_text, "actual_output": actual_output}

        # Add expected output if available
        expected_output = (
            self._get_field_value(item, "reference.answer")
            or self._get_field_value(item, "reference.expected")
            or self._get_field_value(item, "reference.ground_truth")
        )
        if expected_output:
            test_case_kwargs["expected_output"] = expected_output

        # Add retrieval context if available and needed
        # Note: DeepEval has both 'context' and 'retrieval_context' fields
        # Different metrics use different fields, so we set both for compatibility
        if item.context and metric_name in [
            "faithfulness",
            "contextual_precision",
            "contextual_recall",
            "contextual_relevancy",
            "hallucination",
            "summarization",
        ]:
            retrieval_context = [ctx.text for ctx in item.context]
            test_case_kwargs["context"] = retrieval_context  # Used by HallucinationMetric
            test_case_kwargs["retrieval_context"] = retrieval_context  # Used by FaithfulnessMetric

        return LLMTestCase(**test_case_kwargs)  # pyright: ignore[reportPossiblyUnboundVariable]

    def _create_metric_instance(self, metric_name: str, config: Dict[str, Any]) -> Any:
        """Create DeepEval metric instance.

        Args:
            metric_name: Name of the metric
            config: Metric configuration

        Returns:
            DeepEval metric instance
        """
        metric_class = self._metric_classes[metric_name]

        # Build metric configuration
        metric_kwargs = {}

        # Add model configuration if specified
        if "model" in config:
            metric_kwargs["model"] = config["model"]

        # Add threshold if specified
        if "threshold" in config:
            metric_kwargs["threshold"] = config["threshold"]

        # Add metric-specific configurations
        if metric_name == "bias" and "bias_type" in config:
            metric_kwargs["bias_type"] = config["bias_type"]
        elif metric_name == "tool_correctness" and "available_tools" in config:
            metric_kwargs["available_tools"] = config["available_tools"]
        elif metric_name == "prompt_alignment" and "prompt_instructions" in config:
            metric_kwargs["prompt_instructions"] = config["prompt_instructions"]
        elif metric_name == "summarization":
            # Summarization has special parameters
            if "n" in config:
                metric_kwargs["n"] = config["n"]
            if "assessment_questions" in config:
                metric_kwargs["assessment_questions"] = config["assessment_questions"]

        # Create and return metric instance
        return metric_class(**metric_kwargs)

    async def _run_deepeval_evaluation(self, metric: Any, test_case: "LLMTestCase") -> float:
        """Run DeepEval metric evaluation.

        Args:
            metric: DeepEval metric instance
            test_case: Test case to evaluate

        Returns:
            Evaluation score (0.0 to 1.0)
        """
        # Run evaluation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        def _evaluate() -> float:
            metric.measure(test_case)
            return float(metric.score)

        score = await loop.run_in_executor(None, _evaluate)

        # Ensure score is in [0, 1] range
        return self._normalize_score(score)

    def estimate_cost(
        self, metric_name: str, item: DatasetItem, config: Dict[str, Any]
    ) -> Optional[float]:
        """Estimate cost for DeepEval evaluation.

        Args:
            metric_name: Metric name
            item: Dataset item
            config: Configuration

        Returns:
            Estimated cost in USD
        """
        # Rough cost estimation based on OpenAI pricing
        base_cost_per_request = 0.002  # Approximate cost for gpt-3.5-turbo

        # Adjust based on metric complexity
        complexity_multipliers = {
            "faithfulness": 1.5,
            "answer_relevancy": 1.2,
            "contextual_precision": 1.8,
            "contextual_recall": 1.8,
            "contextual_relevancy": 1.3,
            "toxicity": 1.0,
            "bias": 1.1,
            "hallucination": 1.4,
            "summarization": 2.0,  # More complex, multiple questions
            "tool_correctness": 1.3,
            "role_adherence": 1.1,
            "prompt_alignment": 1.2,
        }

        resolved_metric = self._metric_aliases.get(metric_name, metric_name)
        multiplier = complexity_multipliers.get(resolved_metric, 1.0)

        # Adjust based on content length
        input_len = len(
            str(
                self._get_field_value(item, "input.query", "")
                or self._get_field_value(item, "input.question", "")
            )
        )
        output_len = len(str(self._get_field_value(item, "output.answer", "")))
        context_len = 0

        if item.context:
            context_len = sum(len(ctx.text) for ctx in item.context)

        total_chars = input_len + output_len + context_len
        length_multiplier = max(1.0, total_chars / 1000)

        return base_cost_per_request * multiplier * length_multiplier

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate DeepEval-specific configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid
        """
        # Check base configuration
        if not super().validate_config(config):
            return False

        # Allow empty config for introspection (e.g., checking supported metrics)
        if not config:
            return True

        # Check API key
        if "api_key" not in config:
            logger.error("DeepEval requires 'api_key' in configuration")
            return False

        api_key = config["api_key"]
        if not isinstance(api_key, str) or not api_key.strip():
            logger.error("DeepEval API key must be a non-empty string")
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
                logger.warning(f"Model '{model}' may not be supported by DeepEval")

        # Validate threshold if specified
        if "threshold" in config:
            threshold = config["threshold"]
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                logger.error("Threshold must be a number between 0 and 1")
                return False

        return True


# Register adapter with global registry
from ulei.core.registry import get_registry, register_adapter

if DEEPEVAL_AVAILABLE:
    register_adapter("deepeval", DeepEvalAdapter)

    # Register metric mappings
    registry = get_registry()
    registry.register_provider_metrics(
        "deepeval",
        {
            "answer_relevancy": "answer_relevancy",
            "answer_correctness": "correctness",
            "faithfulness": "faithfulness",
            "toxicity": "toxicity",
            "bias": "bias",
        },
    )
