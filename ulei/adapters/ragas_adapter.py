"""
Ragas provider adapter for RAG evaluation metrics.
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
    import ragas
    from datasets import Dataset  # type: ignore
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        AnswerSimilarity,
        ContextEntityRecall,
        ContextPrecision,
        ContextRecall,
        ContextRelevance,
        ContextUtilization,
        Faithfulness,
        ResponseRelevancy,
        SemanticSimilarity,
    )

    RAGAS_AVAILABLE = True

    # Check Ragas version - only support v0.3+
    RAGAS_VERSION = tuple(map(int, ragas.__version__.split(".")[:2]))
    if RAGAS_VERSION < (0, 3):
        raise ProviderError(
            f"Ragas version {ragas.__version__} is not supported. "
            "Please upgrade to Ragas >= 0.3.0 with: pip install 'ragas>=0.3.0'",
            provider="ragas",
        )

except ImportError:
    RAGAS_AVAILABLE = False


class RagasAdapter(BaseAdapter):
    """Adapter for Ragas evaluation provider."""

    # Required configuration keys
    required_config_keys = ["api_key"]

    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize Ragas adapter.

        Args:
            config: Configuration dictionary with API key and other settings
        """
        if not RAGAS_AVAILABLE:
            raise ProviderError(
                "Ragas is not installed. Please install with: pip install ragas", provider="ragas"
            )

        # Use empty dict if no config provided (for introspection)
        super().__init__(config or {})
        self._setup_ragas()

    def _setup_ragas(self) -> None:
        """Setup Ragas configuration for Ragas v0.3+.

        Note: Uses deprecated LangchainLLMWrapper instead of llm_factory
        because llm_factory returns InstructorLLM which has a bug with
        agenerate_prompt method. This will be updated when Ragas fixes the issue.
        """
        # Set up API key
        import os

        if "api_key" in self.config:
            os.environ["OPENAI_API_KEY"] = self.config["api_key"]

        # Create LangChain LLM and wrap it (using deprecated approach that works)
        model_name = self.config.get("default_model", "gpt-3.5-turbo")
        base_url = self.config.get("base_url", "https://api.openai.com/v1")
        api_key = self.config.get("api_key", os.environ.get("OPENAI_API_KEY", "dummy"))

        # Create LangChain LLM
        langchain_llm = ChatOpenAI(
            model=model_name,  # type: ignore
            base_url=base_url,
            api_key=api_key,
        )

        # Wrap with Ragas wrapper (deprecated but works, llm_factory is buggy)
        ragas_llm = LangchainLLMWrapper(langchain_llm)

        # Create metrics with explicit LLM
        # Note: Some metrics require embeddings, we'll handle those separately
        self._metric_map = {
            # Core RAG metrics
            "faithfulness": Faithfulness(llm=ragas_llm),
            "answer_relevancy": AnswerRelevancy(llm=ragas_llm),
            "context_precision": ContextPrecision(llm=ragas_llm),
            "context_recall": ContextRecall(llm=ragas_llm),
            # Additional answer quality metrics
            "answer_correctness": AnswerCorrectness(llm=ragas_llm),
            "answer_similarity": AnswerSimilarity(),  # Uses embeddings by default
            # Context evaluation metrics
            "context_entity_recall": ContextEntityRecall(llm=ragas_llm),
            "context_relevance": ContextRelevance(llm=ragas_llm),
            "context_utilization": ContextUtilization(llm=ragas_llm),
            # Response metrics
            "response_relevancy": ResponseRelevancy(llm=ragas_llm),
            # Semantic similarity (embedding-based)
            "semantic_similarity": SemanticSimilarity(),
        }

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "ragas"

    @property
    def supported_metrics(self) -> List[str]:
        """Return list of supported metrics."""
        return list(self._metric_map.keys())

    def supports_batch_evaluation(self) -> bool:
        """Ragas supports efficient batch evaluation of multiple metrics.

        Returns:
            True, indicating batch evaluation is supported and recommended
        """
        return True

    async def evaluate_metrics_batch(
        self, metric_names: List[str], item: DatasetItem, config: Dict[str, Any]
    ) -> Dict[str, MetricResult]:
        """Evaluate multiple metrics for a single item in one batch call.

        This is more efficient than calling evaluate_metric() multiple times
        as it makes a single API call to evaluate all metrics at once.

        Args:
            metric_names: List of metric names to evaluate
            item: Dataset item to evaluate
            config: Configuration for the evaluation

        Returns:
            Dictionary mapping metric names to their results
        """
        start_time = time.time()
        results = {}

        try:
            # Validate all metrics are supported
            unsupported = [m for m in metric_names if not self.supports_metric(m)]
            if unsupported:
                raise MetricNotSupportedError(
                    f"Metrics not supported by Ragas: {unsupported}",
                    provider=self.provider_name,
                    metric=", ".join(unsupported),
                    supported_metrics=self.supported_metrics,
                )

            # Validate required fields for all metrics
            for metric_name in metric_names:
                self._validate_item_for_metric(item, metric_name)

            # Determine which fields are needed for any of the requested metrics
            needs_context = any(
                m
                in [
                    "faithfulness",
                    "context_precision",
                    "context_recall",
                    "context_entity_recall",
                    "context_relevance",
                    "context_utilization",
                ]
                for m in metric_names
            )
            needs_reference = any(
                m
                in [
                    "answer_correctness",
                    "answer_similarity",
                    "semantic_similarity",
                    "context_recall",
                    "context_precision",
                ]
                for m in metric_names
            )

            # Convert item to Ragas format (include all needed fields)
            dataset_dict = self._item_to_ragas_format(
                item, metric_names[0] if needs_context else metric_names[0]
            )

            # Ensure context and reference are included if any metric needs them
            if needs_context and item.context:
                contexts = [ctx.text for ctx in item.context]
                dataset_dict["retrieved_contexts"] = [contexts]

            # Add reference if available for metrics that need it
            if needs_reference:
                ground_truth = (
                    self._get_field_value(item, "reference.ground_truth")
                    or self._get_field_value(item, "reference.answer")
                    or self._get_field_value(item, "reference.expected")
                )
                if ground_truth:
                    dataset_dict["reference"] = [ground_truth]

            # Create Hugging Face dataset
            dataset = Dataset.from_dict(dataset_dict)

            # Get all metric instances
            metrics = [self._metric_map[name] for name in metric_names]

            # Run batch evaluation (single API call for all metrics)
            self.logger.info(
                f"Batch evaluating {len(metric_names)} metrics for item {item.id}: "
                f"{', '.join(metric_names)}"
            )
            result = await self._run_ragas_evaluation(dataset, metrics, config)

            # Convert result to dict
            if hasattr(result, "to_pandas"):
                df = result.to_pandas()
                result_dict = df.to_dict("list")
            elif hasattr(result, "__dict__"):
                result_dict = vars(result)
            else:
                result_dict = dict(result)

            # Extract scores for each metric
            execution_time = time.time() - start_time

            for metric_name in metric_names:
                if metric_name in result_dict:
                    score_value = result_dict[metric_name]

                    # Handle both list and scalar values
                    if isinstance(score_value, list):
                        if len(score_value) > 0:
                            score = score_value[0]
                        else:
                            self.logger.warning(f"Empty list returned for {metric_name}")
                            score = None
                    else:
                        score = score_value

                    # Handle NaN values and floating point precision issues from Ragas
                    import math

                    if score is not None and math.isnan(score):
                        self.logger.warning(
                            f"Ragas returned NaN for {metric_name} on item {item.id}. "
                            "This is a known Ragas issue. Setting score to None."
                        )
                        score = None
                    elif score is not None and isinstance(score, (int, float)):
                        # Clamp floating point values to [0.0, 1.0] range
                        # Handles precision issues like 1.0000000000000002
                        score = max(0.0, min(1.0, float(score)))

                    cost_estimate = self.estimate_cost(metric_name, item, config)

                    results[metric_name] = self._create_result(
                        metric_name=metric_name,
                        item_id=item.id,
                        score=score,
                        execution_time=execution_time / len(metric_names),  # Divide time
                        cost_estimate=cost_estimate,
                        evidence={"ragas_result": score, "batch_evaluated": True},
                    )
                else:
                    self.logger.warning(
                        f"Metric {metric_name} not found in batch result keys: {result_dict.keys()}"
                    )
                    results[metric_name] = self._create_error_result(
                        metric_name=metric_name,
                        item_id=item.id,
                        error="Metric not found in batch evaluation result",
                        execution_time=execution_time / len(metric_names),
                    )

            self.logger.info(
                f"Batch evaluation completed for item {item.id} in {execution_time:.2f}s"
            )
            return results

        except Exception as e:
            execution_time = time.time() - start_time

            if isinstance(e, (EvaluationError, MetricNotSupportedError)):
                raise

            error_msg = f"Ragas batch evaluation failed: {str(e)}"
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(f"Exception message: {str(e)}")
            self.logger.error(error_msg, exc_info=True)

            # Return error results for all metrics
            for metric_name in metric_names:
                results[metric_name] = self._create_error_result(
                    metric_name=metric_name,
                    item_id=item.id,
                    error=error_msg,
                    execution_time=execution_time / len(metric_names),
                )

            return results

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

            # Debug logging
            self.logger.debug(f"Ragas result type: {type(result)}")
            self.logger.debug(f"Ragas result: {result}")

            # Ragas v0.3 returns an EvaluationResult object, convert to dict
            if hasattr(result, "to_pandas"):
                # Convert to pandas DataFrame then to dict
                df = result.to_pandas()
                result_dict = df.to_dict("list")
                self.logger.debug(f"Converted result_dict keys: {result_dict.keys()}")
            elif hasattr(result, "__dict__"):
                result_dict = vars(result)
            else:
                result_dict = dict(result)

            # Extract score for this metric
            if metric_name in result_dict:
                score_value = result_dict[metric_name]
                self.logger.debug(f"Score value type: {type(score_value)}, value: {score_value}")

                # Handle both list and scalar values
                if isinstance(score_value, list):
                    if len(score_value) > 0:
                        score = score_value[0]
                    else:
                        self.logger.warning(f"Empty list returned for {metric_name}")
                        score = None
                else:
                    score = score_value

                # Handle NaN values and floating point precision issues from Ragas
                import math

                if score is not None and math.isnan(score):
                    self.logger.warning(
                        f"Ragas returned NaN for {metric_name} on item {item.id}. "
                        "This is a known Ragas issue. Setting score to None."
                    )
                    score = None
                elif score is not None and isinstance(score, (int, float)):
                    # Clamp floating point values to [0.0, 1.0] range
                    # Handles precision issues like 1.0000000000000002
                    score = max(0.0, min(1.0, float(score)))

                self.logger.debug(f"Extracted score for {metric_name}: {score}")
            else:
                score = None
                self.logger.warning(
                    f"Metric {metric_name} not found in result keys: {result_dict.keys()}"
                )

            execution_time = time.time() - start_time

            # Estimate cost (approximate)
            cost_estimate = self.estimate_cost(metric_name, item, config)

            return self._create_result(
                metric_name=metric_name,
                item_id=item.id,
                score=score,
                execution_time=execution_time,
                cost_estimate=cost_estimate,
                evidence={"ragas_result": score},
            )

        except Exception as e:
            execution_time = time.time() - start_time

            if isinstance(e, (EvaluationError, MetricNotSupportedError)):
                raise

            # Detailed error logging
            error_msg = f"Ragas evaluation failed: {str(e)}"
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(f"Exception message: {str(e)}")
            self.logger.error(f"Exception repr: {repr(e)}")
            self.logger.error(error_msg, exc_info=True)

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

        # Metric-specific requirements based on Ragas documentation
        if metric_name in [
            "faithfulness",
            "context_precision",
            "context_recall",
            "context_entity_recall",
            "context_relevance",
            "context_utilization",
        ]:
            # Context-based metrics need question, answer, and context
            required_fields.extend(["input.question", "context"])
        elif metric_name in ["answer_relevancy", "response_relevancy"]:
            # Relevancy metrics need question and answer
            required_fields.append("input.question")
        elif metric_name in ["answer_correctness", "answer_similarity", "semantic_similarity"]:
            # Correctness/similarity metrics need reference answer
            # Check if any of the reference field variants exist
            has_reference = (
                self._get_field_value(item, "reference.answer")
                or self._get_field_value(item, "reference.expected")
                or self._get_field_value(item, "reference.ground_truth")
            )
            if not has_reference:
                raise EvaluationError(
                    "Metric requires reference answer but none found (checked reference.answer, reference.expected, reference.ground_truth)",
                    provider=self.provider_name,
                    metric=metric_name,
                    item_id=item.id,
                )
            required_fields.append("input.question")

        # Validate other required fields (not reference)
        if required_fields:
            self._validate_required_fields(item, required_fields)

        # Additional validation for context-based metrics
        if metric_name in [
            "faithfulness",
            "context_precision",
            "context_recall",
            "context_entity_recall",
            "context_relevance",
            "context_utilization",
        ]:
            if not item.context or len(item.context) == 0:
                raise EvaluationError(
                    f"Metric '{metric_name}' requires context but none provided",
                    provider=self.provider_name,
                    metric=metric_name,
                    item_id=item.id,
                )

    def _item_to_ragas_format(self, item: DatasetItem, metric_name: str) -> Dict[str, List]:
        """Convert dataset item to Ragas v0.3+ format.

        Args:
            item: Dataset item to convert
            metric_name: Metric being evaluated

        Returns:
            Dictionary in Ragas dataset format
        """
        # Ragas v0.3+ column names
        ragas_dict = {
            "user_input": [self._get_field_value(item, "input.question", "")],
            "response": [self._get_field_value(item, "output.answer", "")],
        }

        # Add context if available and needed
        context_metrics = [
            "faithfulness",
            "context_precision",
            "context_recall",
            "context_entity_recall",
            "context_relevance",
            "context_utilization",
        ]
        if item.context and metric_name in context_metrics:
            contexts = [ctx.text for ctx in item.context]
            ragas_dict["retrieved_contexts"] = [contexts]

        # Add ground truth/reference if available
        ground_truth = (
            self._get_field_value(item, "reference.ground_truth")
            or self._get_field_value(item, "reference.answer")
            or self._get_field_value(item, "reference.expected")
        )

        # Metrics that need reference answer
        reference_metrics = [
            "context_recall",
            "context_precision",
            "answer_correctness",
            "answer_similarity",
            "semantic_similarity",
        ]
        if ground_truth and metric_name in reference_metrics:
            ragas_dict["reference"] = [ground_truth]

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
            config: Configuration dictionary to validate

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
            logger.error("Ragas requires 'api_key' in configuration")
            return False

        api_key = config["api_key"]
        if not isinstance(api_key, str) or not api_key.strip():
            logger.error("Ragas API key must be a non-empty string")
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
                logger.warning(f"Model '{model}' may not be supported by Ragas")

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
            "context_relevancy": "context_precision",  # Ragas calls it context_precision
            # Note: context_recall not in builtin metrics yet
        },
    )
