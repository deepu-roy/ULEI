"""
Core evaluation orchestrator with batch processing capabilities.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from ulei.core.registry import get_registry
from ulei.core.schemas import (
    CostSummary,
    DatasetItem,
    EvaluationReport,
    EvaluationRun,
    EvaluationSuite,
    ExecutionMetadata,
    MetricResult,
    MetricSpec,
    RunStatus,
)
from ulei.utils.errors import BudgetExceededError, ConfigurationError, DatasetError, EvaluationError

logger = logging.getLogger(__name__)


class Evaluator:
    """Core evaluation orchestrator implementing batch processing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize evaluator with configuration and budget manager.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.registry = get_registry()
        self._setup_logging()
        # Budget manager for cost control
        from ulei.utils.budget import BudgetManager

        self.budget_manager = BudgetManager()

    def _setup_logging(self) -> None:
        """Setup evaluator-specific logging."""
        self.logger = logging.getLogger("ulei.core.evaluator")

    def load_dataset(self, dataset_path: str) -> List[DatasetItem]:
        """Load dataset from file path.

        Args:
            dataset_path: Path to dataset file (JSONL or CSV)

        Returns:
            List of DatasetItem instances

        Raises:
            DatasetError: If dataset loading fails
        """
        try:
            path = Path(dataset_path)
            if not path.exists():
                raise DatasetError(f"Dataset file not found: {path}")

            if path.suffix.lower() == ".jsonl":
                return self._load_jsonl_dataset(path)
            elif path.suffix.lower() == ".csv":
                return self._load_csv_dataset(path)
            else:
                raise DatasetError(
                    f"Unsupported dataset format: {path.suffix}. Supported formats: .jsonl, .csv"
                )

        except Exception as e:
            if isinstance(e, DatasetError):
                raise
            raise DatasetError(f"Failed to load dataset: {e}", str(dataset_path)) from e

    def _load_jsonl_dataset(self, dataset_path: Path) -> List[DatasetItem]:
        """Load dataset from JSONL file.

        Args:
            dataset_path: Path to JSONL file

        Returns:
            List of DatasetItem instances
        """
        import json

        items = []
        with open(dataset_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    item = DatasetItem(**data)
                    items.append(item)
                except (json.JSONDecodeError, ValueError) as e:
                    raise DatasetError(
                        f"Invalid JSON on line {line_num}: {e}", str(dataset_path), line_num
                    ) from e

        self.logger.info(f"Loaded {len(items)} items from {dataset_path}")
        return items

    def _load_csv_dataset(self, dataset_path: Path) -> List[DatasetItem]:
        """Load dataset from CSV file.

        Args:
            dataset_path: Path to CSV file

        Returns:
            List of DatasetItem instances
        """
        import pandas as pd

        try:
            df = pd.read_csv(dataset_path)

            # Validate required columns
            required_cols = ["id", "input", "output"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise DatasetError(f"Missing required columns: {missing_cols}", str(dataset_path))

            items = []
            for idx, row in df.iterrows():
                try:
                    # Parse JSON fields if they're strings
                    input_data = self._parse_json_field(row["input"])
                    output_data = self._parse_json_field(row["output"])
                    reference_data = (
                        self._parse_json_field(row.get("reference")) if "reference" in row else None
                    )

                    item = DatasetItem(
                        id=str(row["id"]),
                        input=input_data,
                        output=output_data,
                        reference=reference_data,
                        metadata=row.get("metadata", {}),
                    )
                    items.append(item)

                except Exception as e:
                    # Cast idx to int since pandas iterrows() returns Hashable index
                    row_num = int(idx) + 2 if isinstance(idx, (int, float)) else len(items) + 2
                    raise DatasetError(
                        f"Invalid data on row {row_num}: {e}",
                        str(dataset_path),
                        row_num,
                    ) from e

            self.logger.info(f"Loaded {len(items)} items from {dataset_path}")
            return items

        except pd.errors.EmptyDataError:
            raise DatasetError("CSV file is empty", str(dataset_path))
        except Exception as e:
            if isinstance(e, DatasetError):
                raise
            raise DatasetError(f"Failed to parse CSV: {e}", str(dataset_path)) from e

    def _parse_json_field(self, field_value: Any) -> Any:
        """Parse JSON field value if it's a string, otherwise return as-is.

        Args:
            field_value: Field value to parse

        Returns:
            Parsed value or original value
        """
        if isinstance(field_value, str):
            try:
                import json

                return json.loads(field_value)
            except json.JSONDecodeError:
                # If it's not valid JSON, treat as plain string
                return field_value
        return field_value

    async def run_evaluation(
        self, suite: EvaluationSuite, dataset: List[DatasetItem], run_id: Optional[str] = None
    ) -> EvaluationReport:
        """Run complete evaluation and return results.

        Args:
            suite: Evaluation suite configuration
            dataset: Dataset items to evaluate
            run_id: Optional run ID (generated if None)

        Returns:
            EvaluationReport with results
        """
        if not run_id:
            run_id = f"run_{uuid.uuid4().hex[:8]}"

        # Create evaluation run
        run = EvaluationRun(run_id=run_id, suite=suite, dataset=dataset, status=RunStatus.PENDING)

        try:
            # Validate configuration
            self._validate_evaluation_setup(suite, dataset)

            # Start execution
            run.status = RunStatus.RUNNING
            run.started_at = datetime.utcnow()

            self.logger.info(
                f"Starting evaluation {run_id} with {len(dataset)} items "
                f"and {len(suite.metrics)} metrics"
            )

            # Execute evaluation
            results = await self._execute_evaluation(suite, dataset)

            # Calculate aggregates
            aggregates = self._calculate_aggregates(results, suite.metrics)

            # Check thresholds
            threshold_status = self._check_thresholds(aggregates, suite.thresholds)

            # Calculate costs
            cost_summary = self._calculate_cost_summary(results)

            # Create execution metadata
            run.completed_at = datetime.utcnow()
            execution_time = (run.completed_at - run.started_at).total_seconds()

            execution_metadata = ExecutionMetadata(
                total_items=len(dataset),
                successful_evaluations=len([r for r in results if r.error is None]),
                failed_evaluations=len([r for r in results if r.error is not None]),
                total_execution_time=execution_time,
                provider_usage=self._calculate_provider_usage(results),
            )

            # Create final report
            report = EvaluationReport(
                run_id=run_id,
                suite_name=suite.name,
                dataset_stats=self._calculate_dataset_stats(dataset),
                results=results,
                aggregates=aggregates,
                threshold_status=threshold_status,
                execution_metadata=execution_metadata,
                cost_summary=cost_summary,
            )

            run.status = RunStatus.COMPLETE
            run.progress = 1.0

            self.logger.info(f"Evaluation {run_id} completed successfully")
            return report

        except Exception as e:
            run.status = RunStatus.FAILED
            run.completed_at = datetime.utcnow()

            self.logger.error(f"Evaluation {run_id} failed: {e}")

            if isinstance(e, (EvaluationError, BudgetExceededError, ConfigurationError)):
                raise

            raise EvaluationError(f"Evaluation failed: {e}") from e

    def _validate_evaluation_setup(
        self, suite: EvaluationSuite, dataset: List[DatasetItem]
    ) -> None:
        """Validate evaluation setup before execution.

        Args:
            suite: Evaluation suite
            dataset: Dataset items

        Raises:
            ConfigurationError: If setup is invalid
        """
        if not dataset:
            raise ConfigurationError("Dataset cannot be empty")

        if not suite.metrics:
            raise ConfigurationError("Suite must have at least one metric")

        # Validate that providers exist for metrics
        for metric in suite.metrics:
            provider_name = self.registry.resolve_provider_for_metric(
                metric.name, suite.provider_priority
            )
            if not provider_name:
                available_providers = self.registry.get_providers_for_metric(metric.name)
                raise ConfigurationError(
                    f"No provider found for metric '{metric.name}'. "
                    f"Available providers: {available_providers}"
                )

    async def _execute_evaluation(
        self, suite: EvaluationSuite, dataset: List[DatasetItem]
    ) -> List[MetricResult]:
        """Execute evaluation for all metrics and dataset items.

        Args:
            suite: Evaluation suite
            dataset: Dataset items

        Returns:
            List of MetricResult instances
        """
        all_results = []
        self.budget_manager.reset()

        # Create semaphore for concurrency control
        worker_count = getattr(suite, "parallel_workers", 4)
        self.logger.info(f"Parallel processing enabled: {worker_count} workers")
        semaphore = asyncio.Semaphore(worker_count)

        # Create tasks for all metric-item combinations
        tasks = []
        for metric in suite.metrics:
            for item in dataset:
                self.logger.debug(f"Scheduling evaluation: metric={metric.name}, item={item.id}")
                task = self._evaluate_single_metric_item(
                    semaphore,
                    suite,
                    metric,
                    item,
                    self.budget_manager.current_cost,
                    suite.budget_limit,
                )
                tasks.append(task)
        self.logger.info(f"Total evaluation tasks scheduled: {len(tasks)}")

        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Task failed: {result}")
                continue

            # At this point, result is MetricResult (type narrowing)
            metric_result = cast(MetricResult, result)
            all_results.append(metric_result)
            # Update cost tracking and enforce budget
            if metric_result.cost_estimate:
                try:
                    self.budget_manager.add_cost(metric_result.cost_estimate)
                    self.logger.debug(f"Cost after task: {self.budget_manager.current_cost:.4f}")
                except Exception as e:
                    self.logger.error(f"Budget enforcement error: {e}")
        self.logger.info(
            f"All evaluation tasks complete. Total cost: {self.budget_manager.current_cost:.2f}"
        )
        return all_results

    async def _evaluate_single_metric_item(
        self,
        semaphore: asyncio.Semaphore,
        suite: EvaluationSuite,
        metric_spec: Any,
        item: DatasetItem,
        current_cost: float,
        budget_limit: Optional[float] = None,
    ) -> MetricResult:
        """Evaluate a single metric for a single item.

        Args:
            semaphore: Concurrency control semaphore
            suite: Evaluation suite
            metric_spec: Metric specification
            item: Dataset item
            current_cost: Current accumulated cost

        Returns:
            MetricResult for this evaluation
        """
        async with semaphore:
            try:
                # Resolve provider
                provider_name = metric_spec.provider or self.registry.resolve_provider_for_metric(
                    metric_spec.name, suite.provider_priority
                )

                if not provider_name:
                    raise EvaluationError(
                        f"No provider available for metric '{metric_spec.name}'",
                        metric=metric_spec.name,
                        item_id=item.id,
                    )

                # Get adapter
                provider_config = suite.providers.get(provider_name, {})
                adapter = self.registry.get_adapter(provider_name, provider_config)

                # Check budget before evaluation
                if budget_limit:
                    estimated_cost = adapter.estimate_cost(
                        metric_spec.name, item, metric_spec.config
                    )
                    if estimated_cost and (current_cost + estimated_cost) > budget_limit:
                        from ulei.utils.budget import BudgetExceededError

                        raise BudgetExceededError(
                            "Evaluation would exceed budget limit",
                            budget_limit=budget_limit,
                            current_cost=current_cost + estimated_cost,
                        )

                # Evaluate with retry
                if hasattr(adapter, "evaluate_metric_with_retry"):
                    # mypy/pylance can't see dynamic attributes checked with hasattr
                    result = await adapter.evaluate_metric_with_retry(  # type: ignore
                        metric_spec.name,
                        item,
                        metric_spec.config,
                        max_retries=suite.retry_policy.max_retries,
                        backoff_factor=suite.retry_policy.backoff_factor,
                    )
                else:
                    result = await adapter.evaluate_metric(
                        metric_spec.name, item, metric_spec.config
                    )

                return result

            except Exception as e:
                self.logger.error(f"Failed to evaluate {metric_spec.name} for item {item.id}: {e}")

                # Create error result
                # Use the provider_name if it was successfully determined, otherwise use "unknown"
                error_provider = "unknown"
                if "provider_name" in locals() and provider_name is not None:
                    error_provider = provider_name

                return MetricResult(
                    metric=metric_spec.name,
                    provider=error_provider,
                    item_id=item.id,
                    score=None,
                    confidence=None,
                    error=str(e),
                    execution_time=0.0,
                    cost_estimate=None,
                )

    def get_adapter(self, provider_name: str) -> Optional[Any]:
        """Get a registered provider adapter.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider adapter or None if not found
        """
        try:
            return self.registry.get_adapter(provider_name)
        except Exception:
            return None

    def _calculate_aggregates(
        self, results: List[MetricResult], metrics: List[MetricSpec]
    ) -> Dict[str, float]:
        """Calculate aggregate statistics for metrics.

        Args:
            results: List of metric results
            metrics: Metric specifications

        Returns:
            Dictionary with aggregate values per metric
        """
        aggregates = {}

        # Group results by metric
        metric_results: Dict[str, List[float]] = {}
        for result in results:
            if result.error is None and result.score is not None:
                if result.metric not in metric_results:
                    metric_results[result.metric] = []
                metric_results[result.metric].append(result.score)

        # Calculate mean for each metric
        for metric_name, scores in metric_results.items():
            if scores:
                aggregates[metric_name] = sum(scores) / len(scores)
            else:
                aggregates[metric_name] = 0.0

        return aggregates

    def _check_thresholds(
        self, aggregates: Dict[str, float], thresholds: Dict[str, float]
    ) -> Dict[str, bool]:
        """Check if aggregated metrics meet thresholds.

        Args:
            aggregates: Aggregated metric values
            thresholds: Threshold values per metric

        Returns:
            Dictionary with pass/fail status per threshold
        """
        threshold_status = {}

        for metric_name, threshold in thresholds.items():
            aggregate_value = aggregates.get(metric_name, 0.0)
            threshold_status[metric_name] = aggregate_value >= threshold
            self.logger.info(
                f"Threshold check: metric={metric_name}, value={aggregate_value:.3f}, threshold={threshold:.3f}, pass={threshold_status[metric_name]}"
            )
        return threshold_status

    def _calculate_cost_summary(self, results: List[MetricResult]) -> Optional[CostSummary]:
        """Calculate cost summary from results.

        Args:
            results: List of metric results

        Returns:
            CostSummary or None if no cost data available
        """
        costs = [r.cost_estimate for r in results if r.cost_estimate is not None]
        if not costs:
            return None

        total_cost = sum(costs)

        # Group by provider
        cost_by_provider: Dict[str, float] = {}
        for result in results:
            if result.cost_estimate:
                provider = result.provider
                cost_by_provider[provider] = (
                    cost_by_provider.get(provider, 0.0) + result.cost_estimate
                )

        # Group by metric
        cost_by_metric: Dict[str, float] = {}
        for result in results:
            if result.cost_estimate:
                metric = result.metric
                cost_by_metric[metric] = cost_by_metric.get(metric, 0.0) + result.cost_estimate

        return CostSummary(
            total_estimated_cost=total_cost,
            cost_by_provider=cost_by_provider,
            cost_by_metric=cost_by_metric,
            budget_utilization=1.0,  # Will be calculated against actual budget in calling code
        )

    def _calculate_provider_usage(self, results: List[MetricResult]) -> Dict[str, int]:
        """Calculate usage statistics per provider.

        Args:
            results: List of metric results

        Returns:
            Dictionary with usage count per provider
        """
        usage: Dict[str, int] = {}
        for result in results:
            provider = result.provider
            usage[provider] = usage.get(provider, 0) + 1
        return usage

    def _calculate_dataset_stats(self, dataset: List[DatasetItem]) -> Dict[str, Any]:
        """Calculate statistics about the dataset.

        Args:
            dataset: List of dataset items

        Returns:
            Dictionary with dataset statistics
        """
        return {
            "total_items": len(dataset),
            "items_with_reference": len([item for item in dataset if item.reference]),
            "items_with_context": len([item for item in dataset if item.context]),
            "unique_ids": len({item.id for item in dataset}),
        }

    def check_thresholds(
        self, results: List[MetricResult], thresholds: Dict[str, float], strategy: str = "mean"
    ) -> Dict[str, bool]:
        """
        Check if evaluation results meet configured thresholds.

        Args:
            results: List of metric results
            thresholds: Dictionary mapping metric names to threshold values
            strategy: Aggregation strategy ('mean', 'median', 'min', 'all')

        Returns:
            Dictionary mapping metric names to pass/fail status
        """
        if not thresholds:
            return {}

        threshold_status = {}

        # Group results by metric
        results_by_metric = self._group_results_by_metric(results)

        for metric_name, threshold_value in thresholds.items():
            if metric_name not in results_by_metric:
                self.logger.warning(f"No results found for threshold metric: {metric_name}")
                threshold_status[metric_name] = False
                continue

            metric_results = results_by_metric[metric_name]
            scores = [r.score for r in metric_results if r.score is not None and r.error is None]

            if not scores:
                self.logger.warning(f"No valid scores found for threshold metric: {metric_name}")
                threshold_status[metric_name] = False
                continue

            # Calculate aggregate score based on strategy
            if strategy == "mean":
                aggregate_score = sum(scores) / len(scores)
            elif strategy == "median":
                sorted_scores = sorted(scores)
                n = len(sorted_scores)
                aggregate_score = (
                    sorted_scores[n // 2]
                    if n % 2 == 1
                    else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
                )
            elif strategy == "min":
                aggregate_score = min(scores)
            elif strategy == "all":
                # All scores must meet threshold
                threshold_status[metric_name] = all(score >= threshold_value for score in scores)
                continue
            else:
                raise ValueError(f"Unknown threshold strategy: {strategy}")

            # Check if aggregate meets threshold
            threshold_status[metric_name] = aggregate_score >= threshold_value

            self.logger.info(
                f"Threshold check for {metric_name}: {aggregate_score:.3f} >= {threshold_value} = {threshold_status[metric_name]}"
            )

        return threshold_status

    def _group_results_by_metric(
        self, results: List[MetricResult]
    ) -> Dict[str, List[MetricResult]]:
        """Group results by metric name."""
        grouped: Dict[str, List[MetricResult]] = {}
        for result in results:
            metric_name = result.metric
            if metric_name not in grouped:
                grouped[metric_name] = []
            grouped[metric_name].append(result)
        return grouped

    def get_overall_status(
        self, threshold_status: Dict[str, bool], required_metrics: Optional[List[str]] = None
    ) -> bool:
        """
        Determine overall evaluation status based on threshold results.

        Args:
            threshold_status: Dictionary of threshold check results
            required_metrics: List of metrics that must pass (if None, all must pass)

        Returns:
            True if evaluation passes, False otherwise
        """
        if not threshold_status:
            return True  # No thresholds = always pass

        if required_metrics:
            # Only check required metrics
            for metric in required_metrics:
                if metric in threshold_status and not threshold_status[metric]:
                    return False
            return True
        else:
            # All thresholds must pass
            return all(threshold_status.values())

    def calculate_failure_reasons(
        self,
        threshold_status: Dict[str, bool],
        thresholds: Dict[str, float],
        results: List[MetricResult],
    ) -> List[str]:
        """
        Generate human-readable failure reasons for CI/CD reporting.

        Args:
            threshold_status: Threshold check results
            thresholds: Configured thresholds
            results: Evaluation results

        Returns:
            List of failure reason strings
        """
        failures = []
        results_by_metric = self._group_results_by_metric(results)

        for metric, passed in threshold_status.items():
            if not passed:
                threshold = thresholds.get(metric, 0.0)
                metric_results = results_by_metric.get(metric, [])
                scores = [
                    r.score for r in metric_results if r.score is not None and r.error is None
                ]

                if scores:
                    actual_score = sum(scores) / len(scores)
                    failures.append(
                        f"{metric}: {actual_score:.3f} < {threshold:.3f} (failed by {threshold - actual_score:.3f})"
                    )
                else:
                    failures.append(f"{metric}: No valid scores available")

        return failures
