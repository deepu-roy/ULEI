"""
Protocol definitions for provider adapters and core interfaces.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ulei.core.schemas import DatasetItem, EvaluationReport, EvaluationSuite, MetricResult


@runtime_checkable
class Metric(Protocol):
    """Protocol for individual metric implementations."""

    @property
    def name(self) -> str:
        """Return the metric name."""
        ...

    @property
    def supported_providers(self) -> List[str]:
        """Return list of providers that support this metric."""
        ...

    def validate_item(self, item: DatasetItem) -> bool:
        """Check if the dataset item has required fields for this metric."""
        ...


@runtime_checkable
class ProviderAdapter(Protocol):
    """Protocol for evaluation provider adapters."""

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        ...

    @property
    def supported_metrics(self) -> List[str]:
        """Return list of metrics this provider supports."""
        ...

    def supports_metric(self, metric_name: str) -> bool:
        """Check if this provider supports the given metric."""
        ...

    async def evaluate_metric(
        self, metric_name: str, item: DatasetItem, config: Dict[str, Any]
    ) -> MetricResult:
        """Evaluate a single metric for a dataset item."""
        ...

    def estimate_cost(
        self, metric_name: str, item: DatasetItem, config: Dict[str, Any]
    ) -> Optional[float]:
        """Estimate the cost for evaluating this metric."""
        ...

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate the provider configuration."""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for evaluation orchestrators."""

    def load_dataset(self, dataset_path: str) -> List[DatasetItem]:
        """Load dataset from file path."""
        ...

    async def run_evaluation(
        self, suite: EvaluationSuite, dataset: List[DatasetItem]
    ) -> EvaluationReport:
        """Run complete evaluation and return results."""
        ...

    def get_adapter(self, provider_name: str) -> Optional[ProviderAdapter]:
        """Get a registered provider adapter."""
        ...


@runtime_checkable
class Reporter(Protocol):
    """Protocol for report generators."""

    @property
    def format_name(self) -> str:
        """Return the report format name."""
        ...

    @property
    def file_extension(self) -> str:
        """Return the file extension for this format."""
        ...

    def generate_report(self, report: EvaluationReport, output_path: str) -> str:
        """Generate report and return the output file path."""
        ...


class BaseAdapter(ABC):
    """Abstract base class for provider adapters."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the adapter with configuration."""
        self.config = config
        self._validate_config(config)

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass

    @property
    @abstractmethod
    def supported_metrics(self) -> List[str]:
        """Return list of metrics this provider supports."""
        pass

    def supports_metric(self, metric_name: str) -> bool:
        """Check if this provider supports the given metric."""
        return metric_name in self.supported_metrics

    @abstractmethod
    async def evaluate_metric(
        self, metric_name: str, item: DatasetItem, config: Dict[str, Any]
    ) -> MetricResult:
        """Evaluate a single metric for a dataset item."""
        pass

    def estimate_cost(
        self, metric_name: str, item: DatasetItem, config: Dict[str, Any]
    ) -> Optional[float]:
        """Estimate the cost for evaluating this metric.

        Default implementation returns None (no cost estimation).
        Override in subclasses to provide cost estimates.
        """
        return None

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate the provider configuration.

        Default implementation always returns True.
        Override in subclasses for specific validation.
        """
        return True

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Internal configuration validation."""
        if not self.validate_config(config):
            raise ValueError(f"Invalid configuration for {self.provider_name}")

    def _create_result(
        self,
        metric_name: str,
        item_id: str,
        score: Optional[float] = None,
        confidence: Optional[float] = None,
        evidence: Optional[Dict[str, Any]] = None,
        execution_time: float = 0.0,
        cost_estimate: Optional[float] = None,
        error: Optional[str] = None,
        raw_response: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """Helper method to create MetricResult instances."""
        return MetricResult(
            metric=metric_name,
            provider=self.provider_name,
            item_id=item_id,
            score=score,
            confidence=confidence,
            evidence=evidence or {},
            execution_time=execution_time,
            cost_estimate=cost_estimate,
            error=error,
            raw_response=raw_response,
        )


class BaseReporter(ABC):
    """Abstract base class for report generators."""

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the report format name."""
        pass

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Return the file extension for this format."""
        pass

    @abstractmethod
    def generate_report(self, report: EvaluationReport, output_path: str) -> str:
        """Generate report and return the output file path."""
        pass

    def _ensure_extension(self, output_path: str) -> str:
        """Ensure the output path has the correct extension."""
        if not output_path.endswith(self.file_extension):
            return f"{output_path}{self.file_extension}"
        return output_path


# Type aliases for convenience
AdapterType = ProviderAdapter
ReporterType = Reporter
EvaluatorType = Evaluator
