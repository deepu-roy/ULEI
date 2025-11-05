"""
Core Pydantic schemas for ULEI data structures - simplified without complex validators.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(str, Enum):
    """Execution status for evaluation runs."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RetryPolicy(BaseModel):
    """Configuration for handling flaky evaluations."""

    max_retries: int = Field(default=3, ge=0, le=10)
    backoff_factor: float = Field(default=1.0, ge=0.1, le=10.0)
    timeout: float = Field(default=60.0, ge=1.0, le=300.0)


class MetricSpec(BaseModel):
    """Specification for a metric to evaluate."""

    name: str = Field(..., min_length=1, max_length=100)
    provider: Optional[str] = Field(None, max_length=50)
    config: Dict[str, Any] = Field(default_factory=dict)


class ContextItem(BaseModel):
    """Retrieved context item for RAG evaluation."""

    model_config = ConfigDict(populate_by_name=True)

    text: str = Field(..., min_length=1)
    source_id: Optional[str] = Field(None, max_length=200, alias="source")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatasetItem(BaseModel):
    """Individual test case for evaluation."""

    id: str = Field(..., min_length=1, max_length=200)
    input: Dict[str, Any] = Field(...)
    reference: Optional[Dict[str, Any]] = None
    context: Optional[List[ContextItem]] = None
    output: Dict[str, Any] = Field(...)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricResult(BaseModel):
    """Standardized output from metric evaluation."""

    metric: str = Field(..., min_length=1, max_length=100)
    provider: str = Field(..., min_length=1, max_length=50)
    item_id: str = Field(..., min_length=1, max_length=200)
    score: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float = Field(..., ge=0.0)
    cost_estimate: Optional[float] = Field(None, ge=0.0)
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CostSummary(BaseModel):
    """Cost breakdown for an evaluation run."""

    total_estimated_cost: float = Field(..., ge=0.0)
    cost_by_provider: Dict[str, float] = Field(default_factory=dict)
    cost_by_metric: Dict[str, float] = Field(default_factory=dict)
    budget_utilization: float = Field(..., ge=0.0, le=1.0)


class ExecutionMetadata(BaseModel):
    """Runtime information for an evaluation run."""

    total_items: int = Field(..., ge=0)
    successful_evaluations: int = Field(..., ge=0)
    failed_evaluations: int = Field(..., ge=0)
    cache_hits: int = Field(default=0, ge=0)
    total_execution_time: float = Field(..., ge=0.0)
    provider_usage: Dict[str, int] = Field(default_factory=dict)


class EvaluationSuite(BaseModel):
    """Configuration object that defines a complete evaluation setup."""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)

    # Dataset specification - either items directly or dataset source
    items: Optional[List[DatasetItem]] = Field(None)
    dataset: Optional[Dict[str, Any]] = Field(None)

    metrics: List[MetricSpec] = Field(...)
    providers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    provider_priority: List[str] = Field(default_factory=list)
    thresholds: Dict[str, float] = Field(default_factory=dict)
    budget_limit: Optional[float] = Field(None, gt=0.0)
    output_formats: List[str] = Field(default=["json"])
    parallel_workers: int = Field(default=4, ge=1, le=20)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    output_dir: str = Field(default="./evaluation_results")
    cache_enabled: bool = Field(default=True)

    def get_items(self) -> List[DatasetItem]:
        """
        Get dataset items, loading from source if necessary.

        Returns:
            List of DatasetItem objects

        Raises:
            ConfigurationError: If dataset cannot be loaded
        """
        if self.items is not None:
            return self.items
        elif self.dataset is not None:
            from ulei.utils.dataset import DatasetLoader

            # Extract dataset configuration
            source = self.dataset.get("source")
            if not source:
                raise ValueError("Dataset configuration must include 'source' field")

            format_hint = self.dataset.get("format")
            loader_kwargs = self.dataset.get("options", {})

            return DatasetLoader.load_dataset(source, format_hint, **loader_kwargs)
        else:
            raise ValueError("No dataset items or source specified")


class EvaluationReport(BaseModel):
    """Comprehensive results from an evaluation run."""

    run_id: str = Field(..., min_length=1, max_length=100)
    suite_name: str = Field(..., min_length=1, max_length=100)
    dataset_stats: Dict[str, Any] = Field(default_factory=dict)
    results: List[MetricResult] = Field(default_factory=list)
    aggregates: Dict[str, float] = Field(default_factory=dict)
    threshold_status: Dict[str, bool] = Field(default_factory=dict)
    execution_metadata: ExecutionMetadata
    cost_summary: Optional[CostSummary] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvaluationRun(BaseModel):
    """Execution context for a specific evaluation."""

    run_id: str = Field(..., min_length=1, max_length=100)
    suite: EvaluationSuite
    dataset: List[DatasetItem] = Field(...)
    status: RunStatus = Field(default=RunStatus.PENDING)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# Provider-specific types for extension
class ProviderConfig(BaseModel):
    """Base configuration for evaluation providers."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    model: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)


class CostModel(BaseModel):
    """Pricing information for budget control."""

    cost_per_request: Optional[float] = Field(None, ge=0.0)
    cost_per_token: Optional[float] = Field(None, ge=0.0)
    currency: str = Field(default="USD")
