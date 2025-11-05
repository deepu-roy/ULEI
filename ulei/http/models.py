"""
Pydantic models for HTTP API request/response schemas.

Implements the data models defined in contracts/http-api.yaml
for online evaluation data ingestion.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ContextItem(BaseModel):
    """Context item for RAG evaluation."""

    text: str = Field(..., description="Context content")
    source_id: Optional[str] = Field(None, description="Source document identifier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class EvaluationEvent(BaseModel):
    """Evaluation event for online shadow evaluation."""

    event_id: Optional[str] = Field(None, description="Unique identifier for this evaluation event")
    suite_name: Optional[str] = Field(
        "production_monitoring", description="Name of evaluation suite to apply"
    )

    input: Dict[str, Any] = Field(..., description="Original input to the model/system")
    output: Dict[str, Any] = Field(..., description="Actual system output to evaluate")
    reference: Optional[Dict[str, Any]] = Field(None, description="Expected/ground truth output")

    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional context and tags"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "evt_abc123",
                "suite_name": "production_rag",
                "input": {
                    "query": "What is the capital of France?",
                    "context": [
                        {
                            "text": "Paris is the capital and most populous city of France.",
                            "source_id": "doc_123",
                        }
                    ],
                },
                "output": {
                    "answer": "Paris is the capital of France.",
                    "citations": ["doc_123"],
                    "latency_ms": 250,
                },
                "reference": {"answer": "Paris"},
                "metadata": {
                    "user_id": "user_456",
                    "session_id": "session_789",
                    "timestamp": "2024-11-05T10:00:00Z",
                    "tags": ["geography", "factual"],
                },
            }
        }


class BatchIngestRequest(BaseModel):
    """Request for batch ingestion of evaluation events."""

    events: List[EvaluationEvent] = Field(
        ..., max_items=100, description="List of evaluation events to ingest"
    )


class BatchIngestResponse(BaseModel):
    """Response for batch ingestion with acceptance/rejection counts."""

    batch_id: str = Field(..., description="Unique identifier for this batch")
    accepted_count: int = Field(..., description="Number of events accepted")
    rejected_count: int = Field(..., description="Number of events rejected")
    rejected_events: List[Dict[str, Any]] = Field(
        default_factory=list, description="Details of rejected events"
    )


class EventStatus(BaseModel):
    """Status tracking for an evaluation event."""

    event_id: str = Field(..., description="Event identifier")
    status: str = Field(..., description="Processing status")
    queued_at: Optional[datetime] = Field(None, description="When event was queued")
    started_at: Optional[datetime] = Field(None, description="When processing started")
    completed_at: Optional[datetime] = Field(None, description="When processing completed")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    results_available: bool = Field(False, description="Whether results are available")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(..., description="Current timestamp")
    queue_size: Optional[int] = Field(None, description="Current queue size")
    processing_active: Optional[bool] = Field(
        None, description="Whether background processing is active"
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class MetricResult(BaseModel):
    """Result from evaluating a single metric."""

    metric: str = Field(..., description="Name of the evaluated metric")
    provider: str = Field(..., description="Provider that evaluated the metric")
    item_id: str = Field(..., description="ID of the evaluated item")
    score: Optional[float] = Field(None, ge=0, le=1, description="Metric score (0-1)")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence in the score")
    evidence: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Supporting evidence"
    )
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    cost_estimate: Optional[float] = Field(None, description="Estimated cost")
    error: Optional[str] = Field(None, description="Error message if evaluation failed")


class EvaluationReportResponse(BaseModel):
    """Evaluation report response for HTTP API."""

    run_id: str = Field(..., description="Unique run identifier")
    suite_name: str = Field(..., description="Name of the evaluation suite")

    dataset_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Statistics about the evaluated dataset"
    )

    results: List[MetricResult] = Field(
        default_factory=list, description="Individual metric evaluation results"
    )

    aggregates: Dict[str, float] = Field(
        default_factory=dict, description="Aggregated metric values"
    )

    threshold_status: Dict[str, bool] = Field(
        default_factory=dict, description="Threshold compliance status per metric"
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Report creation timestamp"
    )


class WindowedEvaluationConfig(BaseModel):
    """Configuration for windowed evaluation aggregation."""

    window_size_minutes: int = Field(60, description="Size of evaluation window in minutes")
    overlap_minutes: int = Field(10, description="Overlap between windows in minutes")
    min_samples_per_window: int = Field(10, description="Minimum samples required per window")
    max_samples_per_window: int = Field(1000, description="Maximum samples per window")
    aggregation_strategy: str = Field("mean", description="Aggregation strategy for metrics")


class AlertConfig(BaseModel):
    """Configuration for threshold-based alerting."""

    enabled: bool = Field(True, description="Whether alerting is enabled")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for alerts")
    threshold_violations_required: int = Field(
        3, description="Consecutive violations to trigger alert"
    )
    cooldown_minutes: int = Field(30, description="Cooldown period between alerts")
    alert_on_regression: bool = Field(True, description="Alert on performance regressions")
    alert_on_threshold_breach: bool = Field(True, description="Alert on threshold breaches")


class ServerConfig(BaseModel):
    """Server configuration model."""

    host: str = Field("0.0.0.0", description="Host to bind to")
    port: int = Field(8000, description="Port to bind to")
    workers: int = Field(1, description="Number of worker processes")

    queue_max_size: int = Field(10000, description="Maximum queue size")
    batch_size: int = Field(50, description="Batch size for processing")
    batch_timeout_seconds: int = Field(30, description="Timeout for batch processing")

    windowed_evaluation: WindowedEvaluationConfig = Field(
        default_factory=WindowedEvaluationConfig, description="Windowed evaluation configuration"
    )

    alerting: AlertConfig = Field(default_factory=AlertConfig, description="Alerting configuration")

    rate_limit_events_per_minute: int = Field(1000, description="Rate limit for event ingestion")

    # Storage configuration
    reports_retention_days: int = Field(30, description="Report retention period in days")
    event_status_retention_days: int = Field(7, description="Event status retention period")


class PrometheusMetrics(BaseModel):
    """Prometheus metrics model."""

    name: str = Field(..., description="Metric name")
    type: str = Field(..., description="Metric type (counter, gauge, histogram)")
    help: str = Field(..., description="Metric description")
    labels: Optional[Dict[str, str]] = Field(default_factory=dict, description="Metric labels")
    value: Union[int, float] = Field(..., description="Metric value")
