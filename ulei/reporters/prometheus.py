"""
Prometheus metrics exporter for ULEI evaluation monitoring.

Exports evaluation metrics, system performance, and queue statistics
in Prometheus format for monitoring and alerting.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

from ulei.core.schemas import EvaluationReport
from ulei.reporters.base import BaseReporter

logger = logging.getLogger(__name__)


class PrometheusMetric:
    """Represents a single Prometheus metric."""

    def __init__(
        self,
        name: str,
        metric_type: str,
        help_text: str,
        labels: Optional[Dict[str, str]] = None,
        value: Union[int, float] = 0,
    ):
        self.name = name
        self.metric_type = metric_type
        self.help_text = help_text
        self.labels = labels or {}
        self.value = value
        self.timestamp = time.time()

    def format_prometheus(self) -> str:
        """Format metric in Prometheus exposition format."""
        lines = []

        # Add HELP line
        lines.append(f"# HELP {self.name} {self.help_text}")

        # Add TYPE line
        lines.append(f"# TYPE {self.name} {self.metric_type}")

        # Format labels
        if self.labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(self.labels.items()))
            metric_line = f"{self.name}{{{label_str}}} {self.value}"
        else:
            metric_line = f"{self.name} {self.value}"

        # Add timestamp if available
        if self.timestamp:
            metric_line += f" {int(self.timestamp * 1000)}"

        lines.append(metric_line)

        return "\n".join(lines)


class PrometheusExporter(BaseReporter):
    """Prometheus metrics exporter for ULEI evaluation data."""

    def __init__(self, metric_prefix: str = "ulei"):
        """
        Initialize Prometheus exporter.

        Args:
            metric_prefix: Prefix for all exported metrics
        """
        self.metric_prefix = metric_prefix
        self.metrics: Dict[str, PrometheusMetric] = {}

        # Initialize core metrics
        self._initialize_core_metrics()

    def _initialize_core_metrics(self):
        """Initialize core system metrics."""

        # Evaluation metrics
        self.add_metric(
            "evaluations_total",
            "counter",
            "Total number of evaluations processed",
            {"status": "completed"},
        )

        self.add_metric(
            "evaluations_total",
            "counter",
            "Total number of evaluations processed",
            {"status": "failed"},
        )

        # Performance metrics
        self.add_metric(
            "evaluation_duration_seconds", "histogram", "Time spent on evaluation processing"
        )

        self.add_metric("queue_size", "gauge", "Current size of the evaluation queue")

        self.add_metric(
            "queue_processing_rate", "gauge", "Rate of queue processing (events/second)"
        )

        # Metric score gauges
        self.add_metric(
            "metric_score_current",
            "gauge",
            "Current metric score value",
            {"metric": "faithfulness", "suite": "production"},
        )

        # System health
        self.add_metric("server_up", "gauge", "Server uptime indicator (1 = up, 0 = down)")

        # Cost tracking
        self.add_metric("evaluation_cost_total", "counter", "Total estimated cost of evaluations")

    def add_metric(
        self,
        name: str,
        metric_type: str,
        help_text: str,
        labels: Optional[Dict[str, str]] = None,
        value: Union[int, float] = 0,
    ):
        """
        Add or update a metric.

        Args:
            name: Metric name (without prefix)
            metric_type: Prometheus metric type (counter, gauge, histogram, summary)
            help_text: Help description
            labels: Metric labels
            value: Metric value
        """
        full_name = f"{self.metric_prefix}_{name}"
        metric_key = self._get_metric_key(full_name, labels)

        self.metrics[metric_key] = PrometheusMetric(
            name=full_name, metric_type=metric_type, help_text=help_text, labels=labels, value=value
        )

    def update_metric(
        self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None
    ):
        """
        Update an existing metric value.

        Args:
            name: Metric name (without prefix)
            value: New metric value
            labels: Metric labels
        """
        full_name = f"{self.metric_prefix}_{name}"
        metric_key = self._get_metric_key(full_name, labels)

        if metric_key in self.metrics:
            self.metrics[metric_key].value = value
            self.metrics[metric_key].timestamp = time.time()
        else:
            logger.warning(f"Metric {metric_key} not found for update")

    def increment_counter(
        self, name: str, increment: Union[int, float] = 1, labels: Optional[Dict[str, str]] = None
    ):
        """
        Increment a counter metric.

        Args:
            name: Counter name (without prefix)
            increment: Amount to increment by
            labels: Metric labels
        """
        full_name = f"{self.metric_prefix}_{name}"
        metric_key = self._get_metric_key(full_name, labels)

        if metric_key in self.metrics:
            self.metrics[metric_key].value += increment
            self.metrics[metric_key].timestamp = time.time()
        else:
            self.add_metric(
                name, "counter", f"Auto-generated counter for {name}", labels, increment
            )

    def export_evaluation_report(self, report: EvaluationReport):
        """
        Export metrics from an evaluation report.

        Args:
            report: Evaluation report to export
        """
        suite_name = report.suite_name

        # Update evaluation counters
        successful_evals = report.execution_metadata.successful_evaluations
        failed_evals = report.execution_metadata.failed_evaluations

        self.increment_counter(
            "evaluations_total", successful_evals, {"status": "completed", "suite": suite_name}
        )

        self.increment_counter(
            "evaluations_total", failed_evals, {"status": "failed", "suite": suite_name}
        )

        # Update execution time
        self.update_metric(
            "evaluation_duration_seconds",
            report.execution_metadata.total_execution_time,
            {"suite": suite_name},
        )

        # Export individual metric scores
        for metric_name, score in report.aggregates.items():
            self.update_metric(
                "metric_score_current", score, {"metric": metric_name, "suite": suite_name}
            )

        # Export threshold status
        for metric_name, passed in report.threshold_status.items():
            self.update_metric(
                "threshold_status", 1 if passed else 0, {"metric": metric_name, "suite": suite_name}
            )

        # Export cost information
        if report.cost_summary and report.cost_summary.total_estimated_cost:
            self.increment_counter(
                "evaluation_cost_total",
                report.cost_summary.total_estimated_cost,
                {"suite": suite_name},
            )

        logger.debug(f"Exported metrics for evaluation report {report.run_id}")

    def export_queue_metrics(self, queue_stats: Dict[str, Any]):
        """
        Export queue-related metrics.

        Args:
            queue_stats: Queue statistics dictionary
        """
        self.update_metric("queue_size", queue_stats.get("current_size", 0))
        self.update_metric("queue_failed_size", queue_stats.get("failed_size", 0))
        self.update_metric("queue_utilization", queue_stats.get("utilization", 0))

        # Calculate processing rate if available
        if "dequeued_total" in queue_stats and "enqueued_total" in queue_stats:
            processing_rate = queue_stats["dequeued_total"] / max(1, queue_stats["enqueued_total"])
            self.update_metric("queue_processing_rate", processing_rate)

    def export_system_metrics(
        self,
        server_uptime: float,
        active_connections: int = 0,
        memory_usage_mb: Optional[float] = None,
    ):
        """
        Export system health metrics.

        Args:
            server_uptime: Server uptime in seconds
            active_connections: Number of active HTTP connections
            memory_usage_mb: Memory usage in megabytes
        """
        self.update_metric("server_up", 1)  # Server is running
        self.update_metric("server_uptime_seconds", server_uptime)
        self.update_metric("active_connections", active_connections)

        if memory_usage_mb is not None:
            self.update_metric("memory_usage_megabytes", memory_usage_mb)

    def generate_prometheus_output(self) -> str:
        """
        Generate Prometheus exposition format output.

        Returns:
            String in Prometheus exposition format
        """
        output_lines = []

        # Group metrics by name to handle multiple label sets
        metrics_by_name: Dict[str, List[PrometheusMetric]] = {}
        for metric in self.metrics.values():
            if metric.name not in metrics_by_name:
                metrics_by_name[metric.name] = []
            metrics_by_name[metric.name].append(metric)

        # Generate output for each metric group
        for metric_name in sorted(metrics_by_name.keys()):
            metric_group = metrics_by_name[metric_name]

            # Add HELP and TYPE lines (from first metric in group)
            first_metric = metric_group[0]
            output_lines.append(f"# HELP {metric_name} {first_metric.help_text}")
            output_lines.append(f"# TYPE {metric_name} {first_metric.metric_type}")

            # Add all metric values
            for metric in metric_group:
                if metric.labels:
                    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(metric.labels.items()))
                    metric_line = f"{metric_name}{{{label_str}}} {metric.value}"
                else:
                    metric_line = f"{metric_name} {metric.value}"

                output_lines.append(metric_line)

            output_lines.append("")  # Empty line between metric groups

        return "\n".join(output_lines)

    def get_metrics_endpoint_response(self) -> tuple[str, str]:
        """
        Get response for /metrics endpoint.

        Returns:
            Tuple of (content, content_type)
        """
        content = self.generate_prometheus_output()
        content_type = "text/plain; version=0.0.4; charset=utf-8"

        return content, content_type

    def _get_metric_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Generate unique key for metric with labels."""
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"

    def reset_metrics(self):
        """Reset all metrics to initial state."""
        self.metrics.clear()
        self._initialize_core_metrics()
        logger.info("Prometheus metrics reset")

    def get_metric_names(self) -> List[str]:
        """Get list of all metric names."""
        return list({metric.name for metric in self.metrics.values()})

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of current metrics.

        Returns:
            Dictionary with metrics summary
        """
        return {
            "total_metrics": len(self.metrics),
            "metric_names": self.get_metric_names(),
            "last_updated": max(
                (metric.timestamp for metric in self.metrics.values()), default=time.time()
            ),
        }


def create_prometheus_middleware(exporter: PrometheusExporter):
    """
    Create FastAPI middleware for automatic metrics collection.

    Args:
        exporter: Prometheus exporter instance

    Returns:
        FastAPI middleware function
    """
    import time

    from fastapi import Request, Response

    async def prometheus_middleware(request: Request, call_next):
        """Middleware to collect HTTP metrics."""
        start_time = time.time()

        # Process request
        response: Response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Update HTTP metrics
        method = request.method
        path = request.url.path
        status_code = response.status_code

        exporter.increment_counter(
            "http_requests_total", 1, {"method": method, "path": path, "status": str(status_code)}
        )

        exporter.update_metric(
            "http_request_duration_seconds", duration, {"method": method, "path": path}
        )

        return response

    return prometheus_middleware
