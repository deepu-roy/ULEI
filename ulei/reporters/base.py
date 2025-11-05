"""
Base reporter class with common reporting functionality.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ulei.core.interfaces import BaseReporter as CoreBaseReporter
from ulei.core.schemas import EvaluationReport, MetricResult
from ulei.utils.errors import ReportingError

logger = logging.getLogger(__name__)


class BaseReporter(CoreBaseReporter):
    """Enhanced base reporter with common functionality."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize reporter with optional configuration.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup reporter-specific logging."""
        self.logger = logging.getLogger(f"ulei.reporters.{self.format_name}")

    def generate_report(self, report: EvaluationReport, output_path: str) -> str:
        """Generate report with error handling and directory creation.

        Args:
            report: Evaluation report to generate
            output_path: Path where to save the report

        Returns:
            Path to the generated report file

        Raises:
            ReportingError: If report generation fails
        """
        try:
            # Ensure output path has correct extension
            output_path = self._ensure_extension(output_path)

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Generate the actual report
            self._generate_report_content(report, output_path)

            # Verify file was created
            if not os.path.exists(output_path):
                raise ReportingError(f"Report file was not created: {output_path}")

            file_size = os.path.getsize(output_path)
            self.logger.info(
                f"Generated {self.format_name} report: {output_path} ({file_size} bytes)"
            )

            return output_path

        except Exception as e:
            error_msg = f"Failed to generate {self.format_name} report: {e}"
            self.logger.error(error_msg)
            raise ReportingError(error_msg) from e

    def _generate_report_content(self, report: EvaluationReport, output_path: str) -> None:
        """Generate the actual report content.

        This method must be implemented by subclasses.

        Args:
            report: Evaluation report data
            output_path: Path where to save the report
        """
        raise NotImplementedError("Subclasses must implement _generate_report_content")

    def _calculate_aggregates(self, report: EvaluationReport) -> Dict[str, Dict[str, float]]:
        """Calculate aggregate statistics from results.

        Args:
            report: Evaluation report containing results

        Returns:
            Dictionary with aggregate statistics per metric
        """
        aggregates = {}

        # Group results by metric
        metric_results: Dict[str, List[MetricResult]] = {}
        for result in report.results:
            if result.error is None and result.score is not None:  # Only successful results
                if result.metric not in metric_results:
                    metric_results[result.metric] = []
                metric_results[result.metric].append(result)

        # Calculate statistics for each metric
        for metric_name, results in metric_results.items():
            if not results:
                continue

            scores = [r.score for r in results if r.score is not None]
            execution_times = [r.execution_time for r in results]
            costs = [r.cost_estimate for r in results if r.cost_estimate is not None]

            stats = {
                "count": len(results),
                "success_rate": len(scores) / len(results) if results else 0.0,
                "mean_score": sum(scores) / len(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "mean_execution_time": sum(execution_times) / len(execution_times)
                if execution_times
                else 0.0,
                "total_cost": sum(costs) if costs else 0.0,
            }

            # Calculate standard deviation
            if len(scores) > 1:
                mean = stats["mean_score"]
                variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
                stats["std_score"] = variance**0.5
            else:
                stats["std_score"] = 0.0

            aggregates[metric_name] = stats

        return aggregates

    def _get_threshold_summary(self, report: EvaluationReport) -> Dict[str, Any]:
        """Get summary of threshold compliance.

        Args:
            report: Evaluation report

        Returns:
            Dictionary with threshold status summary
        """
        total_thresholds = len(report.threshold_status)
        passed_thresholds = sum(1 for status in report.threshold_status.values() if status)

        return {
            "total_thresholds": total_thresholds,
            "passed_thresholds": passed_thresholds,
            "failed_thresholds": total_thresholds - passed_thresholds,
            "pass_rate": passed_thresholds / total_thresholds if total_thresholds > 0 else 0.0,
            "overall_pass": all(report.threshold_status.values())
            if report.threshold_status
            else True,
        }

    def _format_timestamp(
        self, timestamp: datetime, format_str: str = "%Y-%m-%d %H:%M:%S UTC"
    ) -> str:
        """Format timestamp for display.

        Args:
            timestamp: Datetime to format
            format_str: Format string

        Returns:
            Formatted timestamp string
        """
        return timestamp.strftime(format_str)

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable format.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted duration string
        """
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            return f"{hours}h {remaining_minutes}m"

    def _format_percentage(self, value: float) -> str:
        """Format a decimal as a percentage.

        Args:
            value: Decimal value (0.0 to 1.0)

        Returns:
            Formatted percentage string
        """
        return f"{value * 100:.1f}%"

    def _format_cost(self, cost: Optional[float]) -> str:
        """Format cost for display.

        Args:
            cost: Cost value in USD

        Returns:
            Formatted cost string
        """
        if cost is None:
            return "N/A"
        elif cost == 0:
            return "$0.00"
        elif cost < 0.01:
            return f"${cost:.4f}"
        else:
            return f"${cost:.2f}"

    def _group_results_by_metric(
        self, results: List[MetricResult]
    ) -> Dict[str, List[MetricResult]]:
        """Group results by metric name.

        Args:
            results: List of metric results

        Returns:
            Dictionary mapping metric names to lists of results
        """
        grouped: Dict[str, List[MetricResult]] = {}
        for result in results:
            if result.metric not in grouped:
                grouped[result.metric] = []
            grouped[result.metric].append(result)
        return grouped

    def _group_results_by_provider(
        self, results: List[MetricResult]
    ) -> Dict[str, List[MetricResult]]:
        """Group results by provider name.

        Args:
            results: List of metric results

        Returns:
            Dictionary mapping provider names to lists of results
        """
        grouped: Dict[str, List[MetricResult]] = {}
        for result in results:
            if result.provider not in grouped:
                grouped[result.provider] = []
            grouped[result.provider].append(result)
        return grouped

    def _validate_report(self, report: EvaluationReport) -> None:
        """Validate report data before generation.

        Args:
            report: Report to validate

        Raises:
            ReportingError: If report data is invalid
        """
        if not report.run_id:
            raise ReportingError("Report must have a run_id")

        if not report.suite_name:
            raise ReportingError("Report must have a suite_name")

        if not hasattr(report, "execution_metadata"):
            raise ReportingError("Report must have execution_metadata")

    def _get_report_metadata(self, report: EvaluationReport) -> Dict[str, Any]:
        """Extract metadata for report generation.

        Args:
            report: Evaluation report

        Returns:
            Dictionary with metadata for the report
        """
        return {
            "run_id": report.run_id,
            "suite_name": report.suite_name,
            "generated_at": datetime.utcnow(),
            "total_items": report.execution_metadata.total_items,
            "total_results": len(report.results),
            "execution_time": report.execution_metadata.total_execution_time,
            "format": self.format_name,
        }
