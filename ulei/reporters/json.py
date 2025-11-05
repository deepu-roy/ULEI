"""
JSON reporter for generating machine-readable evaluation reports.
"""

import json
from datetime import datetime
from typing import Any, Dict

from ulei.core.schemas import EvaluationReport, MetricResult
from ulei.reporters.base import BaseReporter


class JSONReporter(BaseReporter):
    """Reporter for generating JSON format evaluation reports."""

    @property
    def format_name(self) -> str:
        """Return report format name."""
        return "json"

    @property
    def file_extension(self) -> str:
        """Return file extension for JSON reports."""
        return ".json"

    def _generate_report_content(self, report: EvaluationReport, output_path: str) -> None:
        """Generate JSON report content.

        Args:
            report: Evaluation report data
            output_path: Path where to save the report
        """
        # Validate report
        self._validate_report(report)

        # Build JSON structure
        json_data = self._build_json_structure(report)

        # Write JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, default=self._json_serializer, ensure_ascii=False)

    def _build_json_structure(self, report: EvaluationReport) -> Dict[str, Any]:
        """Build JSON report structure.

        Args:
            report: Evaluation report

        Returns:
            Dictionary representing JSON structure
        """
        # Calculate aggregates for better reporting
        aggregates = self._calculate_aggregates(report)
        threshold_summary = self._get_threshold_summary(report)
        metadata = self._get_report_metadata(report)

        # Group results for better organization
        results_by_metric = self._group_results_by_metric(report.results)
        results_by_provider = self._group_results_by_provider(report.results)

        json_data = {
            # Report metadata
            "report_metadata": {
                "run_id": report.run_id,
                "suite_name": report.suite_name,
                "generated_at": metadata["generated_at"].isoformat(),
                "format": "json",
                "format_version": "1.0",
            },
            # Execution summary
            "execution_summary": {
                "status": "complete",  # Could be derived from report state
                "started_at": None,  # Would need to be added to report schema
                "completed_at": report.created_at.isoformat(),
                "total_execution_time": report.execution_metadata.total_execution_time,
                "total_items": report.execution_metadata.total_items,
                "successful_evaluations": report.execution_metadata.successful_evaluations,
                "failed_evaluations": report.execution_metadata.failed_evaluations,
                "cache_hits": report.execution_metadata.cache_hits,
                "provider_usage": report.execution_metadata.provider_usage,
            },
            # Dataset statistics
            "dataset_stats": report.dataset_stats,
            # Metrics summary
            "metrics_summary": {
                metric: {
                    "aggregate_score": report.aggregates.get(metric, 0.0),
                    "threshold": None,  # Will be filled below
                    "threshold_passed": report.threshold_status.get(metric),
                    **aggregates.get(metric, {}),
                }
                for metric in {result.metric for result in report.results}
            },
            # Overall threshold status
            "threshold_summary": threshold_summary,
            # Cost information
            "cost_summary": self._serialize_cost_summary(report.cost_summary),
            # Detailed results
            "detailed_results": {
                "by_metric": {
                    metric: [self._serialize_result(r) for r in results]
                    for metric, results in results_by_metric.items()
                },
                "by_provider": {
                    provider: [self._serialize_result(r) for r in results]
                    for provider, results in results_by_provider.items()
                },
                "all_results": [self._serialize_result(r) for r in report.results],
            },
            # Performance insights
            "performance_insights": self._generate_performance_insights(report),
            # Errors and warnings
            "issues": self._extract_issues(report),
        }

        # Add threshold information to metrics summary
        for metric, summary in json_data["metrics_summary"].items():
            if metric in report.threshold_status:
                # Find threshold value from suite (would need to be passed or stored)
                summary["threshold"] = None  # Would need suite thresholds

        return json_data

    def _serialize_result(self, result: MetricResult) -> Dict[str, Any]:
        """Serialize a MetricResult to JSON-compatible dict.

        Args:
            result: MetricResult to serialize

        Returns:
            JSON-compatible dictionary
        """
        return {
            "metric": result.metric,
            "provider": result.provider,
            "item_id": result.item_id,
            "score": result.score,
            "confidence": result.confidence,
            "evidence": result.evidence,
            "execution_time": result.execution_time,
            "cost_estimate": result.cost_estimate,
            "error": result.error,
            "created_at": result.created_at.isoformat(),
        }

    def _serialize_cost_summary(self, cost_summary: Any) -> Dict[str, Any]:
        """Serialize cost summary to JSON-compatible format.

        Args:
            cost_summary: CostSummary object or None

        Returns:
            JSON-compatible dictionary
        """
        if cost_summary is None:
            return {
                "total_estimated_cost": 0.0,
                "cost_by_provider": {},
                "cost_by_metric": {},
                "budget_utilization": 0.0,
                "currency": "USD",
            }

        return {
            "total_estimated_cost": cost_summary.total_estimated_cost,
            "cost_by_provider": cost_summary.cost_by_provider,
            "cost_by_metric": cost_summary.cost_by_metric,
            "budget_utilization": cost_summary.budget_utilization,
            "currency": "USD",
        }

    def _generate_performance_insights(self, report: EvaluationReport) -> Dict[str, Any]:
        """Generate performance insights from the evaluation results.

        Args:
            report: Evaluation report

        Returns:
            Performance insights dictionary
        """
        results = report.results
        if not results:
            return {}

        # Calculate execution time statistics
        execution_times = [r.execution_time for r in results if r.execution_time > 0]

        if not execution_times:
            return {}

        execution_times.sort()
        n = len(execution_times)

        insights = {
            "execution_time_stats": {
                "mean": sum(execution_times) / n,
                "median": execution_times[n // 2],
                "min": min(execution_times),
                "max": max(execution_times),
                "p95": execution_times[int(0.95 * n)] if n > 20 else max(execution_times),
            },
            "throughput": {
                "items_per_second": report.execution_metadata.total_items
                / max(report.execution_metadata.total_execution_time, 0.001),
                "avg_time_per_item": report.execution_metadata.total_execution_time
                / max(report.execution_metadata.total_items, 1),
            },
            "error_rate": {
                "total_errors": report.execution_metadata.failed_evaluations,
                "error_rate": report.execution_metadata.failed_evaluations
                / max(report.execution_metadata.total_items, 1),
                "success_rate": report.execution_metadata.successful_evaluations
                / max(report.execution_metadata.total_items, 1),
            },
        }

        # Add cache effectiveness if cache data available
        if report.execution_metadata.cache_hits > 0:
            total_requests = (
                report.execution_metadata.successful_evaluations
                + report.execution_metadata.failed_evaluations
                + report.execution_metadata.cache_hits
            )
            insights["cache_effectiveness"] = {
                "hit_rate": report.execution_metadata.cache_hits / max(total_requests, 1),
                "total_hits": report.execution_metadata.cache_hits,
            }

        return insights

    def _extract_issues(self, report: EvaluationReport) -> Dict[str, Any]:
        """Extract errors and warnings from the evaluation report.

        Args:
            report: Evaluation report

        Returns:
            Dictionary with errors and warnings
        """
        errors = []
        warnings = []

        # Extract errors from results
        for result in report.results:
            if result.error:
                errors.append(
                    {
                        "type": "evaluation_error",
                        "metric": result.metric,
                        "provider": result.provider,
                        "item_id": result.item_id,
                        "message": result.error,
                        "timestamp": result.created_at.isoformat(),
                    }
                )

        # Check for threshold failures
        for metric, passed in report.threshold_status.items():
            if not passed:
                warnings.append(
                    {
                        "type": "threshold_failure",
                        "metric": metric,
                        "message": f"Metric '{metric}' did not meet threshold",
                        "aggregate_score": report.aggregates.get(metric, 0.0),
                    }
                )

        # Check for performance warnings
        if report.execution_metadata.total_execution_time > 300:  # 5 minutes
            warnings.append(
                {
                    "type": "performance_warning",
                    "message": f"Evaluation took {report.execution_metadata.total_execution_time:.1f}s, "
                    "consider optimizing or using parallel processing",
                }
            )

        return {
            "errors": errors,
            "warnings": warnings,
            "error_count": len(errors),
            "warning_count": len(warnings),
        }

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for non-standard types.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable representation
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, "model_dump"):  # Pydantic models
            return obj.model_dump()
        elif hasattr(obj, "dict"):  # Pydantic models (older versions)
            return obj.dict()
        else:
            return str(obj)
