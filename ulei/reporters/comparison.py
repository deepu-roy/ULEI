"""
Comparison report generator for evaluation performance analysis.

Generates reports comparing evaluation runs with statistical significance,
delta visualization, and regression detection.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ulei.core.schemas import EvaluationReport
from ulei.reporters.base import BaseReporter
from ulei.utils.stats import (
    ComparisonResult,
    TrendAnalysis,
    analyze_trend,
    compare_metrics,
    detect_regressions,
    summarize_comparison,
)

logger = logging.getLogger(__name__)


class ComparisonReporter(BaseReporter):
    """Reporter for generating comparison reports between evaluation runs."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize comparison reporter.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir or Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_comparison_report(
        self,
        baseline_report: EvaluationReport,
        comparison_report: EvaluationReport,
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        Generate a comprehensive comparison report.

        Args:
            baseline_report: Baseline evaluation report
            comparison_report: Comparison evaluation report
            output_filename: Optional custom filename

        Returns:
            Path to generated report file
        """
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"comparison_report_{timestamp}.json"

        output_path = self.output_dir / output_filename

        # Get all unique metric names
        baseline_metrics = {r.metric for r in baseline_report.results}
        comparison_metrics = {r.metric for r in comparison_report.results}
        all_metrics = baseline_metrics.union(comparison_metrics)

        # Generate comparisons for each metric
        comparisons = []
        for metric_name in all_metrics:
            comparison = compare_metrics(
                baseline_report.results, comparison_report.results, metric_name
            )
            if comparison:
                comparisons.append(comparison)

        # Detect regressions
        regressions = detect_regressions(
            baseline_report.results, comparison_report.results, list(all_metrics)
        )

        # Generate summary
        summary = summarize_comparison(comparisons)

        # Build report data
        report_data = {
            "comparison_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "baseline_run_id": baseline_report.run_id,
                "comparison_run_id": comparison_report.run_id,
                "baseline_suite": baseline_report.suite_name,
                "comparison_suite": comparison_report.suite_name,
            },
            "summary": summary,
            "metric_comparisons": [self._comparison_to_dict(c) for c in comparisons],
            "detected_regressions": [self._comparison_to_dict(regressions[m]) for m in regressions],
            "baseline_aggregates": baseline_report.aggregates,
            "comparison_aggregates": comparison_report.aggregates,
            "execution_comparison": {
                "baseline_execution_time": baseline_report.execution_metadata.total_execution_time,
                "comparison_execution_time": comparison_report.execution_metadata.total_execution_time,
                "baseline_success_rate": (
                    baseline_report.execution_metadata.successful_evaluations
                    / len(baseline_report.results)
                )
                if baseline_report.results
                else 0,
                "comparison_success_rate": (
                    comparison_report.execution_metadata.successful_evaluations
                    / len(comparison_report.results)
                )
                if comparison_report.results
                else 0,
            },
        }

        # Write report
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"Comparison report generated: {output_path}")
        return output_path

    def generate_trend_analysis_report(
        self,
        reports_over_time: List[tuple[float, EvaluationReport]],
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        Generate trend analysis report from multiple evaluation runs.

        Args:
            reports_over_time: List of (timestamp, report) tuples sorted by time
            output_filename: Optional custom filename

        Returns:
            Path to generated report file
        """
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"trend_analysis_{timestamp}.json"

        output_path = self.output_dir / output_filename

        if len(reports_over_time) < 3:
            logger.warning("Insufficient data points for trend analysis")
            return output_path

        # Get all metrics across all reports
        all_metrics: set[str] = set()
        for _, report in reports_over_time:
            all_metrics.update(r.metric for r in report.results)

        # Analyze trends for each metric
        trend_analyses = []
        results_by_time = [(ts, report.results) for ts, report in reports_over_time]

        for metric_name in all_metrics:
            trend = analyze_trend(results_by_time, metric_name)
            if trend:
                trend_analyses.append(trend)

        # Build report data
        report_data = {
            "trend_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "time_period": {
                    "start": min(ts for ts, _ in reports_over_time),
                    "end": max(ts for ts, _ in reports_over_time),
                    "num_points": len(reports_over_time),
                },
            },
            "trend_analyses": [self._trend_to_dict(t) for t in trend_analyses],
            "summary": {
                "total_metrics": len(trend_analyses),
                "improving_trends": len(
                    [t for t in trend_analyses if t.trend_direction == "improving"]
                ),
                "declining_trends": len(
                    [t for t in trend_analyses if t.trend_direction == "declining"]
                ),
                "stable_metrics": len([t for t in trend_analyses if t.trend_direction == "stable"]),
            },
        }

        # Write report
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"Trend analysis report generated: {output_path}")
        return output_path

    def generate_html_comparison_report(
        self,
        baseline_report: EvaluationReport,
        comparison_report: EvaluationReport,
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        Generate HTML comparison report with visualizations.

        Args:
            baseline_report: Baseline evaluation report
            comparison_report: Comparison evaluation report
            output_filename: Optional custom filename

        Returns:
            Path to generated HTML report
        """
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"comparison_report_{timestamp}.html"

        output_path = self.output_dir / output_filename

        # Generate comparison data
        baseline_metrics = {r.metric for r in baseline_report.results}
        comparison_metrics = {r.metric for r in comparison_report.results}
        all_metrics = baseline_metrics.union(comparison_metrics)

        comparisons = []
        for metric_name in all_metrics:
            comparison = compare_metrics(
                baseline_report.results, comparison_report.results, metric_name
            )
            if comparison:
                comparisons.append(comparison)

        regressions = detect_regressions(
            baseline_report.results, comparison_report.results, list(all_metrics)
        )

        summary = summarize_comparison(comparisons)

        # Generate HTML content
        html_content = self._generate_html_content(
            baseline_report, comparison_report, comparisons, regressions, summary
        )

        # Write HTML file
        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML comparison report generated: {output_path}")
        return output_path

    def _comparison_to_dict(self, comparison: ComparisonResult) -> Dict[str, Any]:
        """Convert ComparisonResult to dictionary."""
        return {
            "metric_name": comparison.metric_name,
            "baseline_mean": comparison.baseline_mean,
            "comparison_mean": comparison.comparison_mean,
            "delta": comparison.delta,
            "delta_percent": comparison.delta_percent,
            "p_value": comparison.p_value,
            "is_significant": comparison.is_significant,
            "confidence_interval": comparison.confidence_interval,
            "sample_sizes": comparison.sample_sizes,
            "effect_size": comparison.effect_size,
            "test_type": comparison.test_type,
        }

    def _trend_to_dict(self, trend: TrendAnalysis) -> Dict[str, Any]:
        """Convert TrendAnalysis to dictionary."""
        return {
            "metric_name": trend.metric_name,
            "slope": trend.slope,
            "intercept": trend.intercept,
            "r_squared": trend.r_squared,
            "p_value": trend.p_value,
            "is_significant_trend": trend.is_significant_trend,
            "trend_direction": trend.trend_direction,
            "confidence_interval": trend.confidence_interval,
        }

    def _generate_html_content(
        self,
        baseline_report: EvaluationReport,
        comparison_report: EvaluationReport,
        comparisons: List[ComparisonResult],
        regressions: Dict[str, ComparisonResult],
        summary: Dict[str, Any],
    ) -> str:
        """Generate HTML content for comparison report."""

        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #333; }}
        .summary {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .metric-comparison {{ border: 1px solid #ddd; margin: 15px 0; padding: 15px; border-radius: 5px; }}
        .improvement {{ border-left: 4px solid #28a745; }}
        .regression {{ border-left: 4px solid #dc3545; }}
        .stable {{ border-left: 4px solid #6c757d; }}
        .delta-positive {{ color: #28a745; font-weight: bold; }}
        .delta-negative {{ color: #dc3545; font-weight: bold; }}
        .delta-neutral {{ color: #6c757d; }}
        .significant {{ background: #fff3cd; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metadata {{ font-size: 0.9em; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Evaluation Comparison Report</h1>

        <div class="metadata">
            <p><strong>Generated:</strong> {generated_at}</p>
            <p><strong>Baseline:</strong> {baseline_run_id} ({baseline_suite})</p>
            <p><strong>Comparison:</strong> {comparison_run_id} ({comparison_suite})</p>
        </div>

        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Total Metrics:</strong> {total_metrics}</p>
            <p><strong>Significant Changes:</strong> {significant_changes}</p>
            <p><strong>Improvements:</strong> <span class="delta-positive">{improvements}</span></p>
            <p><strong>Regressions:</strong> <span class="delta-negative">{regressions}</span></p>
            <p><strong>Stable:</strong> {stable_metrics}</p>
        </div>

        <h2>Metric Comparisons</h2>
        {comparisons_html}

        {regressions_section}

        <h2>Execution Comparison</h2>
        <table>
            <tr><th>Metric</th><th>Baseline</th><th>Comparison</th><th>Change</th></tr>
            <tr>
                <td>Execution Time (s)</td>
                <td>{baseline_exec_time:.2f}</td>
                <td>{comparison_exec_time:.2f}</td>
                <td class="{exec_time_class}">{exec_time_delta:+.2f}</td>
            </tr>
            <tr>
                <td>Success Rate</td>
                <td>{baseline_success_rate:.1%}</td>
                <td>{comparison_success_rate:.1%}</td>
                <td class="{success_rate_class}">{success_rate_delta:+.1%}</td>
            </tr>
        </table>
    </div>
</body>
</html>
        """

        # Generate comparisons HTML
        comparisons_html = ""
        for comparison in comparisons:
            css_class = (
                "improvement"
                if comparison.delta > 0 and comparison.is_significant
                else "regression"
                if comparison.delta < 0 and comparison.is_significant
                else "stable"
            )

            delta_class = (
                "delta-positive"
                if comparison.delta > 0
                else "delta-negative"
                if comparison.delta < 0
                else "delta-neutral"
            )

            significance_class = "significant" if comparison.is_significant else ""

            comparisons_html += f"""
            <div class="metric-comparison {css_class} {significance_class}">
                <h3>{comparison.metric_name}</h3>
                <p><strong>Baseline:</strong> {comparison.baseline_mean:.3f}</p>
                <p><strong>Comparison:</strong> {comparison.comparison_mean:.3f}</p>
                <p><strong>Delta:</strong> <span class="{delta_class}">{comparison.delta:+.3f} ({comparison.delta_percent:+.1f}%)</span></p>
                <p><strong>P-value:</strong> {comparison.p_value:.4f} {"(Significant)" if comparison.is_significant else "(Not significant)"}</p>
                <p><strong>Sample sizes:</strong> {comparison.sample_sizes[0]} vs {comparison.sample_sizes[1]}</p>
                {f"<p><strong>Effect size:</strong> {comparison.effect_size:.3f}</p>" if comparison.effect_size else ""}
            </div>
            """

        # Generate regressions section
        regressions_section = ""
        if regressions:
            regressions_section = "<h2>⚠️ Detected Regressions</h2>"
            for metric_name, regression in regressions.items():
                regressions_section += f"""
                <div class="metric-comparison regression significant">
                    <h3>{metric_name}</h3>
                    <p><strong>Performance declined by {abs(regression.delta_percent):.1f}%</strong></p>
                    <p>Baseline: {regression.baseline_mean:.3f} → Comparison: {regression.comparison_mean:.3f}</p>
                    <p>P-value: {regression.p_value:.4f}</p>
                </div>
                """

        # Calculate execution comparison deltas
        exec_time_delta = (
            comparison_report.execution_metadata.total_execution_time
            - baseline_report.execution_metadata.total_execution_time
        )
        success_rate_delta = (
            (
                (
                    comparison_report.execution_metadata.successful_evaluations
                    / len(comparison_report.results)
                )
                - (
                    baseline_report.execution_metadata.successful_evaluations
                    / len(baseline_report.results)
                )
            )
            if baseline_report.results and comparison_report.results
            else 0
        )

        # Format the HTML
        return html_template.format(
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            baseline_run_id=baseline_report.run_id,
            baseline_suite=baseline_report.suite_name,
            comparison_run_id=comparison_report.run_id,
            comparison_suite=comparison_report.suite_name,
            total_metrics=summary.get("total_metrics", 0),
            significant_changes=summary.get("significant_changes", 0),
            improvements=summary.get("improvements", 0),
            regressions=summary.get("regressions", 0),
            stable_metrics=summary.get("stable_metrics", 0),
            comparisons_html=comparisons_html,
            regressions_section=regressions_section,
            baseline_exec_time=baseline_report.execution_metadata.total_execution_time,
            comparison_exec_time=comparison_report.execution_metadata.total_execution_time,
            exec_time_delta=exec_time_delta,
            exec_time_class="delta-negative" if exec_time_delta > 0 else "delta-positive",
            baseline_success_rate=(
                baseline_report.execution_metadata.successful_evaluations
                / len(baseline_report.results)
            )
            if baseline_report.results
            else 0,
            comparison_success_rate=(
                comparison_report.execution_metadata.successful_evaluations
                / len(comparison_report.results)
            )
            if comparison_report.results
            else 0,
            success_rate_delta=success_rate_delta,
            success_rate_class="delta-positive"
            if success_rate_delta > 0
            else "delta-negative"
            if success_rate_delta < 0
            else "delta-neutral",
        )
