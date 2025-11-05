"""
HTML reporter for generating human-readable evaluation reports.
"""

from typing import Any, Dict

from ulei.core.schemas import EvaluationReport
from ulei.reporters.base import BaseReporter


class HTMLReporter(BaseReporter):
    """Reporter for generating HTML format evaluation reports."""

    @property
    def format_name(self) -> str:
        """Return report format name."""
        return "html"

    @property
    def file_extension(self) -> str:
        """Return file extension for HTML reports."""
        return ".html"

    def _generate_report_content(self, report: EvaluationReport, output_path: str) -> None:
        """Generate HTML report content.

        Args:
            report: Evaluation report data
            output_path: Path where to save the report
        """
        # Validate report
        self._validate_report(report)

        # Generate HTML content
        html_content = self._build_html_report(report)

        # Write HTML file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _build_html_report(self, report: EvaluationReport) -> str:
        """Build complete HTML report.

        Args:
            report: Evaluation report

        Returns:
            Complete HTML report as string
        """
        # Calculate data for report
        aggregates = self._calculate_aggregates(report)
        threshold_summary = self._get_threshold_summary(report)
        metadata = self._get_report_metadata(report)

        # Build HTML sections
        css_styles = self._get_css_styles()
        header_html = self._build_header_section(report, metadata)
        summary_html = self._build_summary_section(report, threshold_summary)
        metrics_html = self._build_metrics_section(report, aggregates)
        results_html = self._build_results_section(report)
        performance_html = self._build_performance_section(report)
        footer_html = self._build_footer_section(metadata)

        # Combine into complete HTML document
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ULEI Evaluation Report - {report.suite_name}</title>
    <style>{css_styles}</style>
</head>
<body>
    <div class="container">
        {header_html}
        {summary_html}
        {metrics_html}
        {results_html}
        {performance_html}
        {footer_html}
    </div>
    <script>{self._get_javascript()}</script>
</body>
</html>"""

        return html_content

    def _build_header_section(self, report: EvaluationReport, metadata: Dict[str, Any]) -> str:
        """Build header section of the report.

        Args:
            report: Evaluation report
            metadata: Report metadata

        Returns:
            HTML header section
        """
        return f"""
        <header class="report-header">
            <div class="header-content">
                <h1>🚀 ULEI Evaluation Report</h1>
                <div class="report-info">
                    <div class="info-item">
                        <span class="label">Suite:</span>
                        <span class="value">{report.suite_name}</span>
                    </div>
                    <div class="info-item">
                        <span class="label">Run ID:</span>
                        <span class="value">{report.run_id}</span>
                    </div>
                    <div class="info-item">
                        <span class="label">Generated:</span>
                        <span class="value">{self._format_timestamp(metadata["generated_at"])}</span>
                    </div>
                    <div class="info-item">
                        <span class="label">Items:</span>
                        <span class="value">{report.execution_metadata.total_items:,}</span>
                    </div>
                    <div class="info-item">
                        <span class="label">Duration:</span>
                        <span class="value">{self._format_duration(report.execution_metadata.total_execution_time)}</span>
                    </div>
                </div>
            </div>
        </header>
        """

    def _build_summary_section(
        self, report: EvaluationReport, threshold_summary: Dict[str, Any]
    ) -> str:
        """Build summary section showing overall status.

        Args:
            report: Evaluation report
            threshold_summary: Threshold compliance summary

        Returns:
            HTML summary section
        """
        overall_status = "✅ PASS" if threshold_summary["overall_pass"] else "❌ FAIL"
        status_class = "pass" if threshold_summary["overall_pass"] else "fail"

        return f"""
        <section class="summary-section">
            <h2>📊 Evaluation Summary</h2>
            <div class="summary-grid">
                <div class="summary-card status-card {status_class}">
                    <h3>Overall Status</h3>
                    <div class="status-value">{overall_status}</div>
                    <p>{threshold_summary["passed_thresholds"]}/{threshold_summary["total_thresholds"]} thresholds passed</p>
                </div>

                <div class="summary-card">
                    <h3>Success Rate</h3>
                    <div class="metric-value">
                        {self._format_percentage(report.execution_metadata.successful_evaluations / max(report.execution_metadata.total_items, 1))}
                    </div>
                    <p>{report.execution_metadata.successful_evaluations:,} / {report.execution_metadata.total_items:,} successful</p>
                </div>

                <div class="summary-card">
                    <h3>Performance</h3>
                    <div class="metric-value">
                        {(report.execution_metadata.total_items / max(report.execution_metadata.total_execution_time, 0.001)):.1f}/sec
                    </div>
                    <p>Average throughput</p>
                </div>

                <div class="summary-card">
                    <h3>Cost Estimate</h3>
                    <div class="metric-value">
                        {self._format_cost(report.cost_summary.total_estimated_cost if report.cost_summary else 0.0)}
                    </div>
                    <p>Total evaluation cost</p>
                </div>
            </div>
        </section>
        """

    def _build_metrics_section(
        self, report: EvaluationReport, aggregates: Dict[str, Dict[str, float]]
    ) -> str:
        """Build metrics section showing metric-wise results.

        Args:
            report: Evaluation report
            aggregates: Aggregated metrics data

        Returns:
            HTML metrics section
        """
        metrics_cards = []

        for metric_name in sorted(aggregates.keys()):
            stats = aggregates[metric_name]
            threshold_passed = report.threshold_status.get(metric_name)

            # Determine status
            if threshold_passed is None:
                status_icon = "ℹ️"
                status_class = "info"
                status_text = "No threshold"
            elif threshold_passed:
                status_icon = "✅"
                status_class = "pass"
                status_text = "Passed"
            else:
                status_icon = "❌"
                status_class = "fail"
                status_text = "Failed"

            # Build metric card
            card_html = f"""
            <div class="metric-card {status_class}">
                <div class="metric-header">
                    <h3>{metric_name}</h3>
                    <span class="status-badge {status_class}">{status_icon} {status_text}</span>
                </div>
                <div class="metric-stats">
                    <div class="stat">
                        <span class="stat-label">Mean Score</span>
                        <span class="stat-value">{stats["mean_score"]:.3f}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Range</span>
                        <span class="stat-value">{stats["min_score"]:.3f} - {stats["max_score"]:.3f}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Std Dev</span>
                        <span class="stat-value">{stats["std_score"]:.3f}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Success Rate</span>
                        <span class="stat-value">{self._format_percentage(stats["success_rate"])}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Avg Time</span>
                        <span class="stat-value">{stats["mean_execution_time"]:.2f}s</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Cost</span>
                        <span class="stat-value">{self._format_cost(stats["total_cost"])}</span>
                    </div>
                </div>
            </div>
            """
            metrics_cards.append(card_html)

        return f"""
        <section class="metrics-section">
            <h2>📈 Metrics Breakdown</h2>
            <div class="metrics-grid">
                {"".join(metrics_cards)}
            </div>
        </section>
        """

    def _build_results_section(self, report: EvaluationReport) -> str:
        """Build detailed results section.

        Args:
            report: Evaluation report

        Returns:
            HTML results section
        """
        # Group results for display
        self._group_results_by_metric(report.results)

        # Build results table
        table_rows = []
        for result in report.results[:100]:  # Limit to first 100 for performance
            status_icon = "✅" if result.error is None else "❌"
            status_class = "success" if result.error is None else "error"

            score_display = f"{result.score:.3f}" if result.score is not None else "N/A"
            error_display = (
                result.error[:100] + "..."
                if result.error and len(result.error) > 100
                else (result.error or "")
            )

            row_html = f"""
            <tr class="{status_class}">
                <td><span class="status-icon">{status_icon}</span></td>
                <td>{result.item_id}</td>
                <td>{result.metric}</td>
                <td>{result.provider}</td>
                <td class="score-cell">{score_display}</td>
                <td>{result.execution_time:.2f}s</td>
                <td>{self._format_cost(result.cost_estimate)}</td>
                <td class="error-cell" title="{result.error or ""}">{error_display}</td>
            </tr>
            """
            table_rows.append(row_html)

        results_table = f"""
        <div class="results-table-container">
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Status</th>
                        <th>Item ID</th>
                        <th>Metric</th>
                        <th>Provider</th>
                        <th>Score</th>
                        <th>Time</th>
                        <th>Cost</th>
                        <th>Error</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(table_rows)}
                </tbody>
            </table>
        </div>
        """

        if len(report.results) > 100:
            results_table += f"""
            <p class="results-note">
                Showing first 100 of {len(report.results)} results.
                Download the JSON report for complete data.
            </p>
            """

        return f"""
        <section class="results-section">
            <h2>📋 Detailed Results</h2>
            {results_table}
        </section>
        """

    def _build_performance_section(self, report: EvaluationReport) -> str:
        """Build performance analysis section.

        Args:
            report: Evaluation report

        Returns:
            HTML performance section
        """
        # Calculate performance metrics
        exec_times = [r.execution_time for r in report.results if r.execution_time > 0]

        if not exec_times:
            return ""

        exec_times.sort()
        n = len(exec_times)

        perf_stats = {
            "mean": sum(exec_times) / n,
            "median": exec_times[n // 2],
            "p95": exec_times[int(0.95 * n)] if n > 20 else max(exec_times),
            "min": min(exec_times),
            "max": max(exec_times),
        }

        return f"""
        <section class="performance-section">
            <h2>⚡ Performance Analysis</h2>
            <div class="perf-grid">
                <div class="perf-card">
                    <h3>Execution Times</h3>
                    <div class="perf-stats">
                        <div>Mean: {perf_stats["mean"]:.2f}s</div>
                        <div>Median: {perf_stats["median"]:.2f}s</div>
                        <div>P95: {perf_stats["p95"]:.2f}s</div>
                        <div>Range: {perf_stats["min"]:.2f}s - {perf_stats["max"]:.2f}s</div>
                    </div>
                </div>

                <div class="perf-card">
                    <h3>Provider Usage</h3>
                    <div class="perf-stats">
                        {self._build_provider_usage_list(report.execution_metadata.provider_usage)}
                    </div>
                </div>

                <div class="perf-card">
                    <h3>Cache Performance</h3>
                    <div class="perf-stats">
                        <div>Cache Hits: {report.execution_metadata.cache_hits:,}</div>
                        <div>Hit Rate: {self._format_percentage(report.execution_metadata.cache_hits / max(len(report.results), 1))}</div>
                    </div>
                </div>
            </div>
        </section>
        """

    def _build_provider_usage_list(self, provider_usage: Dict[str, int]) -> str:
        """Build HTML list of provider usage.

        Args:
            provider_usage: Provider usage statistics

        Returns:
            HTML list of provider usage
        """
        if not provider_usage:
            return "<div>No provider data</div>"

        total_usage = sum(provider_usage.values())
        items = []

        for provider, count in sorted(provider_usage.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_usage) * 100 if total_usage > 0 else 0
            items.append(f"<div>{provider}: {count:,} ({percentage:.1f}%)</div>")

        return "".join(items)

    def _build_footer_section(self, metadata: Dict[str, Any]) -> str:
        """Build footer section.

        Args:
            metadata: Report metadata

        Returns:
            HTML footer section
        """
        return f"""
        <footer class="report-footer">
            <div class="footer-content">
                <p>Generated by ULEI (Unified LLM Evaluation Interface) on {self._format_timestamp(metadata["generated_at"])}</p>
                <p>Report Format: HTML v1.0 | Execution Time: {self._format_duration(metadata["execution_time"])}</p>
            </div>
        </footer>
        """

    def _get_css_styles(self) -> str:
        """Get CSS styles for the HTML report.

        Returns:
            CSS styles as string
        """
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Header */
        .report-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }

        .report-header h1 {
            font-size: 2.5rem;
            margin-bottom: 20px;
            text-align: center;
        }

        .report-info {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 30px;
        }

        .info-item {
            text-align: center;
        }

        .info-item .label {
            display: block;
            font-size: 0.9rem;
            opacity: 0.9;
            margin-bottom: 5px;
        }

        .info-item .value {
            display: block;
            font-size: 1.2rem;
            font-weight: bold;
        }

        /* Sections */
        section {
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        section h2 {
            font-size: 1.8rem;
            margin-bottom: 25px;
            color: #2d3748;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }

        /* Summary Grid */
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }

        .summary-card {
            background: #f7fafc;
            padding: 25px;
            border-radius: 8px;
            text-align: center;
            border: 2px solid #e2e8f0;
        }

        .summary-card.pass {
            border-color: #48bb78;
            background: #f0fff4;
        }

        .summary-card.fail {
            border-color: #f56565;
            background: #fffafa;
        }

        .summary-card h3 {
            font-size: 1rem;
            color: #718096;
            margin-bottom: 10px;
        }

        .status-value, .metric-value {
            font-size: 2rem;
            font-weight: bold;
            margin: 10px 0;
        }

        .summary-card.pass .status-value {
            color: #48bb78;
        }

        .summary-card.fail .status-value {
            color: #f56565;
        }

        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }

        .metric-card {
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px;
            background: white;
        }

        .metric-card.pass {
            border-color: #48bb78;
        }

        .metric-card.fail {
            border-color: #f56565;
        }

        .metric-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .metric-header h3 {
            font-size: 1.2rem;
            color: #2d3748;
        }

        .status-badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
        }

        .status-badge.pass {
            background: #c6f6d5;
            color: #22543d;
        }

        .status-badge.fail {
            background: #fed7d7;
            color: #742a2a;
        }

        .status-badge.info {
            background: #bee3f8;
            color: #2a4365;
        }

        .metric-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }

        .stat {
            display: flex;
            justify-content: space-between;
        }

        .stat-label {
            color: #718096;
            font-size: 0.9rem;
        }

        .stat-value {
            font-weight: bold;
            color: #2d3748;
        }

        /* Results Table */
        .results-table-container {
            overflow-x: auto;
        }

        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

        .results-table th,
        .results-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }

        .results-table th {
            background: #f7fafc;
            font-weight: bold;
            color: #2d3748;
        }

        .results-table tr.success {
            background: #f0fff4;
        }

        .results-table tr.error {
            background: #fffafa;
        }

        .score-cell {
            text-align: right;
            font-family: monospace;
        }

        .error-cell {
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .status-icon {
            font-size: 1.2rem;
        }

        .results-note {
            margin-top: 15px;
            color: #718096;
            font-style: italic;
        }

        /* Performance Grid */
        .perf-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }

        .perf-card {
            background: #f7fafc;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }

        .perf-card h3 {
            margin-bottom: 15px;
            color: #2d3748;
        }

        .perf-stats div {
            margin-bottom: 8px;
            font-size: 0.9rem;
        }

        /* Footer */
        .report-footer {
            text-align: center;
            padding: 20px;
            color: #718096;
            font-size: 0.9rem;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }

            .report-header h1 {
                font-size: 2rem;
            }

            .report-info {
                gap: 15px;
            }

            .summary-grid,
            .metrics-grid {
                grid-template-columns: 1fr;
            }

            section {
                padding: 20px;
            }
        }
        """

    def _get_javascript(self) -> str:
        """Get JavaScript for interactive features.

        Returns:
            JavaScript code as string
        """
        return """
        // Add click-to-copy functionality for run ID
        document.addEventListener('DOMContentLoaded', function() {
            // Add tooltips and interactive features here if needed
            console.log('ULEI HTML Report loaded');
        });
        """
