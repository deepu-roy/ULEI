"""
CLI command for comparing evaluation reports.
"""

import json
import logging
from pathlib import Path

import click

from ulei.core.schemas import EvaluationReport
from ulei.reporters.comparison import ComparisonReporter
from ulei.utils.logging import configure_logging as setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.argument("baseline_report", type=click.Path(exists=True, path_type=Path))
@click.argument("comparison_report", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path.cwd(),
    help="Output directory for comparison reports",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "html", "both"], case_sensitive=False),
    default="both",
    help="Output format for comparison report",
)
@click.option(
    "--regression-threshold",
    type=float,
    default=5.0,
    help="Minimum percentage decline to flag as regression",
)
@click.option(
    "--significance-level",
    type=float,
    default=0.05,
    help="Statistical significance level for comparisons",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def compare(
    baseline_report: Path,
    comparison_report: Path,
    output_dir: Path,
    format: str,
    regression_threshold: float,
    significance_level: float,
    verbose: bool,
) -> None:
    """
    Compare two evaluation reports and generate statistical analysis.

    BASELINE_REPORT: Path to the baseline evaluation report (JSON)
    COMPARISON_REPORT: Path to the comparison evaluation report (JSON)

    This command performs statistical comparison between two evaluation runs,
    detecting improvements, regressions, and calculating significance tests.

    Examples:

    \b
    # Compare two reports with HTML output
    ulei compare baseline.json comparison.json --format html

    \b
    # Compare with custom regression threshold
    ulei compare baseline.json comparison.json --regression-threshold 10.0

    \b
    # Output to specific directory
    ulei compare baseline.json comparison.json -o ./reports/
    """
    # Setup logging
    setup_logging(verbose)

    try:
        logger.info(f"Starting comparison between {baseline_report} and {comparison_report}")

        # Load evaluation reports
        baseline_data = _load_report(baseline_report)
        comparison_data = _load_report(comparison_report)

        # Create reporter
        reporter = ComparisonReporter(output_dir)

        generated_files = []

        # Generate JSON report
        if format in ["json", "both"]:
            json_path = reporter.generate_comparison_report(baseline_data, comparison_data)
            generated_files.append(json_path)
            click.echo(f"✅ JSON comparison report: {json_path}")

        # Generate HTML report
        if format in ["html", "both"]:
            html_path = reporter.generate_html_comparison_report(baseline_data, comparison_data)
            generated_files.append(html_path)
            click.echo(f"✅ HTML comparison report: {html_path}")

        # Load and check for regressions
        if format in ["json", "both"]:
            with open(generated_files[0]) as f:
                report_data = json.load(f)

            regressions = report_data.get("detected_regressions", [])
            if regressions:
                click.echo(f"\n⚠️  {len(regressions)} regression(s) detected:")
                for regression in regressions:
                    metric = regression["metric_name"]
                    delta = regression["delta_percent"]
                    click.echo(f"  - {metric}: {delta:.1f}% decline")

                # Exit with code 2 to indicate regressions found
                raise click.ClickException("Performance regressions detected")
            else:
                click.echo("\n✅ No significant regressions detected")

        # Summary
        click.echo(f"\nComparison complete. Reports generated in: {output_dir}")

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise click.ClickException(f"Comparison failed: {e}")


@click.command()
@click.argument("reports", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path.cwd(),
    help="Output directory for trend analysis report",
)
@click.option(
    "--time-column", default="timestamp", help="Column name for timestamps in report metadata"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def trend(reports: tuple[Path, ...], output_dir: Path, time_column: str, verbose: bool) -> None:
    """
    Analyze performance trends across multiple evaluation reports.

    REPORTS: Paths to evaluation reports (JSON) in chronological order

    This command analyzes performance trends over time, detecting
    improving, declining, or stable metrics with statistical significance.

    Examples:

    \b
    # Analyze trend across multiple reports
    ulei trend report1.json report2.json report3.json

    \b
    # Output to specific directory
    ulei trend *.json -o ./trends/
    """
    # Setup logging
    setup_logging(verbose)

    try:
        if len(reports) < 3:
            raise click.ClickException("Trend analysis requires at least 3 reports")

        logger.info(f"Analyzing trends across {len(reports)} reports")

        # Load reports with timestamps
        reports_with_time = []
        for i, report_path in enumerate(reports):
            report_data = _load_report(report_path)
            # Use index as timestamp if no timestamp in metadata
            timestamp = i  # This could be enhanced to parse actual timestamps
            reports_with_time.append((timestamp, report_data))

        # Create reporter and generate trend analysis
        reporter = ComparisonReporter(output_dir)
        trend_path = reporter.generate_trend_analysis_report(reports_with_time)

        click.echo(f"✅ Trend analysis report: {trend_path}")

        # Load and summarize trends
        with open(trend_path) as f:
            trend_data = json.load(f)

        summary = trend_data.get("summary", {})
        improving = summary.get("improving_trends", 0)
        declining = summary.get("declining_trends", 0)
        stable = summary.get("stable_metrics", 0)

        click.echo("\n📈 Trend Summary:")
        click.echo(f"  Improving: {improving}")
        click.echo(f"  Declining: {declining}")
        click.echo(f"  Stable: {stable}")

        if declining > 0:
            click.echo(f"\n⚠️  {declining} metric(s) showing declining trends")

    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        raise click.ClickException(f"Trend analysis failed: {e}")


def _load_report(report_path: Path) -> EvaluationReport:
    """Load an evaluation report from JSON file."""
    try:
        with open(report_path) as f:
            data = json.load(f)

        # Convert to EvaluationReport object
        return EvaluationReport(**data)

    except Exception as e:
        raise click.ClickException(f"Failed to load report {report_path}: {e}")
