"""
CLI run command for executing evaluation suites.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml

from ulei.core.evaluator import Evaluator as EvaluationEngine
from ulei.core.schemas import DatasetItem, EvaluationSuite
from ulei.utils.config import ConfigLoader


@click.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default="./reports",
    help="Directory to save evaluation reports",
)
@click.option(
    "--format",
    "-f",
    "output_formats",
    type=click.Choice(["json", "html", "all"], case_sensitive=False),
    multiple=True,
    default=["json"],
    help="Output format(s) for reports (can specify multiple)",
)
@click.option(
    "--parallel-workers",
    "-p",
    type=click.IntRange(1, 50),
    default=4,
    help="Number of parallel evaluation workers",
)
@click.option("--cache/--no-cache", default=True, help="Enable or disable response caching")
@click.option(
    "--cache-ttl",
    type=click.IntRange(0, 86400),
    default=3600,
    help="Cache TTL in seconds (0 = no expiry)",
)
@click.option(
    "--retry-attempts",
    type=click.IntRange(0, 10),
    default=3,
    help="Number of retry attempts for failed evaluations",
)
@click.option(
    "--timeout", type=click.IntRange(1, 300), default=60, help="Timeout per evaluation in seconds"
)
@click.option(
    "--dry-run", is_flag=True, help="Validate configuration and show execution plan without running"
)
@click.option("--verbose", "-v", count=True, help="Increase verbosity level (-v, -vv, -vvv)")
@click.option(
    "--fail-on-threshold",
    is_flag=True,
    default=True,
    help="Exit with non-zero code if thresholds are not met (default: enabled)",
)
@click.option(
    "--no-fail-on-threshold",
    "fail_on_threshold",
    flag_value=False,
    help="Do not exit with non-zero code on threshold failures",
)
@click.option(
    "--threshold-strategy",
    type=click.Choice(["mean", "median", "min", "all"], case_sensitive=False),
    default="mean",
    help="Strategy for aggregating scores before threshold checking",
)
def run(
    config_path: Path,
    output_dir: Path,
    output_formats: List[str],
    parallel_workers: int,
    cache: bool,
    cache_ttl: int,
    retry_attempts: int,
    timeout: int,
    dry_run: bool,
    verbose: int,
    fail_on_threshold: bool,
    threshold_strategy: str,
) -> None:
    """
    Run an evaluation suite from a configuration file.

    CONFIG_PATH: Path to the YAML configuration file containing evaluation suite definition.

    Example:
        ulei run my_eval_suite.yaml --output-dir ./reports --format json html
    """
    # Set up logging level based on verbosity
    log_level = _get_log_level(verbose)

    try:
        # Load and validate configuration
        click.echo(f"📋 Loading configuration from {config_path}")
        config_loader = ConfigLoader()
        suite = config_loader.load_suite(str(config_path))

        # Load dataset items
        items = suite.get_items()
        click.echo(
            f"✅ Loaded suite '{suite.name}' with {len(items)} items"
        )  # Validate output formats
        if "all" in output_formats:
            output_formats = ["json", "html"]

        # Show execution plan if dry run
        if dry_run:
            _show_execution_plan(suite, items, output_dir, output_formats, parallel_workers)
            return

        # Run the evaluation
        result = asyncio.run(
            _run_evaluation(
                suite=suite,
                items=items,
                output_dir=output_dir,
                output_formats=list(output_formats),
                parallel_workers=parallel_workers,
                use_cache=cache,
                cache_ttl=cache_ttl if cache_ttl > 0 else None,
                retry_attempts=retry_attempts,
                timeout=timeout,
                log_level=log_level,
                threshold_strategy=threshold_strategy,
            )
        )

        # Report results and handle exit codes
        exit_code = _report_completion(result, output_dir, output_formats, fail_on_threshold)

        if exit_code != 0:
            raise SystemExit(exit_code)

    except SystemExit:
        # Let SystemExit pass through (for proper exit codes)
        raise
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose > 0:
            import traceback

            click.echo(traceback.format_exc(), err=True)
        raise SystemExit(1)


async def _run_evaluation(
    suite: EvaluationSuite,
    items: List[DatasetItem],
    output_dir: Path,
    output_formats: List[str],
    parallel_workers: int,
    use_cache: bool,
    cache_ttl: Optional[int],
    retry_attempts: int,
    timeout: int,
    log_level: str,
    threshold_strategy: str,
) -> Dict[str, Any]:
    """
    Execute the evaluation suite with the given parameters.

    Args:
        suite: Evaluation suite to execute
        items: Dataset items to evaluate
        output_dir: Directory for output files
        output_formats: List of output formats to generate
        parallel_workers: Number of parallel workers
        use_cache: Whether to use caching
        cache_ttl: Cache TTL in seconds
        retry_attempts: Number of retry attempts
        timeout: Timeout per evaluation
        log_level: Logging level

    Returns:
        Evaluation results summary
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize orchestrator
    click.echo("🚀 Starting evaluation...")

    orchestrator_config = {
        "parallel_workers": parallel_workers,
        "cache_enabled": use_cache,
        "cache_ttl": cache_ttl,
        "retry_attempts": retry_attempts,
        "timeout": timeout,
        "log_level": log_level,
    }

    orchestrator = EvaluationEngine(config=orchestrator_config)

    # Execute evaluation
    with click.progressbar(
        length=len(items) * len(suite.metrics), label="Running evaluations"
    ) as bar:

        def progress_callback(completed: int, total: int) -> None:
            """Update progress bar."""
            bar.update(completed - bar.pos)

        # Run evaluation with the loaded items
        report = await orchestrator.run_evaluation(suite, items)

    # Generate reports
    click.echo("📄 Generating reports...")
    report_paths = []

    for format_type in output_formats:
        if format_type == "json":
            from ulei.reporters.json import JSONReporter

            reporter = JSONReporter()
        elif format_type == "html":
            from ulei.reporters.html import HTMLReporter

            reporter = HTMLReporter()  # type: ignore
        else:
            click.echo(f"⚠️  Unknown format: {format_type}", err=True)
            continue

        report_path = (
            output_dir / f"{suite.name}_{report.run_id}.{reporter.file_extension.lstrip('.')}"
        )
        reporter.generate_report(report, str(report_path))
        report_paths.append(report_path)
        click.echo(f"   ✅ {format_type.upper()}: {report_path}")

    # Determine overall pass status and collect failure reasons
    overall_pass = True
    threshold_failures = []

    if suite.thresholds and report.threshold_status:
        overall_pass = all(report.threshold_status.values())

        if not overall_pass:
            # Import evaluator to get failure reasons
            from ulei.core.evaluator import Evaluator

            evaluator = Evaluator()
            threshold_failures = evaluator.calculate_failure_reasons(
                report.threshold_status, suite.thresholds, report.results
            )

    return {
        "suite_name": suite.name,
        "run_id": report.run_id,
        "total_items": len(items),
        "total_metrics": len(suite.metrics),
        "successful_evaluations": report.execution_metadata.successful_evaluations,
        "total_evaluations": len(report.results),
        "execution_time": report.execution_metadata.total_execution_time,
        "report_paths": [str(p) for p in report_paths],
        "overall_pass": overall_pass,
        "threshold_failures": threshold_failures,
    }


def _show_execution_plan(
    suite: EvaluationSuite,
    items: List[DatasetItem],
    output_dir: Path,
    output_formats: List[str],
    parallel_workers: int,
) -> None:
    """
    Display execution plan for dry run.

    Args:
        suite: Evaluation suite
        items: Dataset items
        output_dir: Output directory
        output_formats: Output formats
        parallel_workers: Number of parallel workers
    """
    click.echo("\n🔍 Execution Plan (Dry Run)")
    click.echo("=" * 50)

    click.echo(f"Suite Name: {suite.name}")
    click.echo(f"Items: {len(items)}")
    click.echo(f"Metrics: {len(suite.metrics)} ({', '.join([m.name for m in suite.metrics])})")
    click.echo(f"Total Evaluations: {len(items) * len(suite.metrics)}")
    click.echo(f"Parallel Workers: {parallel_workers}")
    click.echo(f"Output Directory: {output_dir}")
    click.echo(f"Output Formats: {', '.join(output_formats)}")

    if suite.thresholds:
        click.echo(f"Thresholds: {len(suite.thresholds)} configured")
        for metric, threshold in suite.thresholds.items():
            click.echo(f"  - {metric}: {threshold}")
    else:
        click.echo("Thresholds: None configured")

    # Provider breakdown
    provider_counts: Dict[str, int] = {}
    for metric_spec in suite.metrics:
        provider = getattr(metric_spec, "provider", None) or "auto"
        provider_counts[provider] = provider_counts.get(provider, 0) + 1

    click.echo("Provider Usage:")
    for provider, count in provider_counts.items():
        click.echo(f"  - {provider}: {count} metrics")

    # Estimated outputs
    click.echo("\nEstimated Output Files:")
    for format_type in output_formats:
        extension = "json" if format_type == "json" else "html"
        filename = f"{suite.name}_<run_id>.{extension}"
        click.echo(f"  - {output_dir / filename}")

    click.echo("\n✅ Configuration is valid. Use --dry-run=false to execute.")


def _report_completion(
    result: Dict[str, Any],
    output_dir: Path,
    output_formats: List[str],
    fail_on_threshold: bool = True,
) -> int:
    """
    Report evaluation completion summary and return exit code.

    Args:
        result: Evaluation results summary
        output_dir: Output directory
        output_formats: Output formats
        fail_on_threshold: Whether to return non-zero exit code on threshold failure

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    click.echo("\n🎉 Evaluation Complete!")
    click.echo("=" * 50)

    # Summary statistics
    success_rate = (result["successful_evaluations"] / result["total_evaluations"]) * 100

    click.echo(f"Suite: {result['suite_name']}")
    click.echo(f"Run ID: {result['run_id']}")
    click.echo(
        f"Success Rate: {success_rate:.1f}% ({result['successful_evaluations']}/{result['total_evaluations']})"
    )
    click.echo(f"Execution Time: {result['execution_time']:.2f}s")

    # Overall status and exit code determination
    overall_pass = result.get("overall_pass", True)
    threshold_failures = result.get("threshold_failures", [])

    if overall_pass:
        click.echo("Status: ✅ PASS - All thresholds met")
        exit_code = 0
    else:
        click.echo("Status: ❌ FAIL - Some thresholds not met")

        if threshold_failures:
            click.echo("\nThreshold Failures:")
            for failure in threshold_failures:
                click.echo(f"  ❌ {failure}")

        # Determine exit code
        if fail_on_threshold:
            exit_code = 2  # Use 2 to distinguish from general errors (1)
            click.echo(f"\n🚨 Exiting with code {exit_code} due to threshold failures")
        else:
            exit_code = 0
            click.echo("\n⚠️  Threshold failures detected but --no-fail-on-threshold specified")

    # Report files
    click.echo("\nGenerated Reports:")
    for report_path in result["report_paths"]:
        click.echo(f"  📄 {report_path}")

    # Next steps
    click.echo("\nNext Steps:")
    if "html" in output_formats:
        html_reports = [p for p in result["report_paths"] if p.endswith(".html")]
        if html_reports:
            click.echo(f"  👀 Open {html_reports[0]} in your browser")

    if not overall_pass:
        click.echo("  🔍 Review failing metrics in the detailed reports")
        click.echo("  ⚙️  Adjust thresholds or improve system performance")
        click.echo("  🔧 Use --no-fail-on-threshold to suppress exit code")

    click.echo(f"  🔄 Re-run with: ulei run <config> --output-dir {output_dir}")

    return exit_code


def _get_log_level(verbose: int) -> str:
    """
    Convert verbosity count to log level.

    Args:
        verbose: Verbosity level (0-3)

    Returns:
        Log level string
    """
    if verbose == 0:
        return "WARNING"
    elif verbose == 1:
        return "INFO"
    elif verbose == 2:
        return "DEBUG"
    else:
        return "DEBUG"  # Max verbosity


def _parse_config_file(config_path: Path) -> Dict[str, Any]:
    """
    Parse YAML configuration file with error handling.

    Args:
        config_path: Path to config file

    Returns:
        Parsed configuration dictionary

    Raises:
        click.ClickException: If file cannot be parsed
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                config = yaml.safe_load(f)
                if not isinstance(config, dict):
                    raise click.ClickException(
                        f"Configuration file must contain a dictionary, got {type(config).__name__}"
                    )
                return config
            elif config_path.suffix.lower() == ".json":
                config = json.load(f)
                if not isinstance(config, dict):
                    raise click.ClickException(
                        f"Configuration file must contain a dictionary, got {type(config).__name__}"
                    )
                return config
            else:
                raise click.ClickException(
                    f"Unsupported config file format: {config_path.suffix}. "
                    "Use .yaml, .yml, or .json"
                )
    except yaml.YAMLError as e:
        raise click.ClickException(f"Invalid YAML in {config_path}: {e}")
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON in {config_path}: {e}")
    except Exception as e:
        raise click.ClickException(f"Error reading {config_path}: {e}")


if __name__ == "__main__":
    run()
