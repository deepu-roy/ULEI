"""
Statistical analysis utilities for evaluation comparison.

Provides statistical tests, confidence intervals, and significance testing
for comparing evaluation runs and detecting performance changes.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from ulei.core.schemas import MetricResult

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of statistical comparison between two metric sets."""

    metric_name: str
    baseline_mean: float
    comparison_mean: float
    delta: float
    delta_percent: float
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    sample_sizes: Tuple[int, int]
    effect_size: Optional[float] = None
    test_type: str = "t-test"


@dataclass
class TrendAnalysis:
    """Result of trend analysis over multiple evaluation runs."""

    metric_name: str
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    is_significant_trend: bool
    trend_direction: str  # "improving", "declining", "stable"
    confidence_interval: Tuple[float, float]


def extract_metric_scores(results: List[MetricResult], metric_name: str) -> List[float]:
    """
    Extract scores for a specific metric from evaluation results.

    Args:
        results: List of metric results
        metric_name: Name of the metric to extract

    Returns:
        List of scores for the metric (excluding errors/None values)
    """
    scores = []
    for result in results:
        if result.metric == metric_name and result.score is not None and result.error is None:
            scores.append(result.score)

    if not scores:
        logger.warning(f"No valid scores found for metric: {metric_name}")

    return scores


def compare_metrics(
    baseline_results: List[MetricResult],
    comparison_results: List[MetricResult],
    metric_name: str,
    alpha: float = 0.05,
    min_sample_size: int = 3,
) -> Optional[ComparisonResult]:
    """
    Compare metrics between two evaluation runs using statistical tests.

    Args:
        baseline_results: Results from baseline evaluation
        comparison_results: Results from comparison evaluation
        metric_name: Name of metric to compare
        alpha: Significance level for statistical tests
        min_sample_size: Minimum sample size required for testing

    Returns:
        ComparisonResult or None if insufficient data
    """
    baseline_scores = extract_metric_scores(baseline_results, metric_name)
    comparison_scores = extract_metric_scores(comparison_results, metric_name)

    if len(baseline_scores) < min_sample_size or len(comparison_scores) < min_sample_size:
        logger.warning(
            f"Insufficient data for {metric_name}: baseline={len(baseline_scores)}, "
            f"comparison={len(comparison_scores)}, min_required={min_sample_size}"
        )
        return None

    baseline_mean = np.mean(baseline_scores)
    comparison_mean = np.mean(comparison_scores)
    delta = comparison_mean - baseline_mean
    delta_percent = (delta / baseline_mean * 100) if baseline_mean != 0 else 0

    # Perform two-sample t-test
    t_stat, p_value = stats.ttest_ind(comparison_scores, baseline_scores)
    is_significant = p_value < alpha

    # Calculate confidence interval for the difference
    pooled_std = np.sqrt(
        (
            (len(baseline_scores) - 1) * np.var(baseline_scores, ddof=1)
            + (len(comparison_scores) - 1) * np.var(comparison_scores, ddof=1)
        )
        / (len(baseline_scores) + len(comparison_scores) - 2)
    )

    se_diff = pooled_std * np.sqrt(1 / len(baseline_scores) + 1 / len(comparison_scores))
    df = len(baseline_scores) + len(comparison_scores) - 2
    t_critical = stats.t.ppf(1 - alpha / 2, df)

    ci_lower = delta - t_critical * se_diff
    ci_upper = delta + t_critical * se_diff

    # Calculate Cohen's d for effect size
    effect_size = delta / pooled_std if pooled_std > 0 else None

    logger.info(
        f"Comparison for {metric_name}: "
        f"baseline={baseline_mean:.3f}, comparison={comparison_mean:.3f}, "
        f"delta={delta:.3f} ({delta_percent:.1f}%), p-value={p_value:.4f}, "
        f"significant={is_significant}"
    )

    return ComparisonResult(
        metric_name=metric_name,
        baseline_mean=baseline_mean,
        comparison_mean=comparison_mean,
        delta=delta,
        delta_percent=delta_percent,
        p_value=p_value,
        is_significant=is_significant,
        confidence_interval=(ci_lower, ci_upper),
        sample_sizes=(len(baseline_scores), len(comparison_scores)),
        effect_size=effect_size,
        test_type="t-test",
    )


def analyze_trend(
    results_over_time: List[Tuple[float, List[MetricResult]]], metric_name: str, alpha: float = 0.05
) -> Optional[TrendAnalysis]:
    """
    Analyze trend in metric performance over time.

    Args:
        results_over_time: List of (timestamp, results) tuples sorted by time
        metric_name: Name of metric to analyze
        alpha: Significance level for trend test

    Returns:
        TrendAnalysis or None if insufficient data
    """
    if len(results_over_time) < 3:
        logger.warning(f"Insufficient time points for trend analysis: {len(results_over_time)}")
        return None

    # Extract mean scores for each time point
    time_points = []
    mean_scores = []

    for timestamp, results in results_over_time:
        scores = extract_metric_scores(results, metric_name)
        if scores:
            time_points.append(timestamp)
            mean_scores.append(np.mean(scores))

    if len(mean_scores) < 3:
        logger.warning(f"Insufficient valid data points for trend analysis: {len(mean_scores)}")
        return None

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, mean_scores)
    r_squared = r_value**2

    # Determine trend direction
    if p_value < alpha:
        if slope > 0:
            trend_direction = "improving"
        else:
            trend_direction = "declining"
    else:
        trend_direction = "stable"

    # Calculate confidence interval for slope
    df = len(time_points) - 2
    t_critical = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = slope - t_critical * std_err
    ci_upper = slope + t_critical * std_err

    logger.info(
        f"Trend analysis for {metric_name}: "
        f"slope={slope:.6f}, r²={r_squared:.3f}, p-value={p_value:.4f}, "
        f"direction={trend_direction}"
    )

    return TrendAnalysis(
        metric_name=metric_name,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        p_value=p_value,
        is_significant_trend=p_value < alpha,
        trend_direction=trend_direction,
        confidence_interval=(ci_lower, ci_upper),
    )


def calculate_confidence_intervals(
    results: List[MetricResult], metric_name: str, confidence_level: float = 0.95
) -> Optional[Tuple[float, float, float]]:
    """
    Calculate confidence interval for a metric.

    Args:
        results: List of metric results
        metric_name: Name of metric to analyze
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        Tuple of (mean, lower_bound, upper_bound) or None if insufficient data
    """
    scores = extract_metric_scores(results, metric_name)

    if len(scores) < 2:
        logger.warning(f"Insufficient data for confidence interval: {len(scores)} samples")
        return None

    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=1)
    n = len(scores)

    # Calculate confidence interval using t-distribution
    alpha = 1 - confidence_level
    df = n - 1
    t_critical = stats.t.ppf(1 - alpha / 2, df)

    margin_error = t_critical * (std_score / np.sqrt(n))
    lower_bound = mean_score - margin_error
    upper_bound = mean_score + margin_error

    logger.debug(
        f"Confidence interval for {metric_name}: "
        f"{mean_score:.3f} ± {margin_error:.3f} "
        f"[{lower_bound:.3f}, {upper_bound:.3f}]"
    )

    return (mean_score, lower_bound, upper_bound)


def detect_regressions(
    baseline_results: List[MetricResult],
    comparison_results: List[MetricResult],
    metrics: List[str],
    regression_threshold: float = 0.05,
    significance_level: float = 0.05,
) -> Dict[str, ComparisonResult]:
    """
    Detect performance regressions across multiple metrics.

    Args:
        baseline_results: Results from baseline evaluation
        comparison_results: Results from comparison evaluation
        metrics: List of metric names to check
        regression_threshold: Minimum percentage decline to consider regression
        significance_level: Statistical significance level

    Returns:
        Dictionary mapping metric names to comparison results for regressions
    """
    regressions = {}

    for metric_name in metrics:
        comparison = compare_metrics(
            baseline_results, comparison_results, metric_name, alpha=significance_level
        )

        if comparison is None:
            continue

        # Check for significant decline
        if comparison.is_significant and comparison.delta_percent < -regression_threshold:
            regressions[metric_name] = comparison
            logger.warning(
                f"Regression detected in {metric_name}: "
                f"{comparison.delta_percent:.1f}% decline (p={comparison.p_value:.4f})"
            )

    return regressions


def summarize_comparison(comparisons: List[ComparisonResult]) -> Dict[str, Any]:
    """
    Summarize multiple metric comparisons into high-level insights.

    Args:
        comparisons: List of comparison results

    Returns:
        Summary dictionary with overall statistics
    """
    if not comparisons:
        return {}

    significant_improvements = [c for c in comparisons if c.is_significant and c.delta > 0]
    significant_regressions = [c for c in comparisons if c.is_significant and c.delta < 0]

    total_metrics = len(comparisons)
    improved_count = len(significant_improvements)
    regressed_count = len(significant_regressions)

    # Calculate overall effect size (average of absolute effect sizes)
    effect_sizes = [abs(c.effect_size) for c in comparisons if c.effect_size is not None]
    avg_effect_size = np.mean(effect_sizes) if effect_sizes else None

    summary = {
        "total_metrics": total_metrics,
        "significant_changes": improved_count + regressed_count,
        "improvements": improved_count,
        "regressions": regressed_count,
        "stable_metrics": total_metrics - improved_count - regressed_count,
        "average_effect_size": avg_effect_size,
        "largest_improvement": max(significant_improvements, key=lambda x: x.delta_percent)
        if significant_improvements
        else None,
        "largest_regression": min(significant_regressions, key=lambda x: x.delta_percent)
        if significant_regressions
        else None,
    }

    logger.info(
        f"Comparison summary: {improved_count} improvements, {regressed_count} regressions, "
        f"{summary['stable_metrics']} stable metrics out of {total_metrics} total"
    )

    return summary
