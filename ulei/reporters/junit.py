"""
JUnit XML reporter for CI/CD system integration.

Generates JUnit-compatible XML reports that can be consumed by CI systems
like Jenkins, GitHub Actions, GitLab CI, etc.
"""

import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List

from ulei.core.schemas import EvaluationReport, MetricResult
from ulei.reporters.base import BaseReporter


class JUnitReporter(BaseReporter):
    """Reporter for generating JUnit XML format evaluation reports."""

    @property
    def format_name(self) -> str:
        """Return report format name."""
        return "junit"

    @property
    def file_extension(self) -> str:
        """Return file extension for JUnit reports."""
        return ".xml"

    def _generate_report_content(self, report: EvaluationReport, output_path: str) -> None:
        """Generate JUnit XML report content.

        Args:
            report: Evaluation report data
            output_path: Path where to save the report
        """
        # Validate report
        self._validate_report(report)

        # Generate JUnit XML
        xml_content = self._build_junit_xml(report)

        # Write XML file with proper formatting
        self._write_xml_file(xml_content, output_path)

    def _build_junit_xml(self, report: EvaluationReport) -> ET.Element:
        """Build JUnit XML structure.

        Args:
            report: Evaluation report

        Returns:
            XML root element
        """
        # Create root testsuites element
        testsuites = ET.Element("testsuites")

        # Group results by metric for test organization
        results_by_metric = self._group_results_by_metric(report.results)

        # Overall statistics
        total_tests = len(report.results)
        total_failures = len([r for r in report.results if r.error is not None])
        total_time = report.execution_metadata.total_execution_time

        # Set testsuites attributes
        testsuites.set("name", f"ULEI Evaluation: {report.suite_name}")
        testsuites.set("tests", str(total_tests))
        testsuites.set("failures", str(total_failures))
        testsuites.set("errors", "0")  # We treat all issues as failures
        testsuites.set("time", f"{total_time:.3f}")
        testsuites.set("timestamp", datetime.utcnow().isoformat())

        # Create a testsuite for each metric
        for metric_name, metric_results in results_by_metric.items():
            testsuite = self._create_testsuite(metric_name, metric_results, report)
            testsuites.append(testsuite)

        # Add threshold testsuite if thresholds are configured
        if report.threshold_status:
            threshold_suite = self._create_threshold_testsuite(report)
            testsuites.append(threshold_suite)

        return testsuites

    def _create_testsuite(
        self, metric_name: str, results: List[MetricResult], report: EvaluationReport
    ) -> ET.Element:
        """Create a testsuite element for a specific metric.

        Args:
            metric_name: Name of the metric
            results: Results for this metric
            report: Full evaluation report

        Returns:
            XML testsuite element
        """
        testsuite = ET.Element("testsuite")

        # Calculate suite statistics
        failures = len([r for r in results if r.error is not None])
        total_time = sum(r.execution_time for r in results if r.execution_time is not None)

        # Set testsuite attributes
        testsuite.set("name", f"{metric_name}")
        testsuite.set("tests", str(len(results)))
        testsuite.set("failures", str(failures))
        testsuite.set("errors", "0")
        testsuite.set("time", f"{total_time:.3f}")
        testsuite.set("package", f"ulei.metrics.{metric_name}")

        # Add properties
        properties = ET.SubElement(testsuite, "properties")

        # Add metric-specific properties
        self._add_property(properties, "metric.name", metric_name)
        self._add_property(
            properties, "metric.provider", results[0].provider if results else "unknown"
        )

        if results:
            scores = [r.score for r in results if r.score is not None]
            if scores:
                self._add_property(
                    properties, "metric.mean_score", f"{sum(scores) / len(scores):.3f}"
                )
                self._add_property(properties, "metric.min_score", f"{min(scores):.3f}")
                self._add_property(properties, "metric.max_score", f"{max(scores):.3f}")

        # Add threshold info if available
        if metric_name in report.threshold_status:
            threshold_passed = report.threshold_status[metric_name]
            self._add_property(properties, "threshold.configured", "true")
            self._add_property(properties, "threshold.passed", str(threshold_passed).lower())

        # Create testcase for each evaluation
        for result in results:
            testcase = self._create_testcase(result)
            testsuite.append(testcase)

        return testsuite

    def _create_testcase(self, result: MetricResult) -> ET.Element:
        """Create a testcase element for a single evaluation result.

        Args:
            result: Metric result

        Returns:
            XML testcase element
        """
        testcase = ET.Element("testcase")

        # Set testcase attributes
        testcase.set("name", f"{result.item_id}")
        testcase.set("classname", f"ulei.metrics.{result.metric}")
        if result.execution_time is not None:
            testcase.set("time", f"{result.execution_time:.3f}")

        # Add failure if there's an error
        if result.error is not None:
            failure = ET.SubElement(testcase, "failure")
            failure.set("message", "Evaluation failed")
            failure.set("type", "EvaluationError")
            failure.text = result.error

        # Add system-out with details
        system_out = ET.SubElement(testcase, "system-out")
        output_lines = [
            f"Item ID: {result.item_id}",
            f"Metric: {result.metric}",
            f"Provider: {result.provider}",
        ]

        if result.score is not None:
            output_lines.append(f"Score: {result.score:.6f}")

        if result.execution_time is not None:
            output_lines.append(f"Execution Time: {result.execution_time:.3f}s")

        if result.cost_estimate is not None:
            output_lines.append(f"Cost Estimate: ${result.cost_estimate:.6f}")

        system_out.text = "\n".join(output_lines)

        return testcase

    def _create_threshold_testsuite(self, report: EvaluationReport) -> ET.Element:
        """Create a testsuite for threshold compliance checks.

        Args:
            report: Evaluation report

        Returns:
            XML testsuite element for thresholds
        """
        testsuite = ET.Element("testsuite")

        # Calculate threshold statistics
        total_thresholds = len(report.threshold_status)
        failed_thresholds = len(
            [status for status in report.threshold_status.values() if not status]
        )

        testsuite.set("name", "Threshold Compliance")
        testsuite.set("tests", str(total_thresholds))
        testsuite.set("failures", str(failed_thresholds))
        testsuite.set("errors", "0")
        testsuite.set("time", "0.0")  # Thresholds don't have execution time
        testsuite.set("package", "ulei.thresholds")

        # Add properties
        properties = ET.SubElement(testsuite, "properties")
        overall_pass = all(report.threshold_status.values())
        self._add_property(properties, "overall.passed", str(overall_pass).lower())
        self._add_property(properties, "total.thresholds", str(total_thresholds))
        self._add_property(properties, "failed.thresholds", str(failed_thresholds))

        # Create testcase for each threshold
        for metric_name, passed in report.threshold_status.items():
            testcase = ET.Element("testcase")
            testcase.set("name", f"threshold_{metric_name}")
            testcase.set("classname", "ulei.thresholds.ThresholdCheck")
            testcase.set("time", "0.0")

            if not passed:
                failure = ET.SubElement(testcase, "failure")
                failure.set("message", f"Threshold not met for {metric_name}")
                failure.set("type", "ThresholdError")

                # Add details about the failure
                results_by_metric = self._group_results_by_metric(report.results)
                if metric_name in results_by_metric:
                    metric_results = results_by_metric[metric_name]
                    scores = [r.score for r in metric_results if r.score is not None]
                    if scores:
                        actual_score = sum(scores) / len(scores)
                        failure.text = f"Mean score {actual_score:.3f} did not meet threshold"

            # Add system-out with threshold details
            system_out = ET.SubElement(testcase, "system-out")
            system_out.text = f"Metric: {metric_name}\nPassed: {passed}"

            testsuite.append(testcase)

        return testsuite

    def _add_property(self, properties: ET.Element, name: str, value: str) -> None:
        """Add a property to the properties element."""
        prop = ET.SubElement(properties, "property")
        prop.set("name", name)
        prop.set("value", value)

    def _write_xml_file(self, root: ET.Element, output_path: str) -> None:
        """Write XML content to file with proper formatting.

        Args:
            root: XML root element
            output_path: Output file path
        """
        # Create tree and add XML declaration
        tree = ET.ElementTree(root)

        # Write to file with UTF-8 encoding
        with open(output_path, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)

    def _group_results_by_metric(
        self, results: List[MetricResult]
    ) -> Dict[str, List[MetricResult]]:
        """Group results by metric name."""
        grouped: Dict[str, List[MetricResult]] = {}
        for result in results:
            metric_name = result.metric
            if metric_name not in grouped:
                grouped[metric_name] = []
            grouped[metric_name].append(result)
        return grouped


# Convenience function
def generate_junit_report(report: EvaluationReport, output_path: str) -> None:
    """Generate JUnit XML report.

    Args:
        report: Evaluation report
        output_path: Output file path
    """
    reporter = JUnitReporter()
    reporter.generate_report(report, output_path)
