"""
Metric validation framework for ULEI custom metrics.

Provides comprehensive validation for custom metrics including schema validation,
runtime checks, performance monitoring, and error handling.
"""

import asyncio
import inspect
import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import jsonschema
from pydantic import ValidationError

from ulei.core.custom_metrics import (
    BaseMetric,
    MetricComplexity,
    MetricExecutionContext,
    MetricExecutionMode,
    MetricExecutionResult,
    MetricParameter,
    MetricType,
)

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""

    STRICT = "strict"  # Fail on any validation error
    MODERATE = "moderate"  # Warn on minor issues, fail on major ones
    LENIENT = "lenient"  # Warn on all issues, allow execution


class ValidationCategory(Enum):
    """Categories of validation checks."""

    SCHEMA = "schema"  # Configuration schema validation
    PARAMETER = "parameter"  # Parameter validation
    IMPLEMENTATION = "implementation"  # Implementation checks
    RUNTIME = "runtime"  # Runtime behavior validation
    PERFORMANCE = "performance"  # Performance validation
    SECURITY = "security"  # Security validation


@dataclass
class ValidationIssue:
    """Represents a validation issue."""

    category: ValidationCategory
    severity: str  # "error", "warning", "info"
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None
    location: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of metric validation."""

    metric_name: str
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    validation_time_ms: Optional[float] = None

    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(issue.severity == "error" for issue in self.issues)

    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(issue.severity == "warning" for issue in self.issues)

    def get_issues_by_category(self, category: ValidationCategory) -> List[ValidationIssue]:
        """Get issues filtered by category."""
        return [issue for issue in self.issues if issue.category == category]

    def get_issues_by_severity(self, severity: str) -> List[ValidationIssue]:
        """Get issues filtered by severity."""
        return [issue for issue in self.issues if issue.severity == severity]


class MetricValidator:
    """Comprehensive validator for custom metrics."""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        """
        Initialize metric validator.

        Args:
            validation_level: Validation strictness level
        """
        self.validation_level = validation_level
        self.schema_cache: Dict[str, Dict[str, Any]] = {}

        # Load built-in schemas
        self._load_builtin_schemas()

    def _load_builtin_schemas(self):
        """Load built-in validation schemas."""
        # Output schemas for common metric types
        self.schema_cache.update(
            {
                "accuracy": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "f1_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "bleu": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "rouge": {
                    "type": "object",
                    "properties": {
                        "rouge-1": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rouge-2": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rouge-l": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                },
                "semantic_similarity": {"type": "number", "minimum": -1.0, "maximum": 1.0},
            }
        )

    async def validate_metric(
        self, metric: BaseMetric, test_contexts: Optional[List[MetricExecutionContext]] = None
    ) -> ValidationResult:
        """
        Perform comprehensive metric validation.

        Args:
            metric: Metric to validate
            test_contexts: Optional test contexts for runtime validation

        Returns:
            Validation result
        """
        start_time = time.time()
        issues = []

        # Schema validation
        schema_issues = await self._validate_schema(metric)
        issues.extend(schema_issues)

        # Parameter validation
        param_issues = await self._validate_parameters(metric)
        issues.extend(param_issues)

        # Implementation validation
        impl_issues = await self._validate_implementation(metric)
        issues.extend(impl_issues)

        # Runtime validation (if test contexts provided)
        if test_contexts:
            runtime_issues = await self._validate_runtime(metric, test_contexts)
            issues.extend(runtime_issues)

        # Performance validation
        perf_issues = await self._validate_performance(metric)
        issues.extend(perf_issues)

        # Security validation
        security_issues = await self._validate_security(metric)
        issues.extend(security_issues)

        # Determine overall validity
        has_errors = any(issue.severity == "error" for issue in issues)

        if self.validation_level == ValidationLevel.STRICT:
            valid = not has_errors and not any(issue.severity == "warning" for issue in issues)
        elif self.validation_level == ValidationLevel.MODERATE:
            valid = not has_errors
        else:  # LENIENT
            valid = True

        validation_time = (time.time() - start_time) * 1000

        return ValidationResult(
            metric_name=metric.config.name,
            valid=valid,
            issues=issues,
            validation_time_ms=validation_time,
        )

    async def _validate_schema(self, metric: BaseMetric) -> List[ValidationIssue]:
        """Validate metric configuration schema."""
        issues = []
        config = metric.config

        try:
            # Validate using Pydantic
            config.dict()  # This triggers Pydantic validation
        except ValidationError as e:
            for error in e.errors():
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.SCHEMA,
                        severity="error",
                        message=f"Schema validation error: {error['msg']}",
                        details={"field": error.get("loc"), "input": error.get("input")},
                        location=".".join(str(x) for x in error.get("loc", [])),
                    )
                )

        # Validate output schema if specified
        if config.output_schema:
            try:
                jsonschema.Draft7Validator.check_schema(config.output_schema)
            except jsonschema.SchemaError as e:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.SCHEMA,
                        severity="error",
                        message=f"Invalid output schema: {e.message}",
                        suggestion="Check JSON schema syntax and structure",
                    )
                )

        # Check for required fields based on metric type
        if config.type in [MetricType.BLEU, MetricType.ROUGE] and not config.input_requirements.get(
            "requires_reference"
        ):
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.SCHEMA,
                    severity="warning",
                    message=f"Metric type '{config.type.value}' typically requires reference text",
                    suggestion="Set input_requirements.requires_reference = true",
                )
            )

        return issues

    async def _validate_parameters(self, metric: BaseMetric) -> List[ValidationIssue]:
        """Validate metric parameters."""
        issues = []
        config = metric.config

        # Check parameter definitions
        param_names = set()
        for param in config.parameters:
            # Check for duplicate parameter names
            if param.name in param_names:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.PARAMETER,
                        severity="error",
                        message=f"Duplicate parameter name: {param.name}",
                        suggestion="Use unique parameter names",
                    )
                )
            param_names.add(param.name)

            # Validate parameter type
            if param.type not in ["str", "int", "float", "bool", "list", "dict", "any"]:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.PARAMETER,
                        severity="error",
                        message=f"Invalid parameter type: {param.type}",
                        suggestion="Use supported parameter types",
                    )
                )

            # Check required parameters have no default
            if param.required and param.default is not None:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.PARAMETER,
                        severity="warning",
                        message=f"Required parameter '{param.name}' has default value",
                        suggestion="Remove default value or make parameter optional",
                    )
                )

        # Validate provided parameters
        for name, value in metric.parameters.items():
            param_def = next((p for p in config.parameters if p.name == name), None)
            if not param_def:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.PARAMETER,
                        severity="warning",
                        message=f"Unknown parameter: {name}",
                        suggestion="Check parameter name spelling or add to configuration",
                    )
                )
                continue

            # Type validation
            param_issues = self._validate_parameter_value(param_def, value)
            issues.extend(param_issues)

        return issues

    def _validate_parameter_value(
        self, param: MetricParameter, value: Any
    ) -> List[ValidationIssue]:
        """Validate a single parameter value."""
        issues = []

        # Type checking
        if param.type != "any":
            expected_types = {
                "str": str,
                "int": int,
                "float": (int, float),
                "bool": bool,
                "list": list,
                "dict": dict,
            }

            expected_type = expected_types.get(param.type)
            if expected_type and not isinstance(value, expected_type):
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.PARAMETER,
                        severity="error",
                        message=f"Parameter '{param.name}' has wrong type. Expected {param.type}, got {type(value).__name__}",
                        suggestion=f"Convert value to {param.type}",
                    )
                )

        # Validation rules
        if param.validation:
            if "min" in param.validation and isinstance(value, (int, float)):
                if value < param.validation["min"]:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.PARAMETER,
                            severity="error",
                            message=f"Parameter '{param.name}' value {value} is below minimum {param.validation['min']}",
                            suggestion=f"Use value >= {param.validation['min']}",
                        )
                    )

            if "max" in param.validation and isinstance(value, (int, float)):
                if value > param.validation["max"]:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.PARAMETER,
                            severity="error",
                            message=f"Parameter '{param.name}' value {value} exceeds maximum {param.validation['max']}",
                            suggestion=f"Use value <= {param.validation['max']}",
                        )
                    )

            if "choices" in param.validation and value not in param.validation["choices"]:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.PARAMETER,
                        severity="error",
                        message=f"Parameter '{param.name}' value '{value}' not in allowed choices",
                        details={"allowed_choices": param.validation["choices"]},
                        suggestion=f"Use one of: {param.validation['choices']}",
                    )
                )

        return issues

    async def _validate_implementation(self, metric: BaseMetric) -> List[ValidationIssue]:
        """Validate metric implementation."""
        issues = []

        # Check if compute method is properly implemented
        if not hasattr(metric, "compute"):
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.IMPLEMENTATION,
                    severity="error",
                    message="Metric missing required 'compute' method",
                    suggestion="Implement async compute method",
                )
            )
        else:
            # Check if compute method is async
            if not asyncio.iscoroutinefunction(metric.compute):
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.IMPLEMENTATION,
                        severity="error",
                        message="Compute method must be async",
                        suggestion="Use 'async def compute(...)'",
                    )
                )

            # Check method signature
            sig = inspect.signature(metric.compute)
            if len(sig.parameters) != 2:  # self + context
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.IMPLEMENTATION,
                        severity="error",
                        message="Compute method must accept exactly one parameter (context)",
                        suggestion="Use signature: async def compute(self, context: MetricExecutionContext)",
                    )
                )

        # Check if validate_input method exists and is properly implemented
        if hasattr(metric, "validate_input"):
            if not asyncio.iscoroutinefunction(metric.validate_input):
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.IMPLEMENTATION,
                        severity="warning",
                        message="validate_input method should be async",
                        suggestion="Use 'async def validate_input(...)'",
                    )
                )

        # Check for common implementation issues
        try:
            # Check if class properly inherits from BaseMetric
            if not isinstance(metric, BaseMetric):
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.IMPLEMENTATION,
                        severity="error",
                        message="Metric class must inherit from BaseMetric",
                        suggestion="class YourMetric(BaseMetric): ...",
                    )
                )
        except Exception as e:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.IMPLEMENTATION,
                    severity="error",
                    message=f"Implementation validation failed: {e}",
                    details={"exception": str(e)},
                )
            )

        return issues

    async def _validate_runtime(
        self, metric: BaseMetric, test_contexts: List[MetricExecutionContext]
    ) -> List[ValidationIssue]:
        """Validate metric runtime behavior."""
        issues = []

        for i, context in enumerate(test_contexts):
            try:
                # Test metric execution
                result = await asyncio.wait_for(metric.compute(context), timeout=30.0)

                # Validate result structure
                if not isinstance(result, MetricExecutionResult):
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.RUNTIME,
                            severity="error",
                            message=f"Test {i}: compute() must return MetricExecutionResult",
                            suggestion="Return MetricExecutionResult instance",
                        )
                    )
                    continue

                # Validate result values
                if result.value is None and not result.error:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.RUNTIME,
                            severity="warning",
                            message=f"Test {i}: Result has no value and no error",
                            suggestion="Ensure metric returns valid value or error message",
                        )
                    )

                # Validate output schema if specified
                if metric.config.output_schema and result.value is not None:
                    try:
                        jsonschema.validate(result.value, metric.config.output_schema)
                    except jsonschema.ValidationError as e:
                        issues.append(
                            ValidationIssue(
                                category=ValidationCategory.RUNTIME,
                                severity="error",
                                message=f"Test {i}: Output doesn't match schema: {e.message}",
                                details={"output_value": result.value},
                                suggestion="Ensure output matches defined schema",
                            )
                        )

                # Check for reasonable execution time
                if result.execution_time_ms and result.execution_time_ms > 10000:  # 10 seconds
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.RUNTIME,
                            severity="warning",
                            message=f"Test {i}: Slow execution time: {result.execution_time_ms:.0f}ms",
                            suggestion="Consider optimizing metric computation",
                        )
                    )

            except asyncio.TimeoutError:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.RUNTIME,
                        severity="error",
                        message=f"Test {i}: Metric execution timed out",
                        suggestion="Reduce computation time or increase timeout",
                    )
                )
            except Exception as e:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.RUNTIME,
                        severity="error",
                        message=f"Test {i}: Runtime error: {str(e)}",
                        details={"exception": str(e), "traceback": traceback.format_exc()},
                        suggestion="Fix runtime error in metric implementation",
                    )
                )

        return issues

    async def _validate_performance(self, metric: BaseMetric) -> List[ValidationIssue]:
        """Validate metric performance characteristics."""
        issues = []

        # Check complexity classification
        complexity = metric.config.complexity
        timeout = metric.config.timeout_seconds

        # Suggest appropriate timeouts based on complexity
        if complexity == MetricComplexity.LOW and timeout > 5:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.PERFORMANCE,
                    severity="info",
                    message="Low complexity metric has high timeout",
                    suggestion="Consider reducing timeout for better responsiveness",
                )
            )
        elif complexity == MetricComplexity.VERY_HIGH and timeout < 60:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.PERFORMANCE,
                    severity="warning",
                    message="Very high complexity metric has low timeout",
                    suggestion="Consider increasing timeout to prevent timeouts",
                )
            )

        # Check for expensive operations in implementation
        # This is a simplified check - could be enhanced with static analysis
        if hasattr(metric, "compute"):
            try:
                source = inspect.getsource(metric.compute)

                # Check for potentially expensive operations
                expensive_patterns = [
                    ("requests.", "HTTP requests"),
                    ("urllib.", "HTTP requests"),
                    ("openai.", "OpenAI API calls"),
                    ("anthropic.", "Anthropic API calls"),
                    ("time.sleep", "Blocking sleep calls"),
                    ("subprocess.", "Subprocess calls"),
                ]

                for pattern, description in expensive_patterns:
                    if pattern in source:
                        issues.append(
                            ValidationIssue(
                                category=ValidationCategory.PERFORMANCE,
                                severity="info",
                                message=f"Metric contains {description}",
                                suggestion="Consider caching or async alternatives",
                            )
                        )

            except (OSError, TypeError):
                # Can't get source code (e.g., built-in functions)
                pass

        return issues

    async def _validate_security(self, metric: BaseMetric) -> List[ValidationIssue]:
        """Validate metric security aspects."""
        issues = []

        # Check for API key requirements
        if metric.config.api_key_required:
            # Check if API key handling is secure
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.SECURITY,
                    severity="info",
                    message="Metric requires API key",
                    suggestion="Ensure API keys are handled securely and not logged",
                )
            )

        # Check for potentially unsafe operations
        if hasattr(metric, "compute"):
            try:
                source = inspect.getsource(metric.compute)

                # Check for potentially unsafe patterns
                unsafe_patterns = [
                    ("eval(", "Dynamic code evaluation"),
                    ("exec(", "Dynamic code execution"),
                    ("subprocess.", "Subprocess execution"),
                    ("os.system", "System command execution"),
                    ("open(", "File operations"),
                ]

                for pattern, description in unsafe_patterns:
                    if pattern in source:
                        issues.append(
                            ValidationIssue(
                                category=ValidationCategory.SECURITY,
                                severity="warning",
                                message=f"Metric contains {description}",
                                suggestion="Ensure safe handling and validation of inputs",
                            )
                        )

            except (OSError, TypeError):
                pass

        return issues

    def create_test_context(
        self,
        prediction: str = "This is a test prediction.",
        reference: str = "This is a test reference.",
        context: str = "This is test context.",
    ) -> MetricExecutionContext:
        """Create a test execution context for validation."""
        from ulei.core.schemas import DatasetItem

        dataset_item = DatasetItem(
            id="test-item", inputs={"text": context}, expected_outputs={"response": reference}
        )

        return MetricExecutionContext(
            dataset_item=dataset_item,
            prediction=prediction,
            reference=reference,
            context=context,
            execution_mode=MetricExecutionMode.LOCAL,
        )

    def generate_validation_report(self, result: ValidationResult) -> str:
        """Generate human-readable validation report."""
        report = []
        report.append(f"Validation Report for Metric: {result.metric_name}")
        report.append("=" * 50)
        report.append(f"Status: {'✓ VALID' if result.valid else '✗ INVALID'}")

        if result.validation_time_ms:
            report.append(f"Validation Time: {result.validation_time_ms:.1f}ms")

        report.append("")

        # Group issues by category
        for category in ValidationCategory:
            category_issues = result.get_issues_by_category(category)
            if category_issues:
                report.append(f"{category.value.upper()} Issues:")
                report.append("-" * 20)

                for issue in category_issues:
                    icon = (
                        "🔴"
                        if issue.severity == "error"
                        else "🟡"
                        if issue.severity == "warning"
                        else "ℹ️"
                    )
                    report.append(f"{icon} {issue.message}")

                    if issue.suggestion:
                        report.append(f"   💡 Suggestion: {issue.suggestion}")

                    if issue.location:
                        report.append(f"   📍 Location: {issue.location}")

                    report.append("")

        # Summary
        error_count = len(result.get_issues_by_severity("error"))
        warning_count = len(result.get_issues_by_severity("warning"))
        info_count = len(result.get_issues_by_severity("info"))

        report.append("Summary:")
        report.append(f"- Errors: {error_count}")
        report.append(f"- Warnings: {warning_count}")
        report.append(f"- Info: {info_count}")

        return "\n".join(report)
