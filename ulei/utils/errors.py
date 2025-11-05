"""
Custom exception classes for ULEI error handling.
"""

from typing import Any, Dict, Optional


class ULEIError(Exception):
    """Base exception class for all ULEI errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize ULEI error.

        Args:
            message: Error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class ConfigurationError(ULEIError):
    """Raised when there are configuration-related errors."""

    pass


class EvaluationError(ULEIError):
    """Raised when evaluation fails."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        metric: Optional[str] = None,
        item_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize evaluation error with context.

        Args:
            message: Error message
            provider: Provider name where error occurred
            metric: Metric name being evaluated
            item_id: ID of the dataset item
            details: Additional error details
        """
        error_details = details or {}
        if provider:
            error_details["provider"] = provider
        if metric:
            error_details["metric"] = metric
        if item_id:
            error_details["item_id"] = item_id

        super().__init__(message, error_details)
        self.provider = provider
        self.metric = metric
        self.item_id = item_id


class ProviderError(ULEIError):
    """Raised when there are provider-specific errors."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize provider error.

        Args:
            message: Error message
            provider: Provider name
            status_code: HTTP status code if applicable
            details: Additional error details
        """
        error_details = details or {}
        if provider:
            error_details["provider"] = provider
        if status_code:
            error_details["status_code"] = status_code

        super().__init__(message, error_details)
        self.provider = provider
        self.status_code = status_code


class ReportingError(ULEIError):
    """Raised when report generation fails."""

    def __init__(
        self,
        message: str,
        format_name: Optional[str] = None,
        output_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize reporting error.

        Args:
            message: Error message
            format_name: Report format that failed
            output_path: Output path where error occurred
            details: Additional error details
        """
        error_details = details or {}
        if format_name:
            error_details["format"] = format_name
        if output_path:
            error_details["output_path"] = output_path

        super().__init__(message, error_details)
        self.format_name = format_name
        self.output_path = output_path


class DatasetError(ULEIError):
    """Raised when there are dataset-related errors."""

    def __init__(
        self,
        message: str,
        dataset_path: Optional[str] = None,
        line_number: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize dataset error.

        Args:
            message: Error message
            dataset_path: Path to the dataset file
            line_number: Line number where error occurred
            details: Additional error details
        """
        error_details = details or {}
        if dataset_path:
            error_details["dataset_path"] = dataset_path
        if line_number:
            error_details["line_number"] = line_number

        super().__init__(message, error_details)
        self.dataset_path = dataset_path
        self.line_number = line_number


class ValidationError(ULEIError):
    """Raised when validation fails."""

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize validation error.

        Args:
            message: Error message
            field_name: Name of the field that failed validation
            field_value: Value that failed validation
            details: Additional error details
        """
        error_details = details or {}
        if field_name:
            error_details["field_name"] = field_name
        if field_value is not None:
            error_details["field_value"] = field_value

        super().__init__(message, error_details)
        self.field_name = field_name
        self.field_value = field_value


class BudgetExceededError(ULEIError):
    """Raised when budget limit is exceeded."""

    def __init__(
        self,
        message: str,
        budget_limit: Optional[float] = None,
        current_cost: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize budget exceeded error.

        Args:
            message: Error message
            budget_limit: Maximum allowed budget
            current_cost: Current estimated cost
            details: Additional error details
        """
        error_details = details or {}
        if budget_limit is not None:
            error_details["budget_limit"] = budget_limit
        if current_cost is not None:
            error_details["current_cost"] = current_cost

        super().__init__(message, error_details)
        self.budget_limit = budget_limit
        self.current_cost = current_cost


class TimeoutError(ULEIError):
    """Raised when operations timeout."""

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize timeout error.

        Args:
            message: Error message
            timeout_seconds: Timeout duration in seconds
            operation: Name of the operation that timed out
            details: Additional error details
        """
        error_details = details or {}
        if timeout_seconds is not None:
            error_details["timeout_seconds"] = timeout_seconds
        if operation:
            error_details["operation"] = operation

        super().__init__(message, error_details)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class CacheError(ULEIError):
    """Raised when cache operations fail."""

    pass


class MetricNotSupportedError(ProviderError):
    """Raised when a provider doesn't support a requested metric."""

    def __init__(
        self,
        message: str,
        provider: str,
        metric: str,
        supported_metrics: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize metric not supported error.

        Args:
            message: Error message
            provider: Provider name
            metric: Unsupported metric name
            supported_metrics: List of supported metrics
            details: Additional error details
        """
        error_details = details or {}
        error_details["metric"] = metric
        if supported_metrics:
            error_details["supported_metrics"] = supported_metrics

        super().__init__(message, provider, details=error_details)
        self.metric = metric
        self.supported_metrics = supported_metrics


# Error handling utilities
def handle_provider_error(error: Exception, provider: str, context: str = "") -> ProviderError:
    """Convert generic exceptions to ProviderError with context.

    Args:
        error: Original exception
        provider: Provider name
        context: Additional context about where the error occurred

    Returns:
        ProviderError with wrapped exception details
    """
    message = f"Provider '{provider}' error"
    if context:
        message += f" during {context}"
    message += f": {str(error)}"

    details = {"original_error": str(error), "error_type": type(error).__name__}

    # Extract status code if it's an HTTP error
    status_code = None
    if hasattr(error, "response") and hasattr(error.response, "status_code"):
        status_code = error.response.status_code
    elif hasattr(error, "status_code"):
        status_code = error.status_code

    return ProviderError(
        message=message, provider=provider, status_code=status_code, details=details
    )


def handle_evaluation_error(
    error: Exception, provider: str, metric: str, item_id: str
) -> EvaluationError:
    """Convert generic exceptions to EvaluationError with context.

    Args:
        error: Original exception
        provider: Provider name
        metric: Metric name
        item_id: Dataset item ID

    Returns:
        EvaluationError with wrapped exception details
    """
    message = f"Evaluation failed for metric '{metric}' on item '{item_id}': {str(error)}"

    details = {"original_error": str(error), "error_type": type(error).__name__}

    return EvaluationError(
        message=message, provider=provider, metric=metric, item_id=item_id, details=details
    )
