"""
Custom metrics configuration framework for ULEI.

Provides extensible framework for defining, validating, and managing
custom evaluation metrics with flexible configuration and execution.
"""

import asyncio
import importlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import yaml
from pydantic import BaseModel, Field, ValidationError, validator

from ulei.core.schemas import DatasetItem

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of evaluation metrics."""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    BLEU = "bleu"
    ROUGE = "rouge"
    BERTSCORE = "bertscore"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    FACTUAL_ACCURACY = "factual_accuracy"
    HALLUCINATION_DETECTION = "hallucination_detection"
    BIAS_DETECTION = "bias_detection"
    TOXICITY_DETECTION = "toxicity_detection"
    FLUENCY = "fluency"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    GROUNDEDNESS = "groundedness"
    CUSTOM = "custom"


class MetricExecutionMode(Enum):
    """Metric execution modes."""

    LOCAL = "local"  # Execute locally
    REMOTE = "remote"  # Execute via API call
    HYBRID = "hybrid"  # Try local, fallback to remote
    CACHED = "cached"  # Use cached results if available


class MetricComplexity(Enum):
    """Metric computational complexity levels."""

    LOW = "low"  # Fast, simple metrics
    MEDIUM = "medium"  # Moderate computation
    HIGH = "high"  # Expensive computation
    VERY_HIGH = "very_high"  # Very expensive (LLM-based)


@dataclass
class MetricExecutionContext:
    """Context for metric execution."""

    dataset_item: DatasetItem
    prediction: str
    reference: Optional[str] = None
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_mode: MetricExecutionMode = MetricExecutionMode.LOCAL
    timeout_seconds: Optional[float] = None


@dataclass
class MetricExecutionResult:
    """Result of metric execution."""

    metric_name: str
    value: Union[float, int, str, bool, Dict[str, Any]]
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    execution_time_ms: Optional[float] = None
    execution_mode: Optional[MetricExecutionMode] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MetricParameter(BaseModel):
    """Configuration parameter for a metric."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (str, int, float, bool, list, dict)")
    default: Optional[Any] = Field(None, description="Default value")
    required: bool = Field(False, description="Whether parameter is required")
    description: Optional[str] = Field(None, description="Parameter description")
    validation: Optional[Dict[str, Any]] = Field(None, description="Validation rules")

    @validator("type")
    def validate_type(cls, v):
        """Validate parameter type."""
        allowed_types = ["str", "int", "float", "bool", "list", "dict", "any"]
        if v not in allowed_types:
            raise ValueError(f"Parameter type must be one of {allowed_types}")
        return v


class MetricConfiguration(BaseModel):
    """Configuration for a custom metric."""

    name: str = Field(..., description="Unique metric name")
    display_name: Optional[str] = Field(None, description="Human-readable name")
    description: Optional[str] = Field(None, description="Metric description")

    # Metric classification
    type: MetricType = Field(MetricType.CUSTOM, description="Metric type")
    complexity: MetricComplexity = Field(
        MetricComplexity.MEDIUM, description="Computational complexity"
    )
    category: Optional[str] = Field(None, description="Metric category")
    tags: List[str] = Field(default_factory=list, description="Metric tags")

    # Execution configuration
    execution_mode: MetricExecutionMode = Field(
        MetricExecutionMode.LOCAL, description="Execution mode"
    )
    timeout_seconds: float = Field(30.0, description="Execution timeout")
    retry_attempts: int = Field(2, description="Number of retry attempts")

    # Implementation
    implementation: Dict[str, Any] = Field(..., description="Implementation configuration")
    parameters: List[MetricParameter] = Field(default_factory=list, description="Metric parameters")

    # Dependencies
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies")
    api_key_required: bool = Field(False, description="Whether API key is required")

    # Validation
    input_requirements: Dict[str, Any] = Field(
        default_factory=dict, description="Input validation requirements"
    )
    output_schema: Dict[str, Any] = Field(
        default_factory=dict, description="Expected output schema"
    )

    # Metadata
    version: str = Field("1.0.0", description="Metric version")
    author: Optional[str] = Field(None, description="Metric author")
    created: Optional[datetime] = Field(None, description="Creation timestamp")
    updated: Optional[datetime] = Field(None, description="Last update timestamp")

    @validator("name")
    def validate_name(cls, v):
        """Validate metric name."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Metric name must be alphanumeric with underscores or hyphens")
        return v

    @validator("implementation")
    def validate_implementation(cls, v):
        """Validate implementation configuration."""
        if "type" not in v:
            raise ValueError("Implementation must specify 'type'")

        impl_type = v["type"]
        if impl_type == "function":
            if "module" not in v or "function" not in v:
                raise ValueError("Function implementation must specify 'module' and 'function'")
        elif impl_type == "class":
            if "module" not in v or "class" not in v:
                raise ValueError("Class implementation must specify 'module' and 'class'")
        elif impl_type == "api":
            if "endpoint" not in v:
                raise ValueError("API implementation must specify 'endpoint'")
        else:
            raise ValueError(f"Unknown implementation type: {impl_type}")

        return v


class BaseMetric(ABC):
    """Abstract base class for custom metrics."""

    def __init__(self, config: MetricConfiguration, **kwargs):
        """
        Initialize metric.

        Args:
            config: Metric configuration
            **kwargs: Additional parameters
        """
        self.config = config
        self.parameters = kwargs

        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate provided parameters against configuration."""
        config_params = {p.name: p for p in self.config.parameters}

        # Check required parameters
        for param in self.config.parameters:
            if param.required and param.name not in self.parameters:
                if param.default is None:
                    raise ValueError(f"Required parameter '{param.name}' not provided")
                self.parameters[param.name] = param.default

        # Validate parameter types and values
        for name, value in self.parameters.items():
            if name in config_params:
                param = config_params[name]
                self._validate_parameter_value(param, value)

    def _validate_parameter_value(self, param: MetricParameter, value: Any):
        """Validate a parameter value."""
        # Type validation
        if param.type != "any":
            expected_type = {
                "str": str,
                "int": int,
                "float": (int, float),
                "bool": bool,
                "list": list,
                "dict": dict,
            }.get(param.type)

            if expected_type and not isinstance(value, expected_type):
                raise ValueError(f"Parameter '{param.name}' must be of type {param.type}")

        # Custom validation rules
        if param.validation:
            self._apply_validation_rules(param.name, value, param.validation)

    def _apply_validation_rules(self, name: str, value: Any, rules: Dict[str, Any]):
        """Apply custom validation rules."""
        if "min" in rules and value < rules["min"]:
            raise ValueError(f"Parameter '{name}' must be >= {rules['min']}")

        if "max" in rules and value > rules["max"]:
            raise ValueError(f"Parameter '{name}' must be <= {rules['max']}")

        if "choices" in rules and value not in rules["choices"]:
            raise ValueError(f"Parameter '{name}' must be one of {rules['choices']}")

        if "pattern" in rules:
            import re

            if not re.match(rules["pattern"], str(value)):
                raise ValueError(f"Parameter '{name}' does not match pattern {rules['pattern']}")

    @abstractmethod
    async def compute(self, context: MetricExecutionContext) -> MetricExecutionResult:
        """
        Compute the metric.

        Args:
            context: Execution context

        Returns:
            Metric result
        """
        pass

    async def validate_input(self, context: MetricExecutionContext) -> bool:
        """
        Validate input requirements.

        Args:
            context: Execution context

        Returns:
            True if input is valid
        """
        requirements = self.config.input_requirements

        if requirements.get("requires_reference") and not context.reference:
            return False

        if requirements.get("requires_context") and not context.context:
            return False

        if requirements.get("min_prediction_length", 0) > len(context.prediction):
            return False

        return True

    def get_info(self) -> Dict[str, Any]:
        """Get metric information."""
        return {
            "name": self.config.name,
            "display_name": self.config.display_name,
            "description": self.config.description,
            "type": self.config.type.value,
            "complexity": self.config.complexity.value,
            "parameters": [p.dict() for p in self.config.parameters],
            "version": self.config.version,
        }


class CustomMetricRegistry:
    """Registry for managing custom metrics."""

    def __init__(self):
        """Initialize metric registry."""
        self.metrics: Dict[str, Type[BaseMetric]] = {}
        self.configurations: Dict[str, MetricConfiguration] = {}
        self.instances: Dict[str, BaseMetric] = {}

        # Built-in metrics
        self._register_builtin_metrics()

    def _register_builtin_metrics(self):
        """Register built-in metrics."""
        # This would register standard metrics like BLEU, ROUGE, etc.
        pass

    def register_metric(self, metric_class: Type[BaseMetric], config: MetricConfiguration):
        """
        Register a custom metric.

        Args:
            metric_class: Metric implementation class
            config: Metric configuration
        """
        name = config.name

        # Validate metric class
        if not issubclass(metric_class, BaseMetric):
            raise ValueError("Metric class must inherit from BaseMetric")

        # Store registration
        self.metrics[name] = metric_class
        self.configurations[name] = config

        logger.info(f"Registered metric: {name}")

    def register_from_config(self, config_path: Union[str, Path]):
        """
        Register metric from configuration file.

        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load configuration
        with open(config_path) as f:
            if config_path.suffix.lower() in [".yml", ".yaml"]:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration format: {config_path.suffix}")

        # Create configuration object
        config = MetricConfiguration(**config_data)

        # Load implementation
        metric_class = self._load_metric_implementation(config)

        # Register
        self.register_metric(metric_class, config)

    def _load_metric_implementation(self, config: MetricConfiguration) -> Type[BaseMetric]:
        """Load metric implementation from configuration."""
        impl = config.implementation
        impl_type = impl["type"]

        if impl_type == "function":
            return self._create_function_metric(config, impl)
        elif impl_type == "class":
            return self._load_class_metric(impl)
        elif impl_type == "api":
            return self._create_api_metric(config, impl)
        else:
            raise ValueError(f"Unknown implementation type: {impl_type}")

    def _create_function_metric(
        self, config: MetricConfiguration, impl: Dict[str, Any]
    ) -> Type[BaseMetric]:
        """Create metric class from function implementation."""
        # Import function
        module = importlib.import_module(impl["module"])
        func = getattr(module, impl["function"])

        # Create wrapper class
        class FunctionMetric(BaseMetric):
            async def compute(self, context: MetricExecutionContext) -> MetricExecutionResult:
                import time

                start_time = time.time()

                try:
                    # Call function
                    if asyncio.iscoroutinefunction(func):
                        result = await func(context, **self.parameters)
                    else:
                        result = func(context, **self.parameters)

                    execution_time = (time.time() - start_time) * 1000

                    return MetricExecutionResult(
                        metric_name=config.name,
                        value=result,
                        execution_time_ms=execution_time,
                        execution_mode=context.execution_mode,
                    )

                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000
                    return MetricExecutionResult(
                        metric_name=config.name,
                        value=None,
                        error=str(e),
                        execution_time_ms=execution_time,
                        execution_mode=context.execution_mode,
                    )

        return FunctionMetric

    def _load_class_metric(self, impl: Dict[str, Any]) -> Type[BaseMetric]:
        """Load metric class implementation."""
        module = importlib.import_module(impl["module"])
        metric_class = getattr(module, impl["class"])

        if not issubclass(metric_class, BaseMetric):
            raise ValueError("Metric class must inherit from BaseMetric")

        return metric_class

    def _create_api_metric(
        self, config: MetricConfiguration, impl: Dict[str, Any]
    ) -> Type[BaseMetric]:
        """Create metric class for API-based implementation."""
        import aiohttp

        class APIMetric(BaseMetric):
            async def compute(self, context: MetricExecutionContext) -> MetricExecutionResult:
                import time

                start_time = time.time()

                try:
                    # Prepare API request
                    endpoint = impl["endpoint"]
                    method = impl.get("method", "POST")
                    headers = impl.get("headers", {})

                    payload = {
                        "prediction": context.prediction,
                        "reference": context.reference,
                        "context": context.context,
                        "parameters": self.parameters,
                    }

                    timeout = aiohttp.ClientTimeout(total=context.timeout_seconds)

                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.request(
                            method, endpoint, json=payload, headers=headers
                        ) as response:
                            if response.status != 200:
                                raise Exception(f"API request failed: {response.status}")

                            result_data = await response.json()
                            execution_time = (time.time() - start_time) * 1000

                            return MetricExecutionResult(
                                metric_name=config.name,
                                value=result_data.get("value"),
                                confidence=result_data.get("confidence"),
                                explanation=result_data.get("explanation"),
                                execution_time_ms=execution_time,
                                execution_mode=MetricExecutionMode.REMOTE,
                                metadata=result_data.get("metadata"),
                            )

                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000
                    return MetricExecutionResult(
                        metric_name=config.name,
                        value=None,
                        error=str(e),
                        execution_time_ms=execution_time,
                        execution_mode=MetricExecutionMode.REMOTE,
                    )

        return APIMetric

    def get_metric(self, name: str, **parameters) -> BaseMetric:
        """
        Get metric instance.

        Args:
            name: Metric name
            **parameters: Metric parameters

        Returns:
            Metric instance
        """
        if name not in self.metrics:
            raise ValueError(f"Unknown metric: {name}")

        # Create cache key
        param_key = str(sorted(parameters.items()))
        cache_key = f"{name}:{param_key}"

        # Return cached instance if available
        if cache_key in self.instances:
            return self.instances[cache_key]

        # Create new instance
        metric_class = self.metrics[name]
        config = self.configurations[name]
        instance = metric_class(config, **parameters)

        # Cache instance
        self.instances[cache_key] = instance

        return instance

    def list_metrics(self) -> List[Dict[str, Any]]:
        """List all registered metrics."""
        return [
            {"name": name, "config": config.dict(), "available": name in self.metrics}
            for name, config in self.configurations.items()
        ]

    def get_metric_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific metric."""
        if name not in self.configurations:
            return None

        config = self.configurations[name]
        return {
            "name": name,
            "display_name": config.display_name,
            "description": config.description,
            "type": config.type.value,
            "complexity": config.complexity.value,
            "parameters": [p.dict() for p in config.parameters],
            "version": config.version,
            "available": name in self.metrics,
        }

    def validate_configuration(self, config: MetricConfiguration) -> List[str]:
        """
        Validate metric configuration.

        Args:
            config: Configuration to validate

        Returns:
            List of validation errors
        """
        errors = []

        try:
            # Validate configuration object
            config.dict()  # This will trigger pydantic validation
        except ValidationError as e:
            errors.extend([str(error) for error in e.errors()])

        # Check implementation dependencies
        impl = config.implementation
        if impl["type"] in ["function", "class"]:
            try:
                module = importlib.import_module(impl["module"])
                if impl["type"] == "function":
                    if not hasattr(module, impl["function"]):
                        errors.append(
                            f"Function '{impl['function']}' not found in module '{impl['module']}'"
                        )
                else:
                    if not hasattr(module, impl["class"]):
                        errors.append(
                            f"Class '{impl['class']}' not found in module '{impl['module']}'"
                        )
            except ImportError as e:
                errors.append(f"Cannot import module '{impl['module']}': {e}")

        return errors


# Global metric registry
metric_registry = CustomMetricRegistry()
