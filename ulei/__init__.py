"""
Unified LLM Evaluation Interface (ULEI)

A provider-agnostic evaluation framework for LLM and RAG systems.
Supports both offline batch evaluation and online shadow evaluation.
"""

__version__ = "0.1.0"
__author__ = "ULEI Contributors"
__description__ = "Provider-agnostic evaluation for LLM and RAG systems"

# Core exports
from ulei.core.evaluator import Evaluator
from ulei.core.registry import AdapterRegistry
from ulei.core.schemas import (
    DatasetItem,
    EvaluationReport,
    EvaluationRun,
    EvaluationSuite,
    MetricResult,
)

# Common exceptions
from ulei.utils.errors import (
    ConfigurationError,
    EvaluationError,
    ProviderError,
    ULEIError,
)

# Import adapters to trigger registration
# These imports are needed for adapters to self-register with the global registry
try:
    from ulei.adapters import ragas_adapter  # noqa: F401
except ImportError:
    pass  # Ragas not installed

try:
    from ulei.adapters import deepeval_adapter  # noqa: F401
except ImportError:
    pass  # DeepEval not installed

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__description__",
    # Core classes
    "EvaluationSuite",
    "DatasetItem",
    "MetricResult",
    "EvaluationReport",
    "EvaluationRun",
    "Evaluator",
    "AdapterRegistry",
    # Exceptions
    "ULEIError",
    "EvaluationError",
    "ProviderError",
    "ConfigurationError",
]
