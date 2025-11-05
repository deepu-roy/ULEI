"""
Provider adapter registry with resolution and priority handling.
Includes metric registry for mapping logical names to provider-specific metrics.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from ulei.core.interfaces import BaseAdapter
from ulei.utils.errors import ConfigurationError, ProviderError

logger = logging.getLogger(__name__)


class MetricCategory(Enum):
    """Categories of evaluation metrics."""

    ACCURACY = "accuracy"
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    RETRIEVAL = "retrieval"
    SAFETY = "safety"
    BIAS = "bias"
    CUSTOM = "custom"


@dataclass
class MetricDefinition:
    """Definition of a metric with metadata."""

    name: str
    description: str
    category: MetricCategory
    output_type: str  # "score", "classification", "ranking"
    score_range: Optional[tuple] = None  # (min, max) for score metrics
    higher_is_better: bool = True
    requires_context: bool = False
    requires_reference: bool = False
    requires_retrieval: bool = False

    def __post_init__(self) -> None:
        """Validate metric definition."""
        if self.output_type not in ["score", "classification", "ranking"]:
            raise ValueError(f"Invalid output_type: {self.output_type}")

        if self.score_range and len(self.score_range) != 2:
            raise ValueError("score_range must be a tuple of (min, max)")


class MetricRegistry:
    """
    Registry for mapping logical metric names to provider implementations.

    This enables users to specify metrics by logical names (e.g., 'faithfulness')
    and have them automatically mapped to the appropriate provider adapter.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._metrics: Dict[str, MetricDefinition] = {}
        self._provider_mappings: Dict[str, Dict[str, str]] = {}
        self._aliases: Dict[str, str] = {}

        # Register built-in metrics
        self._register_builtin_metrics()

    def register_metric(
        self, definition: MetricDefinition, aliases: Optional[List[str]] = None
    ) -> None:
        """
        Register a metric definition.

        Args:
            definition: Metric definition
            aliases: Alternative names for the metric
        """
        if definition.name in self._metrics:
            raise ValueError(f"Metric '{definition.name}' already registered")

        self._metrics[definition.name] = definition

        # Register aliases
        if aliases:
            for alias in aliases:
                if alias in self._aliases:
                    raise ValueError(f"Alias '{alias}' already exists")
                self._aliases[alias] = definition.name

    def register_provider_mapping(
        self, provider: str, logical_name: str, provider_metric: str
    ) -> None:
        """
        Register mapping from logical metric name to provider-specific metric.

        Args:
            provider: Provider name (e.g., 'ragas', 'deepeval')
            logical_name: Logical metric name
            provider_metric: Provider-specific metric name
        """
        if logical_name not in self._metrics and logical_name not in self._aliases:
            raise ValueError(f"Logical metric '{logical_name}' not registered")

        # Resolve alias to canonical name
        canonical_name = self._aliases.get(logical_name, logical_name)

        # Initialize provider mappings if needed
        if provider not in self._provider_mappings:
            self._provider_mappings[provider] = {}

        self._provider_mappings[provider][canonical_name] = provider_metric

    def get_metric_definition(self, name: str) -> Optional[MetricDefinition]:
        """
        Get metric definition by name or alias.

        Args:
            name: Metric name or alias

        Returns:
            Metric definition if found
        """
        # Resolve alias to canonical name
        canonical_name = self._aliases.get(name, name)
        return self._metrics.get(canonical_name)

    def get_provider_metric(self, provider: str, logical_name: str) -> Optional[str]:
        """
        Get provider-specific metric name for logical metric.

        Args:
            provider: Provider name
            logical_name: Logical metric name

        Returns:
            Provider-specific metric name if mapped
        """
        # Resolve alias to canonical name
        canonical_name = self._aliases.get(logical_name, logical_name)

        return self._provider_mappings.get(provider, {}).get(canonical_name)

    def list_metrics(
        self, category: Optional[MetricCategory] = None, provider: Optional[str] = None
    ) -> List[str]:
        """
        List available metric names.

        Args:
            category: Filter by metric category
            provider: Filter by provider support

        Returns:
            List of metric names
        """
        metrics = []

        for name, definition in self._metrics.items():
            # Filter by category
            if category and definition.category != category:
                continue

            # Filter by provider support
            if provider:
                if provider not in self._provider_mappings:
                    continue
                if name not in self._provider_mappings[provider]:
                    continue

            metrics.append(name)

        return sorted(metrics)

    def validate_metric_config(self, metrics: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Validate metric configuration and return any issues.

        Args:
            metrics: Metric configuration from evaluation suite

        Returns:
            Dictionary of validation issues by metric name
        """
        issues = {}

        for metric_name, config in metrics.items():
            metric_issues = []

            # Check if metric is registered
            definition = self.get_metric_definition(metric_name)
            if not definition:
                metric_issues.append(f"Metric '{metric_name}' not registered")
                issues[metric_name] = metric_issues
                continue

            # Check provider support
            provider = config.get("provider")
            if provider and not self.get_provider_metric(provider, metric_name):
                metric_issues.append(f"Provider '{provider}' does not support '{metric_name}'")

            # Validate threshold against metric properties
            threshold = config.get("threshold")
            if threshold is not None:
                if definition.output_type != "score":
                    metric_issues.append(
                        f"Threshold not supported for {definition.output_type} metrics"
                    )
                elif definition.score_range:
                    min_val, max_val = definition.score_range
                    if not (min_val <= threshold <= max_val):
                        metric_issues.append(
                            f"Threshold {threshold} outside valid range {definition.score_range}"
                        )

            if metric_issues:
                issues[metric_name] = metric_issues

        return issues

    def _register_builtin_metrics(self) -> None:
        """Register built-in metric definitions."""

        # Faithfulness metrics
        self.register_metric(
            MetricDefinition(
                name="faithfulness",
                description="Measures factual consistency of generated answer with given context",
                category=MetricCategory.FAITHFULNESS,
                output_type="score",
                score_range=(0.0, 1.0),
                higher_is_better=True,
                requires_context=True,
            ),
            aliases=["factual_consistency", "groundedness"],
        )

        # Relevance metrics
        self.register_metric(
            MetricDefinition(
                name="answer_relevancy",
                description="Measures how relevant the answer is to the given question",
                category=MetricCategory.RELEVANCE,
                output_type="score",
                score_range=(0.0, 1.0),
                higher_is_better=True,
            ),
            aliases=["relevance", "answer_relevance"],
        )

        self.register_metric(
            MetricDefinition(
                name="context_relevancy",
                description="Measures relevance of retrieved context to the question",
                category=MetricCategory.RELEVANCE,
                output_type="score",
                score_range=(0.0, 1.0),
                higher_is_better=True,
                requires_context=True,
                requires_retrieval=True,
            ),
            aliases=["context_relevance", "context_precision"],
        )

        # Accuracy metrics
        self.register_metric(
            MetricDefinition(
                name="answer_correctness",
                description="Measures factual accuracy of answer against reference",
                category=MetricCategory.ACCURACY,
                output_type="score",
                score_range=(0.0, 1.0),
                higher_is_better=True,
                requires_reference=True,
            ),
            aliases=["correctness", "accuracy"],
        )

        # Safety metrics
        self.register_metric(
            MetricDefinition(
                name="toxicity",
                description="Detects toxic or harmful content in responses",
                category=MetricCategory.SAFETY,
                output_type="score",
                score_range=(0.0, 1.0),
                higher_is_better=False,  # Lower toxicity is better
            )
        )

        self.register_metric(
            MetricDefinition(
                name="bias",
                description="Detects biased content in responses",
                category=MetricCategory.BIAS,
                output_type="score",
                score_range=(0.0, 1.0),
                higher_is_better=False,  # Lower bias is better
            )
        )


class AdapterRegistry:
    """Registry for managing provider adapters with priority-based resolution."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._adapters: Dict[str, Type[BaseAdapter]] = {}
        self._instances: Dict[str, BaseAdapter] = {}
        self._default_priority: List[str] = []
        self._metric_registry = MetricRegistry()

    def register_adapter(self, provider_name: str, adapter_class: Type[BaseAdapter]) -> None:
        """Register a provider adapter class.

        Args:
            provider_name: Unique name for the provider
            adapter_class: Adapter class implementing BaseAdapter

        Raises:
            ConfigurationError: If provider name is already registered
        """
        if provider_name in self._adapters:
            raise ConfigurationError(f"Provider '{provider_name}' is already registered")

        # Validate adapter class implements required interface
        if not issubclass(adapter_class, BaseAdapter):
            raise ConfigurationError("Adapter class must inherit from BaseAdapter")

        self._adapters[provider_name] = adapter_class
        logger.debug(f"Registered adapter for provider: {provider_name}")

    def unregister_adapter(self, provider_name: str) -> None:
        """Remove a provider adapter from the registry.

        Args:
            provider_name: Name of provider to remove
        """
        if provider_name in self._adapters:
            del self._adapters[provider_name]

        if provider_name in self._instances:
            del self._instances[provider_name]

        logger.debug(f"Unregistered adapter for provider: {provider_name}")

    def get_adapter(self, provider_name: str, config: Optional[Dict] = None) -> BaseAdapter:
        """Get an initialized adapter instance.

        Args:
            provider_name: Name of the provider
            config: Optional configuration for the adapter

        Returns:
            Initialized adapter instance

        Raises:
            ProviderError: If provider is not registered or cannot be initialized
        """
        if provider_name not in self._adapters:
            raise ProviderError(f"Provider '{provider_name}' is not registered")

        # Return cached instance if available and no config provided
        if config is None and provider_name in self._instances:
            return self._instances[provider_name]

        try:
            adapter_class = self._adapters[provider_name]
            adapter = adapter_class(config or {})

            # Cache instance if no specific config was provided
            if config is None:
                self._instances[provider_name] = adapter

            return adapter

        except Exception as e:
            raise ProviderError(f"Failed to initialize adapter for '{provider_name}': {e}") from e

    def resolve_provider_for_metric(
        self, metric_name: str, provider_priority: Optional[List[str]] = None
    ) -> Optional[str]:
        """Resolve the best provider for a given metric based on priority.

        Args:
            metric_name: Name of the metric to evaluate
            provider_priority: Custom priority list (uses default if None)

        Returns:
            Provider name that supports the metric, or None if none found
        """
        priority_list = provider_priority or self._default_priority or list(self._adapters.keys())

        for provider_name in priority_list:
            if provider_name in self._adapters:
                try:
                    adapter = self.get_adapter(provider_name)
                    if adapter.supports_metric(metric_name):
                        return provider_name
                except Exception as e:
                    logger.warning(f"Failed to check metric support for {provider_name}: {e}")
                    continue

        return None

    def get_providers_for_metric(self, metric_name: str) -> List[str]:
        """Get all providers that support a given metric.

        Args:
            metric_name: Name of the metric

        Returns:
            List of provider names that support the metric
        """
        providers = []

        for provider_name in self._adapters:
            try:
                adapter = self.get_adapter(provider_name)
                if adapter.supports_metric(metric_name):
                    providers.append(provider_name)
            except Exception as e:
                logger.warning(f"Failed to check metric support for {provider_name}: {e}")
                continue

        return providers

    def list_registered_providers(self) -> List[str]:
        """Get list of all registered provider names.

        Returns:
            List of registered provider names
        """
        return list(self._adapters.keys())

    def list_supported_metrics(self, provider_name: Optional[str] = None) -> List[str]:
        """Get list of supported metrics.

        Args:
            provider_name: Specific provider to check (all providers if None)

        Returns:
            List of metric names

        Raises:
            ProviderError: If specific provider is not registered
        """
        if provider_name is not None:
            if provider_name not in self._adapters:
                raise ProviderError(f"Provider '{provider_name}' is not registered")

            try:
                adapter = self.get_adapter(provider_name)
                return adapter.supported_metrics
            except Exception as e:
                raise ProviderError(f"Failed to get metrics for {provider_name}: {e}") from e

        # Get metrics from all providers
        all_metrics = set()
        for provider in self._adapters:
            try:
                adapter = self.get_adapter(provider)
                all_metrics.update(adapter.supported_metrics)
            except Exception as e:
                logger.warning(f"Failed to get metrics from {provider}: {e}")
                continue

        return sorted(all_metrics)

    def set_default_priority(self, priority_list: List[str]) -> None:
        """Set the default provider priority order.

        Args:
            priority_list: List of provider names in priority order

        Raises:
            ConfigurationError: If priority list contains invalid providers
        """
        invalid_providers = set(priority_list) - set(self._adapters.keys())
        if invalid_providers:
            raise ConfigurationError(f"Invalid providers in priority list: {invalid_providers}")

        self._default_priority = priority_list.copy()
        logger.debug(f"Set default provider priority: {priority_list}")

    def validate_configuration(self, providers_config: Dict[str, Dict]) -> Dict[str, bool]:
        """Validate configuration for multiple providers.

        Args:
            providers_config: Dictionary mapping provider names to config

        Returns:
            Dictionary mapping provider names to validation status
        """
        results = {}

        for provider_name, config in providers_config.items():
            try:
                if provider_name not in self._adapters:
                    results[provider_name] = False
                    continue

                adapter = self.get_adapter(provider_name, config)
                results[provider_name] = adapter.validate_config(config)

            except Exception as e:
                logger.error(f"Configuration validation failed for {provider_name}: {e}")
                results[provider_name] = False

        return results

    def get_metric_registry(self) -> MetricRegistry:
        """Get the metric registry.

        Returns:
            MetricRegistry instance
        """
        return self._metric_registry

    def register_provider_metrics(
        self, provider_name: str, metric_mappings: Dict[str, str]
    ) -> None:
        """
        Register metric mappings for a provider.

        Args:
            provider_name: Provider name
            metric_mappings: Dictionary mapping logical names to provider metrics
        """
        for logical_name, provider_metric in metric_mappings.items():
            self._metric_registry.register_provider_mapping(
                provider_name, logical_name, provider_metric
            )
        logger.debug(f"Registered {len(metric_mappings)} metrics for {provider_name}")

    def clear(self) -> None:
        """Clear all registered adapters and instances."""
        self._adapters.clear()
        self._instances.clear()
        self._default_priority.clear()
        self._metric_registry = MetricRegistry()  # Reset metric registry too
        logger.debug("Cleared adapter registry")


# Global registry instance
_global_registry: Optional[AdapterRegistry] = None


def get_registry() -> AdapterRegistry:
    """Get the global adapter registry instance.

    Returns:
        Global AdapterRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = AdapterRegistry()
    return _global_registry


def register_adapter(provider_name: str, adapter_class: Type[BaseAdapter]) -> None:
    """Register an adapter with the global registry.

    Args:
        provider_name: Unique name for the provider
        adapter_class: Adapter class implementing BaseAdapter
    """
    registry = get_registry()
    registry.register_adapter(provider_name, adapter_class)
