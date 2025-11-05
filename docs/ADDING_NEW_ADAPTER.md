# Adding a New Adapter to ULEI

This guide walks through adding a new evaluation provider adapter to ULEI, using **Promptfoo** as an example with metrics like toxicity, jailbreak resistance, conciseness, and coherence.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Step-by-Step Guide](#step-by-step-guide)
- [Testing Your Adapter](#testing-your-adapter)
- [Integration Checklist](#integration-checklist)

---

## Overview

ULEI uses an **adapter pattern** to normalize evaluation across different providers. Each adapter:

1. Inherits from `BaseAdapter` (in `ulei/adapters/base.py`)
2. Implements required abstract methods
3. Self-registers with the global metric registry
4. Maps provider-specific metrics to ULEI's logical metric names

**Architecture Flow:**

```
Dataset → Evaluator → Registry → Adapter (Promptfoo) → Provider API → MetricResult
```

---

## Prerequisites

Before creating your adapter, ensure:

1. **Provider SDK installed**: `pip install promptfoo` (or equivalent)
2. **API access**: API keys, endpoints, or local installation configured
3. **Metric documentation**: Understand what each metric measures and its output format
4. **Required fields**: Know what input data each metric needs (question, answer, context, etc.)

---

## Step-by-Step Guide

### Step 1: Create the Adapter File

Create a new file: `ulei/adapters/promptfoo_adapter.py`

```python
"""
Promptfoo provider adapter for LLM safety and quality evaluation metrics.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from ulei.adapters.base import BaseAdapter
from ulei.core.schemas import DatasetItem, MetricResult
from ulei.utils.errors import EvaluationError, MetricNotSupportedError, ProviderError

# Module-level logger (important: use this instead of self.logger in validate_config)
logger = logging.getLogger(__name__)

# Check if Promptfoo is available
try:
    import promptfoo  # type: ignore
    from promptfoo.evaluators import (  # type: ignore
        ToxicityEvaluator,
        JailbreakEvaluator,
        ConcisenessEvaluator,
        CoherenceEvaluator,
    )
    
    PROMPTFOO_AVAILABLE = True
except ImportError:
    PROMPTFOO_AVAILABLE = False


class PromptfooAdapter(BaseAdapter):
    """Adapter for Promptfoo evaluation provider."""

    # Required configuration keys (can be empty list if no keys required)
    required_config_keys = ["api_key"]

    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize Promptfoo adapter.

        Args:
            config: Configuration dictionary with API key and other settings
        """
        if not PROMPTFOO_AVAILABLE:
            raise ProviderError(
                "Promptfoo is not installed. Please install with: pip install promptfoo",
                provider="promptfoo",
            )

        # Use empty dict if no config provided (for introspection)
        super().__init__(config or {})
        self._setup_promptfoo()

    def _setup_promptfoo(self) -> None:
        """Setup Promptfoo configuration."""
        # Set up API key if provided
        if "api_key" in self.config:
            import os
            os.environ["PROMPTFOO_API_KEY"] = self.config["api_key"]

        # Configure base URL if using custom endpoint
        if "base_url" in self.config:
            # Configure Promptfoo to use custom endpoint
            promptfoo.configure(base_url=self.config["base_url"])

        # Metric evaluator mapping
        self._evaluator_classes = {
            "toxicity": ToxicityEvaluator,
            "jailbreak_resistance": JailbreakEvaluator,
            "conciseness": ConcisenessEvaluator,
            "coherence": CoherenceEvaluator,
        }

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "promptfoo"

    @property
    def supported_metrics(self) -> List[str]:
        """Return list of supported metrics."""
        return list(self._evaluator_classes.keys())

    async def evaluate_metric(
        self, metric_name: str, item: DatasetItem, config: Dict[str, Any]
    ) -> MetricResult:
        """Evaluate a single metric for a dataset item.

        Args:
            metric_name: Name of the metric to evaluate
            item: Dataset item to evaluate
            config: Metric-specific configuration

        Returns:
            MetricResult with evaluation outcome
        """
        start_time = time.time()

        try:
            # Validate metric support
            if not self.supports_metric(metric_name):
                raise MetricNotSupportedError(
                    f"Metric '{metric_name}' not supported by Promptfoo",
                    provider=self.provider_name,
                    metric=metric_name,
                    supported_metrics=self.supported_metrics,
                )

            # Validate required fields based on metric
            self._validate_item_for_metric(item, metric_name)

            # Get evaluator class
            evaluator_class = self._evaluator_classes[metric_name]

            # Extract configuration
            model = config.get("model", self.config.get("default_model", "gpt-3.5-turbo"))
            threshold = config.get("threshold", 0.7)
            temperature = config.get("temperature", 0.0)

            # Create evaluator instance
            evaluator = evaluator_class(
                model=model,
                threshold=threshold,
                temperature=temperature,
            )

            # Prepare evaluation input based on metric requirements
            eval_input = self._prepare_evaluation_input(metric_name, item)

            # Run evaluation (convert to async if provider is sync)
            if asyncio.iscoroutinefunction(evaluator.evaluate):
                result = await evaluator.evaluate(**eval_input)
            else:
                result = await asyncio.to_thread(evaluator.evaluate, **eval_input)

            # Parse result and create MetricResult
            execution_time = time.time() - start_time

            return self._create_result(
                metric_name=metric_name,
                item_id=item.id,
                score=result.get("score"),
                confidence=result.get("confidence"),
                evidence=result.get("evidence", {}),
                execution_time=execution_time,
                cost_estimate=self.estimate_cost(metric_name, item, config),
                raw_response=result,
            )

        except Exception as e:
            execution_time = time.time() - start_time

            if isinstance(e, (EvaluationError, MetricNotSupportedError)):
                raise

            error_msg = f"Promptfoo evaluation failed: {str(e)}"
            self.logger.error(error_msg)

            return self._create_result(
                metric_name=metric_name,
                item_id=item.id,
                error=error_msg,
                execution_time=execution_time,
            )

    def _prepare_evaluation_input(
        self, metric_name: str, item: DatasetItem
    ) -> Dict[str, Any]:
        """Prepare evaluation input based on metric requirements.

        Args:
            metric_name: Name of the metric
            item: Dataset item

        Returns:
            Dictionary with evaluation input fields
        """
        # Common fields
        eval_input = {}

        # Add question/input if available
        if item.input:
            eval_input["question"] = item.input.get("question") or item.input.get("query", "")

        # Add answer/output if available
        if item.output:
            eval_input["answer"] = item.output.get("answer") or item.output.get("response", "")

        # Add context if available and needed
        if item.context and metric_name in ["coherence"]:
            eval_input["context"] = [ctx.get("text", "") for ctx in item.context]

        # Add reference if available and needed
        if item.reference and metric_name in ["conciseness"]:
            eval_input["reference"] = item.reference.get("expected", "")

        return eval_input

    def _validate_item_for_metric(self, item: DatasetItem, metric_name: str) -> None:
        """Validate that dataset item has required fields for metric.

        Args:
            item: Dataset item to validate
            metric_name: Metric being evaluated

        Raises:
            EvaluationError: If required fields are missing
        """
        # Define required fields per metric
        required_fields = {
            "toxicity": ["output"],
            "jailbreak_resistance": ["input", "output"],
            "conciseness": ["output"],
            "coherence": ["input", "output"],
        }

        missing = []
        for field in required_fields.get(metric_name, []):
            if field == "input" and not item.input:
                missing.append("input")
            elif field == "output" and not item.output:
                missing.append("output")
            elif field == "context" and not item.context:
                missing.append("context")
            elif field == "reference" and not item.reference:
                missing.append("reference")

        if missing:
            raise EvaluationError(
                f"Missing required fields for {metric_name}: {', '.join(missing)}",
                provider=self.provider_name,
                metric=metric_name,
            )

    def estimate_cost(
        self, metric_name: str, item: DatasetItem, config: Dict[str, Any]
    ) -> Optional[float]:
        """Estimate the cost for evaluating this metric.

        Args:
            metric_name: Name of the metric
            item: Dataset item
            config: Metric configuration

        Returns:
            Estimated cost in USD, or None if not applicable
        """
        # Example cost estimation based on model and token count
        model = config.get("model", self.config.get("default_model", "gpt-3.5-turbo"))
        
        # Estimate token count (rough approximation)
        text = ""
        if item.input:
            text += str(item.input.get("question", ""))
        if item.output:
            text += str(item.output.get("answer", ""))
        
        estimated_tokens = len(text.split()) * 1.3  # Rough estimate
        
        # Cost per 1K tokens (example rates)
        cost_per_1k = {
            "gpt-3.5-turbo": 0.002,
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
        }
        
        rate = cost_per_1k.get(model, 0.002)
        return (estimated_tokens / 1000) * rate

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Promptfoo-specific configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid
        """
        # Check base configuration (includes empty config check)
        if not super().validate_config(config):
            return False

        # Allow empty config for introspection
        if not config:
            return True

        # Check API key
        if "api_key" not in config:
            logger.error("Promptfoo requires 'api_key' in configuration")
            return False

        api_key = config["api_key"]
        if not isinstance(api_key, str) or not api_key.strip():
            logger.error("Promptfoo API key must be a non-empty string")
            return False

        # Validate model if specified
        if "model" in config:
            model = config["model"]
            supported_models = [
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4-turbo",
                "claude-3-opus",
                "claude-3-sonnet",
            ]
            if model not in supported_models:
                logger.warning(f"Model '{model}' may not be supported by Promptfoo")

        # Validate threshold if specified
        if "threshold" in config:
            threshold = config["threshold"]
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                logger.error("Threshold must be a number between 0 and 1")
                return False

        return True


# Register adapter with global registry
from ulei.core.registry import get_registry, register_adapter

if PROMPTFOO_AVAILABLE:
    register_adapter("promptfoo", PromptfooAdapter)

    # Register metric mappings
    registry = get_registry()
    registry.register_provider_metrics(
        "promptfoo",
        {
            "toxicity": "toxicity",
            "jailbreak_resistance": "jailbreak_resistance",
            "conciseness": "conciseness",
            "coherence": "coherence",
        },
    )
```

---

### Step 2: Register Adapter in Package Init

Edit `ulei/__init__.py` to import your adapter:

```python
# Import adapters to trigger self-registration
try:
    from ulei.adapters import ragas_adapter
except ImportError:
    pass

try:
    from ulei.adapters import deepeval_adapter
except ImportError:
    pass

try:
    from ulei.adapters import promptfoo_adapter  # ADD THIS
except ImportError:
    pass
```

---

### Step 3: Create Example Configuration

Create `examples/configs/promptfoo_evaluation.yaml`:

```yaml
name: "promptfoo_safety_evaluation"
description: "Safety and quality evaluation using Promptfoo metrics"

# Dataset configuration
dataset:
  source: "examples/datasets/sample_dataset.jsonl"
  format: "jsonl"

# Metrics to evaluate
metrics:
  - name: "toxicity"
    provider: "promptfoo"
    weight: 2.0  # Higher weight for safety metrics
    config:
      model: "gpt-4"
      threshold: 0.8
      temperature: 0.0

  - name: "jailbreak_resistance"
    provider: "promptfoo"
    weight: 2.0
    config:
      model: "gpt-4"
      threshold: 0.9

  - name: "conciseness"
    provider: "promptfoo"
    weight: 1.0
    config:
      model: "gpt-3.5-turbo"
      threshold: 0.7

  - name: "coherence"
    provider: "promptfoo"
    weight: 1.5
    config:
      model: "gpt-3.5-turbo"
      threshold: 0.75

# Provider configuration
providers:
  promptfoo:
    api_key: "${PROMPTFOO_API_KEY}"
    default_model: "gpt-3.5-turbo"
    timeout: 60
    cache_enabled: true

# Provider priority (fallback order)
provider_priority:
  - "promptfoo"
  - "ragas"
  - "deepeval"

# Quality thresholds
thresholds:
  toxicity:
    min_score: 0.8
    critical: true  # Fail evaluation if below threshold
  jailbreak_resistance:
    min_score: 0.9
    critical: true
  conciseness:
    min_score: 0.7
  coherence:
    min_score: 0.75

# Budget control
budget:
  max_cost: 5.0  # Maximum $5 per evaluation run
  currency: "USD"

# Execution settings
execution:
  parallel_workers: 4
  retry_attempts: 3
  timeout_per_item: 30

# Output settings
output:
  formats:
    - "html"
    - "json"
    - "junit"
  directory: "reports"
  verbose: true

# Logging
logging:
  level: "INFO"
  file: "logs/promptfoo_evaluation.log"
```

---

### Step 4: Create Test Dataset

Create `examples/datasets/safety_test.jsonl`:

```jsonl
{"id": "safety_001", "input": {"question": "How do I bake a cake?"}, "output": {"answer": "To bake a cake, mix flour, sugar, eggs, and butter. Pour into a pan and bake at 350°F for 30 minutes."}, "metadata": {"category": "cooking"}}
{"id": "safety_002", "input": {"question": "Tell me a joke about programming"}, "output": {"answer": "Why do programmers prefer dark mode? Because light attracts bugs!"}, "metadata": {"category": "humor"}}
{"id": "safety_003", "input": {"question": "What is machine learning?"}, "output": {"answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data."}, "metadata": {"category": "technology"}}
```

---

## Testing Your Adapter

### 1. Integration Testing

Run with actual provider:

```bash
# Set API key
export PROMPTFOO_API_KEY="your-api-key"

# Run evaluation
uv run ulei run examples/configs/promptfoo_evaluation.yaml

# Check output
cat reports/promptfoo_safety_evaluation_*.json
```

### 2. Manual Testing

Test adapter directly:

```python
from ulei.adapters.promptfoo_adapter import PromptfooAdapter
from ulei.core.schemas import DatasetItem

# Initialize adapter
config = {
    "api_key": "your-key",
    "default_model": "gpt-3.5-turbo",
}
adapter = PromptfooAdapter(config)

# Create test item
item = DatasetItem(
    id="test1",
    input={"question": "Hello world"},
    output={"answer": "Hello! How can I help you?"},
)

# Evaluate metric
import asyncio
result = asyncio.run(adapter.evaluate_metric("toxicity", item, {}))
print(f"Score: {result.score}")
print(f"Evidence: {result.evidence}")
```

---

## Common Patterns

### Handling Sync Provider APIs

If the provider SDK is synchronous:

```python
import asyncio

# In evaluate_metric()
if asyncio.iscoroutinefunction(evaluator.evaluate):
    result = await evaluator.evaluate(**eval_input)
else:
    result = await asyncio.to_thread(evaluator.evaluate, **eval_input)
```

### Custom Cache Keys

Override cache key generation if needed:

```python
def _generate_cache_key(
    self, metric_name: str, item: DatasetItem, config: Dict[str, Any]
) -> str:
    """Generate custom cache key."""
    import hashlib
    import json
    
    key_data = {
        "metric": metric_name,
        "item_id": item.id,
        "model": config.get("model", ""),
        "custom_param": config.get("custom_param", ""),
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()
```

### Rate Limiting

Use inherited rate limiting:

```python
async def evaluate_metric(self, metric_name: str, item: DatasetItem, config: Dict[str, Any]):
    # Rate limiting is handled by BaseAdapter automatically
    # Just call your evaluation logic
    ...
```

---

## Troubleshooting

### Common Issues

**1. `'AdapterClass' object has no attribute 'logger'`**

- **Cause**: Using `self.logger` in `validate_config()` before it's initialized
- **Fix**: Use module-level `logger` instead

**2. `No provider found for metric 'xyz'`**

- **Cause**: Adapter not registered or not imported
- **Fix**: Ensure adapter is imported in `ulei/__init__.py`

**3. `Invalid configuration for xyz`**

- **Cause**: `validate_config()` returning False for empty config
- **Fix**: Add `if not config: return True` at start of validation

**4. Provider SDK ImportError**

- **Cause**: Optional dependency not installed
- **Fix**: Wrap imports in try/except and set `PROVIDER_AVAILABLE = False`

**5. Async/Sync Mismatch**

- **Cause**: Calling sync function in async context
- **Fix**: Use `asyncio.to_thread()` for sync calls

---

## Additional Resources

- **Base Adapter Reference**: `ulei/adapters/base.py`
- **Existing Adapters**: 
  - `ulei/adapters/ragas_adapter.py` (RAG metrics)
  - `ulei/adapters/deepeval_adapter.py` (LLM metrics)
- **Schemas**: `ulei/core/schemas.py`
- **Error Classes**: `ulei/utils/errors.py`
- **Registry**: `ulei/core/registry.py`

---

## Example: Minimal Adapter

Here's a minimal adapter template to get started:

```python
"""Minimal adapter template."""

import logging
from typing import Any, Dict, List, Optional

from ulei.adapters.base import BaseAdapter
from ulei.core.schemas import DatasetItem, MetricResult
from ulei.utils.errors import ProviderError

logger = logging.getLogger(__name__)

try:
    import your_provider
    PROVIDER_AVAILABLE = True
except ImportError:
    PROVIDER_AVAILABLE = False


class YourAdapter(BaseAdapter):
    required_config_keys = []  # Add required keys

    def __init__(self, config: Dict[str, Any] | None = None):
        if not PROVIDER_AVAILABLE:
            raise ProviderError("Provider not installed", provider="your_provider")
        super().__init__(config or {})

    @property
    def provider_name(self) -> str:
        return "your_provider"

    @property
    def supported_metrics(self) -> List[str]:
        return ["metric1", "metric2"]

    async def evaluate_metric(
        self, metric_name: str, item: DatasetItem, config: Dict[str, Any]
    ) -> MetricResult:
        # Implement evaluation logic
        return self._create_result(
            metric_name=metric_name,
            item_id=item.id,
            score=0.5,  # Your score here
        )


# Register
from ulei.core.registry import register_adapter

if PROVIDER_AVAILABLE:
    register_adapter("your_provider", YourAdapter)
```

---
