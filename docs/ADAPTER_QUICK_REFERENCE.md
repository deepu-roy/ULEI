# Adapter Development Quick Reference

Quick reference for adding new evaluation provider adapters to ULEI.

## Minimal Adapter Template

```python
"""Provider adapter for [PROVIDER_NAME]."""

import logging
from typing import Any, Dict, List

from ulei.adapters.base import BaseAdapter
from ulei.core.schemas import DatasetItem, MetricResult
from ulei.utils.errors import ProviderError

logger = logging.getLogger(__name__)

try:
    import provider_sdk
    PROVIDER_AVAILABLE = True
except ImportError:
    PROVIDER_AVAILABLE = False


class MyProviderAdapter(BaseAdapter):
    """Adapter for MyProvider evaluation provider."""
    
    required_config_keys = ["api_key"]  # or [] if none required

    def __init__(self, config: Dict[str, Any] | None = None):
        if not PROVIDER_AVAILABLE:
            raise ProviderError(
                "Provider not installed. Install with: pip install provider-sdk",
                provider="myprovider"
            )
        super().__init__(config or {})
        self._setup_provider()

    def _setup_provider(self) -> None:
        """Setup provider configuration."""
        # Initialize provider SDK, set API keys, etc.
        pass

    @property
    def provider_name(self) -> str:
        return "myprovider"

    @property
    def supported_metrics(self) -> List[str]:
        return ["metric1", "metric2", "metric3"]

    async def evaluate_metric(
        self, metric_name: str, item: DatasetItem, config: Dict[str, Any]
    ) -> MetricResult:
        """Evaluate a single metric."""
        import time
        start_time = time.time()
        
        try:
            # 1. Validate metric support
            if not self.supports_metric(metric_name):
                raise MetricNotSupportedError(
                    f"Metric '{metric_name}' not supported",
                    provider=self.provider_name,
                    metric=metric_name,
                    supported_metrics=self.supported_metrics,
                )
            
            # 2. Call provider API
            result = await self._call_provider_api(metric_name, item, config)
            
            # 3. Return normalized result
            return self._create_result(
                metric_name=metric_name,
                item_id=item.id,
                score=result["score"],
                confidence=result.get("confidence"),
                evidence=result.get("evidence", {}),
                execution_time=time.time() - start_time,
                raw_response=result,
            )
            
        except Exception as e:
            # Handle errors
            return self._create_result(
                metric_name=metric_name,
                item_id=item.id,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    async def _call_provider_api(
        self, metric_name: str, item: DatasetItem, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call provider API and return raw result."""
        # Implement provider-specific API call
        # Use asyncio.to_thread() for sync APIs
        pass

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate provider configuration."""
        if not super().validate_config(config):
            return False
        
        if not config:  # Allow empty for introspection
            return True
        
        # Add provider-specific validation
        # Use logger (module-level), NOT self.logger
        return True


# Register adapter
from ulei.core.registry import get_registry, register_adapter

if PROVIDER_AVAILABLE:
    register_adapter("myprovider", MyProviderAdapter)
    
    # Register metric mappings
    registry = get_registry()
    registry.register_provider_metrics(
        "myprovider",
        {
            "metric1": "metric1",
            "metric2": "metric2",
            "metric3": "metric3",
        },
    )
```

## Critical Rules

### ✅ DO

1. **Import adapter in `ulei/__init__.py`**:
   ```python
   try:
       from ulei.adapters import myprovider_adapter
   except ImportError:
       pass
   ```

2. **Use module-level logger in `validate_config()`**:
   ```python
   logger = logging.getLogger(__name__)  # Module level
   
   def validate_config(self, config):
       if "api_key" not in config:
           logger.error("Missing API key")  # NOT self.logger
           return False
   ```

3. **Allow empty config for introspection**:
   ```python
   def validate_config(self, config):
       if not super().validate_config(config):
           return False
       if not config:  # Allow empty config
           return True
       # ... rest of validation
   ```

4. **Accept None config in __init__**:
   ```python
   def __init__(self, config: Dict[str, Any] | None = None):
       super().__init__(config or {})
   ```

5. **Handle sync APIs with asyncio.to_thread()**:
   ```python
   import asyncio
   result = await asyncio.to_thread(sync_function, arg1, arg2)
   ```

### ❌ DON'T

1. **Don't use `self.logger` in `validate_config()`** - It doesn't exist yet!

2. **Don't require config in `__init__`** - Registry needs to create instances for introspection

3. **Don't raise errors for missing API keys in empty configs** - Allow introspection

4. **Don't forget to register** - Import in `__init__.py` and call `register_adapter()`

5. **Don't call sync code directly in async methods** - Use `asyncio.to_thread()`

## File Locations

```
ulei/
├── __init__.py                    # Import adapter here
├── adapters/
│   ├── base.py                    # Inherit from BaseAdapter
│   ├── myprovider_adapter.py      # Your adapter file
│   ├── ragas_adapter.py           # Reference example
│   └── deepeval_adapter.py        # Reference example
└── core/
    └── registry.py                # Registration happens here

examples/
├── configs/
│   └── myprovider_evaluation.yaml # Example config
└── datasets/
    └── myprovider_test.jsonl      # Example dataset

tests/
└── test_myprovider_adapter.py     # Unit tests

docs/
└── ADDING_NEW_ADAPTER.md          # Full guide
```

## Configuration Example

```yaml
name: "my_evaluation"
description: "Evaluation using MyProvider"

dataset:
  source: "examples/datasets/test.jsonl"
  format: "jsonl"

metrics:
  - name: "metric1"
    provider: "myprovider"
    config:
      model: "gpt-4"
      threshold: 0.8

providers:
  myprovider:
    api_key: "${MY_PROVIDER_API_KEY}"
    default_model: "gpt-3.5-turbo"
    timeout: 60
    cache_enabled: true

thresholds:
  metric1:
    min_score: 0.7
```

## Testing Checklist

```bash
# 1. Test adapter can be instantiated with empty config
python -c "from ulei.adapters.myprovider_adapter import MyProviderAdapter; a = MyProviderAdapter({}); print(a.provider_name)"

# 2. Test supported metrics are reported
python -c "from ulei.adapters.myprovider_adapter import MyProviderAdapter; a = MyProviderAdapter({}); print(a.supported_metrics)"

# 3. Run unit tests
uv run pytest tests/test_myprovider_adapter.py -v

# 4. Run integration test
uv run ulei run examples/configs/myprovider_evaluation.yaml

# 5. Check for errors
uv run ulei run examples/configs/myprovider_evaluation.yaml 2>&1 | grep -i error
```

## Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `'Adapter' object has no attribute 'logger'` | Using `self.logger` in `validate_config()` | Use module-level `logger` instead |
| `No provider found for metric` | Adapter not registered | Import in `ulei/__init__.py` |
| `Invalid configuration` on empty config | `validate_config()` rejects empty config | Add `if not config: return True` |
| `ImportError` when adapter not installed | Missing try/except around SDK import | Wrap imports in try/except |
| `This coroutine should be awaited` | Calling sync function in async method | Use `asyncio.to_thread()` |

## Metric Mapping Patterns

### Simple 1:1 Mapping
```python
registry.register_provider_metrics(
    "myprovider",
    {
        "toxicity": "toxicity",
        "coherence": "coherence",
    },
)
```

### Custom Logical Names
```python
registry.register_provider_metrics(
    "myprovider",
    {
        "my_toxicity": "toxicity",  # Logical: my_toxicity, Provider: toxicity
        "my_coherence": "coherence",
    },
)
```

## DatasetItem Fields

Common field access patterns:

```python
# Question/Query
question = item.input.get("question") or item.input.get("query", "")

# Answer/Response
answer = item.output.get("answer") or item.output.get("response", "")

# Context for RAG
contexts = [ctx.get("text", "") for ctx in item.context] if item.context else []

# Reference/Expected
expected = item.reference.get("expected", "") if item.reference else ""
```

## Next Steps

1. Read full guide: [ADDING_NEW_ADAPTER.md](ADDING_NEW_ADAPTER.md)
2. Study existing adapters: `ulei/adapters/ragas_adapter.py`
3. Copy minimal template above
4. Implement your adapter
5. Test thoroughly
6. Submit PR!

---

**Need help?** Check the [full guide](ADDING_NEW_ADAPTER.md) or review existing adapters in `ulei/adapters/`.
