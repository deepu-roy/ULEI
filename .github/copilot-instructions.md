# ULEI Development Guide

**ULEI (Unified LLM Evaluation Interface)** - Provider-agnostic evaluation framework for LLM and RAG systems. Think "pytest + Terraform-providers" for AI evaluation.

## Architecture Overview

ULEI uses an **adapter pattern** to normalize evaluation across heterogeneous providers (Ragas, DeepEval, local metrics):

```
Dataset (JSONL) → Evaluator → Registry → Adapters (Ragas/DeepEval/Local) → Normalized MetricResult
                      ↓
                  Reporters (HTML/JSON/JUnit/Prometheus)
```

**Key design decisions:**
- **Provider abstraction**: All adapters inherit from `ulei.core.interfaces.BaseAdapter` and return standardized `MetricResult` objects
- **Registry pattern**: `ulei.core.registry.MetricRegistry` maps logical metric names (e.g., "faithfulness") to provider-specific implementations
- **Budget control**: `ulei.utils.budget.BudgetManager` enforces cost limits to prevent API overspending
- **Caching by default**: Adapters cache results by generating cache keys from `hash(metric_name + item.id + config)`

## Project Structure

```
ulei/
  core/           # Core orchestration (evaluator, schemas, registry, interfaces)
  adapters/       # Provider implementations (ragas_adapter.py, deepeval_adapter.py)
  cli/            # Click-based CLI (run.py, compare.py, server.py)
  http/           # FastAPI server for online shadow evaluation
  reporters/      # Output formatters (html.py, json.py, junit.py, prometheus.py)
  utils/          # Helpers (config.py, dataset.py, budget.py, cache.py, retry.py)
  metrics/local/  # Provider-free metrics (BERT-Score, ROUGE, BLEU)
examples/configs/ # Reference YAML configurations
specs/            # Feature specifications and contracts
```

## Development Workflow

### Setup
```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"

# Run linting
ruff check .
ruff format .
```

### Running Evaluations
```bash
# Basic run
ulei run examples/configs/basic_evaluation.yaml

# With specific output formats
ulei run config.yaml --format json html

# CI/CD mode (exits non-zero on threshold failures)
ulei run config.yaml --fail-on-threshold

# Compare two evaluation runs
ulei compare --baseline reports/run1.json --candidate reports/run2.json
```

### Testing
```bash
# Run all tests (when implemented)
pytest

# Run with coverage
pytest --cov=ulei --cov-report=html
```

## Configuration Patterns

### Environment Variable Substitution
All YAML configs support `${ENV_VAR}` syntax. See `ulei.utils.config.ConfigLoader._substitute_env_vars()`.

Example:
```yaml
providers:
  ragas:
    api_key: "${OPENAI_API_KEY}"  # Reads from environment
```

### Dataset Format (JSONL)
Standard structure in `ulei.core.schemas.DatasetItem`:
```jsonl
{"id": "1", "input": {"query": "..."}, "output": {"answer": "..."}, "reference": {"expected": "..."}, "context": [{"text": "...", "source": "doc1"}]}
```

**Critical fields:**
- `id`: Required, unique identifier
- `input`: Required, query/prompt data
- `output`: Required, model response
- `reference`: Optional, ground truth for comparison metrics
- `context`: Optional but required for RAG metrics (faithfulness, context_precision)

### Provider Priority & Fallback
When metric doesn't specify provider, registry uses `provider_priority` from config:
```yaml
provider_priority:
  - "ragas"      # Try Ragas first
  - "deepeval"   # Fallback to DeepEval
```

See `ulei.core.registry.AdapterRegistry.resolve_adapter()` for fallback logic.

## Adding New Components

### New Adapter (Provider)
1. Inherit from `ulei.adapters.base.BaseAdapter`
2. Implement `evaluate_metric()` and `supported_metrics()`
3. Register in `ulei.core.registry` via `register_adapter()`
4. Add provider config to YAML schema

Example stub:
```python
class MyProviderAdapter(BaseAdapter):
    @property
    def provider_name(self) -> str:
        return "myprovider"
    
    async def evaluate_metric(self, metric_name: str, item: DatasetItem, config: Dict[str, Any]) -> MetricResult:
        # Call provider API, normalize response
        return MetricResult(metric=metric_name, provider=self.provider_name, ...)
```

### New Reporter
1. Inherit from `ulei.reporters.base.BaseReporter`
2. Implement `generate_report(run: EvaluationRun, output_path: Path)`
3. Register format in CLI options (`ulei/cli/run.py`)

### New Local Metric
Add to `ulei/metrics/local/` - these require no API calls. See BERT-Score implementation as reference.

## Common Patterns

### Async Evaluation with Retry
All adapters use `evaluate_metric_with_retry()` from `BaseAdapter`:
```python
result = await adapter.evaluate_metric_with_retry(
    metric_name="faithfulness",
    item=dataset_item,
    config={"model": "gpt-3.5-turbo"},
    max_retries=3,
    backoff_factor=2.0
)
```

### Budget Enforcement
`Evaluator` checks budget before each batch:
```python
self.budget_manager.check_budget(estimated_cost)  # Raises BudgetExceededError if over limit
```

### Cache Key Generation
Adapters generate deterministic cache keys:
```python
cache_key = hashlib.sha256(f"{metric_name}:{item.id}:{json.dumps(config, sort_keys=True)}".encode()).hexdigest()
```

## Error Handling

Custom exceptions in `ulei.utils.errors`:
- `ConfigurationError`: Invalid YAML or missing required fields
- `DatasetError`: Dataset loading/parsing failures
- `BudgetExceededError`: Cost limit exceeded
- `ProviderError`: Provider API failures
- `EvaluationError`: Metric evaluation failures

**Pattern**: Adapters catch provider-specific errors and wrap in `EvaluationError` with original traceback.

## CLI Entry Points

Defined in `pyproject.toml`:
```toml
[project.scripts]
ulei = "ulei.cli.main:main"
```

Commands implemented in `ulei/cli/`:
- `run`: Execute evaluation suite
- `compare`: Statistical comparison of two runs
- `server`: Start FastAPI server for online evaluation

## HTTP Server Implementation (Online Shadow Evaluation)

**User-facing API documentation**: See `docs/HTTP_API.md` for endpoint details and integration examples.

### Internal Architecture

**Request flow** (`ulei/http/server.py`):
1. FastAPI receives event via POST → `ULEIServer.ingest_event()`
2. Validates `EvaluationEvent` and converts to `DatasetItem` via `_event_to_dataset_item()`
3. Creates `QueuedEvent` with priority and queues in `EventQueue`
4. Updates `event_statuses` dict for tracking
5. Triggers background processing task if not running
6. Background task polls queue, processes batches, stores results in `completed_reports`

**Key implementation patterns**:
- **Event ID generation**: Auto-generated using `f"evt_{uuid.uuid4().hex[:12]}"` if not provided
- **Queue processing**: Runs as FastAPI `BackgroundTasks`, not a separate thread
- **Validation limits**: Batch endpoint enforces `max_items=100` via Pydantic schema
- **Status tracking**: In-memory dicts (`event_statuses`, `completed_reports`) - no persistence
- **Error wrapping**: Pydantic `ValidationError` → 400, generic `Exception` → 500

**Files to reference**:
- `ulei/http/server.py`: Main FastAPI app and endpoints
- `ulei/http/models.py`: Request/response Pydantic schemas
- `ulei/http/queue.py`: `EventQueue` and `QueuedEvent` implementation
- `specs/001-unified-eval-interface/contracts/http-api.yaml`: OpenAPI spec

## Dependencies

**Core runtime** (Python 3.9+):
- Pydantic 2.x for schemas with validation
- Click for CLI
- FastAPI + Uvicorn for HTTP server
- Ragas + DeepEval as evaluation providers
- SciPy for statistical testing in comparisons

**Development**:
- Ruff for linting/formatting (replaces Black + isort + flake8)
- pytest + pytest-asyncio for testing
- mypy for type checking

## Coding Conventions

- **Type hints required**: All function signatures must have types
- **Pydantic for validation**: Use `Field()` with constraints, not manual validation
- **Async by default**: All evaluation methods are async for parallel execution
- **Logging over print**: Use `logger.info/debug/error` from module-level logger
- **Line length**: 100 characters (configured in `pyproject.toml`)

## Key Files to Reference

- `ulei/core/schemas.py`: All Pydantic models (start here for data structures)
- `ulei/core/evaluator.py`: Main evaluation orchestration logic
- `ulei/core/registry.py`: Provider resolution and metric mapping
- `ulei/adapters/base.py`: Adapter base class with retry/cache logic
- `examples/configs/basic_evaluation.yaml`: Reference configuration

## CI/CD Integration

Exit codes:
- `0`: All metrics pass thresholds
- `1`: One or more metrics below threshold (when `--fail-on-threshold` enabled)
- `2`: Evaluation errors or configuration issues

JUnit XML output for CI tools:
```bash
ulei run config.yaml --format junit --output-dir ./test-results
```

---

**Last updated**: 2025-11-05  
**Spec reference**: `specs/001-unified-eval-interface/`
