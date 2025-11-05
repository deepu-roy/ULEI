# ULEI Quick Reference

## CLI Commands

### Run Evaluation

```bash
# Basic
ulei run config.yaml

# With options
ulei run config.yaml --output-dir ./reports --format json html --parallel-workers 4

# Dry run (validate only)
ulei run config.yaml --dry-run

# CI/CD mode (exit 1 on failure)
ulei run config.yaml --fail-on-threshold

# Verbose output
ulei run config.yaml -vvv

# With caching
ulei run config.yaml --cache --cache-ttl 3600

# With retry
ulei run config.yaml --retry-attempts 5 --timeout 60
```

### Compare Results

```bash
# Basic comparison
ulei compare --baseline run1.json --candidate run2.json

# With output
ulei compare --baseline run1.json --candidate run2.json --output comparison.html

# With significance testing
ulei compare --baseline run1.json --candidate run2.json --significance-level 0.05

# Multiple formats
ulei compare --baseline run1.json --candidate run2.json --format html json
```

### Start Server

```bash
# Basic
ulei server --config eval.yaml

# With options
ulei server --config eval.yaml --port 8080 --workers 4

# With host binding
ulei server --config eval.yaml --host 0.0.0.0 --port 8080
```

### Health Check

```bash
# Check server health
ulei health --host localhost --port 8080
```

### Trend Analysis

```bash
# Analyze trends
ulei trend --reports run_*.json --metric faithfulness --output trend.html
```

## Configuration Templates

### Minimal Config

```yaml
name: "eval_name"
dataset:
  source: "data.jsonl"
  format: "jsonl"
metrics:
  - name: "faithfulness"
    provider: "ragas"
providers:
  ragas:
    api_key: "${OPENAI_API_KEY}"
```

### Full Config

```yaml
name: "comprehensive_eval"
description: "Full evaluation with all options"

dataset:
  source: "data.jsonl"
  format: "jsonl"

metrics:
  - name: "faithfulness"
    provider: "ragas"
    config:
      model: "gpt-3.5-turbo"
      temperature: 0.1
  
  - name: "answer_relevancy"
    provider: "ragas"
  
  - name: "context_precision"
    provider: "ragas"

providers:
  ragas:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-3.5-turbo"
    temperature: 0.1

thresholds:
  faithfulness: 0.8
  answer_relevancy: 0.7
  context_precision: 0.75

budget_limit: 50.0

retry_policy:
  max_retries: 3
  backoff_factor: 2.0
  timeout: 30

cache:
  enabled: true
  backend: "sqlite"
  ttl: 3600

parallel_workers: 4
output_formats: ["json", "html"]
output_dir: "reports"
```

### Provider Configs

#### OpenAI via Ragas

```yaml
providers:
  ragas:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
    temperature: 0.0
    max_tokens: 2000
```

#### DeepEval

```yaml
providers:
  deepeval:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-3.5-turbo"
```

#### Local Metrics

```yaml
metrics:
  - name: "bert_score"
    provider: "local"
  - name: "rouge"
    provider: "local"
  - name: "bleu"
    provider: "local"
```

## Dataset Formats

### JSONL (Recommended)

```jsonl
{"id": "1", "input": {"query": "..."}, "output": {"answer": "..."}, "reference": {"expected": "..."}, "context": [{"text": "...", "source": "..."}]}
```

### Minimal Dataset

```jsonl
{"id": "1", "input": {"query": "What is ML?"}, "output": {"answer": "ML is..."}}
{"id": "2", "input": {"query": "What is DL?"}, "output": {"answer": "DL is..."}}
```

### Full Dataset

```jsonl
{"id": "1", "input": {"query": "What is RAG?"}, "output": {"answer": "RAG is...", "confidence": 0.9}, "reference": {"expected": "Retrieval..."}, "context": [{"text": "RAG combines...", "source": "doc1.pdf", "score": 0.95}]}
```

## Metrics Reference

### Ragas Metrics

| Metric | Range | Description | Requires Context | Requires Reference |
|--------|-------|-------------|-----------------|-------------------|
| `faithfulness` | 0-1 | Answer faithful to context | Yes | No |
| `answer_relevancy` | 0-1 | Answer relevant to query | No | No |
| `context_precision` | 0-1 | Retrieved context precision | Yes | Yes |
| `context_recall` | 0-1 | Context covers reference | Yes | Yes |

### DeepEval Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| `g_eval` | 0-10 | Custom criteria evaluation |
| `summarization` | 0-1 | Summary quality |
| `toxicity` | 0-1 | Toxic content detection |
| `bias` | 0-1 | Bias detection |

### Local Metrics

| Metric | Range | Description | Requires Reference |
|--------|-------|-------------|-------------------|
| `bert_score` | 0-1 | Semantic similarity | Yes |
| `bleu` | 0-1 | N-gram precision | Yes |
| `rouge` | 0-1 | N-gram recall | Yes |

## Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."

# Paths
export DATA_PATH="/path/to/data"
export OUTPUT_DIR="/path/to/reports"

# Use in config
# api_key: "${OPENAI_API_KEY}"
# source: "${DATA_PATH}/data.jsonl"
```

## Common Patterns

### CI/CD Integration

```yaml
# .github/workflows/eval.yml
- name: Evaluate
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    uv run python -m ulei.cli.main run eval.yaml --fail-on-threshold
```

### A/B Testing

```bash
# Run both versions
ulei run version_a.yaml --output-dir reports/a
ulei run version_b.yaml --output-dir reports/b

# Compare
ulei compare --baseline reports/a/*.json --candidate reports/b/*.json
```

### Batch Processing

```bash
# Process multiple configs
for config in configs/*.yaml; do
  ulei run "$config" --output-dir "reports/$(basename $config .yaml)"
done
```

### Progressive Testing

```bash
# Quick smoke test
ulei run eval_small.yaml --parallel-workers 1 --dry-run

# Full evaluation
ulei run eval_full.yaml --parallel-workers 8 --cache
```

## Python API Quick Reference

### Basic Usage

```python
from ulei.core.evaluator import Evaluator
from ulei.utils.config import ConfigLoader

# Load config
loader = ConfigLoader()
suite = loader.load_suite("config.yaml")

# Create evaluator
evaluator = Evaluator()

# Load dataset
items = evaluator.load_dataset("data.jsonl")

# Run evaluation
import asyncio
report = asyncio.run(evaluator.run_evaluation(suite, items))
```

### Create Dataset Programmatically

```python
from ulei.core.schemas import DatasetItem

items = [
    DatasetItem(
        item_id="1",
        input_data={"query": "What is ML?"},
        output_data={"answer": "ML is..."},
        reference_data={"expected": "Machine learning..."},
        context=[{"text": "ML context...", "source": "doc1"}]
    )
]
```

### Access Results

```python
# Aggregated scores
print(report.aggregates["faithfulness"]["mean"])

# Individual results
for result in report.results:
    print(f"{result.item_id}: {result.metric} = {result.score}")

# Threshold status
for metric, status in report.threshold_status.items():
    print(f"{metric}: {'✓' if status['passed'] else '✗'}")

# Cost summary
print(f"Total cost: ${report.cost_summary['total_cost']:.2f}")
```

## Troubleshooting Quick Fixes

### Rate Limiting

```yaml
retry_policy:
  max_retries: 5
  backoff_factor: 3.0
parallel_workers: 2  # Reduce parallelism
```

### Out of Memory

```yaml
parallel_workers: 1
batch_size: 5
cache:
  backend: "sqlite"  # Use disk instead of memory
```

### Timeout Issues

```yaml
retry_policy:
  timeout: 120  # Increase timeout
metrics:
  - name: "faithfulness"
    config:
      max_tokens: 500  # Reduce token limit
```

### API Key Errors

```bash
# Check environment variable
echo $OPENAI_API_KEY

# Set temporarily
export OPENAI_API_KEY="sk-..."

# Or use .env file
# .env
OPENAI_API_KEY=sk-...
```

## Output Files

### JSON Report

```
reports/
  └── eval_name_abc123.json      # Full results
```

### HTML Report

```
reports/
  └── eval_name_abc123.html      # Interactive dashboard
```

### Multiple Formats

```
reports/
  ├── eval_name_abc123.json      # Raw data
  ├── eval_name_abc123.html      # Dashboard
  └── eval_name_abc123.junit.xml # CI/CD integration
```

## Performance Tips

1. **Use caching**: `--cache --cache-ttl 3600`
2. **Parallel execution**: `--parallel-workers 8`
3. **Batch processing**: Split large datasets
4. **Local metrics first**: Test with BERT/BLEU before API calls
5. **Dry run validation**: Always use `--dry-run` first
6. **Progressive scaling**: Start small, scale up
7. **Monitor costs**: Set `budget_limit` in config

## Best Practices

1. **Version control configs**: Store configs in git
2. **Use environment variables**: Never hardcode API keys
3. **Set thresholds**: Define acceptable quality levels
4. **Compare baselines**: Track changes over time
5. **Cache results**: Avoid redundant API calls
6. **Log verbosely in CI**: Use `-vvv` for debugging
7. **Test incrementally**: Start with small datasets
8. **Document metrics**: Explain why you chose specific metrics

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `API key not found` | Missing env variable | `export OPENAI_API_KEY="sk-..."` |
| `Dataset file not found` | Wrong path | Check file path in config |
| `No provider for metric` | Provider not configured | Add provider to config |
| `Threshold not met` | Low scores | Check `--fail-on-threshold` flag |
| `Rate limit exceeded` | Too many requests | Reduce `parallel_workers` |
| `Budget exceeded` | Costs too high | Increase `budget_limit` |
| `Invalid YAML` | Syntax error | Validate YAML syntax |

## Keyboard Shortcuts (Interactive Mode)

| Key | Action |
|-----|--------|
| `Ctrl+C` | Cancel running evaluation |
| `q` | Quit interactive prompt |

## Version Information

```bash
# Check version
uv run python -m ulei.cli.main --version

# Check installed providers
uv run python -c "from ulei.core.registry import get_registry; print(get_registry().list_providers())"
```

## Getting Help

```bash
# General help
ulei --help

# Command help
ulei run --help
ulei compare --help
ulei server --help

# Verbose output
ulei run config.yaml -vvv
```

## Links

- Documentation: `docs/`
- Examples: `examples/`
- GitHub: https://github.com/yourusername/ulei
- Issues: https://github.com/yourusername/ulei/issues
