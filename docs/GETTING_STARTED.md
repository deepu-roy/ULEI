# Getting Started with ULEI

This guide will help you get started with the Unified LLM Evaluation Interface (ULEI) framework for evaluating your LLM and RAG systems.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Configuration Guide](#configuration-guide)
- [Dataset Preparation](#dataset-preparation)
- [Running Evaluations](#running-evaluations)
- [Understanding Results](#understanding-results)
- [Advanced Usage](#advanced-usage)

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or uv package manager

### Install ULEI

**Using uv (recommended):**

```bash
uv add ulei
```

**Using pip:**

```bash
pip install ulei
```

**From source:**

```bash
git clone https://github.com/yourusername/unified-llm-evaluation-framework
cd unified-llm-evaluation-framework
uv sync
```

### Verify Installation

```bash
uv run python -m ulei.cli.main --version
```

## Quick Start

Let's evaluate a simple RAG system in 3 steps:

### Step 1: Create Your Dataset

Create a file `my_data.jsonl` with your evaluation data:

```jsonl
{"id": "1", "input": {"query": "What is machine learning?"}, "output": {"answer": "Machine learning is a subset of AI that enables systems to learn from data."}, "reference": {"expected": "ML is a method of data analysis that automates model building"}, "context": [{"text": "Machine learning is a branch of artificial intelligence...", "source": "ml_guide.pdf"}]}
{"id": "2", "input": {"query": "What is deep learning?"}, "output": {"answer": "Deep learning uses neural networks with multiple layers."}, "reference": {"expected": "Deep learning is a subset of ML using multi-layer neural networks"}, "context": [{"text": "Deep learning is a subset of machine learning...", "source": "dl_intro.pdf"}]}
```

### Step 2: Create Your Configuration

Create `my_eval.yaml`:

```yaml
name: "my_first_evaluation"
description: "Evaluating my RAG system"

# Dataset location
dataset:
  source: "my_data.jsonl"
  format: "jsonl"

# Metrics to evaluate
metrics:
  - name: "faithfulness"
    provider: "ragas"
    config:
      model: "gpt-3.5-turbo"
      temperature: 0.1
  
  - name: "answer_relevancy"
    provider: "ragas"
    config:
      model: "gpt-3.5-turbo"

# Provider configuration
providers:
  ragas:
    api_key: "${OPENAI_API_KEY}"  # Reads from environment variable

# Quality thresholds
thresholds:
  faithfulness: 0.7
  answer_relevancy: 0.7

# Output configuration
output_formats: ["json", "html"]
output_dir: "reports"
```

### Step 3: Run the Evaluation

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Run evaluation
uv run python -m ulei.cli.main run my_eval.yaml

# Or with custom output directory
uv run python -m ulei.cli.main run my_eval.yaml --output-dir ./my_reports
```

That's it! You'll find your results in the `reports/` directory.

## Core Concepts

### Evaluation Suite

An **Evaluation Suite** is a complete configuration that defines:

- Which metrics to evaluate
- Which providers to use
- Quality thresholds
- Dataset location
- Output preferences

### Dataset Items

Each **DatasetItem** represents one evaluation case with:

- `id`: Unique identifier
- `input`: The input to your system (e.g., query, prompt)
- `output`: Your system's actual output
- `reference` (optional): Expected/ground truth output
- `context` (optional): Retrieved context for RAG systems

### Metrics

**Metrics** are the evaluation criteria. ULEI supports:

**RAG Metrics (via Ragas):**

- `faithfulness`: Are outputs faithful to the context?
- `answer_relevancy`: Is the answer relevant to the query?
- `context_precision`: Is the context relevant?
- `context_recall`: Does context cover the expected answer?

**LLM Metrics (via DeepEval):**

- `g_eval`: Custom criteria evaluation
- `summarization`: Summary quality
- `toxicity`: Toxic content detection
- `bias`: Bias detection

**Local Metrics (no API required):**

- `bert_score`: Semantic similarity
- `bleu`: Precision-based similarity
- `rouge`: Recall-based similarity

### Providers

**Providers** are the evaluation backends:

- `ragas`: RAG-specific evaluations
- `deepeval`: General LLM evaluations
- `local`: Offline metrics (BERT-Score, BLEU, ROUGE)

## Configuration Guide

### Basic Configuration Structure

```yaml
# Required fields
name: "evaluation_name"
description: "What this evaluation does"

dataset:
  source: "path/to/data.jsonl"
  format: "jsonl"  # or "csv", "json", "parquet"

metrics:
  - name: "metric_name"
    provider: "provider_name"

# Optional fields
providers:
  provider_name:
    api_key: "${API_KEY}"
    model: "model-name"

thresholds:
  metric_name: 0.8

output_formats: ["json", "html"]
output_dir: "reports"
```

### Provider Configuration Examples

#### Ragas (RAG Evaluation)

```yaml
providers:
  ragas:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"  # or "gpt-3.5-turbo"
    temperature: 0.1
    max_tokens: 1000
```

#### DeepEval (LLM Evaluation)

```yaml
providers:
  deepeval:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-3.5-turbo"
    custom_rubric: |
      Evaluate the response on the following criteria:
      1. Accuracy (0-10)
      2. Completeness (0-10)
      3. Clarity (0-10)
```

#### Local Metrics (No API Key)

```yaml
metrics:
  - name: "bert_score"
    provider: "local"
    config:
      model: "bert-base-uncased"
  
  - name: "rouge"
    provider: "local"
    config:
      rouge_types: ["rouge1", "rouge2", "rougeL"]
```

### Advanced Configuration

#### Budget Control

```yaml
budget_limit: 50.0  # USD
budget_tracking: true
cost_per_1k_tokens:
  gpt-4: 0.03
  gpt-3.5-turbo: 0.002
```

#### Retry Policy

```yaml
retry_policy:
  max_retries: 3
  backoff_factor: 2.0
  timeout: 30
  retryable_errors: ["rate_limit", "timeout", "server_error"]
```

#### Caching

```yaml
cache:
  enabled: true
  backend: "sqlite"  # or "memory"
  ttl: 3600  # seconds
  cache_dir: ".cache/ulei"
```

#### Parallel Execution

```yaml
parallel_workers: 4
batch_size: 10
```

## Dataset Preparation

### Supported Formats

ULEI supports multiple dataset formats:

#### JSONL (Recommended)

```jsonl
{"id": "1", "input": {...}, "output": {...}, "reference": {...}, "context": [...]}
{"id": "2", "input": {...}, "output": {...}, "reference": {...}, "context": [...]}
```

#### JSON

```json
[
  {
    "id": "1",
    "input": {"query": "..."},
    "output": {"answer": "..."},
    "reference": {"expected": "..."},
    "context": [{"text": "...", "source": "..."}]
  }
]
```

#### CSV

```csv
id,query,answer,expected,context
1,"What is ML?","ML is...","Machine learning is...","ML guide text"
2,"What is DL?","DL is...","Deep learning is...","DL intro text"
```

#### Parquet

```python
import pandas as pd

df = pd.DataFrame({
    'id': ['1', '2'],
    'input': [{'query': '...'}, {'query': '...'}],
    'output': [{'answer': '...'}, {'answer': '...'}]
})
df.to_parquet('data.parquet')
```

### Dataset Schema

#### Minimum Required Fields

```python
{
    "id": "unique_identifier",
    "input": {"query": "user question"},
    "output": {"answer": "system response"}
}
```

#### Full Schema (for RAG)

```python
{
    "id": "item_001",
    "input": {
        "query": "What is the capital of France?",
        "additional_params": {}  # Any extra input data
    },
    "output": {
        "answer": "Paris is the capital of France.",
        "confidence": 0.95,  # Optional metadata
        "latency_ms": 250
    },
    "reference": {
        "expected": "Paris",
        "explanation": "Paris has been France's capital since..."
    },
    "context": [
        {
            "text": "Paris is the capital and largest city of France...",
            "source": "geography_book.pdf",
            "score": 0.92,  # Retrieval score
            "metadata": {}
        }
    ]
}
```

### Creating Test Data Programmatically

```python
import json

# Create dataset
dataset = [
    {
        "id": f"test_{i}",
        "input": {"query": f"Question {i}"},
        "output": {"answer": f"Answer {i}"},
        "reference": {"expected": f"Expected {i}"}
    }
    for i in range(100)
]

# Save as JSONL
with open('test_data.jsonl', 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')
```

## Running Evaluations

### Basic Evaluation

```bash
# Simple run
uv run python -m ulei.cli.main run config.yaml

# Dry run (validate without executing)
uv run python -m ulei.cli.main run config.yaml --dry-run
```

### Customizing Output

```bash
# Custom output directory
uv run python -m ulei.cli.main run config.yaml --output-dir ./results

# Multiple formats
uv run python -m ulei.cli.main run config.yaml --format json html

# All formats
uv run python -m ulei.cli.main run config.yaml --format all
```

### Performance Tuning

```bash
# Parallel execution
uv run python -m ulei.cli.main run config.yaml --parallel-workers 8

# With caching
uv run python -m ulei.cli.main run config.yaml --cache --cache-ttl 3600

# Retry configuration
uv run python -m ulei.cli.main run config.yaml --retry-attempts 5 --timeout 60
```

### CI/CD Integration

```bash
# Exit with error if thresholds not met
uv run python -m ulei.cli.main run config.yaml --fail-on-threshold

# Verbose output for debugging
uv run python -m ulei.cli.main run config.yaml -vvv
```

### Comparing Results

```bash
# Compare two evaluation runs
uv run python -m ulei.cli.main compare \
  --baseline reports/baseline_run.json \
  --candidate reports/new_run.json \
  --output comparison_report.html

# With statistical significance testing
uv run python -m ulei.cli.main compare \
  --baseline reports/baseline_run.json \
  --candidate reports/new_run.json \
  --significance-level 0.05 \
  --format html json
```

### Trend Analysis

```bash
# Analyze trends across multiple runs
uv run python -m ulei.cli.main trend \
  --reports reports/run_*.json \
  --metric faithfulness \
  --output trend_analysis.html
```

## Understanding Results

### JSON Report Structure

```json
{
  "run_id": "run_abc123",
  "suite_name": "my_evaluation",
  "timestamp": "2025-11-05T10:30:00Z",
  "dataset_stats": {
    "total_items": 100,
    "items_with_context": 95,
    "items_with_reference": 100
  },
  "results": [
    {
      "item_id": "1",
      "metric": "faithfulness",
      "score": 0.85,
      "error": null,
      "metadata": {
        "latency_ms": 1250,
        "tokens_used": 450
      }
    }
  ],
  "aggregates": {
    "faithfulness": {
      "mean": 0.82,
      "median": 0.85,
      "std": 0.12,
      "min": 0.45,
      "max": 0.98
    }
  },
  "threshold_status": {
    "faithfulness": {
      "threshold": 0.7,
      "actual": 0.82,
      "passed": true
    }
  },
  "cost_summary": {
    "total_cost": 2.45,
    "cost_by_provider": {
      "ragas": 2.45
    },
    "total_tokens": 45000
  },
  "execution_metadata": {
    "total_execution_time": 125.5,
    "successful_evaluations": 98,
    "failed_evaluations": 2
  }
}
```

### HTML Report

HTML reports include:

- **Overview**: Summary statistics and pass/fail status
- **Metric Scores**: Distribution charts and histograms
- **Item Details**: Individual evaluation results
- **Cost Analysis**: Token usage and costs
- **Threshold Comparison**: Visual indicator of threshold compliance

### Interpreting Metrics

#### Faithfulness (0-1)

- **> 0.9**: Excellent - answers are very faithful to context
- **0.7-0.9**: Good - mostly faithful with minor issues
- **0.5-0.7**: Fair - some unfaithful statements
- **< 0.5**: Poor - significant hallucinations

#### Answer Relevancy (0-1)

- **> 0.9**: Highly relevant to the query
- **0.7-0.9**: Relevant with some tangential info
- **0.5-0.7**: Partially relevant
- **< 0.5**: Not relevant to the query

#### Context Precision (0-1)

- **> 0.9**: Very precise retrieval
- **0.7-0.9**: Good retrieval with some noise
- **0.5-0.7**: Many irrelevant documents
- **< 0.5**: Poor retrieval quality

## Advanced Usage

### Environment Variable Substitution

Use environment variables in your config:

```yaml
providers:
  ragas:
    api_key: "${OPENAI_API_KEY}"
    endpoint: "${OPENAI_ENDPOINT:https://api.openai.com/v1}"  # with default
    
dataset:
  source: "${DATA_PATH}/eval_data.jsonl"
```

### Custom Metric Configuration

```yaml
metrics:
  - name: "custom_relevance"
    provider: "deepeval"
    config:
      model: "gpt-4"
      evaluation_params:
        evaluation_steps: ["Extract key claims", "Verify against context"]
        rubric: |
          Score from 1-10 based on:
          - Factual accuracy
          - Completeness
          - Clarity
```

### Conditional Evaluation

Use different configs for different scenarios:

```bash
# Development
uv run python -m ulei.cli.main run eval_dev.yaml --parallel-workers 1

# Staging
uv run python -m ulei.cli.main run eval_staging.yaml --parallel-workers 4

# Production
uv run python -m ulei.cli.main run eval_prod.yaml --parallel-workers 8 --cache
```

### Programmatic Usage

```python
from ulei.core.evaluator import Evaluator
from ulei.utils.config import ConfigLoader
from ulei.core.schemas import DatasetItem

# Load configuration
loader = ConfigLoader()
suite = loader.load_suite("config.yaml")

# Create evaluator
evaluator = Evaluator()

# Load dataset
items = evaluator.load_dataset("data.jsonl")

# Or create items programmatically
items = [
    DatasetItem(
        item_id="1",
        input_data={"query": "What is ML?"},
        output_data={"answer": "Machine learning is..."},
        reference_data={"expected": "ML is..."}
    )
]

# Run evaluation
import asyncio
report = asyncio.run(evaluator.run_evaluation(suite, items))

# Access results
print(f"Mean faithfulness: {report.aggregates['faithfulness']['mean']}")
print(f"Passed thresholds: {report.threshold_status}")
```

### Online/Shadow Evaluation

Start an evaluation server for real-time monitoring:

```bash
# Start server
uv run python -m ulei.cli.main server \
  --config production_eval.yaml \
  --port 8080 \
  --workers 4

# Submit events from your application
curl -X POST http://localhost:8080/v1/eval/events \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "evt_123",
    "input": {"query": "user question"},
    "output": {"answer": "system response"},
    "context": [{"text": "retrieved context"}]
  }'

# Batch submission
curl -X POST http://localhost:8080/v1/eval/events/batch \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {"input": {...}, "output": {...}},
      {"input": {...}, "output": {...}}
    ]
  }'

# Check server health
curl http://localhost:8080/health
```

### Integration with CI/CD

#### GitHub Actions Example

```yaml
name: LLM Evaluation

on:
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install ULEI
        run: |
          pip install uv
          uv add ulei
      
      - name: Run Evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          uv run python -m ulei.cli.main run evaluation.yaml --fail-on-threshold
      
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: evaluation-reports
          path: reports/
      
      - name: Compare with Baseline
        if: github.event_name == 'pull_request'
        run: |
          # Download baseline from main branch
          uv run python -m ulei.cli.main compare \
            --baseline baseline/report.json \
            --candidate reports/*.json \
            --output comparison.html
```

## Next Steps

- **Explore Examples**: Check the `examples/` directory for more configurations
- **Read API Documentation**: See detailed API docs in `docs/API.md`
- **Custom Providers**: Learn to create custom evaluation providers in `docs/CUSTOM_PROVIDERS.md`
- **Performance Tuning**: Optimize for large-scale evaluations in `docs/PERFORMANCE.md`
- **Community**: Join discussions and get help at [GitHub Discussions](https://github.com/yourusername/ulei/discussions)

## Troubleshooting

### Common Issues

**API Key Errors:**

```bash
# Ensure environment variable is set
echo $OPENAI_API_KEY
export OPENAI_API_KEY="sk-..."
```

**Dataset Loading Errors:**

```bash
# Validate your dataset
uv run python -c "
from ulei.utils.dataset import DatasetLoader
items = DatasetLoader.load_dataset('data.jsonl')
print(f'Loaded {len(items)} items')
"
```

**Configuration Errors:**

```bash
# Use dry-run to validate
uv run python -m ulei.cli.main run config.yaml --dry-run
```

**Rate Limiting:**

```yaml
# Add retry configuration
retry_policy:
  max_retries: 5
  backoff_factor: 2.0

# Or reduce parallel workers
parallel_workers: 2
```

### Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: [GitHub Issues](https://github.com/yourusername/ulei/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ulei/discussions)
- **Examples**: See `examples/` for working configurations

## Summary

You now know how to:

- ✅ Install and configure ULEI
- ✅ Prepare evaluation datasets
- ✅ Create evaluation configurations
- ✅ Run evaluations and interpret results
- ✅ Compare model performance
- ✅ Integrate with CI/CD pipelines
- ✅ Use advanced features like caching and parallel execution

Happy evaluating! 🚀
