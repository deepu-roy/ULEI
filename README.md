# ULEI - Unified LLM Evaluation Interface

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**The "pytest + Terraform-providers" of AI evaluation** - A unified Python interface providing provider-agnostic evaluation of LLM and RAG systems.

## Features

- **Provider-Agnostic**: Switch between evaluation providers (Ragas, DeepEval) via configuration
- **Batch & Online**: Support for both offline CI/CD evaluation and online shadow evaluation
- **Cost Control**: Built-in budget limits and retry policies to prevent cost overruns
- **Statistical Analysis**: Compare model performance with statistical significance testing
- **Multiple Output Formats**: HTML, JSON, JUnit XML, and Prometheus metrics export
- **Local & Cloud**: Local filesystem storage with optional S3/GCS integration

## Quick Start

### Installation

```bash
# Using pip
pip install ulei

# Using uv (recommended)
uv add ulei

# From source
git clone https://github.com/yourusername/unified-llm-evaluation-framework
cd unified-llm-evaluation-framework
uv sync
```

### Basic Usage

1. **Create an evaluation suite configuration**:

```yaml
# evaluation.yaml
name: "rag_evaluation"
description: "Evaluate RAG system performance"
metrics:
  - name: "faithfulness"
    provider: "ragas"
  - name: "answer_relevancy"
    provider: "ragas"
providers:
  ragas:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-3.5-turbo"
thresholds:
  faithfulness: 0.8
  answer_relevancy: 0.7
output_formats: ["html", "json"]
```

2. **Run evaluation**:

```bash
# Basic evaluation
ulei run evaluation.yaml

# With custom output directory
ulei run evaluation.yaml --output-dir ./reports

# With specific formats
ulei run evaluation.yaml --format json html

# CI/CD mode with exit codes on threshold failures
ulei run evaluation.yaml --fail-on-threshold

# Compare two runs
ulei compare --baseline reports/run1.json --candidate reports/run2.json
```

3. **Start online evaluation server**:

```bash
ulei server --port 8080 --config evaluation.yaml
```

### Dataset Format

ULEI supports JSONL datasets with the following structure:

```jsonl
{"id": "1", "input": {"query": "What is RAG?"}, "output": {"answer": "RAG stands for..."}, "reference": {"expected": "Retrieval Augmented Generation"}, "context": [{"text": "RAG retrieves...", "source": "doc1"}]}
{"id": "2", "input": {"query": "How does it work?"}, "output": {"answer": "It combines..."}, "context": [{"text": "RAG uses retrieval...", "source": "doc2"}]}
```

## Architecture

ULEI implements a provider adapter pattern that normalizes evaluation results across different providers:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Your Data     │    │      ULEI        │    │   Providers     │
│                 │    │                  │    │                 │
│  ┌───────────┐  │    │  ┌─────────────┐ │    │  ┌───────────┐  │
│  │ Dataset   │──┼────┼──│ Evaluator   │─┼────┼──│ Ragas     │  │
│  └───────────┘  │    │  └─────────────┘ │    │  └───────────┘  │
│                 │    │         │        │    │  ┌───────────┐  │
│  ┌───────────┐  │    │  ┌─────────────┐ │    │  │ DeepEval  │  │
│  │ Config    │──┼────┼──│ Registry    │─┼────┼──│           │  │
│  └───────────┘  │    │  └─────────────┘ │    │  └───────────┘  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Supported Providers

- **Ragas**: RAG-specific metrics (faithfulness, answer relevancy, context precision)
- **DeepEval**: General LLM metrics (G-Eval, summarization, toxicity)
- **Local Metrics**: BERT-Score, BLEU, ROUGE (no API required)

## Use Cases

### 1. CI/CD Integration

```yaml
# .github/workflows/evaluation.yml
- name: Run LLM Evaluation
  run: |
    ulei run ci_evaluation.yaml --fail-on-threshold
    ulei compare --baseline reports/main_baseline.json --candidate reports/${{ github.sha }}.json
```

### 2. A/B Testing

```bash
# Compare model versions
ulei run evaluation.yaml --output-dir reports/model_a
ulei run evaluation_v2.yaml --output-dir reports/model_b
ulei compare --baseline reports/model_a/*.json --candidate reports/model_b/*.json --significance-level 0.05
```

### 3. Production Monitoring

```bash
# Start shadow evaluation server
ulei server --config production_monitoring.yaml --port 8080

# Send production data for evaluation
curl -X POST http://localhost:8080/v1/eval/events \
  -H "Content-Type: application/json" \
  -d '{"input": {"query": "..."}, "output": {"answer": "..."}}'
```

## Configuration

### Provider Configuration

```yaml
providers:
  ragas:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
    temperature: 0.1
  deepeval:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-3.5-turbo"
    custom_rubric: |
      Rate the response on accuracy (1-10)
```

### Budget Control

```yaml
budget_limit: 50.0 # USD
retry_policy:
  max_retries: 3
  backoff_factor: 2
  timeout: 30
```

### Output Configuration

```yaml
output_formats: ["html", "json", "junit"]
output_dir: "./evaluation_results"
cloud_storage:
  provider: "s3"
  bucket: "my-evaluation-results"
  prefix: "experiments/"
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/unified-llm-evaluation-framework
cd unified-llm-evaluation-framework

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Testing

```bash
# Run tests with uv
uv run pytest

# Run with coverage
uv run pytest --cov=ulei --cov-report=html

# Type checking
uv run mypy ulei/

# Linting
uv run ruff check ulei/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] Additional providers (LangChain, Weights & Biases)
- [ ] Custom metric definition DSL
- [ ] Real-time dashboard for online evaluation
- [ ] Integration with MLOps platforms (MLflow, Kubeflow)
- [ ] Advanced statistical analysis (confidence intervals, power analysis)
