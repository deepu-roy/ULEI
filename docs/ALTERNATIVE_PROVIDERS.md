# Using Alternative LLM Providers with ULEI

This guide explains how to use Ollama, Azure OpenAI, and other LLM providers instead of OpenAI's API with ULEI.

## Overview

ULEI's evaluation providers (Ragas, DeepEval) use LLM-as-a-Judge for metrics like faithfulness and answer relevancy. While they default to OpenAI, you can configure them to use:

- **Ollama** - Run models locally (free, private)
- **Azure OpenAI** - Use Microsoft's hosted OpenAI models
- **vLLM** - High-performance inference server
- **LocalAI** - OpenAI-compatible local API
- **Any OpenAI-compatible endpoint**

## Quick Start with Ollama

### 1. Install and Start Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.1

# Start Ollama (it runs on http://localhost:11434)
ollama serve
```

### 2. Configure ULEI for Ollama

Use the provided configuration:

```yaml
# examples/configs/ollama_evaluation.yaml
name: "ollama_evaluation"
description: "Evaluation using Ollama local models"

providers:
  ragas:
    base_url: "http://localhost:11434/v1"  # Ollama's OpenAI-compatible endpoint
    api_key: "ollama"  # Dummy key (required by libraries)
    default_model: "llama3.1"
    timeout: 120  # Local inference may take longer

metrics:
  - name: "faithfulness"
    provider: "ragas"
    config:
      model: "llama3.1"
      temperature: 0.1
```

### 3. Run Evaluation

```bash
ulei run examples/configs/ollama_evaluation.yaml
```

## Supported Models

### Ollama Models

Popular models that work well for evaluation:

| Model | Size | Best For |
|-------|------|----------|
| `llama3.1` | 8B | General evaluation, fast |
| `llama3.1:70b` | 70B | High-quality evaluation (slower) |
| `mistral` | 7B | Fast, good reasoning |
| `mixtral` | 47B | Excellent for complex evaluations |
| `phi3` | 3.8B | Very fast, lightweight |

Pull models with:

```bash
ollama pull llama3.1
ollama pull mistral
```

### Azure OpenAI Models

Use your Azure OpenAI deployment:

```yaml
providers:
  ragas:
    base_url: "${AZURE_OPENAI_ENDPOINT}/openai/deployments/${AZURE_DEPLOYMENT_NAME}"
    api_key: "${AZURE_OPENAI_API_KEY}"
    api_version: "2024-02-01"
    default_model: "gpt-4"
```

Environment variables needed:

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_DEPLOYMENT_NAME="gpt-4-deployment"
export AZURE_OPENAI_API_KEY="your-azure-key"
```

## Configuration Examples

### Example 1: Ollama with Multiple Models

```yaml
name: "multi_model_ollama"

providers:
  ragas:
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"
    timeout: 120

metrics:
  - name: "faithfulness"
    config:
      model: "llama3.1:70b"  # Use larger model for faithfulness
      temperature: 0.0
  
  - name: "answer_relevancy"
    config:
      model: "mistral"  # Faster model for relevancy
      temperature: 0.1
```

### Example 2: vLLM Server

```yaml
providers:
  ragas:
    base_url: "http://localhost:8000/v1"  # vLLM default port
    api_key: "none"
    default_model: "meta-llama/Llama-3.1-8B-Instruct"
    timeout: 60
```

Start vLLM:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

### Example 3: LocalAI

```yaml
providers:
  ragas:
    base_url: "http://localhost:8080/v1"
    api_key: "local"
    default_model: "gpt-3.5-turbo"  # LocalAI model alias
    timeout: 90
```

### Example 4: HTTP Server with Ollama

```yaml
# server_ollama.yaml
name: "production_monitoring_ollama"
description: "Shadow evaluation with Ollama"

providers:
  ragas:
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"
    default_model: "llama3.1"
    timeout: 120

metrics:
  - name: "faithfulness"
    provider: "ragas"
  - name: "answer_relevancy"
    provider: "ragas"

# Start server
ulei server --suite-config examples/configs/server_ollama.yaml --port 8080
```

## Performance Considerations

### Ollama Performance Tips

1. **Use GPU**: Ollama automatically uses GPU if available

   ```bash
   # Check GPU usage
   nvidia-smi  # For NVIDIA
   ```

2. **Adjust parallel workers**: Lower for local models

   ```yaml
   parallel_workers: 1  # Prevent overload
   ```

3. **Increase timeouts**: Local inference takes longer

   ```yaml
   retry_policy:
     timeout: 120  # vs 30 for OpenAI
   ```

4. **Model selection trade-offs**:
   - Small models (3-8B): Fast but may miss nuances
   - Large models (70B+): High quality but slower

### Cost Comparison

| Provider | Cost | Speed | Privacy |
|----------|------|-------|---------|
| OpenAI API | $0.01-0.06/1K tokens | Fast | Sent to OpenAI |
| Azure OpenAI | $0.01-0.06/1K tokens | Fast | Stays in Azure |
| Ollama | Free | Slower (local) | Fully private |
| vLLM | Free (your GPU) | Fast (optimized) | Fully private |

## Testing Ollama Setup

### 1. Test Ollama is Running

```bash
curl http://localhost:11434/v1/models
```

Expected response:

```json
{
  "object": "list",
  "data": [
    {"id": "llama3.1", "object": "model", ...}
  ]
}
```

### 2. Test Evaluation

```bash
# Run a single evaluation
ulei run examples/configs/ollama_evaluation.yaml

# Check logs for model usage
tail -f ./ollama_results/*.log
```

### 3. Test HTTP Server

```bash
# Start server with Ollama
ulei server --suite-config examples/configs/ollama_evaluation.yaml --port 8080

# Send test event
curl -X POST http://localhost:8080/v1/eval/events \
  -H "Content-Type: application/json" \
  -d '{
    "input": {"query": "What is AI?"},
    "output": {"answer": "AI is artificial intelligence..."},
    "context": [{"text": "AI refers to...", "source": "doc1"}]
  }'
```

## Troubleshooting

### Issue: "Connection refused" to Ollama

**Solution:**

```bash
# Make sure Ollama is running
ollama serve

# Test connectivity
curl http://localhost:11434/api/tags
```

### Issue: Evaluation times out

**Solution:** Increase timeouts

```yaml
retry_policy:
  timeout: 180  # 3 minutes
providers:
  ragas:
    timeout: 180
```

### Issue: "Model not found"

**Solution:** Pull the model first

```bash
ollama pull llama3.1
ollama list  # Verify it's available
```

### Issue: Poor evaluation quality with small models

**Solution:** Use larger models or adjust prompts

```yaml
metrics:
  - name: "faithfulness"
    config:
      model: "llama3.1:70b"  # Use 70B for better quality
```

### Issue: Out of memory with large models

**Solution:** 

1. Use smaller models
2. Reduce parallel workers to 1
3. Close other applications

## Advanced: Custom LLM Endpoints

### Generic OpenAI-Compatible Endpoint

```yaml
providers:
  ragas:
    base_url: "https://your-custom-llm.com/v1"
    api_key: "${YOUR_API_KEY}"
    default_model: "your-model-name"
    timeout: 60
    # Additional headers if needed
    headers:
      X-Custom-Header: "value"
```

### Using Multiple Providers Simultaneously

```yaml
provider_priority:
  - "ragas_ollama"
  - "ragas_openai"  # Fallback to OpenAI if Ollama fails

providers:
  ragas_ollama:
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"
    default_model: "llama3.1"
  
  ragas_openai:
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    default_model: "gpt-3.5-turbo"
```

## Best Practices

### 1. Model Selection by Metric

Different metrics may benefit from different models:

```yaml
metrics:
  - name: "faithfulness"
    config:
      model: "llama3.1:70b"  # Needs strong reasoning
  
  - name: "answer_relevancy"
    config:
      model: "mistral"  # Faster model is fine
  
  - name: "toxicity"
    config:
      model: "phi3"  # Lightweight works well
```

### 2. Development vs Production

**Development (Ollama):**

```yaml
providers:
  ragas:
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"
```

**Production (OpenAI/Azure):**

```yaml
providers:
  ragas:
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
```

### 3. Caching for Cost Savings

Enable aggressive caching with local models:

```yaml
cache:
  enabled: true
  ttl: 604800  # 7 days for local models
```

## See Also

- [Ollama Documentation](https://ollama.com/docs)
- [vLLM Documentation](https://docs.vllm.ai)
- [Azure OpenAI Setup](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [ULEI Configuration Guide](GETTING_STARTED.md)
- [HTTP API Reference](HTTP_API.md)
