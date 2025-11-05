#!/bin/bash
# Test script to verify Ollama integration with ULEI

set -e

echo "🔍 Testing Ollama Integration with ULEI"
echo "========================================"
echo

# Check if Ollama is installed
echo "1. Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed"
    echo "   Install from: https://ollama.com/install"
    exit 1
fi
echo "✅ Ollama is installed"
echo

# Check if Ollama is running
echo "2. Checking if Ollama is running..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama is not running"
    echo "   Start with: ollama serve"
    exit 1
fi
echo

# Check available models
echo "3. Checking available models..."
MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys, json; print(', '.join([m['name'] for m in json.load(sys.stdin).get('models', [])]))")
if [ -z "$MODELS" ]; then
    echo "❌ No models available"
    echo "   Pull a model with: ollama pull llama3.1"
    exit 1
fi
echo "✅ Available models: $MODELS"
echo

# Test OpenAI-compatible endpoint
echo "4. Testing OpenAI-compatible endpoint..."
if curl -s http://localhost:11434/v1/models > /dev/null 2>&1; then
    echo "✅ OpenAI-compatible endpoint is working"
else
    echo "❌ OpenAI-compatible endpoint not responding"
    exit 1
fi
echo

# Test ULEI configuration
echo "5. Validating ULEI Ollama configuration..."
if [ -f "examples/configs/ollama_evaluation.yaml" ]; then
    echo "✅ Ollama configuration file exists"
else
    echo "❌ Ollama configuration file not found"
    exit 1
fi
echo

echo "========================================"
echo "✅ All checks passed!"
echo
echo "You can now run evaluations with Ollama:"
echo "  ulei run examples/configs/ollama_evaluation.yaml"
echo
echo "Or start the HTTP server:"
echo "  ulei server --suite-config examples/configs/ollama_evaluation.yaml --port 8080"
echo
